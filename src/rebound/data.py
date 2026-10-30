import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .media import cache_path, audio_samples, interpolate, input_identity, save_npz
from .protocol import view_periods, rebase_periods


def reference_views(config, row):
    with np.load(cache_path(config['reference_cache'], row), allow_pickle=False) as bank:
        if str(bank['input_identity']) != input_identity(config, row):
            raise ValueError('Stale reference input cache')
        times = bank['times'].copy()
        fixed_audio = bank['fixed_audio'].astype(np.float32)
    with np.load(row['avhubert_path'], allow_pickle=False) as speech:
        audio_h = speech['audio_features'][:len(times)].astype(np.float32)
        video_h = speech['video_features'][:len(times)].astype(np.float32)
    with np.load(cache_path(config['feature_cache'], row), allow_pickle=False) as regional:
        visual = regional['features'].astype(np.float32)
        video_times = regional['times']
    audio = np.concatenate((audio_h, fixed_audio), -1)
    video = np.concatenate((video_h, interpolate(visual[:, 0], video_times, times),
                            interpolate(visual[:, 1:].std(1), video_times, times)), -1)
    return [audio.astype(np.float32), video.astype(np.float32)], times


def fit_pca(config, rows, dimension=512, frame_budget=200000):
    if not rows or any(row['group'] != 'Real' or row['partition'] != 'fit' for row in rows):
        raise ValueError('PCA may use genuine fitting families only')
    rng = np.random.RandomState(config['seed'])
    values = [[], []]
    sampled = 0
    for i in rng.permutation(len(rows)):
        views, times = reference_views(config, rows[int(i)])
        positions = rng.choice(len(times), min(32, len(times)), replace=False)
        values[0].append(views[0][positions])
        values[1].append(views[1][positions])
        sampled += len(positions)
        if sampled >= frame_budget:
            break
    means, matrices, scales = [], [], []
    for samples in values:
        x = torch.from_numpy(np.concatenate(samples)).double()
        if min(x.shape) < dimension:
            raise ValueError('Not enough genuine samples/dimensions for configured PCA')
        mean = x.mean(0)
        covariance = (x - mean).T @ (x - mean) / (len(x) - 1)
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[:dimension]
        means.append(mean.float())
        matrices.append(vectors[:, order].float())
        scales.append(eigenvalues[order].clamp_min(1e-4).sqrt().float())
    return {'means': means, 'matrices': matrices, 'scales': scales, 'sampled_frames': sampled,
            'fit_keys_sha256': hashlib.sha256('\n'.join(sorted(r['file'] for r in rows)).encode()).hexdigest()}


def projected_reference(config, row, pca):
    fingerprint = pca.get('fingerprint')
    output = None
    if fingerprint:
        output = cache_path(Path(config['workspace']) / 'projected', row)
        signature = hashlib.sha256((fingerprint + input_identity(config, row)).encode()).hexdigest()
        if output.exists():
            with np.load(output, allow_pickle=False) as saved:
                if str(saved['signature']) != signature:
                    raise ValueError('Stale projected genuine view cache')
                return torch.from_numpy(saved['values'].copy()), torch.from_numpy(saved['times'].copy()).float()
    views, times = reference_views(config, row)
    arrays = [torch.from_numpy(x) for x in views]
    times = torch.from_numpy(times.copy()).float()
    values = [((x - pca['means'][m]) @ pca['matrices'][m]) / pca['scales'][m] for m, x in enumerate(arrays)]
    values = torch.stack(values)
    if output is not None:
        # Float32 is lossless relative to the uncached path; no new feature
        # quantization is introduced by this training throughput optimization.
        save_npz(output, values=values.numpy(), times=times.numpy(), signature=np.asarray(signature))
    return values, times


class Windows(Dataset):
    """All deterministic windows at a fixed hop; no discarded forged regions."""
    def __init__(self, config, rows, kind, pca=None):
        self.config, self.rows, self.kind, self.pca = config, rows, kind, pca
        self.windows = []
        crop, hop = config['crop_seconds'], config['crop_hop_seconds']
        for i, row in enumerate(rows):
            if kind in ('genuine', 'expert'):
                path = cache_path(config['reference_cache'], row)
                with np.load(path, allow_pickle=False) as bank:
                    duration = float(bank['duration'])
            else:
                path = cache_path(config['evidence_cache'], row)
                with np.load(path, allow_pickle=False) as bank:
                    duration = len(bank['times']) / config['reference_hz']
            last = max(0., duration - crop)
            starts = [k * hop for k in range(int(last // hop) + 1)]
            # Keep the final window on the shared 80ms alignment grid.
            final = max(0., math.ceil(last / .08) * .08)
            if final > starts[-1] + 1e-6:
                starts.append(final)
            self.windows.extend((i, first, duration) for first in starts)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        i, first, duration = self.windows[index]
        row, config = self.rows[i], self.config
        frames = round(config['crop_seconds'] * config['reference_hz'])
        start = round(first * config['reference_hz'])
        n = min(frames, max(0, round(duration * config['reference_hz']) - start))
        if n < 2:
            raise ValueError('Window has insufficient genuine context')
        valid = torch.arange(frames) < n
        times = torch.arange(frames).float() / config['reference_hz']
        base = {'valid': valid, 'times': times, 'group': row['group'], 'index': index,
                'periods': rebase_periods(view_periods(row), first, min(config['crop_seconds'], duration - first))}
        if self.kind == 'genuine':
            value, _ = projected_reference(config, row, self.pca)
            value = value[:, start:start + n]
            base['reference'] = torch.nn.functional.pad(value, (0, 0, 0, frames - n))
        elif self.kind == 'evidence':
            with np.load(cache_path(config['evidence_cache'], row), allow_pickle=False) as bank:
                value = torch.from_numpy(bank['evidence'][:, start:start + n].astype(np.float32))
            base['evidence'] = torch.nn.functional.pad(value, (0, 0, 0, frames - n))
        elif self.kind == 'expert':
            waveform = audio_samples(row['video_path'], config['sample_rate'])
            audio_start = round(first * config['sample_rate'])
            actual = min(round((duration - first) * config['sample_rate']), round(config['crop_seconds'] * config['sample_rate']))
            waveform = torch.from_numpy(waveform[audio_start:audio_start + actual].copy())
            count = min(len(waveform), actual)
            base['audio'] = torch.nn.functional.pad(waveform[:count], (0, round(config['crop_seconds'] * config['sample_rate']) - count))
            base['audio_length'] = count
            with np.load(cache_path(config['feature_cache'], row), allow_pickle=False) as bank:
                offset = round(first / .08)
                positions = round(config['crop_seconds'] / .08)
                values = torch.from_numpy(bank['features'][offset:offset + positions].astype(np.float32))
            available = min(len(values), math.ceil(n / 2))
            base['video'] = torch.nn.functional.pad(values[:available], (0, 0, 0, 0, 0, positions - available))
            base['video_valid'] = torch.arange(positions) < available
            base['video_frames'] = n
            base['audio_periods'] = rebase_periods(view_periods(row, 'audio'), first, config['crop_seconds'])
            base['video_periods'] = rebase_periods(view_periods(row, 'video'), first, config['crop_seconds'])
            base['modality'] = int(row['modify_video']) + 2 * int(row['modify_audio'])
        else:
            raise ValueError('Unknown dataset kind')
        return base


def collate(batch):
    ragged = {'periods', 'audio_periods', 'video_periods', 'group'}
    result = {}
    for key in batch[0]:
        if key in ragged:
            result[key] = [row[key] for row in batch]
        else:
            result[key] = torch.utils.data.default_collate([row[key] for row in batch])
    return result
