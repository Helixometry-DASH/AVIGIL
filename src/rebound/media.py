"""Raw media and frozen regional features. All GPU execution is CLI-gated."""
import importlib
import json
import math
import os
import hashlib
import uuid
import time
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json, sha256


def audio_samples(path, sample_rate=16000):
    import av
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError('Missing audio stream: ' + str(path))
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=sample_rate)
        for frame in container.decode(audio=0):
            for value in resampler.resample(frame):
                chunks.append(value.to_ndarray().reshape(-1))
        for value in resampler.resample(None):
            chunks.append(value.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError('Empty decoded audio')
    return np.concatenate(chunks).astype(np.float32)


def video_frames(path):
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('Missing video stream')
        stream = container.streams.video[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - 25) > .01:
            raise ValueError('Benchmark extraction requires 25FPS media; explicit timestamp resampling is needed otherwise')
        stream.thread_type = 'AUTO'
        frames = [frame.to_ndarray(format='rgb24', width=224, height=224) for frame in container.decode(video=0)]
    if not frames:
        raise ValueError('Empty decoded video')
    return np.stack(frames)


def cache_path(root, row):
    key = Path(row['file']).with_suffix('.npz')
    if key.is_absolute() or '..' in key.parts:
        raise ValueError('Unsafe cache key')
    return Path(root) / key


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **values)
    for attempt in range(10):
        try:
            if temporary.exists():
                os.replace(temporary, path)
            return
        except OSError:
            import time
            time.sleep(0.05)


def input_identity(config, row):
    paths = [row['video_path'], row['avhubert_path'], cache_path(config['feature_cache'], row),
             Path(config['feature_cache']) / 'signature.json']
    stats = [(str(path), Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in paths]
    return hashlib.sha256(json.dumps(stats).encode()).hexdigest()


def extract_regions(config, rows, device, rank=0, world=1, window_batch=16):
    from .native import import_vendor
    import_vendor()
    module = importlib.import_module('avigil.models.backbone.backbone')
    weights = Path(config['backbone']) / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    signature = {'schema': 'rebound_regions_v1', 'regions': config['regions'],
                 'backbone_sha256': sha256(weights), 'fps': 25, 'tubelet_frames': 2,
                 'odd_tail': 'repeat last frame, retain final half tubelet'}
    root = Path(config['feature_cache'])
    signature_path = root / 'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Feature cache has incompatible backbone/extraction provenance')
    if rank == 0 and not signature_path.exists():
        atomic_json(signature_path, signature)
    backbone = module.load_backbone(repo=config['backbone'], device=str(device))
    for number in range(rank, len(rows), world):
        row = rows[number]
        path = cache_path(root, row)
        raw = Path(row['video_path'])
        stat = raw.stat()
        identity = str(stat.st_size) + ':' + str(stat.st_mtime_ns)
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if str(old['raw_identity'].item()) != identity:
                    raise ValueError('Raw media changed underneath cached features')
            continue
        frames = video_frames(raw)
        count = len(frames)
        total = math.ceil(count / 2)
        starts = module.window_starts(count, 8)
        needed = starts[-1] + module.WINDOW
        tensor = torch.from_numpy(frames).to(device)
        if needed > count:
            tensor = torch.cat((tensor, tensor[-1:].expand(needed - count, -1, -1, -1)), 0)
        tensor = module.normalize(tensor)
        accumulator = torch.zeros(total, 1 + config['regions'] ** 2, 768, device=device)
        denominator = torch.zeros(total, device=device)
        weights = module.center_weights(device=device)
        with torch.no_grad():
            for begin in range(0, len(starts), window_batch):
                chunk = starts[begin:begin + window_batch]
                batch = torch.stack([tensor[:, first:first + 16] for first in chunk])
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    tokens = module.pool_regions(module.window_tokens(backbone, batch), config['regions']).float()
                for i, first in enumerate(chunk):
                    left, right = first // 2, min(first // 2 + 8, total)
                    if right > left:
                        w = weights[:right - left]
                        accumulator[left:right] += tokens[i, :right - left] * w[:, None, None]
                        denominator[left:right] += w
        if (denominator <= 0).any():
            raise RuntimeError('Uncovered regional feature positions')
        values = (accumulator / denominator[:, None, None]).cpu().numpy().astype(np.float16)
        save_npz(path, features=values, decoded_frames=np.int64(count),
                 times=(np.arange(total) * .08 + .04).astype(np.float64), raw_identity=np.asarray(identity))
        if number % 100 == rank:
            print(json.dumps({'stage': 'extract-regions', 'rank': rank, 'completed_index': number, 'total': len(rows)}), flush=True)


def interpolate(values, times, query):
    values, times, query = np.asarray(values), np.asarray(times), np.asarray(query)
    if len(times) == 1:
        return np.repeat(values[:1], len(query), axis=0)
    hi = np.searchsorted(times, query).clip(1, len(times) - 1)
    lo = hi - 1
    ratio = ((query - times[lo]) / (times[hi] - times[lo])).clip(0, 1)
    return values[lo] * (1 - ratio[:, None]) + values[hi] * ratio[:, None]


def logmel_views(waveform, target_times):
    """Fixed log-mel and spectral-flux views; no forgery-pretrained audio net."""
    waveform = torch.from_numpy(waveform)
    if len(waveform) < 512:
        waveform = torch.nn.functional.pad(waveform, (0, 512 - len(waveform)))
    power = torch.stft(waveform, n_fft=512, hop_length=160, win_length=512,
                       window=torch.hann_window(512), center=False, return_complex=True).abs().square()
    edges = 700 * (torch.pow(10., torch.linspace(0, 2595 * np.log10(1 + 8000 / 700), 72) / 2595) - 1)
    frequencies = torch.arange(257) * (16000 / 512)
    left = (frequencies[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]).clamp_min(1e-5)
    right = (edges[2:, None] - frequencies[None]) / (edges[2:, None] - edges[1:-1, None]).clamp_min(1e-5)
    filters = torch.minimum(left, right).clamp_min(0)
    mel = (filters @ power).clamp_min(1e-8).log().T.numpy()
    flux = np.vstack((np.zeros((1, 70)), np.diff(mel, axis=0)))
    times = (np.arange(len(mel)) * 160 + 256) / 16000
    return interpolate(np.concatenate((mel, flux), -1), times, target_times).astype(np.float32)


def build_reference_inputs(config, rows, rank=0, world=1):
    assist = Path(config['workspace']) / 'extraction_assist.json'
    while assist.exists():
        state = json.loads(assist.read_text())
        if state['status'] == 'FAILED':
            raise RuntimeError('Assisting extractor failed: ' + state.get('error', 'unknown'))
        if state['status'] != 'RUNNING':
            break
        if not Path('/proc/' + str(state['pid'])).exists():
            raise RuntimeError('Assisting extractor disappeared before completing cache writes')
        time.sleep(5)
    for index in range(rank, len(rows), world):
        row = rows[index]
        output = cache_path(config['reference_cache'], row)
        regional_path = cache_path(config['feature_cache'], row)
        with np.load(regional_path, allow_pickle=False) as regional:
            visual = regional['features'].astype(np.float32)
            visual_times = regional['times']
            frame_count = int(regional['decoded_frames'])
        if output.exists():
            with np.load(output, allow_pickle=False) as old:
                if str(old['input_identity']) != input_identity(config, row):
                    raise ValueError('Stale reference input cache')
            continue
        with np.load(row['avhubert_path'], allow_pickle=False) as speech:
            audio_h = speech['audio_features'].astype(np.float32)
            video_h = speech['video_features'].astype(np.float32)
        waveform = audio_samples(row['video_path'])
        length = min(len(audio_h), len(video_h), frame_count, int(len(waveform) / 16000 * 25))
        if length < 2:
            raise ValueError('Insufficient common AV timeline')
        times = np.arange(length) / 25
        # Existing speech banks and the regional cache remain the authoritative
        # sources; do not duplicate thousands of feature channels per frame.
        audio_view = logmel_views(waveform, times)
        save_npz(output, fixed_audio=audio_view.astype(np.float16), times=times,
                 duration=np.float64(length / 25), input_identity=np.asarray(input_identity(config, row)),
                 schema=np.asarray('speech+fixed_logmel_flux; speech+regional_global_std_v1'))
        if index % 100 == rank:
            print(json.dumps({'stage': 'reference-inputs', 'rank': rank, 'completed_index': index, 'total': len(rows)}), flush=True)
