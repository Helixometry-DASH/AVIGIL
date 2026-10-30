"""Real-only audio-visual agreement reference and a cross-modal boundary head.

Reference statistics (multi-scale CCA between AV-HuBERT audio/video streams and
the genuine evidence scale) are fitted on genuine fitting-family clips only.
Source forgeries train only the boundary head; target-modality forgeries are
never read. The same features and head serve A->V and V->A.

Feature recipe (65 dims), all derived from genuine-referenced quantities:
  * canonical agreement of static, first- and second-difference streams at
    four canonical bands (unit-normalized vectors, so velocity-norm changes
    caused by time stretching or lip re-synthesis do not flip sign),
  * cross-predictive audio surprise channels, despiked during non-boundary
    silence dips and suppressed before agreement drops (removes the visual
    tubelet halo and leading-silence spikes),
  * an end pin: last frame of an acoustic closure that follows an agreement
    drop. It aligns the end step of both manipulation types to the same frame.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn

FEATURE_DIM = 65
BANDS = (16, 32, 64, 128)
# Audio-bank channels holding the video->audio cross-predictive surprise.
CROSS_SURPRISE = (25, 31)


def _fit_cca(a, v, reg=1e-2, k=128):
    mu_a, mu_v = a.mean(0), v.mean(0)
    ac, vc = a - mu_a, v - mu_v
    n = len(ac)
    caa = ac.T @ ac / n + reg * np.eye(a.shape[1])
    cvv = vc.T @ vc / n + reg * np.eye(v.shape[1])
    cav = ac.T @ vc / n
    ea, ua = np.linalg.eigh(caa)
    ev, uv = np.linalg.eigh(cvv)
    wa = ua @ np.diag(1 / np.sqrt(np.maximum(ea, 1e-5))) @ ua.T
    wv = uv @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-5))) @ uv.T
    u, s, vt = np.linalg.svd(wa @ cav @ wv, full_matrices=False)
    return {'mu_a': torch.from_numpy(mu_a.astype(np.float32)), 'mu_v': torch.from_numpy(mu_v.astype(np.float32)),
            'pa': torch.from_numpy((wa @ u[:, :k]).astype(np.float32)),
            'pv': torch.from_numpy((wv @ vt[:k].T).astype(np.float32)),
            'rho': torch.from_numpy(s[:k].astype(np.float32))}


def _speech(row, length):
    with np.load(row['avhubert_path'], allow_pickle=False) as bank:
        a = bank['audio_features'][:length].astype(np.float32)
        v = bank['video_features'][:length].astype(np.float32)
    if len(a) < length or len(v) < length:
        a = np.pad(a, ((0, length - len(a)), (0, 0)), mode='edge')
        v = np.pad(v, ((0, length - len(v)), (0, 0)), mode='edge')
    return a, v


def fit_reference(config, rows, clips=600):
    from .media import cache_path
    if not rows or any(r['group'] != 'Real' or r['partition'] != 'fit' for r in rows):
        raise ValueError('Cross-modal reference may use genuine fitting families only')
    rng = np.random.RandomState(config['seed'])
    chosen = [rows[int(i)] for i in rng.permutation(len(rows))[:clips]]
    streams = {0: ([], []), 1: ([], []), 2: ([], [])}
    evidence = []
    for row in chosen:
        with np.load(cache_path(config['evidence_cache'], row), allow_pickle=False) as bank:
            ev = bank['evidence'].astype(np.float32)
        a, v = _speech(row, ev.shape[1])
        n = ev.shape[1]
        if n <= 16:
            continue
        streams[0][0].append(a[8:n - 8:2]); streams[0][1].append(v[8:n - 8:2])
        da, dv = a[2:n] - a[:n - 2], v[2:n] - v[:n - 2]
        streams[1][0].append(da[7:n - 9:2]); streams[1][1].append(dv[7:n - 9:2])
        d2a, d2v = a[4:n] - a[:n - 4], v[4:n] - v[:n - 4]
        streams[2][0].append(d2a[6:n - 10:2]); streams[2][1].append(d2v[6:n - 10:2])
        evidence.append(ev[:, 8:n - 8:2])
    cca = [_fit_cca(np.concatenate(streams[m][0]).astype(np.float64), np.concatenate(streams[m][1]).astype(np.float64))
           for m in range(3)]
    scale = np.concatenate(evidence, 1).std(axis=1, keepdims=True) + 1e-5
    return {'cca': cca, 'evidence_scale': torch.from_numpy(scale.astype(np.float32)), 'clips': len(evidence),
            'fit_keys_sha256': hashlib.sha256('\n'.join(sorted(r['file'] for r in chosen)).encode()).hexdigest()}


def load_reference(config):
    from .components import checkpoint_path, metadata
    from .io import sha256
    path = checkpoint_path(config, 'cca')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['metadata'] != metadata(config, 'cca'):
        raise ValueError('Cross-modal reference provenance mismatch')
    ref = state['reference']
    return {'cca': [{k: v.numpy() for k, v in c.items()} for c in ref['cca']],
            'evidence_scale': ref['evidence_scale'].numpy(), 'fingerprint': sha256(path)}


def _align(x, hold, damp, first=2):
    t = len(x)
    padded = np.pad(x, ((6, 0), (0, 0)), mode='edge')
    previous = np.max(np.stack([padded[6 - d:t + 6 - d] for d in range(first, 7)]), 0)
    return ((1 - hold) * x + hold * np.maximum(x, previous)) * (1 - damp)


def _stream(a, v, cca, bridge, hold, damp):
    t = len(a)
    u, w = (a - cca['mu_a']) @ cca['pa'], (v - cca['mu_v']) @ cca['pv']
    lo, hi = min(6, t // 4), max(t - 6, t - t // 4)
    out = []
    for k in BANDS:
        uk, wk, rk = u[:, :k], w[:, :k], cca['rho'][:k]
        un = uk / (np.linalg.norm(uk, axis=-1, keepdims=True) + 1e-6)
        wn = wk / (np.linalg.norm(wk, axis=-1, keepdims=True) + 1e-6)
        cos = (un * wn).sum(-1, keepdims=True)
        wcos = (un * wn * rk).sum(-1, keepdims=True) / rk.sum()
        residual = ((un - wn) ** 2 / (2 * (1 - rk + .05))).mean(-1, keepdims=True) * k
        drop = np.median(cos[lo:hi]) - cos
        wdrop = np.median(wcos[lo:hi]) - wcos
        out += [_align(np.maximum(drop, .15 * bridge), hold, damp), _align(np.maximum(wdrop, .15 * bridge), hold, damp),
                _align(residual - np.median(residual[lo:hi]), hold, damp), _align(residual, hold, damp), -cos]
    return np.concatenate(out, -1)


def compute_features(evidence, fixed_audio, a, v, ref):
    """evidence (2,T,32), fixed_audio (T,>=64), a/v (T,768) -> (T,65) float32."""
    t = evidence.shape[1]
    pad1 = lambda x, n: np.pad(x, ((n, n), (0, 0)), mode='edge')
    da, dv = pad1(a, 1)[2:] - pad1(a, 1)[:-2], pad1(v, 1)[2:] - pad1(v, 1)[:-2]
    d2a, d2v = pad1(a, 2)[4:] - pad1(a, 2)[:-4], pad1(v, 2)[4:] - pad1(v, 2)[:-4]
    lo, hi = min(6, t // 4), max(t - 6, t - t // 4)
    z = (evidence - np.median(evidence[:, lo:hi], axis=1, keepdims=True)) / ref['evidence_scale']
    surprise = z[0][:, list(CROSS_SURPRISE)]
    surprise_max = surprise.max(-1)
    c0 = ref['cca'][0]
    u32, w32 = (a - c0['mu_a']) @ c0['pa'][:, :32], (v - c0['mu_v']) @ c0['pv'][:, :32]
    cos32 = ((u32 / (np.linalg.norm(u32, axis=-1, keepdims=True) + 1e-6)) *
             (w32 / (np.linalg.norm(w32, axis=-1, keepdims=True) + 1e-6))).sum(-1)
    drop32 = np.median(cos32[lo:hi]) - cos32
    mel = fixed_audio[:t, :64].mean(-1)
    mel = mel - np.median(mel[lo:hi])
    dip = np.tanh(np.maximum(0, -mel - 1.2) / 1.5)
    dp = np.pad(dip, (2, 2), mode='edge')
    pin = np.maximum(0, dp[1:t + 1] - np.maximum(dp[2:t + 2], dp[3:t + 3]))
    sp = np.pad(surprise_max, (8, 8), mode='edge')
    cp = np.pad(drop32, (10, 10), mode='edge')
    peak = np.maximum(sp[7:t + 7], np.maximum(sp[8:t + 8], sp[9:t + 9]))
    after = (sp[10:t + 10] + sp[11:t + 11] + sp[12:t + 12]) / 3
    before_drop = sum(cp[i:t + i] for i in range(1, 6)) / 5
    after_drop = sum(cp[i:t + i] for i in range(12, 16)) / 4
    gate = (np.tanh(np.maximum(0, 5 * (before_drop - .5 * np.maximum(0, after_drop)))) *
            np.tanh(np.maximum(0, 5 * (before_drop - .08))))
    end_pin = pin * np.tanh(2.5 * np.maximum(0, peak - after)) * gate
    q = np.pad(end_pin, (1, 1))
    end_peak = end_pin * (end_pin >= q[:-2]) * (end_pin >= q[2:])
    ep = np.pad(end_peak, (5, 5))
    hold = np.minimum(1, np.max(np.stack([ep[5 + d:t + 5 + d] for d in range(5)]), 0) * 2.5)[:, None]
    past = np.max(np.stack([ep[5 - d:t + 5 - d] for d in range(1, 6)]), 0)
    damp = np.minimum(.85, past * 2.5 * (1 - hold[:, 0]))[:, None]
    aligned32 = _align(drop32[:, None], hold, damp)[:, 0]
    s4 = np.pad(surprise, ((4, 4), (0, 0)), mode='edge')
    interpolated = .5 * (s4[2:t + 2] + s4[7:t + 7])
    start_dip = (dip * (1 - gate))[:, None]
    clean = (1 - start_dip) * surprise + start_dip * np.minimum(surprise, interpolated)
    keep = np.maximum(gate, np.tanh(np.maximum(0, (aligned32 - .04) * 8)))[:, None]
    clean = _align(clean * keep, hold, damp)
    bridge = clean.max(-1, keepdims=True)
    cliff = clean - np.pad(clean, ((0, 2), (0, 0)), mode='edge')[2:]
    x = np.concatenate([_stream(a, v, ref['cca'][0], bridge, hold, damp),
                        _stream(da, dv, ref['cca'][1], bridge, hold, damp),
                        _stream(d2a, d2v, ref['cca'][2], bridge, hold, damp),
                        clean, cliff, end_peak[:, None]], -1).astype(np.float32)
    if x.shape[1] != FEATURE_DIM:
        raise RuntimeError('Unexpected cross-modal feature width')
    return x


def row_features(config, row, ref):
    """Cached per-clip features keyed by the reference checkpoint fingerprint."""
    from .media import cache_path, save_npz
    from .pipeline import evidence_signature
    out = cache_path(Path(config['workspace']) / 'crossmodal_features', row)
    if out.exists():
        try:
            with np.load(out, allow_pickle=True) as saved:
                if str(saved['signature']) == ref['fingerprint']:
                    return saved['x'].copy(), saved['times'].copy()
        except Exception:
            pass
    with np.load(cache_path(config['evidence_cache'], row), allow_pickle=False) as bank:
        if str(bank['signature']) != evidence_signature(config):
            raise ValueError('Stale evidence cache: ' + row['file'])
        evidence = bank['evidence'].astype(np.float32)
        times = bank['times'].astype(np.float32)
    with np.load(cache_path(config['reference_cache'], row), allow_pickle=False) as bank:
        fixed_audio = bank['fixed_audio'].astype(np.float32)
    a, v = _speech(row, evidence.shape[1])
    x = compute_features(evidence, fixed_audio, a, v, ref)
    save_npz(out, x=x, times=times, signature=np.asarray(ref['fingerprint']))
    return x, times


class CrossModalBoundary(nn.Module):
    """Dilated temporal convolution + BiGRU with continuous start/end offsets."""
    def __init__(self, dimension=FEATURE_DIM, width=128, dropout=.15):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))
        self.conv = nn.Sequential(nn.Dropout1d(dropout),
                                  nn.Conv1d(dimension, width, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.GELU(),
                                  nn.Conv1d(width, width, 5, padding=8, dilation=4), nn.GELU())
        self.gru = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.Conv1d(width, 3, 1)
        self.offsets = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    @torch.no_grad()
    def fit_standardization(self, values, groups):
        if set(groups) != {'Real'}:
            raise ValueError('Standardization uses genuine clips only')
        x = torch.cat(values)
        self.mean.copy_(x.mean(0))
        scale = x.std(0).clamp_min(1e-3)
        scale[-1] = scale[-1].clamp_min(.1)
        self.scale.copy_(scale)

    def forward(self, x, times, valid):
        x = ((x - self.mean) / self.scale).masked_fill(~valid[..., None], 0)
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        g, _ = self.gru(c)
        h = self.norm(c + g).transpose(1, 2)
        frame, start, end = self.heads(h).unbind(1)
        so, eo = self.offsets(h).tanh().unbind(1)
        hop = torch.cat((times[:, 1:] - times[:, :-1], times[:, -1:] - times[:, -2:-1]), -1)
        return {'frame_logits': frame, 'start_logits': start, 'end_logits': end,
                'start_offset': so, 'end_offset': eo, 'start_time': times + hop * so,
                'end_time': times + hop * eo, 'times': times, 'valid': valid}


def batch(items, device):
    length = max(len(x) for x, _ in items)
    xs = torch.stack([torch.nn.functional.pad(torch.from_numpy(x), (0, 0, 0, length - len(x))) for x, _ in items])
    times = torch.arange(length).float()[None].expand(len(items), -1) / 25
    valid = torch.stack([torch.arange(length) < len(x) for x, _ in items])
    return xs.to(device), times.to(device), valid.to(device)


@torch.no_grad()
def predict(config, model, x, times, device):
    from .inference import decoding
    t = torch.from_numpy(times).float()[None].to(device)
    xv = torch.from_numpy(x)[None].to(device)
    output = model(xv, t, torch.ones(1, len(x), dtype=torch.bool, device=device))
    output = {k: v.cpu() if torch.is_tensor(v) else v for k, v in output.items()}
    output['clip_duration'] = float(len(x) / 25)
    return decoding(config, output)


def train(config, runtime, source):
    """Source-only training; checkpoint selected by source-dev AP75."""
    from .components import checkpoint_path, metadata
    from .protocol import selected_rows, source_training_rows, view_periods
    from .targets import targets, localization_loss
    from .metrics import temporal_metrics
    from .io import atomic_torch
    cfg = config['crossmodal']
    device = runtime.device
    ref = load_reference(config)
    rng = np.random.RandomState(config['seed'])
    rows = source_training_rows(selected_rows(config, source, 'train'))
    rows = [rows[int(i)] for i in rng.permutation(len(rows))[:cfg['train_clips']]]
    dev = selected_rows(config, source, 'dev')
    dev = [dev[int(i)] for i in rng.permutation(len(dev))[:cfg['dev_clips']]]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        data = list(pool.map(lambda r: (*row_features(config, r, ref), view_periods(r), r['group']), rows))
        held = list(pool.map(lambda r: (*row_features(config, r, ref), view_periods(r)), dev))
    runtime.log({'stage': 'crossmodal-features', 'source': source, 'total': len(data)})
    torch.manual_seed(config['seed'])
    model = CrossModalBoundary().to(device)
    model.fit_standardization([torch.from_numpy(x) for x, _, _, g in data if g == 'Real'], ['Real'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    best, best_state = -1., None
    for epoch in range(cfg['epochs']):
        model.train()
        order = np.random.RandomState(config['seed'] + epoch).permutation(len(data))
        total = 0.
        for k in range(0, len(order), cfg['batch_size']):
            chunk = [data[int(j)] for j in order[k:k + cfg['batch_size']]]
            xs, ts, valid = batch([(x, t) for x, t, _, _ in chunk], device)
            loss, _ = localization_loss(model(xs, ts, valid), targets(ts, valid, [p for _, _, p, _ in chunk]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss)
        sched.step()
        model.eval()
        proposals = [predict(config, model, x, t, device) for x, t, _ in held]
        score = temporal_metrics(proposals, [p for _, _, p in held])['ap_pct']['0.75']
        runtime.log({'stage': 'crossmodal', 'source': source, 'epoch': epoch,
                     'train_loss': total / max(1, len(order) // cfg['batch_size']), 'source_dev_ap75': score})
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    atomic_torch(checkpoint_path(config, 'crossmodal', source),
                 {'model': best_state, 'metadata': metadata(config, 'crossmodal', source), 'best': best,
                  'selection': 'source-dev AP75 only', 'train_keys_sha256':
                  hashlib.sha256('\n'.join(sorted(r['file'] for r in rows)).encode()).hexdigest()})
    return best

