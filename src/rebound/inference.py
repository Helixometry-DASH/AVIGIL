"""Independent private and shared proposals with genuine-only tail ranking."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .assemble import blend_predictions, window_starts
from .components import checkpoint_path, load_model, boundary, TransportPair, specification
from .data import Windows, collate
from .decode import decode
from .io import atomic_json, sha256
from .media import cache_path
from .metrics import suppress, temporal_metrics
from .native import NativeExperts
from .protocol import selected_rows, genuine_rows, view_periods
from .runtime import on_device


def decoding(config, prediction):
    kwargs = {k: v for k, v in config['decode'].items() if k != 'shared_weight'}
    return decode(prediction, **kwargs)


@torch.no_grad()
def expert_predictions(config, model, row, device):
    # Strip labels before constructing inference windows.
    blind = {**row, 'fake_periods': [], 'modify_audio': False, 'modify_video': False}
    dataset = Windows(config, [blind], 'expert')
    windows = []
    for i, (_, first, duration) in enumerate(dataset.windows):
        b = on_device(collate([dataset[i]]), device)
        output = model(b['audio'], b['video'], b['video_valid'], b['audio_length'], b['video_frames'])
        windows.append((first, output))
    count = (round(duration * 16000) - 128) // 243
    timeline = (torch.arange(count, device=device).float() * 243 + 64) / 16000
    output = blend_predictions(windows, timeline, torch.ones(count, dtype=torch.bool, device=device))
    output['clip_duration'] = duration
    return decoding(config, output)


@torch.no_grad()
def shared_predictions(config, flow, head, row, device):
    from .pipeline import validate_evidence
    validate_evidence(config, [row])
    with np.load(cache_path(config['evidence_cache'], row), allow_pickle=False) as bank:
        evidence = torch.from_numpy(bank['evidence'].astype(np.float32)).to(device)
        timeline = torch.from_numpy(bank['times'].copy()).float().to(device)
    width, hop = round(config['crop_seconds'] * 25), round(config['crop_hop_seconds'] * 25)
    routes = []
    for modality in range(2):
        windows = []
        for first in window_starts(len(timeline), width, hop):
            n = min(width, len(timeline) - first)
            x = torch.nn.functional.pad(evidence[:, first:first + n], (0, 0, 0, width - n))[None]
            valid = (torch.arange(width, device=device) < n)[None]
            # Fit and inference use the same crop-local causal lag.
            z = flow(x)[:, modality]
            times = torch.arange(width, device=device).float() / 25
            prediction = head(z, times, valid)
            windows.append((first / 25, prediction))
        result = blend_predictions(windows, timeline, torch.ones(len(timeline), device=device, dtype=torch.bool))
        result['clip_duration'] = len(timeline) / 25
        routes.append(decoding(config, result))
    return routes


def run_signature(config, source):
    checkpoints = {k: sha256(checkpoint_path(config, k, source if k == 'experts' else None))
                   for k in ('experts', 'boundary', 'transport', 'genuine', 'pca')}
    if checkpoint_path(config, 'cca').exists():
        checkpoints['cca'] = sha256(checkpoint_path(config, 'cca'))
    if checkpoint_path(config, 'crossmodal', source).exists():
        checkpoints['crossmodal'] = sha256(checkpoint_path(config, 'crossmodal', source))
    return {'source': source, 'specification': specification(config), 'decode': config['decode'],
            'checkpoints': checkpoints}


class Predictor:
    def __init__(self, config, source, device, calibrated=True):
        self.config, self.source, self.device = config, source, device
        self.experts = load_model(config, 'experts', NativeExperts(source), source, device)
        self.flow = load_model(config, 'transport', TransportPair(config), device=device)
        self.head = load_model(config, 'boundary', boundary(config), device=device)
        self.ref = None
        self.crossmodal = None
        if checkpoint_path(config, 'cca').exists():
            from .crossmodal import load_reference
            self.ref = load_reference(config)
        if checkpoint_path(config, 'crossmodal', source).exists():
            from .crossmodal import CrossModalBoundary
            self.crossmodal = load_model(config, 'crossmodal', CrossModalBoundary(), source=source, device=device)
        self.calibration = None
        if calibrated:
            state = json.loads((Path(config['workspace']) / ('calibration_' + source + '.json')).read_text())
            if state['signature'] != run_signature(config, source):
                raise ValueError('Calibration/checkpoint signature mismatch')
            self.calibration = state['genuine_route_maxima']

    def routes(self, row):
        private = expert_predictions(self.config, self.experts, row, self.device)
        if self.crossmodal is not None and self.ref is not None:
            from .crossmodal import row_features, predict
            x, times = row_features(self.config, row, self.ref)
            cm_props = predict(self.config, self.crossmodal, x, times, self.device)
            return [private, cm_props, cm_props]
        return [private] + shared_predictions(self.config, self.flow, self.head, row, self.device)

    def combine(self, routes):
        shared_weight = self.config['decode']['shared_weight']
        if not .25 <= shared_weight <= .75:
            raise ValueError('Shared proposal route must retain a nonzero fixed budget')
        weights = (1 - shared_weight, shared_weight / 2, shared_weight / 2)
        proposals = []
        for i, route in enumerate(routes):
            null = np.sort(self.calibration[i]) if self.calibration is not None else None
            for score, start, end in route:
                if score < 0:
                    continue
                rarity = ((1 + np.searchsorted(null, float(score), side='left')) / (len(null) + 1)) * float(score) if null is not None else float(score)
                proposals.append((float(rarity) * weights[i], float(start), float(end)))
        return suppress(proposals, budget=self.config['decode']['budget'], threshold=self.config['decode']['nms_iou'])


def calibrate(config, source, runtime):
    if runtime.world != 1:
        raise ValueError('Calibration is one coordinator process; no duplicate calibration ranks')
    model = Predictor(config, source, runtime.device, calibrated=False)
    rows = genuine_rows(selected_rows(config, source, 'train'), 'calibration')
    if len(rows) < 50:
        raise ValueError('At least 50 held-out genuine calibration clips are required')
    maxima = [[], [], []]
    for i, row in enumerate(rows):
        for m, proposals in enumerate(model.routes(row)):
            maxima[m].append(max(0., float(proposals[:, 0].max())))
        if i % 100 == 0:
            runtime.log({'stage': 'calibration', 'completed': i, 'total': len(rows)})
    atomic_json(Path(config['workspace']) / ('calibration_' + source + '.json'),
                {'signature': run_signature(config, source), 'genuine_route_maxima': maxima,
                 'calibration_keys_sha256': hashlib.sha256('\n'.join(sorted(r['file'] for r in rows)).encode()).hexdigest(),
                 'interpretation': 'empirical genuine-clip tail ranking; not a fake probability or conformal guarantee'})


def infer(config, source, target, runtime):
    model = Predictor(config, source, runtime.device)
    rows = selected_rows(config, source, 'test', target=target, final_evaluation=True)
    folder = Path(config['workspace']) / ('inference_' + source + '_to_' + target)
    signature = run_signature(config, source)
    for i in range(runtime.rank, len(rows), runtime.world):
        row = rows[i]
        output = folder / (Path(row['file']).stem + '.json')
        if output.exists():
            old = json.loads(output.read_text())
            if old['signature'] != signature or old['file'] != row['file']:
                raise ValueError('Stale prediction file')
            continue
        routes = model.routes(row)
        proposals = model.combine(routes)
        atomic_json(output, {'file': row['file'], 'signature': signature,
                    'proposals': proposals[proposals[:, 0] >= 0].tolist(),
                    'routes': [p[p[:, 0] >= 0].tolist() for p in routes]})
        if i % 100 == runtime.rank:
            runtime.log({'stage': 'inference', 'index': i, 'total': len(rows), 'source': source, 'target': target})
    runtime.barrier()


def evaluate(config, source, target):
    rows = selected_rows(config, source, 'test', target=target, final_evaluation=True)
    folder = Path(config['workspace']) / ('inference_' + source + '_to_' + target)
    proposals, routes = [], [[], [], []]
    signature = run_signature(config, source)
    for row in rows:
        value = json.loads((folder / (Path(row['file']).stem + '.json')).read_text())
        if value['signature'] != signature or value['file'] != row['file']:
            raise ValueError('Prediction provenance mismatch')
        proposals.append(value['proposals'])
        for i in range(3):
            routes[i].append(value['routes'][i])
    ground = [view_periods(row) for row in rows]
    from .metrics import prediction_first_metrics
    result = {'canonical': temporal_metrics(proposals, ground), 'prediction_first': prediction_first_metrics(proposals, ground),
              'individual_routes': {name: temporal_metrics(p, ground) for name, p in zip(('private', 'shared_audio', 'shared_video'), routes)},
              'source': source, 'target': target, 'signature': signature}
    if source == 'mixed_source':
        result['individual_routes_prediction_first'] = {name: prediction_first_metrics(p, ground)
                            for name, p in zip(('private', 'shared_audio', 'shared_video'), routes)}
        result['manipulation_groups_with_genuine_controls'] = {}
        for label in ('A-only', 'V-only', 'AV'):
            indices = [i for i, row in enumerate(rows) if row['group'] in ('Real', label)]
            subset_ground = [ground[i] for i in indices]
            result['manipulation_groups_with_genuine_controls'][label] = {
                'combined': temporal_metrics([proposals[i] for i in indices], subset_ground),
                'private': temporal_metrics([routes[0][i] for i in indices], subset_ground)}
    atomic_json(folder / 'metrics.json', result)
    return result
