"""Separate checkpoints for genuine reference, shared head, and source experts."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .genuine import GenuineConfig, GenuinePredictors
from .boundary import BoundaryConfig, SharedContinuousBoundary
from .transport import RealReferenceTransport
from .io import sha256, restore_checkpoint

SCHEMA = 'rebound_speech_fixed_audio_regional2_v1'


def checkpoint_path(config, stage, source=None, latest=False):
    root = config.get('reference_workspace', config['workspace']) if stage in ('pca', 'genuine', 'transport', 'boundary', 'cca') else config['workspace']
    return Path(root) / (stage + ('_' + source if source else '')) / ('last.pt' if latest else 'best.pt')


def specification(config):
    keys = ('seed', 'regions', 'genuine', 'flow', 'boundary', 'crop_seconds', 'crop_hop_seconds')
    return hashlib.sha256(json.dumps({k: config[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def metadata(config, stage, source=None):
    value = {'stage': stage, 'source': source, 'schema': SCHEMA, 'specification': specification(config)}
    dependencies = {'genuine': ['pca'], 'transport': ['genuine', 'pca'],
                    'experts': [], 'boundary': [], 'pca': [],
                    'cca': ['pca'], 'crossmodal': ['pca']}
    value['dependencies'] = {name: sha256(checkpoint_path(config, name)) for name in dependencies[stage]}
    if stage not in ('boundary', 'cca'):
        sources = [source] if source else ['audio_source', 'visual_source']
        value['input_provenance'] = {'metadata': sha256(config['metadata']),
            'source_manifests': {split + '_' + s: sha256(Path(config['legacy_manifests']) / (split + '_' + s + '.csv'))
                                 for s in sources for split in ('train', 'dev')},
            'regional_schema': sha256(Path(config['feature_cache']) / 'signature.json')}
    return value


def predictor(config):
    return GenuinePredictors(GenuineConfig(**config['genuine']))


def boundary(config):
    return SharedContinuousBoundary(BoundaryConfig(**config['boundary']))


def load_model(config, stage, module, source=None, device='cpu'):
    restore_checkpoint(checkpoint_path(config, stage, source), module,
                       expected=metadata(config, stage, source), device=device)
    return module.to(device).eval().requires_grad_(False)


def load_crossmodal(config, source, device='cpu'):
    from .crossmodal import CrossModalBoundary
    path = checkpoint_path(config, 'crossmodal', source)
    if not path.exists():
        return None
    model = CrossModalBoundary()
    return load_model(config, 'crossmodal', model, source=source, device=device)


def load_pca(config):
    state = torch.load(checkpoint_path(config, 'pca'), map_location='cpu', weights_only=True)
    if state['metadata'] != metadata(config, 'pca'):
        raise ValueError('PCA provenance mismatch')
    state['pca']['fingerprint'] = sha256(checkpoint_path(config, 'pca'))
    return state['pca']


def lag_context(x):
    # Causal lag, never the observation whose density is being evaluated.
    return torch.nn.functional.pad(x[..., :-1, :], (0, 0, 1, 0))


class TransportPair(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flows = nn.ModuleList([RealReferenceTransport(**config['flow']) for _ in range(2)])

    def forward(self, evidence, valid=None, groups=None):
        if groups is not None:
            # The undefined first lag and derivative are never fitted.
            mask = valid.clone()
            mask[:, 0] = False
            return sum(flow.genuine_nll(evidence[:, m], mask, groups=groups,
                       context=lag_context(evidence[:, m])) for m, flow in enumerate(self.flows)) / 2
        return torch.stack([flow(evidence[:, m], lag_context(evidence[:, m]))[0]
                            for m, flow in enumerate(self.flows)], 1)
