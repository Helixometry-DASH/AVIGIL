"""Explicit stages for caches, reference fitting, and source-only training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .components import (predictor, boundary, load_model, load_pca, TransportPair,
                         checkpoint_path, metadata, lag_context, SCHEMA)
from .protocol import selected_rows, genuine_rows, source_training_rows
from .data import Windows, projected_reference, fit_pca, collate
from .evidence import EvidenceBank
from .media import cache_path, save_npz, input_identity
from .assemble import window_starts, edge_weights
from .runtime import fit
from .io import atomic_torch, sha256
from .native import NativeExperts
from .losses import expert_loss
from .proxies import sample_reference_changes
from .targets import targets, localization_loss
from torch.utils.data import Dataset


def common_real(config, split, part='fit'):
    first = genuine_rows(selected_rows(config, 'audio_source', split), part)
    other = genuine_rows(selected_rows(config, 'visual_source', split), part)
    if {r['file'] for r in first} != {r['file'] for r in other}:
        raise ValueError('Common real reference requires identical genuine A/V coverage')
    return first


def preparation_rows(config):
    result = {}
    for split in ('train', 'dev'):
        for source in config.get('experiment_sources', ('audio_source', 'visual_source')):
            for row in selected_rows(config, source, split):
                result[row['file']] = row
    return list(result.values())


def evidence_signature(config):
    return hashlib.sha256(json.dumps({'schema': SCHEMA, 'predictor': sha256(checkpoint_path(config, 'genuine')),
              'pca': sha256(checkpoint_path(config, 'pca')), 'seed': config['seed'],
              'crop': config['crop_seconds'], 'hop': config['crop_hop_seconds']}, sort_keys=True).encode()).hexdigest()


def validate_evidence(config, rows):
    signature = evidence_signature(config)
    for row in rows:
        with np.load(cache_path(config['evidence_cache'], row), allow_pickle=False) as bank:
            if str(bank['signature']) != signature:
                raise ValueError('Stale evidence cache: ' + row['file'])
            if str(bank['input_identity']) != input_identity(config, row):
                raise ValueError('Evidence inputs changed: ' + row['file'])


@torch.no_grad()
def cache_evidence(config, rows, runtime):
    model = load_model(config, 'genuine', predictor(config), device=runtime.device)
    bank = EvidenceBank(config['genuine']['feature_dim'], config['seed']).to(runtime.device)
    pca, signature = load_pca(config), evidence_signature(config)
    width, hop = round(config['crop_seconds'] * 25), round(config['crop_hop_seconds'] * 25)
    for index in range(runtime.rank, len(rows), runtime.world):
        row = rows[index]
        path = cache_path(config['evidence_cache'], row)
        if path.exists():
            with np.load(path, allow_pickle=False) as previous:
                if str(previous['signature']) != signature:
                    raise ValueError('Refuse stale evidence cache; select a new cache directory')
                if str(previous['input_identity']) != input_identity(config, row):
                    raise ValueError('Evidence inputs changed; select a new cache directory')
            continue
        values, times = projected_reference(config, row, pca)
        result = torch.zeros(2, len(times), 32, device=runtime.device)
        denominator = torch.zeros(len(times), device=runtime.device)
        for first in window_starts(len(times), width, hop):
            n = min(width, len(times) - first)
            x = torch.nn.functional.pad(values[:, first:first + n], (0, 0, 0, width - n))[None].to(runtime.device)
            valid = (torch.arange(width, device=runtime.device) < n)[None]
            with runtime.amp():
                residuals, surprises = model.evidence(x, valid)
                evidence = bank(residuals, surprises, valid)[0, :, :n]
            weight = edge_weights(n, runtime.device)
            result[:, first:first + n] += evidence * weight[None, :, None]
            denominator[first:first + n] += weight
        if (denominator == 0).any():
            raise RuntimeError('Evidence coverage gap')
        save_npz(path, evidence=(result / denominator[None, :, None]).cpu().numpy().astype(np.float16),
                 times=times.numpy(), signature=np.asarray(signature), input_identity=np.asarray(input_identity(config, row)))
        if index % 100 == runtime.rank:
            runtime.log({'stage': 'cache-evidence', 'index': index, 'total': len(rows)})


class ProxyDataset(Dataset):
    def __init__(self, length, config, validation=False):
        self.length, self.config, self.validation = length, config, validation

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seed = self.config['seed'] + index + (100000000 if self.validation else 0)
        z, times, valid, periods = sample_reference_changes(4, 100, step=.04, seed=seed)
        # Each index includes genuine and all alternative families over batches.
        return {'z': z, 'times': times[None].expand(4, -1), 'valid': valid, 'periods': periods, 'group': 'synthetic'}


def proxy_collation_loss(model, batch, step):
    z = batch['z'].flatten(0, 1)
    times, valid = batch['times'].flatten(0, 1), batch['valid'].flatten(0, 1)
    periods = [p for example in batch['periods'] for p in example]
    output = model(z, times, valid)
    crop = [(0., 4.)] * len(z)
    return localization_loss(output, targets(times, valid, periods, crop_ranges=crop))[0]


def train_stage(config, stage, runtime, source=None, resume=False):
    if config.get('reference_workspace') and stage in ('genuine', 'transport', 'boundary'):
        raise ValueError('Inherited reference checkpoints are read-only; use a new reference workspace to refit')
    if stage == 'genuine':
        pca = load_pca(config)
        train = Windows(config, common_real(config, 'train'), 'genuine', pca)
        dev = Windows(config, common_real(config, 'dev', None), 'genuine', pca)
        model = predictor(config)
        loss = lambda module, b, step: module(b['reference'], b['valid'], step)
        return fit(runtime, model, train, dev, loss, stage, config['train']['genuine_epochs'],
                   config['train']['lr_genuine'], resume=resume)
    if stage == 'transport':
        rows, val = common_real(config, 'train', 'reference'), common_real(config, 'dev', None)
        validate_evidence(config, rows + val)
        train, dev = Windows(config, rows, 'evidence'), Windows(config, val, 'evidence')
        model = TransportPair(config)
        # Equal deterministic samples per window, fitting only reserved real families.
        sample = torch.cat([train[i]['evidence'][:, train[i]['valid']][:, 1::4] for i in range(min(2048, len(train)))], 1)
        for m, flow in enumerate(model.flows):
            flow.fit_standardization(sample[m], groups=['Real'])
        loss = lambda module, b, step: module(b['evidence'], b['valid'], b['group'])
        return fit(runtime, model, train, dev, loss, stage, config['train']['transport_epochs'],
                   config['train']['lr_transport'], resume=resume)
    if stage == 'boundary':
        # One dataset item is four episodes; derive actual number of optimizer steps.
        count = config['train']['proxy_steps'] * config['train']['effective_batch']
        train, dev = ProxyDataset(count, config), ProxyDataset(256, config, True)
        return fit(runtime, boundary(config), train, dev, proxy_collation_loss, stage, 1,
                   config['train']['lr_boundary'], resume=resume)
    if stage == 'experts':
        from .inference import expert_predictions
        from .metrics import temporal_metrics
        from .protocol import view_periods
        rows = source_training_rows(selected_rows(config, source, 'train'))
        val = selected_rows(config, source, 'dev')
        train, dev = Windows(config, rows, 'expert'), Windows(config, val, 'expert')
        model = NativeExperts(source)
        def loss(module, b, step):
            output = module(b['audio'], b['video'], b['video_valid'], b['audio_length'], b['video_frames'])
            return expert_loss(output, b)[0]
        def score(module):
            shard = [(index, expert_predictions(config, module, val[index], runtime.device))
                     for index in range(runtime.rank, len(val), runtime.world)]
            if runtime.world > 1:
                import torch.distributed as dist
                gathered = [None] * runtime.world if runtime.rank == 0 else None
                dist.gather_object(shard, gathered, dst=0)
                if runtime.rank:
                    return 0.
                shard = [item for rank in gathered for item in rank]
            proposals = [p for index, p in sorted(shard, key=lambda item: item[0])]
            return temporal_metrics(proposals, [view_periods(r) for r in val])['ap_pct']['0.95']
        return fit(runtime, model, train, dev, loss, stage, config['train']['expert_epochs'],
                   config['train']['lr_experts'], source=source, resume=resume, score_function=score)
    if stage == 'crossmodal':
        from .crossmodal import train as train_crossmodal
        return train_crossmodal(config, runtime, source)
    raise ValueError('Unknown fitting stage')


def save_pca(config):
    if config.get('reference_workspace'):
        raise ValueError('Inherited PCA checkpoint is read-only')
    value = fit_pca(config, common_real(config, 'train'))
    atomic_torch(checkpoint_path(config, 'pca'), {'pca': value, 'metadata': metadata(config, 'pca')})


def save_cca(config):
    if config.get('reference_workspace'):
        raise ValueError('Inherited CCA checkpoint is read-only')
    from .crossmodal import fit_reference
    value = fit_reference(config, common_real(config, 'train', 'fit'))
    atomic_torch(checkpoint_path(config, 'cca'), {'reference': value, 'metadata': metadata(config, 'cca')})
