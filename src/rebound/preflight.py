"""Read-only inventory: never instantiate a model, decode media, or open tests."""
import importlib.metadata
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from .io import ROOT
from .media import cache_path
from .protocol import selected_rows


def mounted(path, mount):
    value = str(path)
    return Path(str(mount) + value.replace('/', '\\')) if mount and value.startswith('/') else Path(value)


def inspect(config, mount=None):
    cfg = dict(config)
    for key in ('metadata', 'legacy_manifests', 'backbone', 'data_root'):
        cfg[key] = str(mounted(cfg[key], mount))
    result = {'project': 'ReBound', 'training_started': False, 'inference_started': False,
              'test_manifests_opened': False, 'blockers': [], 'dependencies': {}, 'sources': {}}
    result['python'] = sys.version.split()[0]
    result['environment_scope'] = 'Current local interpreter; server dependencies require a server preflight'
    if sys.version_info < (3, 10):
        result['blockers'].append('Full deployment requires Python >=3.10')
    from packaging.specifiers import SpecifierSet
    requirements = {'torch': '>=2.4,<2.6', 'numpy': '>=1.26,<3', 'av': '>=13,<15',
                    'transformers': '>=4.46,<4.49', 'timm': '>=1.0', 'safetensors': '>=0.4',
                    'easydict': '>=1.13', 'omegaconf': '>=2.3'}
    for package in ('torch', 'numpy', 'av', 'transformers', 'safetensors', 'timm', 'easydict', 'omegaconf'):
        try:
            result['dependencies'][package] = importlib.metadata.version(package)
            if result['dependencies'][package] not in SpecifierSet(requirements[package]):
                result['blockers'].append('Dependency version outside deployment range: ' + package + requirements[package])
        except importlib.metadata.PackageNotFoundError:
            result['dependencies'][package] = 'missing'
            result['blockers'].append('Missing Python package: ' + package)
    result['backbone_available'] = (Path(cfg['backbone']) / 'model.safetensors').exists()
    if not result['backbone_available']:
        result['blockers'].append('Frozen VideoMAEv2 checkpoint unavailable')
    rows = {}
    try:
        for source in config.get('experiment_sources', ('audio_source', 'visual_source')):
            result['sources'][source] = {}
            families = []
            for split in ('train', 'dev'):
                selected = selected_rows(cfg, source, split)
                rows.update({r['file']: r for r in selected})
                families.append({r['family'] for r in selected})
                result['sources'][source][split] = {'clips': len(selected), 'groups': dict(Counter(r['group'] for r in selected)),
                    'partitions': dict(Counter(r['partition'] for r in selected))}
            if families[0] & families[1]:
                result['blockers'].append('Train/dev original-family overlap: ' + source)
        samples = list(rows.values())[::max(1, len(rows) // 32)]
        result['sampled_inputs'] = {'sampled_clips': len(samples),
             'missing_raw': sum(not mounted(r['video_path'], mount).exists() for r in samples),
             'missing_avhubert': sum(not mounted(r['avhubert_path'], mount).exists() for r in samples)}
        if result['sampled_inputs']['missing_raw'] or result['sampled_inputs']['missing_avhubert']:
            result['blockers'].append('Sampled raw/speech inputs missing')
    except (OSError, KeyError, ValueError) as error:
        result['blockers'].append('Manifest audit: ' + str(error))
    result['unique_train_dev_clips'] = len(rows)
    result['caches'] = {}
    for name in ('feature_cache', 'reference_cache', 'evidence_cache'):
        present = sum(cache_path(config[name], row).exists() for row in rows.values())
        result['caches'][name] = {'present': present, 'required_train_dev': len(rows)}
        if present < len(rows):
            result['blockers'].append(name + ' requires preparation')
    disk_root = mounted('/home/inzi15', mount) if mount else ROOT
    if disk_root.exists():
        free = shutil.disk_usage(disk_root).free / 2**30
        if mount or free > 1e6:
            result['disk_free_GiB'] = None
            result['disk_note'] = 'Mounted filesystem capacity is not trustworthy; verify df -h on the server before cache extraction.'
        else:
            result['disk_free_GiB'] = round(free, 2)
    # Metadata duration only; no test labels or test CSVs are read for this budget.
    if rows:
        frames = sum(round(float(r.get('duration', 10.)) * 25) for r in rows.values())
        regional = frames / 2 * (config['regions']**2 + 1) * 768 * 2
        result['estimated_uncompressed_train_dev_GiB'] = {
            'regional': round(regional / 2**30, 2),
            'fixed_audio': round(frames * 140 * 2 / 2**30, 2),
            'evidence': round(frames * 2 * 32 * 2 / 2**30, 2)}
    result['status'] = 'DATA_PREPARATION_REQUIRED' if result['blockers'] else 'READY_FOR_EXPLICIT_EXECUTION'
    result['limits'] = ['Read-only path sampling does not validate every media file.',
                        'No trained ReBound checkpoint or ReBound AP result exists.',
                        'Final test caches require additional disk budget before evaluation.']
    return result
