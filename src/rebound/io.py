import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _sha256(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=256)
def _sha256(path, size, modified_ns):
    # Checkpoint banks are large; rehash only when file identity changes.
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            if temporary.exists():
                os.replace(temporary, path)
            return
        except OSError:
            import time
            time.sleep(0.05)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    pinned_backbone = ROOT / 'backbones/videomaev2-base'
    if (pinned_backbone / 'model.safetensors').is_file():
        # Prefer the verified project-owned copy over a mutable external path.
        pin = ROOT / 'PINNED_INPUTS.json'
        if pin.exists():
            expected = json.loads(pin.read_text())['backbone_sha256']
            if sha256(pinned_backbone / 'model.safetensors') != expected:
                raise ValueError('Project-owned frozen backbone changed')
            config['backbone'] = str(pinned_backbone)
            if Path(config['legacy_manifests']).name == 'manifests' and 'smoke' not in str(config['legacy_manifests']):
                config['legacy_manifests'] = str(ROOT / 'configs/manifests')
    for key in ('workspace', 'feature_cache', 'reference_cache', 'evidence_cache'):
        target = Path(config[key])
        config[key] = str(target if target.is_absolute() else ROOT / target)
    if config.get('reference_workspace'):
        reference = Path(config['reference_workspace'])
        config['reference_workspace'] = str(reference if reference.is_absolute() else ROOT / reference)
    if config['regions'] not in (2, 4) or config['sample_rate'] != 16000 or config['fps'] != 25:
        raise ValueError('This implementation requires 16kHz/25FPS and a 2x2 or 4x4 region grid')
    return config


def code_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'src').rglob('*.py')}


def save_checkpoint(path, module, optimizer, metadata, epoch, best, bad_epochs=0):
    state = {'model': module.state_dict(), 'metadata': metadata, 'epoch': epoch,
             'best': best, 'bad_epochs': bad_epochs, 'torch_rng': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    atomic_torch(path, state)


def restore_checkpoint(path, module, optimizer=None, expected=None, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in (expected or {}).items():
        if state['metadata'].get(key) != value:
            raise ValueError('Checkpoint provenance mismatch: ' + key)
    module.load_state_dict(state['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value):
                    item[key] = value.to(device)
        torch.set_rng_state(state['torch_rng'])
        if 'cuda_rng' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state
