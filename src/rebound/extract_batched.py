"""Prefetched multi-clip VideoMAEv2 windows with adaptive inference batching."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from .io import atomic_json, sha256
from .media import video_frames, cache_path, save_npz
from .native import import_vendor


def extract(config, rows, device, rank=0, world=1):
    import_vendor()
    module = importlib.import_module('avigil.models.backbone.backbone')
    weights = Path(config['backbone']) / 'model.safetensors'
    signature = {'schema': 'rebound_regions_v1', 'regions': config['regions'],
                 'backbone_sha256': sha256(weights), 'fps': 25, 'tubelet_frames': 2,
                 'odd_tail': 'repeat last frame, retain final half tubelet'}
    root = Path(config['feature_cache'])
    signature_path = root / 'signature.json'
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError('Feature cache schema differs')
    elif rank == 0:
        atomic_json(signature_path, signature)
    backbone = module.load_backbone(repo=config['backbone'], device=str(device))
    pending = []
    for index in range(rank, len(rows), world):
        row = rows[index]
        stat = Path(row['video_path']).stat()
        identity = str(stat.st_size) + ':' + str(stat.st_mtime_ns)
        path = cache_path(root, row)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['raw_identity']) != identity:
                    raise ValueError('Raw media changed: ' + row['file'])
        else:
            pending.append((row, path, identity))
    if not pending:
        return
    mean = torch.tensor(module.PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(module.PIXEL_STD, device=device).view(1, 3, 1, 1, 1)
    center = module.center_weights(device=device)
    capacity, best, ceiling_batch = 16, 0, 129
    free, _ = torch.cuda.mem_get_info(device)
    ceiling_bytes = torch.cuda.memory_allocated(device) + .94 * free
    video_batch = 8
    with ThreadPoolExecutor(max_workers=2) as executor, torch.no_grad():
        # Keep one decode group ahead while the current group runs on the GPU.
        futures = {}
        def submit(begin):
            for j in range(begin, min(begin + video_batch, len(pending))):
                futures[j] = executor.submit(video_frames, pending[j][0]['video_path'])
        submit(0)
        for begin in range(0, len(pending), video_batch):
            if shutil.disk_usage(root).free < 20 * 2**30:
                raise RuntimeError('Less than 20 GiB free during extraction')
            submit(begin + video_batch)
            end = min(begin + video_batch, len(pending))
            inputs, counts, accumulators, denominators, positions = [], [], [], [], []
            for number in range(begin, end):
                raw = futures.pop(number).result()
                count = len(raw)
                starts = module.window_starts(count, 8)
                pad = starts[-1] + 16 - count
                if pad > 0:
                    raw = np.concatenate((raw, np.repeat(raw[-1:], pad, axis=0)))
                inputs.append(torch.from_numpy(raw))
                counts.append(count)
                size = math.ceil(count / 2)
                accumulators.append(torch.zeros(size, 1 + config['regions']**2, 768, device=device))
                denominators.append(torch.zeros(size, device=device))
                positions.extend((number - begin, first) for first in starts)
            cursor = 0
            while cursor < len(positions):
                selected = positions[cursor:cursor + capacity]
                batch = tokens = None
                try:
                    batch = torch.stack([inputs[number][first:first + 16] for number, first in selected])
                    batch = batch.to(device).permute(0, 4, 1, 2, 3).float().div_(255.)
                    batch.sub_(mean).div_(std)
                    torch.cuda.reset_peak_memory_stats(device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        tokens = module.pool_regions(module.window_tokens(backbone, batch), config['regions']).float()
                    peak = torch.cuda.max_memory_allocated(device)
                    if peak > ceiling_bytes and len(selected) > 1:
                        ceiling_batch = min(ceiling_batch, len(selected))
                        best = min(best, ceiling_batch - 1)
                        capacity = max(1, (best + ceiling_batch) // 2)
                        del batch, tokens
                        torch.cuda.empty_cache()
                        continue
                except torch.cuda.OutOfMemoryError:
                    ceiling_batch = min(ceiling_batch, len(selected))
                    best = min(best, ceiling_batch - 1)
                    if ceiling_batch <= 1:
                        raise
                    capacity = max(1, (best + ceiling_batch) // 2)
                    del batch, tokens
                    torch.cuda.empty_cache()
                    continue
                for local, (number, first) in enumerate(selected):
                    left, right = first // 2, min(first // 2 + 8, len(denominators[number]))
                    w = center[:right - left]
                    accumulators[number][left:right] += tokens[local, :right - left] * w[:, None, None]
                    denominators[number][left:right] += w
                cursor += len(selected)
                if len(selected) == capacity:
                    best = max(best, capacity)
                    capacity = min(128, best * 2 if ceiling_batch == 129 else (best + ceiling_batch) // 2)
                    capacity = max(best, capacity)
                del batch, tokens
            for local, number in enumerate(range(begin, end)):
                if (denominators[local] <= 0).any():
                    raise RuntimeError('Uncovered visual tubelet')
                row, path, identity = pending[number]
                values = (accumulators[local] / denominators[local][:, None, None]).cpu().numpy().astype(np.float16)
                save_npz(path, features=values, decoded_frames=np.int64(counts[local]),
                         times=np.arange(len(values)) * .08 + .04, raw_identity=np.asarray(identity))
            print(json.dumps({'stage': 'extract-batched', 'rank': rank, 'completed': end, 'pending': len(pending),
                              'window_batch_capacity': best, 'peak_GiB': peak / 2**30}), flush=True)
