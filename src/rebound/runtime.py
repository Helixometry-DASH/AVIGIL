"""Single GPU or torchrun DDP; explicit memory headroom and resumable fitting."""
import contextlib
import json
import math
import os
import random
import gc
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .data import collate
from .io import save_checkpoint, restore_checkpoint, code_hashes, atomic_json
from .components import checkpoint_path, metadata


class Runtime:
    def __init__(self, config):
        self.rank, self.world = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1))
        self.local = int(os.getenv('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local)
            self.device = torch.device('cuda', self.local)
        else:
            self.device = torch.device('cpu')
        if self.world > 1:
            dist.init_process_group('nccl' if self.device.type == 'cuda' else 'gloo', timeout=timedelta(hours=12))
        seed = config['seed'] + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.config = config

    def amp(self):
        enabled = self.device.type == 'cuda' and self.config['train']['amp']
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def close(self):
        if self.world > 1:
            dist.destroy_process_group()

    def log(self, value, stage=None, source=None):
        if self.rank:
            return
        line = json.dumps(value, allow_nan=False)
        print(line, flush=True)
        if stage:
            folder = checkpoint_path(self.config, stage, source).parent
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'train.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(line + '\n')


def on_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def tune_batch(runtime, module, dataset, loss_function, lr):
    """Future execution only: real forward/backward/Adam probes, then exact reset.

    The smallest safe batch across selected ranks wins. Ten percent of currently
    free VRAM is reserved; deliberately filling all VRAM would make runs fragile.
    """
    cfg, device = runtime.config, runtime.device
    if device.type != 'cuda' or not cfg['train'].get('auto_batch', False):
        return cfg['train']['batch_size']
    snapshot = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    limit = int(cfg['train'].get('max_batch_size', 32))
    available, _ = torch.cuda.mem_get_info(device)
    ceiling = torch.cuda.memory_allocated(device) + int(available * .9)
    best = 0
    size = 1
    try:
        while size <= limit:
            opt = batch = loss = None
            success = False
            try:
                module.load_state_dict(snapshot)
                module.train()
                torch.cuda.reset_peak_memory_stats(device)
                batch = on_device(collate([dataset[i % len(dataset)] for i in range(size)]), device)
                opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=lr)
                with runtime.amp():
                    loss = loss_function(module, batch, 0)
                loss.backward()
                opt.step()
                torch.cuda.synchronize(device)
                success = torch.cuda.max_memory_allocated(device) <= ceiling
            except torch.cuda.OutOfMemoryError:
                success = False
            finally:
                module.zero_grad(set_to_none=True)
                del opt, batch, loss
                gc.collect()
                torch.cuda.empty_cache()
            if not success:
                break
            best = size
            size *= 2
    finally:
        module.load_state_dict(snapshot)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        del snapshot
    shared = torch.tensor(best, device=device)
    if runtime.world > 1:
        dist.all_reduce(shared, op=dist.ReduceOp.MIN)
    if int(shared) < 1:
        raise RuntimeError('No safe batch fits on every selected GPU; use a smaller architecture/configuration')
    return int(shared)


def fit(runtime, module, train, validation, loss_function, stage, epochs, lr, source=None,
        resume=False, score_function=None):
    """Validation loss is minimized, or source-only development AP is maximized.

    Batches are explicit configuration, never selected from test performance.
    Resume is at epoch boundaries. Worker sampling is deterministic by epoch.
    """
    cfg, dev = runtime.config, runtime.device
    module.to(dev)
    opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=lr, weight_decay=.01)
    meta = metadata(cfg, stage, source)
    initial, best, bad = 0, float('-inf'), 0
    last = checkpoint_path(cfg, stage, source, latest=True)
    if resume:
        state = restore_checkpoint(last, module, opt, expected=meta, device=dev)
        initial, best, bad = state['epoch'] + 1, state['best'], state['bad_epochs']
    if runtime.rank == 0:
        atomic_json(last.parent / 'run.json', {'config': cfg, 'metadata': meta, 'code_hashes': code_hashes(),
                    'world_size': runtime.world, 'resume': resume})
    batch_size = tune_batch(runtime, module, train, loss_function, lr)
    accumulation = max(1, math.ceil(cfg['train']['effective_batch'] / (batch_size * runtime.world)))
    if stage == 'boundary':
        # Keep optimizer-step budget exact after adaptive microbatch selection.
        train.length = cfg['train']['proxy_steps'] * batch_size * runtime.world * accumulation
    sampler = DistributedSampler(train, runtime.world, runtime.rank, shuffle=True, seed=cfg['seed'])
    loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=cfg['train']['workers'],
                        collate_fn=collate, pin_memory=dev.type == 'cuda')
    runtime.log({'event': 'batch_plan', 'epochs': epochs, 'training_windows': len(train),
                 'batch_per_gpu': batch_size, 'gpu_count': runtime.world,
                 'gradient_accumulation': accumulation, 'effective_batch': batch_size * runtime.world * accumulation,
                 'batches_per_epoch': len(loader), 'optimizer_updates_per_epoch': math.ceil(len(loader) / accumulation)}, stage, source)
    wrapped = DistributedDataParallel(module, device_ids=[runtime.local] if dev.type == 'cuda' else None) if runtime.world > 1 else module
    for epoch in range(initial, epochs):
        sampler.set_epoch(epoch)
        module.train()
        opt.zero_grad(set_to_none=True)
        total = 0.
        for step, batch in enumerate(loader):
            batch = on_device(batch, dev)
            end = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            remainder = len(loader) % accumulation
            divisor = remainder if remainder and step >= len(loader) - remainder else accumulation
            sync = contextlib.nullcontext() if end or runtime.world == 1 else wrapped.no_sync()
            with sync, runtime.amp():
                loss = loss_function(wrapped, batch, epoch * len(loader) + step)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss; no result checkpoint accepted')
                (loss / divisor).backward()
            total += float(loss.detach())
            if end:
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                runtime.log({'epoch': epoch, 'step': step, 'loss': float(loss.detach()),
                             'allocated_GiB': torch.cuda.memory_allocated(dev) / 2**30 if dev.type == 'cuda' else 0}, stage, source)
        module.eval()
        if runtime.world > 1:
            for buffer in module.buffers():
                dist.broadcast(buffer, 0)
        # Exact validation sharding: never duplicate records to equalize ranks.
        local = torch.tensor([0., 0.], device=dev)
        with torch.no_grad():
            for index in range(runtime.rank, len(validation), runtime.world):
                batch = on_device(collate([validation[index]]), dev)
                with runtime.amp():
                    loss = loss_function(module, batch, index)
                local += torch.tensor([float(loss), 1.], device=dev)
        if runtime.world > 1:
            dist.all_reduce(local)
        if local[1] == 0:
            raise ValueError('Empty validation dataset')
        validation_loss = float(local[0] / local[1])
        score = -validation_loss
        if score_function:
            # The callback shards exact source-dev clips and gathers proposals.
            score_value = score_function(module)
            score_tensor = torch.tensor([score_value if runtime.rank == 0 else 0.], device=dev)
            if runtime.world > 1:
                dist.broadcast(score_tensor, 0)
            score = float(score_tensor[0])
        improved = score > best
        bad, best = (0, score) if improved else (bad + 1, best)
        if runtime.rank == 0:
            if improved:
                save_checkpoint(checkpoint_path(cfg, stage, source), module, None, meta, epoch, best)
            save_checkpoint(last, module, opt, meta, epoch, best, bad)
        runtime.log({'epoch': epoch, 'mean_train_loss_rank0': total / max(1, len(loader)),
                     'validation_loss': validation_loss, 'selection_score': score, 'best_score': best,
                     'effective_batch': batch_size * runtime.world * accumulation}, stage, source)
        runtime.barrier()
        if bad >= cfg['train']['patience']:
            break
