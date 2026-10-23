"""Task-agnostic training loop.

The LR schedule is per-*step* cosine annealing from ``base_lr`` down to
``lr_min`` over the whole run, not a per-epoch schedule.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.varlen import LengthBucketSampler, collate_varlen
from ..tasks.base import Task
from ..utils.logging import RunDir
from ..utils.seed import worker_init


def cosine_annealing(step: int, total_steps: int, lr_ratio_min: float) -> float:
    """Upstream's lambda: multiplier on base_lr, decaying to lr_min/base_lr."""
    return lr_ratio_min + (1 - lr_ratio_min) * 0.5 * (
        1 + math.cos(step / max(total_steps, 1) * math.pi)
    )


class Trainer:
    def __init__(self, cfg, task: Task, run: RunDir, logger) -> None:
        self.cfg = cfg
        self.task = task
        self.run = run
        self.log = logger
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = task.build_model().to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        self.log.info(f"model {cfg.model.name}: {n_params/1e6:.3f}M params on {self.device}")

        self.train_loader = self._loader("train", train=True)
        self.val_loader = self._loader("val", train=False)
        self.log.info(
            f"train {len(self.train_loader.dataset)} clips "
            f"({len(self.train_loader)} batches) | "
            f"val {len(self.val_loader.dataset)} clips "
            f"({len(self.val_loader)} batches)"
        )
        for split, loader in (("train", self.train_loader), ("val", self.val_loader)):
            ds = loader.dataset
            if hasattr(ds, "class_balance"):
                self.log.info(f"  {split} balance: {ds.class_balance()}")

        self.optimizer = self._optimizer()
        self.steps_per_epoch = self._effective_batches(self.train_loader, "limit_train_batches")
        total_steps = self.steps_per_epoch * cfg.train.epochs
        lr_min = float(cfg.train.get("lr_min", cfg.train.lr * 0.05))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: cosine_annealing(s, total_steps, lr_min / cfg.train.lr),
        )
        self.log.info(
            f"schedule: per-step cosine, {total_steps} steps, "
            f"lr {cfg.train.lr:g} -> {lr_min:g}"
        )

        self.amp = bool(cfg.amp) and self.device.type == "cuda"
        self.amp_dtype = getattr(torch, str(cfg.get("amp_dtype", "bfloat16")))
        # bf16 needs no GradScaler; fp16 does
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp and self.amp_dtype == torch.float16
        )

        metric, mode = task.monitor
        self.monitor, self.mode = metric, mode
        self.best = math.inf if mode == "min" else -math.inf
        self.global_step = 0

    # ---- setup helpers ----------------------------------------------------
    def _loader(self, split: str, train: bool) -> DataLoader:
        ds = self.task.build_dataset(split, train=train)
        lc = self.cfg.data.loader

        if bool(self.cfg.task.get("full_length", False)):
            # Whole clips, nothing cropped. Training uses length-bucketed
            # batches (~0.2% padding); evaluation uses batch_size=1 so padding
            # is exactly zero and the score is the model's view of the real,
            # untouched waveform.
            if train:
                sampler = LengthBucketSampler(
                    ds.durations(), batch_size=lc.batch_size, shuffle=True,
                    drop_last=bool(lc.get("drop_last", True)), seed=int(self.cfg.seed),
                )
                self._train_sampler = sampler
                return DataLoader(
                    ds, batch_sampler=sampler, collate_fn=collate_varlen,
                    num_workers=lc.num_workers, pin_memory=lc.get("pin_memory", True),
                    persistent_workers=bool(lc.num_workers and lc.get("persistent_workers", True)),
                    prefetch_factor=lc.get("prefetch_factor", 4) if lc.num_workers else None,
                    worker_init_fn=worker_init,
                )
            return DataLoader(
                ds, batch_size=1, shuffle=False, collate_fn=collate_varlen,
                num_workers=lc.num_workers, pin_memory=lc.get("pin_memory", True),
                worker_init_fn=worker_init,
            )

        # Evaluation may need a different batch size from training. The audio
        # localization collate stacks, so it needs equal shapes: with
        # task.eval_crop="full" the clips are 2.3-19.9 s and only batch_size=1
        # is safe. Set data.loader.eval_batch_size to override, or leave it and
        # a full-length eval crop will force 1 rather than crash in collate.
        eval_bs = int(lc.get("eval_batch_size", 0) or lc.batch_size)
        needs_single = (
            str(self.cfg.task.get("eval_crop", "")) == "full"
            and not self.task.collate_pads
            and not lc.get("eval_batch_size")
        )
        batch_size = int(lc.batch_size) if train else (1 if needs_single else eval_bs)
        if needs_single:
            self.log.info("eval_crop=full with a non-padding collate -> eval batch_size=1")

        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=self.task.collate,
            shuffle=train,
            num_workers=lc.num_workers,
            pin_memory=lc.get("pin_memory", True),
            drop_last=bool(train and lc.get("drop_last", True)),
            persistent_workers=bool(lc.num_workers and lc.get("persistent_workers", True)),
            prefetch_factor=lc.get("prefetch_factor", 4) if lc.num_workers else None,
            worker_init_fn=worker_init,
        )

    def _optimizer(self) -> torch.optim.Optimizer:
        tc = self.cfg.train
        kind = str(tc.optimizer).lower()
        params = self.model.parameters()
        betas = tuple(tc.get("betas", (0.9, 0.999)))
        if kind == "adam":
            return torch.optim.Adam(
                params, lr=tc.lr, betas=betas, weight_decay=tc.weight_decay,
                amsgrad=bool(tc.get("amsgrad", False)),
            )
        if kind == "adamw":
            return torch.optim.AdamW(
                params, lr=tc.lr, betas=betas, weight_decay=tc.weight_decay
            )
        if kind == "sgd":
            return torch.optim.SGD(
                params, lr=tc.lr, momentum=0.9, weight_decay=tc.weight_decay, nesterov=True
            )
        raise ValueError(f"unknown optimizer {tc.optimizer!r}")

    def _effective_batches(self, loader: DataLoader, limit_key: str) -> int:
        limit = self.cfg.get(limit_key, None)
        return min(len(loader), int(limit)) if limit else len(loader)

    def _to_device(self, batch: dict) -> dict:
        return {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    # ---- loops ------------------------------------------------------------
    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        limit = self._effective_batches(self.train_loader, "limit_train_batches")
        accum = int(self.cfg.train.get("accum_steps", 1))
        running: dict[str, float] = defaultdict(float)
        seen = 0
        t0 = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(self.train_loader):
            if i >= limit:
                break
            batch = self._to_device(batch)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
                loss, stats = self.task.compute_loss(self.model, batch)

            self.scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                if self.cfg.train.get("grad_clip", 0):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.cfg.train.grad_clip)
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            for k, v in stats.items():
                running[k] += v
            seen += 1

            if seen % max(1, limit // 10) == 0:
                avg = {k: v / seen for k, v in running.items()}
                self.log.info(
                    f"  epoch {epoch} [{seen}/{limit}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    + f" lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

        out = {k: v / max(seen, 1) for k, v in running.items()}
        out["epoch_seconds"] = time.time() - t0
        out["lr"] = self.optimizer.param_groups[0]["lr"]
        return out

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, limit_key: str = "limit_val_batches") -> dict:
        self.model.eval()
        limit = self._effective_batches(loader, limit_key)
        collected: dict[str, list] = defaultdict(list)
        for i, batch in enumerate(loader):
            if i >= limit:
                break
            batch = self._to_device(batch)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
                pred = self.task.predict(self.model, batch)
            for k, v in pred.items():
                collected[k].append(v.detach().float().cpu().numpy())
        merged = {k: np.concatenate(v) for k, v in collected.items()}
        return self.task.evaluate(merged), merged

    # ---- checkpoints ------------------------------------------------------
    def save(self, epoch: int, metrics: dict, best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": metrics,
            "monitor": {"name": self.monitor, "mode": self.mode, "best": self.best},
        }
        ckpt_dir = Path(self.run.path) / "checkpoints"
        torch.save(state, ckpt_dir / "last.pt")
        if best:
            torch.save(state, ckpt_dir / "best.pt")
            self.log.info(f"  new best {self.monitor}={metrics[self.monitor]:.4f} -> best.pt")

        keep = int(self.cfg.log.get("keep_last", 3))
        if keep and self.cfg.log.get("save_every", 0):
            torch.save(state, ckpt_dir / f"epoch{epoch:03d}.pt")
            epochs = sorted(ckpt_dir.glob("epoch*.pt"))
            for old in epochs[:-keep]:
                old.unlink()

    def _is_better(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        return value < self.best if self.mode == "min" else value > self.best

    # ---- entry point ------------------------------------------------------
    def fit(self) -> dict:
        history = []
        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            ds = getattr(self.train_loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
            if hasattr(self, "_train_sampler"):
                self._train_sampler.set_epoch(epoch)
            tr = self.train_epoch(epoch)
            self.run.log_metrics(epoch, "train", tr)

            val, _ = self.evaluate(self.val_loader)
            self.run.log_metrics(epoch, "val", val)
            self.log.info(
                f"epoch {epoch} val: "
                + " ".join(
                    f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, float)
                )
            )

            better = self._is_better(val.get(self.monitor, math.nan))
            if better:
                self.best = val[self.monitor]
            self.save(epoch, val, best=better)
            history.append({"epoch": epoch, "train": tr, "val": val})

        summary = {"best": {self.monitor: self.best}, "epochs": len(history)}
        (Path(self.run.path) / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def test(self) -> dict:
        """Evaluate the best checkpoint on the test split."""
        ckpt = Path(self.run.path) / "checkpoints" / "best.pt"
        if ckpt.is_file():
            state = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            self.log.info(f"loaded best.pt from epoch {state['epoch']}")
        else:
            self.log.warning("no best.pt found; testing the current weights")

        loader = self._loader("test", train=False)
        metrics, raw = self.evaluate(loader, limit_key="limit_test_batches")
        self.run.log_metrics(-1, "test", metrics)
        np.savez_compressed(Path(self.run.path) / "test_scores.npz", **raw)
        self.log.info(
            "TEST: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        )
        (Path(self.run.path) / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics
