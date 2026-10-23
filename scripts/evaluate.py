#!/usr/bin/env python
"""Evaluate a trained run on a dataset split.

Points at a finished run directory, rebuilds the task from the config saved
beside it, loads a checkpoint, and runs the same evaluation path the trainer
uses.

    python scripts/evaluate.py experiments/av_localization/m5/<timestamp> \
        --ckpt best.pt --split test --out test_metrics.json

Trailing ``key=value`` arguments override the saved config, which is how to
point at moved data (``data.root=...``) or limit the run
(``limit_test_batches=20``).

``--crop full`` scores whole clips by sliding the encoder over each recording
and reassembling the predictions, which is the default. ``--refine soft``
reads each boundary as the probability-weighted centroid of the boundary head,
giving a continuous time rather than a grid point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from avigil.engine.trainer import Trainer
from avigil.tasks import build_task
from avigil.utils.logging import setup_logger

AP_IOU = [0.5, 0.75, 0.9, 0.95]


class _Run:
    """Writes nowhere the original run can see, so scoring never mutates it."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)

    def log_metrics(self, *a, **k) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a timestamped dir under experiments/")
    ap.add_argument("--ckpt", default="last.pt", help="best.pt | last.pt | epochNNN.pt")
    ap.add_argument("--crop", default="full", choices=["full", "head", "center"])
    ap.add_argument("--refine", default="soft", choices=["soft", "argmax", "none"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto (1 when crop=full)")
    ap.add_argument("--out", default="", help="write metrics JSON here")
    ap.add_argument("overrides", nargs="*",
                    help="key=value dotlist overrides applied to the run's config")
    # collected with parse_known_args so overrides may appear in any position
    args, unknown = ap.parse_known_args()
    overrides = list(args.overrides)
    stray = []
    for token in unknown:
        if "=" in token and not token.startswith("-"):
            overrides.append(token)
        else:
            stray.append(token)
    if stray:
        ap.error("unrecognized arguments: " + " ".join(stray))

    rd = Path(args.run_dir)
    cfg = OmegaConf.load(rd / "config.yaml")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg.task.eval_crop = args.crop
    cfg.eval.boundary_refine = args.refine
    cfg.eval.ap_iou = AP_IOU
    # full-length audio cannot be batched: collate_loc stacks, so shapes must match
    audio = str(cfg.task.name).startswith("t4")
    cfg.data.loader.batch_size = args.batch_size or (
        1 if (args.crop == "full" and audio) else int(cfg.data.loader.batch_size)
    )
    cfg.data.loader.drop_last = False

    out = Path(args.out) if args.out else rd / f"test_{args.crop}_{args.refine}.json"
    log = setup_logger("eval_run", out.with_suffix(".log"))
    scratch = out.parent / f"_eval_{rd.name}_{args.crop}_{args.refine}"
    (scratch / "checkpoints").mkdir(parents=True, exist_ok=True)

    task = build_task(cfg.task.name, cfg)
    tr = Trainer(cfg, task, _Run(scratch), log)
    state = torch.load(rd / "checkpoints" / args.ckpt, map_location=tr.device, weights_only=False)
    tr.model.load_state_dict(state["model"])
    log.info(f"{rd.parent.name}: {args.ckpt} from epoch {state['epoch']}, "
             f"crop={args.crop} refine={args.refine} bs={cfg.data.loader.batch_size}")

    metrics, _ = tr.evaluate(tr._loader(args.split, train=False), limit_key="limit_test_batches")
    metrics.update({"_epoch": state["epoch"], "_crop": args.crop,
                    "_refine": args.refine, "_ckpt": args.ckpt, "_split": args.split})
    out.write_text(json.dumps(metrics, indent=2))

    print(f"\n{rd.parent.name}  ({args.split}, {args.crop}, {args.refine}, epoch {state['epoch']})")

    def line(label: str, key: str, fmt: str = "{:.4f}", scale: float = 1.0) -> None:
        """Print a metric only if this task reported it."""
        v = metrics.get(key)
        if v is not None:
            print(f"  {label:<10} " + fmt.format(v * scale))

    for k in AP_IOU:
        line(f"AP@{k}", f"AP@{k}")
    line("AR@10", "AR@10")
    line("bnd MAE", "boundary_mae", "{:.1f} ms", 1000.0)
    line("mod acc", "mod_acc")
    if metrics.get("clip_eer") is not None:
        print(f"  clip EER   {metrics['clip_eer'] * 100:.2f}%"
              + (f"   AUC {metrics['clip_auc']:.4f}" if metrics.get("clip_auc") is not None else ""))
    if metrics.get("n_gt_segments") is not None:
        print(f"  segments   {metrics['n_gt_segments']:.0f} ground truth, "
              f"{metrics.get('n_proposals', 0):.0f} predicted")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
