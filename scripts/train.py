#!/usr/bin/env python
"""Train one experiment.

    python scripts/train.py configs/tasks/av_localization/m5.yaml
    python scripts/train.py configs/tasks/av_localization/m5.yaml --test
    python scripts/train.py <config> train.epochs=8 data.loader.batch_size=32

Positional overrides use OmegaConf dotlist syntax and win over the files.
Results land in experiments/<log.name>/<timestamp>/ -- config.yaml,
metrics.jsonl, checkpoints/, and (with --test) test_metrics.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from omegaconf import OmegaConf  # noqa: E402

from avigil.engine import Trainer  # noqa: E402
from avigil.tasks import build_task  # noqa: E402
from avigil.utils import RunDir, load_config, save_config, set_seed, setup_logger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to an experiment config yaml")
    ap.add_argument("overrides", nargs="*", help="key=value dotlist overrides")
    ap.add_argument("--test", action="store_true", help="evaluate best.pt on the test split after training")
    ap.add_argument("--test-only", action="store_true", help="skip training; evaluate only")
    ap.add_argument("--name", default=None, help="override log.name")
    ap.add_argument("--dry-run", action="store_true", help="build everything, print config, run nothing")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.name:
        cfg.log.name = args.name

    set_seed(int(cfg.seed), deterministic=bool(cfg.get("deterministic", False)))

    run = RunDir(cfg.log.out_dir, cfg.log.name)
    logger = setup_logger("avigil", Path(run.path) / "train.log")
    save_config(cfg, run.path)
    logger.info(f"run dir: {run.path}")
    logger.info(f"config:  {cfg.config_path}")

    task = build_task(cfg.task.name, cfg)

    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        ds = task.build_dataset("train", train=True)
        stats = (
            ds.class_balance() if hasattr(ds, "class_balance")
            else ds.segment_stats() if hasattr(ds, "segment_stats")
            else {}
        )
        logger.info(f"dry-run: train dataset has {len(ds)} clips, {stats}")
        item = ds[0]
        logger.info(
            "dry-run: first item "
            + ", ".join(f"{k}={tuple(v.shape)}" for k, v in item.items() if hasattr(v, "shape"))
        )
        return 0

    trainer = Trainer(cfg, task, run, logger)
    if not args.test_only:
        summary = trainer.fit()
        logger.info(f"best {trainer.monitor} = {summary['best'][trainer.monitor]:.4f}")
    if args.test or args.test_only:
        trainer.test()

    logger.info(f"done -> {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
