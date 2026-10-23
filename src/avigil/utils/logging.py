"""Run logging: one directory per experiment, holding everything needed to
reconstruct the result -- resolved config, metric history as JSONL, and
checkpoints.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "avigil", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class RunDir:
    """``experiments/<name>/<timestamp>/`` -- never overwrites a previous run."""

    def __init__(self, root: str | Path, name: str, timestamp: bool = True) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / name / stamp if timestamp else Path(root) / name
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "checkpoints").mkdir(exist_ok=True)
        self.metrics_file = self.path / "metrics.jsonl"

    def log_metrics(self, step: int, split: str, metrics: dict) -> None:
        row = {"step": step, "split": split, **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    def __fspath__(self) -> str:
        return str(self.path)
