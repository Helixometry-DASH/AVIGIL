"""Config loading: YAML with a one-level `extends`, merged over configs/base.yaml.

Deliberately small -- Hydra's rewriting of the working directory and logging
makes experiment provenance harder to follow, and every run here needs exactly
one thing: the fully resolved config saved next to its results.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _load_yaml(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{path} must contain a mapping")
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Resolve an experiment config.

    Merge order, later winning:
        configs/base.yaml
        configs/data/<base.defaults.data>.yaml   (under key `data`)
        the file named by `extends`, if present
        the experiment file itself
        CLI dotlist overrides
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_ROOT / path
    exp = _load_yaml(path)

    base = _load_yaml(CONFIG_ROOT / "base.yaml")
    defaults = base.pop("defaults", {})
    cfg = base

    data_name = exp.get("defaults", {}).get("data") or defaults.get("data")
    if data_name:
        data_cfg = _load_yaml(CONFIG_ROOT / "data" / f"{data_name}.yaml")
        cfg = OmegaConf.merge(cfg, {"data": data_cfg})

    if "extends" in exp:
        parent = (path.parent / exp.pop("extends")).resolve()
        cfg = OmegaConf.merge(cfg, _load_yaml(parent))

    exp.pop("defaults", None)
    cfg = OmegaConf.merge(cfg, exp)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    cfg.config_path = str(path)
    return cfg


def save_config(cfg: DictConfig, out_dir: str | Path) -> Path:
    out = Path(out_dir) / "config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(cfg))
    return out
