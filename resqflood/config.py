"""Load config/config.yaml (and optional config/local.yaml) and resolve data paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config(dict):
    """Dict with helpers. `cfg.data("01_Boundary/x.geojson")` returns an absolute path inside DADU_DATA."""

    @property
    def data_root(self) -> Path:
        p = Path(self["paths"]["dadu_data"])
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def work_dir(self) -> Path:
        p = Path(self["paths"]["work"])
        p = p if p.is_absolute() else REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def data(self, rel: str) -> Path:
        return self.data_root / rel

    def flood_mask(self, date_key: str | None = None) -> Path:
        key = date_key or self["flood"]["default_date"]
        return self.data(self["flood"]["daily_masks_dir"]) / f"dadu_tuwien_flood_{key}.tif"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else REPO_ROOT / "config" / "config.yaml"
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    local = path.parent / "local.yaml"
    if local.exists():
        cfg = _merge(cfg, yaml.safe_load(local.read_text()) or {})
    return Config(cfg)
