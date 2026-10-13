"""Cut aligned SAR/label pairs into fixed-size training tiles and assign them to train/val/test by date."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def split_for(date_key: str, split: dict) -> str:
    d = int(date_key[:8])
    for name, dates in split.items():
        if d in [int(x) for x in dates]:
            return name
    return "unused"


def make_tiles(aligned: dict, date_key: str, out_root: Path, split: dict, size: int = 256,
               min_valid: float = 0.5) -> dict:
    """Save tiles as .npz (vv, vh, label, valid). Tiles with < min_valid valid pixels are skipped."""
    part = split_for(date_key, split)
    out = Path(out_root) / part
    out.mkdir(parents=True, exist_ok=True)
    H, W = aligned["label"].shape
    n = skipped = 0
    for r in range(0, H - size + 1, size):
        for c in range(0, W - size + 1, size):
            sl = (slice(r, r + size), slice(c, c + size))
            valid = aligned["valid"][sl]
            if valid.mean() < min_valid:
                skipped += 1
                continue
            np.savez_compressed(out / f"{date_key}_r{r}_c{c}.npz", vv=aligned["vv"][sl], vh=aligned["vh"][sl],
                                label=(aligned["label"][sl] == 1).astype(np.uint8), valid=valid.astype(np.uint8))
            n += 1
    return {"date": date_key, "split": part, "tiles": n, "skipped": skipped}
