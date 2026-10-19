"""JRC Global Surface Water v1.4 tiles: find the tiles that cover an area and download missing ones automatically.

Tiles are 10x10 degrees, named by their top-left corner, e.g. occurrence_60E_30Nv1_4_2021.tif covers 60-70E, 20-30N.
Source: https://global-surface-water.appspot.com (Pekel et al. 2016), files on storage.googleapis.com.
"""
from __future__ import annotations

import glob
import math
import os
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/global-surface-water/downloads2021"
LAYERS = ("occurrence", "seasonality")


def tile_ids(west, south, east, north):
    ids = []
    for lon in range(int(math.floor(west / 10) * 10), int(math.ceil(east / 10) * 10), 10):
        for top in range(int(math.floor(south / 10) * 10) + 10, int(math.ceil(north / 10) * 10) + 10, 10):
            lo = f"{abs(lon)}{'E' if lon >= 0 else 'W'}"
            la = f"{abs(top)}{'N' if top >= 0 else 'S'}"
            ids.append(f"{lo}_{la}")
    return ids


def find_local(jrc_dir, layer, tid):
    if not jrc_dir:
        return None
    hits = glob.glob(os.path.join(jrc_dir, "**", f"{layer}_{tid}v1_4_2021.tif"), recursive=True)
    return hits[0] if hits else None


def ensure_tiles(bbox, jrc_dir, download=True, log=print):
    """Return {'occurrence': [...], 'seasonality': [...]} covering bbox (west, south, east, north).
    Missing tiles are downloaded into <jrc_dir>/auto_downloaded/ when download=True."""
    out = {k: [] for k in LAYERS}
    missing = []
    for tid in tile_ids(*bbox):
        for layer in LAYERS:
            p = find_local(jrc_dir, layer, tid)
            if p is None and download and jrc_dir:
                dest = Path(jrc_dir) / "auto_downloaded" / layer / f"{layer}_{tid}v1_4_2021.tif"
                dest.parent.mkdir(parents=True, exist_ok=True)
                url = f"{BASE}/{layer}/{layer}_{tid}v1_4_2021.tif"
                try:
                    log(f"downloading JRC tile {dest.name} ...")
                    tmp = dest.with_suffix(".part")
                    urllib.request.urlretrieve(url, tmp)
                    tmp.replace(dest)
                    p = str(dest)
                except Exception as e:
                    log(f"[warn] could not download {url}: {e}")
            if p:
                out[layer].append(p)
            else:
                missing.append(f"{layer}_{tid}")
    out["missing"] = missing
    return out
