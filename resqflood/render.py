"""Rendering + saving of detector results (overlay PNG, class GeoTIFF, JSON summary)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLORS = {1: (255, 170, 0), 2: (225, 25, 35), 3: (30, 90, 225), 4: (0, 200, 210)}
LEGEND = ("red = FLOOD · orange = possible flood (weak evidence, not counted) · blue = permanent water (river/lake) · "
          "cyan = normal seasonal / river-corridor water")


def sar_gray(vv_lee_db, valid):
    g = np.clip((np.nan_to_num(vv_lee_db, nan=-25) + 25) / 25, 0, 1)
    rgb = np.repeat((g * 255).astype("uint8")[..., None], 3, -1)
    rgb[~valid] = (0, 0, 0)
    return rgb


def overlay(res, alpha=0.6):
    base = sar_gray(res["vv_lee_db"], res["valid"]).astype("float32")
    out = base.copy()
    for k, c in COLORS.items():
        m = res["classes"] == k
        out[m] = (1 - alpha) * base[m] + alpha * np.array(c, "float32")
    return out.astype("uint8")


def summary(res):
    keys = ["verdict", "mode", "flood_km2", "flood_pct", "confirmed_by_both_pct", "verdict_flood_km2",
            "flood_in_seasonal_zone_km2", "possible_flood_pct", "permanent_water_km2", "seasonal_water_km2",
            "permanent_water_source", "valid_pct", "vv_threshold_db", "unet_thresholds", "notes"]
    return {k: res[k] for k in keys}


def save(res, out_dir, transform=None, crs=None, name="result"):
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay(res)).save(out / f"{name}_overlay.png")
    Image.fromarray(sar_gray(res["vv_lee_db"], res["valid"])).save(out / f"{name}_sar.png")
    json.dump(summary(res), open(out / f"{name}_summary.json", "w"), indent=2)
    if transform is not None and crs is not None:
        import rasterio
        H, W = res["classes"].shape
        prof = dict(driver="GTiff", height=H, width=W, count=1, crs=crs, transform=transform, compress="deflate", tiled=True)
        with rasterio.open(out / f"{name}_classes.tif", "w", dtype="uint8", nodata=255, **prof) as o:
            o.write(res["classes"], 1)
            o.update_tags(values="0 dry, 1 possible flood, 2 flood, 3 permanent water, 4 seasonal water, 255 no data")
        if res["prob"] is not None:
            with rasterio.open(out / f"{name}_unet_prob.tif", "w", dtype="uint8", nodata=255, **prof) as o:
                o.write(np.where(res["valid"], np.round(res["prob"] * 100), 255).astype("uint8"), 1)
    return out
