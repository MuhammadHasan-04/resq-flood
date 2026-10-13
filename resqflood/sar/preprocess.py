"""Mosaic SAR tiles, convert to dB, and align SAR to a flood mask on the same pixel grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

NODATA_IN = -32768.0   # Planetary Computer RTC fill value
NODATA_DB = -9999.0


def mosaic_to_db(tile_files: list[Path], out_path: Path, clip_geom=None) -> Path:
    srcs = [rasterio.open(f) for f in tile_files]
    arr, T = merge(srcs, nodata=NODATA_IN, method="first")
    lin = arr[0].astype("float32")
    ok = lin > 0
    db = np.where(ok, 10 * np.log10(np.where(ok, lin, 1)), NODATA_DB).astype("float32")
    prof = dict(driver="GTiff", height=db.shape[0], width=db.shape[1], count=1, dtype="float32", crs=srcs[0].crs,
                transform=T, nodata=NODATA_DB, compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
    for s in srcs:
        s.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as o:
        o.write(db, 1)
    if clip_geom is not None:
        from rasterio.mask import mask
        with rasterio.open(out_path) as s:
            c, t = mask(s, clip_geom, crop=True, nodata=NODATA_DB)
            p = s.profile
        p.update(height=c.shape[1], width=c.shape[2], transform=t)
        with rasterio.open(out_path, "w", **p) as o:
            o.write(c)
    return out_path


def align_to_mask(sar_paths: dict, mask_path: Path) -> dict:
    """Resample SAR bands onto the mask grid. Returns {'vv','vh','label','valid','transform','crs'}."""
    with rasterio.open(mask_path) as m:
        label = m.read(1)
        T, crs, shape = m.transform, m.crs, label.shape
    out = {"label": label, "transform": T, "crs": crs}
    for band, p in sar_paths.items():
        dst = np.full(shape, NODATA_DB, "float32")
        with rasterio.open(p) as s:
            reproject(s.read(1), dst, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=crs,
                      resampling=Resampling.bilinear, src_nodata=s.nodata, dst_nodata=NODATA_DB)
        out[band] = dst
    out["valid"] = (label != 255) & np.all([out[b] != NODATA_DB for b in sar_paths], axis=0)
    return out
