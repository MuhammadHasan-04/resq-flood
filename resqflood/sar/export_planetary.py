"""Download Sentinel-1 RTC (VV, VH) for Dadu from Microsoft Planetary Computer, tile by tile.

This is how the 30 Aug 2022 scene in DADU_DATA was produced. The data API returns a GeoTIFF crop per
request, so no account is needed. Some networks block the host; if so, run this on another connection
or use the GEE script in DADU_DATA/02_Sentinel1_SAR/c_matching_scene_list_for_TUWien.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

TILE_W, TILE_H = 0.32, 0.33     # degrees per request
PIX = 0.0003                    # ~30 m


def search_scenes(cfg, date_key: str, bbox) -> list[dict]:
    """Find RTC items on the acquisition day of a TU Wien mask key such as '20220830_D078'."""
    day = datetime.strptime(date_key[:8], "%Y%m%d")
    body = {"collections": [cfg["sar"]["collection"]], "bbox": list(bbox),
            "datetime": f"{day:%Y-%m-%d}T00:00:00Z/{day + timedelta(days=1):%Y-%m-%d}T00:00:00Z", "limit": 50}
    r = requests.post(cfg["sar"]["stac_url"] + "/search", json=body, timeout=60)
    r.raise_for_status()
    orbit = int(date_key.split("_")[1][1:]) if "_" in date_key else None
    feats = r.json()["features"]
    if orbit is not None:
        feats = [f for f in feats if f["properties"].get("sat:relative_orbit") == orbit] or feats
    return feats


def download_tiles(cfg, item_id: str, bbox, out_dir: Path, bands=("vv", "vh")) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = bbox
    files = []
    nx_, ny_ = int(np.ceil((x1 - x0) / TILE_W)), int(np.ceil((y1 - y0) / TILE_H))
    w, h = int(round(TILE_W / PIX)), int(round(TILE_H / PIX))
    for band in bands:
        for r in range(ny_):
            for c in range(nx_):
                b = (x0 + c * TILE_W, y0 + r * TILE_H, x0 + (c + 1) * TILE_W, y0 + (r + 1) * TILE_H)
                f = out_dir / f"{item_id}_{band}_r{r}c{c}.tif"
                if f.exists():
                    files.append(f)
                    continue
                url = (f"{cfg['sar']['data_api']}/item/bbox/{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}/{w}x{h}.tif"
                       f"?collection={cfg['sar']['collection']}&item={item_id}&assets={band}&return_mask=false")
                for attempt in range(3):
                    try:
                        resp = requests.get(url, timeout=180)
                        resp.raise_for_status()
                        f.write_bytes(resp.content)
                        files.append(f)
                        break
                    except requests.RequestException:
                        if attempt == 2:
                            raise
    return files
