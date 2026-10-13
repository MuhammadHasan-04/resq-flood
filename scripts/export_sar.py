"""Download and build the Sentinel-1 VV/VH dB GeoTIFFs for one TU Wien mask date.
python scripts/export_sar.py --date 20220908_A042
Writes <DADU_DATA>/02_Sentinel1_SAR/matched_scenes/<date>/dadu_S1_<date>_{VV,VH}_dB.tif
"""
import argparse

import _common  # noqa: F401
import geopandas as gpd

from resqflood.config import load_config
from resqflood.sar.export_planetary import download_tiles, search_scenes
from resqflood.sar.preprocess import mosaic_to_db

ap = argparse.ArgumentParser()
ap.add_argument("--date", required=True)
a = ap.parse_args()
cfg = load_config()
bnd = gpd.read_file(cfg.data(cfg["study_area"]["boundary"]))
x0, y0, x1, y1 = bnd.total_bounds
bbox = (round(x0 - 0.04, 2), round(y0 - 0.03, 2), round(x1 + 0.03, 2), round(y1 + 0.03, 2))
items = search_scenes(cfg, a.date, bbox)
print("scenes:", [i["id"] for i in items])
out = cfg.data(cfg["sar"]["out_dir"]) / a.date
for band in ("vv", "vh"):
    tiles = []
    for it in items:
        tiles += download_tiles(cfg, it["id"], bbox, out / "raw_tiles", bands=(band,))
    p = mosaic_to_db(tiles, out / f"dadu_S1_{a.date}_{band.upper()}_dB.tif", clip_geom=bnd.geometry)
    print("wrote", p)
