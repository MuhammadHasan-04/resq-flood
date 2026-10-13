"""Least-cost boat path on the pilot cost grid (default launch near Dadu -> Jhandiani).
python scripts/boat_demo.py --date 20220830_D078
"""
import argparse
import json

import _common  # noqa: F401
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import LineString

from resqflood.boat.costgrid import boat_path, build_cost_grid
from resqflood.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None)
ap.add_argument("--start", default="67.775,26.735")
ap.add_argument("--end", default="67.781454,26.588606")
a = ap.parse_args()
cfg = load_config()
key = a.date or cfg["flood"]["default_date"]
roads = gpd.read_file(cfg.data(cfg["roads"]["file"]))
g = build_cost_grid(cfg, cfg.flood_mask(key), roads)
res = boat_path(g, tuple(map(float, a.start.split(","))), tuple(map(float, a.end.split(","))))
print(json.dumps({k: v for k, v in res.items() if k != "coords"}, indent=1), "| flood level proxy", round(g["flood_level_m"], 1), "m")
gpd.GeoDataFrame({"length_km": [res["length_km"]]}, geometry=[LineString(res["coords"])], crs=4326).to_file(cfg.work_dir / f"boat_{key}.geojson", driver="GeoJSON")
H, W = g["cost"].shape
with rasterio.open(cfg.work_dir / f"costgrid_{key}.tif", "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                   crs="EPSG:4326", transform=g["transform"], compress="deflate") as o:
    o.write(g["cost"], 1)
print("wrote", cfg.work_dir / f"boat_{key}.geojson")
