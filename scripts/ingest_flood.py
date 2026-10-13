"""Score all Dadu roads against one dated flood mask.
python scripts/ingest_flood.py --date 20220908_A042
"""
import argparse
import json

import _common  # noqa: F401
import geopandas as gpd

from resqflood.config import load_config
from resqflood.floodmask.ingest import ingest_mask, summary

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None, help="mask key, e.g. 20220830_D078 (default from config)")
ap.add_argument("--mask", default=None, help="path to any flood mask GeoTIFF (e.g. U-Net output) instead of --date")
a = ap.parse_args()
cfg = load_config()
mask = a.mask or cfg.flood_mask(a.date)
key = a.date or cfg["flood"]["default_date"]
roads = gpd.read_file(cfg.data(cfg["roads"]["file"]))
scored = ingest_mask(roads, mask, cfg)
out = cfg.work_dir / f"roads_scored_{key}.geojson"
scored.to_file(out, driver="GeoJSON")
s = summary(scored)
(cfg.work_dir / f"roads_summary_{key}.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s, indent=1)); print("wrote", out)
