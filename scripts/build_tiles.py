"""Align a date's SAR to its TU Wien mask, report baselines, and cut training tiles.
python scripts/build_tiles.py --date 20220830_D078 --vv <path> --vh <path>
(--vv/--vh default to the matched_scenes output of export_sar.py)
"""
import argparse
import json

import _common  # noqa: F401

from resqflood.config import load_config
from resqflood.sar.baseline import fixed_vv, otsu_vv
from resqflood.sar.preprocess import align_to_mask
from resqflood.sar.tiling import make_tiles

ap = argparse.ArgumentParser()
ap.add_argument("--date", required=True)
ap.add_argument("--vv"); ap.add_argument("--vh")
ap.add_argument("--no-tiles", action="store_true")
a = ap.parse_args()
cfg = load_config()
base = cfg.data(cfg["sar"]["out_dir"]) / a.date
vv = a.vv or base / f"dadu_S1_{a.date}_VV_dB.tif"
vh = a.vh or base / f"dadu_S1_{a.date}_VH_dB.tif"
al = align_to_mask({"vv": vv, "vh": vh}, cfg.flood_mask(a.date))
res = {"date": a.date, "baselines": [otsu_vv(al), fixed_vv(al)]}
if not a.no_tiles:
    res["tiles"] = make_tiles(al, a.date, cfg.work_dir / "tiles", cfg["sar"]["split"], cfg["sar"]["tile_size"])
(cfg.work_dir / f"baseline_{a.date}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
