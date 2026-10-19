"""Detect floods in any SAR input with the fused detector (U-Net + fallback) and save the results.

  python scripts/predict_scene.py --model runs/BEST/best_model.json --input VV.tif VH.tif --out runs/pred_dadu
  python scripts/predict_scene.py --model runs/BEST/best_model.json --input scene_2band.tif --out runs/pred_x
  python scripts/predict_scene.py --model runs/BEST/best_model.json --input quicklook.png --pixel-m 20 --out runs/pred_png
  python scripts/predict_scene.py --model ... --input flood.png --location "67.6 26.6 67.8 26.8" --out runs/pred_png_located
  python scripts/predict_scene.py --model ... --input flood.png --reference before_flood.png --out runs/pred_change
  python scripts/predict_scene.py --fallback-only --input VV.tif VH.tif --out runs/pred_fb      (no model needed)

Outputs in --out: result_overlay.png, result_sar.png, result_summary.json and, for GeoTIFF input,
result_classes.tif (0 dry, 1 possible, 2 flood, 3 permanent water, 4 seasonal water) + result_unet_prob.tif.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resqflood import render  # noqa: E402
from resqflood.detect import FloodDetector  # noqa: E402
from resqflood.sar_io import align_to, load_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="GeoTIFF(s) (VV [+VH]) or a PNG/JPG")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="runs/BEST/best_model.json or a run folder with best.pt + eval/metrics.json")
    ap.add_argument("--norm", default="../outputs/FLOOD_TRAINING_DATA/02_splits/norm_stats.json")
    ap.add_argument("--jrc-dir", default="../outputs/datasets/jrc_global_surface_water_pakistan")
    ap.add_argument("--fallback-only", action="store_true")
    ap.add_argument("--pixel-m", type=float, default=20.0, help="pixel size of PNG/JPG inputs in metres")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--location", default="", help='PNG/JPG location: "west south east north" or "lat, lon" '
                    '(centre; uses --pixel-m). Sidecar .pgw/.bbox.txt files are read automatically.')
    ap.add_argument("--reference", nargs="+", help="PRE-FLOOD image(s) of the same area: its water counts as permanent")
    a = ap.parse_args()
    if a.fallback_only or not a.model:
        det = FloodDetector(jrc_dir=a.jrc_dir)
    else:
        det = FloodDetector.from_model_file(a.model, jrc_dir=a.jrc_dir, norm_stats=a.norm, tta=not a.no_tta)
        if det.error:
            print("[warn] U-Net could not be loaded:", det.error)
    data = load_any(a.input, a.pixel_m, a.location)
    rvv = rvh = None
    if a.reference:
        ref = load_any(a.reference, a.pixel_m, a.location)
        rvv, rvh, note = align_to(ref, data)
        data["notes"].append(note)
    res = det.detect(data["vv"], data["vh"], data["transform"], data["crs"], ref_vv_lin=rvv, ref_vh_lin=rvh)
    res["notes"] = data["notes"] + res["notes"]
    render.save(res, a.out, data["transform"], data["crs"])
    print(json.dumps(render.summary(res), indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
