"""Run the fused flood detector on hand-picked test chips and save a visual report in a separate folder.

  python scripts/test_cases.py --model runs/BEST/best_model.json --data ../outputs/FLOOD_TRAINING_DATA --out runs/TEST_CASES

Picks chips that were NEVER used in training: Dadu/Johi floods, other test floods, dry chips (must say "no flood"),
low-flood chips, and Sen1Floods11 hand-labelled chips from 2017. For each chip it saves SAR / truth / detection
images and a row in test_cases.csv (verdict vs truth, IoU), plus one overview image test_cases_overview.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resqflood import render  # noqa: E402
from resqflood.detect import FloodDetector  # noqa: E402
from resqflood.train.core import Confusion  # noqa: E402
from resqflood.train.data import read_manifest  # noqa: E402

DADU = (67.14, 26.21, 68.04, 27.47)


def pick(rows, n_each=3):
    test = [r for r in rows if r["split"] == "test"]
    ext = [r for r in rows if r["split"] == "test_external_s1f_hand"]
    ff = lambda r: float(r["flood_frac"])  # noqa: E731
    dadu = [r for r in test if DADU[0] <= float(r["lon"]) <= DADU[2] and DADU[1] <= float(r["lat"]) <= DADU[3]]
    out = {}
    out["Dadu flood"] = sorted(dadu, key=ff)[-n_each:]
    other = [r for r in test if r not in dadu]
    out["Other test flood"] = sorted(other, key=ff)[-n_each:]
    out["Low flood (<10%)"] = [r for r in sorted(test, key=ff) if 0.02 < ff(r) < 0.10][:n_each]
    out["Dry (no flood)"] = [r for r in sorted(test, key=ff) if ff(r) < 0.005][:n_each]
    out["Sen1Floods11 hand 2017"] = sorted(ext, key=lambda r: float(r["water_frac"]))[-n_each:]
    return out


def load_chip(root, r):
    with rasterio.open(Path(root) / "01_chips" / r["sar_file"]) as s:
        a = s.read().astype("float32")
        T, crs = s.transform, s.crs
    with rasterio.open(Path(root) / "01_chips" / r["label_file"]) as s:
        lab = s.read()
    nod = (a == -32768).any(0)
    vv = np.where(nod, np.nan, 10 ** (a[0] / 1000.0))
    vh = np.where(nod, np.nan, 10 ** (a[1] / 1000.0))
    if r["dataset"].startswith("sen1floods11"):          # 10 m -> 20 m like in training
        k = 2
        H, W = vv.shape
        vv = np.nanmean(vv[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k), axis=(1, 3))
        vh = np.nanmean(vh[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k), axis=(1, 3))
        lab = lab[:, :H // k * k:k, :W // k * k:k]
        T = T * T.scale(k, k)
    return vv.astype("float32"), vh.astype("float32"), T, crs, lab


def load_pre(path):
    """Pre-monsoon chip: linear VV/VH; truth = no flood anywhere (water unknown)."""
    with rasterio.open(path) as s:
        a = s.read().astype("float32")
        T, crs = s.transform, s.crs
    a[a <= -30000] = np.nan
    v = np.isfinite(a).all(0)
    lab = np.stack([np.full(v.shape, 255, "uint8"), np.where(v, 0, 255).astype("uint8"),
                    np.zeros(v.shape, "uint8"), v.astype("uint8")])
    return a[0], a[1], T, crs, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="../outputs/FLOOD_TRAINING_DATA")
    ap.add_argument("--jrc-dir", default="../outputs/datasets/jrc_global_surface_water_pakistan")
    ap.add_argument("--out", default="runs/TEST_CASES")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    det = FloodDetector.from_model_file(a.model, jrc_dir=a.jrc_dir,
                                        norm_stats=str(Path(a.data) / "02_splits" / "norm_stats.json"))
    print("detector mode:", det.mode, ("| " + det.error) if det.error else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = pick(read_manifest(a.data), a.n)
    pre_dir = Path(a.data) / "00_raw_downloads" / "sentinel1_rtc_pre_flood_2022"
    if (pre_dir / "pre_flood_meta.json").exists():
        meta = json.load(open(pre_dir / "pre_flood_meta.json"))
        ids = [m[0] for m in meta if m[1] == "test"][: a.n]
        groups["River/lake, NO flood (Apr 2022)"] = [dict(chip_id=f.stem, date=f.stem.split("_")[-1], lon="", lat="",
                                                          flood_frac="0", pre_file=str(f))
                                                     for i in ids for f in sorted(pre_dir.glob(f"PRE_{i}_*.tif"))]
    rows, tiles = [], []
    for g, rs in groups.items():
        for r in rs:
            if "pre_file" in r:
                vv, vh, T, crs, lab = load_pre(r["pre_file"])
            else:
                vv, vh, T, crs, lab = load_chip(a.data, r)
            res = det.detect(vv, vh, T, crs)
            d = out / f"{r['chip_id']}"
            render.save(res, d, T, crs, name=r["chip_id"])
            water, flood = lab[0], lab[1]
            v = flood != 255
            truth_flood = (flood == 1).sum() >= 0.01 * max(v.sum(), 1)
            c_u, c_f = Confusion(), Confusion()
            if res["prob"] is not None:
                c_u.add((res["classes"] == 2) | (res["classes"] == 3) | (res["classes"] == 4), water)
            c_f.add((res["classes"] == 2) | (res["classes"] == 4), flood)
            pred_flood = res["verdict"] == "FLOOD DETECTED"
            row = dict(group=g, chip_id=r["chip_id"], date=r["date"], lon=r["lon"], lat=r["lat"],
                       truth_flood_pct=round(100 * float(r["flood_frac"]), 1), truth="flood" if truth_flood else "no flood",
                       verdict=res["verdict"], correct=pred_flood == truth_flood,
                       flood_pct=res["flood_pct"], possible_pct=res["possible_flood_pct"],
                       flood_km2=res["flood_km2"], confirmed_by_both_pct=res["confirmed_by_both_pct"], water_iou=c_u.result()["iou"],
                       flood_iou=c_f.result()["iou"], flood_precision=c_f.result()["precision"],
                       mode=res["mode"])
            rows.append(row)
            truth_rgb = render.sar_gray(res["vv_lee_db"], res["valid"]).copy()
            truth_rgb[water == 1] = (30, 120, 230)
            truth_rgb[flood == 1] = (220, 20, 30)
            truth_rgb[water == 255] = (128, 128, 128)
            tiles.append((f"{r['chip_id']} · {g}", render.sar_gray(res["vv_lee_db"], res["valid"]), truth_rgb,
                          render.overlay(res), row))
            print(f"{g:24s} {r['chip_id']:20s} truth {row['truth']:8s} -> {res['verdict']:18s} "
                  f"flood {res['flood_pct']:5.1f}%  flood IoU {row['flood_iou']:.3f}")
    with open(out / "test_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    acc = np.mean([r["correct"] for r in rows])
    json.dump(dict(n=len(rows), verdict_accuracy=round(float(acc), 3), mode=det.mode,
                   params=det.params, vv_threshold_db=det.vv_threshold_db),
              open(out / "test_cases_summary.json", "w"), indent=2)
    _overview(tiles, out / "test_cases_overview.png")
    print(f"\nflood yes/no verdict correct on {acc * 100:.0f}% of {len(rows)} test cases -> {out}")


def _overview(tiles, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(tiles)
    fig, ax = plt.subplots(n, 3, figsize=(10.5, 3.4 * n))
    ax = np.atleast_2d(ax)
    for i, (t, sar, truth, det, row) in enumerate(tiles):
        ax[i, 0].imshow(sar)
        ax[i, 0].set_title(t, fontsize=8)
        ax[i, 1].imshow(truth)
        ax[i, 1].set_title(f"truth: {row['truth']} ({row['truth_flood_pct']}%)\nred flood, blue water, grey unknown", fontsize=8)
        ax[i, 2].imshow(det)
        ok = "correct" if row["correct"] else "WRONG"
        ax[i, 2].set_title(f"{row['verdict']} ({ok})\nred flood, orange possible, blue permanent, cyan seasonal", fontsize=8)
    for x in ax.ravel():
        x.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=80)
    plt.close()


if __name__ == "__main__":
    main()
