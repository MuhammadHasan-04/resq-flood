"""Build the U-Net training set (TU Wien 2022 Pakistan + Sen1Floods11 Pakistan 2017).

Usage:
  python scripts/build_training_data.py --tuwien-sar DIR --tuwien-masks DIR --chips chip_candidates.csv \
      --s1f DIR --jrc DIR --out FLOOD_TRAINING_DATA/01_chips [--only tuwien|s1f] [--start N --stop M]

Every chip gets two GeoTIFFs with the same grid:
  sar/<id>.tif     int16 x4 : VV_dB, VH_dB, VV_lee7_dB, VH_lee7_dB   (value / 100 = dB, -32768 = nodata)
  labels/<id>.tif  uint8 x4 : water, flood, permanent_water, valid   (0/1, 255 = ignore)
and one row in <out>/<dataset>_chips.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resqflood.sar import training_data as td  # noqa: E402

ENL = {"tuwien": 4.4, "s1f": 4.4}   # ESA nominal ENL for IW GRDH; data-driven estimates in 04_report/enl_estimates.json


def otsu(x):
    x = x[np.isfinite(x)]
    if x.size < 100:
        return np.nan
    h, e = np.histogram(x, 256)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    wb, wf = w, w[-1] - w
    mb = m / np.maximum(wb, 1)
    mf = (m[-1] - m) / np.maximum(wf, 1)
    return float(c[np.argmax(wb * wf * (mb - mf) ** 2)])


def iou(pred, lab):
    ok = lab != 255
    p, t = pred[ok], lab[ok] == 1
    u = (p | t).sum()
    return float((p & t).sum() / u) if u else np.nan


def qc(stack, labels):
    """Otsu on VV (raw and Lee) vs the water label: a sanity check that SAR and label line up."""
    out = {}
    for name, b in (("raw", 0), ("lee", 2)):
        thr = otsu(stack[b])
        out[f"otsu_thr_{name}"] = round(thr, 2) if np.isfinite(thr) else ""
        out[f"otsu_iou_{name}"] = round(iou(np.nan_to_num(stack[b], nan=99) < thr, labels[0]), 4) if np.isfinite(thr) else ""
    v = labels[3] == 1
    w = (labels[0] == 1) & v
    d = v & (labels[0] == 0)
    out["vv_water_mean_db"] = round(float(np.nanmean(stack[2][w])), 2) if w.sum() > 50 else ""
    out["vv_land_mean_db"] = round(float(np.nanmean(stack[2][d])), 2) if d.sum() > 50 else ""
    return out


def jrc_reader(jrc_dir):
    occ = sorted(glob.glob(os.path.join(jrc_dir, "**", "occurrence_*.tif"), recursive=True))
    sea = sorted(glob.glob(os.path.join(jrc_dir, "**", "seasonality_*.tif"), recursive=True))
    return td.JRC(occ, sea), [Path(p).name for p in occ + sea]


def build_tuwien(a, jrc, out):
    rows = list(csv.DictReader(open(a.chips)))[a.start:a.stop]
    (out / "sar").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    res = []
    for r in rows:
        cid = r["chip_id"]
        src = glob.glob(os.path.join(a.tuwien_sar, f"{cid}_*.tif"))[0]
        with rasterio.open(src) as s:
            lin = s.read().astype("float32")
            T, crs = s.transform, s.crs
        lin[lin <= -30000] = np.nan
        stack, sar_ok = td.sar_stack(lin[0], lin[1], ENL["tuwien"])
        tuw = td.tuwien_label_on_grid(os.path.join(a.tuwien_masks, r["mask_file"]), int(r["bi"]), int(r["bj"]),
                                      T, crs, lin.shape[1:])
        perm, _ = jrc.permanent(T, crs, lin.shape[1:])
        # TU Wien excludes its reference (permanent) water as "masked" (255); JRC tells us it is water.
        lab_ok = (tuw != 255) | (perm == 1)
        labels = td.make_labels(tuw, lab_ok, sar_ok, perm, label_includes_permanent=False)
        tags = dict(source="Sentinel-1 RTC gamma0 (Planetary Computer) + TU Wien FLOOD-HM-MASKED",
                    acquisition=r["ts"], orbit=r["orbit"], equi7_tile=r["tile"])
        td.write_chip(out / "sar" / f"{cid}.tif", out / "labels" / f"{cid}.tif", stack, labels, T, crs, tags)
        st = td.chip_stats(labels)
        res.append(dict(chip_id=cid, dataset="tuwien_pakistan_2022", label_type="tuwien_auto",
                        date=r["ts"][:8], acquisition=r["ts"], orbit=r["orbit"], tile=r["tile"],
                        lon=r["lon"], lat=r["lat"], sample_bin=r["bin"], perm_source="JRC_GSW_v1.4",
                        sar_file=f"tuwien_pakistan_2022/sar/{cid}.tif",
                        label_file=f"tuwien_pakistan_2022/labels/{cid}.tif", **st, **qc(stack, labels)))
    return res


def build_s1f(a, jrc, out):
    base = Path(a.s1f)
    items = []
    for p in sorted(glob.glob(str(base / "**/S1Hand/*.tif"), recursive=True)):
        items.append(("hand", p))
    for p in sorted(glob.glob(str(base / "**/S1Weak/*.tif"), recursive=True)):
        items.append(("weak", p))
    items = items[a.start:a.stop]
    (out / "sar").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    def find(name):
        g = glob.glob(str(base / f"**/{name}"), recursive=True)
        return g[0] if g else None

    res = []
    for kind, p in items:
        num = Path(p).stem.split("_")[1]
        cid = f"S1F_{kind}_{num}"
        try:
            with rasterio.open(p) as s:
                db = s.read().astype("float32")
                T, crs = s.transform, s.crs
        except Exception as e:  # truncated download -> excluded and reported
            print("SKIP corrupt", p, e)
            with open(out / "skipped_corrupt.txt", "a") as o:
                o.write(f"{Path(p).name}\t{e}\n")
            continue
        lin =np.where(np.isfinite(db), 10 ** (db / 10), np.nan).astype("float32")
        stack, sar_ok = td.sar_stack(lin[0], lin[1], ENL["s1f"])
        shape = db.shape[1:]
        if kind == "hand":
            with rasterio.open(find(f"Pakistan_{num}_LabelHand.tif")) as s:
                lab = s.read(1)
            with rasterio.open(find(f"Pakistan_{num}_JRCWaterHand.tif")) as s:
                perm = s.read(1).astype("uint8")
            perm_source = "Sen1Floods11 JRCWaterHand"
            ltype = "hand"
        else:
            with rasterio.open(find(f"Pakistan_{num}_S2IndexLabelWeak.tif")) as s:
                lab = s.read(1)
            perm, _ = jrc.permanent(T, crs, shape)
            cov = float((perm != 255).mean())
            perm_source = "JRC_GSW_v1.4" if cov > 0.999 else ("JRC_GSW_v1.4 (partial)" if cov > 0 else "none (outside JRC tiles)")
            ltype = "weak_S2index"
        lab_ok = lab != -1
        labels = td.make_labels((lab == 1).astype("uint8"), lab_ok, sar_ok, perm, label_includes_permanent=True)
        tags = dict(source="Sen1Floods11 v1.1 (S1 GRD sigma0, GEE) Pakistan", label=ltype)
        td.write_chip(out / "sar" / f"{cid}.tif", out / "labels" / f"{cid}.tif", stack, labels, T, crs, tags)
        st = td.chip_stats(labels)
        lon = (T.c + T.a * shape[1] / 2)
        lat = (T.f + T.e * shape[0] / 2)
        res.append(dict(chip_id=cid, dataset="sen1floods11_pakistan_2017", label_type=ltype, date="2017",
                        acquisition="", orbit="", tile="", lon=round(lon, 5), lat=round(lat, 5), sample_bin="",
                        perm_source=perm_source, sar_file=f"sen1floods11_pakistan_2017/sar/{cid}.tif",
                        label_file=f"sen1floods11_pakistan_2017/labels/{cid}.tif", **st, **qc(stack, labels)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuwien-sar")
    ap.add_argument("--tuwien-masks")
    ap.add_argument("--chips")
    ap.add_argument("--s1f")
    ap.add_argument("--jrc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["tuwien", "s1f"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    jrc, jrc_files = jrc_reader(a.jrc)
    for name, fn, sub in (("tuwien", build_tuwien, "tuwien_pakistan_2022"), ("s1f", build_s1f, "sen1floods11_pakistan_2017")):
        if a.only and a.only != name:
            continue
        rows = fn(a, jrc, out / sub)
        if not rows:
            continue
        f = out / f"{sub}_chips_{a.start:04d}.csv"
        with open(f, "w", newline="") as o:
            w = csv.DictWriter(o, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows), "->", f)


if __name__ == "__main__":
    main()
