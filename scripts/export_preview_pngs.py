"""Export viewable PNGs of EVERY chip: the raw image as downloaded and the same image after pre-processing,
with the same file name in parallel folders, so they can be browsed side by side in Windows Explorer.

  python scripts/export_preview_pngs.py                     # all datasets (~1,100 chips, a few minutes)
  python scripts/export_preview_pngs.py --only tuwien       # one dataset: tuwien | s1f | dry

Output (inside FLOOD_TRAINING_DATA/04_preview_png/):
  1_RAW/<dataset>/<chip>.png         VV exactly as downloaded (linear backscatter, speckle, no processing)
  2_PROCESSED/<dataset>/<chip>.png   VV after pre-processing (dB + Lee 7x7) = what the model receives
  3_LABEL/<dataset>/<chip>.png       label on the same pixels (blue water, red flood-only*, beige land, grey ignore)
  README.txt                         how the folders match
(* permanent water is shown dark blue, flood water light blue)
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DS = {"tuwien": "tuwien_pakistan_2022", "s1f": "sen1floods11_pakistan_2017", "dry": "dry_season_2022"}


def to_png(img8, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8).save(path, optimize=True)


def stretch(a, lo, hi):
    g = np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    return (g * 255).astype("uint8")


def read_raw_vv(src):
    """src: path or bytes of the downloaded GeoTIFF -> VV linear (NaN = no data)."""
    with rasterio.open(io.BytesIO(src) if isinstance(src, bytes) else src) as s:
        a = s.read(1).astype("float32")
    a[(a <= -30000) | ~np.isfinite(a)] = np.nan
    if np.nanmedian(a) < 0:      # Sen1Floods11 is delivered in dB
        a = 10 ** (a / 10)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "outputs" / "FLOOD_TRAINING_DATA"))
    ap.add_argument("--s1f-raw", default=str(ROOT.parent / "outputs" / "datasets" / "sen1floods11_all_pakistan"))
    ap.add_argument("--only", choices=list(DS))
    a = ap.parse_args()
    data = Path(a.data)
    out = data / "04_preview_png"
    rows = list(csv.DictReader(open(data / "02_splits" / "manifest.csv")))
    raw_dir = data / "00_raw_downloads"
    zips = {}
    for z in sorted(glob.glob(str(raw_dir / "sentinel1_rtc_chips_tuwien" / "*.zip"))):
        for n in zipfile.ZipFile(z).namelist():
            zips[os.path.basename(n).split("_")[0]] = (z, n)
    s1f = {Path(p).stem.split("_")[1]: p for p in glob.glob(str(Path(a.s1f_raw) / "**" / "S1*" / "*.tif"), recursive=True)
           if "S1Hand" in p or "S1Weak" in p}
    lab_cols = np.array([[233, 228, 212], [120, 170, 240], [154, 154, 154], [20, 70, 200]], "uint8")
    n_done = {k: 0 for k in DS.values()}
    open_zip = {}
    for r in rows:
        ds = r["dataset"]
        if a.only and ds != DS[a.only]:
            continue
        cid = r["chip_id"]
        try:
            if ds == DS["tuwien"]:
                z, n = zips[cid]
                open_zip.setdefault(z, zipfile.ZipFile(z))
                raw = read_raw_vv(open_zip[z].read(n))
            elif ds == DS["dry"]:
                raw = read_raw_vv(str(raw_dir / "sentinel1_rtc_dry_season_2022" / f"{cid}.tif"))
            else:
                raw = read_raw_vv(s1f[cid.split("_")[-1]])
        except Exception as e:
            print(f"[skip raw] {cid}: {type(e).__name__} {e}")
            raw = None
        with rasterio.open(data / "01_chips" / r["sar_file"]) as s:
            vv_lee = s.read(3).astype("float32")
        vv_lee[vv_lee <= -30000] = np.nan
        vv_lee /= 100.0
        with rasterio.open(data / "01_chips" / r["label_file"]) as s:
            water, perm = s.read(1), s.read(3)
        if raw is not None:
            to_png(stretch(raw, 0, float(np.nanpercentile(raw, 99))), out / "1_RAW" / ds / f"{cid}.png")
        to_png(stretch(vv_lee, -22, -2), out / "2_PROCESSED" / ds / f"{cid}.png")
        idx = np.where(water == 255, 2, np.where((water == 1) & (perm == 1), 3, water)).astype("uint8")
        to_png(lab_cols[idx], out / "3_LABEL" / ds / f"{cid}.png")
        n_done[ds] += 1
    (out / "README.txt").write_text(
        "Every chip has the SAME file name in each folder, so you can compare them directly:\n\n"
        "  1_RAW\\<dataset>\\<chip>.png        Sentinel-1 VV exactly as downloaded (linear, speckle, no processing)\n"
        "  2_PROCESSED\\<dataset>\\<chip>.png  the same pixels after pre-processing: dB + Lee 7x7 filter (model input)\n"
        "  3_LABEL\\<dataset>\\<chip>.png      label on the same pixels: light blue = flood water, dark blue = permanent\n"
        "                                     river/lake, beige = dry land, grey = ignored (unknown)\n\n"
        "Datasets: tuwien_pakistan_2022 (542 flood-season chips), sen1floods11_pakistan_2017 (276),\n"
        "dry_season_2022 (282). Split, date and QC of every chip: ..\\02_splits\\manifest.csv\n"
        "The actual data files (GeoTIFF) are in ..\\00_raw_downloads (raw) and ..\\01_chips (processed).\n")
    print("done:", n_done, "->", out)


if __name__ == "__main__":
    main()
