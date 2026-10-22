"""Before/after pre-processing images for several chips + an index (manual) with every file path.

  python scripts/make_before_after.py                         # the default demo set (6 chips)
  python scripts/make_before_after.py --chips TW0385 TW0028   # your own chip ids (from 02_splits/manifest.csv)

For every chip it writes  <data>/03_report/before_after/<chip>/
  1_RAW_as_downloaded.png      Sentinel-1 VV exactly as downloaded (linear gamma0, speckle)
  2_RAW_in_dB_no_filter.png    the same pixels converted to dB only
  3_PROCESSED_dB_Lee7x7.png    dB + Lee 7x7 speckle filter = what the model receives
  4_LABEL_aligned.png          the label on the same pixels (blue water, beige land, grey ignore)
  0_SIDE_BY_SIDE.png           all four in one picture (for slides)
  RAW_downloaded.tif           the original downloaded file (open in QGIS)
and a manual:  <data>/03_report/before_after/INDEX.md  (+ index.csv) listing every path and the numbers.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = [  # chip, why it is interesting
    ("TW0385", "Dadu-area flood, 15 Sep 2022 (test set)"),
    ("TW0138", "Flood next to the Indus, Dadu/Johi (test set)"),
    ("TW0208", "North Sindh flood, 3 Sep 2022 (training set)"),
    ("TW0028", "Small flood, about 3% of the chip"),
    ("DRY_TW0016_20220420", "Dry season (Apr 2022): river only, no flood"),
    ("DRY_TW0017_20220216", "Dry season (Feb 2022): dry land"),
]


def win_path(p: Path) -> str:
    """/mnt/c/Users/... -> C:\\Users\\... (WSL); other paths unchanged."""
    s = str(p.resolve())
    if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
        return s[5].upper() + ":\\" + s[7:].replace("/", "\\")
    return s


def otsu(v):
    v = v[np.isfinite(v)]
    h, e = np.histogram(v, 256)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    wf = w[-1] - w
    return float(c[np.argmax(w * wf * (m / np.maximum(w, 1) - (m[-1] - m) / np.maximum(wf, 1)) ** 2)])


def iou(pred, lab):
    ok, t = lab != 255, lab == 1
    u = ((pred | t) & ok).sum()
    return float((pred & t & ok).sum() / u) if u else float("nan")


def find_raw(data: Path, cid: str, out: Path):
    """Copy the originally downloaded file of this chip to out; returns the path or None."""
    raw = data / "00_raw_downloads"
    if cid.startswith("DRY_"):
        f = raw / "sentinel1_rtc_dry_season_2022" / f"{cid}.tif"
        if f.exists():
            out.write_bytes(f.read_bytes())
            return out
        return None
    for z in sorted(glob.glob(str(raw / "sentinel1_rtc_chips_tuwien" / "*.zip"))):
        with zipfile.ZipFile(z) as zz:
            hit = [n for n in zz.namelist() if os.path.basename(n).startswith(cid + "_")]
            if hit:
                out.write_bytes(zz.read(hit[0]))
                return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "outputs" / "FLOOD_TRAINING_DATA"))
    ap.add_argument("--chips", nargs="*", help="chip ids (default: a demo set of 6)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    data = Path(a.data)
    man = {r["chip_id"]: r for r in csv.DictReader(open(data / "02_splits" / "manifest.csv"))}
    todo = [(c, "") for c in a.chips] if a.chips else DEFAULT
    base = data / "03_report" / "before_after"
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    cmap_lab = ListedColormap(["#e9e4d4", "#1f5fd1", "#9a9a9a"])
    for cid, why in todo:
        r = man.get(cid)
        if r is None:
            print(f"[skip] {cid}: not in manifest.csv")
            continue
        d = base / cid
        d.mkdir(exist_ok=True)
        with rasterio.open(data / "01_chips" / r["sar_file"]) as s:
            p = s.read().astype("float32")
        p[p <= -30000] = np.nan
        p /= 100.0
        vv_db, vv_lee = p[0], p[2]
        with rasterio.open(data / "01_chips" / r["label_file"]) as s:
            lab = s.read(1)
        raw_tif = find_raw(data, cid, d / "RAW_downloaded.tif")
        if raw_tif:
            with rasterio.open(raw_tif) as s:
                raw = s.read(1).astype("float32")
            raw[raw <= -30000] = np.nan
            if np.nanmedian(raw) < 0:          # some sources are already in dB
                raw = 10 ** (raw / 10)
        else:
            raw = 10 ** (vv_db / 10)
        panels = [
            (raw, "1_RAW_as_downloaded.png", "RAW: VV as downloaded\n(linear, speckle, no processing)", 0, np.nanpercentile(raw, 99)),
            (vv_db, "2_RAW_in_dB_no_filter.png", "Step 1: converted to dB\n(no filter yet)", -22, -2),
            (vv_lee, "3_PROCESSED_dB_Lee7x7.png", "PROCESSED: dB + Lee 7x7 filter\n(what the model receives)", -22, -2),
        ]
        for img, fn, title, lo, hi in panels:
            plt.figure(figsize=(6, 6.4))
            plt.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            plt.title(title, fontsize=11)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(d / fn, dpi=120)
            plt.close()
        plt.figure(figsize=(6, 6.4))
        plt.imshow(np.where(lab == 255, 2, lab), cmap=cmap_lab, vmin=0, vmax=2, interpolation="nearest")
        plt.title("LABEL on the same pixels\n(blue water, beige land, grey ignore)", fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(d / "4_LABEL_aligned.png", dpi=120)
        plt.close()
        # numbers
        land, water = lab == 0, lab == 1
        sp_raw = float(np.nanstd(vv_db[land])) if land.any() else float("nan")
        sp_lee = float(np.nanstd(vv_lee[land])) if land.any() else float("nan")
        i_raw = iou(np.nan_to_num(vv_db, nan=99) < otsu(vv_db), lab)
        i_lee = iou(np.nan_to_num(vv_lee, nan=99) < otsu(vv_lee), lab)
        # side by side
        fig, ax = plt.subplots(1, 4, figsize=(20, 5.6))
        for k, (img, _, title, lo, hi) in enumerate(panels):
            ax[k].imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax[k].set_title(title, fontsize=11)
        ax[3].imshow(np.where(lab == 255, 2, lab), cmap=cmap_lab, vmin=0, vmax=2, interpolation="nearest")
        ax[3].set_title("LABEL (blue = water)", fontsize=11)
        for x in ax:
            x.axis("off")
        fig.suptitle(f"{cid}  ({r['date']}, {r['split']})  {why}   |   speckle on land {sp_raw:.2f} -> {sp_lee:.2f} dB"
                     f"   |   Otsu IoU vs label {i_raw:.2f} -> {i_lee:.2f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(d / "0_SIDE_BY_SIDE.png", dpi=100)
        plt.close()
        rows.append(dict(chip=cid, what=why, date=r["date"], split=r["split"], water_pct=round(100 * float(r["water_frac"]), 1),
                         speckle_raw_db=round(sp_raw, 2), speckle_lee_db=round(sp_lee, 2),
                         otsu_iou_raw=round(i_raw, 3), otsu_iou_lee=round(i_lee, 3),
                         folder=win_path(d), processed_tif=win_path(data / "01_chips" / r["sar_file"]),
                         label_tif=win_path(data / "01_chips" / r["label_file"]),
                         raw_tif=win_path(raw_tif) if raw_tif else "(raw file not found; dB band used)"))
        print(f"{cid}: speckle {sp_raw:.2f} -> {sp_lee:.2f} dB, Otsu IoU {i_raw:.3f} -> {i_lee:.3f}  -> {d}")
    if not rows:
        return
    with open(base / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    L = ["# Before / after pre-processing: image index", "",
         "Each chip folder contains:",
         "- `0_SIDE_BY_SIDE.png`: all steps in one picture (best for slides)",
         "- `1_RAW_as_downloaded.png`: Sentinel-1 VV exactly as downloaded (linear, speckle)",
         "- `2_RAW_in_dB_no_filter.png`: converted to dB only",
         "- `3_PROCESSED_dB_Lee7x7.png`: dB + Lee 7x7 filter, i.e. what the model receives",
         "- `4_LABEL_aligned.png`: the flood label on exactly the same pixels",
         "- `RAW_downloaded.tif`: the original downloaded file (open in QGIS)", "",
         "Speckle = standard deviation of VV on land pixels (lower = less noise). "
         "Otsu IoU = how well the same automatic threshold matches the label (higher = water/land easier to separate).", ""]
    for x in rows:
        L += [f"## {x['chip']}: {x['what'] or x['split']}",
              f"- Date {x['date']}, split {x['split']}, water {x['water_pct']} % of the chip",
              f"- Speckle on land: {x['speckle_raw_db']} dB raw -> {x['speckle_lee_db']} dB after Lee",
              f"- Otsu IoU vs label: {x['otsu_iou_raw']} raw -> {x['otsu_iou_lee']} after Lee",
              f"- Images folder: `{x['folder']}`",
              f"- Before (raw): `{x['folder']}\\1_RAW_as_downloaded.png`",
              f"- After (processed): `{x['folder']}\\3_PROCESSED_dB_Lee7x7.png`",
              f"- Side by side: `{x['folder']}\\0_SIDE_BY_SIDE.png`",
              f"- Raw data file: `{x['raw_tif']}`",
              f"- Processed data file: `{x['processed_tif']}` (band 1 VV dB, band 3 VV Lee; values / 100 = dB)",
              f"- Label file: `{x['label_tif']}`", ""]
    (base / "INDEX.md").write_text("\n".join(L))
    print("index:", base / "INDEX.md")


if __name__ == "__main__":
    main()
