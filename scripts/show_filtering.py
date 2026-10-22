"""Show a training chip BEFORE and AFTER the Lee 7x7 speckle filter (for demos / the mid evaluation).

  python scripts/show_filtering.py                      # picks a flooded test chip from the Dadu area
  python scripts/show_filtering.py --chip TW0208        # any chip id from manifest.csv
  python scripts/show_filtering.py --chip TW0208 --live # also re-runs lee_filter() live on the raw data

Saves runs/filtering_demo/<chip>_before_after.png and prints the numbers.

What the figure shows
  row 1: VV raw (dB) | VV after Lee 7x7 | the label (blue = water, grey = ignore)
  row 2: VH raw (dB) | VH after Lee 7x7 | histograms of water vs land pixels, raw vs filtered
The filter itself is resqflood/sar/training_data.py -> lee_filter().
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resqflood.sar import training_data as td  # noqa: E402


def otsu(v):
    v = v[np.isfinite(v)]
    h, e = np.histogram(v, 256)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    wf = w[-1] - w
    return float(c[np.argmax(w * wf * (m / np.maximum(w, 1) - (m[-1] - m) / np.maximum(wf, 1)) ** 2)])


def iou(pred, lab):
    ok = lab != 255
    t = lab == 1
    return float((pred & t & ok).sum() / max(((pred | t) & ok).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "outputs" / "FLOOD_TRAINING_DATA"))
    ap.add_argument("--chip", default="", help="chip id, e.g. TW0208 (default: a flooded Dadu test chip)")
    ap.add_argument("--live", action="store_true", help="re-run lee_filter() on the raw values and compare")
    ap.add_argument("--out", default=str(ROOT / "runs" / "filtering_demo"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(Path(a.data) / "02_splits" / "manifest.csv")))
    if a.chip:
        r = next((x for x in rows if x["chip_id"] == a.chip), None)
        if r is None:
            raise SystemExit(f"chip {a.chip} not in manifest.csv")
    else:
        r = max((x for x in rows if x["split"] == "test" and 0.2 < float(x["water_frac"]) < 0.6),
                key=lambda x: float(x["otsu_iou_lee"] or 0) - float(x["otsu_iou_raw"] or 0))
    c = Path(a.data) / "01_chips"
    with rasterio.open(c / r["sar_file"]) as s:
        sar = s.read().astype("float32")
        desc = s.descriptions
    sar[sar <= -30000] = np.nan
    sar /= 100.0                                   # stored as dB x 100
    vv_raw, vh_raw, vv_lee, vh_lee = sar
    with rasterio.open(c / r["label_file"]) as s:
        lab = s.read(1)

    # ---- the numbers
    print(f"chip {r['chip_id']}  date {r['date']}  split {r['split']}  bands {desc}")
    water, land = (lab == 1), (lab == 0)
    for name, raw, lee in (("VV", vv_raw, vv_lee), ("VH", vh_raw, vh_lee)):
        print(f"  {name}: speckle (std on land)  raw {np.nanstd(raw[land]):.2f} dB -> Lee {np.nanstd(lee[land]):.2f} dB | "
              f"water {np.nanmean(raw[water]):.1f} dB, land {np.nanmean(raw[land]):.1f} dB")
    t_raw, t_lee = otsu(vv_raw), otsu(vv_lee)
    i_raw, i_lee = iou(np.nan_to_num(vv_raw, nan=99) < t_raw, lab), iou(np.nan_to_num(vv_lee, nan=99) < t_lee, lab)
    print(f"  same Otsu threshold method vs label: raw IoU {i_raw:.3f} (thr {t_raw:.1f} dB) -> Lee IoU {i_lee:.3f} (thr {t_lee:.1f} dB)")

    if a.live:   # prove the stored band = lee_filter() applied to the raw data
        lin = 10 ** (vv_raw / 10)
        valid = np.isfinite(lin)
        live = td.to_db(td.lee_filter(np.where(valid, lin, np.nan), valid, 7, 4.4))
        d = np.nanmax(np.abs(live - vv_lee))
        print(f"  live lee_filter() vs stored VV_lee7 band: max difference {d:.3f} dB (int16 rounding = 0.005 dB)")

    # ---- the figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.6))
    for j, (img, title) in enumerate(((vv_raw, "VV BEFORE (raw, speckle)"), (vv_lee, "VV AFTER Lee 7x7"))):
        ax[0, j].imshow(img, cmap="gray", vmin=-22, vmax=-2)
        ax[0, j].set_title(title)
    for j, (img, title) in enumerate(((vh_raw, "VH BEFORE (raw)"), (vh_lee, "VH AFTER Lee 7x7"))):
        ax[1, j].imshow(img, cmap="gray", vmin=-28, vmax=-8)
        ax[1, j].set_title(title)
    ax[0, 2].imshow(np.where(lab == 255, 2, lab), cmap=ListedColormap(["#e9e4d4", "#1f5fd1", "#9a9a9a"]),
                    vmin=0, vmax=2, interpolation="nearest")
    ax[0, 2].set_title("label: blue water, beige land, grey ignore")
    bins = np.linspace(-30, 0, 120)
    h = ax[1, 2]
    h.hist(vv_raw[water & np.isfinite(vv_raw)], bins, alpha=.35, color="tab:blue", label="water, raw")
    h.hist(vv_raw[land & np.isfinite(vv_raw)], bins, alpha=.35, color="tab:orange", label="land, raw")
    h.hist(vv_lee[water & np.isfinite(vv_lee)], bins, histtype="step", lw=2, color="navy", label="water, Lee")
    h.hist(vv_lee[land & np.isfinite(vv_lee)], bins, histtype="step", lw=2, color="darkred", label="land, Lee")
    h.set_title(f"VV histogram: Otsu IoU raw {i_raw:.2f} -> Lee {i_lee:.2f}")
    h.set_xlabel("dB")
    h.legend(fontsize=8)
    for x in ax.ravel()[[0, 1, 2, 3, 4]]:
        x.axis("off")
    fig.suptitle(f"{r['chip_id']} ({r['date']}, {r['split']}): before vs after speckle filtering", fontsize=13)
    plt.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{r['chip_id']}_before_after.png"
    plt.savefig(f, dpi=110)
    print("saved", f)


if __name__ == "__main__":
    main()
