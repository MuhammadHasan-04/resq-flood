"""Geographic train/val/test split + QC flags + normalisation statistics for the built chips.

Usage: python scripts/make_splits.py --chips-dir FLOOD_TRAINING_DATA/01_chips --out FLOOD_TRAINING_DATA/02_splits
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# 1-degree cells (lon_floor, lat_floor). Test holds out Dadu/Johi (the ResQFlood pilot district) + one
# northern-Sindh cell, val holds out the lower Indus (Hyderabad) + the Punjab/Cholistan cells.
TEST_CELLS = {(67, 26), (67, 27), (69, 27)}
VAL_CELLS = {(68, 25), (70, 28), (70, 29), (71, 29), (66, 25), (66, 26)}
CLIP = {"VV": (-23.0, 0.0), "VH": (-28.0, -5.0)}     # Yadav et al. 2022 ranges
HALF = 0.04608


def boxes_overlap(a, b, h=HALF):
    return abs(a[0] - b[0]) < 2 * h and abs(a[1] - b[1]) < 2 * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cd, out = Path(a.chips_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.concat([pd.read_csv(p) for p in sorted(cd.glob("tuwien_pakistan_2022_chips_*.csv"))], ignore_index=True)
    s = pd.concat([pd.read_csv(p) for p in sorted(cd.glob("sen1floods11_pakistan_2017_chips_*.csv"))], ignore_index=True)

    cell = list(zip(np.floor(t.lon).astype(int), np.floor(t.lat).astype(int)))
    t["cell"] = [f"{x}E_{y}N" for x, y in cell]
    t["split"] = ["test" if c in TEST_CELLS else "val" if c in VAL_CELLS else "train" for c in cell]
    # spatial buffer: a training chip may not overlap any val/test chip footprint (any date)
    held = t[t.split != "train"][["lon", "lat"]].values
    drop = [i for i, r in t.iterrows() if r.split == "train" and any(boxes_overlap((r.lon, r.lat), h) for h in held)]
    t.loc[t.index.isin(drop) & (t.split == "train"), "split"] = "excluded_buffer"

    s["cell"] = ""
    s["split"] = np.where(s.label_type == "hand", "test_external_s1f_hand", "train_aux_s1f_weak")

    df = pd.concat([t, s], ignore_index=True)
    # QC flags
    flags = []
    for _, r in df.iterrows():
        f = []
        if r.valid_frac < 0.5:
            f.append("low_valid")
        if r.water_frac > 0.01 and pd.notna(r.vv_water_mean_db) and r.vv_water_mean_db > r.vv_land_mean_db - 2:
            f.append("label_sar_disagree")
        if str(r.perm_source).startswith("none"):
            f.append("no_permanent_water_ref")
        flags.append(";".join(f))
    df["qc_flags"] = flags
    df["use_for_training"] = df.split.isin(["train", "train_aux_s1f_weak"]) & ~df.qc_flags.str.contains("low_valid|label_sar_disagree")
    df.to_csv(out / "manifest.csv", index=False)
    for sp in sorted(df.split.unique()):
        (out / f"{sp}.txt").write_text("\n".join(df[df.split == sp].chip_id) + "\n")

    # normalisation statistics on the TU Wien training split (clipped dB), 20k random valid px per chip
    rng = np.random.default_rng(42)
    samples = {k: [] for k in ["VV_dB", "VH_dB", "VV_lee7_dB", "VH_lee7_dB", "VV_minus_VH_dB", "VV_minus_VH_lee7_dB"]}
    tr = df[(df.split == "train") & (df.dataset == "tuwien_pakistan_2022")]
    for _, r in tr.iterrows():
        with rasterio.open(cd / r.sar_file) as src:
            x = src.read().astype("float32")
        ok = (x != -32768).all(0)
        idx = np.flatnonzero(ok)
        idx = rng.choice(idx, min(20000, idx.size), replace=False)
        v = x.reshape(4, -1)[:, idx] / 100
        vv, vh, vvl, vhl = v
        samples["VV_dB"].append(np.clip(vv, *CLIP["VV"]))
        samples["VH_dB"].append(np.clip(vh, *CLIP["VH"]))
        samples["VV_lee7_dB"].append(np.clip(vvl, *CLIP["VV"]))
        samples["VH_lee7_dB"].append(np.clip(vhl, *CLIP["VH"]))
        samples["VV_minus_VH_dB"].append(vv - vh)
        samples["VV_minus_VH_lee7_dB"].append(vvl - vhl)
    stats = {}
    for k, v in samples.items():
        v = np.concatenate(v)
        stats[k] = dict(mean=round(float(v.mean()), 4), std=round(float(v.std()), 4),
                        p1=round(float(np.percentile(v, 1)), 2), p99=round(float(np.percentile(v, 99)), 2))
    # class balance on training labels
    bal = {}
    for lab in ("water", "flood"):
        pos = tot = 0
        for _, r in tr.iterrows():
            with rasterio.open(cd / r.label_file) as src:
                l = src.read(1 if lab == "water" else 2)
            pos += int((l == 1).sum())
            tot += int((l != 255).sum())
        bal[lab] = dict(positive_px=pos, labelled_px=tot, positive_frac=round(pos / tot, 4),
                        suggested_pos_weight=round((tot - pos) / max(pos, 1), 2))
    json.dump(dict(computed_on="TU Wien train split, clipped dB, 20k px per chip, seed 42",
                   clip_db=CLIP, channels=stats, class_balance=bal,
                   recommended_input=["VV_lee7_dB", "VH_lee7_dB", "VV_minus_VH_lee7_dB"],
                   ignore_index=255), open(out / "norm_stats.json", "w"), indent=2)
    print(df.groupby(["dataset", "split"]).size())
    print(df.qc_flags.value_counts())
    print(json.dumps(bal, indent=1))


if __name__ == "__main__":
    main()
