"""Run the trained flood U-Net on a full Sentinel-1 scene and write flood / water GeoTIFFs.

  python scripts/predict_scene.py --ckpt runs\\unet_r34_lee\\best.pt --vv VV.tif --vh VH.tif --out runs\\pred_dadu_20220830 \
      --norm ..\\outputs\\FLOOD_TRAINING_DATA\\02_splits\\norm_stats.json --jrc-dir ..\\outputs\\datasets\\jrc_global_surface_water_pakistan

Input units are detected automatically: linear gamma0/sigma0, dB, or dB x 100 (int16, as in DADU_DATA).
The scene is resampled to the training grid (0.00018 deg, ~20 m), Lee 7x7 filtered in linear units exactly as in
training, z-scored, predicted with overlapping 512 px windows (+ optional TTA), and thresholded.
Outputs: water_prob.tif (0-100), water_mask.tif, flood_mask.tif (water minus JRC permanent water, if --jrc-dir).
--method otsu runs the threshold fallback instead of the U-Net (no checkpoint needed).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, calculate_default_transform, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resqflood.sar import training_data as td  # noqa: E402
from resqflood.train.core import build_unet, predict_prob  # noqa: E402


def read_linear(path, res):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        nod = s.nodata
        bad = ~np.isfinite(a)
        if nod is not None:
            bad |= a == nod
        bad |= a <= -30000
        v = a[~bad]
        if s.dtypes[0] in ("int16", "int32") and np.median(v) < -100:
            lin = 10 ** (a / 1000.0)          # dB x 100
        elif np.median(v) < 0:
            lin = 10 ** (a / 10.0)            # dB
        else:
            lin = a                          # linear
        lin[bad] = np.nan
        T, W, H = calculate_default_transform(s.crs, "EPSG:4326", s.width, s.height, *s.bounds, resolution=res)
        out = np.full((H, W), np.nan, "float32")
        reproject(lin, out, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs="EPSG:4326",
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return out, T


def sliding(models, x, device, tile=512, overlap=128, tta=True):
    C, H, W = x.shape
    step = tile - overlap
    acc = np.zeros((H, W), "float32")
    wsum = np.zeros((H, W), "float32")
    w1 = np.hanning(tile + 2)[1:-1].astype("float32")
    win = np.maximum(np.outer(w1, w1), 1e-3)
    def starts(n):
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        return s if s[-1] == n - tile else s + [n - tile]
    for r in starts(H):
        for c in starts(W):
            patch = x[:, r:r + tile, c:c + tile]
            p = predict_prob(models, patch, device, tta=tta)
            ww = win[:p.shape[0], :p.shape[1]]
            acc[r:r + p.shape[0], c:c + p.shape[1]] += p * ww
            wsum[r:r + p.shape[0], c:c + p.shape[1]] += ww
    return acc / np.maximum(wsum, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vv", required=True)
    ap.add_argument("--vh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--norm", help="02_splits/norm_stats.json (required for the U-Net)")
    ap.add_argument("--jrc-dir", help="folder with JRC occurrence_*.tif / seasonality_*.tif")
    ap.add_argument("--threshold", type=float, default=0, help="0 = use the value from eval/metrics.json if found, else 0.5")
    ap.add_argument("--method", default="unet", choices=["unet", "otsu"])
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vv, T = read_linear(a.vv, td.PIX_DEG)
    vh, _ = read_linear(a.vh, td.PIX_DEG)
    stack, valid = td.sar_stack(vv, vh, enl=4.4)          # dB: VV, VH, VV_lee, VH_lee
    H, W = valid.shape
    print(f"scene resampled to {W} x {H} px at {td.PIX_DEG} deg, valid {valid.mean() * 100:.1f}%")

    if a.method == "unet":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models, cfg = [], None
        for p in a.ckpt:
            ck = torch.load(p, map_location=device, weights_only=False)
            cfg = ck["config"]
            m, _ = build_unet(cfg["encoder"], cfg["in_channels"], "none", cfg.get("attention", "scse"))
            m.load_state_dict(ck["model"])
            models.append(m.to(device).eval())
        if not models:
            raise SystemExit("--ckpt is required for --method unet")
        st = json.load(open(a.norm))
        clip = st["clip_db"]
        names = {"lee": [("VV_lee7_dB", 2, "VV"), ("VH_lee7_dB", 3, "VH"), ("VV_minus_VH_lee7_dB", None, None)],
                 "raw": [("VV_dB", 0, "VV"), ("VH_dB", 1, "VH"), ("VV_minus_VH_dB", None, None)]}[cfg["channels"]]
        chans = []
        for n, b, cl in names:
            if b is None:
                v = stack[names[0][1]] - stack[names[1][1]]
            else:
                v = np.clip(stack[b], *clip[cl])
            z = (v - st["channels"][n]["mean"]) / st["channels"][n]["std"]
            chans.append(np.clip(np.where(valid, z, 0), -6, 6))
        x = np.stack(chans).astype("float32")
        thr = a.threshold
        mj = Path(a.ckpt[0]).parent / "eval" / "metrics.json"
        if not thr:
            thr = json.load(open(mj))["threshold"] if mj.exists() else 0.5
        prob = sliding(models, x, device, tta=not a.no_tta)
        water = (prob >= thr) & valid
        print(f"U-Net threshold {thr}")
    else:
        v = stack[2][valid]
        h, e = np.histogram(v, 256)
        c = (e[:-1] + e[1:]) / 2
        w = np.cumsum(h)
        m = np.cumsum(h * c)
        wf = w[-1] - w
        thr_db = c[np.argmax(w * wf * (m / np.maximum(w, 1) - (m[-1] - m) / np.maximum(wf, 1)) ** 2)]
        prob = np.where(stack[2] < thr_db, 1.0, 0.0)
        water = (stack[2] < thr_db) & valid
        print(f"Otsu fallback threshold {thr_db:.2f} dB (VV Lee)")

    prof = dict(driver="GTiff", height=H, width=W, count=1, crs="EPSG:4326", transform=T, compress="deflate", tiled=True)
    with rasterio.open(out / "water_prob.tif", "w", dtype="uint8", nodata=255, **prof) as o:
        o.write(np.where(valid, np.round(prob * 100), 255).astype("uint8"), 1)
    with rasterio.open(out / "water_mask.tif", "w", dtype="uint8", nodata=255, **prof) as o:
        o.write(np.where(valid, water, 255).astype("uint8"), 1)
    if a.jrc_dir:
        occ = sorted(glob.glob(os.path.join(a.jrc_dir, "**", "occurrence_*.tif"), recursive=True))
        sea = sorted(glob.glob(os.path.join(a.jrc_dir, "**", "seasonality_*.tif"), recursive=True))
        perm, _ = td.JRC(occ, sea).permanent(T, "EPSG:4326", (H, W))
        flood = water & (perm != 1)
        with rasterio.open(out / "flood_mask.tif", "w", dtype="uint8", nodata=255, **prof) as o:
            o.write(np.where(valid, flood, 255).astype("uint8"), 1)
        px_km2 = (td.PIX_DEG * 111.32) ** 2 * np.cos(np.deg2rad(T.f + T.e * H / 2))
        print(f"water {water.sum() * px_km2:.1f} km2, permanent {((perm == 1) & water).sum() * px_km2:.1f} km2, "
              f"flood {flood.sum() * px_km2:.1f} km2")
    print("saved", out)


if __name__ == "__main__":
    main()
