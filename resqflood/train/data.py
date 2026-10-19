"""In-memory chip datasets, augmentation and sampling for U-Net training.

All chips of a split are read once (float16, ~2 MB per chip for 3 channels), so the GPU is never
waiting on disk and num_workers can stay 0 on Windows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from resqflood.sar.dataset import CHANNELS, load_chip

IGNORE = 255


def read_manifest(root):
    return list(csv.DictReader(open(Path(root) / "02_splits" / "manifest.csv")))


def select_rows(rows, split, include_aux=False, train_qc=True, include_dry=False):
    keep = {split}
    if include_aux and split == "train":
        keep.add("train_aux_s1f_weak")
    if include_dry and split in ("train", "val"):
        keep.add(split + "_dry")
    out = []
    for r in rows:
        if r["split"] not in keep:
            continue
        if r["split"].startswith("train") and train_qc and r["use_for_training"] != "True":
            continue
        out.append(r)
    return out


class ChipCache:
    """Loads x (C,H,W float16, z-scored), water/flood labels and the permanent-water band for a list of chips."""

    def __init__(self, root, rows, channels="lee", s1f_scale=0.5, verbose=True, use_disk_cache=True):
        self.root = Path(root)
        self.stats = json.load(open(self.root / "02_splits" / "norm_stats.json"))
        self.rows, self.channels = rows, channels
        self.x, self.water, self.flood, self.perm = [], [], [], []
        t = time.time()
        key = hashlib.md5((channels + str(s1f_scale) + "|".join(r["chip_id"] for r in rows)).encode()).hexdigest()[:12]
        cache_f = self.root / "cache" / f"chips_{channels}_{key}.npz"
        if use_disk_cache and cache_f.exists():
            z = np.load(cache_f, allow_pickle=True)
            self.x, self.water, self.flood, self.perm = (list(z[k]) for k in ("x", "water", "flood", "perm"))
            if verbose:
                print(f"  loaded {len(rows)} chips from cache {cache_f.name} ({time.time() - t:.0f}s)", flush=True)
            return
        c = self.root / "01_chips"
        for i, r in enumerate(rows):
            x, _ = load_chip(c / r["sar_file"], None, channels, self.stats)
            with rasterio.open(c / r["label_file"]) as s:
                lab = s.read()                       # water, flood, permanent, valid
            nod = np.all(x == 0, axis=0)
            w, f, p = lab[0].copy(), lab[1].copy(), lab[2].copy()
            w[nod] = IGNORE
            f[nod] = IGNORE
            if r["dataset"].startswith("sen1floods11") and s1f_scale != 1:
                # Sen1Floods11 is 10 m; the model is trained on ~20 m pixels -> resample to the same scale
                x, w, f, p = _rescale(x, w, f, p, s1f_scale)
            self.x.append(np.clip(x, -6, 6).astype("float16"))
            self.water.append(w)
            self.flood.append(f)
            self.perm.append(p)
            if verbose and (i + 1) % 100 == 0:
                print(f"  loaded {i + 1}/{len(rows)} chips ({time.time() - t:.0f}s)", flush=True)
        if verbose:
            print(f"  loaded {len(rows)} chips in {time.time() - t:.0f}s", flush=True)
        if use_disk_cache:
            try:
                cache_f.parent.mkdir(exist_ok=True)
                obj = lambda L: np.array(L, dtype=object) if len({a.shape for a in L}) > 1 else np.stack(L)  # noqa: E731
                np.savez(cache_f, x=obj(self.x), water=obj(self.water), flood=obj(self.flood), perm=obj(self.perm))
            except Exception as e:
                print("[warn] could not write chip cache:", e)

    def target(self, i, name):
        return self.water[i] if name == "water" else self.flood[i]


def _rescale(x, w, f, p, s):
    k = int(round(1 / s))
    C, H, W = x.shape
    x2 = x[:, :H // k * k, :W // k * k].reshape(C, H // k, k, W // k, k).mean((2, 4))

    def lab(a):
        return a[: H // k * k: k, : W // k * k: k].copy()
    return x2.astype("float32"), lab(w), lab(f), lab(p)


class TrainChips(Dataset):
    """Random crops with flips/rot90, calibration jitter and water-biased crop centres."""

    def __init__(self, cache: ChipCache, target="water", crop=384, epoch_mult=2, water_bias=0.5,
                 gain_db=1.0, noise=0.03, seed=42, scale=(0.75, 1.33)):
        self.c, self.target, self.crop = cache, target, crop
        self.scale = scale
        self.n = len(cache.rows) * epoch_mult
        self.water_bias, self.gain_db, self.noise = water_bias, gain_db, noise
        self.rng = np.random.default_rng(seed)
        st = cache.stats["channels"]
        names = CHANNELS[cache.channels]
        # a calibration offset of d dB shifts VV and VH bands equally (ratio bands unchanged)
        self.gain_vec = np.array([(1.0 / st[n]["std"]) if ("minus" not in n) else 0.0 for n in names], "float32")
        flood_frac = np.array([float(r["water_frac"] if target == "water" else r["flood_frac"]) for r in cache.rows])
        w = np.where(flood_frac > 0.01, 2.0, 1.0)
        self.chip_p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        rng = self.rng
        i = rng.choice(len(self.chip_p), p=self.chip_p)
        x = self.c.x[i]
        y = self.c.target(i, self.target)
        H, W = y.shape
        out_cs = min(self.crop, H, W)
        # scale augmentation: crop a larger/smaller window and resize to out_cs (robust to 10-30 m pixels)
        sc = rng.uniform(*self.scale) if self.scale and self.scale[1] > self.scale[0] else 1.0
        cs = int(np.clip(round(out_cs / sc), 64, min(H, W)))
        if rng.random() < self.water_bias and (y == 1).any():
            ys, xs = np.nonzero(y == 1)
            j = rng.integers(len(ys))
            r0 = int(np.clip(ys[j] - cs // 2 + rng.integers(-cs // 4, cs // 4 + 1), 0, H - cs))
            c0 = int(np.clip(xs[j] - cs // 2 + rng.integers(-cs // 4, cs // 4 + 1), 0, W - cs))
        else:
            r0, c0 = int(rng.integers(0, H - cs + 1)), int(rng.integers(0, W - cs + 1))
        xc = x[:, r0:r0 + cs, c0:c0 + cs].astype("float32")
        yc = y[r0:r0 + cs, c0:c0 + cs]
        k = int(rng.integers(4))
        xc, yc = np.rot90(xc, k, (1, 2)), np.rot90(yc, k)
        if rng.random() < 0.5:
            xc, yc = xc[:, :, ::-1], yc[:, ::-1]
        if self.gain_db > 0:
            valid = np.any(xc != 0, axis=0)
            xc = xc + (rng.uniform(-self.gain_db, self.gain_db) * self.gain_vec)[:, None, None] * valid
        if self.noise > 0:
            xc = xc + rng.normal(0, self.noise, xc.shape).astype("float32")
        xt = torch.from_numpy(np.ascontiguousarray(xc, dtype="float32"))
        yt = torch.from_numpy(np.ascontiguousarray(yc).astype("int64"))
        if cs != out_cs:
            xt = torch.nn.functional.interpolate(xt[None], size=(out_cs, out_cs), mode="bilinear", align_corners=False)[0]
            yt = torch.nn.functional.interpolate(yt[None, None].float(), size=(out_cs, out_cs), mode="nearest")[0, 0].long()
        return xt, yt


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))


def pad_to(x, mult=32):
    """Pad a (B,C,H,W) tensor so H and W are multiples of `mult` (U-Net needs /32)."""
    H, W = x.shape[-2:]
    ph, pw = (math.ceil(H / mult) * mult - H), (math.ceil(W / mult) * mult - W)
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    return x, H, W
