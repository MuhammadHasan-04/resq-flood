"""PyTorch dataset for the preprocessed flood chips (FLOOD_TRAINING_DATA).

    from resqflood.sar.dataset import FloodChips
    ds = FloodChips(root, split="train", target="water", channels="lee")
    x, y = ds[0]     # x: float32 [C,512,512] z-scored, y: int64 [512,512] in {0,1,255}

Channels: "lee" -> VV_lee7, VH_lee7, VV-VH (lee); "raw" -> VV, VH, VV-VH (raw); "all" -> the 4 dB bands + both ratios.
Normalisation = clip (VV -23..0 dB, VH -28..-5 dB) then z-score with 02_splits/norm_stats.json; nodata -> 0 after
z-scoring and label 255. Use CrossEntropyLoss(ignore_index=255) or BCE with a mask.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

TARGET_BAND = {"water": 1, "flood": 2}
CHANNELS = {
    "lee": ["VV_lee7_dB", "VH_lee7_dB", "VV_minus_VH_lee7_dB"],
    "raw": ["VV_dB", "VH_dB", "VV_minus_VH_dB"],
    "all": ["VV_dB", "VH_dB", "VV_lee7_dB", "VH_lee7_dB", "VV_minus_VH_dB", "VV_minus_VH_lee7_dB"],
}


def load_chip(sar_path, label_path=None, channels="lee", stats=None, target="water"):
    with rasterio.open(sar_path) as s:
        a = s.read().astype("float32")
    nod = (a == -32768).any(0)
    a = a / 100.0
    vv, vh, vvl, vhl = a
    clip = stats["clip_db"] if stats else {"VV": (-23, 0), "VH": (-28, -5)}
    bands = {"VV_dB": np.clip(vv, *clip["VV"]), "VH_dB": np.clip(vh, *clip["VH"]),
             "VV_lee7_dB": np.clip(vvl, *clip["VV"]), "VH_lee7_dB": np.clip(vhl, *clip["VH"]),
             "VV_minus_VH_dB": vv - vh, "VV_minus_VH_lee7_dB": vvl - vhl}
    x = []
    for n in CHANNELS[channels]:
        b = bands[n]
        if stats:
            b = (b - stats["channels"][n]["mean"]) / stats["channels"][n]["std"]
        x.append(np.where(nod, 0, b))
    x = np.stack(x).astype("float32")
    y = None
    if label_path is not None:
        with rasterio.open(label_path) as s:
            y = s.read(TARGET_BAND[target]).astype("int64")
        y[nod] = 255
    return x, y


class FloodChips:
    def __init__(self, root, split="train", target="water", channels="lee", include_aux=False, transform=None):
        self.root = Path(root)
        self.stats = json.load(open(self.root / "02_splits" / "norm_stats.json"))
        rows = list(csv.DictReader(open(self.root / "02_splits" / "manifest.csv")))
        keep = {split} | ({"train_aux_s1f_weak"} if include_aux and split == "train" else set())
        # QC filtering only on training data; evaluation splits keep every chip (no cherry-picking)
        self.rows = [r for r in rows if r["split"] in keep and
                     (r["use_for_training"] == "True" if split == "train" else True)]
        self.target, self.channels, self.transform = target, channels, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        c = self.root / "01_chips"
        x, y = load_chip(c / r["sar_file"], c / r["label_file"], self.channels, self.stats, self.target)
        if self.transform:
            x, y = self.transform(x, y)
        try:
            import torch
            return torch.from_numpy(x), torch.from_numpy(y)
        except ImportError:
            return x, y
