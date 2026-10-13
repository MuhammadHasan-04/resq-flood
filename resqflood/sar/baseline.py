"""Baselines the U-Net must beat: Otsu threshold and fixed dB threshold on VV."""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu


def metrics(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    p, y = pred[valid], truth[valid]
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"IoU": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0, "precision": round(prec, 4),
            "recall": round(rec, 4), "F1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4), "pixels": int(valid.sum())}


def otsu_vv(aligned: dict, sample_step: int = 10) -> dict:
    vv, valid, y = aligned["vv"], aligned["valid"], aligned["label"] == 1
    t = float(threshold_otsu(vv[valid][::sample_step]))
    return {"method": "otsu_vv", "threshold_db": round(t, 2), **metrics(vv < t, y, valid)}


def fixed_vv(aligned: dict, threshold_db: float = -18.0) -> dict:
    vv, valid, y = aligned["vv"], aligned["valid"], aligned["label"] == 1
    return {"method": f"fixed_vv_{threshold_db}", "threshold_db": threshold_db, **metrics(vv < threshold_db, y, valid)}
