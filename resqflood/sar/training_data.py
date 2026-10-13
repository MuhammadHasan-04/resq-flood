"""Build U-Net training chips: Sentinel-1 VV/VH (raw + Lee-filtered, dB) with flood / water / permanent-water labels.

Design decisions (see docs/PREPROCESSING.md for the sources):
* Speckle: classic Lee filter, 7x7 window, applied to LINEAR backscatter (the multiplicative noise
  model only holds in linear units), then converted to dB. Raw dB is kept as well so the effect of the
  filter can be tested during training (ablation).
* Units: dB = 10*log10(gamma0 or sigma0). Stored as int16 = round(dB*100), nodata -32768.
* Permanent water: JRC Global Surface Water v1.4 - seasonality == 12 months OR occurrence >= 80 %.
  label_water = flood OR permanent water, label_flood = flood with permanent water set to 255 (ignore).
* Invalid pixels (SAR nodata, TU Wien "masked" class 255, Sen1Floods11 -1) are 255 in every label band.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

IGNORE = 255
DB_SCALE = 100.0
NODATA_I16 = -32768
SAR_BANDS = ["VV_dB", "VH_dB", "VV_lee7_dB", "VH_lee7_dB"]
LABEL_BANDS = ["water", "flood", "permanent_water", "valid"]
CHIP = 512
PIX_DEG = 0.00018           # ~20 m, close to the TU Wien 20 m Equi7 grid
HALF_DEG = PIX_DEG * CHIP / 2


# ---------------------------------------------------------------- speckle
def lee_filter(lin: np.ndarray, valid: np.ndarray, size: int = 7, enl: float = 4.4) -> np.ndarray:
    """Classic Lee (1980) filter on linear intensity.

    k = var_x / var_z with var_x = (var_z - mean^2 * Cu^2) / (1 + Cu^2), Cu^2 = 1/ENL.
    Invalid pixels are excluded from the local statistics (normalised convolution) and stay invalid.
    """
    v = valid.astype("float64")
    x = np.where(valid, lin, 0).astype("float64")
    w = uniform_filter(v, size, mode="reflect")
    w_safe = np.where(w > 0, w, 1)
    m = uniform_filter(x, size, mode="reflect") / w_safe
    m2 = uniform_filter(x * x, size, mode="reflect") / w_safe
    var_z = np.maximum(m2 - m * m, 0)
    cu2 = 1.0 / enl
    var_x = np.maximum((var_z - m * m * cu2) / (1 + cu2), 0)
    k = np.where(var_z > 0, var_x / np.where(var_z > 0, var_z, 1), 0)
    out = m + k * (x - m)
    return np.where(valid, out, np.nan).astype("float32")


def estimate_enl(lin: np.ndarray, valid: np.ndarray, size: int = 9, q: float = 0.10) -> float:
    """ENL = mean^2/var over the most homogeneous windows (lowest coefficient of variation decile)."""
    v = valid.astype("float64")
    x = np.where(valid, lin, 0).astype("float64")
    w = uniform_filter(v, size)
    ok = w > 0.999
    if ok.sum() < 1000:
        return float("nan")
    m = uniform_filter(x, size)[ok]
    var = np.maximum(uniform_filter(x * x, size)[ok] - m * m, 1e-12)
    cv = np.sqrt(var) / np.maximum(m, 1e-12)
    sel = cv <= np.quantile(cv, q)
    return float(np.median((m[sel] ** 2) / var[sel]))


def to_db(lin: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(lin) & (lin > 0), 10 * np.log10(np.maximum(lin, 1e-10)), np.nan).astype("float32")


def db_to_i16(db: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(db), np.clip(np.round(db * DB_SCALE), -32767, 32767), NODATA_I16).astype("int16")


def sar_stack(vv_lin, vh_lin, enl, lee_size=7):
    valid = np.isfinite(vv_lin) & np.isfinite(vh_lin) & (vv_lin > 0) & (vh_lin > 0)
    vv_l = lee_filter(vv_lin, valid, lee_size, enl)
    vh_l = lee_filter(vh_lin, valid, lee_size, enl)
    stack = np.stack([to_db(np.where(valid, vv_lin, np.nan)), to_db(np.where(valid, vh_lin, np.nan)),
                      to_db(vv_l), to_db(vh_l)])
    return stack, valid


# ---------------------------------------------------------------- permanent water (JRC GSW)
class JRC:
    """Reads JRC GSW occurrence + seasonality tiles and resamples them (nearest) to any chip grid."""

    def __init__(self, occurrence_tiles, seasonality_tiles, occ_thr=80):
        self.occ = [rasterio.open(p) for p in occurrence_tiles]
        self.sea = [rasterio.open(p) for p in seasonality_tiles]
        self.occ_thr = occ_thr

    def _sample(self, srcs, T, crs, shape):
        out = np.full(shape, 255, "uint8")
        for s in srcs:
            dst = np.full(shape, 255, "uint8")
            reproject(rasterio.band(s, 1), dst, dst_transform=T, dst_crs=crs, resampling=Resampling.nearest,
                      src_nodata=255, dst_nodata=255)
            out = np.where(out == 255, dst, out)
        return out

    def permanent(self, T, crs, shape):
        """Returns uint8: 1 permanent water, 0 not, 255 not covered by the tiles."""
        occ = self._sample(self.occ, T, crs, shape)
        sea = self._sample(self.sea, T, crs, shape)
        covered = (occ != 255) | (sea != 255)
        perm = ((sea == 12) | ((occ != 255) & (occ >= self.occ_thr))).astype("uint8")
        return np.where(covered, perm, 255).astype("uint8"), occ


# ---------------------------------------------------------------- labels
def make_labels(water_or_flood: np.ndarray, label_valid: np.ndarray, sar_valid: np.ndarray, perm: np.ndarray,
                label_includes_permanent: bool):
    """Build the 4 label bands.

    water_or_flood: 0/1 array from the source label. TU Wien marks only *flood* (permanent water is
    excluded by the TU Wien reference-water mask), Sen1Floods11 LabelHand marks *all water*.
    """
    valid = label_valid & sar_valid
    perm_known = perm != 255
    perm1 = perm == 1
    src1 = water_or_flood == 1
    if label_includes_permanent:
        water = src1
        flood = src1 & ~perm1
    else:
        water = src1 | perm1
        flood = src1 & ~perm1
    l_water = np.where(valid, water, IGNORE).astype("uint8")
    l_flood = np.where(valid & perm_known, flood, IGNORE).astype("uint8")
    l_flood[valid & perm1] = IGNORE          # never score the model on permanent rivers/lakes for "flood"
    l_perm = np.where(perm_known, perm1, IGNORE).astype("uint8")
    return np.stack([l_water, l_flood, l_perm, valid.astype("uint8")])


# ---------------------------------------------------------------- writing
def write_chip(sar_path, lab_path, stack_db, labels, T, crs, tags=None):
    common = dict(driver="GTiff", height=stack_db.shape[1], width=stack_db.shape[2], crs=crs, transform=T,
                  compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(sar_path, "w", count=4, dtype="int16", nodata=NODATA_I16, predictor=2, **common) as o:
        o.write(db_to_i16(stack_db))
        for i, n in enumerate(SAR_BANDS, 1):
            o.set_band_description(i, n)
        o.update_tags(scale="0.01 (value/100 = dB)", **(tags or {}))
    with rasterio.open(lab_path, "w", count=4, dtype="uint8", nodata=None, **common) as o:
        o.write(labels)
        for i, n in enumerate(LABEL_BANDS, 1):
            o.set_band_description(i, n)
        o.update_tags(values="0 = no, 1 = yes, 255 = ignore", **(tags or {}))


def chip_grid(lon, lat):
    return from_origin(round(lon - HALF_DEG, 5), round(lat + HALF_DEG, 5), PIX_DEG, PIX_DEG)


def tuwien_label_on_grid(mask_path, bi, bj, T, crs, shape, margin=128):
    """Reproject one 512-block (+margin) of a TU Wien Equi7 mask onto the chip grid (nearest)."""
    with rasterio.open(mask_path) as m:
        r0, c0 = max(bi * CHIP - margin, 0), max(bj * CHIP - margin, 0)
        win = Window(c0, r0, min(CHIP + 2 * margin, m.width - c0), min(CHIP + 2 * margin, m.height - r0))
        a = m.read(1, window=win)
        wT = m.window_transform(win)
        dst = np.full(shape, 255, "uint8")
        reproject(a, dst, src_transform=wT, src_crs=m.crs, dst_transform=T, dst_crs=crs,
                  resampling=Resampling.nearest, src_nodata=None, dst_nodata=255)
    return dst


def chip_stats(labels):
    w, f, p, v = labels
    n = labels.shape[1] * labels.shape[2]
    nv = max(int((v == 1).sum()), 1)
    return dict(valid_frac=round(float((v == 1).sum()) / n, 4),
                water_frac=round(float((w == 1).sum()) / nv, 4),
                flood_frac=round(float((f == 1).sum()) / nv, 4),
                perm_frac=round(float((p == 1).sum()) / n, 4),
                flood_ignored_frac=round(float(((f == IGNORE) & (v == 1)).sum()) / nv, 4))
