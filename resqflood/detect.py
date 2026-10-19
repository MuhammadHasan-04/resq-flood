"""ResQFlood flood detector: U-Net (primary) + radar-threshold fallback, with permanent / seasonal water handling.

    det = FloodDetector.from_model_file("runs/M2/best_model.json")      # or a run folder (runs/M1_unet_r34_lee)
    res = det.detect(vv_linear, vh_linear, transform=T, crs="EPSG:4326")

Per-pixel classes (res["classes"])
  0 dry · 1 possible flood (weak evidence, NOT counted) · 2 FLOOD · 3 permanent water · 4 seasonal water · 255 no data

Flood map (fusion of the two methods; thresholds tuned on the validation split by scripts/tune_detector.py)
  flood = U-Net probability >= t_high                                  (model is confident)
       OR (U-Net probability >= t_low AND radar threshold says water)  (both methods agree)
  "confirmed by both" statistics are reported separately.
Normal water (never counted as flood)
  * permanent water: JRC seasonality == 12 months or occurrence >= 80 %, and/or water in a pre-flood reference image
  * seasonal water zone (JRC occurrence >= seasonal_occ, not permanent): shown as class 4 only if the tuning showed
    that this does not reduce flood accuracy; otherwise the pixels stay flood and are reported as
    "flood inside the normal seasonal-water zone"
Verdict ("FLOOD DETECTED" / "NO FLOOD DETECTED")
  counts flood pixels outside permanent water, outside a riverbank buffer and (optionally) outside the seasonal
  zone, after removing specks smaller than min_blob_px, and compares them with min_frac / min_km2.
Fallback: if the U-Net cannot be loaded or fails, the radar threshold alone is used and reported as such.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

from resqflood.sar import training_data as td

DEFAULT_VV_THR_DB = -15.5
DEFAULT_PARAMS = dict(t_high=0.5, t_low=0.5, seasonal_occ=0, seasonal_as_class=False, verdict_buffer_px=1,
                      verdict_exclude_seasonal=True, verdict_seasonal_occ=50, min_blob_px=10, min_frac=0.005,
                      min_km2=0.5, perm_occ=80)


def otsu(v):
    v = v[np.isfinite(v)]
    if v.size < 100:
        return DEFAULT_VV_THR_DB
    h, e = np.histogram(v, 256)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    wf = w[-1] - w
    return float(c[np.argmax(w * wf * (m / np.maximum(w, 1) - (m[-1] - m) / np.maximum(wf, 1)) ** 2)])


def normalise(stack, valid, stats, channels="lee"):
    """stack = [VV, VH, VV_lee, VH_lee] dB -> z-scored model input (C,H,W)."""
    clip = stats["clip_db"]
    spec = {"lee": [("VV_lee7_dB", 2, "VV"), ("VH_lee7_dB", 3, "VH"), ("VV_minus_VH_lee7_dB", None, None)],
            "raw": [("VV_dB", 0, "VV"), ("VH_dB", 1, "VH"), ("VV_minus_VH_dB", None, None)]}[channels]
    out = []
    for name, b, cl in spec:
        v = stack[spec[0][1]] - stack[spec[1][1]] if b is None else np.clip(stack[b], *clip[cl])
        z = (v - stats["channels"][name]["mean"]) / stats["channels"][name]["std"]
        out.append(np.clip(np.where(valid, np.nan_to_num(z), 0), -6, 6))
    return np.stack(out).astype("float32")


def sliding_prob(models, x, device, tile=512, overlap=128, tta=True):
    from resqflood.train.core import predict_prob
    C, H, W = x.shape
    step = tile - overlap

    def starts(n):
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        return s if s[-1] == n - tile else s + [n - tile]
    acc = np.zeros((H, W), "float32")
    wsum = np.zeros((H, W), "float32")
    w1 = np.hanning(tile + 2)[1:-1].astype("float32")
    win = np.maximum(np.outer(w1, w1), 1e-3)
    for r in starts(H):
        for c in starts(W):
            p = predict_prob(models, x[:, r:r + tile, c:c + tile], device, tta=tta)
            ww = win[:p.shape[0], :p.shape[1]]
            acc[r:r + p.shape[0], c:c + p.shape[1]] += p * ww
            wsum[r:r + p.shape[0], c:c + p.shape[1]] += ww
    return acc / np.maximum(wsum, 1e-6)


def jrc_masks(occ, sea, valid, params, verdict_zone=False):
    """occ / sea: JRC layers on the grid (255 = no data) or None.
    Returns permanent, seasonal (map class) and - with verdict_zone=True - the 'normally wet' zone used by the verdict."""
    if occ is None:
        z = np.zeros(valid.shape, bool)
        return (z, z.copy(), z.copy()) if verdict_zone else (z, z.copy())
    perm = ((sea == 12) | ((occ != 255) & (occ >= params["perm_occ"]))) & valid
    so = params.get("seasonal_occ", 0)
    seasonal = ((occ != 255) & (occ >= so) & (occ > 0) & ~perm & valid) if so else np.zeros_like(perm)
    if not verdict_zone:
        return perm, seasonal
    vo = params.get("verdict_seasonal_occ", 50)
    zone = ((occ != 255) & (occ >= vo) & (occ > 0) & ~perm & valid) if vo else np.zeros_like(perm)
    return perm, seasonal, zone


def decide(prob, fb, valid, perm, seasonal, params, px_km2, verdict_zone=None):
    """Shared decision logic (used by the app, predict_scene, test_cases and tune_detector).
    prob: U-Net probability (None -> fallback only). fb: fallback water mask. Returns classes + statistics."""
    p = params
    if prob is not None:
        flood = ((prob >= p["t_high"]) | ((prob >= p["t_low"]) & fb)) & valid
        both = flood & fb & (prob >= p["t_low"])
        weak = (((prob >= min(p["t_low"], 0.3)) | fb) & ~flood) & valid
    else:
        flood, both, weak = fb & valid, fb & valid, np.zeros_like(fb)
    classes = np.zeros(valid.shape, "uint8")
    classes[weak & ~perm] = 1
    classes[flood & ~perm] = 2
    if p.get("seasonal_as_class"):
        classes[flood & seasonal] = 4
    classes[perm & (flood | weak | fb)] = 3
    classes[~valid] = 255
    # ---- verdict: count only flood outside normal water, riverbank buffer and tiny specks
    cnt = flood & ~perm & valid
    if p.get("verdict_buffer_px", 0) and perm.any():
        cnt &= ~ndimage.binary_dilation(perm, iterations=int(p["verdict_buffer_px"]))
    vz = seasonal if verdict_zone is None else verdict_zone
    if p.get("verdict_exclude_seasonal", False) and vz.any():
        cnt &= ~vz
    if p.get("min_blob_px", 0) and cnt.any():
        lab, n = ndimage.label(cnt)
        sizes = ndimage.sum(cnt, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= p["min_blob_px"]
        cnt = keep[lab]
    nvalid = max(int(valid.sum()), 1)
    n_flood = int((classes == 2).sum())
    n_cnt = int(cnt.sum())
    frac = n_cnt / nvalid
    verdict = (frac >= p["min_frac"]) or (n_cnt * px_km2 >= p["min_km2"])
    return dict(classes=classes, flood=flood & ~perm, verdict_mask=cnt, verdict=bool(verdict),
                stats=dict(flood_px=n_flood, both_px=int((both & ~perm).sum()), verdict_px=n_cnt,
                           seasonal_flood_px=int((flood & seasonal & ~perm).sum()), nvalid=nvalid,
                           possible_px=int((classes == 1).sum()), perm_px=int((classes == 3).sum()),
                           seasonal_px=int((classes == 4).sum())))


class FloodDetector:
    def __init__(self, ckpts=(), norm_stats=None, unet_threshold=0.5, vv_threshold_db=None, jrc_dir=None,
                 tta=True, params=None, download_jrc=True):
        self.models, self.cfg, self.device, self.error = [], None, None, None
        self.stats = json.load(open(norm_stats)) if norm_stats and Path(norm_stats).exists() else None
        self.vv_threshold_db, self.jrc_dir, self.tta, self.download_jrc = vv_threshold_db, jrc_dir, tta, download_jrc
        self.params = dict(DEFAULT_PARAMS, t_high=unet_threshold, t_low=unet_threshold)
        if params:
            self.params.update(params)
        self.unet_threshold = self.params["t_high"]
        if ckpts and self.stats:
            try:
                import torch
                from resqflood.train.core import build_unet
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for p in ckpts:
                    ck = torch.load(p, map_location=self.device, weights_only=False)
                    self.cfg = ck["config"]
                    m, _ = build_unet(self.cfg["encoder"], self.cfg["in_channels"], "none", self.cfg.get("attention", "scse"))
                    m.load_state_dict(ck["model"])
                    self.models.append(m.to(self.device).eval())
            except Exception as e:      # model unusable -> fallback-only mode
                self.models, self.error = [], f"{type(e).__name__}: {e}"
        elif ckpts and not self.stats:
            self.error = "norm_stats.json not found"

    @classmethod
    def from_model_file(cls, path, jrc_dir=None, **kw):
        """A model folder (best.pt + eval/metrics.json [+ detector.json]) or a best_model.json (M2)."""
        p = Path(path)
        if p.is_dir():
            mj = p / "eval" / "metrics.json"
            m = json.load(open(mj)) if mj.exists() else {}
            spec = dict(checkpoints=[str(p / "best.pt")], unet_threshold=m.get("threshold", 0.5),
                        vv_threshold_db=m.get("vv_threshold_db"))
            dj = p / "detector.json"
        else:
            spec = json.load(open(p))
            base = p.parent
            spec["checkpoints"] = [c if os.path.isabs(c) else str((base / c).resolve()) for c in spec["checkpoints"]]
            dj = p.parent / "detector.json"
        params = None
        if dj.exists():
            d = json.load(open(dj))
            params = d["params"]
            spec["vv_threshold_db"] = d.get("vv_threshold_db", spec.get("vv_threshold_db"))
        ns = kw.pop("norm_stats", None) or spec.get("norm_stats")
        return cls(spec["checkpoints"], ns, spec.get("unet_threshold", 0.5), spec.get("vv_threshold_db"),
                   jrc_dir or spec.get("jrc_dir"), params=params, **kw)

    @property
    def mode(self):
        return "U-Net + fallback (fused)" if self.models else "fallback only (radar threshold)"

    # ------------------------------------------------------------------ pieces (also used by tune_detector)
    def unet_prob(self, stack, valid):
        x = normalise(stack, valid, self.stats, self.cfg["channels"])
        return sliding_prob(self.models, x, self.device, tta=self.tta)

    def jrc_layers(self, transform, crs, shape, notes):
        from rasterio.transform import array_bounds
        from rasterio.warp import transform_bounds
        from resqflood.jrc import ensure_tiles
        H, W = shape
        b = array_bounds(H, W, transform)
        bb = transform_bounds(crs, "EPSG:4326", *b) if str(crs).upper() != "EPSG:4326" else b
        tiles = ensure_tiles(bb, self.jrc_dir, download=self.download_jrc, log=lambda m: notes.append(m))
        if tiles["missing"]:
            notes.append("JRC tiles unavailable: " + ", ".join(tiles["missing"]))
        if tiles["occurrence"] and tiles["seasonality"]:
            return td.JRC(tiles["occurrence"], tiles["seasonality"]).layers(transform, crs, shape)
        return None, None

    # ------------------------------------------------------------------ main entry
    def detect(self, vv_lin, vh_lin=None, transform=None, crs=None, pixel_m=20.0, ref_vv_lin=None, ref_vh_lin=None,
               _water_only=False):
        """vv_lin / vh_lin: linear backscatter on one grid (NaN = no data). vh may be None (single polarisation).
        ref_vv_lin / ref_vh_lin: optional PRE-FLOOD image on the same grid; its water is treated as permanent."""
        notes = []
        if vh_lin is None:
            ratio = self.stats["channels"]["VV_minus_VH_lee7_dB"]["mean"] if self.stats else 6.9
            vh_lin = vv_lin / (10 ** (ratio / 10))
            notes.append("Only one polarisation (VV) available: VH approximated, U-Net reliability is reduced.")
        stack, valid = td.sar_stack(vv_lin.astype("float32"), vh_lin.astype("float32"), enl=4.4)
        if valid.sum() < 100:
            raise ValueError("the image contains no valid radar pixels")
        vv_lee = stack[2]
        thr_db = self.vv_threshold_db if self.vv_threshold_db is not None else otsu(vv_lee[valid])
        fb = (vv_lee < thr_db) & valid
        prob = None
        if self.models:
            try:
                prob = self.unet_prob(stack, valid)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                notes.append(f"U-Net failed ({self.error}); fallback used alone.")
        if _water_only:
            p = self.params
            return (((prob >= p["t_high"]) | ((prob >= p["t_low"]) & fb)) & valid) if prob is not None else fb
        perm_sources = []
        occ = sea = None
        if transform is not None and crs is not None and self.jrc_dir:
            occ, sea = self.jrc_layers(transform, crs, valid.shape, notes)
            if occ is not None:
                perm_sources.append("JRC Global Surface Water")
        perm, seasonal, vzone = jrc_masks(occ, sea, valid, self.params, verdict_zone=True)
        if ref_vv_lin is not None:
            if ref_vv_lin.shape != valid.shape:
                notes.append("Reference image has a different size from the flood image; it was ignored.")
            else:
                pre = self.detect(ref_vv_lin, ref_vh_lin, None, None, pixel_m, _water_only=True)
                perm = perm | (pre & valid)
                perm_sources.append("pre-flood reference image")
        if not perm_sources:
            notes.append("Permanent water NOT removed: give the image a location (GeoTIFF, world file or corner "
                         "coordinates) or a pre-flood reference image. All water is treated as possible flood.")
        if transform is not None and crs is not None and str(crs).upper().endswith("4326"):
            lat = transform.f + transform.e * valid.shape[0] / 2
            px_km2 = abs(transform.a * transform.e) * (111.32 ** 2) * np.cos(np.deg2rad(lat))
        else:
            px_km2 = (pixel_m / 1000) ** 2
        d = decide(prob, fb, valid, perm, seasonal, self.params, px_km2, verdict_zone=vzone)
        st = d["stats"]
        nv = st["nvalid"]
        if prob is None:
            notes.append("Fallback-only result: confirm with a second source before acting on it.")
        flood_px = int(d["flood"].sum())
        return dict(
            verdict="FLOOD DETECTED" if d["verdict"] else "NO FLOOD DETECTED",
            mode=self.mode if prob is not None else "fallback only (radar threshold)",
            flood_pct=round(100 * flood_px / nv, 2), flood_km2=round(flood_px * px_km2, 2),
            confirmed_by_both_pct=round(100 * st["both_px"] / nv, 2),
            flood_in_seasonal_zone_km2=round(st["seasonal_flood_px"] * px_km2, 2),
            verdict_flood_km2=round(st["verdict_px"] * px_km2, 2),
            possible_flood_pct=round(100 * st["possible_px"] / nv, 2),
            permanent_water_km2=round(st["perm_px"] * px_km2, 2), seasonal_water_km2=round(st["seasonal_px"] * px_km2, 2),
            permanent_water_source=" + ".join(perm_sources) if perm_sources else "not removed",
            valid_pct=round(100 * valid.mean(), 1), vv_threshold_db=round(float(thr_db), 2),
            unet_thresholds=dict(t_high=self.params["t_high"], t_low=self.params["t_low"]) if prob is not None else None,
            notes=notes, classes=d["classes"], prob=prob, fallback=fb, vv_lee_db=vv_lee, valid=valid,
            # backwards-compatible names
            confirmed_flood_pct=round(100 * flood_px / nv, 2), confirmed_flood_km2=round(flood_px * px_km2, 2))
