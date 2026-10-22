"""Tune and verify the full flood detector (U-Net + fallback + permanent/seasonal water + verdict) for one model.

  python scripts/tune_detector.py --model runs/M1_unet_r34_lee            (model folder)
  python scripts/tune_detector.py --model runs/M2/best_model.json         (auto_tune result)

1. Runs the model (8-view TTA) once on every validation / test / external chip, exactly as the app does.
2. VALIDATION ONLY: chooses
   - the fusion thresholds (t_high, t_low) and the radar threshold -> best flood IoU;
   - whether the JRC seasonal-water zone may be shown as its own class -> only if flood IoU does not drop;
   - the verdict rules (riverbank buffer, seasonal exclusion, speck size, minimum flooded share/area)
     -> best flood / no-flood accuracy on chips.
3. Reports everything on the untouched test sets, including "river / permanent water but no flood" chips.
Writes <model>/detector.json (used automatically by the app) and <model>/detector_eval/DETECTOR_REPORT.md.
The model weights are never modified.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import ndimage  # noqa: E402
from resqflood.detect import DEFAULT_PARAMS, FloodDetector, decide, jrc_masks, synth_vh  # noqa: E402
from resqflood.sar import training_data as td  # noqa: E402
from resqflood.train.data import read_manifest  # noqa: E402
from test_cases import DADU, load_chip  # noqa: E402

SPLITS = ["val", "test", "test_external_s1f_hand", "val_dry", "test_dry"]


def collect(det, data, rows, cache_f, log_every=10, vv_only=False):
    """vv_only=True: the VH band is dropped and approximated exactly as the app does for VV-only GeoTIFFs and PNG/JPG."""
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        return z["chips"].tolist()
    chips, t0 = [], time.time()
    for i, r in enumerate(rows):
        vv, vh, T, crs, lab = load_chip(data, r)
        if vv_only:
            vh = synth_vh(vv, det.stats)
        stack, valid = td.sar_stack(vv, vh, enl=4.4)
        prob = det.unet_prob(stack, valid, single_pol=vv_only) if det.models else None
        notes = []
        occ, sea = det.jrc_layers(T, crs, valid.shape, notes) if det.jrc_dir else (None, None)
        lat = T.f + T.e * valid.shape[0] / 2
        px_km2 = abs(T.a * T.e) * 111.32 ** 2 * np.cos(np.deg2rad(lat))
        chips.append(dict(id=r["chip_id"], split=r["split"], lon=float(r["lon"]), lat=float(r["lat"]),
                          prob=None if prob is None else prob.astype("float16"), vv=stack[2].astype("float16"),
                          valid=valid, occ=occ, sea=sea, water=lab[0], flood=lab[1], perm_lab=lab[2], px_km2=px_km2))
        if (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(rows)} chips ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(cache_f, chips=np.array(chips, dtype=object))
    return chips


def collect_pre(det, pre_dir, cache_f, log_every=10, vv_only=False):
    """Pre-monsoon (Mar-May 2022) Sentinel-1 chips of river/lake locations: real scenes with permanent water and
    NO flood. Truth: flood = 0 everywhere (the 2022 floods started in June); water truth unknown (255)."""
    import rasterio
    if cache_f.exists():
        return np.load(cache_f, allow_pickle=True)["chips"].tolist()
    meta = {m[0]: m for m in json.load(open(Path(pre_dir) / "pre_flood_meta.json"))}
    chips, t0 = [], time.time()
    files = sorted(Path(pre_dir).glob("PRE_*.tif"))
    for i, f in enumerate(files):
        cid = f.stem.split("_")[1]
        m = meta.get(cid)
        with rasterio.open(f) as src:
            a = src.read().astype("float32")
            T, crs = src.transform, src.crs
        a[a <= -30000] = np.nan
        stack, valid = td.sar_stack(a[0], synth_vh(a[0], det.stats) if vv_only else a[1], enl=4.4)
        prob = det.unet_prob(stack, valid, single_pol=vv_only)
        occ, sea = det.jrc_layers(T, crs, valid.shape, [])
        lat = T.f + T.e * valid.shape[0] / 2
        px_km2 = abs(T.a * T.e) * 111.32 ** 2 * np.cos(np.deg2rad(lat))
        perm_lab = np.zeros(valid.shape, "uint8") if occ is None else (((sea == 12) | ((occ != 255) & (occ >= 80))) & valid).astype("uint8")
        chips.append(dict(id=f.stem, split="pre_flood_" + (m[1] if m else "unknown"), lon=float(m[2]) if m else 0,
                          lat=float(m[3]) if m else 0, prob=prob.astype("float16"), vv=stack[2].astype("float16"),
                          valid=valid, occ=occ, sea=sea, water=np.full(valid.shape, 255, "uint8"),
                          flood=np.where(valid, 0, 255).astype("uint8"), perm_lab=perm_lab, px_km2=px_km2))
        if (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(files)} pre-flood chips ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(cache_f, chips=np.array(chips, dtype=object))
    return chips


def masks_for(c, params, zone=False):
    if c["occ"] is not None:
        return jrc_masks(c["occ"], c["sea"], c["valid"], params, verdict_zone=zone)
    perm = (c["perm_lab"] == 1) & c["valid"]       # outside the local JRC tiles: the chip's own permanent-water band
    z = np.zeros_like(perm)
    return (perm, z, z.copy()) if zone else (perm, z)


def truth_flood(c):
    v = c["flood"] != 255
    return (c["flood"] == 1).sum() >= 0.01 * max(v.sum(), 1)


def river_only(c, perm):
    return (not truth_flood(c)) and (c["split"].startswith("pre_flood") or perm.sum() >= 0.005 * c["valid"].sum())


def pixel_scores(chips, params, vv_thr, seasonal_class=None):
    """Flood IoU (vs flood label, permanent ignored) and water IoU (vs water label) over all chips."""
    tp = fp = fn = wtp = wfp = wfn = 0
    for c in chips:
        prob = None if c["prob"] is None else c["prob"].astype("float32")
        fb = (c["vv"].astype("float32") < vv_thr) & c["valid"]
        if prob is None:
            water = fb
        else:
            water = ((prob >= params["t_high"]) | ((prob >= params["t_low"]) & fb)) & c["valid"]
        perm, seas = masks_for(c, params)
        flood = water & ~perm
        if seasonal_class:
            flood &= ~seas
        y = c["flood"]
        ok = y != 255
        t = y == 1
        tp += int((flood & t & ok).sum())
        fp += int((flood & ~t & ok).sum())
        fn += int((~flood & t & ok).sum())
        yw = c["water"]
        okw = yw != 255
        tw = yw == 1
        wtp += int((water & tw & okw).sum())
        wfp += int((water & ~tw & okw).sum())
        wfn += int((~water & tw & okw).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    if tp + fn == 0:     # scene set without any flood (river-only): report false-alarm share instead
        npx = sum(int((c["flood"] != 255).sum()) for c in chips)
        return dict(flood_iou=float("nan"), flood_f1=float("nan"), flood_precision=float("nan"), flood_recall=float("nan"),
                    water_iou=float("nan"), false_flood_pct=round(100 * fp / max(npx, 1), 3))
    return dict(flood_iou=round(iou, 4), flood_f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                flood_precision=round(prec, 4), flood_recall=round(rec, 4),
                water_iou=round(wtp / max(wtp + wfp + wfn, 1), 4))


def verdicts(chips, params, vv_thr):
    out = []
    for c in chips:
        prob = None if c["prob"] is None else c["prob"].astype("float32")
        fb = (c["vv"].astype("float32") < vv_thr) & c["valid"]
        perm, seas, zone = masks_for(c, params, zone=True)
        d = decide(prob, fb, c["valid"], perm, seas, params, c["px_km2"], verdict_zone=zone)
        out.append(dict(id=c["id"], truth=bool(truth_flood(c)), pred=d["verdict"], river_only=bool(river_only(c, perm))))
    return out


def chip_scores(v):
    t = np.array([x["truth"] for x in v])
    p = np.array([x["pred"] for x in v])
    ro = np.array([x["river_only"] for x in v])
    tpr = (p & t).sum() / max(t.sum(), 1)
    tnr = (~p & ~t).sum() / max((~t).sum(), 1)
    return dict(chip_accuracy=round(float((p == t).mean()), 4), balanced_accuracy=round(float((tpr + tnr) / 2), 4),
                flood_chips_found=f"{int((p & t).sum())}/{int(t.sum())}",
                no_flood_chips_correct=f"{int((~p & ~t).sum())}/{int((~t).sum())}",
                river_only_chips_correct=f"{int((~p & ro).sum())}/{int(ro.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="../outputs/FLOOD_TRAINING_DATA")
    ap.add_argument("--jrc-dir", default="../outputs/datasets/jrc_global_surface_water_pakistan")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--pre-flood", default="../outputs/FLOOD_TRAINING_DATA/00_raw_downloads/sentinel1_rtc_pre_flood_2022",
                    help="folder with pre-monsoon river/lake chips (no-flood scenario test)")
    ap.add_argument("--skip-vv-only", action="store_true", help="do not tune the single-polarisation (VV/PNG) settings")
    ap.add_argument("--vv-model", nargs="*", default=None, help="best.pt of the single-polarisation U-Net "
                    "(train_unet.py --single-pol); used for VV-only / PNG / JPG input")
    a = ap.parse_args()
    mp = Path(a.model)
    mdir = mp if mp.is_dir() else mp.parent
    out = mdir / "detector_eval"
    out.mkdir(parents=True, exist_ok=True)
    norm = str(Path(a.data) / "02_splits" / "norm_stats.json")
    det = FloodDetector.from_model_file(a.model, jrc_dir=a.jrc_dir, norm_stats=norm, tta=not a.no_tta)
    if not det.models:
        raise SystemExit(f"model could not be loaded: {det.error}")
    if a.vv_model:
        det.models_vv = []
        det.load_vv_models([str(Path(p).resolve()) for p in a.vv_model])
        if not det.models_vv:
            raise SystemExit(det.error)
        print("single-polarisation model:", a.vv_model)
    base_thr = det.unet_threshold
    base_vv = det.vv_threshold_db
    rows = read_manifest(a.data)
    chips = {}
    for s in SPLITS:
        rs = [r for r in rows if r["split"] == s]
        print(f"[{s}] running the detector on {len(rs)} chips ...", flush=True)
        chips[s] = collect(det, a.data, rs, out / f"cache_{s}{'_notta' if a.no_tta else ''}.npz")
    val = chips["val"]
    pre = []
    if a.pre_flood and (Path(a.pre_flood) / "pre_flood_meta.json").exists():
        print("[pre-flood river/lake scenes] running the detector ...", flush=True)
        pre = collect_pre(det, a.pre_flood, out / f"cache_pre_flood{'_notta' if a.no_tta else ''}.npz")
    pre_val = [c for c in pre if c["split"] == "pre_flood_val"] + chips.get("val_dry", [])
    pre_test = [c for c in pre if c["split"] != "pre_flood_val"]
    if pre_test:
        chips["river_only_pre_flood (test+other areas)"] = pre_test

    params, vv_thr, s_best, cs_val = tune(val, chips.get("val_dry", []), pre_val, base_thr, base_vv)

    # ---- single polarisation (VV only GeoTIFF, every PNG/JPG): same tuning on the same validation scenes, VH removed
    vv_only = {}
    if not a.skip_vv_only:
        print("\n=== single-polarisation (VV only / PNG / JPG) input ===", flush=True)
        sfx = ("_notta" if a.no_tta else "") + ("_vvmodel_" + __import__("hashlib").md5("|".join(
            f"{c}{Path(c).stat().st_mtime}" for c in det.vv_ckpts).encode()).hexdigest()[:8]
                                                 if det.models_vv else "")
        for s in ["val", "val_dry", "test", "test_dry"]:
            rs = [r for r in rows if r["split"] == s]
            print(f"[VV only · {s}] running the detector on {len(rs)} chips ...", flush=True)
            vv_only[s] = collect(det, a.data, rs, out / f"cache_vvonly_{s}{sfx}.npz", vv_only=True)
        pre1 = collect_pre(det, a.pre_flood, out / f"cache_vvonly_pre_flood{sfx}.npz", vv_only=True) if pre else []
        pv1 = [c for c in pre1 if c["split"] == "pre_flood_val"] + vv_only["val_dry"]
        pt1 = [c for c in pre1 if c["split"] != "pre_flood_val"]
        params1, vv_thr1, s1, cs1 = tune(vv_only["val"], vv_only["val_dry"], pv1, base_thr, base_vv)
        if pt1:
            vv_only["river_only_pre_flood (test+other areas)"] = pt1
        chips.update({f"VV only · {k}": v for k, v in vv_only.items()})

    # ---- 4. report on all splits
    report = dict(model=str(a.model), tta=not a.no_tta, vv_threshold_db=vv_thr, params=params, splits={})
    for s, cs in chips.items():
        p_, t_ = (params1, vv_thr1) if s.startswith("VV only") else (params, vv_thr)
        groups = {s: cs}
        if s.endswith("test"):
            groups[s.replace("test", "test_dadu_district")] = [c for c in cs if DADU[0] <= c["lon"] <= DADU[2] and DADU[1] <= c["lat"] <= DADU[3]]
        for g, cc in groups.items():
            report["splits"][g] = dict(
                n_chips=len(cc),
                tuned_detector=dict(**pixel_scores(cc, p_, t_, p_["seasonal_as_class"]),
                                    **chip_scores(verdicts(cc, p_, t_))),
                unet_alone=pixel_scores(cc, dict(DEFAULT_PARAMS, t_high=base_thr, t_low=base_thr), base_vv),
                strict_and=pixel_scores(cc, dict(DEFAULT_PARAMS, t_high=1.01, t_low=base_thr), base_vv),
                fallback_only=pixel_scores([dict(c, prob=None) for c in cc], p_, t_))
    res = dict(params=params, vv_threshold_db=vv_thr, chosen_on="validation split only",
               val_flood_iou=s_best["flood_iou"], val_chip=cs_val)
    if vv_only:
        if det.models_vv:
            import os
            res["checkpoints_vv_only"] = [os.path.relpath(c, mdir) for c in det.vv_ckpts]
        res.update(params_vv_only=params1, vv_threshold_db_vv_only=vv_thr1, val_flood_iou_vv_only=s1["flood_iou"],
                   val_chip_vv_only=cs1)
        report.update(params_vv_only=params1, vv_threshold_db_vv_only=vv_thr1)
    json.dump(res, open(mdir / "detector.json", "w"), indent=2)
    json.dump(report, open(out / "detector_metrics.json", "w"), indent=2)
    _md(report, out / "DETECTOR_REPORT.md")
    print((out / "DETECTOR_REPORT.md").read_text())
    print("detector settings saved to", mdir / "detector.json")


def tune(val, val_dry, pre_val, base_thr, base_vv):
    """Chooses fusion thresholds, seasonal handling and verdict rules on validation scenes only."""
    chips = {"val_dry": val_dry}
    # ---- 1. fusion thresholds + radar threshold (val flood IoU)
    best = None
    for th, tl, dv in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01], [0.15, 0.2, 0.3, 0.4, 0.5], [-1.0, 0.0, 1.0]):
        if tl > th and th <= 1:
            continue
        p = dict(DEFAULT_PARAMS, t_high=th, t_low=min(tl, th))
        s = pixel_scores(val + chips.get("val_dry", []), p, base_vv + dv)   # dry-season chips punish false alarms
        if best is None or s["flood_iou"] > best[0]["flood_iou"]:
            best = (s, th, min(tl, th), base_vv + dv)
    s_best, th, tl, vv_thr = best
    params = dict(DEFAULT_PARAMS, t_high=th, t_low=tl)
    vv_all = val + chips.get("val_dry", [])
    ref_unet = pixel_scores(vv_all, dict(DEFAULT_PARAMS, t_high=base_thr, t_low=base_thr), base_vv)
    ref_and = pixel_scores(vv_all, dict(DEFAULT_PARAMS, t_high=1.01, t_low=base_thr), base_vv)
    print(f"fusion: t_high {th}, t_low {tl}, radar threshold {vv_thr} dB -> val flood IoU {s_best['flood_iou']} "
          f"(U-Net alone {ref_unet['flood_iou']}, strict AND {ref_and['flood_iou']})")

    # ---- 2. seasonal zone: may it be its own class without hurting flood IoU?
    seasonal_choice = dict(seasonal_occ=0, seasonal_as_class=False)
    for so in (1, 5, 10, 25, 50):            # lowest occurrence threshold that does not cost flood accuracy
        p = dict(params, seasonal_occ=so)
        s = pixel_scores(vv_all, p, vv_thr, seasonal_class=True)
        print(f"  seasonal zone occ>={so}% as separate class -> val flood IoU {s['flood_iou']}")
        if s["flood_iou"] >= s_best["flood_iou"] - 0.001 and not seasonal_choice["seasonal_as_class"]:
            seasonal_choice = dict(seasonal_occ=so, seasonal_as_class=True)
    if not seasonal_choice["seasonal_as_class"]:
        seasonal_choice = dict(seasonal_occ=50, seasonal_as_class=False)   # zone used for the verdict only
    params.update(seasonal_choice)
    print("seasonal handling:", seasonal_choice)

    # ---- 3. verdict rules (val chips + val-area river-only scenes; ties -> fewer exclusions)
    vchips = val + pre_val
    base = {}
    for c in vchips:                       # flood map once per chip
        prob = c["prob"].astype("float32")
        fb = (c["vv"].astype("float32") < vv_thr) & c["valid"]
        perm, seas = masks_for(c, params)
        flood = ((prob >= params["t_high"]) | ((prob >= params["t_low"]) & fb)) & c["valid"] & ~perm
        zones = {vo: (masks_for(c, dict(params, verdict_seasonal_occ=vo), zone=True)[2] if vo else None)
                 for vo in (1, 5, 10, 25, 50)}
        base[c["id"]] = (flood, perm, zones)
    counts = {}                            # verdict pixel counts for every (buffer, exclude-seasonal, blob)
    for buf, ex, blob in itertools.product([0, 1, 2, 3], [0, 1, 5, 10, 25, 50], [0, 10, 25, 50, 100]):
        arr = []
        for c in vchips:
            flood, perm, zones = base[c["id"]]
            cnt = flood.copy()
            if buf and perm.any():
                cnt &= ~ndimage.binary_dilation(perm, iterations=buf)
            if ex and zones[ex] is not None and zones[ex].any():
                cnt &= ~zones[ex]
            if blob and cnt.any():
                lab, n = ndimage.label(cnt)
                sizes = ndimage.sum(cnt, lab, range(1, n + 1))
                keep = np.zeros(n + 1, bool)
                keep[1:] = sizes >= blob
                cnt = keep[lab]
            arr.append((int(cnt.sum()), max(int(c["valid"].sum()), 1), c["px_km2"]))
        counts[(buf, ex, blob)] = arr
    truth = np.array([truth_flood(c) for c in vchips])
    ro = np.array([river_only(c, base[c["id"]][1]) for c in vchips])
    print(f"verdict tuning on {len(vchips)} validation scenes ({int(truth.sum())} with flood, {int((~truth).sum())} without)")
    cand = []
    for (buf, ex, blob), arr in counts.items():
        n, nv, km = (np.array(x, float) for x in zip(*arr))
        for mf, mk in itertools.product([0.002, 0.005, 0.01, 0.02, 0.04], [0.25, 0.5, 1.0, 2.0, 4.0]):
            pred = (n / nv >= mf) | (n * km >= mk)
            tpr = (pred & truth).sum() / max(truth.sum(), 1)
            tnr = (~pred & ~truth).sum() / max((~truth).sum(), 1)
            simplicity = -(buf + (2 if ex else 0) + blob / 25)
            cand.append(((round((tpr + tnr) / 2, 4), round(float((pred == truth).mean()), 4), simplicity),
                         dict(params, verdict_buffer_px=buf, verdict_exclude_seasonal=bool(ex), verdict_seasonal_occ=ex or 50,
                              min_blob_px=blob, min_frac=mf, min_km2=mk),
                         dict(balanced_accuracy=round((tpr + tnr) / 2, 4), chip_accuracy=round(float((pred == truth).mean()), 4),
                              river_only_correct=f"{int((~pred & ro).sum())}/{int(ro.sum())}")))
    cand.sort(key=lambda x: x[0], reverse=True)
    _, params, cs_val = cand[0]
    print("verdict rules:", {k: params[k] for k in ("verdict_buffer_px", "verdict_exclude_seasonal", "verdict_seasonal_occ", "min_blob_px",
                                                     "min_frac", "min_km2")}, "->", cs_val)

    return params, vv_thr, s_best, cs_val

def _md(r, path):
    L = [f"# Flood detector evaluation ({r['model']})", "",
         f"Settings chosen on the validation split only: {json.dumps(r['params'])}, radar threshold {r['vv_threshold_db']} dB.", ""]
    if r.get("params_vv_only"):
        L += [f"Single-polarisation input (VV only, PNG/JPG; rows 'VV only · ...'): {json.dumps(r['params_vv_only'])}, "
              f"radar threshold {r['vv_threshold_db_vv_only']} dB.", ""]
    L += [
         "| Split | Chips | Flood IoU | Flood F1 | Precision | Recall | Water IoU | Flood/no-flood verdict accuracy | Flood chips found | No-flood chips correct | River-only chips correct | False-flood % of pixels (no-flood scenes) | U-Net alone flood IoU | Strict AND flood IoU | Fallback only flood IoU |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in r["splits"].items():
        t = s["tuned_detector"]
        L.append(f"| {k} | {s['n_chips']} | **{t['flood_iou']:.3f}** | {t['flood_f1']:.3f} | {t['flood_precision']:.3f} | "
                 f"{t['flood_recall']:.3f} | {t['water_iou']:.3f} | **{t['chip_accuracy']:.3f}** | {t['flood_chips_found']} | "
                 f"{t['no_flood_chips_correct']} | {t['river_only_chips_correct']} | {t.get('false_flood_pct', '')} | "
                 f"{s['unet_alone']['flood_iou']:.3f} | "
                 f"{s['strict_and']['flood_iou']:.3f} | {s['fallback_only']['flood_iou']:.3f} |")
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
