"""Full verification of a flood-detection model: every scenario, every input format, every edge case.

  python scripts/full_verification.py --model runs/M2/best_model.json
  python scripts/full_verification.py --model runs/M1_unet_r34_lee

Everything runs through the same FloodDetector the app uses, on data never used for training or tuning.
Each check has a pass target; results go to <model>/verification/VERIFICATION_REPORT.md (+ CSVs and figures).

Sections
  1  Scenario accuracy (held-out chips with labels)
     S1 flood season, held-out regions         S2 Dadu district            S3 flood next to a river/lake
     S4 small floods (1-10 %)                   S5 dry season, no flood     S6 rivers/lakes only, April 2022
     S7 external: Sen1Floods11 2017 hand labels (reported, no target: different region/year/labelling)
  2  Change detection: flood image + pre-flood image of the same place; and pre-flood vs itself (must be no flood)
  3  Input-format robustness: the same scenes as GeoTIFF linear / dB / dB x100 / separate VV+VH / UTM / 10 m / 30 m /
     VV only / half no-data / Sentinel-1 GRD .zip (raw DN + calibration table) / small crop / PNG + corners /
     PNG + centre point / PNG + world file / 16-bit PNG / JPEG / PNG without location -> verdict must stay correct and the flooded area consistent
  4  Invalid inputs: colour photo, optical multispectral raster, blank image, all-no-data raster, truncated file,
     text file named .tif, empty file -> must be refused with a message, never crash
  5  Fallback mode (no model) on flood and no-flood scenes
  6  Stability: identical result when run twice; TTA vs no-TTA agreement
  7  Demo pack (DEMO_IMAGES/expected.csv)
  8  Scenarios S1, S2, S4, S5, S6 again with VV-only input (what every PNG/JPG and single-polarisation GeoTIFF gives)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))
from resqflood import render  # noqa: E402
from resqflood.detect import FloodDetector  # noqa: E402
from resqflood.sar_io import InputError, align_to, load_any  # noqa: E402
from resqflood.train.core import Confusion  # noqa: E402
from resqflood.train.data import read_manifest  # noqa: E402
from test_cases import DADU, load_chip, load_pre  # noqa: E402

TARGETS = {
    "S1 flood season, held-out regions": dict(flood_iou=0.85, verdict=0.95),
    "S2 Dadu district": dict(flood_iou=0.88, verdict=0.95),
    "S3 flood next to a river/lake": dict(flood_iou=0.85, verdict=0.95),
    "S4 small floods (1-10 %)": dict(verdict=0.85),
    "S5 dry season, no flood": dict(verdict=0.90),
    "S6 rivers/lakes only, April 2022": dict(verdict=0.90),
    "S7 external Sen1Floods11 2017 (no target)": dict(),
}


def truth_of(lab):
    v = lab[1] != 255
    return bool((lab[1] == 1).sum() >= 0.01 * max(v.sum(), 1))


class V:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        norm = str(Path(a.data) / "02_splits" / "norm_stats.json")
        self.det = FloodDetector.from_model_file(a.model, jrc_dir=a.jrc_dir, norm_stats=norm, tta=not a.no_tta)
        if not self.det.models:
            raise SystemExit(f"model not loaded: {self.det.error}")
        self.fb = FloodDetector(jrc_dir=a.jrc_dir, norm_stats=norm, vv_threshold_db=self.det.vv_threshold_db,
                                params=self.det.params)
        self.rows = read_manifest(a.data)
        self.results, self.checks, self.fails = [], [], []

    def check(self, section, name, ok, detail, target=""):
        self.checks.append(dict(section=section, check=name, result="PASS" if ok else "FAIL", detail=detail, target=target))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}" + (f"  (target {target})" if target else ""), flush=True)

    # ------------------------------------------------------------------ 1 scenarios
    def scenario_sets(self):
        test = [r for r in self.rows if r["split"] == "test"]
        dadu = [r for r in test if DADU[0] <= float(r["lon"]) <= DADU[2] and DADU[1] <= float(r["lat"]) <= DADU[3]]
        river = [r for r in test if float(r["perm_frac"]) >= 0.005 and float(r["flood_frac"]) >= 0.01]
        small = [r for r in test if 0.01 <= float(r["flood_frac"]) < 0.10]
        dry = [r for r in self.rows if r["split"] == "test_dry"]
        ext = [r for r in self.rows if r["split"] == "test_external_s1f_hand"]
        pre_dir = Path(self.a.pre_flood)
        meta = json.load(open(pre_dir / "pre_flood_meta.json")) if (pre_dir / "pre_flood_meta.json").exists() else []
        pre = [dict(chip_id=f.stem, pre_file=str(f), split="pre") for m in meta if m[1] != "val"
               for f in sorted(pre_dir.glob(f"PRE_{m[0]}_*.tif"))]
        sets = {"S1 flood season, held-out regions": test, "S2 Dadu district": dadu,
                "S3 flood next to a river/lake": river, "S4 small floods (1-10 %)": small,
                "S5 dry season, no flood": dry, "S6 rivers/lakes only, April 2022": pre,
                "S7 external Sen1Floods11 2017 (no target)": ext}
        if self.a.quick:
            sets = {k: v[:12] for k, v in sets.items()}
        return sets

    def load(self, r):
        if "pre_file" in r:
            return load_pre(r["pre_file"])
        return load_chip(self.a.data, r)

    def run_single_pol(self):
        """Same scenarios when only VV is available (single-polarisation GeoTIFF, and every PNG/JPG input)."""
        self.run_scenarios(single=True)

    def run_scenarios(self, single=False):
        print("\n[8] Scenario accuracy with VV-only input (single polarisation / PNG / JPG)" if single
              else "\n[1] Scenario accuracy", flush=True)
        sec = "8 VV-only scenarios" if single else "1 scenarios"
        cache = {}
        for name, rows in self.scenario_sets().items():
            if single and name.startswith(("S3", "S7")):
                continue
            cf, n_ok, n = Confusion(), 0, 0
            for r in rows:
                key = r["chip_id"]
                if key not in cache:
                    vv, vh, T, crs, lab = self.load(r)
                    res = self.det.detect(vv, None if single else vh, T, crs)
                    cache[key] = (res["verdict"], res["classes"], lab, res["flood_km2"], res)
                verdict, classes, lab, km2, res = cache[key]
                truth = truth_of(lab)
                pred = verdict == "FLOOD DETECTED"
                ok = pred == truth
                n_ok += ok
                n += 1
                cf.add((classes == 2) | (classes == 4), lab[1])
                self.results.append(dict(section=sec, set=name, case=key, truth="flood" if truth else "no flood",
                                         got=verdict, correct=ok, flood_km2=km2))
                if not ok:
                    self.fails.append((f"{'VV only · ' if single else ''}{name}: {key}", res, lab))
            m = cf.result()
            acc = n_ok / max(n, 1)
            t = dict(TARGETS[name])
            if single and "flood_iou" in t:      # one band carries less information: IoU target 0.05 lower
                t["flood_iou"] = round(t["flood_iou"] - 0.05, 2)
            has_flood = (m["tp"] + m["fn"]) > 0
            detail = f"{n} scenes · verdict correct {n_ok}/{n} ({acc * 100:.0f}%)" + \
                     (f" · flood IoU {m['iou']:.3f} · precision {m['precision']:.3f} · recall {m['recall']:.3f}" if has_flood
                      else f" · false-flood pixels {100 * m['fp'] / max(m['fp'] + m['tn'], 1):.2f}%")
            ok = True
            tgt = []
            if "verdict" in t:
                ok &= acc >= t["verdict"]
                tgt.append(f"verdict ≥ {t['verdict'] * 100:.0f}%")
            if "flood_iou" in t and has_flood:
                ok &= m["iou"] >= t["flood_iou"]
                tgt.append(f"IoU ≥ {t['flood_iou']}")
            self.check(sec, ("VV only · " if single else "") + name, ok if t else True, detail, ", ".join(tgt) or "report only")

    # ------------------------------------------------------------------ 2 change detection
    def run_change(self):
        print("\n[2] Change detection with a pre-flood image", flush=True)
        pre_dir = Path(self.a.pre_flood)
        if not (pre_dir / "pre_flood_meta.json").exists():
            self.check("2 change detection", "pre-flood pairs", False, "pre-flood folder missing")
            return
        meta = json.load(open(pre_dir / "pre_flood_meta.json"))
        by_id = {r["chip_id"]: r for r in self.rows}
        n1 = ok1 = n2 = ok2 = 0
        for m in meta:
            if m[1] == "val" or m[0] not in by_id:
                continue
            r = by_id[m[0]]
            pf = sorted(pre_dir.glob(f"PRE_{m[0]}_*.tif"))
            if not pf:
                continue
            vv, vh, T, crs, lab = load_chip(self.a.data, r)
            pvv, pvh, pT, pcrs, _ = load_pre(pf[0])
            ref = align_to(dict(vv=pvv, vh=pvh, transform=pT, crs=pcrs), dict(vv=vv, transform=T, crs=crs))
            res = self.det.detect(vv, vh, T, crs, ref_vv_lin=ref[0], ref_vh_lin=ref[1])
            truth = truth_of(lab)
            good = (res["verdict"] == "FLOOD DETECTED") == truth
            n1 += 1
            ok1 += good
            self.results.append(dict(section="2 change", set="flood image + pre-flood image", case=m[0],
                                     truth="flood" if truth else "no flood", got=res["verdict"], correct=good,
                                     flood_km2=res["flood_km2"]))
            res2 = self.det.detect(pvv, pvh, pT, pcrs, ref_vv_lin=pvv, ref_vh_lin=pvh)
            good2 = res2["verdict"] == "NO FLOOD DETECTED"
            n2 += 1
            ok2 += good2
            self.results.append(dict(section="2 change", set="pre-flood image vs itself", case=m[0], truth="no flood",
                                     got=res2["verdict"], correct=good2, flood_km2=res2["flood_km2"]))
        self.check("2 change detection", "flood image + pre-flood image of the same place", ok1 >= 0.9 * max(n1, 1),
                   f"{ok1}/{n1} correct", "≥ 90%")
        self.check("2 change detection", "pre-flood image compared with itself (must be NO FLOOD)", ok2 == n2,
                   f"{ok2}/{n2} correct", "100%")

    # ------------------------------------------------------------------ 3 formats
    def variants(self, src_lin, T, crs, d, name):
        """Write the same scene in many formats. Returns list of (variant, files, location, pixel_m, kind)."""
        from PIL import Image
        from rasterio.warp import Resampling, calculate_default_transform, reproject
        vv, vh = src_lin
        H, W = vv.shape
        prof = dict(driver="GTiff", height=H, width=W, crs=crs, transform=T)
        out = []

        def w(fn, arrs, dtype, nodata=None, desc=None, p=None):
            pp = dict(p or prof, count=len(arrs), dtype=dtype)
            if nodata is not None:
                pp["nodata"] = nodata
            with rasterio.open(d / fn, "w", **pp) as o:
                for i, a_ in enumerate(arrs, 1):
                    o.write(a_.astype(dtype), i)
                    if desc:
                        o.set_band_description(i, desc[i - 1])
            return str(d / fn)

        db = [10 * np.log10(np.clip(x, 1e-6, None)) for x in (vv, vh)]
        out.append(("GeoTIFF 2-band linear", [w(f"{name}_lin.tif", [vv, vh], "float32", desc=["VV", "VH"])], "", 20, "raster"))
        out.append(("GeoTIFF 2-band dB", [w(f"{name}_db.tif", db, "float32", -9999, ["VV", "VH"])], "", 20, "raster"))
        out.append(("GeoTIFF dB x100 int16", [w(f"{name}_dbx100.tif", [np.round(x * 100) for x in db], "int16", -32768,
                                                ["VV", "VH"])], "", 20, "raster"))
        out.append(("two files VV + VH", [w(f"{name}_VV.tif", [vv], "float32"), w(f"{name}_VH.tif", [vh], "float32")],
                    "", 20, "raster"))
        out.append(("VV band only", [w(f"{name}_onlyvv.tif", [vv], "float32")], "", 20, "raster"))
        # UTM reprojection
        utm = "EPSG:32642"
        T2, W2, H2 = calculate_default_transform(crs, utm, W, H, *_bounds(T, H, W), resolution=20)
        arrs = []
        for x in (vv, vh):
            dst = np.full((H2, W2), np.nan, "float32")
            reproject(x, dst, src_transform=T, src_crs=crs, dst_transform=T2, dst_crs=utm, resampling=Resampling.bilinear,
                      src_nodata=np.nan, dst_nodata=np.nan)
            arrs.append(dst)
        out.append(("GeoTIFF in UTM 42N", [w(f"{name}_utm.tif", arrs, "float32", desc=["VV", "VH"],
                                             p=dict(driver="GTiff", height=H2, width=W2, crs=utm, transform=T2))],
                    "", 20, "raster"))
        for res_m, tag in ((10, "10 m pixels"), (30, "30 m pixels")):
            f = 20 / res_m
            H3, W3 = int(H * f), int(W * f)
            T3 = T * T.scale(W / W3, H / H3)
            arrs = []
            for x in (vv, vh):
                dst = np.full((H3, W3), np.nan, "float32")
                reproject(x, dst, src_transform=T, src_crs=crs, dst_transform=T3, dst_crs=crs,
                          resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
                arrs.append(dst)
            out.append((f"GeoTIFF {tag}", [w(f"{name}_{res_m}m.tif", arrs, "float32", desc=["VV", "VH"],
                                                p=dict(driver="GTiff", height=H3, width=W3, crs=crs, transform=T3))],
                        "", 20, "raster"))
        half = [x.copy() for x in (vv, vh)]
        for x in half:
            x[:, : W // 2] = np.nan
        out.append(("half of the image no-data", [w(f"{name}_half.tif", half, "float32", desc=["VV", "VH"])], "", 20,
                    "raster_half"))
        # PNG family
        g = np.clip((db[0] + 25) / 25 * 255, 0, 255).astype("uint8")
        wb, sb, eb, nb = _bounds(T, H, W)
        Image.fromarray(g).save(d / f"{name}_corners.png")
        out.append(("PNG + typed corners", [str(d / f"{name}_corners.png")], f"{wb} {sb} {eb} {nb}", 20, "png"))
        Image.fromarray(g).save(d / f"{name}_centre.png")
        px = (eb - wb) * 111320 * np.cos(np.deg2rad((sb + nb) / 2)) / W
        out.append(("PNG + centre point", [str(d / f"{name}_centre.png")], f"{(sb + nb) / 2}, {(wb + eb) / 2}", px, "png"))
        Image.fromarray(g).save(d / f"{name}_world.png")
        (d / f"{name}_world.pgw").write_text(f"{T.a}\n0\n0\n{T.e}\n{T.c + T.a / 2}\n{T.f + T.e / 2}\n")
        out.append(("PNG + world file", [str(d / f"{name}_world.png"), str(d / f"{name}_world.pgw")], "", 20, "png"))
        g16 = np.clip((db[0] + 25) / 25 * 65535, 0, 65535).astype("uint16")
        Image.fromarray(g16).save(d / f"{name}_16bit.png")
        out.append(("16-bit PNG + corners", [str(d / f"{name}_16bit.png")], f"{wb} {sb} {eb} {nb}", 20, "png"))
        Image.fromarray(g).convert("L").save(d / f"{name}.jpg", quality=85)
        out.append(("JPEG + corners", [str(d / f"{name}.jpg")], f"{wb} {sb} {eb} {nb}", 20, "png"))
        out.append(("Sentinel-1 GRD .zip (raw DN + calibration)", [_safe_zip(vv, vh, T, H, W, d, name)], "", 20, "raster"))
        crop = [x[: 96, : 96] for x in (vv, vh)]
        out.append(("small crop 96 x 96 px", [w(f"{name}_small.tif", crop, "float32", desc=["VV", "VH"],
                                                p=dict(driver="GTiff", height=96, width=96, crs=crs, transform=T))],
                    "", 20, "small"))
        Image.fromarray(g).save(d / f"{name}_noloc.png")
        out.append(("PNG without location", [str(d / f"{name}_noloc.png")], "", 20, "png_noloc"))
        return out

    def run_formats(self):
        print("\n[3] Input-format robustness", flush=True)
        by_id = {r["chip_id"]: r for r in self.rows}
        bases = []
        flood_id = next((r["chip_id"] for r in self.rows if r["split"] == "test" and float(r["perm_frac"]) > 0.02
                         and float(r["flood_frac"]) > 0.3), None)
        if flood_id:
            bases.append(("flood_with_river", by_id[flood_id], True))
        pre = sorted(Path(self.a.pre_flood).glob(f"PRE_{flood_id}_*.tif")) if flood_id else []
        if pre:
            bases.append(("river_only_april", dict(chip_id=pre[0].stem, pre_file=str(pre[0])), False))
        dry = next((r for r in self.rows if r["split"] == "test_dry" and float(r["water_frac"]) < 0.002), None)
        if dry:
            bases.append(("dry_land", dry, False))
        tmp = Path(tempfile.mkdtemp(prefix="resq_formats_"))
        for name, r, truth in bases:
            vv, vh, T, crs, lab = self.load(r)
            base = self.det.detect(vv, vh, T, crs)
            right = (base["verdict"] == "FLOOD DETECTED") == truth
            self.check("3 formats", f"{name} · reference GeoTIFF answer is correct", right,
                       f"{base['verdict']} (truth: {'flood' if truth else 'no flood'})")
            base_km2 = base["flood_km2"]
            fl = (base["classes"] == 2)
            half_share = float(fl[:, fl.shape[1] // 2:].sum()) / max(float(fl.sum()), 1.0)   # flood kept in "half" variant
            for variant, files, loc, px, kind in self.variants((vv, vh), T, crs, tmp, name):
                try:
                    data = load_any(files, px, loc or None)
                    res = self.det.detect(data["vv"], data["vh"], data["transform"], data["crs"], pixel_m=px)
                    got = res["verdict"]
                    km2 = res["flood_km2"]
                except Exception as e:
                    got, km2 = f"ERROR {type(e).__name__}: {e}"[:120], None
                    res = None
                exp = base["verdict"]          # format robustness = same answer as the reference GeoTIFF
                if kind == "png_noloc":
                    ok = got != "" and not got.startswith("ERROR")          # cannot remove rivers: any verdict, no crash
                    note = "location-free PNG: river cannot be separated (documented limit)"
                elif kind == "small":
                    ok = not got.startswith("ERROR")
                    note = "tiny crop: must run without error"
                else:
                    ok = got == exp
                    note = ""
                    if ok and base["verdict"] == "FLOOD DETECTED" and km2 is not None and base_km2 > 0:
                        scale = half_share if kind == "raster_half" else 1.0
                        tol = 0.35 if kind == "png" else 0.20
                        rel = abs(km2 - base_km2 * scale) / max(base_km2 * scale, 1e-6)
                        ok = rel <= tol
                        note = f"flood {km2} km² vs reference {round(base_km2 * scale, 2)} km² ({rel * 100:.0f}% diff, limit {tol * 100:.0f}%)"
                self.results.append(dict(section="3 formats", set=name, case=variant, truth=exp, got=got, correct=ok,
                                         flood_km2=km2))
                self.check("3 formats", f"{name} · {variant}", ok, f"{got} {note}".strip(), "same answer as reference")
                if not ok and res is not None:
                    self.fails.append((f"format {name}: {variant}", res, None))
        shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ 4 invalid
    def run_invalid(self):
        print("\n[4] Invalid inputs", flush=True)
        from PIL import Image
        d = Path(tempfile.mkdtemp(prefix="resq_invalid_"))
        rng = np.random.default_rng(0)
        photo = np.clip(np.stack([np.full((200, 300), 120), np.full((200, 300), 170), np.full((200, 300), 90)], -1)
                        + rng.normal(0, 25, (200, 300, 3)), 0, 255).astype("uint8")
        Image.fromarray(photo).save(d / "colour_photo.jpg")
        with rasterio.open(d / "optical_4band.tif", "w", driver="GTiff", height=100, width=100, count=4, dtype="uint16",
                           crs="EPSG:4326", transform=rasterio.transform.from_origin(67, 27, 0.0002, 0.0002)) as o:
            o.write((rng.random((4, 100, 100)) * 3000).astype("uint16"))
        Image.fromarray(np.full((80, 80), 100, "uint8")).save(d / "blank.png")
        with rasterio.open(d / "all_nodata.tif", "w", driver="GTiff", height=50, width=50, count=2, dtype="float32",
                           nodata=-9999, crs="EPSG:4326", transform=rasterio.transform.from_origin(67, 27, 0.0002, 0.0002)) as o:
            o.write(np.full((2, 50, 50), -9999, "float32"))
        good = d / "good.tif"
        with rasterio.open(good, "w", driver="GTiff", height=256, width=256, count=2, dtype="float32", crs="EPSG:4326",
                           transform=rasterio.transform.from_origin(67, 27, 0.0002, 0.0002)) as o:
            o.write((rng.random((2, 256, 256)) * 0.1 + 0.01).astype("float32"))
        (d / "truncated.tif").write_bytes(good.read_bytes()[:3000])
        (d / "text_named.tif").write_text("this is not an image")
        (d / "empty.png").write_bytes(b"")
        (d / "document.pdf").write_bytes(b"%PDF-1.4")
        for f in sorted(d.iterdir()):
            if f.name == "good.tif":
                continue
            try:
                data = load_any([str(f)])
                self.det.detect(data["vv"], data["vh"], data["transform"], data["crs"])
                ok, msg = False, "was processed (should have been refused)"
            except InputError as e:
                ok, msg = True, f"refused: {str(e)[:90]}"
            except Exception as e:
                ok, msg = False, f"crashed: {type(e).__name__}: {str(e)[:80]}"
            self.results.append(dict(section="4 invalid", set="invalid input", case=f.name, truth="refused", got=msg,
                                     correct=ok, flood_km2=None))
            self.check("4 invalid inputs", f.name, ok, msg)
        shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ 5 fallback
    def run_fallback(self):
        print("\n[5] Fallback mode (no U-Net)", flush=True)
        sets = self.scenario_sets()
        for name in ("S1 flood season, held-out regions", "S6 rivers/lakes only, April 2022"):
            rows = sets[name][: 20]
            n_ok, cf = 0, Confusion()
            for r in rows:
                vv, vh, T, crs, lab = self.load(r)
                res = self.fb.detect(vv, vh, T, crs)
                n_ok += (res["verdict"] == "FLOOD DETECTED") == truth_of(lab)
                cf.add((res["classes"] == 2) | (res["classes"] == 4), lab[1])
                assert res["mode"].startswith("fallback")
            m = cf.result()
            iou = f", flood IoU {m['iou']:.3f}" if m["tp"] + m["fn"] else ""
            self.check("5 fallback", f"fallback only · {name}", True,
                       f"verdict {n_ok}/{len(rows)} correct{iou} (mode correctly reported as fallback)", "runs, labelled")

    # ------------------------------------------------------------------ 6 stability
    def run_stability(self):
        print("\n[6] Stability", flush=True)
        rows = [r for r in self.rows if r["split"] == "test"][:6]
        same = True
        agree = 0
        for r in rows:
            vv, vh, T, crs, lab = self.load(r)
            a1 = self.det.detect(vv, vh, T, crs)
            a2 = self.det.detect(vv, vh, T, crs)
            same &= bool((a1["classes"] == a2["classes"]).all())
            tta = self.det.tta
            self.det.tta = not tta
            b = self.det.detect(vv, vh, T, crs)
            self.det.tta = tta
            agree += a1["verdict"] == b["verdict"]
        self.check("6 stability", "identical result when run twice", same, "pixel-identical" if same else "differs")
        self.check("6 stability", "TTA vs no-TTA verdict agreement", agree == len(rows), f"{agree}/{len(rows)}")

    # ------------------------------------------------------------------ 7 demo
    def run_demo(self):
        print("\n[7] Demo pack", flush=True)
        from flood_app import App
        app = App(self.a.data, self.a.jrc_dir, self.a.demo, self.a.model)
        app.cache[app.current] = self.det
        head, rows, _ = app.run_demo(not self.a.no_tta, None)
        for r in rows:
            ok = r[4].endswith("PASS")
            self.results.append(dict(section="7 demo", set="demo", case=r[0], truth=r[2], got=r[3], correct=ok, flood_km2=""))
            self.check("7 demo pack", r[0], ok, f"expected {r[2]}, got {r[3]}")

    # ------------------------------------------------------------------ report
    def report(self, seconds):
        with open(self.out / "checks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.checks[0].keys()))
            w.writeheader()
            w.writerows(self.checks)
        with open(self.out / "all_cases.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
            w.writeheader()
            w.writerows(self.results)
        n_pass = sum(c["result"] == "PASS" for c in self.checks)
        L = [f"# Verification report – {self.a.model}", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · TTA {not self.a.no_tta} · {len(self.results)} test cases · "
             f"{len(self.checks)} checks · run time {seconds / 60:.1f} min", "",
             f"## Result: {n_pass}/{len(self.checks)} checks passed", ""]
        sec = None
        for c in self.checks:
            if c["section"] != sec:
                sec = c["section"]
                L += ["", f"### {sec}", "", "| Check | Result | Details | Target |", "|---|---|---|---|"]
            L.append(f"| {c['check']} | {'✅' if c['result'] == 'PASS' else '❌'} {c['result']} | {c['detail']} | {c['target']} |")
        if self.fails:
            L += ["", "## Wrong answers (see failures.png)", ""] + [f"- {f[0]}" for f in self.fails[:40]]
        (self.out / "VERIFICATION_REPORT.md").write_text("\n".join(L) + "\n")
        self._fig()
        print(f"\n{n_pass}/{len(self.checks)} checks passed -> {self.out / 'VERIFICATION_REPORT.md'}")

    def _fig(self):
        if not self.fails:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fl = self.fails[:16]
        fig, ax = plt.subplots(len(fl), 2, figsize=(8, 3.6 * len(fl)))
        ax = np.atleast_2d(ax)
        for i, (t, res, lab) in enumerate(fl):
            ax[i, 0].imshow(render.sar_gray(res["vv_lee_db"], res["valid"]))
            ax[i, 0].set_title(t[:60], fontsize=8)
            ax[i, 1].imshow(render.overlay(res))
            ax[i, 1].set_title(f"{res['verdict']} · flood {res['flood_km2']} km²", fontsize=8)
            for x in ax[i]:
                x.axis("off")
        plt.tight_layout()
        plt.savefig(self.out / "failures.png", dpi=70)
        plt.close()


def _safe_zip(vv, vh, T, H, W, d, name):
    """A raw Sentinel-1 GRD product as delivered (.zip / SAFE): radar geometry (flipped), uint16 DN,
    ground control points and a range-dependent sigma0 calibration table."""
    import os
    import zipfile
    from rasterio.control import GroundControlPoint
    A = np.linspace(300, 520, W)
    g = [GroundControlPoint(row=i, col=j, x=(T * (j, H - i))[0], y=(T * (j, H - i))[1], z=0)
         for i in np.linspace(0, H, 6) for j in np.linspace(0, W, 6)]
    zp = Path(d) / f"S1A_IW_GRDH_1SDV_{name}.zip"
    base = f"S1A_IW_GRDH_1SDV_{name}.SAFE"
    with zipfile.ZipFile(zp, "w") as z:
        for pol, a in (("vv", vv), ("vh", vh)):
            n = f"s1a-iw-grd-{pol}-20220830t010203-001"
            dn = np.clip(np.sqrt(np.nan_to_num(a, nan=0)) * A[None, :], 0, 65535).astype("uint16")[::-1]
            tif = Path(d) / f"{n}.tiff"
            with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint16") as o:
                o.gcps = (g, "EPSG:4326")
                o.write(dn, 1)
            z.write(tif, f"{base}/measurement/{n}.tiff")
            os.remove(tif)
            idx = np.linspace(0, W - 1, 20).astype(int)
            px, sv = " ".join(map(str, idx)), " ".join(f"{A[i]:.3f}" for i in idx)
            vecs = "".join(f"<calibrationVector><line>{ln}</line><pixel count='20'>{px}</pixel>"
                           f"<sigmaNought count='20'>{sv}</sigmaNought></calibrationVector>" for ln in (0, H // 2, H))
            z.writestr(f"{base}/annotation/calibration/calibration-{n}.xml",
                       f"<calibration><calibrationVectorList>{vecs}</calibrationVectorList></calibration>")
    return str(zp)


def _bounds(T, H, W):
    return (T.c, T.f + T.e * H, T.c + T.a * W, T.f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT.parent / "outputs" / "FLOOD_TRAINING_DATA"))
    ap.add_argument("--jrc-dir", default=str(ROOT.parent / "outputs" / "datasets" / "jrc_global_surface_water_pakistan"))
    ap.add_argument("--demo", default=str(ROOT.parent / "outputs" / "DEMO_IMAGES"))
    ap.add_argument("--pre-flood", default=str(ROOT.parent / "outputs" / "FLOOD_TRAINING_DATA" / "00_raw_downloads" /
                                               "sentinel1_rtc_pre_flood_2022"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--quick", action="store_true", help="12 scenes per set (smoke test)")
    ap.add_argument("--sections", default="12345678")
    a = ap.parse_args()
    mp = Path(a.model)
    a.out = a.out or str((mp if mp.is_dir() else mp.parent) / "verification")
    t0 = time.time()
    v = V(a)
    print("model:", a.model, "|", v.det.mode, "| params:", v.det.params)
    steps = {"1": v.run_scenarios, "2": v.run_change, "3": v.run_formats, "4": v.run_invalid, "5": v.run_fallback,
             "6": v.run_stability, "7": v.run_demo, "8": v.run_single_pol}
    for s in a.sections:
        try:
            steps[s]()
        except Exception as e:
            traceback.print_exc()
            v.check(f"{s} (section)", "section ran", False, f"{type(e).__name__}: {e}")
    v.report(time.time() - t0)
    sys.exit(0 if all(c["result"] == "PASS" for c in v.checks) else 1)


if __name__ == "__main__":
    main()
