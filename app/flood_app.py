"""ResQFlood flood-detection interface (runs locally, open http://localhost:7860).

  python app/flood_app.py                    # picks runs/M2 if it exists, else runs/M1_unet_r34_lee
  python app/flood_app.py --model runs/M1_unet_r34_lee --port 7860

Tabs
  1. Detect floods  : upload one or many images (Sentinel-1 GeoTIFFs, VV+VH pairs, PNG/JPG), optional pre-flood
                      images, type locations for PNGs in the table, press Detect -> verdict per image, maps, downloads.
  2. Demo set       : runs the prepared demo images (DEMO_IMAGES) and checks every answer against the expected one.
  3. Test-set chips : held-out chips with ground truth next to the detection.
  4. How it works   : decision logic, legend and the measured accuracy of the loaded model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from resqflood import render  # noqa: E402
from resqflood.detect import FloodDetector  # noqa: E402
from resqflood.sar_io import IMG_EXT, WORLD_EXT, InputError, align_to, load_any  # noqa: E402

RASTER_EXT = (".tif", ".tiff", ".img", ".jp2", ".vrt")
POL_RE = re.compile(r"(?i)(^|[_\-. ])(vv|vh)(?=$|[_\-. ])")
PRE_RE = re.compile(r"(?i)(^|[_\-. ])(pre|before|preflood|pre_flood|reference|ref|dry)(?=$|[_\-. ])")


# ------------------------------------------------------------------ helpers
def find_models():
    out = {}
    m2 = ROOT / "runs" / "M2" / "best_model.json"
    if m2.exists():
        out["M2 (new model)"] = str(m2)
    for d in sorted((ROOT / "runs").glob("*")):
        if (d / "best.pt").exists() and d.name.startswith("M1"):
            out[f"M1 ({d.name})"] = str(d)
    return out


NICE = {"val": "Validation – flood season", "test": "Test – held-out regions (flood season)",
        "test_dadu_district": "Test – Dadu district", "test_external_s1f_hand": "External – Sen1Floods11 2017, hand-labelled",
        "val_dry": "Validation – dry season (no flood)", "test_dry": "Test – dry season (no flood)",
        "river_only_pre_flood (test+other areas)": "Test – rivers/lakes in April 2022 (no flood)"}


def model_card(model_path):
    p = Path(model_path)
    d = p if p.is_dir() else p.parent
    f = d / "detector_eval" / "detector_metrics.json"
    if not f.exists():
        return "_Detector not yet tuned for this model (run scripts/tune_detector.py)._"
    m = json.load(open(f))
    lines = ["| Evaluation set | Scenes | Flood IoU (map accuracy) | Correct flood / no-flood answer |", "|---|---|---|---|"]
    for k, s in m["splits"].items():
        t = s["tuned_detector"]
        iou = "– (no flood present)" if t["flood_iou"] != t["flood_iou"] else f"{t['flood_iou']:.3f}"
        lines.append(f"| {NICE.get(k, k)} | {s['n_chips']} | {iou} | {t['chip_accuracy'] * 100:.0f}% |")
    return "\n".join(lines)


def norm_key(name, keep_case=False):
    stem = Path(name).stem
    k = POL_RE.sub(r"\1", stem).strip("_-. ")
    return k if keep_case else k.lower()


def pre_key(name):
    return PRE_RE.sub(r"\1", norm_key(name)).strip("_-. ")


def stage(files, workdir):
    """Copy uploads into one folder (so sidecar files sit next to their image)."""
    out = []
    Path(workdir).mkdir(parents=True, exist_ok=True)
    for f in files or []:
        src = Path(f if isinstance(f, str) else f.name)
        dst = Path(workdir) / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        out.append(dst)
    return out


def group_scenes(paths):
    """-> list of dict(name, kind, files). VV/VH single-band rasters with matching names become one scene."""
    rasters = [p for p in paths if p.suffix.lower() in RASTER_EXT]
    images = [p for p in paths if p.suffix.lower() in IMG_EXT]
    scenes, used = [], set()
    for p in paths:                      # a complete Sentinel-1 GRD product (.zip / SAFE): one scene, calibrated
        if p.suffix.lower() == ".zip":
            scenes.append(dict(name=p.stem, kind="Sentinel-1 GRD .zip", files=[str(p)]))
    for p in rasters:
        if p in used:
            continue
        pol = POL_RE.search(p.stem)
        mate = None
        if pol:
            for q in rasters:
                if q != p and q not in used and norm_key(q.name) == norm_key(p.name) and POL_RE.search(q.stem):
                    mate = q
                    break
        files = [p] + ([mate] if mate else [])
        used.update(files)
        scenes.append(dict(name=norm_key(p.name, keep_case=True) if mate else p.stem, kind="raster (VV+VH files)" if mate else "raster",
                           files=[str(f) for f in files]))
    for p in images:
        side = None
        for ext in WORLD_EXT:
            if p.with_suffix(ext).exists():
                side = "world file"
        if p.with_suffix(".bbox.txt").exists() or p.with_name(p.stem + ".bbox.txt").exists():
            side = "bbox file"
        scenes.append(dict(name=p.stem, kind="image", files=[str(p)], sidecar=side))
    other = [p.name for p in paths if p.suffix.lower() not in RASTER_EXT + IMG_EXT + WORLD_EXT + (".txt", ".zip")]
    return scenes, other


def pair_reference(scene, refs, n_scenes):
    """Match a pre-flood image by name (dadu.png <-> dadu_before.png); a single reference is used for a single image."""
    if not refs:
        return None
    k = pre_key(scene["name"])
    for r in refs:
        rk = pre_key(r["name"])
        if rk == k or (len(rk) > 3 and (k.startswith(rk) or rk.startswith(k))):
            return r
    return refs[0] if (len(refs) == 1 and n_scenes == 1) else None


def prob_img(res):
    if res["prob"] is None:
        return None
    p = (np.clip(res["prob"], 0, 1) * 255).astype("uint8")
    rgb = np.stack([255 - p, 255 - p // 3, np.full_like(p, 255)], -1)
    rgb[~res["valid"]] = 0
    return rgb


def fb_img(res):
    g = render.sar_gray(res["vv_lee_db"], res["valid"]).copy()
    g[res["fallback"]] = (0, 200, 255)
    return g


def verdict_md(name, res, extra=""):
    flood = res["verdict"] == "FLOOD DETECTED"
    head = f"### {'🌊' if flood else '✅'} {name}: **{res['verdict']}**"
    lines = [head]
    if extra:
        lines.append(extra)
    lines += [f"- **Flood:** {res['flood_km2']} km² ({res['flood_pct']}% of the image) · confirmed by both methods: "
              f"{res['confirmed_by_both_pct']}%",
              f"- **Permanent water** (rivers/lakes, not flood): {res['permanent_water_km2']} km² · "
              f"**seasonal water**: {res['seasonal_water_km2']} km² · source: {res['permanent_water_source']}",
              f"- Flood counted for the verdict (outside rivers, banks and specks): {res['verdict_flood_km2']} km²",
              f"- Detector: {res['mode']} · radar threshold {res['vv_threshold_db']} dB · "
              f"U-Net thresholds {res['unet_thresholds']}"]
    if res["notes"]:
        lines.append("**Notes:**\n" + "\n".join(f"- {n}" for n in res["notes"]))
    return "\n".join(lines)


# ------------------------------------------------------------------ app
class App:
    def __init__(self, data_root, jrc_dir, demo_dir, default_model):
        self.data_root = data_root if data_root and Path(data_root).exists() else None
        self.jrc_dir, self.demo_dir = jrc_dir, demo_dir
        self.norm = str(Path(data_root) / "02_splits" / "norm_stats.json") if data_root else None
        self.models = find_models()
        if default_model and str(default_model) not in self.models.values():
            self.models[f"{Path(default_model).name if Path(default_model).is_dir() else Path(default_model).parent.name} (selected)"] = str(default_model)
        self.cache = {}
        self.current = default_model or (list(self.models.values())[0] if self.models else None)
        self.last = {}

    def det(self, model_path=None):
        key = model_path or self.current
        if key not in self.cache:
            if key and Path(key).exists():
                self.cache[key] = FloodDetector.from_model_file(key, jrc_dir=self.jrc_dir, norm_stats=self.norm)
            else:
                self.cache[key] = FloodDetector(jrc_dir=self.jrc_dir, norm_stats=self.norm)
        return self.cache[key]

    # ---------------- one scene
    def run_scene(self, scene, location, pixel_m, tta, ref=None, outdir=None):
        d = self.det()
        d.tta = tta
        t0 = time.time()
        data = load_any(scene["files"], pixel_m, location or None)
        rvv = rvh = None
        if ref:
            r = load_any(ref["files"], pixel_m, location or None)
            rvv, rvh, note = align_to(r, data)
            data["notes"].append(f"Pre-flood image '{ref['name']}' used. {note}")
        res = d.detect(data["vv"], data["vh"], data["transform"], data["crs"], pixel_m=pixel_m,
                       ref_vv_lin=rvv, ref_vh_lin=rvh)
        res["notes"] = data["notes"] + res["notes"]
        res["seconds"] = round(time.time() - t0, 1)
        if outdir:
            render.save(res, outdir, data["transform"], data["crs"], name=re.sub(r"[^\w\-.]", "_", scene["name"]))
        return res

    # ---------------- tab 1 callbacks
    def on_upload(self, files, refs):
        if not files:
            return [], "Upload images to start."
        wd = Path(tempfile.mkdtemp(prefix="resq_in_"))
        paths = stage(files, wd)
        scenes, other = group_scenes(paths)
        rows = []
        for s in scenes:
            if s["kind"].startswith("raster"):
                loc_note = "from the GeoTIFF"
            elif s["kind"].startswith("Sentinel-1"):
                loc_note = "from the product (ground control points)"
            elif s.get("sidecar"):
                loc_note = f"from {s['sidecar']}"
            else:
                loc_note = "type it → or leave empty"
            rows.append([s["name"], s["kind"], "", loc_note])
        msg = f"{len(scenes)} image(s) ready."
        if other:
            msg += f" Ignored unsupported files: {', '.join(other)}."
        if refs:
            msg += f" {len(refs)} pre-flood file(s) will be matched by name."
        return rows, msg

    def on_detect(self, files, refs, table, pixel_m, tta, model_label):
        if model_label and model_label in self.models:
            self.current = self.models[model_label]
        if not files:
            return "Upload at least one image.", None, [], None, None, None, None, None, None, gr_update_choices([])
        wd = Path(tempfile.mkdtemp(prefix="resq_run_"))
        scenes, _ = group_scenes(stage(files, wd / "in"))
        rscenes, _ = group_scenes(stage(refs, wd / "ref")) if refs else ([], [])
        locs = {}
        try:
            for row in (table.values.tolist() if hasattr(table, "values") else table or []):
                if row and row[0] and len(row) > 2 and str(row[2]).strip():
                    locs[str(row[0])] = str(row[2]).strip()
        except Exception:
            pass
        out = wd / "results"
        summary, gallery = [], []
        self.last = {}
        for s in scenes:
            ref = pair_reference(s, rscenes, len(scenes))
            try:
                res = self.run_scene(s, locs.get(s["name"]), pixel_m, tta, ref, out)
                self.last[s["name"]] = res
                summary.append([s["name"], res["verdict"], res["flood_km2"], res["flood_pct"], res["confirmed_by_both_pct"],
                                res["permanent_water_km2"], res["seasonal_water_km2"], res["permanent_water_source"],
                                res["mode"], len(res["notes"]), res["seconds"]])
                gallery.append((render.overlay(res), f"{s['name']} · {res['verdict']} · flood {res['flood_km2']} km²"))
            except InputError as e:
                summary.append([s["name"], f"NOT PROCESSED: {e}", "", "", "", "", "", "", "", "", ""])
            except Exception as e:
                traceback.print_exc()
                summary.append([s["name"], f"ERROR: {type(e).__name__}: {e}", "", "", "", "", "", "", "", "", ""])
        if out.exists():
            with open(out / "report.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(SUMMARY_COLS)
                w.writerows(summary)
            zp = wd / "resqflood_results.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
                for p in out.iterdir():
                    z.write(p, p.name)
        else:
            zp = None
        n_fl = sum(1 for r in summary if r[1] == "FLOOD DETECTED")
        n_ok = sum(1 for r in summary if r[1] in ("FLOOD DETECTED", "NO FLOOD DETECTED"))
        head = (f"**{n_ok}/{len(summary)} images processed · {n_fl} with FLOOD DETECTED** · model: "
                f"{model_label or 'default'} ({self.det().mode})")
        first = next(iter(self.last), None)
        detail = self.detail(first) if first else ("", None, None, None, None)
        return (head, summary, gallery, str(zp) if zp else None, *detail, gr_update_choices(list(self.last)))

    def detail(self, name):
        if not name or name not in self.last:
            return "", None, None, None, None
        res = self.last[name]
        return (verdict_md(name, res), render.overlay(res), render.sar_gray(res["vv_lee_db"], res["valid"]),
                prob_img(res), fb_img(res))

    # ---------------- tab 2: demo set
    def demo_cases(self):
        f = Path(self.demo_dir) / "expected.csv" if self.demo_dir else None
        if not f or not f.exists():
            return []
        return list(csv.DictReader(open(f)))

    def run_demo(self, tta, model_label):
        out = None
        for out in self.run_demo_iter(tta, model_label):
            pass
        return out

    def run_demo_iter(self, tta, model_label):
        if model_label and model_label in self.models:
            self.current = self.models[model_label]
        rows, gallery, ok = [], [], 0
        cases = self.demo_cases()
        for i, c in enumerate(cases):
            yield f"Running case {i + 1}/{len(cases)}: {c['case']} …", rows, gallery
            files = [str(Path(self.demo_dir) / x) for x in c["files"].split(";")]
            ref = None
            if c.get("reference"):
                ref = dict(name=Path(c["reference"]).stem, files=[str(Path(self.demo_dir) / c["reference"])])
            scene = dict(name=c["case"], files=files)
            warned = False
            try:
                res = self.run_scene(scene, c.get("location") or None, float(c.get("pixel_m") or 20), tta, ref)
                got = res["verdict"]
                warned = any("WARNING" in n for n in res["notes"])
                extra = (f"{res['flood_km2']} km² flood, {res['permanent_water_km2']} km² permanent water "
                         f"({res['permanent_water_source']})" + (" · WARNING shown" if warned else ""))
                gallery.append((render.overlay(res), f"{c['case']} · {got}"))
            except InputError as e:
                got, extra = "REJECTED", str(e)[:160]
            except Exception as e:
                got, extra = "ERROR", f"{type(e).__name__}: {e}"[:160]
            exp = c["expected"]
            passed = (exp == "ANY" and got != "ERROR") or (exp == "WARNED" and warned) or got == exp
            ok += passed
            rows.append([c["case"], c["type"], c["expected"], got, "✅ PASS" if passed else "❌ FAIL", extra, c["what_it_shows"]])
        n = len(rows)
        head = f"## Demo check: {ok}/{n} cases correct" if n else "No demo set found (run scripts/make_demo_pack.py)."
        yield head, rows, gallery

    # ---------------- tab 3: test chips
    def chip_choices(self):
        out = {}
        if not self.data_root:
            return out
        from resqflood.train.data import read_manifest
        for r in read_manifest(self.data_root):
            if r["split"] in ("test", "test_external_s1f_hand"):
                out[f"{r['chip_id']} | {r['split']} | {r['date']} | truth flood {100 * float(r['flood_frac']):.1f}%"] = r
        return out

    def run_chip(self, label, tta, model_label):
        if model_label and model_label in self.models:
            self.current = self.models[model_label]
        if not label:
            return "Pick a chip.", None, None, None
        from test_cases import load_chip
        r = self.chips[label]
        vv, vh, T, crs, lab = load_chip(self.data_root, r)
        d = self.det()
        d.tta = tta
        res = d.detect(vv, vh, T, crs)
        truth = render.sar_gray(res["vv_lee_db"], res["valid"]).copy()
        truth[lab[0] == 1] = (30, 120, 230)
        truth[lab[1] == 1] = (220, 20, 30)
        truth[lab[0] == 255] = (128, 128, 128)
        v = lab[1] != 255
        tf = (lab[1] == 1).sum() >= 0.01 * max(v.sum(), 1)
        ok = (res["verdict"] == "FLOOD DETECTED") == tf
        extra = (f"**Ground truth:** {'flood' if tf else 'no flood'} ({100 * float(r['flood_frac']):.1f}% flooded) → "
                 f"verdict **{'CORRECT' if ok else 'WRONG'}**")
        return verdict_md(r["chip_id"], res, extra), render.sar_gray(res["vv_lee_db"], res["valid"]), truth, render.overlay(res)


SUMMARY_COLS = ["image", "verdict", "flood km²", "flood %", "confirmed by both %", "permanent water km²",
                "seasonal water km²", "permanent-water source", "detector", "notes", "seconds"]


def gr_update_choices(choices):
    import gradio as gr
    return gr.update(choices=choices, value=choices[0] if choices else None)


HOW = """
### How a decision is made
1. **Radar preparation** – the image is put on a 20 m grid, speckle-filtered (Lee 7×7) and converted to dB.
2. **Two independent water detectors**
   - the **U-Net** (deep learning model trained on 379 chips of the 2022 Pakistan floods) gives a water probability;
   - the **fallback**: a radar-darkness threshold on VV (tuned on validation data). If the model is missing or fails,
     the fallback runs alone and the result says so.
3. **Fusion** – a pixel is FLOOD when the U-Net is confident, or when the U-Net is moderately sure *and* the fallback
   agrees. The thresholds were chosen on the validation split only.
4. **Normal water is never flood**
   - **permanent water** (blue): JRC Global Surface Water (water ≥ 80 % of the time or all 12 months) – needs a
     location; tiles anywhere in the world are downloaded automatically – and/or water in a **pre-flood image**;
   - **seasonal water** (cyan): areas that are normally wet part of the year.
5. **Verdict** – FLOOD DETECTED when enough flood remains outside rivers/lakes, their banks and tiny specks.

### Colours
🟥 flood · 🟧 possible flood (weak evidence, not counted) · 🟦 permanent water · 🩵 seasonal water · ⬛ no data

### Inputs
- **Sentinel-1 GeoTIFF** (best): VV+VH in one file, or two files whose names contain VV and VH. Linear, dB or dB×100.
- **PNG/JPG** of a radar image: type its location in the table (`west south east north` or `lat, lon` of the centre),
  or upload a world file (.pgw/.jgw) / `<name>.bbox.txt` with it. Without a location, permanent water cannot be removed.
- **Pre-flood image** (optional): same area before the flood; its water counts as normal water. Matched by file name
  (e.g. `dadu_before.png` ↔ `dadu.png`), or used for the single uploaded image.
- Optical colour photos / multispectral rasters are flagged: this detector reads radar.
"""


def build(app: App):
    import gradio as gr
    labels = list(app.models)
    cur_label = next((k for k, v in app.models.items() if v == app.current), labels[0] if labels else None)
    app.chips = app.chip_choices()
    with gr.Blocks(title="ResQFlood · SAR flood detection") as ui:
        with gr.Row():
            gr.Markdown("# 🌊 ResQFlood · SAR flood detection\nU-Net + radar-threshold fallback · permanent & seasonal "
                        "water removal · Sentinel-1 GeoTIFF or PNG/JPG")
            model_dd = gr.Dropdown(choices=labels, value=cur_label, label="Model", scale=0, min_width=320)
        with gr.Accordion("Measured accuracy of the selected model (held-out data)", open=False):
            card = gr.Markdown(model_card(app.current) if app.current else "_No trained model found: fallback only._")
        model_dd.change(lambda l: model_card(app.models.get(l)) if l else "", model_dd, card)

        with gr.Tab("1 · Detect floods"):
            with gr.Row():
                with gr.Column(scale=3):
                    files = gr.File(label="Images during/after the flood (GeoTIFF, VV/VH pairs, Sentinel-1 GRD .zip, PNG/JPG + optional "
                                          ".pgw/.bbox.txt)", file_count="multiple", type="filepath")
                    refs = gr.File(label="Optional: PRE-FLOOD images of the same areas (water there = normal water)",
                                   file_count="multiple", type="filepath")
                with gr.Column(scale=2):
                    status = gr.Markdown("Upload images to start.")
                    table = gr.Dataframe(headers=["image", "type", "location (W S E N  or  lat, lon)", "location source"],
                                         datatype=["str", "str", "str", "str"], interactive=True, wrap=True,
                                         label="Locations (type for PNG/JPG; GeoTIFFs already have one)")
                    with gr.Accordion("Settings", open=False):
                        pix = gr.Number(value=20, label="Pixel size of PNG/JPG images (metres)")
                        tta = gr.Checkbox(value=True, label="Test-time augmentation (more accurate, slower)")
                    go = gr.Button("🔍 Detect floods", variant="primary", size="lg")
            head = gr.Markdown()
            summ = gr.Dataframe(headers=SUMMARY_COLS, label="Results", wrap=True)
            gal = gr.Gallery(label="Flood maps (click to enlarge)", columns=4, height=360)
            dl = gr.File(label="Download all results (maps, GeoTIFF classes, JSON, report.csv)")
            with gr.Accordion("Details of one image", open=True):
                pick = gr.Dropdown(label="Image", choices=[])
                det_md = gr.Markdown()
                with gr.Row():
                    i1 = gr.Image(label="Detection overlay")
                    i2 = gr.Image(label="Radar (VV, Lee filtered)")
                    i3 = gr.Image(label="U-Net water probability")
                    i4 = gr.Image(label="Fallback threshold water (cyan)")
            files.change(app.on_upload, [files, refs], [table, status])
            refs.change(app.on_upload, [files, refs], [table, status])
            go.click(app.on_detect, [files, refs, table, pix, tta, model_dd],
                     [head, summ, gal, dl, det_md, i1, i2, i3, i4, pick])
            pick.change(app.detail, pick, [det_md, i1, i2, i3, i4])

        with gr.Tab("2 · Demo set"):
            gr.Markdown("Prepared real Sentinel-1 scenes of every situation (flood, flood + river, river/lake only, "
                        "dry land, PNG with typed coordinates, PNG with world file, pre-flood comparison, full Dadu "
                        "district, and invalid inputs). Each has an expected answer; the check shows PASS/FAIL.")
            dgo = gr.Button("▶ Run all demo cases", variant="primary")
            dtta = gr.Checkbox(value=True, label="Test-time augmentation")
            dhead = gr.Markdown()
            dtab = gr.Dataframe(headers=["case", "input type", "expected", "result", "check", "details", "what it shows"],
                                wrap=True)
            dgal = gr.Gallery(columns=4, height=360, label="Demo flood maps")
            dgo.click(app.run_demo_iter, [dtta, model_dd], [dhead, dtab, dgal])

        with gr.Tab("3 · Test-set chips"):
            gr.Markdown("Held-out chips (never used for training) with the ground truth next to the detection.")
            with gr.Row():
                dd = gr.Dropdown(choices=list(app.chips), label="Choose a test chip")
                tta2 = gr.Checkbox(value=True, label="Test-time augmentation")
                b2 = gr.Button("Run", variant="primary")
            v2 = gr.Markdown()
            with gr.Row():
                s2 = gr.Image(label="Radar")
                t2 = gr.Image(label="Ground truth (red flood, blue other water, grey unknown)")
                o2 = gr.Image(label="Detection")
            b2.click(app.run_chip, [dd, tta2, model_dd], [v2, s2, t2, o2])

        with gr.Tab("4 · How it works"):
            gr.Markdown(HOW)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="runs/M2/best_model.json or runs/M1_unet_r34_lee (default: newest)")
    ap.add_argument("--data", default=str(ROOT.parent / "outputs" / "FLOOD_TRAINING_DATA"))
    ap.add_argument("--jrc-dir", default=str(ROOT.parent / "outputs" / "datasets" / "jrc_global_surface_water_pakistan"))
    ap.add_argument("--demo", default=str(ROOT.parent / "outputs" / "DEMO_IMAGES"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="also create a temporary public link")
    a = ap.parse_args()
    app = App(a.data, a.jrc_dir, a.demo, a.model)
    print("models found:", app.models, "| using:", app.current)
    d = app.det()
    print("detector:", d.mode, d.error or "")
    build(app).launch(server_name="0.0.0.0", server_port=a.port, share=a.share, show_error=True)


if __name__ == "__main__":
    main()
