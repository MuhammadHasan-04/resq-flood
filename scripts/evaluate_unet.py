"""Evaluate trained U-Net checkpoint(s) on val / test / Dadu / Sen1Floods11-hand, against threshold baselines.

  python scripts/evaluate_unet.py --data ..\\outputs\\FLOOD_TRAINING_DATA --ckpt runs\\unet_r34_lee\\best.pt
  (several --ckpt values = ensemble, probabilities averaged)

Steps: 8-view test-time augmentation -> probability threshold tuned on VAL only -> metrics on every split.
"flood" metrics = predicted water minus JRC permanent water, scored against the flood label.
Baselines on the same pixels: global VV threshold tuned on val, and per-chip Otsu on VV (Lee).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resqflood.sar.dataset import CHANNELS  # noqa: E402
from resqflood.train.core import Confusion, best_threshold, build_unet, predict_prob  # noqa: E402
from resqflood.train.data import ChipCache, read_manifest, select_rows  # noqa: E402

DADU_BBOX = (67.14, 26.21, 68.04, 27.47)


def otsu(v):
    v = v[np.isfinite(v)]
    h, e = np.histogram(v, 256)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    wf = w[-1] - w
    mb, mf = m / np.maximum(w, 1), (m[-1] - m) / np.maximum(wf, 1)
    return c[np.argmax(w * wf * (mb - mf) ** 2)]


def load_models(paths, device):
    models, cfg = [], None
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        c = ck["config"]
        m, _ = build_unet(c["encoder"], c["in_channels"], "none", c.get("attention", "scse"))
        m.load_state_dict(ck["model"])
        models.append(m.to(device).eval())
        if cfg and (cfg["channels"], cfg["target"]) != (c["channels"], c["target"]):
            raise SystemExit("all checkpoints in an ensemble must use the same --channels and --target")
        cfg = c
        print(f"loaded {p}  (epoch {ck.get('epoch')}, val IoU {ck.get('val', {}).get('iou')})")
    return models, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--threshold", type=float, default=0, help="0 = tune on val")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cfg = load_models(a.ckpt, device)
    out = Path(a.out or Path(a.ckpt[0]).parent / "eval")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(a.data)
    splits = {s: select_rows(rows, s) for s in ["val", "test", "test_external_s1f_hand", "val_dry", "test_dry"]}
    if a.limit:
        splits = {k: v[:a.limit] for k, v in splits.items()}
    vv_name = CHANNELS[cfg["channels"]][0]

    res = {}
    per_chip = []
    for name, rws in splits.items():
        if not rws:
            continue
        print(f"[{name}] {len(rws)} chips")
        c = ChipCache(a.data, rws, cfg["channels"], verbose=False)
        st = c.stats["channels"][vv_name]
        probs = [predict_prob(models, c.x[i], device, tta=not a.no_tta) for i in range(len(rws))]
        vv_db = [c.x[i][0].astype("float32") * st["std"] + st["mean"] for i in range(len(rws))]
        res[name] = dict(c=c, probs=probs, vv=vv_db, rows=rws)

    v = res["val"]
    vp, vy = list(v["probs"]), list(v["c"].water)
    if "val_dry" in res:          # dry-season validation chips also count, so false alarms on dry ground are punished
        vp += res["val_dry"]["probs"]
        vy += res["val_dry"]["c"].water
    thr, val_iou = (a.threshold, None) if a.threshold else best_threshold(vp, vy)
    # baseline: one global VV threshold tuned on val
    best_vv = (-18.0, -1)
    for t in np.arange(-24, -9.9, 0.5):
        cf = Confusion()
        for vv, y in zip(v["vv"], v["c"].water):
            cf.add(vv < t, y)
        if cf.result()["iou"] > best_vv[1]:
            best_vv = (float(t), cf.result()["iou"])
    print(f"U-Net probability threshold (tuned on val): {thr}   VV baseline threshold: {best_vv[0]} dB")

    metrics = dict(threshold=thr, vv_threshold_db=best_vv[0], tta=not a.no_tta, checkpoints=a.ckpt,
                   channels=cfg["channels"], target_trained=cfg["target"], splits={})
    for name, r in res.items():
        groups = {name: list(range(len(r["rows"])))}
        if name == "test":
            dadu = [i for i, rw in enumerate(r["rows"]) if DADU_BBOX[0] <= float(rw["lon"]) <= DADU_BBOX[2]
                    and DADU_BBOX[1] <= float(rw["lat"]) <= DADU_BBOX[3]]
            if dadu:
                groups["test_dadu_district"] = dadu
        for gname, idx in groups.items():
            cw, cf_, bw, bo, cfu = Confusion(), Confusion(), Confusion(), Confusion(), Confusion()
            det = Confusion()          # chip level: "is there flood in this chip?" (>= 1% of valid pixels)
            for i in idx:
                p = r["probs"][i] >= thr
                yw, yf, perm = r["c"].water[i], r["c"].flood[i], r["c"].perm[i]
                fb = r["vv"][i] < best_vv[0]
                cw.add(p, yw)
                cf_.add(p & (perm != 1), yf)
                bw.add(fb, yw)
                cfu.add(p & fb, yw)
                v = yf != 255
                truth = (yf == 1).sum() >= 0.01 * max(v.sum(), 1)
                pred = (p & fb & (perm != 1) & v).sum() >= 0.01 * max(v.sum(), 1)
                det.add(np.array([pred]), np.array([1 if truth else 0]))
                bo.add(r["vv"][i] < otsu(r["vv"][i][yw != 255]), yw)
                if gname == name:
                    one = Confusion()
                    one.add(p, yw)
                    rw = r["rows"][i]
                    per_chip.append(dict(chip_id=rw["chip_id"], split=name, date=rw["date"], lon=rw["lon"], lat=rw["lat"],
                                         water_frac=rw["water_frac"], unet_iou=one.result()["iou"],
                                         unet_f1=one.result()["f1"]))
            metrics["splits"][gname] = dict(n_chips=len(idx), unet_water=cw.result(), unet_flood=cf_.result(),
                                            fused_confirmed=cfu.result(), chip_detection=det.result(),
                                            baseline_vv_threshold=bw.result(), baseline_otsu=bo.result())
            m = metrics["splits"][gname]
            print(f"  {gname:24s} U-Net water IoU {m['unet_water']['iou']:.3f} F1 {m['unet_water']['f1']:.3f} | "
                  f"flood IoU {m['unet_flood']['iou']:.3f} | fused P {m['fused_confirmed']['precision']:.3f} | "
                  f"chip acc {m['chip_detection']['accuracy']:.3f} | VV-thr IoU {m['baseline_vv_threshold']['iou']:.3f} | "
                  f"Otsu IoU {m['baseline_otsu']['iou']:.3f}")
    if "val_dry" in res:              # combined validation score used for model selection (flood season + dry season)
        cc = Confusion()
        for r in (res["val"], res["val_dry"]):
            for p_, y_ in zip(r["probs"], r["c"].water):
                cc.add(p_ >= thr, y_)
        metrics["splits"]["val_all"] = dict(n_chips=len(res["val"]["rows"]) + len(res["val_dry"]["rows"]), unet_water=cc.result())
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    with open(out / "per_chip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_chip[0].keys()))
        w.writeheader()
        w.writerows(per_chip)
    _summary(metrics, out / "summary.md")
    _figure(res, thr, out / "predictions_test.png")
    print(f"saved {out}")


def _summary(m, path):
    L = [f"# Evaluation  (threshold {m['threshold']}, TTA {m['tta']}, channels {m['channels']})", "",
         "| Split | Chips | U-Net water IoU | F1 | Precision | Recall | Pixel accuracy | U-Net flood IoU | Fused (confirmed) IoU / precision | Chip flood yes/no accuracy | VV threshold IoU | Otsu IoU |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in m["splits"].items():
        u = s["unet_water"]
        L.append(f"| {k} | {s['n_chips']} | {u['iou']:.3f} | {u['f1']:.3f} | {u['precision']:.3f} | {u['recall']:.3f} | "
                 f"{u['accuracy']:.3f} | {s['unet_flood']['iou']:.3f} | {s['fused_confirmed']['iou']:.3f} / {s['fused_confirmed']['precision']:.3f} | "
                 f"{s['chip_detection']['accuracy']:.3f} | {s['baseline_vv_threshold']['iou']:.3f} | {s['baseline_otsu']['iou']:.3f} |")
    Path(path).write_text("\n".join(L) + "\n")


def _figure(res, thr, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = res.get("test") or res["val"]
        order = np.argsort([-float(x["water_frac"]) for x in r["rows"]])
        pick = list(order[:: max(1, len(order) // 6)][:6])
        fig, ax = plt.subplots(3, len(pick), figsize=(3.3 * len(pick), 10))
        ax = np.atleast_2d(ax)
        for j, i in enumerate(pick):
            y = r["c"].water[i]
            p = r["probs"][i] >= thr
            err = np.zeros(y.shape + (3,))
            err[(p == 1) & (y == 1)] = (0.1, 0.6, 0.2)
            err[(p == 1) & (y == 0)] = (0.9, 0.1, 0.1)
            err[(p == 0) & (y == 1)] = (0.1, 0.3, 0.9)
            err[y == 255] = (0.6, 0.6, 0.6)
            ax[0, j].imshow(r["vv"][i], cmap="gray", vmin=-22, vmax=-2)
            ax[0, j].set_title(r["rows"][i]["chip_id"], fontsize=9)
            ax[1, j].imshow(r["probs"][i], cmap="Blues", vmin=0, vmax=1)
            ax[1, j].set_title("U-Net water probability", fontsize=9)
            ax[2, j].imshow(err)
            ax[2, j].set_title("green TP, red FP, blue FN", fontsize=9)
        for x in ax.ravel():
            x.axis("off")
        plt.tight_layout()
        plt.savefig(png, dpi=90)
    except Exception as e:
        print("[warn] figure failed:", e)


if __name__ == "__main__":
    main()
