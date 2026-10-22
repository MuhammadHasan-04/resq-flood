"""Automatic model search: train several U-Net configurations, keep the best on VALIDATION, then train extra
seeds of the winner and build the final ensemble. Everything is selected on val only; test is only reported.

  python scripts/auto_tune.py --data ../outputs/FLOOD_TRAINING_DATA            (6 configs + 2 seeds, ~4-5 h on an RTX 5060)
  python scripts/auto_tune.py --data ../outputs/FLOOD_TRAINING_DATA --quick    (3 configs + 1 seed, ~2-2.5 h)

Runs already finished (best.pt + eval/metrics.json present) are skipped, so the script can be stopped and restarted.
Result: runs/M2/best_model.json (used by the app, predict_scene.py and test_cases.py) and runs/leaderboard.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "M2_C1_r34_lee_dice": ["--include-dry"],
    "M2_C2_r34_lee_lovasz": ["--include-dry", "--loss", "bce_lovasz"],
    "M2_C3_r34_lee_lr1e-4": ["--include-dry", "--lr", "1e-4"],
    "M2_C4_effb4_lee_dice": ["--include-dry", "--encoder", "efficientnet-b4", "--batch", "6"],
    "M2_C5_r34_raw_dice": ["--include-dry", "--channels", "raw"],
    "M2_C6_r34_lee_plus_weak": ["--include-dry", "--include-aux"],
}
QUICK = ["M2_C1_r34_lee_dice", "M2_C2_r34_lee_lovasz", "M2_C6_r34_lee_plus_weak"]
FINAL = "M2"          # final model folder: runs/M2 (M1 is never modified)


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[warn] command failed with code {r.returncode}")
    return r.returncode == 0


def train_eval(data, name, extra, epochs, seed=42):
    d = ROOT / "runs" / name
    if not (d / "best.pt").exists() or not (d / "log.csv").exists() or _epochs_done(d) < epochs:
        resume = ["--resume", str(d / "last.pt")] if (d / "last.pt").exists() and _epochs_done(d) < epochs else []
        if not run([PY, "scripts/train_unet.py", "--data", data, "--out", str(d), "--epochs", str(epochs),
                    "--seed", str(seed)] + extra + resume):
            return None
    if not (d / "eval" / "metrics.json").exists():
        if not run([PY, "scripts/evaluate_unet.py", "--data", data, "--ckpt", str(d / "best.pt")]):
            return None
    return json.load(open(d / "eval" / "metrics.json"))


def _epochs_done(d):
    try:
        return sum(1 for _ in open(d / "log.csv")) - 1
    except FileNotFoundError:
        return 0


def row(name, m):
    s = m["splits"]
    g = lambda sp, k="unet_water", f="iou": s.get(sp, {}).get(k, {}).get(f, "")  # noqa: E731
    sel = g("val_all") if "val_all" in s else g("val")
    return dict(run=name, select_score=sel, val_iou=g("val"), val_dry_false_alarm_px=g("val_dry", f="fp"), val_f1=g("val", f="f1"), test_iou=g("test"), test_f1=g("test", f="f1"),
                dadu_iou=g("test_dadu_district"), external_iou=g("test_external_s1f_hand"),
                test_fused_precision=g("test", "fused_confirmed", "precision"),
                test_chip_accuracy=g("test", "chip_detection", "accuracy"),
                threshold=m["threshold"], vv_threshold_db=m["vv_threshold_db"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../outputs/FLOOD_TRAINING_DATA")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="extra seeds for the winner (default 7 123; quick: 7)")
    a = ap.parse_args()
    if a.seeds is None:
        a.seeds = [7] if a.quick else [7, 123]
    names = QUICK if a.quick else list(CONFIGS)
    board = []
    old = ROOT / "runs" / "M1_unet_r34_lee"
    if (old / "best.pt").exists():      # re-evaluate M1 on the same (extended) validation/test sets; weights untouched
        if not (old / "eval_v2" / "metrics.json").exists():
            run([PY, "scripts/evaluate_unet.py", "--data", a.data, "--ckpt", str(old / "best.pt"), "--out", str(old / "eval_v2")])
        if (old / "eval_v2" / "metrics.json").exists():
            board.append(row("M1_unet_r34_lee (first run)", json.load(open(old / "eval_v2" / "metrics.json"))))
    for n in names:
        m = train_eval(a.data, n, CONFIGS[n], a.epochs)
        if m:
            board.append(row(n, m))
            _write_board(board)
    cand = [b for b in board if not b["run"].endswith("(first run)")]
    if not cand:
        raise SystemExit("no run finished")
    best = max(cand, key=lambda b: b["select_score"])
    print(f"\n=== best configuration on validation: {best['run']} (score {best['select_score']}) ===")
    ckpts = [str(ROOT / "runs" / best["run"] / "best.pt")]
    for sd in a.seeds:
        n = f"{best['run']}_seed{sd}"
        m = train_eval(a.data, n, CONFIGS[best["run"]], a.epochs, seed=sd)
        if m:
            board.append(row(n, m))
            ckpts.append(str(ROOT / "runs" / n / "best.pt"))
            _write_board(board)
    final_dir = ROOT / "runs" / FINAL
    final = dict(best_run=best["run"])
    if len(ckpts) > 1:
        run([PY, "scripts/evaluate_unet.py", "--data", a.data, "--out", str(final_dir / "eval"), "--ckpt"] + ckpts)
        m_ens = json.load(open(final_dir / "eval" / "metrics.json"))
        board.append(row(f"ENSEMBLE {best['run']} x{len(ckpts)}", m_ens))
        _write_board(board)
        m_single = json.load(open(ROOT / "runs" / best["run"] / "eval" / "metrics.json"))
        key = "val_all" if "val_all" in m_ens["splits"] else "val"
        use_ens = m_ens["splits"][key]["unet_water"]["iou"] >= m_single["splits"][key]["unet_water"]["iou"]
        m = m_ens if use_ens else m_single
        if not use_ens:
            ckpts = ckpts[:1]
    else:
        m = json.load(open(ROOT / "runs" / best["run"] / "eval" / "metrics.json"))
    final_dir.mkdir(parents=True, exist_ok=True)
    final.update(checkpoints=[f"../{Path(c).parent.name}/best.pt" for c in ckpts],   # relative to runs/BEST
                 unet_threshold=m["threshold"], vv_threshold_db=m["vv_threshold_db"],
                 norm_stats=str((Path(a.data) / "02_splits" / "norm_stats.json").resolve()),
                 channels=m["channels"], val_iou=m["splits"]["val"]["unet_water"]["iou"],
                 test_iou=m["splits"]["test"]["unet_water"]["iou"])
    json.dump(final, open(final_dir / "best_model.json", "w"), indent=2)
    print("\nFINAL MODEL ->", final_dir / "best_model.json")
    print(json.dumps(final, indent=2))
    # tune + verify the complete detector (fusion, permanent/seasonal water, verdict) for M2, then the demo checks
    run([PY, "scripts/tune_detector.py", "--model", str(final_dir / "best_model.json"), "--data", a.data])
    run([PY, "scripts/test_cases.py", "--model", str(final_dir / "best_model.json"), "--data", a.data,
         "--out", str(ROOT / "runs" / "TEST_CASES_M2")])
    # full verification (every scenario, format and edge case) for M2 and, for comparison, M1
    run([PY, "scripts/full_verification.py", "--model", str(final_dir / "best_model.json"), "--data", a.data])
    if (old / "best.pt").exists() and not (old / "verification" / "VERIFICATION_REPORT.md").exists():
        run([PY, "scripts/full_verification.py", "--model", str(old), "--data", a.data])
    print("\nDONE. Compare:\n  runs/M2/verification/VERIFICATION_REPORT.md\n  runs/M1_unet_r34_lee/verification/VERIFICATION_REPORT.md")


def _write_board(board):
    f = ROOT / "runs" / "leaderboard.csv"
    keys = list(board[0].keys())
    with open(f, "w", newline="") as o:
        w = csv.DictWriter(o, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(board, key=lambda b: -(b["select_score"] or 0)))
    print("\nLEADERBOARD (sorted by validation score: flood-season + dry-season validation IoU):")
    for b in sorted(board, key=lambda b: -(b["select_score"] or 0)):
        print(f"  {b['run']:38s} val {b['val_iou']}  test {b['test_iou']}  dadu {b['dadu_iou']}  external {b['external_iou']}")


if __name__ == "__main__":
    main()
