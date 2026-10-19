"""Run every demo case (DEMO_IMAGES/expected.csv) through the same code the app uses and print PASS/FAIL.

  python scripts/acceptance_test.py --model runs/M2/best_model.json
  python scripts/acceptance_test.py --model runs/M1_unet_r34_lee
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
from flood_app import App  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=str(ROOT.parent / "outputs" / "FLOOD_TRAINING_DATA"))
    ap.add_argument("--jrc-dir", default=str(ROOT.parent / "outputs" / "datasets" / "jrc_global_surface_water_pakistan"))
    ap.add_argument("--demo", default=str(ROOT.parent / "outputs" / "DEMO_IMAGES"))
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="case names to skip (e.g. the large Dadu scene on CPU)")
    a = ap.parse_args()
    app = App(a.data, a.jrc_dir, a.demo, a.model)
    print("model:", app.current, "|", app.det().mode, app.det().error or "")
    if a.skip:
        cases = app.demo_cases
        app.demo_cases = lambda: [c for c in cases() if c["case"] not in a.skip]
    head, rows, _ = app.run_demo(not a.no_tta, None)
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[4].split()[-1]:4s}  {r[0]:{w}s}  expected {r[2]:18s} got {r[3]:18s} {r[5]}")
    print(head.replace("#", "").strip())
    sys.exit(0 if all(r[4].endswith("PASS") for r in rows) else 1)


if __name__ == "__main__":
    main()
