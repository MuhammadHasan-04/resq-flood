# ResQFlood

SAR flood detection and flood-aware relief routing for the **Johi–Dadu pilot sector**, Dadu district, Sindh.
Final Year Project (F26-163) · Hasan Altaf, Gulsher Khan, Haleema Naeem · Supervisor: Bushra Mehmood

## What is in this repo

| Module | Folder | Owner | Phase 1 (Mid FYP-1) deliverable |
|---|---|---|---|
| SAR flood detection | `resqflood/sar/` | Hasan | Matched Sentinel-1 scenes for every TU Wien date, aligned and tiled; Otsu baseline per date |
| Flood-mask ingestion | `resqflood/floodmask/` | Gulsher | Re-score all 5,248 roads for any dated mask (or U-Net output) with one command |
| Road routing | `resqflood/routing/` | Gulsher | A\* with flood-aware costs and 3 alternative routes |
| Boat routing | `resqflood/boat/` | Haleema | Cost grid + least-cost boat path for any date |
| Crowd reports | `resqflood/reports/` | Haleema | Report model, snapping, duplicate/conflict checks, credibility score |
| API | `resqflood/api/` | all | FastAPI service tying the modules together |
| Frontend | `frontend/` | Haleema | Map UI (to be built) |

Data is **not** stored in git. All code reads the `DADU_DATA` folder (see `data/README.md`).

## Setup (Windows)

```powershell
git clone <repo-url> ResQFlood
cd ResQFlood
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Point the code at your data: create `config/local.yaml` (it is git-ignored, so each member can use their own path):

```yaml
paths:
  dadu_data: C:/Users/<you>/Documents/Codex/2026-08-24/th/outputs/DADU_DATA
```

Then check everything is found:

```powershell
python scripts/check_data.py
```

## Run the pipeline

| Step | Command | Output (in `runs/`) |
|---|---|---|
| Score roads for a flood date | `python scripts/ingest_flood.py --date 20220908_A042` | `roads_scored_<date>.geojson`, summary JSON |
| Road routes Johi → Dadu | `python scripts/route_demo.py --k 3` | `routes_<date>.geojson` |
| Boat path | `python scripts/boat_demo.py` | `boat_<date>.geojson`, `costgrid_<date>.tif` |
| Download SAR for a date | `python scripts/export_sar.py --date 20220908_A042` | GeoTIFFs in `DADU_DATA/02_Sentinel1_SAR/matched_scenes/` |
| Align SAR, baselines, tiles | `python scripts/build_tiles.py --date 20220908_A042` | `tiles/<split>/*.npz`, `baseline_<date>.json` |
| API | `uvicorn resqflood.api.main:app --reload` then open http://127.0.0.1:8000/docs | |
| Tests | `pytest -q` | |

### Reference results (30 Aug 2022 mask, reproduced by this code)

| Check | Result |
|---|---|
| Roads scored | 5,248 roads, 4,665 km; 922.6 km flood-exposed |
| Johi → Dadu, shortest by distance | 18.64 km, 3.02 km flooded |
| Johi → Dadu, flood-aware (A\*) | 21.18 km, 1.50 km flooded; alternatives 26.31 km / 19.82 km |
| Boat path Dadu → Jhandiani | 23.45 km, 342 cells, 1 dry cell, 9 road crossings |
| SAR Otsu baseline vs TU Wien (same pass) | IoU 0.884, F1 0.939 (threshold −14.08 dB) |
| Training tiles from this date | 183 tiles of 256 × 256 |

## Repository layout

```
config/          config.yaml (shared settings); local.yaml (your paths, not committed)
resqflood/       Python package: sar, floodmask, routing, boat, reports, api
scripts/         command-line entry points
tests/           unit tests on synthetic data (no DADU_DATA needed)
eval/            fixed route test cases (route_cases.csv)
docs/            pipeline, Phase 1 plan, data inventory
frontend/        web map UI (placeholder)
data/            only README; real data lives in DADU_DATA
runs/            generated outputs (git-ignored)
```

## Working together

See `CONTRIBUTING.md`: one branch per task, pull request into `main`, tests must pass.
