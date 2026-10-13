# Phase 1 (Mid FYP-1) plan

Deliverables on the slide: **export matched Sentinel-1 scenes for Dadu** and **flood-mask ingestion**.

| Week | Hasan (SAR) | Gulsher (roads) | Haleema (reports, boat, UI) |
|---|---|---|---|
| 1 | Set up env; run `export_sar.py` for 2–3 dates | Run `ingest_flood.py` for all 10 dates; compare summaries | Run API locally; read `reports/validate.py` |
| 2 | Export remaining dates; check alignment visually | Verify the 49 flooded bridges on imagery; fix bridge/tunnel connectivity | Move reports from memory to PostgreSQL/PostGIS |
| 3 | `build_tiles.py` for every date; check split balance | Fill `eval/route_cases.csv` with 10–20 sector cases | Boat path for every date; cost-weight sensitivity |
| 4 | Otsu + fixed baselines for all dates (table) | Route results for all cases × dates (table) | Frontend: flood mask, roads by state, routes |
| 5 | Prepare U-Net data loader (PyTorch) | Hook routing to reports (`verified_statuses`) | Report form + map in frontend |
| 6 | Demo + report section | Demo + report section | End-to-end demo, rehearse |

Done = every item reproducible from `scripts/` and listed in the Mid FYP-1 report.
