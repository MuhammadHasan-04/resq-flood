# Data

Nothing in this folder is committed except this file. All datasets live in the shared **DADU_DATA** folder
(`th/outputs/DADU_DATA` on Hasan's laptop). Copy that folder to your machine and set its path in `config/local.yaml`.

Main inputs used by the code (paths relative to DADU_DATA):

| Layer | Path |
|---|---|
| Dadu boundary / pilot sector | `01_Boundary/` |
| Sentinel-1 SAR (30 Aug 2022) | `02_Sentinel1_SAR/a_Dadu_2022-08-30_RTC_VV_VH/` |
| Matched SAR for other dates (created by `scripts/export_sar.py`) | `02_Sentinel1_SAR/matched_scenes/` |
| TU Wien flood masks, 10 dates | `03_TU_Wien_Flood_Masks/a_daily_masks_clipped_to_Dadu/` |
| OSM roads and other layers | `04_OSM_Network/geojson/` |
| Copernicus DEM, slope | `05_DEM_Copernicus_GLO30/clipped/` |
| JRC permanent water | `06_JRC_Surface_Water/` |

The full inventory is in `DADU_DATA/00_README_DADU_DATA.md`.
