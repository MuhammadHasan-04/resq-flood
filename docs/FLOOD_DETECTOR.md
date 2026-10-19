# ResQFlood flood detector: how it decides, how it was verified, how to run it

## Decision pipeline (resqflood/detect.py)
1. **Input** (`resqflood/sar_io.py`). Accepted:
   - Sentinel-1 GeoTIFF: VV+VH in one file, or two files named …VV… / …VH…. Values may be linear, dB, dB×100 or 16-bit DN.
   - PNG/JPG radar quick-look, located by typed corners `W S E N`, a centre point `lat, lon`, a `.pgw` world file or `<name>.bbox.txt`.
   - Everything is resampled to the 20 m training grid.
   - Optical / colour / multispectral images and empty images are rejected with an explanation.
2. **Radar preparation**: Lee 7×7 speckle filter in linear units, then dB.
3. **Two water detectors**
   - **U-Net** (primary): water probability, with 8-view test-time augmentation and sliding 512 px windows for large scenes.
   - **Fallback**: a radar-darkness threshold on VV (Lee), tuned on validation data. If the model cannot be loaded or fails, the fallback runs alone and the result says *fallback only*.
4. **Fusion**: a pixel is FLOOD if the U-Net probability ≥ `t_high`, or if it is ≥ `t_low` *and* the fallback agrees. "Confirmed by both" is reported separately.
5. **Normal water is never flood**
   - **Permanent water** (blue) comes from JRC Global Surface Water: water all 12 months, or ≥ 80 % of observations since 1984. This needs a location; tiles anywhere in the world are downloaded automatically. Water in an optional **pre-flood image** of the same area also counts as permanent.
   - **Seasonal / river-corridor water** (cyan): areas that are wet part of the year (JRC occurrence ≥ `seasonal_occ`). It is shown as its own class only because validation showed this costs no flood accuracy.
6. **Verdict**: the detector counts flood outside permanent water, the riverbank buffer and the normally-wet zone (JRC occurrence ≥ `verdict_seasonal_occ`), ignoring specks smaller than `min_blob_px`. It answers "FLOOD DETECTED" if that flood covers ≥ `min_frac` of the image or ≥ `min_km2`.

All thresholds are chosen by `scripts/tune_detector.py` on validation data only:
- validation chips from the flood season;
- dry-season validation chips;
- April 2022 river/lake scenes from validation areas.

They are stored in `<model>/detector.json`, and the model weights are never changed.

## Data used to check the "river only / no flood" case
- 282 dry-season (Jan–May 2022) Sentinel-1 RTC scenes at training, validation and test locations, labelled from 38 years of JRC history:
  - water where JRC shows permanent water;
  - not water where it has never been water;
  - ignored where it is uncertain.
- 30 April 2022 scenes of river and lake locations, used as the river-only scenario test.
- M2 is trained with the dry-season training scenes (`--include-dry`), so it learns that dark dry ground and riverbeds are not water.

## Verification
- `runs/<model>/detector_eval/DETECTOR_REPORT.md`: pixel accuracy (flood IoU) and flood/no-flood accuracy on every held-out set, compared with the U-Net alone, a strict AND of both methods, and the fallback only.
- `scripts/acceptance_test.py`: the demo pack (`outputs/DEMO_IMAGES`, 19 cases) with expected answers. This is the same check as the app's "Demo set" tab.
- `scripts/test_cases.py`: a visual sheet of held-out chips.
- `tests/`: unit tests, including "river only → NO FLOOD", "flood next to river → FLOOD" and "no model → fallback only".

## Known limits (stated honestly)
- **PNG/JPG without location or pre-flood image**: permanent water cannot be separated, so all water counts as possible flood. The app says so.
- **PNG/JPG inputs**: they carry only one polarisation, so VH is approximated. GeoTIFF VV+VH is always more accurate.
- **Other countries / years**: the Sen1Floods11 2017 Punjab/KP hand-labelled set is much harder (flood IoU ≈ 0.18 with M1): small rivers, different processing, hand labels.
- **Flooded vegetation and urban flooding**: these do not look dark to radar, so any radar method under-detects them.
