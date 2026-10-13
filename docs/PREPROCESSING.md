# SAR flood training-data preprocessing

The full report with figures is in `FLOOD_TRAINING_DATA/03_report/Flood_Preprocessing_Report.html`.

1. `scripts/sample_tuwien_chips.py`: samples 512-px blocks (≥70% valid) from every TU Wien FLOOD-HM-MASKED mask, stratified by flood fraction → `chip_candidates.csv` (542 chips).
2. The Sentinel-1 RTC chips (VV + VH linear gamma0, 0.00018° grid, the same acquisition as the mask) are downloaded from the Planetary Computer data API.
3. `scripts/build_training_data.py` (`resqflood/sar/training_data.py`) applies these steps:
   - Lee 7×7 on linear intensity (ENL 4.4);
   - conversion to dB;
   - nearest-neighbour reprojection of the TU Wien label;
   - JRC permanent water (seasonality = 12 or occurrence ≥ 80);
   - labels: water, flood (permanent → ignore), permanent_water, valid;
   - per-chip QC.
4. `scripts/make_splits.py`:
   - split by 1° cells, with Dadu/Johi as test and a spatial buffer;
   - Sen1Floods11 hand chips as the external test;
   - QC flags;
   - normalisation statistics from the training split only.
5. `resqflood/sar/dataset.py`: `FloodChips` clips, z-scores and returns the tensors (ignore_index = 255).
