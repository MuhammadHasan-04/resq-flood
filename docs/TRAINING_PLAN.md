# U-Net flood-detection training plan (ResQFlood)

## Data (from FLOOD_TRAINING_DATA, see 03_report)

| Split | Chips | Role |
|---|---|---|
| train | 379 TU Wien 2022 | fitting |
| val | 71 TU Wien 2022 | early stopping, threshold tuning, model choice |
| test | 75 TU Wien 2022 (Dadu/Johi held out) | final score, never used for any decision |
| test_external_s1f_hand | 28 Sen1Floods11 2017, hand labels | independent generalisation check (resampled 10 m to 20 m) |
| train_aux_s1f_weak | 139 that pass QC | optional extra training data (ablation A2) |

Chip size is 512×512 at about 20 m. Training pixels are 17% water. Label 255 is ignored everywhere.

## Main model M1

| Item | Setting |
|---|---|
| Architecture | U-Net (segmentation_models_pytorch), ResNet-34 encoder, ImageNet weights, scSE decoder attention, 1 output logit |
| Input | 3 channels: VV_lee7, VH_lee7, VV−VH (dB). Clipped to VV −23..0 and VH −28..−5, z-scored with the train statistics, clamped to ±6 |
| Target | `water` (flood ∪ permanent). Flood = water − JRC permanent water at inference |
| Loss | 0.5·BCE + 0.5·soft Dice, masked to valid pixels |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4, gradient clip 1.0 |
| LR schedule | linear warm-up for 3 epochs, then cosine to 1e-6 (per step) |
| Epochs | 80 max; early stop when val IoU has not improved for 20 epochs |
| Batch / crop | random 384 px crops, 2 crops per chip per epoch. Batch is chosen from GPU memory: <5 GB → 4, <9 GB → 8, <17 GB → 16 |
| Sampling | chips with >1% water drawn 2× as often; 50% of crops centred on a water pixel |
| Augmentation | random rot90 and flip (8 dihedral views), ±1 dB calibration offset on VV and VH, Gaussian noise σ 0.03 |
| Regularisation | EMA of weights (decay 0.999, with warm-up). Validation and checkpoints use the EMA model |
| Precision | mixed precision (AMP) on CUDA |
| Model selection | best val IoU at threshold 0.5 → `best.pt` |
| Evaluation | 8-view TTA; probability threshold tuned on val (0.20–0.80); IoU, F1, precision and recall per split, plus the Dadu subset; baselines on the same pixels |

A CPU sanity run (60 chips, ResNet-18, no pretraining, 3 epochs) already reached val IoU 0.68, so the pipeline learns.

## Ablations (each run is one changed setting)

| Run | Flag | Question |
|---|---|---|
| A1 | `--channels raw` | Does the Lee filter help the U-Net? |
| A2 | `--include-aux` | Do the weak Sen1Floods11 chips help (especially on the external test)? |
| A3 | `--target flood` | Train on water, or directly on flood? |
| A4 | `--encoder efficientnet-b4` | Bigger and better encoder? |

## Final model
Take the best configuration and train it with 3 seeds (42, 7, 123). The final model is the average of the 3 models' probabilities, with TTA (`evaluate_unet.py --ckpt a b c`).

## Fallbacks
1. **CUDA out of memory** → `--batch 4` or `--crop 256`.
2. **No internet for ImageNet weights** → automatic random init. Add `--epochs 120 --patience 30`.
3. **Under-fitting** (val IoU < 0.6 after 30 epochs) → `--lr 5e-4`, `--pos-weight 2`, or `--encoder resnet18`.
4. **Over-fitting** (train loss falls while val IoU drops) → EMA is already on. Lower `--epochs`, or use A2's extra data.
5. **U-Net not better than the baselines** → `predict_scene.py --method otsu` (Otsu on VV-Lee) or the val-tuned VV threshold. Both are always reported next to the U-Net.
6. **Operational products** → TU Wien / Copernicus GFM flood masks for 2022 dates.
7. **Training stopped / laptop slept** → `--resume runs\NAME\last.pt`.

## Success criteria
- Test water IoU ≥ 0.75 and Dadu IoU ≥ 0.75.
- The U-Net beats the VV-threshold and Otsu baselines on every split.
- The external Sen1Floods11 test is reported honestly (expected to be lower: different year, sensor processing and labels).

## Outputs per run
`runs/NAME/` contains:
- `best.pt`, `last.pt`
- `config.json`
- `log.csv`
- `curves.png`
- `eval/metrics.json`, `eval/summary.md`, `eval/per_chip.csv`
- `eval/predictions_test.png`
