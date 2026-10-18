# ResQFlood - full training programme (PowerShell, run from the ResQFlood folder with the venv active)
#   .\scripts\run_all_training.ps1            -> main model M1 + evaluation
#   .\scripts\run_all_training.ps1 -Ablations -> also A1-A4
#   .\scripts\run_all_training.ps1 -Ensemble  -> also seeds 7 and 123 + 3-model ensemble evaluation
param([switch]$Ablations, [switch]$Ensemble)
$ErrorActionPreference = "Stop"
$DATA = "..\outputs\FLOOD_TRAINING_DATA"

python scripts\check_env.py --data $DATA

# ---- M1: main U-Net (ResNet-34, Lee input, water target)
python scripts\train_unet.py --data $DATA --out runs\M1_unet_r34_lee --seed 42
python scripts\evaluate_unet.py --data $DATA --ckpt runs\M1_unet_r34_lee\best.pt

if ($Ablations) {
  python scripts\train_unet.py --data $DATA --out runs\A1_raw_input   --channels raw
  python scripts\evaluate_unet.py --data $DATA --ckpt runs\A1_raw_input\best.pt
  python scripts\train_unet.py --data $DATA --out runs\A2_plus_weak   --include-aux
  python scripts\evaluate_unet.py --data $DATA --ckpt runs\A2_plus_weak\best.pt
  python scripts\train_unet.py --data $DATA --out runs\A3_flood_target --target flood
  python scripts\evaluate_unet.py --data $DATA --ckpt runs\A3_flood_target\best.pt
  python scripts\train_unet.py --data $DATA --out runs\A4_effb4       --encoder efficientnet-b4
  python scripts\evaluate_unet.py --data $DATA --ckpt runs\A4_effb4\best.pt
}

if ($Ensemble) {
  python scripts\train_unet.py --data $DATA --out runs\M1_seed7   --seed 7
  python scripts\train_unet.py --data $DATA --out runs\M1_seed123 --seed 123
  python scripts\evaluate_unet.py --data $DATA --out runs\ENSEMBLE_eval --ckpt runs\M1_unet_r34_lee\best.pt runs\M1_seed7\best.pt runs\M1_seed123\best.pt
}
