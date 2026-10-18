"""Check that the training environment works: Python packages, GPU, data folder, one forward/backward pass.

  python scripts/check_env.py --data ..\\outputs\\FLOOD_TRAINING_DATA
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    ok = True
    import numpy, rasterio, torch  # noqa: E401
    import segmentation_models_pytorch as smp
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | smp {smp.__version__} | rasterio {rasterio.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU OK: {p.name}, {p.total_memory / 1e9:.1f} GB, CUDA {torch.version.cuda}")
    else:
        ok = False
        print("!! No CUDA GPU visible to PyTorch. Install the CUDA build of torch (see docs/TRAINING_PLAN.md, step 1).")
    d = Path(a.data)
    for f in ["02_splits/manifest.csv", "02_splits/norm_stats.json", "01_chips/tuwien_pakistan_2022/sar/TW0000.tif"]:
        e = (d / f).exists()
        ok &= e
        print(("OK  " if e else "!!  missing ") + str(d / f))
    from resqflood.train.core import BCEDiceLoss, build_unet
    from resqflood.train.data import ChipCache, TrainChips, read_manifest, select_rows
    rows = select_rows(read_manifest(d), "train")[:4]
    c = ChipCache(d, rows, "lee", verbose=False, use_disk_cache=False)
    ds = TrainChips(c, crop=384)
    x, y = ds[0]
    print(f"sample chip OK: x {tuple(x.shape)} mean {x.mean():.2f}, labels {sorted(set(y.unique().tolist()))}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, w = build_unet("resnet34", 3, "imagenet")
    print(f"U-Net ResNet-34 built, encoder weights: {w}")
    m.to(dev)
    xb = torch.stack([ds[i][0] for i in range(2)]).to(dev)
    yb = torch.stack([ds[i][1] for i in range(2)]).to(dev)
    t = time.time()
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        out = m(xb)
    loss = BCEDiceLoss()(out, yb)
    loss.backward()
    print(f"forward+backward OK (loss {loss.item():.3f}, {time.time() - t:.2f}s)")
    print("\nREADY TO TRAIN" if ok else "\nFix the items marked !! first")


if __name__ == "__main__":
    main()
