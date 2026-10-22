"""Train the ResQFlood flood U-Net on FLOOD_TRAINING_DATA.

Example (Windows, from the ResQFlood folder):
  python scripts/train_unet.py --data ..\\outputs\\FLOOD_TRAINING_DATA --out runs\\unet_r34_lee

Default = the main configuration from docs/TRAINING_PLAN.md:
U-Net + ResNet-34 (ImageNet) + scSE attention, input [VV_lee, VH_lee, VV-VH], target "water",
BCE+Dice, AdamW 3e-4, cosine with 3 warm-up epochs, 50 epochs (full schedule, best epoch kept),
EMA 0.999, AMP, 384 px crops with 0.75-1.33 zoom augmentation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resqflood.train.core import EMA, Confusion, build_unet, make_loss, predict_prob  # noqa: E402
from resqflood.train.data import ChipCache, TrainChips, read_manifest, select_rows, to_single_pol, worker_init  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="FLOOD_TRAINING_DATA folder")
    ap.add_argument("--out", default="runs/unet_r34_lee")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet", help="imagenet | none")
    ap.add_argument("--attention", default="scse", help="scse | none")
    ap.add_argument("--channels", default="lee", choices=["lee", "raw", "all"])
    ap.add_argument("--target", default="water", choices=["water", "flood"])
    ap.add_argument("--include-aux", action="store_true", help="add QC-passed Sen1Floods11 weak chips to training")
    ap.add_argument("--include-dry", action="store_true", help="add dry-season chips (train_dry to training, val_dry to "
                    "validation): teaches the model that dark dry ground / riverbeds are not water")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--loss", default="bce_dice", choices=["bce_dice", "bce_lovasz"])
    ap.add_argument("--scale-aug", type=float, nargs=2, default=[0.75, 1.33], help="random zoom range; 1 1 disables")
    ap.add_argument("--batch", type=int, default=0, help="0 = choose from GPU memory")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--epoch-mult", type=int, default=2, help="random crops per chip per epoch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999, help="0 disables EMA")
    ap.add_argument("--patience", type=int, default=0, help="early stop after N epochs without val gain; 0 = off "
                    "(run the full cosine schedule - the LR must decay before the model settles)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N chips per split")
    ap.add_argument("--resume", default="", help="path to last.pt to continue")
    ap.add_argument("--init", default="", help="start from the weights of this best.pt (fine-tuning)")
    ap.add_argument("--single-pol", action="store_true", help="train/validate on VV-only inputs (VH approximated exactly "
                    "as the detector does for single-polarisation GeoTIFFs and PNG/JPG images)")
    return ap.parse_args()


def auto_batch(device, crop):
    if device.type != "cuda":
        return 2
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    base = 4 if gb < 5 else 8 if gb < 9 else 16 if gb < 17 else 24
    return max(2, int(base * (384 / crop) ** 2))


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not a.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("[warn] no CUDA GPU found - training on CPU will be very slow")
    a.batch = a.batch or auto_batch(device, a.crop)

    rows = read_manifest(a.data)
    tr_rows = select_rows(rows, "train", a.include_aux, include_dry=a.include_dry)
    va_rows = select_rows(rows, "val", include_dry=a.include_dry)
    if a.limit:
        tr_rows, va_rows = tr_rows[:a.limit], va_rows[:max(2, a.limit // 4)]
    print(f"train chips: {len(tr_rows)}  val chips: {len(va_rows)}  batch: {a.batch}  crop: {a.crop}")
    print("loading training chips into memory ...")
    tr_cache = ChipCache(a.data, tr_rows, a.channels)
    print("loading validation chips ...")
    va_cache = ChipCache(a.data, va_rows, a.channels)
    if a.single_pol:
        to_single_pol(tr_cache)
        to_single_pol(va_cache)
        print("single-polarisation mode: VH replaced by VV - mean(VV-VH) in training and validation")

    ds = TrainChips(tr_cache, a.target, a.crop, a.epoch_mult, seed=a.seed, scale=tuple(a.scale_aug))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, drop_last=True,
                                     num_workers=a.workers, pin_memory=device.type == "cuda",
                                     worker_init_fn=worker_init if a.workers else None,
                                     persistent_workers=a.workers > 0)
    in_ch = tr_cache.x[0].shape[0]
    model, used_w = build_unet(a.encoder, in_ch, a.weights, a.attention)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
        print("initialised from", a.init)
    model.to(device)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    loss_fn = make_loss(a.loss, a.pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = len(dl)
    total = a.epochs * steps_per_epoch
    warm = a.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 1e-6 + 0.5 * (a.lr - 1e-6) * (1 + math.cos(math.pi * p))

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, best_ep, step = 1, -1.0, 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema and ck.get("ema"):
            ema.model.load_state_dict(ck["ema"])
        start, best, best_ep = ck["epoch"] + 1, ck["best"], ck["best_epoch"]
        step = (start - 1) * steps_per_epoch
        print(f"resumed from epoch {ck['epoch']} (best val IoU {best:.4f})")

    cfg = vars(a) | dict(encoder_weights_used=used_w, in_channels=in_ch, n_train=len(tr_rows), n_val=len(va_rows),
                         device=str(device), torch=torch.__version__)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    log_f = out / "log.csv"
    if not a.resume:
        with open(log_f, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_iou", "val_f1", "val_precision", "val_recall", "sec"])

    def save(path, epoch):
        torch.save(dict(model=model.state_dict(), ema=ema.model.state_dict() if ema else None, opt=opt.state_dict(),
                        epoch=epoch, best=best, best_epoch=best_ep, config=cfg), path)

    epoch = start - 1
    try:
        for epoch in range(start, a.epochs + 1):
            t0 = time.time()
            model.train()
            losses = []
            for x, y in dl:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if ema:
                    ema.update(model)
                losses.append(loss.item())
                step += 1
            # ---- validation (EMA weights, full 512 chips, no TTA, threshold 0.5)
            em = (ema.model if ema else model).eval()
            conf = Confusion()
            for i in range(len(va_cache.rows)):
                p = predict_prob([em], va_cache.x[i], device, tta=False, amp=amp)
                conf.add(p >= 0.5, va_cache.target(i, a.target))
            r = conf.result()
            sec = time.time() - t0
            with open(log_f, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{lr_at(step - 1):.2e}", f"{np.mean(losses):.4f}", r["iou"], r["f1"],
                                        r["precision"], r["recall"], f"{sec:.0f}"])
            flag = ""
            if r["iou"] > best:
                best, best_ep, flag = r["iou"], epoch, "  * best"
                torch.save(dict(model=em.state_dict(), config=cfg, epoch=epoch, val=r), out / "best.pt")
            print(f"epoch {epoch:3d}/{a.epochs}  loss {np.mean(losses):.4f}  val IoU {r['iou']:.4f}  "
                  f"F1 {r['f1']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}  ({sec:.0f}s){flag}", flush=True)
            save(out / "last.pt", epoch)
            if a.patience and epoch - best_ep >= a.patience:
                print(f"early stop: no val improvement for {a.patience} epochs")
                break
    except KeyboardInterrupt:
        print("interrupted - saving last.pt (continue with --resume)")
        save(out / "last.pt", epoch)
    print(f"best val IoU {best:.4f} at epoch {best_ep}  ->  {out / 'best.pt'}")
    _plot(log_f, out / "curves.png")


def _plot(log_f, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        d = pd.read_csv(log_f)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(d.epoch, d.train_loss)
        ax[0].set_title("training loss (BCE + Dice)")
        ax[1].plot(d.epoch, d.val_iou, label="IoU")
        ax[1].plot(d.epoch, d.val_f1, label="F1")
        ax[1].set_title("validation (EMA model)")
        ax[1].legend()
        for x in ax:
            x.set_xlabel("epoch")
            x.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(png, dpi=110)
    except Exception as e:
        print("[warn] could not plot curves:", e)


if __name__ == "__main__":
    main()
