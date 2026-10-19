"""Model, loss, metrics, EMA and inference helpers for the flood U-Net."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from resqflood.train.data import IGNORE, pad_to


# ------------------------------------------------------------------ model
def build_unet(encoder="resnet34", in_ch=3, weights="imagenet", attention="scse"):
    import segmentation_models_pytorch as smp
    kw = dict(encoder_name=encoder, in_channels=in_ch, classes=1,
              decoder_attention_type=None if attention in (None, "none") else attention)
    if weights not in (None, "none"):
        try:
            return smp.Unet(encoder_weights=weights, **kw), weights
        except Exception as e:  # no internet / blocked hub -> train from scratch (fallback)
            print(f"[warn] could not load '{weights}' encoder weights ({type(e).__name__}: {e}). "
                  f"Falling back to random init.")
    return smp.Unet(encoder_weights=None, **kw), "none"


# ------------------------------------------------------------------ loss
class BCEDiceLoss(torch.nn.Module):
    """Masked BCE + soft Dice on valid pixels (label 255 ignored)."""

    def __init__(self, bce_w=0.5, dice_w=0.5, pos_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_w, self.dice_w, self.smooth = bce_w, dice_w, smooth
        self.pos_weight = pos_weight

    def forward(self, logits, y):
        logits = logits[:, 0].float()
        m = (y != IGNORE).float()
        t = (y == 1).float()
        pw = torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=pw)
        bce = (bce * m).sum() / m.sum().clamp(min=1)
        p = torch.sigmoid(logits) * m
        inter = (p * t).sum()
        dice = 1 - (2 * inter + self.smooth) / (p.sum() + (t * m).sum() + self.smooth)
        return self.bce_w * bce + self.dice_w * dice


class BCELovaszLoss(torch.nn.Module):
    """Masked BCE + Lovasz hinge (a direct surrogate of IoU, Berman et al. 2018)."""

    def __init__(self, bce_w=0.5, pos_weight=1.0):
        super().__init__()
        self.bce = BCEDiceLoss(bce_w=1.0, dice_w=0.0, pos_weight=pos_weight)
        self.w = bce_w

    def forward(self, logits, y):
        lov = []
        lg = logits[:, 0].float()
        for b in range(lg.shape[0]):
            v = y[b] != IGNORE
            if v.sum() == 0:
                continue
            lov.append(_lovasz_hinge_flat(lg[b][v], (y[b][v] == 1).float()))
        lov = torch.stack(lov).mean() if lov else lg.sum() * 0
        return self.w * self.bce(logits, y) + (1 - self.w) * lov


def _lovasz_hinge_flat(logits, labels):
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt = labels[perm]
    gts = gt.sum()
    inter = gts - gt.cumsum(0)
    union = gts + (1 - gt).cumsum(0)
    jac = 1.0 - inter / union
    if jac.numel() > 1:
        jac[1:] = jac[1:] - jac[:-1]
    return torch.dot(F.relu(errors_sorted), jac)


def make_loss(name="bce_dice", pos_weight=1.0):
    return BCELovaszLoss(pos_weight=pos_weight) if name == "bce_lovasz" else BCEDiceLoss(pos_weight=pos_weight)


# ------------------------------------------------------------------ metrics
class Confusion:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, pred: np.ndarray, y: np.ndarray):
        v = y != IGNORE
        p, t = pred[v].astype(bool), y[v] == 1
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())
        self.tn += int((~p & ~t).sum())

    def result(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        return dict(iou=round(iou, 4), f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4),
                    accuracy=round(acc, 4), tp=tp, fp=fp, fn=fn, tn=tn)


# ------------------------------------------------------------------ EMA
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.updates = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))   # warm-up so early epochs are usable
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ------------------------------------------------------------------ inference
D8 = [(k, f) for k in range(4) for f in (False, True)]


@torch.no_grad()
def predict_prob(models, x: np.ndarray, device, tta=True, amp=True):
    """x: (C,H,W) float -> probability (H,W) float32, averaged over models and (optionally) 8 dihedral views."""
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype="float32"))[None].to(device)
    xt, H, W = pad_to(xt)
    views = D8 if tta else [(0, False)]
    acc = torch.zeros(xt.shape[-2:], device=device)
    n = 0
    for m in models:
        for k, f in views:
            v = torch.rot90(xt, k, (2, 3))
            if f:
                v = torch.flip(v, (3,))
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                o = torch.sigmoid(m(v).float())[0, 0]
            if f:
                o = torch.flip(o, (1,))
            acc += torch.rot90(o, -k, (0, 1))
            n += 1
    return (acc / n)[:H, :W].cpu().numpy()


def best_threshold(probs, ys, grid=np.arange(0.20, 0.81, 0.05)):
    best = (0.5, -1)
    for t in grid:
        c = Confusion()
        for p, y in zip(probs, ys):
            c.add(p >= t, y)
        iou = c.result()["iou"]
        if iou > best[1]:
            best = (round(float(t), 2), iou)
    return best
