"""Streaming segmentation metrics backed by a K×K confusion matrix.

Usage:

    conf = new_confusion(num_classes)
    for logits, target in loader:
        confusion_update(conf, logits, target)   # vectorized, no per-pixel loop
    ious = iou_per_class(conf)
    miou = mean_iou(conf)

Row index = ground-truth class, column index = predicted class. Classes absent
from both target and prediction (empty union) get NaN, and mean_iou excludes
them from the average instead of counting them as 0.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def new_confusion(num_classes: int, device: torch.device | str = "cpu") -> Tensor:
    """Return a zeroed K×K int64 confusion matrix."""
    return torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)


@torch.no_grad()
def confusion_update(conf: Tensor, logits_or_pred: Tensor, target: Tensor) -> Tensor:
    """Accumulate a batch into `conf` (in place) and return it.

    `logits_or_pred` is either logits with a class dimension (B×K×H×W, or
    K×H×W for a single sample) — reduced with argmax — or already-decoded
    class labels with the same shape as `target` (B×H×W or H×W).
    """
    num_classes = conf.shape[0]
    if conf.shape != (num_classes, num_classes):
        raise ValueError(f"conf must be square, got {tuple(conf.shape)}")
    pred = logits_or_pred
    if pred.dim() == target.dim() + 1:
        pred = pred.argmax(dim=-3)  # class dimension of B×K×H×W or K×H×W
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(pred.shape)} does not match target "
            f"{tuple(target.shape)}"
        )
    pred = pred.reshape(-1).to(device=conf.device, dtype=torch.int64)
    tgt = target.reshape(-1).to(device=conf.device, dtype=torch.int64)
    if tgt.numel() == 0:
        return conf
    if bool(tgt.max() >= num_classes) or bool(pred.max() >= num_classes):
        raise ValueError(f"class index out of range for K={num_classes}")
    idx = tgt * num_classes + pred
    conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    return conf


def _tp_fp_fn(conf: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    conf = conf.to(torch.float64)
    tp = conf.diagonal()
    fn = conf.sum(dim=1) - tp  # ground-truth pixels missed
    fp = conf.sum(dim=0) - tp  # predicted pixels that are wrong
    return tp, fp, fn


def iou_per_class(conf: Tensor) -> dict[int, float]:
    """IoU = TP / (TP + FP + FN) per class; NaN when the union is empty."""
    tp, fp, fn = _tp_fp_fn(conf)
    union = tp + fp + fn
    iou = tp / union  # 0/0 -> NaN for classes absent from target and prediction
    return {c: float(iou[c]) for c in range(conf.shape[0])}


def dice_per_class(conf: Tensor) -> dict[int, float]:
    """Dice = 2·TP / (2·TP + FP + FN) per class; NaN when the denominator is 0."""
    tp, fp, fn = _tp_fp_fn(conf)
    denom = 2.0 * tp + fp + fn
    dice = 2.0 * tp / denom  # 0/0 -> NaN
    return {c: float(dice[c]) for c in range(conf.shape[0])}


def mean_iou(conf: Tensor) -> float:
    """Mean of per-class IoUs, excluding absent classes (NaN), not zeroing them."""
    values = [v for v in iou_per_class(conf).values() if not math.isnan(v)]
    if not values:
        return float("nan")
    return sum(values) / len(values)
