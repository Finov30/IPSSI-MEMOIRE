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


# -- calibration (chap. 7.4/8.5): does the model's confidence match its actual
# accuracy? Reliability under from-scratch training, not (only) segmentation
# quality — a confident-but-wrong model is worse in an opposability context
# than one that is uncertain when it should be. Streaming, same idiom as the
# confusion matrix above: an accumulator dict updated per batch.


def new_calibration(n_bins: int = 15, device: torch.device | str = "cpu") -> dict:
    """Zeroed streaming accumulator for ECE (Guo et al. 2017) and Brier score."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    return {
        "n_bins": n_bins,
        "bin_confidence_sum": torch.zeros(n_bins, dtype=torch.float64, device=device),
        "bin_correct_sum": torch.zeros(n_bins, dtype=torch.float64, device=device),
        "bin_count": torch.zeros(n_bins, dtype=torch.int64, device=device),
        "brier_sum": 0.0,
        "brier_count": 0,
    }


@torch.no_grad()
def calibration_update(state: dict, logits: Tensor, target: Tensor) -> dict:
    """Accumulate one batch into `state` (in place, per-pixel) and return it.

    `logits` is B×K×H×W (or K×H×W). Confidence = max softmax probability;
    correct = argmax matches `target`. Brier is the standard multiclass form
    (sum over classes of (prob - one_hot)^2), averaged per pixel — reduces to
    the familiar binary Brier score when K=2.
    """
    probs = torch.softmax(logits.to(torch.float64), dim=-3)
    confidence, prediction = probs.max(dim=-3)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match target "
            f"{tuple(target.shape)}"
        )
    target = target.to(device=probs.device, dtype=torch.int64)
    correct = (prediction == target).to(torch.float64).reshape(-1)
    confidence = confidence.reshape(-1)
    if confidence.numel() == 0:
        return state

    n_bins = state["n_bins"]
    bin_idx = torch.clamp((confidence * n_bins).long(), max=n_bins - 1)
    device = state["bin_count"].device
    bin_idx = bin_idx.to(device)
    state["bin_confidence_sum"] += torch.bincount(
        bin_idx, weights=confidence.to(device), minlength=n_bins
    )
    state["bin_correct_sum"] += torch.bincount(
        bin_idx, weights=correct.to(device), minlength=n_bins
    )
    state["bin_count"] += torch.bincount(bin_idx, minlength=n_bins)

    num_classes = probs.shape[-3]
    onehot = torch.nn.functional.one_hot(target, num_classes).movedim(-1, -3).to(probs.dtype)
    sq_error = (probs - onehot).pow(2).sum(dim=-3)
    state["brier_sum"] += float(sq_error.sum())
    state["brier_count"] += int(sq_error.numel())
    return state


def expected_calibration_error(state: dict) -> float:
    """ECE: bin-population-weighted mean |accuracy - confidence| per bin; NaN if empty."""
    bin_count = state["bin_count"]
    total = int(bin_count.sum())
    if total == 0:
        return float("nan")
    count = bin_count.to(torch.float64)
    safe_count = count.clamp(min=1)
    bin_acc = state["bin_correct_sum"] / safe_count
    bin_conf = state["bin_confidence_sum"] / safe_count
    weights = count / total
    return float((weights * (bin_acc - bin_conf).abs()).sum())


def brier_score(state: dict) -> float:
    """Mean multiclass Brier score over every accumulated pixel; NaN if empty."""
    if state["brier_count"] == 0:
        return float("nan")
    return state["brier_sum"] / state["brier_count"]
