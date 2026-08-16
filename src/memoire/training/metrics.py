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
from collections.abc import Sequence

import numpy as np
import torch
from scipy import ndimage
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


# -- mAP (chap. 7.4): instance-level average precision from a semantic model --
#
# The U-Net predicts per-pixel classes, not per-instance detections, so
# "instances" here are the connected components of the predicted class mask
# (chap. 6.1 justifies segmentation over detection, but chap. 7.4 still wants
# a detection-style metric to sit next to IoU/Dice). Ground truth stays
# instance-level (DamageSegDataset.instance_targets, distinct from the merged
# training mask). AP uses COCO's 101-point interpolated precision-recall
# integral; mAP is averaged over the 0.50:0.05:0.95 IoU thresholds, with
# mAP@0.5 reported separately since it is the more common single number.
#
# Unlike the confusion matrix / calibration accumulators, this cannot reduce
# a batch to a running sum online: precision at a given confidence depends on
# every prediction ranked above it, so the accumulator keeps raw
# (score, is_tp) pairs and only sorts/integrates in mean_average_precision().

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boolean masks; 0.0 when both are empty (no match, not NaN)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def connected_components(binary_mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components of a boolean mask, as a list of boolean masks."""
    structure = np.ones((3, 3), dtype=int)
    labeled, n = ndimage.label(binary_mask, structure=structure)
    return [labeled == k for k in range(1, n + 1)]


@torch.no_grad()
def predicted_instances(
    logits: Tensor, positive_classes: Sequence[int]
) -> list[tuple[int, float, np.ndarray]]:
    """Pseudo-detections for one image: one per connected component of the
    argmax mask, per class in ``positive_classes`` (background excluded by the
    caller). Score = mean softmax probability of that class over the
    component's pixels. ``logits`` is K×H×W (single image, no batch dim).
    """
    probs = torch.softmax(logits, dim=0)
    pred = probs.argmax(dim=0).cpu().numpy()
    probs_np = probs.cpu().numpy()
    instances: list[tuple[int, float, np.ndarray]] = []
    for c in positive_classes:
        for component in connected_components(pred == c):
            score = float(probs_np[c][component].mean())
            instances.append((c, score, component))
    return instances


def new_map(num_classes: int, iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS) -> dict:
    """Zeroed streaming accumulator for mean average precision."""
    thresholds = tuple(iou_thresholds)
    if not thresholds:
        raise ValueError("iou_thresholds must be non-empty")
    return {
        "num_classes": num_classes,
        "iou_thresholds": thresholds,
        "n_gt": dict.fromkeys(range(num_classes), 0),
        "matches": {c: {t: [] for t in thresholds} for c in range(num_classes)},
    }


def map_update(
    state: dict,
    pred_instances: list[tuple[int, float, np.ndarray]],
    gt_instances: list[tuple[int, np.ndarray]],
) -> dict:
    """Accumulate one image's predictions/ground truth (in place) and return `state`.

    Standard greedy COCO-style matching per class per threshold: predictions
    sorted by score descending, each claims its highest-IoU unclaimed ground
    truth instance of the same class; a claim below ``threshold`` is a false
    positive and leaves that ground truth instance available to a later
    (lower-scoring) prediction.
    """
    for class_id, _mask in gt_instances:
        state["n_gt"][class_id] = state["n_gt"].get(class_id, 0) + 1

    preds_by_class: dict[int, list[tuple[float, np.ndarray]]] = {}
    for class_id, score, mask in pred_instances:
        preds_by_class.setdefault(class_id, []).append((score, mask))
    gt_by_class: dict[int, list[np.ndarray]] = {}
    for class_id, mask in gt_instances:
        gt_by_class.setdefault(class_id, []).append(mask)

    for threshold in state["iou_thresholds"]:
        for class_id, preds in preds_by_class.items():
            gt_masks = gt_by_class.get(class_id, [])
            claimed = [False] * len(gt_masks)
            for score, mask in sorted(preds, key=lambda p: -p[0]):
                best_iou, best_j = 0.0, -1
                for j, gt_mask in enumerate(gt_masks):
                    if claimed[j]:
                        continue
                    iou = mask_iou(mask, gt_mask)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                is_tp = best_iou >= threshold
                if is_tp:
                    claimed[best_j] = True
                state["matches"][class_id][threshold].append((score, is_tp))
    return state


def _average_precision(matches: list[tuple[float, bool]], n_gt: int) -> float:
    """COCO-style 101-point interpolated AP; NaN if the class has no ground truth."""
    if n_gt == 0:
        return float("nan")
    if not matches:
        return 0.0
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for _score, is_tp in sorted(matches, key=lambda m: -m[0]):
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)
    precisions_arr = np.array(precisions)
    recalls_arr = np.array(recalls)
    total = 0.0
    for level in range(101):
        r = level / 100.0
        at_or_above = precisions_arr[recalls_arr >= r]
        total += float(at_or_above.max()) if at_or_above.size else 0.0
    return total / 101.0


def mean_average_precision(state: dict) -> dict:
    """Finalize an mAP accumulator: per-class AP, mAP per threshold, and the
    two headline numbers (``map_50``, ``map`` = mean over all thresholds).
    Class 0 (background) is never a detection target and is excluded.
    """
    num_classes = state["num_classes"]
    thresholds = state["iou_thresholds"]
    ap_per_class: dict[int, dict[float, float]] = {
        c: {
            t: _average_precision(state["matches"][c][t], state["n_gt"].get(c, 0))
            for t in thresholds
        }
        for c in range(1, num_classes)
    }

    def _mean(values: list[float]) -> float:
        finite = [v for v in values if not math.isnan(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    map_per_threshold = {
        t: _mean([ap_per_class[c][t] for c in range(1, num_classes)]) for t in thresholds
    }
    return {
        "ap_per_class": ap_per_class,
        "map_per_threshold": map_per_threshold,
        "map_50": map_per_threshold.get(0.5, float("nan")),
        "map": _mean(list(map_per_threshold.values())),
    }
