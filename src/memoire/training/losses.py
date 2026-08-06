"""Segmentation losses: weighted sum of cross-entropy and multiclass soft-Dice.

Convention (docs/CONVENTIONS-TRAINING.md): logits are B×K×H×W, targets are
B×H×W int64 class indices. The Dice term is averaged over the classes actually
present in the target, so a batch with no positive pixel (background only)
still yields a finite, differentiable loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DiceCELoss(nn.Module):
    """total = ce_weight * CrossEntropy + dice_weight * (1 - mean soft-Dice).

    Soft-Dice per class c (softmax probabilities p, one-hot targets t):

        dice_c = (2 * sum(p_c * t_c) + eps) / (sum(p_c) + sum(t_c) + eps)

    The mean runs only over classes present in the target (sum(t_c) > 0);
    absent classes are excluded rather than counted as failures, which keeps
    the loss NaN-free on empty targets.
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.dim() != 4:
            raise ValueError(f"expected logits B×K×H×W, got shape {tuple(logits.shape)}")
        if target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match logits "
                f"{tuple(logits.shape)}"
            )
        ce = F.cross_entropy(logits, target)
        dice = self._soft_dice_loss(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice

    def _soft_dice_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
        dims = (0, 2, 3)  # reduce over batch and spatial dims, keep classes
        intersection = (probs * onehot).sum(dims)
        target_sum = onehot.sum(dims)
        cardinality = probs.sum(dims) + target_sum
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        present = target_sum > 0
        if not bool(present.any()):
            # Degenerate empty target: no class to score, contribute zero while
            # keeping the graph connected to the logits.
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()
