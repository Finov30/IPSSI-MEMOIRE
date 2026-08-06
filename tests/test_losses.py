"""Tests for DiceCELoss with hand-computed reference values."""

import math

import pytest
import torch

from memoire.training.losses import DiceCELoss


def _binary_logits_from_mask(mask: torch.Tensor, magnitude: float = 20.0) -> torch.Tensor:
    """B×2×H×W logits strongly voting for the classes in `mask` (B×H×W)."""
    onehot = torch.nn.functional.one_hot(mask, 2).permute(0, 3, 1, 2).float()
    return (2.0 * onehot - 1.0) * magnitude


class TestForwardValues:
    def test_perfect_prediction_near_zero(self):
        target = torch.tensor([[[0, 1], [1, 1]]])
        logits = _binary_logits_from_mask(target)
        loss = DiceCELoss()(logits, target)
        # Saturated softmax: CE ~ 0 and soft-Dice ~ 1 on both classes.
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_uniform_logits_2x2_hand_computed(self):
        # Uniform logits: softmax = [0.5, 0.5] at every pixel.
        # Target 1×1×2 = [0, 1].
        # CE = -log(0.5) = ln 2.
        # Class 0: inter = 0.5, sum(p0) = 1, sum(t0) = 1 -> dice = 1/2.
        # Class 1: identical -> dice = 1/2. Dice loss = 1 - 1/2 = 1/2.
        logits = torch.zeros(1, 2, 1, 2)
        target = torch.tensor([[[0, 1]]])
        loss = DiceCELoss(ce_weight=1.0, dice_weight=1.0)(logits, target)
        assert loss.item() == pytest.approx(math.log(2.0) + 0.5, abs=1e-5)

    def test_nonuniform_2x2_hand_computed(self):
        # logits: class0 = 0, class1 = ln 3 everywhere -> p1 = 3/4, p0 = 1/4.
        # Target 2×2 = [[1, 1], [1, 0]].
        # CE = -(3·ln(3/4) + ln(1/4)) / 4.
        # Class 1: sum(t1) = 3, sum(p1) = 3, inter = 3·(3/4) -> dice = 4.5/6 = 0.75.
        # Class 0: sum(t0) = 1, sum(p0) = 1, inter = 1/4     -> dice = 0.5/2 = 0.25.
        # Dice loss = 1 - (0.75 + 0.25)/2 = 0.5.
        logits = torch.stack(
            [torch.zeros(2, 2), torch.full((2, 2), math.log(3.0))]
        ).unsqueeze(0)
        target = torch.tensor([[[1, 1], [1, 0]]])
        expected_ce = -(3.0 * math.log(0.75) + math.log(0.25)) / 4.0
        loss = DiceCELoss()(logits, target)
        assert loss.item() == pytest.approx(expected_ce + 0.5, abs=1e-5)

    def test_weights_are_applied(self):
        logits = torch.zeros(1, 2, 1, 2)
        target = torch.tensor([[[0, 1]]])
        loss = DiceCELoss(ce_weight=2.0, dice_weight=0.5)(logits, target)
        assert loss.item() == pytest.approx(2.0 * math.log(2.0) + 0.5 * 0.5, abs=1e-5)


class TestEmptyTarget:
    def test_all_background_target_is_finite(self):
        # No positive pixel at all: only the background class is present, the
        # Dice mean must skip the absent damage class instead of producing NaN.
        torch.manual_seed(0)
        logits = torch.randn(2, 2, 8, 8, requires_grad=True)
        target = torch.zeros(2, 8, 8, dtype=torch.int64)
        loss = DiceCELoss()(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_perfect_on_all_background_near_zero(self):
        target = torch.zeros(1, 4, 4, dtype=torch.int64)
        logits = _binary_logits_from_mask(target)
        loss = DiceCELoss()(logits, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestOptimization:
    def test_loss_decreases_on_separable_toy_problem(self):
        # Free logits on a fixed mask: a few SGD steps must strictly reduce
        # the loss (plumbing check that gradients flow through both terms).
        torch.manual_seed(0)
        target = torch.zeros(1, 4, 4, dtype=torch.int64)
        target[:, :, 2:] = 1  # right half is damage: trivially separable
        logits = torch.nn.Parameter(0.01 * torch.randn(1, 2, 4, 4))
        criterion = DiceCELoss()
        optimizer = torch.optim.SGD([logits], lr=1.0)
        initial = criterion(logits, target).item()
        for _ in range(25):
            optimizer.zero_grad()
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        final = criterion(logits, target).item()
        assert final < initial
        assert final < 0.5 * initial


class TestValidation:
    def test_rejects_bad_logits_rank(self):
        with pytest.raises(ValueError):
            DiceCELoss()(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4, dtype=torch.int64))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            DiceCELoss()(torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 5, dtype=torch.int64))
