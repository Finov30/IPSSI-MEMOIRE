"""Tests for the streaming confusion matrix and IoU/Dice metrics."""

import math

import pytest
import torch

from memoire.training.metrics import (
    confusion_update,
    dice_per_class,
    iou_per_class,
    mean_iou,
    new_confusion,
)


def _binary_logits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(mask, 2).permute(0, 3, 1, 2).float()
    return 2.0 * onehot - 1.0


class TestPerfectAndInverse:
    def test_perfect_prediction_iou_one_everywhere(self):
        target = torch.tensor([[[0, 1], [2, 1]]])
        conf = new_confusion(3)
        confusion_update(conf, target.clone(), target)
        ious = iou_per_class(conf)
        dices = dice_per_class(conf)
        assert ious == {0: 1.0, 1: 1.0, 2: 1.0}
        assert dices == {0: 1.0, 1: 1.0, 2: 1.0}
        assert mean_iou(conf) == pytest.approx(1.0)

    def test_inverse_prediction_iou_zero(self):
        target = torch.tensor([[[0, 0], [1, 1]]])
        pred = 1 - target
        conf = new_confusion(2)
        confusion_update(conf, pred, target)
        ious = iou_per_class(conf)
        assert ious[0] == 0.0
        assert ious[1] == 0.0
        assert mean_iou(conf) == pytest.approx(0.0)


class TestHandComputed2x2:
    # target = [[0, 1], [1, 1]], pred = [[0, 1], [0, 1]]
    # conf (rows = target, cols = pred): [[1, 0], [1, 2]]
    # class 0: TP=1, FP=1, FN=0 -> IoU = 1/2,  Dice = 2/3
    # class 1: TP=2, FP=0, FN=1 -> IoU = 2/3,  Dice = 4/5
    # mean IoU = (1/2 + 2/3) / 2 = 7/12

    def _conf(self):
        target = torch.tensor([[[0, 1], [1, 1]]])
        pred = torch.tensor([[[0, 1], [0, 1]]])
        conf = new_confusion(2)
        return confusion_update(conf, pred, target)

    def test_confusion_counts(self):
        assert self._conf().tolist() == [[1, 0], [1, 2]]

    def test_iou_values(self):
        ious = iou_per_class(self._conf())
        assert ious[0] == pytest.approx(1.0 / 2.0)
        assert ious[1] == pytest.approx(2.0 / 3.0)

    def test_dice_values(self):
        dices = dice_per_class(self._conf())
        assert dices[0] == pytest.approx(2.0 / 3.0)
        assert dices[1] == pytest.approx(4.0 / 5.0)

    def test_mean_iou(self):
        assert mean_iou(self._conf()) == pytest.approx(7.0 / 12.0)


class TestAbsentClass:
    def test_absent_class_is_nan_and_excluded_from_mean(self):
        # K=3 but class 2 never appears in target nor prediction.
        # conf = [[10, 0, 0], [2, 8, 0], [0, 0, 0]]
        # IoU0 = 10/12, IoU1 = 8/10, IoU2 = NaN (empty union)
        # mean = (10/12 + 8/10) / 2 — NOT divided by 3, NaN excluded not zeroed.
        conf = torch.tensor([[10, 0, 0], [2, 8, 0], [0, 0, 0]], dtype=torch.int64)
        ious = iou_per_class(conf)
        assert ious[0] == pytest.approx(10.0 / 12.0)
        assert ious[1] == pytest.approx(8.0 / 10.0)
        assert math.isnan(ious[2])
        assert math.isnan(dice_per_class(conf)[2])
        assert mean_iou(conf) == pytest.approx((10.0 / 12.0 + 8.0 / 10.0) / 2.0)

    def test_all_absent_gives_nan_mean(self):
        assert math.isnan(mean_iou(new_confusion(3)))


class TestStreaming:
    def test_two_updates_equal_one(self):
        torch.manual_seed(0)
        t1 = torch.randint(0, 3, (2, 8, 8))
        p1 = torch.randint(0, 3, (2, 8, 8))
        t2 = torch.randint(0, 3, (1, 8, 8))
        p2 = torch.randint(0, 3, (1, 8, 8))
        streamed = new_confusion(3)
        confusion_update(streamed, p1, t1)
        confusion_update(streamed, p2, t2)
        combined = new_confusion(3)
        confusion_update(combined, torch.cat([p1, p2]), torch.cat([t1, t2]))
        assert torch.equal(streamed, combined)
        assert int(streamed.sum()) == 3 * 8 * 8

    def test_update_returns_same_tensor_in_place(self):
        conf = new_confusion(2)
        out = confusion_update(conf, torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
        assert out is conf
        assert int(conf.sum()) == 2


class TestInputForms:
    def test_logits_input_matches_argmax_pred(self):
        torch.manual_seed(1)
        target = torch.randint(0, 2, (2, 4, 4))
        logits = torch.randn(2, 2, 4, 4)
        from_logits = confusion_update(new_confusion(2), logits, target)
        from_pred = confusion_update(new_confusion(2), logits.argmax(dim=1), target)
        assert torch.equal(from_logits, from_pred)

    def test_unbatched_logits(self):
        target = torch.tensor([[0, 1], [1, 1]])
        logits = _binary_logits_from_mask(target.unsqueeze(0)).squeeze(0)  # K×H×W
        conf = confusion_update(new_confusion(2), logits, target)
        assert torch.equal(conf.diagonal(), torch.tensor([1, 3]))

    def test_rejects_out_of_range_class(self):
        with pytest.raises(ValueError):
            confusion_update(new_confusion(2), torch.tensor([[0, 2]]), torch.tensor([[0, 1]]))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            confusion_update(new_confusion(2), torch.tensor([[0, 1, 0]]), torch.tensor([[0, 1]]))
