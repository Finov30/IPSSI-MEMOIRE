"""Tests for the streaming confusion matrix and IoU/Dice metrics."""

import math

import numpy as np
import pytest
import torch

from memoire.training.metrics import (
    brier_score,
    calibration_update,
    confusion_update,
    connected_components,
    dice_per_class,
    expected_calibration_error,
    iou_per_class,
    map_update,
    mask_iou,
    mean_average_precision,
    mean_iou,
    new_calibration,
    new_confusion,
    new_map,
    predicted_instances,
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


class TestCalibration:
    # 4 single-pixel "images", K=2, logits chosen so softmax gives exact
    # round numbers: softmax([0, ln(3)]) = [0.25, 0.75].
    #   pixel1: target=1, logits=[0, ln3] -> probs=[.25,.75], pred=1, correct
    #   pixel2: target=0, logits=[0, ln3] -> probs=[.25,.75], pred=1, wrong
    #   pixel3: target=1, logits=[ln3, 0] -> probs=[.75,.25], pred=0, wrong
    #   pixel4: target=0, logits=[ln3, 0] -> probs=[.75,.25], pred=0, correct
    # All 4 share confidence=0.75 -> same bin (n_bins=10 -> bin 7).
    # accuracy_in_bin = 2/4 = 0.5, confidence_in_bin = 0.75 -> ECE = 0.25.
    # Brier (sum over classes of (prob-onehot)^2), mean over 4 pixels = 0.625.
    _LN3 = math.log(3.0)

    def _logits_and_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.tensor(
            [[0.0, self._LN3], [0.0, self._LN3], [self._LN3, 0.0], [self._LN3, 0.0]]
        ).T.reshape(1, 2, 2, 2)
        target = torch.tensor([[1, 0], [1, 0]]).reshape(1, 2, 2)
        return logits, target

    def test_hand_computed_ece_and_brier(self):
        logits, target = self._logits_and_target()
        state = new_calibration(n_bins=10)
        calibration_update(state, logits, target)
        assert expected_calibration_error(state) == pytest.approx(0.25)
        assert brier_score(state) == pytest.approx(0.625)

    def test_perfect_confident_prediction_is_zero_ece_and_brier(self):
        target = torch.tensor([[0, 1], [1, 0]]).reshape(1, 2, 2)
        logits = _binary_logits_from_mask(target) * 1000.0  # ~1.0 confidence, always correct
        state = new_calibration(n_bins=10)
        calibration_update(state, logits, target)
        assert expected_calibration_error(state) == pytest.approx(0.0, abs=1e-6)
        assert brier_score(state) == pytest.approx(0.0, abs=1e-6)

    def test_streaming_two_updates_equal_one(self):
        torch.manual_seed(0)
        logits1 = torch.randn(2, 3, 8, 8)
        target1 = torch.randint(0, 3, (2, 8, 8))
        logits2 = torch.randn(1, 3, 8, 8)
        target2 = torch.randint(0, 3, (1, 8, 8))

        streamed = new_calibration(n_bins=15)
        calibration_update(streamed, logits1, target1)
        calibration_update(streamed, logits2, target2)

        combined = new_calibration(n_bins=15)
        calibration_update(
            combined, torch.cat([logits1, logits2]), torch.cat([target1, target2])
        )

        assert expected_calibration_error(streamed) == pytest.approx(
            expected_calibration_error(combined)
        )
        assert brier_score(streamed) == pytest.approx(brier_score(combined))
        assert streamed["brier_count"] == combined["brier_count"] == 3 * 8 * 8

    def test_empty_accumulator_is_nan(self):
        state = new_calibration()
        assert math.isnan(expected_calibration_error(state))
        assert math.isnan(brier_score(state))

    def test_rejects_non_positive_n_bins(self):
        with pytest.raises(ValueError):
            new_calibration(n_bins=0)

    def test_rejects_shape_mismatch(self):
        state = new_calibration()
        with pytest.raises(ValueError):
            calibration_update(state, torch.randn(1, 2, 3, 3), torch.zeros(1, 2, 2, dtype=torch.int64))


class TestMaskIouAndComponents:
    def test_mask_iou_hand_computed(self):
        a = np.zeros((4, 4), dtype=bool)
        a[:2, :2] = True  # 4 pixels
        b = np.zeros((4, 4), dtype=bool)
        b[:2, :3] = True  # 6 pixels, overlaps a's 4 pixels entirely
        assert mask_iou(a, b) == pytest.approx(4.0 / 6.0)

    def test_mask_iou_empty_union_is_zero(self):
        empty = np.zeros((3, 3), dtype=bool)
        assert mask_iou(empty, empty) == 0.0

    def test_connected_components_splits_disjoint_blobs(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[4:6, 4:6] = True
        components = connected_components(mask)
        assert len(components) == 2
        covered = np.zeros_like(mask)
        for comp in components:
            assert comp.shape == mask.shape
            covered |= comp
        assert np.array_equal(covered, mask)

    def test_connected_components_empty_mask_is_empty_list(self):
        assert connected_components(np.zeros((4, 4), dtype=bool)) == []


class TestPredictedInstances:
    def test_extracts_one_instance_per_component_with_mean_score(self):
        # K=2, 4x4: top-left 2x2 strongly favors class 1, rest favors class 0.
        logits = torch.full((2, 4, 4), -5.0)
        logits[1, :2, :2] = 5.0
        instances = predicted_instances(logits, positive_classes=[1])
        assert len(instances) == 1
        class_id, score, mask = instances[0]
        assert class_id == 1
        assert score == pytest.approx(torch.softmax(logits, dim=0)[1, 0, 0].item(), rel=1e-5)
        assert mask.sum() == 4

    def test_no_instances_when_class_never_predicted(self):
        logits = torch.zeros(2, 3, 3)
        logits[0] += 5.0  # class 0 wins everywhere
        assert predicted_instances(logits, positive_classes=[1]) == []


class TestMeanAveragePrecision:
    def _mask(self, region: tuple[slice, slice]) -> np.ndarray:
        mask = np.zeros((4, 4), dtype=bool)
        mask[region] = True
        return mask

    def test_hand_computed_ap_one_tp_one_fp(self):
        # 1 class, 2 images, single threshold=0.5.
        # Image A: pred exactly matches GT (IoU=1.0, score=0.9) -> TP.
        # Image B: pred and GT disjoint (IoU=0.0, score=0.8) -> FP.
        # n_gt=2 total. Matches sorted by score: [(0.9,TP),(0.8,FP)].
        # precision/recall points: (1.0, 0.5) then (0.5, 0.5).
        # 101-pt interpolation: precision=1.0 for r in [0,0.5] (51 points),
        # 0.0 for r in (0.5,1.0] (50 points) -> AP = 51/101.
        top_left = self._mask((slice(0, 2), slice(0, 2)))
        bottom_right = self._mask((slice(2, 4), slice(2, 4)))
        state = new_map(num_classes=2, iou_thresholds=(0.5,))
        map_update(state, [(1, 0.9, top_left)], [(1, top_left)])
        map_update(state, [(1, 0.8, top_left)], [(1, bottom_right)])
        result = mean_average_precision(state)
        expected_ap = 51.0 / 101.0
        assert result["ap_per_class"][1][0.5] == pytest.approx(expected_ap)
        assert result["map_50"] == pytest.approx(expected_ap)
        assert result["map"] == pytest.approx(expected_ap)

    def test_perfect_predictions_give_map_one_across_default_thresholds(self):
        region = self._mask((slice(0, 2), slice(0, 2)))
        state = new_map(num_classes=2)  # default 0.50:0.05:0.95 thresholds
        map_update(state, [(1, 0.99, region)], [(1, region)])
        result = mean_average_precision(state)
        assert result["map_50"] == pytest.approx(1.0)
        assert result["map"] == pytest.approx(1.0)

    def test_missed_ground_truth_instance_is_never_a_free_tp(self):
        region = self._mask((slice(0, 2), slice(0, 2)))
        state = new_map(num_classes=2, iou_thresholds=(0.5,))
        map_update(state, [], [(1, region)])  # no prediction at all for a real instance
        result = mean_average_precision(state)
        assert result["ap_per_class"][1][0.5] == pytest.approx(0.0)

    def test_background_class_excluded_from_results(self):
        result = mean_average_precision(new_map(num_classes=3, iou_thresholds=(0.5,)))
        assert 0 not in result["ap_per_class"]
        assert set(result["ap_per_class"]) == {1, 2}

    def test_empty_accumulator_is_nan(self):
        result = mean_average_precision(new_map(num_classes=2, iou_thresholds=(0.5,)))
        assert math.isnan(result["map_50"])
        assert math.isnan(result["map"])
        assert math.isnan(result["ap_per_class"][1][0.5])

    def test_rejects_empty_iou_thresholds(self):
        with pytest.raises(ValueError):
            new_map(num_classes=2, iou_thresholds=())
