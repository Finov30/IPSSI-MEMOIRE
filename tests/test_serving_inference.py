"""The torch engine, from a checkpoint written on the spot (chap. 5.4).

No training here: a randomly initialised toy U-Net is saved through the very
function the training loop uses (``save_checkpoint``), then reloaded by
``load_engine``. What is under test is the reload contract — the checkpoint
alone must describe the architecture — and the output frame: mask and boxes in
the ORIGINAL pixels of the photo, never in the network's square canvas.

CPU only and deliberately tiny (base_channels=8, depth=2, 64x64): a serving
test must never contend for the GPU.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from memoire.serving.inference import class_names_for, file_sha256, load_engine
from memoire.serving.preprocess import open_image
from memoire.training.train import build_model, num_classes_for, resolve_config, save_checkpoint

CONFIG = {
    "mode": "binary",
    "model": "unet",
    "input_size": 64,
    "base_channels": 8,
    "depth": 2,
    "gn_groups": 4,
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    config = resolve_config(dict(CONFIG))
    model = build_model(config, num_classes_for(config))
    path = tmp_path_factory.mktemp("serving_ckpt") / "best.pt"
    save_checkpoint(path, model, config, iteration=200, best_val_iou=0.31)
    return path


def photo_bytes(width: int, height: int) -> bytes:
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_engine_rebuilds_the_model_from_the_checkpoint_alone(checkpoint: Path) -> None:
    """No YAML is passed: serving a model described by a file edited after
    training is how a service ends up serving a different network."""
    engine = load_engine(checkpoint, device="cpu")
    assert engine.info.name == "unet"
    assert engine.info.mode == "binary"
    assert engine.info.input_size == 64
    assert engine.info.num_classes == 2
    assert engine.info.class_names == ("background", "damage")
    assert engine.info.iteration == 200
    assert engine.info.checkpoint_sha256 == file_sha256(checkpoint)
    assert engine.device.type == "cpu"
    assert not engine.model.training  # eval mode, or dropout/norm would drift


@pytest.mark.parametrize(("width", "height"), [(160, 90), (90, 160), (64, 64), (37, 101)])
def test_predict_returns_a_mask_in_the_original_frame(
    checkpoint: Path, width: int, height: int
) -> None:
    engine = load_engine(checkpoint, device="cpu", score_threshold=0.0, min_area_px=1)
    result = engine.predict(open_image(photo_bytes(width, height)))

    assert result.mask.shape == (height, width)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 1}
    assert 0.0 <= result.damage_pixel_ratio <= 1.0
    assert result.inference_ms > 0.0
    for instance in result.instances:
        x, y, w, h = instance.bbox
        assert 0 <= x and 0 <= y and x + w <= width and y + h <= height
        assert instance.class_name == "damage"
        assert instance.area_px >= 1


def test_thresholds_filter_instances_out(checkpoint: Path) -> None:
    permissive = load_engine(checkpoint, device="cpu", score_threshold=0.0, min_area_px=1)
    strict = load_engine(checkpoint, device="cpu", score_threshold=1.01, min_area_px=1)
    image = open_image(photo_bytes(128, 96))
    assert strict.predict(image).instances == []
    assert len(permissive.predict(image).instances) >= len(strict.predict(image).instances)


def test_load_engine_refuses_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_engine(tmp_path / "nope.pt")


def test_load_engine_refuses_an_input_size_the_architecture_cannot_take(
    tmp_path: Path,
) -> None:
    """UNet raises on a size that is not a multiple of 2**depth; saying so at
    load time beats discovering it on the first photo of the day."""
    config = resolve_config({**CONFIG, "input_size": 70})
    model = build_model(config, num_classes_for(config))
    path = tmp_path / "bad.pt"
    save_checkpoint(path, model, config, iteration=1, best_val_iou=0.0)
    with pytest.raises(ValueError, match="not a multiple"):
        load_engine(path, device="cpu")


def test_class_names_follow_the_mode() -> None:
    assert class_names_for(resolve_config({"mode": "binary"})) == ("background", "damage")
    assert class_names_for(resolve_config({"mode": "multiclass"})) == (
        "background", "large", "fine",
    )
