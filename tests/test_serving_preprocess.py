"""Train/serve parity of the letterbox (chap. 5.4).

``memoire.serving.preprocess`` reimplements the geometry that lives in the
private methods of ``DamageSegDataset`` (the training package was frozen by a
running campaign, so no refactor was possible). Duplicated preprocessing is
how a served model quietly stops matching the evaluated one, so the guard is
mechanical: the serving tensor must equal the dataset's BIT FOR BIT, on
several aspect ratios.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from memoire.serving.preprocess import (
    UnreadableImageError,
    bounding_box,
    encode_mask_png,
    letterbox_geometry,
    letterbox_image,
    open_image,
    unletterbox_mask,
)
from memoire.training.dataset import DamageSegDataset

INPUT_SIZE = 64
GEOMETRIES = [(160, 90), (90, 160), (64, 64), (37, 101)]


def _make_corpus(root: Path) -> Path:
    """Mini COCO export, one image per geometry, deterministic pixels."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    images, annotations = [], []
    for i, (width, height) in enumerate(GEOMETRIES):
        rng = np.random.default_rng(i)
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        file_name = f"img_{i:03d}.png"
        Image.fromarray(pixels).save(images_dir / file_name)
        images.append(
            {
                "id": i + 1, "file_name": file_name, "width": width, "height": height,
                "split": "train", "source": "synthetic",
                "memoire_image_id": f"synthetic/img_{i:03d}",
                "memoire_file_path": str(images_dir / file_name),
            }
        )
        annotations.append(
            {
                "id": i + 1, "image_id": i + 1, "category_id": 1, "iscrowd": 0,
                "area": 36.0, "bbox": [2, 2, 6, 6],
                "segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]],
            }
        )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "dent"}],
    }
    path = root / "corpus.json"
    path.write_text(json.dumps(coco), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    return _make_corpus(tmp_path_factory.mktemp("serving_parity"))


@pytest.fixture(scope="module")
def dataset(corpus: Path) -> DamageSegDataset:
    return DamageSegDataset(corpus, None, "train", mode="binary", input_size=INPUT_SIZE)


def test_letterbox_tensor_matches_the_training_dataset(dataset: DamageSegDataset) -> None:
    for i in range(len(dataset)):
        expected, _ = dataset[i]
        image = Image.open(dataset._paths[i]).convert("RGB")
        canvas, _ = letterbox_image(image, INPUT_SIZE)
        served = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
        served = served.permute(2, 0, 1).contiguous()
        assert torch.equal(served, expected), f"train/serve skew on image {i}"


def test_letterbox_geometry_matches_the_private_dataset_method(
    dataset: DamageSegDataset,
) -> None:
    for width, height in GEOMETRIES:
        expected = dataset._letterbox_geometry(width, height)
        geometry = letterbox_geometry(width, height, INPUT_SIZE)
        assert (geometry.new_w, geometry.new_h, geometry.left, geometry.top) == expected


def test_letterbox_pads_with_black_and_preserves_the_ratio() -> None:
    image = Image.new("RGB", (100, 50), (255, 0, 0))
    canvas, geometry = letterbox_image(image, INPUT_SIZE)
    assert canvas.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert geometry.new_w == INPUT_SIZE and geometry.new_h == INPUT_SIZE // 2
    assert canvas[: geometry.top].max() == 0  # top band untouched


def test_unletterbox_round_trips_to_the_original_resolution() -> None:
    for width, height in GEOMETRIES:
        geometry = letterbox_geometry(width, height, INPUT_SIZE)
        canvas_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        canvas_mask[
            geometry.top : geometry.top + geometry.new_h,
            geometry.left : geometry.left + geometry.new_w,
        ] = 1
        native = unletterbox_mask(canvas_mask, geometry, width, height)
        assert native.shape == (height, width)
        assert native.min() == 1  # the padding bands are gone, not resampled in


def test_unletterbox_drops_a_false_positive_in_the_padding_band() -> None:
    """The letterbox pads with background-labelled pixels that the loss counts,
    so damage predicted there is possible — cropping must discard it."""
    width, height = 100, 50
    geometry = letterbox_geometry(width, height, INPUT_SIZE)
    canvas_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    canvas_mask[: geometry.top, :] = 1  # entirely inside the top black band
    native = unletterbox_mask(canvas_mask, geometry, width, height)
    assert native.max() == 0


def test_unletterbox_rejects_a_mask_of_the_wrong_size() -> None:
    geometry = letterbox_geometry(100, 50, INPUT_SIZE)
    with pytest.raises(ValueError, match="expected"):
        unletterbox_mask(np.zeros((8, 8), dtype=np.uint8), geometry, 100, 50)


def test_bounding_box() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:6, 4:9] = True
    assert bounding_box(mask) == (4, 3, 5, 3)
    assert bounding_box(np.zeros((4, 4), dtype=bool)) == (0, 0, 0, 0)


def test_open_image_rejects_undecodable_bytes() -> None:
    with pytest.raises(UnreadableImageError):
        open_image(b"not an image at all")


def test_open_image_rejects_a_truncated_jpeg(tmp_path: Path) -> None:
    """The failure mode ``memoire.data.image_check`` was written for: a valid
    header, broken data — only a full ``load()`` catches it."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 64), (120, 30, 30)).save(path, quality=95)
    truncated = path.read_bytes()[: len(path.read_bytes()) // 2]
    with pytest.raises(UnreadableImageError):
        open_image(truncated)


def test_encode_mask_png_round_trips_class_ids() -> None:
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[2:5, 3:6] = 2
    decoded = np.asarray(Image.open(io.BytesIO(encode_mask_png(mask))))
    assert np.array_equal(decoded, mask)


def _decompression_bomb_png(side: int = 30000) -> bytes:
    """A tiny, syntactically valid PNG whose header announces a huge image.

    69 bytes on the wire, 900 megapixels once believed: the shape of a
    decompression bomb, and — with one flipped byte in a legitimate header —
    the shape of a transport corruption too.
    """
    import struct
    import zlib

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 16))
        + chunk(b"IEND", b"")
    )


def test_open_image_rejects_a_decompression_bomb() -> None:
    """PIL's DecompressionBombError inherits from Exception, not OSError: it
    used to escape open_image, then handle_record, then the whole loop — a
    69-byte message able to kill the consumer on every restart."""
    with pytest.raises(UnreadableImageError):
        open_image(_decompression_bomb_png())


def test_open_image_rejects_whatever_a_plugin_raises(monkeypatch) -> None:
    """Image decoders raise struct.error, TypeError, IndexError... on hostile
    input. None of it may leave this function as anything but a permanent
    error."""
    import struct

    for exception in (struct.error("bad header"), TypeError("nope"), IndexError("nope")):
        def _raise(*_args, exc=exception, **_kwargs):
            raise exc

        monkeypatch.setattr(Image, "open", _raise)
        with pytest.raises(UnreadableImageError):
            open_image(b"whatever")


def test_open_image_lets_a_memory_error_through(monkeypatch) -> None:
    """Out of memory is about the machine, not about these bytes: converting
    it into a permanent error would bury a valid photo in the DLQ."""
    def _raise(*_args, **_kwargs):
        raise MemoryError("not now")

    monkeypatch.setattr(Image, "open", _raise)
    with pytest.raises(MemoryError):
        open_image(b"whatever")
