"""Tests for memoire.data.image_check (corrupted/unreadable image filtering)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from memoire.data.image_check import filter_unreadable, is_readable


def _write_valid_image(path: Path) -> None:
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)


def _write_truncated_image(path: Path) -> None:
    """A real JPEG header followed by nothing — decodes the header, fails on pixels."""
    _write_valid_image(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])


def test_is_readable_true_for_valid_image(tmp_path):
    path = tmp_path / "ok.png"
    _write_valid_image(path)
    assert is_readable(str(path)) is True


def test_is_readable_false_for_truncated_image(tmp_path):
    path = tmp_path / "bad.png"
    _write_truncated_image(path)
    assert is_readable(str(path)) is False


def test_is_readable_false_for_missing_file(tmp_path):
    assert is_readable(str(tmp_path / "does_not_exist.png")) is False


def test_filter_unreadable_drops_only_corrupted_records(tmp_path):
    good_path = tmp_path / "good.png"
    bad_path = tmp_path / "bad.png"
    _write_valid_image(good_path)
    _write_truncated_image(bad_path)

    records = [
        {"image_id": "a", "file_path": str(good_path), "instances": [{}, {}]},
        {"image_id": "b", "file_path": str(bad_path), "instances": [{}, {}, {}]},
    ]
    kept, n_images_dropped, n_instances_dropped = filter_unreadable(records, source="testsource")
    assert n_images_dropped == 1
    assert n_instances_dropped == 3
    assert [r["image_id"] for r in kept] == ["a"]


def test_filter_unreadable_keeps_everything_when_all_valid(tmp_path):
    paths = [tmp_path / f"img{i}.png" for i in range(3)]
    for p in paths:
        _write_valid_image(p)
    records = [
        {"image_id": str(i), "file_path": str(p), "instances": []} for i, p in enumerate(paths)
    ]
    kept, n_images_dropped, n_instances_dropped = filter_unreadable(records, source="testsource")
    assert n_images_dropped == 0
    assert n_instances_dropped == 0
    assert len(kept) == 3
