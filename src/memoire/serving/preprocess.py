"""Letterbox preprocessing for a single, arbitrary image (chap. 5.4).

The training path letterboxes inside ``DamageSegDataset`` (private
``_letterbox_geometry`` / ``_letterbox``, ``memoire/training/dataset.py``),
which can only be built from a processed COCO JSON: it cannot preprocess one
photo arriving on a Kafka topic. The geometry is reimplemented here as free
functions, deliberately *not* by refactoring the dataset — a training
campaign was running when this path was written and
``memoire/training/`` was frozen.

Duplicated code is a train/serve skew risk, so it is guarded mechanically:
``tests/test_serving_preprocess.py`` requires bit-for-bit equality between
this module's tensor and ``DamageSegDataset.__getitem__``'s on several aspect
ratios. Folding both onto a single shared helper is the follow-up once the
campaign is over.

Conventions preserved verbatim from ``docs/CONVENTIONS-TRAINING.md``:
BILINEAR resize, ratio preserved, centred on a black canvas, ``/255`` and
nothing else (no ImageNet mean/std — the model was trained from scratch on
raw [0, 1] pixels).

This module imports PIL and numpy only: no torch, so the message layer can
import it without pulling the training extra.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


class UnreadableImageError(ValueError):
    """The bytes do not decode into an image — PERMANENT error, goes to DLQ.

    Mirrors ``memoire.data.image_check.is_readable``: opening only reads the
    header, a truncated JPEG raises on ``load()`` alone (16 VehiDE files
    behaved exactly like that). Same rule as the corpus loaders — dropped,
    never silently (``docs/CONVENTIONS.md``), here as a dead letter plus a
    counter.
    """


@dataclass(frozen=True)
class LetterboxGeometry:
    """Where the resized image sits inside the square canvas."""

    input_size: int
    new_w: int
    new_h: int
    left: int
    top: int


def letterbox_geometry(width: int, height: int, input_size: int) -> LetterboxGeometry:
    """``width x height`` fitted into ``input_size`` squared, ratio preserved.

    Same arithmetic as ``DamageSegDataset._letterbox_geometry`` (including the
    ``max(1, round(...))`` guard against a degenerate side and the floor
    division that biases the padding to the top-left by one pixel on odd
    remainders).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    scale = min(input_size / width, input_size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    left = (input_size - new_w) // 2
    top = (input_size - new_h) // 2
    return LetterboxGeometry(input_size, new_w, new_h, left, top)


#: Failures that say "this machine is short of memory right now", not "these
#: bytes are bad". They are deliberately NOT converted into a permanent
#: :class:`UnreadableImageError`: dead-lettering a valid photo because the
#: container was under pressure buries it, whereas letting it escape as a
#: transient failure replays it after a restart (see ``service.RESOURCE_ERRORS``).
RESOURCE_ERRORS: tuple[type[BaseException], ...] = (MemoryError,)


def open_image(data: bytes) -> Image.Image:
    """Decode image bytes to RGB, fully (not just the header).

    Raises :class:`UnreadableImageError` on anything PIL refuses, so the
    caller can route the message to the DLQ instead of crashing the consumer.

    The ``except`` is broad on purpose. These bytes come off a topic, so they
    are hostile input, and PIL's decoders raise far more than ``OSError`` on
    them: ``Image.DecompressionBombError`` inherits straight from
    ``Exception`` (a 69-byte PNG whose IHDR announces 20000x20000 is enough),
    and individual plugins surface ``struct.error``, ``TypeError`` or
    ``IndexError`` on a corrupt header. Any of those escaping this function
    kills the consumer loop mid-record — no commit, no dead letter — and the
    container restarts on the same offset: the poison pill the error taxonomy
    exists to prevent. The only exceptions let through are the resource ones,
    which are about the machine rather than the message.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except RESOURCE_ERRORS:  # transient: the caller retries, then stops
        raise
    except Exception as exc:  # PIL raises OSError on truncated data, and more
        raise UnreadableImageError(f"{type(exc).__name__}: {exc}") from exc


def letterbox_image(image: Image.Image, input_size: int) -> tuple[np.ndarray, LetterboxGeometry]:
    """``(canvas HxWx3 uint8, geometry)`` — the network's input, still uint8.

    Kept in uint8 on purpose: the ``float32 / 255`` conversion belongs to the
    torch side (``memoire.serving.inference``), which is exactly where the
    training path does it (inline in ``__getitem__``), so the parity test can
    compare the two tensors bit for bit.
    """
    geometry = letterbox_geometry(image.width, image.height, input_size)
    resized = image.resize((geometry.new_w, geometry.new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    canvas.paste(resized, (geometry.left, geometry.top))
    return np.asarray(canvas, dtype=np.uint8), geometry


def unletterbox_mask(
    mask: np.ndarray, geometry: LetterboxGeometry, width: int, height: int
) -> np.ndarray:
    """Bring a canvas-sized mask back to the original ``width x height``.

    Crops away the padding bands then resizes NEAREST (a class-id map must
    never be interpolated). Nothing in the training code does this — training
    only ever needs the forward direction — yet without it every published
    box would be expressed in the network's frame instead of the photo's.
    """
    if mask.shape != (geometry.input_size, geometry.input_size):
        raise ValueError(
            f"mask is {mask.shape}, expected "
            f"({geometry.input_size}, {geometry.input_size})"
        )
    crop = mask[
        geometry.top : geometry.top + geometry.new_h,
        geometry.left : geometry.left + geometry.new_w,
    ]
    resized = Image.fromarray(crop.astype(np.uint8), mode="L").resize(
        (width, height), Image.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8)


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` of a boolean mask's non-zero pixels; zeros if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Serialise a class-id mask as an 8-bit PNG (the blob a message points to).

    Palette-free 'L' mode on purpose: the pixel VALUE is the class id, so a
    consumer reads meaning without a palette table (the mapping is published
    in the message under ``mask.class_values``).
    """
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
