"""Segmentation dataset over the processed COCO exports (``data/processed/``).

Design constraints (see ``docs/CONVENTIONS-TRAINING.md``):

- images are selected by the ``split`` field carried by each image entry;
- masks are rasterised with pycocotools (``annToMask``), instances are merged
  (union in binary mode, per canonical class in multiclass mode, class ids
  following the ``categories`` order of the JSON);
- image and mask are letterboxed to a square ``input_size`` canvas (aspect
  ratio preserved, background padding, BILINEAR for the image, NEAREST for
  the mask);
- images are loaded lazily, at ``__getitem__`` time only (PIL, RGB).
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

MODES = ("binary", "multiclass")


class DamageSegDataset(Dataset):
    """Damage segmentation dataset backed by a processed COCO JSON.

    ``__getitem__`` returns ``(image, mask)`` with ``image`` a float32
    ``3xSxS`` tensor in ``[0, 1]`` and ``mask`` an int64 ``SxS`` tensor
    (``S = input_size``). Mask values: ``{0: background, 1: damage}`` in
    binary mode, ``{0: background, 1..K}`` in multiclass mode, where class
    ``k`` is the k-th entry of the JSON ``categories`` sorted by id.

    The image file is resolved from the ``memoire_file_path`` extra when
    present, otherwise from ``images_root / file_name``.
    """

    def __init__(
        self,
        coco_json: Path,
        images_root: Path | None,
        split: str,
        mode: str = "binary",
        input_size: int = 512,
        augment: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        self.coco_json = Path(coco_json)
        self.images_root = Path(images_root) if images_root is not None else None
        self.split = split
        self.mode = mode
        self.input_size = int(input_size)
        self.augment = augment
        self.generator = generator

        with self.coco_json.open("r", encoding="utf-8") as fh:
            dataset = json.load(fh)

        self._coco = COCO()
        self._coco.dataset = dataset
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco.createIndex()

        categories = sorted(dataset.get("categories", []), key=lambda cat: cat["id"])
        self._class_index = {cat["id"]: i for i, cat in enumerate(categories, start=1)}
        if mode == "binary":
            self.class_names = ["background", "damage"]
        else:
            self.class_names = ["background"] + [cat["name"] for cat in categories]
        self.num_classes = len(self.class_names)

        self.images = [img for img in dataset["images"] if img.get("split") == split]
        if not self.images:
            available = sorted({str(img.get("split")) for img in dataset["images"]})
            raise ValueError(
                f"no image with split={split!r} in {self.coco_json} "
                f"(available split values: {available})"
            )
        # Paths are resolved eagerly (pure string work, no disk access) so a
        # misconfigured images_root fails at construction time, not mid-epoch.
        self._paths = [self._resolve_path(img) for img in self.images]
        # Instance count per image (density axis, chap. 7.1): cheap from the
        # already-built COCO index, no annotation decoding needed.
        self.instance_counts = [
            len(self._coco.getAnnIds(imgIds=[img["id"]])) for img in self.images
        ]

    # -- resolution -----------------------------------------------------------

    def _resolve_path(self, image: dict) -> Path:
        file_path = image.get("memoire_file_path")
        if file_path:
            return Path(file_path)
        if self.images_root is None:
            raise ValueError(
                "images_root is required when an image entry has no "
                f"'memoire_file_path' (image {image.get('memoire_image_id')!r})"
            )
        return self.images_root / image["file_name"]

    # -- mask building --------------------------------------------------------

    def _build_mask(self, image: dict) -> np.ndarray:
        """Rasterise all instances of one image into a single class mask."""
        mask = np.zeros((image["height"], image["width"]), dtype=np.uint8)
        ann_ids = self._coco.getAnnIds(imgIds=[image["id"]])
        anns = self._coco.loadAnns(ann_ids)
        # Deterministic overlap resolution: higher category id wins.
        anns = sorted(anns, key=lambda ann: (ann["category_id"], ann["id"]))
        for ann in anns:
            if not ann.get("segmentation"):
                continue
            instance = self._coco.annToMask(ann)
            value = 1 if self.mode == "binary" else self._class_index[ann["category_id"]]
            mask[instance > 0] = value
        return mask

    # -- letterbox ------------------------------------------------------------

    def _letterbox(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """Resize to fit ``input_size`` (ratio preserved) and pad with background."""
        size = self.input_size
        width, height = image.size
        scale = min(size / width, size / height)
        new_w = max(1, round(width * scale))
        new_h = max(1, round(height * scale))

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask_img = Image.fromarray(mask, mode="L").resize((new_w, new_h), Image.NEAREST)

        left = (size - new_w) // 2
        top = (size - new_h) // 2
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(image, (left, top))
        out_mask = np.zeros((size, size), dtype=np.uint8)
        out_mask[top : top + new_h, left : left + new_w] = np.asarray(mask_img, dtype=np.uint8)
        return canvas, out_mask

    # -- Dataset API ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        info = self.images[i]
        image = Image.open(self._paths[i]).convert("RGB")
        mask = self._build_mask(info)
        image, mask = self._letterbox(image, mask)

        image_t = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        image_t = image_t.permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask.astype(np.int64))

        if self.augment and bool(torch.rand((), generator=self.generator) < 0.5):
            image_t = torch.flip(image_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])
        return image_t, mask_t
