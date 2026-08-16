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

from memoire.data.taxonomy import Taxonomy

MODES = ("binary", "multiclass")

# Fixed, corpus-independent order for multiclass mode (chap. 6.4): background
# + the two size groups. Never derived from a COCO file's own local
# `categories` list — that numbering is per-file (alphabetical order of
# whatever canonical classes happen to appear in that one export), so
# combining several corpus JSONs used to silently misalign mask values
# (mask value 2 meaning a different canonical class per source).
MULTICLASS_GROUP_NAMES = ("background", "large", "fine")


class DamageSegDataset(Dataset):
    """Damage segmentation dataset backed by a processed COCO JSON.

    ``__getitem__`` returns ``(image, mask)`` with ``image`` a float32
    ``3xSxS`` tensor in ``[0, 1]`` and ``mask`` an int64 ``SxS`` tensor
    (``S = input_size``). Mask values: ``{0: background, 1: damage}`` in
    binary mode; ``{0: background, 1: large, 2: fine}`` in multiclass mode
    (chap. 6.4 — not the 13 canonical classes), via ``taxonomy.size()``, fixed
    regardless of which corpus JSON this is (multiclass mode requires
    ``taxonomy``).

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
        taxonomy: Taxonomy | None = None,
        copy_paste: bool = False,
        copy_paste_prob: float = 0.5,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if mode == "multiclass" and taxonomy is None:
            raise ValueError(
                "mode='multiclass' requires a taxonomy (background/large/fine grouping, chap. 6.4)"
            )
        if not 0.0 <= copy_paste_prob <= 1.0:
            raise ValueError(f"copy_paste_prob must be in [0, 1], got {copy_paste_prob}")

        self.coco_json = Path(coco_json)
        self.images_root = Path(images_root) if images_root is not None else None
        self.split = split
        self.mode = mode
        self.input_size = int(input_size)
        self.augment = augment
        self.generator = generator
        self.copy_paste = copy_paste
        self.copy_paste_prob = float(copy_paste_prob)

        with self.coco_json.open("r", encoding="utf-8") as fh:
            dataset = json.load(fh)

        self._coco = COCO()
        self._coco.dataset = dataset
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco.createIndex()

        categories = sorted(dataset.get("categories", []), key=lambda cat: cat["id"])
        if mode == "binary":
            # Union of all instances regardless of class (_build_mask never
            # consults _class_index in binary mode).
            self._class_index = {}
            self.class_names = ["background", "damage"]
        else:
            # Global mapping: every category's canonical NAME (already the
            # taxonomy-harmonised name at export time) determines its group id
            # via the taxonomy, fixed regardless of this file's own local
            # category ids — see MULTICLASS_GROUP_NAMES above.
            self._class_index = {
                cat["id"]: MULTICLASS_GROUP_NAMES.index(taxonomy.size(cat["name"]))
                for cat in categories
            }
            self.class_names = list(MULTICLASS_GROUP_NAMES)
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

    def _letterbox_geometry(self, width: int, height: int) -> tuple[int, int, int, int]:
        """``(new_w, new_h, left, top)`` fitting ``width x height`` into ``input_size``."""
        size = self.input_size
        scale = min(size / width, size / height)
        new_w = max(1, round(width * scale))
        new_h = max(1, round(height * scale))
        left = (size - new_w) // 2
        top = (size - new_h) // 2
        return new_w, new_h, left, top

    def _letterbox_mask(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Letterbox a single mask (NEAREST) onto this dataset's canvas geometry."""
        new_w, new_h, left, top = self._letterbox_geometry(width, height)
        mask_img = Image.fromarray(mask, mode="L").resize((new_w, new_h), Image.NEAREST)
        out_mask = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
        out_mask[top : top + new_h, left : left + new_w] = np.asarray(mask_img, dtype=np.uint8)
        return out_mask

    def _letterbox(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """Resize to fit ``input_size`` (ratio preserved) and pad with background."""
        width, height = image.size
        new_w, new_h, left, top = self._letterbox_geometry(width, height)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))
        canvas.paste(image, (left, top))
        out_mask = self._letterbox_mask(mask, width, height)
        return canvas, out_mask

    def instance_targets(self, i: int) -> list[tuple[int, np.ndarray]]:
        """Per-instance ``(class_id, letterboxed boolean mask)`` for image ``i``.

        Unlike ``__getitem__``'s merged semantic mask (overlaps collapse to one
        class per pixel), instances stay distinct here — needed for
        instance-level mAP (chap. 7.4); the per-pixel train/val path never
        needs this.
        """
        info = self.images[i]
        width, height = info["width"], info["height"]
        ann_ids = self._coco.getAnnIds(imgIds=[info["id"]])
        targets: list[tuple[int, np.ndarray]] = []
        for ann in self._coco.loadAnns(ann_ids):
            if not ann.get("segmentation"):
                continue
            instance = self._coco.annToMask(ann)
            class_id = 1 if self.mode == "binary" else self._class_index[ann["category_id"]]
            targets.append((class_id, self._letterbox_mask(instance, width, height) > 0))
        return targets

    # -- copy-paste augmentation -----------------------------------------------

    def _load_letterboxed(self, i: int) -> tuple[Image.Image, np.ndarray]:
        """Letterboxed ``(image, mask)`` for index ``i``, before any augmentation."""
        info = self.images[i]
        image = Image.open(self._paths[i]).convert("RGB")
        mask = self._build_mask(info)
        return self._letterbox(image, mask)

    def _copy_paste(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """Paste a randomly scaled instance from another training image onto
        this one (Ghiasi et al. 2021, chap. 7.3): a substitute for
        pre-training that stays strictly endogenous to the target corpus (no
        external data) — increases object/context diversity by recombining
        instances already in the training split.

        Operates in letterboxed canvas space (both images already the same
        ``input_size``), not native resolution — pixels/mask values are
        copied only where the (possibly rescaled) source mask is non-zero, so
        the source's own background never overwrites the target.
        """
        source_idx = int(torch.randint(0, len(self), (1,), generator=self.generator).item())
        src_image, src_mask = self._load_letterboxed(source_idx)
        if not bool((src_mask > 0).any()):
            return image, mask  # nothing to paste (e.g. an unannotated image)

        size = self.input_size
        scale = float(
            torch.empty((), dtype=torch.float32).uniform_(0.5, 1.5, generator=self.generator)
        )
        new_size = max(1, round(size * scale))
        src_image_np = np.asarray(src_image.resize((new_size, new_size), Image.BILINEAR))
        src_mask_np = np.asarray(
            Image.fromarray(src_mask, mode="L").resize((new_size, new_size), Image.NEAREST)
        )

        # Scaled source may be larger than the canvas: crop to fit before placing.
        crop = min(new_size, size)
        src_image_np = src_image_np[:crop, :crop]
        src_mask_np = src_mask_np[:crop, :crop]

        max_offset = size - crop
        if max_offset > 0:
            off_x = int(torch.randint(0, max_offset + 1, (1,), generator=self.generator).item())
            off_y = int(torch.randint(0, max_offset + 1, (1,), generator=self.generator).item())
        else:
            off_x = off_y = 0

        target_image_np = np.asarray(image).copy()
        paste = src_mask_np > 0
        ys, xs = slice(off_y, off_y + crop), slice(off_x, off_x + crop)
        region_image = target_image_np[ys, xs]
        region_mask = mask[ys, xs]
        region_image[paste] = src_image_np[paste]
        region_mask[paste] = src_mask_np[paste]
        target_image_np[ys, xs] = region_image
        mask[ys, xs] = region_mask
        return Image.fromarray(target_image_np), mask

    # -- Dataset API ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._load_letterboxed(i)

        if (
            self.augment
            and self.copy_paste
            and bool(torch.rand((), generator=self.generator) < self.copy_paste_prob)
        ):
            image, mask = self._copy_paste(image, mask)

        image_t = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        image_t = image_t.permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask.astype(np.int64))

        if self.augment and bool(torch.rand((), generator=self.generator) < 0.5):
            image_t = torch.flip(image_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])
        return image_t, mask_t
