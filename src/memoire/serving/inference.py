"""Serving-side engine: a checkpoint in, a native-resolution mask out (chap. 5.4).

This is the only module of the package that imports torch, and the only one
the service cannot exercise without it — hence the
:class:`~memoire.serving.ports.SegmentationEngine` protocol, which lets the
service tests run with a constant engine.

Two deliberate differences with ``scripts/evaluate_map.py``, which reloads a
checkpoint the same way:

1. **The checkpoint is the single source of truth.** ``train()`` resolves the
   config (``resolve_config``) *before* ``save_checkpoint``, so
   ``checkpoint["config"]`` already carries model/input_size/mode/depth/…
   No ``--config`` YAML is needed, and none is accepted: a served model whose
   architecture came from a YAML edited after training would silently fail to
   load — or worse, load into a differently-shaped network.
2. **The device defaults to CPU.** ``resolve_device("auto")`` would grab the
   GPU, which on the training machine is running a multi-day campaign. Only
   an explicit ``device="cuda"`` (or ``"auto"``) opts into it.

Instance extraction reuses ``memoire.training.metrics.predicted_instances``
verbatim (8-connected components of the argmax mask, score = mean softmax
probability over the component): the served instances are the same objects
the thesis' mAP is computed on (chap. 7.4), not a second, subtly different
implementation.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from memoire.serving.ports import DamageInstance, InferenceResult, ModelInfo
from memoire.serving.preprocess import bounding_box, letterbox_image, unletterbox_mask
from memoire.training.dataset import MULTICLASS_GROUP_NAMES
from memoire.training.metrics import predicted_instances
from memoire.training.train import (
    build_model,
    load_checkpoint,
    num_classes_for,
    resolve_config,
    resolve_device,
)

logger = logging.getLogger("memoire.serving.inference")


def class_names_for(config: dict) -> tuple[str, ...]:
    """Published class names, without instantiating a dataset."""
    if config["mode"] == "binary":
        return ("background", "damage")
    return tuple(MULTICLASS_GROUP_NAMES)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Checkpoint digest, published in every mask message.

    Downstream deduplication key together with ``photo_id``: at-least-once
    delivery replays photos, and inference is deterministic, so a duplicate
    is only recognisable as such if the model is identified.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TorchSegmentationEngine:
    """A loaded checkpoint, ready to segment arbitrary photos."""

    def __init__(
        self,
        model: torch.nn.Module,
        info: ModelInfo,
        device: torch.device,
        score_threshold: float = 0.5,
        min_area_px: int = 64,
    ) -> None:
        self.model = model
        self.info = info
        self.device = device
        self.score_threshold = float(score_threshold)
        self.min_area_px = int(min_area_px)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> InferenceResult:
        """Segment one PIL image; mask and boxes come back in ITS pixels."""
        started = time.perf_counter()
        width, height = image.size
        canvas, geometry = letterbox_image(image, self.info.input_size)
        # Exactly the training conversion (dataset.__getitem__): float32 /255,
        # HWC -> CHW, nothing else — no ImageNet normalisation anywhere in
        # this project. Parity is asserted in tests/test_serving_preprocess.py.
        tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).contiguous()

        logits = self.model(tensor.unsqueeze(0).to(self.device))[0]
        canvas_mask = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
        mask = unletterbox_mask(canvas_mask, geometry, width, height)

        instances: list[DamageInstance] = []
        for class_id, score, component in predicted_instances(
            logits, list(range(1, self.info.num_classes))
        ):
            if score < self.score_threshold:
                continue
            # Un-letterboxing before measuring is what makes the padding bands
            # harmless: the letterbox pads with background-labelled pixels and
            # the loss counts them (docs/CONVENTIONS-TRAINING.md), so a false
            # positive there is possible — cropping drops it instead of
            # publishing a box outside the photo.
            native = unletterbox_mask(component.astype(np.uint8), geometry, width, height) > 0
            area_px = int(native.sum())
            if area_px < self.min_area_px:
                continue
            instances.append(
                DamageInstance(
                    class_id=int(class_id),
                    class_name=self.info.class_names[class_id],
                    score=float(score),
                    bbox=bounding_box(native),
                    area_px=area_px,
                )
            )
        instances.sort(key=lambda inst: inst.score, reverse=True)
        return InferenceResult(
            mask=mask,
            instances=instances,
            damage_pixel_ratio=float((mask > 0).mean()),
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )


def load_engine(
    checkpoint_path: Path | str,
    device: str = "cpu",
    score_threshold: float = 0.5,
    min_area_px: int = 64,
) -> TorchSegmentationEngine:
    """Rebuild the trained network from ``best.pt`` alone."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    torch_device = resolve_device(device)
    if torch_device.type == "cuda":
        logger.warning(
            "serving on CUDA (%s): make sure no training campaign is using that GPU",
            torch_device,
        )
    checkpoint = load_checkpoint(path, map_location=str(torch_device))
    config = resolve_config(checkpoint.get("config") or {})
    num_classes = num_classes_for(config)
    input_size = int(config["input_size"])
    depth = int(config["depth"])
    if input_size % (2**depth) != 0:
        # UNet/PlainEncoderDecoder both raise on a non-divisible input; failing
        # here says why, instead of at the first photo of the day.
        raise ValueError(
            f"input_size={input_size} is not a multiple of 2**depth={2**depth}: "
            "this checkpoint cannot be served as-is"
        )
    model = build_model(config, num_classes).to(torch_device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    info = ModelInfo(
        name=str(config["model"]),
        mode=str(config["mode"]),
        input_size=input_size,
        num_classes=num_classes,
        class_names=class_names_for(config),
        checkpoint_sha256=file_sha256(path),
        run_id=path.parent.as_posix(),
        iteration=checkpoint.get("iteration"),
    )
    logger.info(
        "loaded %s (%s, %d classes, input_size=%d) from %s on %s",
        info.name, info.mode, info.num_classes, info.input_size, path, torch_device,
    )
    return TorchSegmentationEngine(model, info, torch_device, score_threshold, min_area_px)
