"""Iteration-based training loop for the from-scratch damage segmentation U-Net.

Implements the Phase 2 conventions (``docs/CONVENTIONS-TRAINING.md``):

- seeds fixed everywhere (``random``, ``numpy``, ``torch``, DataLoader
  ``generator`` + ``worker_init_fn``) so that two runs with the same config
  produce the same losses;
- AdamW with a linear-warmup + cosine schedule driven by *iterations* (never
  epochs — the schedule length is a config parameter, never hard-coded);
- periodic validation (``val_every``) with per-class IoU/Dice streamed through
  a confusion matrix (``memoire.training.metrics``);
- ``last.pt`` / ``best.pt`` checkpoints carrying ``state_dict`` + config +
  iteration (best = highest mean damage IoU on the validation split);
- optional MLflow tracking with a JSONL fallback (``history.jsonl`` in
  ``output_dir``) that is always written and never blocks a run;
- ``subset_n_images``: deterministic (seeded) subsampling of the train split,
  the lever behind the data-volume curve of the thesis.

Entry point: ``train(config) -> dict`` summary. CLI wrapper in
``scripts/train.py``, defaults in ``configs/train.yaml``.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from memoire.data.taxonomy import load_taxonomy
from memoire.model.unet import UNet
from memoire.training import metrics as seg_metrics
from memoire.training.dataset import MULTICLASS_GROUP_NAMES, DamageSegDataset
from memoire.training.losses import DiceCELoss

logger = logging.getLogger("memoire.training")

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "corpus": ["data/processed/cardd.json"],
    "images_root": None,
    "mode": "binary",  # "binary" | "multiclass" (background/large/fine, chap. 6.4)
    "taxonomy_path": "configs/taxonomy.yaml",  # needed for multiclass's large/fine grouping
    "input_size": 512,
    "in_channels": 3,
    "base_channels": 32,
    "depth": 4,
    "gn_groups": 8,
    "batch_size": 4,
    "num_workers": 0,
    "augment": True,
    "copy_paste": False,        # Ghiasi et al. 2021 (chap. 7.3), endogenous H3 ablation
    "copy_paste_prob": 0.5,     # per train-item probability, only when copy_paste=True
    "subset_n_images": None,
    "density_bucket": None,  # [min, max|null] instances/image; null = volume axis
    "lr": 3.0e-4,
    "weight_decay": 1.0e-2,
    "ce_weight": 1.0,
    "dice_weight": 1.0,
    "iterations": 200,
    "warmup_iterations": 20,
    "val_every": 50,
    "device": "auto",  # "auto" | "cpu" | "cuda" | "cuda:N"
    "output_dir": "runs/dev",
    "mlflow": {"tracking_uri": None, "experiment": "memoire-seg", "run_name": None},
}


# -- config / seeding ---------------------------------------------------------


def resolve_config(config: dict) -> dict:
    """Merge a user config over :data:`DEFAULTS` (shallow, user keys win)."""
    resolved = dict(DEFAULTS)
    resolved.update(config or {})
    mlflow_cfg = dict(DEFAULTS["mlflow"])
    mlflow_cfg.update((config or {}).get("mlflow") or {})
    resolved["mlflow"] = mlflow_cfg
    return resolved


def set_global_seed(seed: int) -> None:
    """Seed every RNG the training loop touches."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_damage_datasets(dataset: Dataset):
    """Yield every :class:`DamageSegDataset` nested inside Subset/ConcatDataset wrappers."""
    if isinstance(dataset, DamageSegDataset):
        yield dataset
    elif isinstance(dataset, Subset):
        yield from _iter_damage_datasets(dataset.dataset)
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _iter_damage_datasets(child)


def seed_worker(worker_id: int) -> None:
    """Derive per-worker numpy/random/augmentation seeds from the torch base seed.

    ``torch.initial_seed()`` already differs per worker (PyTorch derives it from
    the DataLoader's ``generator``). The flip-augmentation ``torch.Generator``
    passed into every ``DamageSegDataset`` is a plain object copied verbatim into
    each forked worker process, so without this it starts identical (same state,
    same seed) in every worker and produces correlated flip decisions across
    workers instead of independent ones.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        for dataset in _iter_damage_datasets(worker_info.dataset):
            if dataset.generator is not None:
                dataset.generator.manual_seed(worker_seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def num_classes_for(config: dict) -> int:
    """2 in binary mode; 3 in multiclass mode (background/large/fine, chap. 6.4)."""
    if config["mode"] == "binary":
        return 2
    return len(MULTICLASS_GROUP_NAMES)


# -- data ---------------------------------------------------------------------


def subset_indices(n_total: int, n_keep: int, seed: int) -> list[int]:
    """Deterministic sample of ``n_keep`` indices out of ``n_total`` (sorted)."""
    n_keep = min(int(n_keep), n_total)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n_keep))


def flat_instance_counts(dataset: Dataset) -> list[int]:
    """Per-image instance count, flattened in the same order as ``dataset``'s indices.

    ``dataset`` is whatever :func:`build_dataset` returns: a single
    :class:`DamageSegDataset`, or a :class:`ConcatDataset` of several — never
    a :class:`Subset` (density selection runs before subsetting, on the full
    corpus). ``ConcatDataset`` concatenates its children's global index ranges
    in ``self.datasets`` order, so this must walk them in the same order.
    """
    if isinstance(dataset, DamageSegDataset):
        return list(dataset.instance_counts)
    if isinstance(dataset, ConcatDataset):
        counts: list[int] = []
        for child in dataset.datasets:
            counts.extend(flat_instance_counts(child))
        return counts
    raise TypeError(f"unsupported dataset type for density sampling: {type(dataset)!r}")


def density_indices(
    counts: Sequence[int], n_keep: int, seed: int, density_min: int, density_max: int | None
) -> list[int]:
    """Deterministic sample of ``n_keep`` indices whose instance count falls in
    ``[density_min, density_max]`` (``density_max=None`` means unbounded, i.e.
    the "N+" bucket of chap. 7.1's density axis).

    Density-axis counterpart of :func:`subset_indices`: that one samples
    uniformly across the whole corpus (varying volume, density unconstrained);
    this one samples only from images matching one density stratum (volume
    held constant via ``n_keep``, density varied via the bucket) — the two
    axes He et al.'s unpaired hypothesis (chap. 2.4) needs decoupled.
    """
    pool = [
        i
        for i, c in enumerate(counts)
        if c >= density_min and (density_max is None or c <= density_max)
    ]
    if len(pool) < n_keep:
        bucket = f"[{density_min}, {'+inf' if density_max is None else density_max}]"
        raise ValueError(
            f"density bucket {bucket} has only {len(pool)} image(s), cannot sample "
            f"{n_keep} (reduce n_keep or widen the bucket)"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(pool, int(n_keep)))


def build_dataset(config: dict, split: str, generator: torch.Generator | None = None) -> Dataset:
    """Concatenate one :class:`DamageSegDataset` per corpus JSON for ``split``."""
    images_root = config.get("images_root")
    # One shared Taxonomy for every corpus (chap. 6.4's large/fine grouping is
    # global, not per-file) — loaded once here rather than per-DamageSegDataset.
    taxonomy = (
        load_taxonomy(config["taxonomy_path"]) if config["mode"] == "multiclass" else None
    )
    datasets: list[Dataset] = [
        DamageSegDataset(
            coco_json=Path(corpus_json),
            images_root=Path(images_root) if images_root else None,
            split=split,
            mode=config["mode"],
            input_size=int(config["input_size"]),
            augment=bool(config["augment"]) and split == "train",
            generator=generator,
            taxonomy=taxonomy,
            copy_paste=bool(config.get("copy_paste", False)),
            copy_paste_prob=float(config.get("copy_paste_prob", 0.5)),
        )
        for corpus_json in config["corpus"]
    ]
    if not datasets:
        raise ValueError("config['corpus'] must list at least one processed COCO JSON")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


# -- schedule -----------------------------------------------------------------


def lr_factor(step: int, warmup_iterations: int, total_iterations: int) -> float:
    """Linear warmup to 1 over ``warmup_iterations``, then cosine decay to 0.

    ``step`` counts completed iterations (0-based, LambdaLR convention).
    """
    if warmup_iterations > 0 and step < warmup_iterations:
        return (step + 1) / warmup_iterations
    span = max(1, total_iterations - warmup_iterations)
    progress = min(max((step - warmup_iterations) / span, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# -- evaluation ---------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    loss_fn: torch.nn.Module,
) -> dict:
    """Full pass over ``loader``: mean loss + per-class IoU/Dice."""
    was_training = model.training
    model.eval()
    conf = seg_metrics.new_confusion(num_classes, device=device)
    total_loss, batches = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(loss_fn(logits, masks))
        batches += 1
        # confusion_update moves its inputs to conf.device itself; conf already
        # lives on `device`, so this stays on-GPU instead of a full B×K×H×W sync.
        conf = seg_metrics.confusion_update(conf, logits, masks)
    if was_training:
        model.train()

    iou = seg_metrics.iou_per_class(conf)
    dice = seg_metrics.dice_per_class(conf)
    damage = [v for c, v in iou.items() if c != 0 and not math.isnan(v)]
    return {
        "val_loss": total_loss / max(1, batches),
        "mean_iou_damage": sum(damage) / len(damage) if damage else 0.0,
        "iou_per_class": iou,
        "dice_per_class": dice,
    }


# -- tracking -----------------------------------------------------------------


def _flatten(config: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)
    return flat


class RunTracker:
    """Metric sink: JSONL always (``history.jsonl``), MLflow when available.

    MLflow is used only if ``config['mlflow']['tracking_uri']`` is set *and*
    the package imports; every MLflow call is wrapped so a broken tracking
    server can never interrupt a run.
    """

    def __init__(self, config: dict, output_dir: Path) -> None:
        self.history_path = output_dir / "history.jsonl"
        self._fh = self.history_path.open("w", encoding="utf-8")
        self._mlflow = None
        mlflow_cfg = config.get("mlflow") or {}
        if mlflow_cfg.get("tracking_uri"):
            try:
                import mlflow

                mlflow.set_tracking_uri(str(mlflow_cfg["tracking_uri"]))
                mlflow.set_experiment(mlflow_cfg.get("experiment") or "memoire-seg")
                mlflow.start_run(run_name=mlflow_cfg.get("run_name"))
                mlflow.log_params(_flatten(config))
                self._mlflow = mlflow
            except Exception:
                logger.warning("MLflow unavailable, falling back to JSONL only", exc_info=True)
                self._mlflow = None

    def log(self, event: dict) -> None:
        self._fh.write(json.dumps(event, default=str) + "\n")
        self._fh.flush()
        if self._mlflow is None:
            return
        try:
            step = int(event.get("iteration", 0))
            kind = event.get("type", "train")
            for key, value in event.items():
                if key in ("iteration", "type"):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self._mlflow.log_metric(f"{kind}/{key}", float(value), step=step)
        except Exception:
            logger.debug("MLflow metric logging failed", exc_info=True)

    def log_artifact(self, path: Path) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception:
            logger.debug("MLflow artifact logging failed", exc_info=True)

    def close(self) -> None:
        self._fh.close()
        if self._mlflow is None:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            logger.debug("MLflow run teardown failed", exc_info=True)


# -- checkpoints --------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict,
    iteration: int,
    best_val_iou: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "iteration": iteration,
            "best_val_iou": best_val_iou,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)


# -- training loop ------------------------------------------------------------


def train(config: dict) -> dict:
    """Run one seeded training; return a summary dict (also saved as JSON)."""
    config = resolve_config(config)
    seed = int(config["seed"])
    set_global_seed(seed)
    device = resolve_device(config["device"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    # Data. A dedicated generator feeds the flip augmentation; the loader
    # generator drives shuffling. Both derive from the config seed.
    augment_generator = torch.Generator().manual_seed(seed + 1)
    train_dataset = build_dataset(config, "train", generator=augment_generator)
    val_dataset = build_dataset(config, "val")

    subset_n = config.get("subset_n_images")
    density_bucket = config.get("density_bucket")
    if density_bucket is not None:
        if subset_n is None:
            raise ValueError("density_bucket requires subset_n_images (volume held constant)")
        density_min, density_max = density_bucket
        indices = density_indices(
            flat_instance_counts(train_dataset), subset_n, seed, int(density_min),
            None if density_max is None else int(density_max),
        )
        train_dataset = Subset(train_dataset, indices)
    elif subset_n is not None:
        train_dataset = Subset(
            train_dataset, subset_indices(len(train_dataset), subset_n, seed)
        )
    if len(train_dataset) == 0:
        raise ValueError("empty train split")

    loader_generator = torch.Generator().manual_seed(seed)
    num_workers = int(config["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
        worker_init_fn=seed_worker,
        drop_last=False,
        pin_memory=device.type == "cuda",
        # Keep workers alive across epochs: re-forking would reset the copied
        # augmentation generator and replay the same flip sequence each epoch.
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )

    # Model / optimisation.
    num_classes = num_classes_for(config)
    model = UNet(
        in_channels=int(config["in_channels"]),
        num_classes=num_classes,
        base_channels=int(config["base_channels"]),
        depth=int(config["depth"]),
        gn_groups=int(config["gn_groups"]),
    ).to(device)
    loss_fn = DiceCELoss(
        ce_weight=float(config["ce_weight"]), dice_weight=float(config["dice_weight"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    total_iterations = int(config["iterations"])
    warmup_iterations = int(config["warmup_iterations"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup_iterations, total_iterations)
    )
    val_every = max(1, int(config["val_every"]))

    tracker = RunTracker(config, output_dir)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    train_losses: list[float] = []
    best_val_iou = -math.inf
    best_iteration = 0
    last_val: dict | None = None
    iteration = 0
    start_time = time.time()

    try:
        model.train()
        while iteration < total_iterations:
            for images, masks in train_loader:
                if iteration >= total_iterations:
                    break
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, masks)
                loss.backward()
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                iteration += 1

                loss_value = float(loss.detach())
                train_losses.append(loss_value)
                tracker.log(
                    {
                        "type": "train",
                        "iteration": iteration,
                        "loss": loss_value,
                        "lr": lr_used,
                    }
                )

                if iteration % val_every == 0 or iteration == total_iterations:
                    val = evaluate(model, val_loader, device, num_classes, loss_fn)
                    last_val = val
                    tracker.log(
                        {
                            "type": "val",
                            "iteration": iteration,
                            "val_loss": val["val_loss"],
                            "mean_iou_damage": val["mean_iou_damage"],
                            "iou_per_class": val["iou_per_class"],
                            "dice_per_class": val["dice_per_class"],
                        }
                    )
                    if val["mean_iou_damage"] > best_val_iou:
                        best_val_iou = val["mean_iou_damage"]
                        best_iteration = iteration
                        save_checkpoint(best_path, model, config, iteration, best_val_iou)
                    save_checkpoint(last_path, model, config, iteration, best_val_iou)

        summary = {
            "iterations": iteration,
            "seed": seed,
            "device": str(device),
            "num_classes": num_classes,
            "num_train_images": len(train_dataset),
            "num_val_images": len(val_dataset),
            "train_losses": train_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "best_val_iou": None if last_val is None else best_val_iou,
            "best_iteration": best_iteration,
            "last_val": last_val,
            "elapsed_seconds": time.time() - start_time,
            "output_dir": str(output_dir),
            "checkpoints": {"last": str(last_path), "best": str(best_path)},
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        tracker.log_artifact(output_dir / "summary.json")
        return summary
    finally:
        tracker.close()
