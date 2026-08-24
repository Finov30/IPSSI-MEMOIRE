"""Tests de scripts/make_result_figures.py — l'agrégation des runs de campagne
en figures et tableaux du chapitre 8.

Le point sensible n'est pas le rendu matplotlib mais la collecte : la source de
vérité est le couple ``config.json`` + ``summary.json`` de chaque run, jamais le
``volume_curve.csv`` agrégé (qui n'est écrit qu'à la toute fin d'une campagne).
Une run interrompue ne doit donc jamais être comptée, et les trois axes doivent
rester séparés — un mélange silencieux entre axe volume et axe densité
fausserait les deux courbes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_result_figures.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_result_figures", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mrf = _load_script()


def _write_run(
    root: Path,
    campagne: str,
    name: str,
    *,
    seed: int = 42,
    n_train: int = 500,
    iou: float = 0.4,
    model: str = "unet",
    copy_paste: bool = False,
    density_bucket: list | None = None,
    with_summary: bool = True,
) -> Path:
    run_dir = root / campagne / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "model": model,
                "copy_paste": copy_paste,
                "density_bucket": density_bucket,
                "subset_n_images": n_train,
                "corpus": ["a.json", "b.json", "c.json"],
                "iterations": 30000,
            }
        ),
        encoding="utf-8",
    )
    if with_summary:
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "iterations": 30000,
                    "seed": seed,
                    "num_train_images": n_train,
                    "best_val_iou": iou,
                    "best_iteration": 26000,
                    "final_train_loss": 0.3,
                    "elapsed_seconds": 3600.0,
                }
            ),
            encoding="utf-8",
        )
    return run_dir


def test_collect_ignores_runs_without_summary(tmp_path):
    """Une run interrompue (pas de summary.json) n'est pas un résultat."""
    _write_run(tmp_path, "volume-curve", "n500_seed42")
    _write_run(tmp_path, "volume-curve", "n500_seed43", seed=43, with_summary=False)

    df = mrf.collect_runs(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["seed"] == 42


def test_collect_reads_config_and_summary(tmp_path):
    _write_run(
        tmp_path, "ablation-baseline", "n2800_seed44",
        seed=44, n_train=2800, iou=0.33, model="baseline",
    )

    row = mrf.collect_runs(tmp_path).iloc[0]

    assert row["campagne"] == "ablation-baseline"
    assert row["model"] == "baseline"
    assert row["num_train_images"] == 2800
    assert row["best_val_iou"] == pytest.approx(0.33)
    assert row["density_bucket"] is None


@pytest.mark.parametrize(
    ("bucket", "expected"),
    [(None, None), ([1, 1], "1"), ([2, 2], "2"), ([4, None], "4+"), ([2, 5], "2-5")],
)
def test_bucket_label(bucket, expected):
    assert mrf._bucket_label(bucket) == expected


def test_aggregate_reports_seed_count_and_spread(tmp_path):
    for seed, iou in ((42, 0.30), (43, 0.34), (44, 0.32)):
        _write_run(tmp_path, "volume-curve", f"n500_seed{seed}", seed=seed, iou=iou)

    agg = mrf.aggregate(mrf.collect_runs(tmp_path), "num_train_images")

    assert len(agg) == 1
    assert agg.iloc[0]["n_seeds"] == 3
    assert agg.iloc[0]["iou_moyenne"] == pytest.approx(0.32)
    assert agg.iloc[0]["iou_min"] == pytest.approx(0.30)
    assert agg.iloc[0]["iou_max"] == pytest.approx(0.34)


def test_density_runs_are_excluded_from_the_volume_curve(tmp_path):
    """Les deux axes ne doivent jamais se mélanger (chap. 7.1)."""
    _write_run(tmp_path, "volume-curve", "n500_seed42", iou=0.40)
    _write_run(
        tmp_path, "density-curve", "d4_seed42",
        iou=0.10, density_bucket=[4, None],
    )
    out = tmp_path / "figures"
    out.mkdir()

    df = mrf.collect_runs(tmp_path)
    assert mrf.fig_volume_curve(df, out) is not None

    import pandas as pd

    table = pd.read_csv(out / "tableau-8-1-courbe-volume.csv")
    assert len(table) == 1
    # La run de densité (IoU 0.10) ne doit pas tirer la moyenne du point volume.
    assert table.iloc[0]["iou_moyenne"] == pytest.approx(0.40)


def test_ablation_reference_comes_from_the_matching_volume_point(tmp_path):
    """La référence de H3 est reprise de la courbe de volume au même ancrage,
    jamais rejouée — et jamais prise à un autre volume (chap. 7.6)."""
    _write_run(tmp_path, "volume-curve", "n2800_seed42", n_train=2800, iou=0.41)
    _write_run(tmp_path, "volume-curve", "n500_seed42", n_train=500, iou=0.20)
    _write_run(
        tmp_path, "ablation-baseline", "n2800_seed42",
        n_train=2800, iou=0.33, model="baseline",
    )
    out = tmp_path / "figures"
    out.mkdir()

    assert mrf.fig_ablations(mrf.collect_runs(tmp_path), out) is not None

    import pandas as pd

    table = pd.read_csv(out / "tableau-8-3-ablations.csv").set_index("condition")
    assert set(table.index) == {"Référence (U-Net)", "U-Net sans skips"}
    # 0.41 (ancrage 2800) et non 0.305 (moyenne avec le point 500).
    assert table.loc["Référence (U-Net)", "iou_moyenne"] == pytest.approx(0.41)


def test_figures_are_skipped_when_an_axis_has_no_run(tmp_path):
    _write_run(tmp_path, "volume-curve", "n500_seed42")
    out = tmp_path / "figures"
    out.mkdir()

    df = mrf.collect_runs(tmp_path)

    assert mrf.fig_volume_curve(df, out) is not None
    assert mrf.fig_density_curve(df, out) is None
    assert mrf.fig_ablations(df, out) is None
