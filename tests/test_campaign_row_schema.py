"""Les deux chemins de campagne doivent écrire le MÊME schéma de ligne.

`scripts/run_volume_curve.py` (via memoire.training.volume_curve.run_campaign) et
le DAG Airflow écrivent tous deux `runs/volume-curve/volume_curve.csv`, avec le
même `write_csv` — dont l'en-tête est déduit des clés de la première ligne. Un
consommateur du CSV (scripts/make_result_figures.py, une figure du chapitre 8)
ne peut pas savoir lequel des deux chemins l'a produit : si leurs clés
divergent, la lecture casse ou, pire, tombe silencieusement sur une colonne
absente.

Ce test compare les clés par analyse syntaxique plutôt qu'à l'exécution : le
DAG importe `airflow`, absent de l'environnement de test (et masqué ici par le
répertoire `airflow/` du dépôt lui-même).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = REPO_ROOT / "airflow" / "dags" / "memoire_pipeline.py"
VOLUME_CURVE_PATH = REPO_ROOT / "src" / "memoire" / "training" / "volume_curve.py"


def _returned_dict_keys(path: Path, func_name: str) -> list[str]:
    """Clés du littéral dict retourné par ``func_name`` dans ``path``.

    Cherche le dernier ``return {...}`` de la fonction — dans run_campaign le
    dict est construit dans un ``rows.append({...})``, on accepte donc aussi
    cette forme.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue
        for sub in ast.walk(node):
            literal = None
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                literal = sub.value
            elif (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "append"
                and sub.args
                and isinstance(sub.args[0], ast.Dict)
            ):
                literal = sub.args[0]
            if literal is not None:
                return [
                    k.value for k in literal.keys if isinstance(k, ast.Constant)
                ]
    raise AssertionError(f"pas de littéral dict trouvé dans {func_name} ({path})")


def test_airflow_and_cli_write_the_same_columns():
    cli = _returned_dict_keys(VOLUME_CURVE_PATH, "run_campaign")
    dag = _returned_dict_keys(DAG_PATH, "train_run")

    assert dag == cli, (
        "le DAG Airflow et le CLI écrivent le même volume_curve.csv : leurs "
        f"colonnes doivent coïncider.\n  CLI     : {cli}\n  Airflow : {dag}"
    )


def test_the_volume_column_is_named_subset_n_images():
    """Le nom de colonne fait partie du contrat : make_result_figures.py et
    toute figure du chapitre 8 s'y réfèrent."""
    for path, func in ((VOLUME_CURVE_PATH, "run_campaign"), (DAG_PATH, "train_run")):
        keys = _returned_dict_keys(path, func)
        assert "subset_n_images" in keys, f"{func} n'écrit pas subset_n_images : {keys}"
        assert "point" not in keys, f"{func} écrit encore l'ancienne colonne 'point' : {keys}"


def test_density_campaign_adds_only_the_bucket_column():
    """L'axe densité réutilise le schéma volume, préfixé de density_bucket."""
    volume = _returned_dict_keys(VOLUME_CURVE_PATH, "run_campaign")
    density = _returned_dict_keys(VOLUME_CURVE_PATH, "run_density_campaign")

    assert density[0] == "density_bucket"
    assert density[1:] == volume


@pytest.mark.parametrize("column", ["best_iteration", "num_train_images", "best_val_iou"])
def test_airflow_does_not_drop_columns(column):
    """Airflow omettait best_iteration : un CSV produit par le DAG perdait la
    colonne pour tout le monde, puisque write_csv déduit l'en-tête de la
    première ligne."""
    assert column in _returned_dict_keys(DAG_PATH, "train_run")
