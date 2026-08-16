"""Tests de l'harmonisation des taxonomies (configs/taxonomy.yaml)."""

from pathlib import Path

import pytest

from memoire.data.taxonomy import (
    CoverageError,
    ExcludedClassError,
    Taxonomy,
    TaxonomyConfigError,
    UnknownClassError,
    UnknownSourceError,
    load_taxonomy,
)

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "configs" / "taxonomy.yaml"

# Les 21 classes sources réelles, vérifiées par comptage direct sur les
# annotations, et leur classe canonique attendue.
EXPECTED = {
    "vehide": {
        "tray_son": "scratch",
        "mop_lom": "dent",
        "rach": "tear",
        "mat_bo_phan": "missing_part",
        "be_den": "lamp_broken",
        "thung": "puncture",
        "vo_kinh": "glass_shatter",
    },
    "cardd": {
        "dent": "dent",
        "scratch": "scratch",
        "crack": "crack",
        "glass shatter": "glass_shatter",
        "lamp broken": "lamp_broken",
        "tire flat": "tire_flat",
    },
    "hitl": {
        "Scratch": "scratch",
        "Dent": "dent",
        "Broken part": "broken_part",
        "Paint chip": "paint_chip",
        "Missing part": "missing_part",
        "Flaking": "flaking",
        "Corrosion": "corrosion",
        "Cracked": "crack",
    },
}

KEEP_SPECIFIC = {
    ("vehide", "rach"),
    ("vehide", "thung"),
    ("hitl", "Broken part"),
    ("hitl", "Paint chip"),
    ("hitl", "Flaking"),
    ("hitl", "Corrosion"),
}


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_PATH)


def test_all_sources_declared(taxonomy):
    assert set(taxonomy.sources) == {"vehide", "cardd", "hitl"}


def test_source_classes_are_exactly_the_observed_ones(taxonomy):
    for source, classes in EXPECTED.items():
        mapped = {taxonomy.mapping(source, c).source_class for c in classes}
        assert mapped == set(classes)
        # Aucune classe fantôme dans le yaml au-delà des classes observées.
        taxonomy.validate_coverage(source, classes)
        with pytest.raises(UnknownClassError):
            taxonomy.canonical(source, "definitely_not_a_class")


@pytest.mark.parametrize(
    ("source", "source_class", "expected_canonical"),
    [
        (source, source_class, canonical)
        for source, classes in EXPECTED.items()
        for source_class, canonical in classes.items()
    ],
)
def test_canonical_for_every_source_class(taxonomy, source, source_class, expected_canonical):
    assert taxonomy.canonical(source, source_class) == expected_canonical


def test_total_source_class_count(taxonomy):
    assert sum(len(classes) for classes in EXPECTED.values()) == 21


def test_decisions(taxonomy):
    for source, classes in EXPECTED.items():
        for source_class in classes:
            decision = taxonomy.decision(source, source_class)
            expected = "keep_specific" if (source, source_class) in KEEP_SPECIFIC else "map"
            assert decision == expected, (source, source_class)


def test_every_arbitration_is_documented(taxonomy):
    for source, classes in EXPECTED.items():
        for source_class in classes:
            assert taxonomy.mapping(source, source_class).note.strip()


def test_canonical_classes_are_snake_case_with_description_and_group(taxonomy):
    assert len(taxonomy.canonical_names) == 13
    for name, cls in taxonomy.canonical_classes.items():
        assert name == name.lower()
        assert " " not in name
        assert cls.description.strip()
        assert cls.group.strip()
        assert taxonomy.group(name) == cls.group


def test_every_expected_canonical_exists(taxonomy):
    expected_canonicals = {c for classes in EXPECTED.values() for c in classes.values()}
    assert expected_canonicals == set(taxonomy.canonical_names)


# --- size grouping (chap. 6.4: multiclass = background/large/fine) ---

EXPECTED_LARGE = {"broken_part", "glass_shatter", "lamp_broken", "tire_flat", "missing_part"}
EXPECTED_FINE = {
    "scratch", "dent", "crack", "tear", "puncture", "paint_chip", "flaking", "corrosion",
}


def test_every_canonical_class_has_a_size(taxonomy):
    for name in taxonomy.canonical_names:
        assert taxonomy.size(name) in {"large", "fine"}


def test_size_grouping_matches_the_documented_split(taxonomy):
    assert EXPECTED_LARGE | EXPECTED_FINE == set(taxonomy.canonical_names)
    assert EXPECTED_LARGE.isdisjoint(EXPECTED_FINE)
    for name in EXPECTED_LARGE:
        assert taxonomy.size(name) == "large"
    for name in EXPECTED_FINE:
        assert taxonomy.size(name) == "fine"


def test_size_raises_on_unknown_class(taxonomy):
    with pytest.raises(UnknownClassError):
        taxonomy.size("definitely_not_a_class")


def test_size_raises_when_unset(tmp_path):
    tax = load_taxonomy(_write(tmp_path, MINIMAL_YAML))
    with pytest.raises(TaxonomyConfigError, match="size"):
        tax.size("scratch")


def test_load_rejects_invalid_size(tmp_path):
    bad = MINIMAL_YAML.replace("group: surface", "group: surface\n    size: medium")
    with pytest.raises(TaxonomyConfigError):
        load_taxonomy(_write(tmp_path, bad))


def test_shared_classes_bridge_at_least_two_sources(taxonomy):
    reachable_from = {}
    for source, classes in EXPECTED.items():
        for source_class in classes:
            if taxonomy.decision(source, source_class) == "map":
                canonical = taxonomy.canonical(source, source_class)
                reachable_from.setdefault(canonical, set()).add(source)
    # tire_flat est la seule classe "map" mono-source (référentiel CarDD).
    mono = {c for c, sources in reachable_from.items() if len(sources) == 1}
    assert mono == {"tire_flat"}


def test_validate_coverage_accepts_full_observed_sets(taxonomy):
    for source, classes in EXPECTED.items():
        taxonomy.validate_coverage(source, list(classes))
        taxonomy.validate_coverage(source, [])  # sous-ensemble vide : couvert


def test_validate_coverage_raises_on_unknown_class(taxonomy):
    with pytest.raises(CoverageError) as excinfo:
        taxonomy.validate_coverage("vehide", ["tray_son", "mystery_damage"])
    assert "mystery_damage" in str(excinfo.value)
    assert "tray_son" not in str(excinfo.value)


def test_validate_coverage_raises_on_unknown_source(taxonomy):
    with pytest.raises(UnknownSourceError):
        taxonomy.validate_coverage("crashcar", ["scratch"])


def test_canonical_raises_on_unknown_source_and_class(taxonomy):
    with pytest.raises(UnknownSourceError):
        taxonomy.canonical("crashcar", "scratch")
    with pytest.raises(UnknownClassError) as excinfo:
        taxonomy.canonical("cardd", "glass_shatter")  # nom canonique != nom source
    assert "glass_shatter" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Comportements sur yaml synthétiques (exclusion, erreurs de config)
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
canonical_classes:
  scratch:
    description: "Rayure."
    group: surface
sources:
  demo:
    Scratch:
      canonical: scratch
      decision: map
      note: "Identité."
    Shadow:
      canonical: null
      decision: exclude
      note: "Ombre portée, pas un dommage."
"""


def _write(tmp_path, text):
    path = tmp_path / "taxonomy.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_excluded_class_is_covered_but_has_no_canonical(tmp_path):
    tax = load_taxonomy(_write(tmp_path, MINIMAL_YAML))
    tax.validate_coverage("demo", ["Scratch", "Shadow"])
    assert tax.decision("demo", "Shadow") == "exclude"
    with pytest.raises(ExcludedClassError):
        tax.canonical("demo", "Shadow")


def test_load_rejects_unknown_decision(tmp_path):
    bad = MINIMAL_YAML.replace("decision: map", "decision: merge")
    with pytest.raises(TaxonomyConfigError):
        load_taxonomy(_write(tmp_path, bad))


def test_load_rejects_mapping_to_unknown_canonical(tmp_path):
    bad = MINIMAL_YAML.replace("canonical: scratch", "canonical: scrach")
    with pytest.raises(TaxonomyConfigError):
        load_taxonomy(_write(tmp_path, bad))


def test_load_rejects_missing_note(tmp_path):
    bad = MINIMAL_YAML.replace('note: "Identité."', 'note: ""')
    with pytest.raises(TaxonomyConfigError):
        load_taxonomy(_write(tmp_path, bad))


def test_load_rejects_excluded_class_with_canonical(tmp_path):
    bad = MINIMAL_YAML.replace("canonical: null", "canonical: scratch")
    with pytest.raises(TaxonomyConfigError):
        load_taxonomy(_write(tmp_path, bad))


def test_load_rejects_non_snake_case_canonical(tmp_path):
    bad = MINIMAL_YAML.replace("  scratch:", "  Scratch Mark:", 1).replace(
        "canonical: scratch", "canonical: Scratch Mark"
    )
    with pytest.raises(TaxonomyConfigError):
        load_taxonomy(_write(tmp_path, bad))
