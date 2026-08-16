"""Harmonisation des taxonomies de classes entre sources (mémoire, chap. 4.5).

L'arbitrage (fusion, conservation spécifique, exclusion) vit exclusivement
dans ``configs/taxonomy.yaml`` : documenté et modifiable sans toucher au code.
Ce module se contente de charger, valider et servir ce fichier.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_DECISIONS = frozenset({"map", "keep_specific", "exclude"})
VALID_SIZES = frozenset({"large", "fine"})

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class TaxonomyError(Exception):
    """Erreur de base du module taxonomie."""


class TaxonomyConfigError(TaxonomyError):
    """Fichier de taxonomie invalide (structure, décision inconnue, etc.)."""


class UnknownSourceError(TaxonomyError):
    """Source non déclarée dans la taxonomie."""


class UnknownClassError(TaxonomyError):
    """Classe source non arbitrée dans la taxonomie."""


class ExcludedClassError(TaxonomyError):
    """Classe source explicitement exclue : pas de classe canonique."""


class CoverageError(TaxonomyError):
    """Des classes observées dans les données ne sont pas arbitrées."""


@dataclass(frozen=True)
class CanonicalClass:
    """Classe canonique : nom snake_case, description fr, groupe large.

    ``size`` (chap. 6.4, optionnel) : regroupement à 2 valeurs {large, fine}
    utilisé par le mode multiclasse d'entraînement (background + larges +
    fines) — pas les 13 classes canoniques, intraitables sous la contrainte
    from-scratch stricte. ``None`` tant que non renseigné dans le yaml ;
    ``Taxonomy.size()`` lève si consultée sans valeur.
    """

    name: str
    description: str
    group: str
    size: str | None = None


@dataclass(frozen=True)
class ClassMapping:
    """Arbitrage d'une classe source : cible canonique, décision, justification."""

    source_class: str
    canonical: str | None
    decision: str
    note: str


class Taxonomy:
    """Taxonomie harmonisée : classes canoniques + arbitrages par source."""

    def __init__(
        self,
        canonical_classes: Mapping[str, CanonicalClass],
        mappings: Mapping[str, Mapping[str, ClassMapping]],
    ) -> None:
        self._canonical_classes = dict(canonical_classes)
        self._mappings = {src: dict(classes) for src, classes in mappings.items()}
        self._validate()

    def _validate(self) -> None:
        for name, cls in self._canonical_classes.items():
            if not _SNAKE_CASE_RE.match(name):
                raise TaxonomyConfigError(
                    f"Canonical class name {name!r} is not snake_case"
                )
            if not cls.description or not cls.group:
                raise TaxonomyConfigError(
                    f"Canonical class {name!r} must have a non-empty "
                    "'description' and 'group'"
                )
            if cls.size is not None and cls.size not in VALID_SIZES:
                raise TaxonomyConfigError(
                    f"Canonical class {name!r}: invalid size {cls.size!r}; "
                    f"expected one of {sorted(VALID_SIZES)} or unset"
                )
        for source, classes in self._mappings.items():
            for source_class, mapping in classes.items():
                where = f"{source}/{source_class}"
                if mapping.decision not in VALID_DECISIONS:
                    raise TaxonomyConfigError(
                        f"Invalid decision {mapping.decision!r} for {where}; "
                        f"expected one of {sorted(VALID_DECISIONS)}"
                    )
                if not mapping.note:
                    raise TaxonomyConfigError(
                        f"Missing 'note' for {where}: every arbitration must be "
                        "documented in the yaml"
                    )
                if mapping.decision == "exclude":
                    if mapping.canonical is not None:
                        raise TaxonomyConfigError(
                            f"Excluded class {where} must have canonical: null"
                        )
                else:
                    if mapping.canonical is None:
                        raise TaxonomyConfigError(
                            f"Missing 'canonical' for {where} "
                            f"(decision {mapping.decision!r})"
                        )
                    if mapping.canonical not in self._canonical_classes:
                        raise TaxonomyConfigError(
                            f"{where} maps to unknown canonical class "
                            f"{mapping.canonical!r}"
                        )

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self._mappings)

    @property
    def canonical_classes(self) -> dict[str, CanonicalClass]:
        return dict(self._canonical_classes)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(self._canonical_classes)

    def group(self, canonical_name: str) -> str:
        """Groupe large d'une classe canonique."""
        try:
            return self._canonical_classes[canonical_name].group
        except KeyError:
            raise UnknownClassError(
                f"Unknown canonical class {canonical_name!r}"
            ) from None

    def size(self, canonical_name: str) -> str:
        """Regroupement large/fine (chap. 6.4) d'une classe canonique.

        Lève ``UnknownClassError`` si la classe est inconnue, ou
        ``TaxonomyConfigError`` si elle est connue mais sans ``size`` renseigné
        dans le yaml (jamais None silencieux pour l'entraînement multiclasse).
        """
        try:
            cls = self._canonical_classes[canonical_name]
        except KeyError:
            raise UnknownClassError(f"Unknown canonical class {canonical_name!r}") from None
        if cls.size is None:
            raise TaxonomyConfigError(
                f"Canonical class {canonical_name!r} has no 'size' (large/fine) "
                "set in the taxonomy — required for multiclass training"
            )
        return cls.size

    def mapping(self, source: str, source_class: str) -> ClassMapping:
        """Arbitrage complet (canonical, decision, note) d'une classe source."""
        try:
            classes = self._mappings[source]
        except KeyError:
            raise UnknownSourceError(
                f"Unknown source {source!r}; known sources: {sorted(self._mappings)}"
            ) from None
        try:
            return classes[source_class]
        except KeyError:
            raise UnknownClassError(
                f"Unknown class {source_class!r} for source {source!r}; "
                f"known classes: {sorted(classes)}"
            ) from None

    def decision(self, source: str, source_class: str) -> str:
        return self.mapping(source, source_class).decision

    def canonical(self, source: str, source_class: str) -> str:
        """Classe canonique d'une classe source ; lève si inconnue ou exclue."""
        mapping = self.mapping(source, source_class)
        if mapping.decision == "exclude":
            raise ExcludedClassError(
                f"Class {source_class!r} from source {source!r} is excluded: "
                f"{mapping.note}"
            )
        assert mapping.canonical is not None  # guaranteed by _validate
        return mapping.canonical

    def validate_coverage(self, source: str, observed_classes: Iterable[str]) -> None:
        """Lève CoverageError si une classe observée n'est pas arbitrée."""
        try:
            classes = self._mappings[source]
        except KeyError:
            raise UnknownSourceError(
                f"Unknown source {source!r}; known sources: {sorted(self._mappings)}"
            ) from None
        unmapped = sorted(set(observed_classes) - set(classes))
        if unmapped:
            raise CoverageError(
                f"Observed classes for source {source!r} without an arbitration "
                f"in the taxonomy: {unmapped}"
            )


def _require_str(value: object, where: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaxonomyConfigError(f"{where}: field {field!r} must be a non-empty string")
    return value.strip()


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Charge et valide ``configs/taxonomy.yaml``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise TaxonomyConfigError(f"{path}: top level must be a mapping")
    for key in ("canonical_classes", "sources"):
        if not isinstance(raw.get(key), dict) or not raw[key]:
            raise TaxonomyConfigError(f"{path}: missing or empty section {key!r}")

    canonical_classes: dict[str, CanonicalClass] = {}
    for name, spec in raw["canonical_classes"].items():
        where = f"canonical_classes/{name}"
        if not isinstance(spec, dict):
            raise TaxonomyConfigError(f"{where}: must be a mapping")
        size = spec.get("size")
        canonical_classes[str(name)] = CanonicalClass(
            name=str(name),
            description=_require_str(spec.get("description"), where, "description"),
            group=_require_str(spec.get("group"), where, "group"),
            size=_require_str(size, where, "size") if size is not None else None,
        )

    mappings: dict[str, dict[str, ClassMapping]] = {}
    for source, classes in raw["sources"].items():
        if not isinstance(classes, dict) or not classes:
            raise TaxonomyConfigError(f"sources/{source}: must be a non-empty mapping")
        per_source: dict[str, ClassMapping] = {}
        for source_class, spec in classes.items():
            where = f"sources/{source}/{source_class}"
            if not isinstance(spec, dict):
                raise TaxonomyConfigError(f"{where}: must be a mapping")
            canonical = spec.get("canonical")
            if canonical is not None:
                canonical = _require_str(canonical, where, "canonical")
            per_source[str(source_class)] = ClassMapping(
                source_class=str(source_class),
                canonical=canonical,
                decision=_require_str(spec.get("decision"), where, "decision"),
                note=_require_str(spec.get("note"), where, "note"),
            )
        mappings[str(source)] = per_source

    return Taxonomy(canonical_classes, mappings)
