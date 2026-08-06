"""Splits anti-fuite par ``group_id`` (jamais par image).

Le split est déterministe : l'ordre d'attribution repose sur un hash SHA-256
seedé du ``group_id``, pas sur ``random.shuffle``. La stratification approche
l'équilibre des densités d'instances (instances par image, par groupe).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

DEFAULT_FRACTIONS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def _stable_hash(key: str, seed: int) -> int:
    """Hash entier stable (indépendant de PYTHONHASHSEED) d'une clé seedée."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _group_by_id(records: Sequence[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["group_id"]].append(rec)
    return dict(groups)


def _validate_fractions(fractions: Mapping[str, float]) -> None:
    if not fractions:
        raise ValueError("fractions must not be empty")
    if any(f < 0 for f in fractions.values()):
        raise ValueError(f"fractions must be non-negative, got {dict(fractions)}")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")


def _density_strata(
    groups: Mapping[str, list[dict]], gids: Sequence[str], n_strata: int
) -> list[list[str]]:
    """Partitionne les groupes en strates de densité (instances/image) croissante."""
    if not gids:
        return []

    def density(gid: str) -> float:
        recs = groups[gid]
        return sum(len(r["instances"]) for r in recs) / len(recs)

    ordered = sorted(gids, key=lambda g: (density(g), g))
    n_strata = max(1, min(n_strata, len(ordered)))
    strata: list[list[str]] = [[] for _ in range(n_strata)]
    for i, gid in enumerate(ordered):
        strata[i * n_strata // len(ordered)].append(gid)
    return [s for s in strata if s]


def make_group_split(
    records: Sequence[dict],
    fractions: Mapping[str, float] | None = None,
    seed: int = 0,
    keep_official: bool = False,
    n_strata: int = 3,
) -> dict[str, list[dict]]:
    """Répartit les records en splits par ``group_id``, jamais par image.

    - Déterministe à seed fixé (hash stable des ``group_id``).
    - ``keep_official=True`` : les groupes portant un ``split_hint`` officiel
      (CarDD) restent dans leur split ; seuls les autres sont répartis.
    - Stratification approximative par densité d'instances par groupe.

    Retourne ``{nom_split: [records]}`` ; ``check_no_leak`` est appelé avant
    de retourner.
    """
    fractions = dict(fractions) if fractions is not None else dict(DEFAULT_FRACTIONS)
    _validate_fractions(fractions)

    groups = _group_by_id(records)
    split_names = list(fractions)
    assignment: dict[str, str] = {}
    unassigned: list[str] = []

    for gid in groups:
        hints = {rec["split_hint"] for rec in groups[gid]}
        if keep_official and len(hints) == 1:
            hint = next(iter(hints))
            if hint in fractions:
                assignment[gid] = hint
                continue
        elif keep_official and len(hints) > 1 and hints - {None}:
            raise ValueError(f"group {gid!r} has conflicting official split hints: {hints}")
        unassigned.append(gid)

    total_images = len(records)
    assigned_images = {name: 0 for name in split_names}
    for gid, name in assignment.items():
        assigned_images[name] += len(groups[gid])

    # Capacité restante par split (en images), pour tendre vers les fractions
    # globales même quand une partie des groupes est déjà fixée officiellement.
    capacity = {
        name: max(fractions[name] * total_images - assigned_images[name], 0.0)
        for name in split_names
    }
    capacity_total = sum(capacity.values())
    if capacity_total <= 0:
        ratios = {name: fractions[name] for name in split_names}
    else:
        ratios = {name: capacity[name] / capacity_total for name in split_names}

    for stratum in _density_strata(groups, unassigned, n_strata):
        ordered = sorted(stratum, key=lambda g: (_stable_hash(g, seed), g))
        stratum_images = sum(len(groups[g]) for g in ordered)
        boundaries: list[tuple[str, float]] = []
        cum = 0.0
        for name in split_names:
            cum += ratios[name] * stratum_images
            boundaries.append((name, cum))
        boundaries[-1] = (boundaries[-1][0], float(stratum_images))

        placed = 0
        idx = 0
        for gid in ordered:
            while idx < len(boundaries) - 1 and placed >= boundaries[idx][1]:
                idx += 1
            assignment[gid] = boundaries[idx][0]
            placed += len(groups[gid])

    splits: dict[str, list[dict]] = {name: [] for name in split_names}
    for rec in records:
        splits[assignment[rec["group_id"]]].append(rec)

    check_no_leak(splits)
    return splits


def check_no_leak(splits: Mapping[str, Sequence[dict]]) -> None:
    """Lève ``AssertionError`` si un ``group_id`` apparaît dans deux splits.

    Garde-fou bloquant (branché en CI) contre la fuite véhicule/sinistre
    entre train, val et test.
    """
    first_seen: dict[str, str] = {}
    offenders: dict[str, set[str]] = defaultdict(set)
    for name, recs in splits.items():
        for rec in recs:
            gid = rec["group_id"]
            if gid in first_seen and first_seen[gid] != name:
                offenders[gid].update({first_seen[gid], name})
            else:
                first_seen.setdefault(gid, name)
    if offenders:
        detail = ", ".join(
            f"{gid!r} in {sorted(names)}" for gid, names in sorted(offenders.items())
        )
        raise AssertionError(f"group leakage across splits: {detail}")
