"""Corrupted/unreadable image filtering, shared across all three loaders.

A file existing on disk with valid dimensions in its annotation metadata
does not guarantee it actually decodes: found empirically — 16 VehiDE JPEGs
(0.11%) raise ``OSError`` (truncated / broken data stream) only when fully
decoded, not when just opened for header/size. CarDD and HitL never open the
image file at load time at all (dimensions come from their own JSON/ann
metadata), so the same risk applies to them structurally even though the
current corpus has zero corrupted files there. Filtered here — once, after
loading, for every source — rather than inside each format-specific loader:
dropped and counted, never silently (docs/CONVENTIONS.md).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PIL import Image

logger = logging.getLogger(__name__)


def is_readable(path: str) -> bool:
    """True if the image at ``path`` fully decodes (not just a valid header)."""
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except OSError:
        return False


def filter_unreadable(records: Sequence[dict], source: str) -> tuple[list[dict], int, int]:
    """Drop records whose image file doesn't decode.

    Returns ``(kept, n_images_dropped, n_instances_dropped)`` — the instance
    count is reported too so a corrupted image's discarded annotations don't
    show up as an unexplained instance-count gap in the corpus report.
    """
    kept: list[dict] = []
    dropped: list[str] = []
    n_instances_dropped = 0
    for rec in records:
        if is_readable(rec["file_path"]):
            kept.append(rec)
        else:
            dropped.append(rec["file_path"])
            n_instances_dropped += len(rec.get("instances", []))
    if dropped:
        preview = ", ".join(dropped[:5])
        more = f", ... ({len(dropped) - 5} more)" if len(dropped) > 5 else ""
        logging.getLogger(f"memoire.data.{source}").warning(
            "%s: skipped %d corrupted/unreadable image file(s) (%d instance(s) with them): %s%s",
            source, len(dropped), n_instances_dropped, preview, more,
        )
    return kept, len(dropped), n_instances_dropped
