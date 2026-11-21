"""P15 — scientific report, feature/hypothesis/experiment/failure registries (assembled from stage outputs)."""
from __future__ import annotations

from ..reporting import build_report


def run(ctx) -> dict:
    return build_report(ctx)
