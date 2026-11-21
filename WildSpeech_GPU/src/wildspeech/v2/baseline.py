"""The strongest validated tabular comparator.

V2 compares every new representation against the best *tabular* model that P10 (or P01) validated — not only
against the weaker screening-only ridge. Families allowed in the comparator: SCREENING, RECORDING, BEHAVIOR,
ITEM, LINGUISTIC (no acoustic family). The choice is read from runtime artefacts; when none exist the configured
fallback is used and recorded as such.
"""
from __future__ import annotations

import pandas as pd

from . import TABULAR_FAMILIES


def _tabular_only(fams: str) -> bool:
    return all(f in TABULAR_FAMILIES for f in str(fams).split("+"))


def strongest_tabular(ctx) -> dict:
    ev = ctx.cfg.get_path("v2.evaluation", {}) or {}
    choice = ev.get("baseline", "auto")
    if isinstance(choice, dict):
        return {"model": choice["model"], "families": list(choice["families"]), "source": "config", "name": "V2_BASELINE"}
    candidates = []
    bench = ctx.read_stage_table("P10", "model_benchmark")
    if bench is not None and not bench.empty:
        b = bench[(bench["status"] == "ok") & bench["families"].map(_tabular_only)]
        candidates += [{"model": r.model, "families": str(r.families).split("+"), "r2_mean": float(r.r2_mean),
                        "source": f"P10:{r.name}", "v1_name": r.name} for r in b.itertuples()]
    p01 = ctx.read_stage_table("P01", "tabular_baselines")
    if p01 is not None and not p01.empty and {"model", "families", "r2_mean"} <= set(p01.columns):
        b = p01[p01["families"].map(_tabular_only) & (p01.get("status", "ok") == "ok") & (p01["model"] != "MEAN")]
        candidates += [{"model": r.model, "families": str(r.families).split("+"), "r2_mean": float(r.r2_mean),
                        "source": f"P01:{getattr(r, 'name', '')}", "v1_name": getattr(r, "name", None)} for r in b.itertuples()]
    if candidates:
        best = max(candidates, key=lambda c: c["r2_mean"])
        best["name"] = "V2_BASELINE"
        return best
    fb = ev.get("baseline_fallback", {"model": "GAM", "families": ["SCREENING", "BEHAVIOR", "ITEM"]})
    return {"model": fb["model"], "families": list(fb["families"]), "source": "fallback(no P10/P01 artefact)",
            "name": "V2_BASELINE", "r2_mean": None}


def conservative(ctx) -> dict:
    c = ctx.cfg.get_path("v2.evaluation.conservative", {"model": "RIDGE", "families": ["SCREENING", "RECORDING"]})
    return {"model": c["model"], "families": list(c["families"]), "name": "V2_CONSERVATIVE", "source": "config"}


def baseline_table(ctx) -> pd.DataFrame:
    s, c = strongest_tabular(ctx), conservative(ctx)
    return pd.DataFrame([{"role": "strongest_tabular", **{k: ("+".join(v) if isinstance(v, list) else v) for k, v in s.items()}},
                         {"role": "conservative", **{k: ("+".join(v) if isinstance(v, list) else v) for k, v in c.items()}}])
