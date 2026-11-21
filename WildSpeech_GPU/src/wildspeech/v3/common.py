"""Shared start-of-stage logic for P23-P29.

begin(ctx, sid)
    1. lineage report for every declared upstream (hard + soft) -> tables/upstream_lineage.csv
       * stale V3 upstream          -> V3StaleUpstreamError (v3.runtime.enforce_fresh_v3_upstream)
       * stale V1/V2 upstream       -> recorded as a warning; V3 reads their artefacts read-only and never
                                       re-plans them (the cause is audited in P23)
       * hard upstream not completed -> error
    2. frozen V2 validation registry (the same folds as P01-P22)
    3. V3 registries
"""
from __future__ import annotations

import pandas as pd


class V3StaleUpstreamError(RuntimeError):
    pass


def lineage_report(ctx, sid: str) -> pd.DataFrame:
    from ..orchestrator import STAGES, Pipeline

    pipe = Pipeline(ctx)
    spec = STAGES[sid]
    rows = []
    for d in spec.deps + spec.soft:
        st = pipe.status(d)
        status = st.get("status", "never_run")
        stale = pipe.is_stale(d) if status == "completed" else None
        rows.append({"stage": sid, "upstream": d, "kind": "hard" if d in spec.deps else "soft", "status": status,
                     "stale": stale, "recorded_fingerprint": st.get("fingerprint"),
                     "current_fingerprint": pipe.fingerprint(d) if status == "completed" else None,
                     "finished": st.get("finished"), "profile": st.get("profile")})
    return pd.DataFrame(rows)


def archive_previous_outcome(ctx, sid: str) -> None:
    """A NOT_EXECUTED marker from an EARLIER run must not survive a run that now produces results: it is moved
    (never deleted) to tables/_history/not_executed_<timestamp>.csv before the stage starts."""
    import os
    import time
    from pathlib import Path

    d = Path(ctx.stage_dir(sid)) / "tables"
    p = d / "not_executed.csv"
    if p.exists():
        h = d / "_history"
        h.mkdir(parents=True, exist_ok=True)
        os.replace(p, h / f"not_executed_{time.strftime('%Y%m%dT%H%M%S', time.localtime(p.stat().st_mtime))}.csv")
    o = Path(ctx.stage_dir(sid)) / "_outcome.json"
    if o.exists():
        h = d / "_history"
        h.mkdir(parents=True, exist_ok=True)
        os.replace(o, h / f"outcome_{time.strftime('%Y%m%dT%H%M%S', time.localtime(o.stat().st_mtime))}.json")


def begin(ctx, sid: str):
    from . import V3_STAGE_IDS
    from ..v2.validation_registry import get_registry
    from .registry import v3_registries

    archive_previous_outcome(ctx, sid)
    rep = lineage_report(ctx, sid)
    ctx.save_table(rep, sid, "upstream_lineage")
    if not rep.empty:
        hard_missing = rep[(rep["kind"] == "hard") & (rep["status"] != "completed")]
        if not hard_missing.empty:
            raise V3StaleUpstreamError(f"{sid}: hard upstream not completed: {list(hard_missing['upstream'])}")
        from ..orchestrator import protected_stages

        stale = rep[(rep["status"] == "completed") & (rep["stale"] == True)]  # noqa: E712
        prot = protected_stages(ctx.cfg)
        for d in stale.loc[stale["upstream"].isin(prot), "upstream"]:
            ctx.logger.warning("%s: protected upstream %s has a changed fingerprint; its completed artefacts are used "
                               "as produced (force it explicitly to recompute)", sid, d)
        bad = stale[stale["upstream"].isin(V3_STAGE_IDS) & ~stale["upstream"].isin(prot)]
        for d in stale.loc[~stale["upstream"].isin(V3_STAGE_IDS), "upstream"]:
            ctx.logger.warning("%s: V1/V2 upstream %s is stale under the current config; its artefacts are read as "
                               "they were produced (see upstream_lineage.csv / P23)", sid, d)
        if not bad.empty and bool(ctx.cfg.get_path("v3.runtime.enforce_fresh_v3_upstream", True)):
            raise V3StaleUpstreamError(f"{sid}: V3 upstream stale: {list(bad['upstream'])}; re-run or force them")
    reg = get_registry(ctx)
    ctx.save_table(reg.summary(), sid, "validation_registry_summary")
    return reg, v3_registries(ctx)


def folds(ctx, scheme: str):
    """Registry folds with the V3 repeat limit applied (v3.runtime.max_repeats); never mutates v2.* settings."""
    from ..v2.validation_registry import get_registry

    fs = get_registry(ctx).folds(scheme)
    mr = ctx.cfg.get_path("v3.runtime.max_repeats")
    if mr and scheme in ("repeated_kfold", "sample_rate_equal_n", "instrument_equal_n"):
        fs = [f for f in fs if f.repeat < int(mr)]
    return fs


def available_schemes(ctx, wanted: list[str]) -> list[str]:
    from ..v2.validation_registry import get_registry

    have = set(get_registry(ctx).schemes())
    return [s for s in wanted if s in have]


def student_meta(ctx) -> pd.DataFrame:
    """Train student metadata used ONLY for grouping/stratification (school_id is never a feature)."""
    from .. import ID

    b = ctx.bundle()
    cols = [ID] + [c for c in ("instrument", "original_sample_rate", "school_id") if c in b.train_students]
    m = b.train_students[cols].copy()
    m[ID] = m[ID].astype(str)
    return m.drop_duplicates(ID).set_index(ID)


def not_executed(ctx, sid: str, reason: str, **extra) -> dict:
    """Uniform NOT_EXECUTED summary + table, so missing prerequisites never look like results."""
    import time

    from ..io_utils import write_json
    from . import NOT_EXECUTED

    flat = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in extra.items()}
    ctx.save_table(pd.DataFrame([{"stage": sid, "status": NOT_EXECUTED, "reason": reason, **flat}]), sid, "not_executed")
    write_json(ctx.stage_dir(sid) / "_outcome.json", {"stage": sid, "status": NOT_EXECUTED, "reason": reason,
                                                      "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                      "results_are_real": False})
    ctx.logger.warning("%s NOT_EXECUTED: %s", sid, reason)
    return {"status": NOT_EXECUTED, "reason": reason, **extra}
