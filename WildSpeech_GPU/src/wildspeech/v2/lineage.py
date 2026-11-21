"""Artefact lineage: V2 stages refuse to consume outdated upstream artefacts.

`require_fresh(ctx, stage)` is called at the start of every V2 stage. For each upstream stage (hard + soft) that
is completed, the fingerprint recorded in its _status.json must equal the fingerprint recomputed from the current
config and *its* upstream statuses. A mismatch means its tables were produced under a different state than the
one this stage would be compared against -> StaleUpstreamError (unless v2.runtime.enforce_fresh_upstream=false).
The orchestrator normally prevents this by re-running stale dependencies first (Pipeline.resolve).
"""
from __future__ import annotations

import pandas as pd


class StaleUpstreamError(RuntimeError):
    pass


def upstream_report(ctx, sid: str) -> pd.DataFrame:
    from ..orchestrator import STAGES, Pipeline

    pipe = Pipeline(ctx)
    spec = STAGES[sid]
    rows = []
    for d in spec.deps + spec.soft:
        st = pipe.status(d)
        status = st.get("status", "never_run")
        stale = pipe.is_stale(d) if status == "completed" else None
        rows.append({"stage": sid, "upstream": d, "kind": "hard" if d in spec.deps else "soft", "status": status,
                     "stale": stale, "recorded_fingerprint": st.get("fingerprint"), "finished": st.get("finished")})
    return pd.DataFrame(rows)


def require_fresh(ctx, sid: str) -> pd.DataFrame:
    rep = upstream_report(ctx, sid)
    ctx.save_table(rep, sid, "upstream_lineage")
    from . import V2_STAGE_IDS

    stale = rep[(rep["status"] == "completed") & (rep["stale"] == True)]  # noqa: E712
    # stale hard deps or stale V2 artefacts are errors; a stale *soft V1* artefact (e.g. P10 used only to pick
    # the comparator, which V2 re-evaluates on the frozen registry) is recorded as a warning.
    bad = stale[(stale["kind"] == "hard") | stale["upstream"].isin(V2_STAGE_IDS)]
    for d in stale.loc[~stale.index.isin(bad.index), "upstream"]:
        ctx.logger.warning("%s: soft V1 upstream %s is stale; V2 re-evaluates what it needs on the frozen registry", sid, d)
    hard_missing = rep[(rep["kind"] == "hard") & (rep["status"] != "completed")]
    if not hard_missing.empty:
        raise StaleUpstreamError(f"{sid}: hard upstream not completed: {list(hard_missing['upstream'])}")
    if not bad.empty and bool(ctx.cfg.get_path("v2.runtime.enforce_fresh_upstream", True)):
        raise StaleUpstreamError(f"{sid}: upstream artefacts are stale (fingerprint changed since they were produced): "
                                 f"{list(bad['upstream'])}. Re-run them (they are re-planned automatically when "
                                 "v2.runtime.refresh_stale_soft_deps=true) or force them explicitly.")
    return rep


def completed(ctx, sid: str) -> bool:
    from ..orchestrator import Pipeline

    return Pipeline(ctx).status(sid).get("status") == "completed"
