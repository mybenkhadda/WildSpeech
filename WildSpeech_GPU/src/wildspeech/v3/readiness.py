"""Readiness boards for V3: what is planned, checked, installed, loadable, smoke-tested, inferred and REAL.

The notebook and the CLI used to show one "status" per stage, so an eligibility-only P24 or a P25 that found no
inputs could look like progress. These boards keep the steps apart:

model_readiness(ctx)    one row per speech model, left to right in the order the steps must happen:
    reviews            review decisions complete? (asr.reviews board; Pasketti overlap is never auto-approved)
    download_auth      reviews pass AND ALLOW_MODEL_DOWNLOADS (+ v3.asr.downloads.authorized_models)
    installed          checkpoint at the pinned commit with every manifest file
    technical          weights load + synthetic-audio forward pass (asr.inference.load_check; no competition data)
    smoke              few training-clip decode passed (inference_readiness dimension)
    inference          execution state of the accepted P24 cache (never_run / partial / complete / ...)
    next_action        the single next thing to do for that model
stage_readiness(ctx)    one row per V3 stage: last run status, whether its results are REAL (completed and not
                        NOT_EXECUTED), the input it is waiting for, and whether saved tables are current or stale
table_state(ctx, s, n)  MISSING | NO_RESULTS | STALE | CURRENT for one saved table

Filesystem / registry only: nothing here loads weights, downloads, or runs a stage.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

V3_PIPELINE = ("P23", "P24", "P25", "P26", "P27", "P28", "P29")
TABLE_STATES = ("MISSING", "NO_RESULTS", "STALE", "CURRENT")
STAGE_INPUT = {"P25": ("text", 1), "P26": ("text", 1), "P27": ("text", 2), "P28": ("any", 1), "P29": ("any", 1)}


# ============================================================================================== tables
def _run_window(st: dict) -> tuple[float | None, float | None]:
    fin = st.get("finished")
    if not fin:
        return None, None
    try:
        end = time.mktime(time.strptime(str(fin)[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:  # noqa: BLE001
        return None, None
    return end - float(st.get("seconds") or 0.0) - 2.0, end + 2.0


def table_state(ctx, stage: str, name: str) -> tuple[str, str]:
    """(state, detail). STALE = the file predates the stage's last recorded run (it is a leftover of an earlier run
    and does not describe the current state, e.g. results from before a NOT_EXECUTED re-run)."""
    from ..io_utils import is_no_results
    from ..orchestrator import Pipeline

    p = ctx.table_path(stage, name)
    if p is None:
        return "MISSING", f"{stage}/{name} not written"
    st = Pipeline(ctx).status(stage)
    start, _end = _run_window(st)
    if start is not None and p.stat().st_mtime < start:
        return "STALE", (f"written {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}, before the "
                         f"last {stage} run ({st.get('status')}, {st.get('finished')})")
    df = ctx.read_stage_table(stage, name)
    if df is None or is_no_results(df):
        return "NO_RESULTS", (df.attrs.get("reason", "zero rows") if df is not None else "unreadable")
    return "CURRENT", f"{len(df)} rows"


def stage_outcome(ctx, stage: str) -> dict:
    """The stage's own outcome record (NOT_EXECUTED marker or P24 outcome), else {}."""
    from ..io_utils import read_json

    d = Path(ctx.stage_dir(stage))
    o = read_json(d / "_outcome.json", {}) or {}
    if stage == "P24":
        p = read_json(d / "p24_outcome.json", {}) or {}
        o = {**o, **{k: p.get(k) for k in ("stage_outcome", "inference_executed", "accepted_models") if k in p}}
    return o


# ============================================================================================== models
def model_readiness(ctx, deep: bool = False) -> pd.DataFrame:
    from ..asr.inference import load_check_record
    from ..asr.model_manager import downloads_allowed
    from ..asr.model_registry import by_priority, document_for, proposals_path, reviews_path
    from ..asr.outputs import execution_state
    from ..asr.reviews import load_proposals, load_reviews, model_decisions, review_board
    from ..stages.p24_asr import eligibility

    doc = document_for(ctx)
    specs, recs, _ = eligibility(ctx, True, deep, False)
    board = review_board(doc.specs, load_reviews(reviews_path(ctx)), load_proposals(proposals_path(ctx)),
                         doc.allowed_licenses, doc.license_policy, ctx.paths.root)
    dec = model_decisions(board).set_index("model_key") if not board.empty else pd.DataFrame()
    keys = by_priority([k for k in recs if k in specs], specs)
    try:
        es = execution_state(ctx, keys, ("train",)).set_index("model_key")
    except Exception:  # noqa: BLE001 - data bundle unavailable: inference state unknown
        es = pd.DataFrame()
    rows = []
    for k in keys:
        s, r = specs[k], recs[k]
        d = dec.loc[k] if (not dec.empty and k in dec.index) else {}
        reviews_ok = bool(d.get("reviews_complete", False)) if len(d) else False
        rejected = bool(d.get("rejected", False)) if len(d) else False
        dl_flag, dl_why = downloads_allowed(ctx.cfg, None, k)
        installed = r.groups.get("checkpoint") == "PASS"
        lc = load_check_record(ctx, k)
        technical = ("NOT_CHECKED" if not lc else
                     ("LOAD_OK" if lc.get("ok") and lc.get("revision") == r.snapshot_revision else
                      ("STALE_CHECK" if lc.get("revision") != r.snapshot_revision else "LOAD_FAILED")))
        smoke = r.checks.get("smoke_test", "NOT_PROBED")
        inf = es.loc[k, "state"] if (not es.empty and k in es.index) else "unknown"
        acc = bool(es.loc[k, "accepted"]) if (not es.empty and k in es.index) else False
        if rejected or r.lifecycle == "INELIGIBLE":
            nxt = "none: model rejected / ineligible"
        elif not reviews_ok:
            nxt = "review decision: python -m wildspeech.asr.reviews commands --model " + k
        elif not installed and not (r.download_allowed and dl_flag):
            nxt = f"authorize download (ALLOW_MODEL_DOWNLOADS=True{', add to DOWNLOAD_MODELS' if 'authorized_models' in dl_why else ''})"
        elif not installed:
            nxt = f"python -m wildspeech v3-install --model {k} --execute"
        elif r.groups.get("framework") != "PASS":
            nxt = "framework: " + "; ".join(a for a in r.actions if a)[:200]
        elif technical != "LOAD_OK":
            nxt = f"load check (RUN_SMOKE=True or python -m wildspeech v3-smoke --model {k} --load-only)"
        elif smoke != "PASS":
            nxt = f"smoke test (python -m wildspeech v3-smoke --model {k})"
        elif inf != "complete":
            nxt = "P24 inference (RUN_PIPELINE=True, stage P24)"
        else:
            nxt = "done: outputs accepted"
        rows.append({"model_key": k, "priority": s.priority, "role": s.role, "output": s.output,
                     "lifecycle": r.lifecycle, "reviews": "REJECTED" if rejected else ("COMPLETE" if reviews_ok else "PENDING"),
                     "outstanding_decisions": (d.get("outstanding_decisions", "") if len(d) else ""),
                     "download_auth": bool(r.download_allowed and dl_flag), "download_flag": dl_why,
                     "installed": installed, "checkpoint_detail": r.results.get("manifest_files").detail
                     if r.results.get("manifest_files") else "", "technical": technical,
                     "technical_error": lc.get("error", "") if lc else "", "smoke": smoke, "inference": inf,
                     "accepted_output": acc, "eligible_for_inference": r.eligible, "next_action": nxt})
    return pd.DataFrame(rows)


# ============================================================================================== stages
def stage_readiness(ctx) -> pd.DataFrame:
    from ..asr.outputs import accepted_models
    from ..orchestrator import Pipeline, protected_stages

    pipe = Pipeline(ctx)
    prot = protected_stages(ctx.cfg)
    try:
        text_models = accepted_models(ctx, ("train",), "text")
        any_models = accepted_models(ctx, ("train",), "any")
    except Exception:  # noqa: BLE001
        text_models, any_models = [], []
    rows = []
    for sid in V3_PIPELINE:
        st = pipe.status(sid)
        status = st.get("status", "never_run")
        oc = stage_outcome(ctx, sid)
        real = status == "completed" and oc.get("status") != "NOT_EXECUTED"
        if sid == "P24":
            real = bool(oc.get("inference_executed")) and status == "completed"
        need = STAGE_INPUT.get(sid)
        if need:
            have = text_models if need[0] == "text" else any_models
            inputs = f"{len(have)}/{need[1]} accepted ASR model(s): {have or 'none'}"
            inputs_ok = len(have) >= need[1]
        elif sid == "P24":
            inputs, inputs_ok = "eligible models (see model_readiness)", None
        else:
            inputs, inputs_ok = "V1/V2 artefacts", True
        rows.append({"stage": sid, "enabled": pipe.enabled(sid), "protected": sid in prot, "last_status": status,
                     "finished": st.get("finished"), "outcome": oc.get("stage_outcome") or oc.get("status") or "",
                     "results_are_real": bool(real), "inputs": inputs, "inputs_ready": inputs_ok,
                     "up_to_date": (not pipe.is_stale(sid)) if status == "completed" else None,
                     "note": str(oc.get("reason") or st.get("note") or "")[:300]})
    return pd.DataFrame(rows)


def explain_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Annotate a dry-run plan so 'would_run' is never read as a result."""
    meaning = {"would_run": "PLANNED (nothing executed yet)", "cached": "COMPLETED earlier, up to date",
               "protected": "COMPLETED earlier, protected (force explicitly to recompute)",
               "blocked": "BLOCKED by a hard dependency", "disabled": "DISABLED by config",
               "not_executed": "NOT_EXECUTED (missing real inputs; no results)", "failed": "FAILED (see _status.json)",
               "completed": "COMPLETED in this call"}
    if plan is None or plan.empty:
        return plan
    out = plan.copy()
    out["meaning"] = out["status"].map(meaning).fillna(out["status"])
    return out
