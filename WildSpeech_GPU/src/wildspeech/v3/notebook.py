"""Thin helpers for notebooks/WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER.ipynb.

All computation lives in the package. These helpers load contexts, run stages through the orchestrator, and
display saved tables/figures. The eligibility helpers read local files only. They never load weights (unless
deep=True, which loads configs and tokenizers) and never download anything.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

V3_STAGES = ("P21", "P23", "P24", "P25", "P26", "P27", "P28", "P29")


def context(profile: str = "V3_FULL", overrides: dict | None = None):
    from ..context import build_context

    return build_context(profile, overrides=overrides)


def plan(ctx, stages: list[str] | None = None) -> pd.DataFrame:
    from ..orchestrator import Pipeline
    from .readiness import explain_plan

    return explain_plan(Pipeline(ctx).run(stages, dry_run=True))


def force_list(force, stages: list[str]) -> list[str]:
    """FORCE may be a bool (True = recompute every requested stage) or a list of stage ids."""
    if force is True:
        return list(stages)
    if not force:
        return []
    return [s for s in force if isinstance(s, str)]


def run_stages(ctx, stages: list[str], force=(), enabled: bool = False) -> pd.DataFrame:
    """Runs only when `enabled` is True (the notebook passes RUN_PIPELINE); otherwise returns the dry-run plan.
    Every row is annotated (`meaning`) so PLANNED / NOT_EXECUTED are never mistaken for results. Protected stages
    (v3.orchestration.protected_stages, default P23) are recomputed only when named in FORCE explicitly."""
    from ..orchestrator import Pipeline
    from .readiness import explain_plan

    f = force_list(force, stages)
    if force is True:                     # FORCE=True never silently recomputes a protected stage
        from ..orchestrator import protected_stages

        f = [s for s in f if s not in protected_stages(ctx.cfg)]
    if not enabled:
        print("RUN_PIPELINE is False -> showing the plan only (nothing executed)")
        return explain_plan(Pipeline(ctx).run(stages, force=f, dry_run=True))
    return explain_plan(Pipeline(ctx).run(stages, force=f))


def status(ctx, stages=V3_STAGES) -> pd.DataFrame:
    """Stage status + the truth about P24: a 'completed' P24 without inference is flagged (pre-fix runs)."""
    from ..orchestrator import Pipeline
    from ..stages.p24_asr import outcome

    t = Pipeline(ctx).status_table()
    t = t[t["stage"].isin(stages)].copy()
    o = outcome(ctx)
    t["asr_truth"] = ""
    m = t["stage"] == "P24"
    t.loc[m, "asr_truth"] = f"{o.get('stage_outcome')} (inference_executed={o.get('inference_executed')})"
    if (t.loc[m, "status"] == "completed").any() and not o.get("inference_executed"):
        t.loc[m, "asr_truth"] += "  <- recorded 'completed' by the old orchestrator but NO inference ran"
    return t


def table(ctx, stage: str, name: str, n: int | None = 20, allow_stale: bool = False) -> pd.DataFrame | None:
    """A saved stage table, or None with an explicit reason: MISSING (never written), NO_RESULTS (empty result;
    never a pandas EmptyDataError), STALE (leftover of an earlier run of the stage; shown only with
    allow_stale=True)."""
    from .readiness import stage_outcome, table_state

    state, detail = table_state(ctx, stage, name)
    oc = stage_outcome(ctx, stage)
    tag = f" | stage outcome: {oc.get('status') or oc.get('stage_outcome')} — {str(oc.get('reason', ''))[:160]}" if oc else ""
    if state == "MISSING":
        print(f"[{stage}] '{name}': MISSING ({detail}){tag}")
        return None
    if state == "NO_RESULTS":
        print(f"[{stage}] '{name}': NO_RESULTS ({detail}){tag}")
        return None
    if state == "STALE" and not allow_stale:
        print(f"[{stage}] '{name}': STALE — {detail}; not shown (allow_stale=True to inspect){tag}")
        return None
    df = ctx.read_stage_table(stage, name)
    return df.head(n) if n else df


def tables(ctx, stage: str, names, n: int | None = None) -> None:
    """Display several stage tables with explicit MISSING / NO_RESULTS / STALE messages."""
    from IPython.display import display

    for t in names:
        df = table(ctx, stage, t, n=n)
        if df is not None:
            print(f"--- {stage}/{t}")
            display(df)


def figures(ctx, stage: str) -> list[Path]:
    d = ctx.stage_dir(stage) / "figures"
    return sorted(d.glob("*.png"))


def show_figures(ctx, stage: str, limit: int = 12) -> None:
    from IPython.display import Image, display

    fs = figures(ctx, stage)
    if not fs:
        print(f"[{stage}] no figures yet")
    for f in fs[:limit]:
        print(f.name)
        display(Image(filename=str(f)))


# ====================================================================================== P24 eligibility
def eligibility(ctx, deep: bool = False, verify_hashes: bool = False) -> pd.DataFrame:
    """Full eligibility table (registry facts + 13 dimensions + lifecycle + reasons + actions). Local files only."""
    from ..stages.p24_asr import eligibility as elig

    _, _, t = elig(ctx, deep=deep, verify_hashes=verify_hashes)
    return t


SUMMARY_COLS = ["model_key", "priority", "family", "role", "output", "lifecycle", "eligible", "download_allowed",
                "legal_status", "provenance_status", "checkpoint_status", "framework_status", "architecture",
                "snapshot_origin", "snapshot_kind", "snapshot_revision", "license", "gpu_need_gb", "manifest_gb"]


def eligibility_report(ctx, deep: bool = False, verify_hashes: bool = False) -> dict[str, pd.DataFrame]:
    """{summary, dimensions, checks, actions, evidence, files, restrictions, invalid_reviews, downloads, policy,
    license_policy, proposals, locks, matrix, check_matrix}."""
    from ..stages.p24_asr import check_matrix, eligibility_tables, status_matrix
    from ..stages.p24_asr import eligibility as elig

    specs, recs, t = elig(ctx, deep=deep, verify_hashes=verify_hashes)
    out = {"summary": t[[c for c in SUMMARY_COLS if c in t]].sort_values("priority") if "priority" in t else t}
    out.update(eligibility_tables(ctx, specs, recs))
    out["checks"] = out.pop("eligibility_checks")
    out["dimensions"] = out.pop("eligibility_dimensions")
    out["actions"] = out.pop("required_actions")
    out["evidence"] = out.pop("model_evidence")
    out["files"] = out.pop("model_files")
    out["restrictions"] = out.pop("model_restrictions")
    out["downloads"] = out.pop("download_commands")
    out["policy"] = out.pop("competition_policy")
    out["proposals"] = out.pop("review_proposals")
    out["matrix"] = status_matrix(recs)
    out["check_matrix"] = check_matrix(recs)
    out["records"] = recs
    return out


def show_eligibility_matrix(ctx, deep: bool = False, rep: dict | None = None) -> None:
    from IPython.display import display

    from . import viz as V

    rep = rep or eligibility_report(ctx, deep)
    display(V.status_matrix(rep["matrix"], "Eligibility by dimension (live, from local files)"))
    life = rep["summary"][["model_key", "lifecycle"] + (["priority"] if "priority" in rep["summary"] else [])]
    display(V.lifecycle(life, "Model lifecycle (live)"))


def environment() -> pd.DataFrame:
    from ..asr.environment import framework_environment

    return pd.DataFrame([framework_environment()]).T.rename(columns={0: "value"})


def installed_checkpoints(ctx) -> pd.DataFrame:
    """Where each model's snapshot was found (or searched), its revision, kind and manifest problems. Filesystem only."""
    from ..asr.model_registry import document_for
    from ..asr.snapshots import detect_architecture, manifest_problems, manifest_status, resolve_snapshot

    rows = []
    for k, s in document_for(ctx).specs.items():
        sn = resolve_snapshot(s, ctx.cfg.get_path("paths.hf_cache_dir"))
        arch = detect_architecture(sn.path) if sn.path else None
        probs = manifest_problems(manifest_status(sn.path, s)) if sn.path else []
        rows.append({"model_key": k, "installed": sn.path is not None and not probs, "origin": sn.origin, "kind": sn.kind,
                     "path": sn.path, "revision": sn.revision, "revision_source": sn.revision_source, "pinned": s.revision,
                     "architecture": arch, "manifest_gb": round(s.manifest_bytes() / 1e9, 3),
                     "problems": "; ".join(probs) if sn.path else sn.reason, "searched": "; ".join(sn.searched)})
    return pd.DataFrame(rows)


def component_checks(ctx) -> pd.DataFrame:
    """Processor / tokenizer contract (configs, processors, tokenizers from local snapshots; no weights)."""
    from ..asr.environment import component_check
    from ..asr.model_registry import document_for
    from ..asr.snapshots import detect_architecture, resolve_snapshot

    rows = []
    for k, s in document_for(ctx).specs.items():
        sn = resolve_snapshot(s, ctx.cfg.get_path("paths.hf_cache_dir"))
        if sn.path:
            rows.append(component_check(s, sn.path, detect_architecture(sn.path)))
    return pd.DataFrame(rows)


def install_plan(ctx, models: list[str] | None = None) -> pd.DataFrame:
    """What the model manager would download / reuse / convert (filesystem only)."""
    from ..asr.model_manager import install_plan as plan

    return plan(ctx, models)


def install(ctx, models: list[str] | None = None, allow: bool = False, verify: bool = True) -> pd.DataFrame:
    """Install APPROVED models only. allow=False -> dry run (the notebook passes ALLOW_MODEL_DOWNLOADS)."""
    from ..asr.model_manager import install as inst

    return inst(ctx, models, allow=allow, verify=verify)


def verify(ctx, models: list[str] | None = None) -> pd.DataFrame:
    from ..asr.model_manager import verify_installed

    return verify_installed(ctx, models)


def smoke(ctx, models: list[str] | None = None, clips: int = 8) -> pd.DataFrame:
    """Few-clip decode (training split) for every READY_FOR_INFERENCE model, one GPU model at a time."""
    import gc

    from ..acoustic.batching import VRAMMonitor
    from ..asr.inference import run_smoke
    from ..asr.model_registry import by_priority
    from ..stages.p24_asr import eligibility as elig

    specs, recs, _ = elig(ctx, deep=True)
    keys = by_priority([k for k, r in recs.items() if r.eligible and (not models or k in models)], specs)
    rows = []
    for k in keys:
        r = run_smoke(ctx, specs[k], recs[k], clips)
        rows.append({x: y for x, y in r.items() if x != "examples"})
        gc.collect()
        VRAMMonitor.release()
    skipped = [k for k in (models or list(recs)) if k in recs and k not in keys]
    for k in skipped:
        rows.append({"model_key": k, "ok": False, "error": f"not READY_FOR_INFERENCE (lifecycle {recs[k].lifecycle})"})
    return pd.DataFrame(rows)


def smoke_examples(ctx, model_key: str, n: int = 8) -> pd.DataFrame | None:
    from ..asr.checkpointing import ASRCheckpointStore

    df = ASRCheckpointStore(ctx.paths.v3_cache / "asr_smoke", model_key).load("train")
    return None if df.empty else df.head(n)


# ===================================================================================== P24 progress / outputs
def execution_state(ctx) -> pd.DataFrame:
    from ..asr.model_registry import document_for
    from ..asr.outputs import execution_state as es

    return es(ctx, list(ctx.cfg.get_path("v3.asr.models", list(document_for(ctx).specs))))


def output_gate(ctx, splits=("train",)) -> pd.DataFrame:
    from ..asr.outputs import scan_outputs

    return scan_outputs(ctx, None, splits)


def p24_outcome(ctx) -> dict:
    from ..stages.p24_asr import outcome

    return outcome(ctx)


def transcripts(ctx, model_key: str | None = None, n: int = 20, split: str = "train") -> pd.DataFrame | None:
    """Expected text vs recognised transcript / phones for ACCEPTED outputs only (training split by default)."""
    from .. import FILE
    from ..asr.outputs import load_accepted_asr

    a = load_accepted_asr(ctx, (split,))
    if a.empty:
        print("no accepted ASR output yet (P24 has not produced real inference)")
        return None
    if model_key:
        a = a[a["model_key"] == model_key]
    clips = (ctx.bundle().train_clips if split == "train" else ctx.bundle().test_clips)
    keep = [c for c in (FILE, "task", "expected_text", "item_score") if c in clips]
    m = a.merge(clips[keep].drop_duplicates(FILE), on=FILE, how="left")
    cols = [c for c in ("model_key", FILE, "task", "expected_text", "item_score", "transcript", "phones",
                        "mean_token_logprob", "expected_ll", "precision", "truncated") if c in m]
    return m[cols].sort_values(["model_key", FILE]).groupby("model_key").head(n)


def phoneme_feature_summary(ctx) -> pd.DataFrame | None:
    """Coverage / mean of every P25 clip feature per model (accepted outputs only)."""
    from ..phonetics.builder import load_phonetic_tables

    wide, _ = load_phonetic_tables(ctx, ("train",))
    if wide.empty or len(wide.columns) <= 1:
        print("no P25 phonetic features yet")
        return None
    num = wide.drop(columns=[c for c in wide.columns if "::" not in c])
    s = pd.DataFrame({"coverage": num.notna().mean(), "mean": num.mean(), "sd": num.std()})
    s.index.name = "feature"
    s = s.reset_index()
    s["model"] = s["feature"].str.split("::").str[0]
    return s.sort_values(["model", "feature"])


def coverage_figure(ctx):
    """Per-model coverage of the training clips (accepted rows / candidates) as horizontal bars."""
    from . import viz as V

    st = execution_state(ctx)
    if st.empty:
        return V.empty("ASR coverage", "no models configured")
    d = st[st["split"] == "train"].assign(coverage_pct=lambda x: 100 * x["coverage"].fillna(0.0))
    return V.bars(d, "model_key", "coverage_pct", "ASR coverage of training clips (accepted outputs)",
                  ylabel="% of candidate clips", ref=100.0)


def report(ctx) -> str:
    from .report import build_report, export_root_tables

    export_root_tables(ctx)
    return build_report(ctx)


# ===================================================================================== frozen encoder embeddings
def embedding_summary(ctx, splits=("train",)) -> pd.DataFrame:
    """Per accepted model with pooled embeddings: rows, dimension, layers, finite share (no values displayed)."""
    import numpy as np

    from ..asr.checkpointing import ASRCheckpointStore
    from ..asr.outputs import usable_pairs

    rows = []
    for k, s in sorted(usable_pairs(ctx, tuple(splits))):
        e = ASRCheckpointStore(ctx.paths.v3_cache, k).load_embeddings(s)
        if e.empty:
            continue
        cols = [c for c in e.columns if c.startswith("e") and c[1:].isdigit()]
        X = e[cols].to_numpy(np.float32)
        prov = ASRCheckpointStore(ctx.paths.v3_cache, k).provenance(s)
        rows.append({"model_key": k, "split": s, "rows": len(e), "dim": len(cols), "finite_share": float(np.isfinite(X).mean()),
                     "mean_norm": float(np.nanmean(np.linalg.norm(np.nan_to_num(X), axis=1))),
                     "output": prov.get("output"), "layers": prov.get("encoder_layers")})
    return pd.DataFrame(rows)


def lifecycle_table(ctx) -> pd.DataFrame:
    rep = eligibility_report(ctx)
    return rep["summary"]


# ===================================================================================== readiness / reviews
def model_readiness(ctx, deep: bool = False) -> pd.DataFrame:
    """reviews -> download authorization -> installed -> technical load -> smoke -> inference -> next action."""
    from .readiness import model_readiness as mr

    return mr(ctx, deep)


def stage_readiness(ctx) -> pd.DataFrame:
    from .readiness import stage_readiness as sr

    return sr(ctx)


def review_board(ctx) -> dict[str, pd.DataFrame]:
    """{'models': one row per model (decisions still yours), 'checks': one row per model x check (+ commands)}."""
    from ..asr.model_registry import document_for, proposals_path, reviews_path
    from ..asr.reviews import load_proposals, load_reviews, model_decisions
    from ..asr.reviews import review_board as rb

    doc = document_for(ctx)
    b = rb(doc.specs, load_reviews(reviews_path(ctx)), load_proposals(proposals_path(ctx)), doc.allowed_licenses,
           doc.license_policy, ctx.paths.root)
    return {"models": model_decisions(b), "checks": b}


def review_commands(ctx, models: list[str] | None = None) -> None:
    """Print the exact commands for every open review decision (run them in a terminal, not here)."""
    b = review_board(ctx)["checks"]
    sel = b[b["decision_needed"]] if not b.empty else b
    if models:
        sel = sel[sel["model_key"].isin(models)]
    if sel.empty:
        print("no open review decisions")
        return
    for r in sel.to_dict("records"):
        print(f"\n## {r['model_key']} · {r['check']} — {r['status']}")
        if r.get("open_questions") or r.get("outstanding"):
            print("   open questions:", r.get("open_questions") or r.get("outstanding"))
        for kind in ("accept", "accept_after_deciding", "record", "reject", "defer"):
            if r.get(f"cmd_{kind}"):
                print(f"   [{kind}] {r[f'cmd_{kind}']}")


def revision_inventory(ctx, models: list[str] | None = None) -> pd.DataFrame:
    """Which cached revision (snapshot) of each repo holds which manifest file. Filesystem only."""
    from ..asr.model_registry import document_for
    from ..asr.snapshots import hf_hub_dir
    from ..asr.snapshots import revision_inventory as ri

    doc = document_for(ctx)
    hub = hf_hub_dir(ctx.cfg.get_path("paths.hf_cache_dir"))
    rows = [r for k, s in doc.specs.items() if not models or k in models for r in ri(s, hub)]
    return pd.DataFrame(rows)


def repair_plan(ctx, models: list[str] | None = None) -> pd.DataFrame:
    """Per manifest file of the PINNED snapshot: present, repairable offline from a sha256 blob, or to download."""
    from ..asr.model_manager import repair_plan as rp

    return rp(ctx, models)


def load_checks(ctx, models: list[str] | None = None, device: str | None = None) -> pd.DataFrame:
    """TECHNICAL load check per model whose checkpoint + framework dimensions pass (reviews NOT required): weights
    load, forward pass on synthetic audio, output contract. One GPU model at a time. Never touches competition
    audio. Results -> cache/v3/asr_load_checks/<model>.json."""
    import gc

    from ..acoustic.batching import VRAMMonitor
    from ..asr.inference import load_check, technically_ready
    from ..asr.model_registry import by_priority
    from ..stages.p24_asr import eligibility as elig

    specs, recs, _ = elig(ctx, deep=True)
    rows = []
    for k in by_priority([k for k in recs if k in specs and (not models or k in models)], specs):
        ok, bad = technically_ready(recs[k])
        if not ok:
            rows.append({"model_key": k, "ok": False, "stage": "not_attempted",
                         "error": "not technically ready: " + ", ".join(bad),
                         "next": "; ".join(a for a in recs[k].actions if "review" not in a)[:240]})
            continue
        r = load_check(ctx, specs[k], recs[k], device)
        rows.append({x: y for x, y in r.items() if x != "outputs"})
        gc.collect()
        VRAMMonitor.release()
    return pd.DataFrame(rows)


def p23_results(ctx) -> None:
    """Display the stored P23 fusion-audit results (never re-runs P23)."""
    tables(ctx, "P23", ("baseline_reproduction", "fusion_verdicts", "v2_budget_cap", "task_ensemble_summary",
                        "lineage_audit", "v3_repairs"))


EXECUTION_SEQUENCE = """\
Manual execution sequence (each step is a deliberate switch; nothing runs by default)
  1. Reviews (terminal):  python -m wildspeech.asr.reviews board
                          python -m wildspeech.asr.reviews commands          # exact accept / reject / defer lines
     Pasketti models: decide the ReadNet / Reach Every Reader overlap yourself (accept with mitigation, reject, or
     defer). No command approves them automatically.
  2. Install:            ALLOW_MODEL_DOWNLOADS = True (optionally DOWNLOAD_MODELS = [...]) -> section 3 cell
                          (or: python -m wildspeech v3-install --model <key> --execute)
  3. Technical check:    RUN_SMOKE = True -> section 4 (load check on synthetic audio, then a few-clip smoke decode
                          for models that are fully eligible)
  4. Inference:          RUN_PIPELINE = True -> section 5 (P24; resumable; interrupt-safe)
  5. Downstream:         sections 7-12 (P25 -> P26 -> P27 -> P28 -> P29). Each stage refuses to run (NOT_EXECUTED)
                          without accepted real ASR outputs; P27 needs >= 2 recognisers.
  6. Report:             section 15
"""
