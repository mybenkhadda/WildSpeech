"""P24 — local speech-model installation and inference (eligible models only).

1. ELIGIBILITY (13 dimensions -> lifecycle) for every configured model, from the evidence registry, the manual
   reviews, the local files (deep processor/tokenizer check; sha256 verification when v3.asr.downloads.verify_sha256)
   and the runtime state. Tables: model_eligibility, eligibility_dimensions, eligibility_checks (long),
   required_actions, model_evidence, model_files, model_restrictions, competition_policy, license_policy,
   review_proposals, invalid_reviews, download_commands, install_plan, framework_environment, locks,
   execution_state_before; outputs/V3/P24_ASR/model_evidence.json (+ outputs/V3/model_eligibility.csv,
   outputs/V3/model_evidence.json). Figures: eligibility matrix (13 dimensions) and lifecycle readiness.
2. INSTALL: models whose evidence dimensions all pass (READY_FOR_DOWNLOAD / CHECKPOINT_INCOMPLETE / pending .bin
   conversion) are installed by asr.model_manager ONLY when ALLOW_MODEL_DOWNLOADS is enabled; eligibility is then
   re-evaluated from disk. Nothing is downloaded for a model with a pending or rejected review.
3. SMOKE TEST (v3.asr.smoke.enabled): a few task-stratified training clips per eligible model into
   cache/v3/asr_smoke (never accepted as output). A failed smoke test skips that model (lifecycle FAILED).
4. INFERENCE: models pass in priority order, one GPU model at a time, resumable, on the configured splits; each run
   holds an INFERENCE_RUNNING heartbeat lock and records _last_run.json.
5. Throughput / memory per model and batch; real-output inventory; lifecycle after the run; transcription
   examples (training data only).

Stage outcome (p24_outcome.json + summary["stage_outcome"]):
  ELIGIBILITY_ONLY      no model eligible -> stage status NOT_EXECUTED ('not_executed', never 'completed')
  INFERENCE_FAILED      eligible models existed but none produced accepted output -> RuntimeError (status 'failed')
  INFERENCE_PARTIAL     accepted outputs exist, but some model/split failed, is incomplete, or is not eligible
  INFERENCE_COMPLETE    every configured model produced complete accepted outputs on every split
No audio leaves the machine; models are loaded with local_files_only=True and never with trust_remote_code.
"""
from __future__ import annotations

import gc
import json

import pandas as pd

from .. import FILE
from ..asr.checkpointing import ASRCheckpointStore
from ..asr.eligibility import DIMENSIONS, checker_for
from ..asr.inference import ASRInferenceEngine, run_smoke, throughput_summary
from ..asr.model_registry import (by_priority, document_for, evidence_table, files_table, preferred_models,
                                  proposals_path, registry_table, restrictions_table)
from ..asr.outputs import execution_state, scan_outputs
from ..io_utils import write_json
from ..v3 import viz as V
from ..v3.common import begin, not_executed
from ..v3.registry import save_figure_v3

S = "P24"
OUTCOMES = ("ELIGIBILITY_ONLY", "INFERENCE_FAILED", "INFERENCE_PARTIAL", "INFERENCE_COMPLETE")
INSTALLABLE = ("READY_FOR_DOWNLOAD", "CHECKPOINT_INCOMPLETE", "FRAMEWORK_INCOMPATIBLE")


def configured_keys(ctx, specs) -> list[str]:
    return list(ctx.cfg.get_path("v3.asr.models", list(specs)) or list(specs))


def eligibility(ctx, probe_files: bool = True, deep: bool = False, verify_hashes: bool = False, env: dict | None = None):
    """-> (specs, records, table). Local files only; loads no weights (deep=True instantiates processors/tokenizers)."""
    from ..asr.run_state import save_hash_cache

    doc = document_for(ctx)
    keys = configured_keys(ctx, doc.specs)
    ch = checker_for(ctx, env=env)
    recs = ch.check_all(doc.specs, keys, probe_files, deep, verify_hashes)
    save_hash_cache(ctx, ch.hash_cache)
    table = registry_table({k: doc.specs[k] for k in keys if k in doc.specs}, recs)
    for k in recs:
        if k not in doc.specs:
            table = pd.concat([table, pd.DataFrame([{"model_key": k, **recs[k].as_row()}])], ignore_index=True)
    return doc.specs, recs, table


def eligibility_tables(ctx, specs, recs) -> dict[str, pd.DataFrame]:
    from ..asr.reviews import load_proposals, proposal_table
    from ..asr.run_state import lock_table
    from ..asr.snapshots import download_commands

    doc = document_for(ctx)
    keys = [k for k in recs if k in specs]
    checks = pd.DataFrame([r for k in recs for r in recs[k].check_rows()])
    dims = pd.DataFrame([r for k in recs for r in recs[k].dimension_rows()])
    acts = pd.DataFrame([{"model_key": k, "priority": getattr(specs.get(k), "priority", 99), "lifecycle": r.lifecycle,
                          "step": i + 1, "action": a} for k, r in recs.items() for i, a in enumerate(r.actions)])
    inv = pd.DataFrame([iv for r in recs.values() for iv in r.invalid_reviews])
    dl = pd.DataFrame([{**download_commands(specs[k]), "lifecycle": recs[k].lifecycle,
                        "download_allowed": recs[k].download_allowed} for k in keys
                       if recs[k].checks.get("checkpoint_present") != "PASS"])
    lp = doc.license_policy
    lic = pd.DataFrame([{"license_id": x, "class": "permissive", "conditions": ""} for x in lp.permissive] +
                       [{"license_id": x, "class": (v or {}).get("class", "conditional"),
                         "conditions": " | ".join(f"{c.get('id')}: {c.get('text', '').strip()}" for c in lp.conditions(x))}
                        for x, v in lp.conditional.items()] +
                       [{"license_id": x, "class": "prohibited", "conditions": ""} for x in lp.prohibited])
    return {"eligibility_checks": checks, "eligibility_dimensions": dims, "required_actions": acts,
            "invalid_reviews": inv, "download_commands": dl,
            "model_evidence": evidence_table({k: specs[k] for k in keys}),
            "model_files": files_table({k: specs[k] for k in keys}),
            "model_restrictions": restrictions_table({k: specs[k] for k in keys}),
            "competition_policy": pd.DataFrame([doc.policy.as_row()]), "license_policy": lic,
            "review_proposals": proposal_table(load_proposals(proposals_path(ctx))),
            "locks": pd.DataFrame(lock_table(ctx, keys))}


def status_matrix(recs) -> pd.DataFrame:
    """models x dimension -> status string (13 dimensions), for the eligibility figure."""
    return pd.DataFrame({d: [recs[k].dimensions.get(d, "NOT_PROBED") for k in recs] for d in DIMENSIONS},
                        index=list(recs))


def check_matrix(recs) -> pd.DataFrame:
    """models x (dimension · check) -> status string (every check)."""
    cols = [(d, c) for d, cs in DIMENSIONS.items() for c in cs]
    return pd.DataFrame({f"{d}·{c}": [recs[k].checks.get(c, "NOT_PROBED") for k in recs] for d, c in cols},
                        index=list(recs))


def evidence_json(ctx, specs, recs) -> dict:
    doc = document_for(ctx)
    return {"competition_policy": doc.policy.as_row(),
            "license_policy": {"permissive": doc.license_policy.permissive, "conditional": doc.license_policy.conditional,
                               "prohibited": doc.license_policy.prohibited, "rule": doc.license_policy.rule,
                               "source_url": doc.license_policy.source_url},
            "models": {k: {**(specs[k].evidence_record() if k in specs else {}), "eligibility": recs[k].evidence_json()}
                       for k in recs}}


def save_eligibility(ctx, specs, recs, table, env: dict, suffix: str = "") -> None:
    ctx.save_table(table, S, "model_eligibility" + suffix)
    if not suffix:
        for name, t in eligibility_tables(ctx, specs, recs).items():
            ctx.save_table(t, S, name)
        ev = evidence_json(ctx, specs, recs)
        write_json(ctx.stage_dir(S) / "model_evidence.json", ev)
        write_json(ctx.paths.v3_root / "model_evidence.json", ev)
        table.to_csv(ctx.paths.v3_root / "model_eligibility.csv", index=False)
    life = pd.DataFrame([{"model_key": k, "lifecycle": r.lifecycle, "priority": getattr(specs.get(k), "priority", 99)}
                         for k, r in recs.items()])
    tag = "after" if suffix else "before"
    save_figure_v3(ctx, S, f"model_readiness_{tag}", V.lifecycle(life, f"Model lifecycle ({tag} this run)"), life,
                   f"Model readiness / lifecycle status ({tag} the run)")
    mat = status_matrix(recs)
    save_figure_v3(ctx, S, f"eligibility_matrix_{tag}", V.status_matrix(mat, f"Eligibility by dimension ({tag})"),
                   mat.reset_index().rename(columns={"index": "model_key"}), f"Eligibility dimensions per model ({tag})")


def _install(ctx, specs, recs) -> pd.DataFrame:
    from ..asr.model_manager import downloads_allowed, install

    allow, why = downloads_allowed(ctx.cfg)
    want = [k for k, r in recs.items() if r.download_allowed and r.lifecycle in INSTALLABLE]
    if not want:
        return pd.DataFrame()
    if not allow:
        ctx.logger.warning("P24: %s could be installed but downloads are disabled (%s)", want, why)
    t = install(ctx, want, allow=allow, verify=bool(ctx.cfg.get_path("v3.asr.downloads.verify_sha256", True)))
    return t


def run(ctx) -> dict:
    import os

    from ..acoustic.batching import VRAMMonitor
    from ..asr.environment import framework_environment

    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # inference loads use local_files_only; the model manager
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")    # temporarily lifts this only inside install()
    reg, v3r = begin(ctx, S)
    a = ctx.cfg.get_path("v3.asr", {}) or {}
    splits = list(a.get("splits", ["train"]))
    verify = bool(ctx.cfg.get_path("v3.asr.downloads.verify_sha256", True))
    env = framework_environment()
    ctx.save_table(pd.DataFrame([env]), S, "framework_environment")
    specs, recs, table = eligibility(ctx, True, True, verify, env)
    save_eligibility(ctx, specs, recs, table, env)
    keys = list(recs)
    ctx.save_table(execution_state(ctx, keys, splits), S, "execution_state_before")
    for r in table.to_dict("records"):
        v3r.log_model(r["model_key"], **{k: v for k, v in r.items() if k != "model_key"}, stage=S)
    # ---- 2. install (gated by reviews AND ALLOW_MODEL_DOWNLOADS)
    from ..asr.model_manager import install_plan

    ctx.save_table(install_plan(ctx, keys, recs), S, "install_plan")
    inst = pd.DataFrame()
    if any(r.download_allowed and r.lifecycle in INSTALLABLE for r in recs.values()):
        inst = _install(ctx, specs, recs)                  # goes online only inside model_manager._online()
        ctx.save_table(inst, S, "install_results")
        if not inst.empty and (inst["action"].isin(["installed", "converted", "verified"])).any():
            specs, recs, table = eligibility(ctx, True, True, verify, env)
            ctx.save_table(table, S, "model_eligibility_after_install")
    eligible = by_priority([k for k, r in recs.items() if r.eligible], specs)
    try:                                              # reviews -> download auth -> installed -> load -> smoke -> inference
        from ..v3.readiness import model_readiness

        ctx.save_table(model_readiness(ctx), S, "model_readiness")
        from ..asr.model_manager import repair_plan
        from ..v3.notebook import revision_inventory

        ctx.save_table(revision_inventory(ctx, keys), S, "revision_inventory")
        ctx.save_table(repair_plan(ctx, keys), S, "repair_plan")
    except Exception as e:  # noqa: BLE001 - diagnostics only
        ctx.logger.warning("P24 readiness tables failed: %s", e)
    base = {"inference_executed": False, "eligible": eligible, "not_eligible": [k for k in keys if k not in eligible],
            "lifecycle": {k: r.lifecycle for k, r in recs.items()}, "cuda_available": env.get("cuda_available"),
            "gpu": env.get("gpu_name"), "installed": inst[["model_key", "action"]].to_dict("records") if not inst.empty else []}
    if not eligible:
        out = {**base, "stage_outcome": "ELIGIBILITY_ONLY", "accepted_models": []}
        write_json(ctx.stage_dir(S) / "p24_outcome.json", out)
        return not_executed(ctx, S, "eligibility only: no speech model is READY_FOR_INFERENCE (see "
                                    "model_eligibility.csv / required_actions.csv); no inference was run",
                            n_models=len(recs), **out)
    # ---- 3. smoke tests
    sm = a.get("smoke", {}) or {}
    smoke_rows, runnable = [], []
    for k in eligible:
        st = recs[k].checks.get("smoke_test")
        if bool(sm.get("enabled", True)) and (st != "PASS" or bool(sm.get("always", False))):
            res = run_smoke(ctx, specs[k], recs[k], int(sm.get("clips", 8)))
            smoke_rows.append(res)
            gc.collect()
            VRAMMonitor.release()
            if not res.get("ok"):
                ctx.logger.error("P24 smoke test failed for %s: %s", k, res.get("error"))
                continue
        runnable.append(k)
    ctx.save_table(pd.DataFrame([{x: y for x, y in r.items() if x != "examples"} for r in smoke_rows]), S, "smoke_tests")
    ctx.save_table(pd.DataFrame([dict(e, model_key=r["model_key"]) for r in smoke_rows for e in r.get("examples", [])]),
                   S, "smoke_examples")
    # ---- 4. inference
    runs, batches = [], []
    for k in runnable:
        eng = ASRInferenceEngine(ctx, specs[k], recs[k])
        for split in splits:
            try:
                res = eng.run_locked(split)
            except Exception as e:  # noqa: BLE001 - recorded per model; other models still run
                res = {"model_key": k, "split": split, "status": "failed", "error": f"{type(e).__name__}: {e}"}
                ctx.logger.exception("P24 %s/%s failed", k, split)
                gc.collect()
                VRAMMonitor.release()
            runs.append(res)
        bt = eng.batch_table()
        if not bt.empty:
            batches.append(bt)
        v3r.log_model(k, stage=S, last_run=str([r.get("status") for r in runs if r.get("model_key") == k]),
                      model_version=next((r.get("model_version") for r in runs if r.get("model_key") == k and r.get("model_version")), None))
    run_t = pd.DataFrame(runs)
    ctx.save_table(run_t, S, "inference_runs")
    bt = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
    ctx.save_table(bt, S, "batch_log", parquet=True)
    tp = throughput_summary(bt, runs)
    ctx.save_table(tp, S, "throughput")
    if not tp.empty and (tp["clips"] > 0).any():
        save_figure_v3(ctx, S, "gpu_throughput_memory", V.throughput(tp[tp["clips"] > 0], "ASR inference throughput and memory"),
                       tp, "GPU inference throughput and peak memory per model")
    # ---- 5. real-output inventory (accepted caches only count), lifecycle after, examples
    inv = scan_outputs(ctx, eligible, splits)
    ctx.save_table(inv, S, "asr_outputs_inventory")
    state = execution_state(ctx, keys, splits)
    ctx.save_table(state, S, "execution_state")
    s2, recs2, table2 = eligibility(ctx, True, False, False, env)
    save_eligibility(ctx, s2, recs2, table2, env, suffix="_after")
    acc = sorted(set(inv.loc[inv["accepted"], "model_key"])) if not inv.empty else []
    pref = preferred_models(specs, acc)
    ctx.save_table(pd.DataFrame([pref]), S, "preferred_models")
    ctx.save_table(_examples(ctx, [k for k in acc if specs.get(k) is None or specs[k].output != "embeddings"],
                             int(a.get("examples_per_task", 4))), S, "transcription_examples")
    if not acc:
        out = {**base, "stage_outcome": "INFERENCE_FAILED", "accepted_models": [],
               "runs": run_t[["model_key", "split", "status"]].to_dict("records") if not run_t.empty else [],
               "smoke_failed": [r["model_key"] for r in smoke_rows if not r.get("ok")]}
        write_json(ctx.stage_dir(S) / "p24_outcome.json", out)
        raise RuntimeError("P24: eligible models produced no accepted output: " + json.dumps(out, default=str)[:900])
    complete = (not state.empty and (state["state"] == "complete").all() and len(eligible) == len(keys))
    outcome = "INFERENCE_COMPLETE" if complete else "INFERENCE_PARTIAL"
    out = {**base, "inference_executed": True, "stage_outcome": outcome, "accepted_models": acc, **pref,
           "lifecycle_after": {k: r.lifecycle for k, r in recs2.items()},
           "runs": run_t[["model_key", "split", "status"]].to_dict("records") if not run_t.empty else [],
           "smoke_failed": [r["model_key"] for r in smoke_rows if not r.get("ok")]}
    write_json(ctx.stage_dir(S) / "p24_outcome.json", out)
    return {"status": "done", **out}


def _examples(ctx, models: list[str], n_ex: int) -> pd.DataFrame:
    clips = ctx.bundle().train_clips[[FILE, "task", "expected_text", "item_score"]].drop_duplicates(FILE)
    ex = []
    for k in models:
        df = ASRCheckpointStore(ctx.paths.v3_cache, k).load("train")
        if df.empty:
            continue
        m = df.merge(clips, on=FILE, how="inner")
        for t, g in m.groupby("task"):
            for r in g.sort_values(FILE).head(n_ex).to_dict("records"):
                ex.append({"model_key": k, "task": t, FILE: r[FILE], "expected_text": r["expected_text"],
                           "item_score": r["item_score"], "transcript": r.get("transcript", ""),
                           "phones": r.get("phones", ""), "ipa": r.get("ipa", ""),
                           "mean_token_logprob": r.get("mean_token_logprob"),
                           "expected_ll": r.get("expected_ll"), "precision": r.get("precision")})
    return pd.DataFrame(ex)


def outcome(ctx) -> dict:
    """The last recorded P24 outcome (notebook / downstream display)."""
    from ..io_utils import read_json

    return read_json(ctx.stage_dir(S) / "p24_outcome.json", {}) or {"stage_outcome": "NEVER_RUN",
                                                                    "inference_executed": False}
