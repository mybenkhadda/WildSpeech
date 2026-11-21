"""P22 — failure recovery, deployability, champion selection and (opt-in) validated candidate submissions.

1. Deployability of every P21 model (finite / extreme / quantile-range / fold-R2 / repeat-SD / coverage checks);
   unstable models are QUARANTINED (registry + table), never selected.
2. Diagnosis of V1's catastrophic task ensemble (instrumented re-run on V1's own fold features) and the repaired
   RobustTaskEnsemble evaluated on exactly the same folds and features.
3. Failure populations defined from the comparator's OOF errors and label-free metadata; recovery matrix:
   which modality reduces the error of which population (paired bootstrap CI).
4. Target-quantile, school and sample-rate generalisation summaries.
5. Candidate selection (conservative, best tabular, harmonized acoustic, phonetic multimodal, residual fusion,
   final ensemble) + models/v2_champion_spec.json and v2_champion_config.json.
6. Only when v2.submission.enabled: test-split feature extraction for the modalities the candidates need,
   fit on all training students, independent re-load, safety checks, validated CSVs in submissions/V2.
7. outputs/V2/scientific_report.html.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID
from ..diagnostics.failure_recovery import define_populations, population_profiles, recovery_matrix
from ..diagnostics.prediction_safety import PredictionSafetyChecker, TaskEnsembleDiagnosis, summarize_task_ensemble_diagnosis
from ..diagnostics.scientific_validation import full_metrics
from ..io_utils import write_json
from ..models.champion_selection import V2SubmissionBuilder, blend_oof, ensemble_members, select_candidates
from ..models.compact_regression import CompactSpec, StudentLevelEvaluator
from ..models.preprocessing import TabularPreprocessor
from ..models.residual_fusion import RobustTaskEnsemble
from ..validation.metrics import incremental_r2, metric_row, subgroup_metrics
from ..validation.runner import load_oof
from ..v2 import visualization_v2 as V
from ..v2.providers import build_providers
from ..v2.registry import save_figure_v2
from ._common import get_runner
from ._v2common import begin, student_meta

S = "P22"


def _fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"repeat": r, "fold": f, **metric_row(d["y_true"], d["y_pred"])}
                         for (r, f), d in oof.groupby(["repeat", "fold"])])


def _repaired_v1_task_ensemble(ctx, runner) -> pd.DataFrame:
    tasks = sorted(ctx.bundle().all_clips["task"].unique())
    rows = []
    for f in runner.folds("repeated_kfold"):
        Xtr, Xva = runner.fold_data(f)
        ytr = runner.y.reindex(Xtr.index).to_numpy(float)
        prep = TabularPreprocessor().fit(Xtr)
        Ptr, Pva = prep.transform(Xtr), prep.transform(Xva)
        for X_raw, P in ((Xtr, Ptr), (Xva, Pva)):
            for t in tasks:
                c = f"BEHAVIOR__{t}__n_items__value"
                P[f"__avail__{t}"] = (X_raw[c].fillna(0).to_numpy(float) > 0).astype(float) if c in X_raw else 1.0
        base_cols = [c for c in Ptr.columns if c.startswith(("SCREENING__", "RECORDING__"))]
        m = RobustTaskEnsemble(base_cols=base_cols, tasks=tuple(tasks), seed=ctx.seed + f.repeat * 31 + f.fold).fit(Ptr, ytr)
        rows.append(pd.DataFrame({ID: Xva.index, "repeat": f.repeat, "fold": f.fold,
                                  "y_true": runner.y.reindex(Xva.index).to_numpy(float), "y_pred": m.predict(Pva), "fold_label": ""}))
    return pd.concat(rows, ignore_index=True)


def _ensure_test_features(ctx, needs: set[str]) -> pd.DataFrame:
    """Resumable test-split extraction for every modality the selected candidates use."""
    from ..acoustic.extractor import AcousticExtractionRunner
    from ..harmonization.bandwidth import HarmonizedExtractionRunner, Variant
    from ..v2.lineage import completed
    from ..v2.providers import selected_variant

    log = []
    v1 = AcousticExtractionRunner(ctx).run("test")
    log.append({"modality": "V1 acoustic families (test)", **{k: str(v) for k, v in v1.items()}})
    if needs & {"HARM", "SOFT", "SOFTMETA"}:
        vn = selected_variant(ctx)
        vv = (ctx.cfg.get_path("v2.harmonization.variants", {}) or {}).get(vn, {})
        r = HarmonizedExtractionRunner(ctx).run("test", [Variant(vn, vv.get("lowpass_hz"), vv.get("highpass_hz"), bool(vv.get("amp_norm", False)))])
        log.append({"modality": f"harmonized {vn} (test)", **{k: str(v) for k, v in r.items()}})
    if (needs & {"PHON", "CPX", "SOFT"}) and completed(ctx, "P18"):
        from ..phonetic.runner_v2 import PhoneticV2Runner

        r = PhoneticV2Runner(ctx).run("test")
        log.append({"modality": "phonetic (test)", **{k: str(v) for k, v in r.items() if k != "eligibility"}})
    if "REF" in needs and completed(ctx, "P19"):
        from ..stages.p19_reference import _variant
        from ..reference.templates import SequenceExtractionRunner

        r = SequenceExtractionRunner(ctx, _variant(ctx)).run("test")
        log.append({"modality": "reference sequences (test)", **{k: str(v) for k, v in r.items()}})
    if (needs & {"FM", "SOFT", "REF"}) and completed(ctx, "P20"):
        from ..foundation.encoders import EmbeddingV2Runner

        r = EmbeddingV2Runner(ctx).run("test")
        log.append({"modality": "embeddings (test)", **{k: str(v) for k, v in r.get("models", {}).items()}})
    return pd.DataFrame(log)


def run(ctx) -> dict:
    reg, v2r = begin(ctx, S)
    fc = ctx.cfg.get_path("v2.failure", {}) or {}
    nb = int(fc.get("bootstrap", 1000))
    specs = ctx.read_stage_json("P21", "specs", {}) or {}
    mc = ctx.read_stage_table("P21", "model_comparison_p21")
    if mc is None or mc.empty:
        raise RuntimeError("P21 model_comparison_p21 missing — run P21 first")
    b = ctx.bundle()
    y = b.y.copy()
    y.index = y.index.astype(str)
    checker = PredictionSafetyChecker(y.to_numpy(float), ctx.cfg)

    # ---------------------------------------------------------------- deployability
    dep_rows, check_rows, oofs = [], [], {}
    for name in mc["name"].unique():
        oof = load_oof(ctx, "P21", name)
        if oof is None:
            continue
        oofs[name] = oof
        ch = checker.check(oof, _fold_metrics(oof), len(y))
        ok, bad = checker.deployable(ch)
        check_rows.append(ch.assign(model=name))
        dep_rows.append({"model": name, "deployable": ok, "failed_checks": ";".join(bad), "source": "P21"})
        if not ok:
            v2r.failures.upsert([{"failure_id": f"quarantine:{name}", "kind": "quarantine", "model": name, "reasons": ";".join(bad), "stage": S}])
    # V1 task ensemble (original) and its repair, on V1's fold features
    runner = get_runner(ctx, with_audio=True)
    v1_te = load_oof(ctx, "P11", "F_TASK_ENSEMBLE_RIDGE")
    te_summary = pd.DataFrame()
    if v1_te is not None:
        ch = checker.check(v1_te, _fold_metrics(v1_te), len(y))
        ok, bad = checker.deployable(ch)
        check_rows.append(ch.assign(model="V1:F_TASK_ENSEMBLE_RIDGE"))
        dep_rows.append({"model": "V1:F_TASK_ENSEMBLE_RIDGE", "deployable": ok, "failed_checks": ";".join(bad), "source": "P11"})
        if not ok:
            v2r.failures.upsert([{"failure_id": "quarantine:V1:F_TASK_ENSEMBLE_RIDGE", "kind": "quarantine",
                                  "model": "V1:F_TASK_ENSEMBLE_RIDGE", "reasons": ";".join(bad), "stage": S}])
        diag = TaskEnsembleDiagnosis(ctx, runner).run()
        for k, d in diag.items():
            ctx.save_table(d, S, k)
        te_summary = summarize_task_ensemble_diagnosis(diag)
        ctx.save_table(te_summary, S, "task_ensemble_diagnosis")
        rep = _repaired_v1_task_ensemble(ctx, runner)
        oofs["V1_TASK_ENSEMBLE_REPAIRED"] = rep
        ch = checker.check(rep, _fold_metrics(rep), len(y))
        ok, bad = checker.deployable(ch)
        check_rows.append(ch.assign(model="V1_TASK_ENSEMBLE_REPAIRED"))
        dep_rows.append({"model": "V1_TASK_ENSEMBLE_REPAIRED", "deployable": ok, "failed_checks": ";".join(bad), "source": "P22"})
        tcmp = pd.DataFrame([{"model": "V1:F_TASK_ENSEMBLE_RIDGE", **full_metrics(v1_te)}, {"model": "V1_TASK_ENSEMBLE_REPAIRED", **full_metrics(rep)}])
        ctx.save_table(tcmp, S, "task_ensemble_repair")
        if not diag["te_folds"].empty:
            save_figure_v2(ctx, S, "task_ensemble_fold_predictions", V.bar_ci(diag["te_folds"].assign(f=diag["te_folds"]["repeat"].astype(str) + "/" + diag["te_folds"]["fold"].astype(str)),
                           "f", "pred_max", None, None, "V1 task ensemble: max validation prediction per fold"), diag["te_folds"],
                           "Diagnosis: prediction alignment / extrapolation per fold")
    dep = pd.DataFrame(dep_rows)
    ctx.save_table(dep, S, "deployability")
    ctx.save_table(pd.concat(check_rows, ignore_index=True) if check_rows else pd.DataFrame(), S, "safety_checks")
    deployable = dict(zip(dep["model"], dep["deployable"])) if not dep.empty else {}

    # --------------------------------------------------------------- failure recovery
    base_oof = oofs.get("V2_BASELINE")
    provs = build_providers(ctx, include=("PHON", "CPX"))
    extra = pd.concat([p.table(sorted(y.index), sorted(y.index)) for p in provs.values()], axis=1) if provs else None
    cand_models = {n: o for n, o in oofs.items() if deployable.get(n, False) and n != "V2_BASELINE"
                   and (n.startswith(("P21_SINGLE_", "P21_ALL_", "P21_FUSION_", "P21_INDICES")) or n in ("V2_CONSERVATIVE", "V1_TASK_ENSEMBLE_REPAIRED"))}
    if base_oof is not None and cand_models:
        P = define_populations(ctx, base_oof, cand_models, extra, float(fc.get("quantile", 0.85)),
                               float(fc.get("extreme_target_quantile", 0.05)), float(fc.get("disagreement_quantile", 0.85)))
        ctx.save_table(P.reset_index().rename(columns={"index": ID}), S, "failure_populations")
        rec = recovery_matrix(P, base_oof, cand_models, nb, ctx.seed)
        ctx.save_table(rec, S, "recovery_matrix")
        if not rec.empty:
            save_figure_v2(ctx, S, "failure_recovery_matrix", V.recovery_heatmap(rec, "MAE reduction vs comparator by failure population"), rec,
                           "Which modality recovers which failure type")
        prof_src = extra if extra is not None else pd.DataFrame(index=P.index)
        ctx.save_table(population_profiles(P, prof_src) if prof_src.shape[1] else pd.DataFrame(), S, "population_profiles")
        for pop in P.columns:
            rr = rec[(rec["population"] == pop) & rec["recovered"]] if not rec.empty else pd.DataFrame()
            v2r.failures.upsert([{"failure_id": f"population:{pop}", "kind": "population", "n": int(P[pop].sum()), "stage": S,
                                  "recovered_by": ";".join(rr["model"].astype(str)) if not rr.empty else ""}])
        meta = student_meta(ctx).reset_index()
        q = [subgroup_metrics(o, meta, [c for c in ("instrument", "original_sample_rate") if c in meta]).assign(model=n)
             for n, o in {"V2_BASELINE": base_oof, **cand_models}.items()]
        ctx.save_table(pd.concat(q, ignore_index=True), S, "target_quantile_and_domain_diagnostics")
    allres = ctx.read_stage_table("P21", "all_results")
    if allres is not None and not allres.empty:
        gen = allres[allres["scheme"].isin(["school_kfold", "loso", "sample_rate_transfer", "sample_rate_equal_n", "instrument_transfer"])]
        ctx.save_table(gen, S, "generalization_summary")

    # ----------------------------------------------------------------- candidates
    chosen = select_candidates(mc, specs, deployable)
    members = ensemble_members(mc, chosen)
    cand_rows = []
    for role, name in chosen.items():
        r = mc[mc["name"] == name]
        cand_rows.append({"role": role, "model": name, **({k: r.iloc[0][k] for k in ("r2_mean", "r2_sd", "delta_r2", "delta_ci_low", "delta_ci_high")
                                                           if k in r} if not r.empty else {})})
    ens_oof = None
    if members and all(m_ in oofs for m_ in members):
        ens_oof = blend_oof([oofs[m_] for m_ in members])
        ch = checker.check(ens_oof, _fold_metrics(ens_oof), len(y))
        ok, bad = checker.deployable(ch)
        inc = incremental_r2(base_oof, ens_oof, nb, 0) if base_oof is not None else {}
        cand_rows.append({"role": "final_ensemble", "model": "+".join(members), **full_metrics(ens_oof), **inc, "deployable": ok,
                          "failed_checks": ";".join(bad)})
    else:
        cand_rows.append({"role": "final_ensemble", "model": chosen.get("best_tabular"),
                          "note": "no non-baseline candidate with delta-R2 CI > 0; ensemble falls back to best_tabular"})
    cands = pd.DataFrame(cand_rows)
    champion_role = "final_ensemble" if members else "best_tabular"
    spec_out = {"chosen": chosen, "ensemble_members": members, "champion_role": champion_role,
                "specs": {n: specs.get(n) for n in set(filter(None, list(chosen.values()) + members))},
                "selection_rule": "deployable only; robust score r2_mean - r2_sd within role; ensemble = equal weights of "
                                  "candidates with ΔR² CI > 0 over the strongest tabular comparator"}
    write_json(ctx.paths.models / "v2_champion_spec.json", spec_out)
    write_json(ctx.paths.models / "v2_champion_config.json", {"profile": ctx.profile, "stages": ["P22"], "config": ctx.cfg.to_dict()})

    # ----------------------------------------------------------------- submissions
    sub_rows = []
    if ctx.cfg.get_path("v2.submission.enabled", False):
        if any("subsampled" in n for n in b.notes):
            raise RuntimeError("data subsampled by profile; V2 submissions require the full test set")
        needs = set()
        for n in set(filter(None, list(chosen.values()) + members)):
            needs |= set((specs.get(n) or {}).get("providers", []))
        if ctx.cfg.get_path("v2.submission.extract_test_features", True):
            ctx.save_table(_ensure_test_features(ctx, needs), S, "test_feature_extraction")
        for k in [k for k in ctx._cache if k == "v2_clip_frame" or (isinstance(k, tuple) and k and k[0] == "v2_bank")]:
            ctx._cache.pop(k)                                   # rebuild clip tables with the new test features
        provs_full = build_providers(ctx, splits=("train", "test"))
        evf = StudentLevelEvaluator(ctx, S, providers=provs_full)
        builder = V2SubmissionBuilder(ctx, evf, S)
        test_ids = list(b.submission_format[ID].astype(str))
        preds = {}
        for role, name in chosen.items():
            if not name or not specs.get(name):
                sub_rows.append({"role": role, "status": "not_available"})
                continue
            spec = CompactSpec(**{**specs[name], "stack_models": tuple(specs[name].get("stack_models", ()))})
            missing = [p for p in spec.providers if p not in provs_full]
            if missing:
                sub_rows.append({"role": role, "model": name, "status": f"refused: test features missing for {missing}"})
                continue
            p, meta = builder.predict_member(spec, test_ids)
            preds[name] = p
            sub_rows.append({"model": name, **builder.write(role, p, meta, checker)})
        if members and all(m_ in preds for m_ in members):
            sub_rows.append({"model": "+".join(members), **builder.write("final_ensemble", np.mean([preds[m_] for m_ in members], axis=0),
                                                                          {"members": members}, checker)})
    else:
        sub_rows.append({"role": "all", "status": "test inference disabled (v2.submission.enabled=false)"})
    subs = pd.DataFrame(sub_rows)
    ctx.save_table(subs, S, "submissions")
    if not subs.empty and "role" in subs:
        st = subs.set_index("role")["status"].to_dict() if subs["role"].notna().all() else {}
        cands["submission_status"] = cands["role"].map(st)
    ctx.save_table(cands, S, "candidates")

    from ..v2.research_report_v2 import build_report

    rep = build_report(ctx)
    return {"deployable": int(dep["deployable"].sum()) if not dep.empty else 0, "quarantined": int((~dep["deployable"]).sum()) if not dep.empty else 0,
            "champion_role": champion_role, "report": rep.get("html"), "submissions": len(sub_rows)}
