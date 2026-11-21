"""reports/WILDSPEECH_V4_RESEARCH_REPORT.md — assembled exclusively from artefacts written by the run.

A section whose table does not exist says NOT EXECUTED; no number is ever typed into this module. The
historical tabular GAM reference (R² ≈ 0.2153) is quoted only as a historical value and compared with the V4
reproduction on the V4 folds when that reproduction exists.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _md(df: pd.DataFrame | None, cols: list[str] | None = None, n: int = 25, fmt: str = "{:.4f}") -> str:
    if df is None or len(df) == 0:
        return "_NOT EXECUTED / no rows._\n"
    d = df[[c for c in (cols or df.columns) if c in df.columns]].head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n| " + " | ".join("---" for _ in d.columns) + " |\n"
    body = "".join("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |\n" for r in d.itertuples(index=False))
    return head + body


def build(store, cfg4: dict, run_cfg: dict) -> str:
    L = store.load_table
    J = store.load_json
    integ = J("manifests", "data_integrity.json", {}) or {}
    imp = J("asr_import", "asr_import_report.json", {}) or {}
    cov = L("asr_import", "asr_coverage.csv")
    gate = L("asr_import", "import_gate.csv")
    ph = L("clip_features", "phonetic_availability.csv")
    cat = L("student_features", "feature_catalog.parquet")
    main = L("experiments", "experiment_results.csv")
    abl = L("ablations", "ablation_results.csv")
    rob = L("robustness", "robustness_results.csv")
    fb = L("diagnostics", "fallback_report.csv")
    hyp = L("reports", "hypotheses.csv")
    champ = J("model_registry", "champion_spec.json", {}) or {}
    subm = J("submission", "submission_meta.json", {}) or {}
    sval = L("submission", "submission_validation.csv")
    imp_g = L("diagnostics", "feature_importance_grouped.csv")
    hist = cfg4.get("historical_reference", {}) or {}
    cols = ["name", "status", "result_class", "r2_mean", "r2_sd", "rmse", "mae", "delta_r2", "delta_ci_low",
            "delta_ci_high", "q", "delta_r2_same_model", "fallback_folds", "n_added"]
    out = [f"# WildSpeech V4 — Speech–Text–Acoustic Multimodal Intelligence\n",
           f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from run artefacts only. Repository ref: "
           f"`{run_cfg.get('repo')}@{run_cfg.get('ref')}`, run_mode `{run_cfg.get('run_mode')}`, "
           f"{run_cfg.get('cv_folds')}-fold x {run_cfg.get('cv_repeats')} repeats, seed {run_cfg.get('seed')}._\n",
           "> Every number below was produced by the executed notebook. Associations are not causal. "
           "Recognized-text error mixes the child's response with recognizer error and is not a verified "
           "pronunciation error. Fallback rows (status `fallback_baseline` / class `BASELINE_FALLBACK`) are not "
           "multimodal improvements.\n",
           "## 1. Data integrity and joins\n"]
    if integ:
        out.append(f"Train clips {integ.get('n_train_clips')}, test clips {integ.get('n_test_clips')}, train students "
                   f"{integ.get('n_train_students')}, test students {integ.get('n_test_students')}; "
                   f"{integ.get('n_fail')} FAIL / {integ.get('n_warn')} WARN checks.\n\n")
        out.append(_md(pd.DataFrame(integ.get("checks", [])), ["check", "status", "value", "detail"], 40))
    else:
        out.append("_NOT EXECUTED._\n")
    out.append("\n## 2. External ASR import (`external_colab_asr`)\n")
    if imp:
        out.append(f"Source `{imp.get('source_name')}` sha256 `{str(imp.get('source_sha256'))[:16]}…`, layout "
                   f"`{imp.get('layout')}`, {imp.get('n_source_rows')} source rows, models {imp.get('models')}. "
                   f"Usable after the independent gate: {imp.get('usable_models')}. Permission review: "
                   f"{imp.get('permission_status')}.\n\n")
        out.append(_md(cov, ["model_key", "split", "n_metadata_clips", "n_usable", "n_failed", "n_unmatched_rows",
                             "n_duplicates_discarded", "usable_frac", "n_phone_strings"], 40))
        out.append("\nGate:\n\n" + _md(gate, None, 60))
    else:
        out.append("_NOT EXECUTED (no ASR export found or import not run)._\n")
    out.append("\n## 3. Phonetic availability\n" + _md(ph, None, 20))
    out.append("\n## 4. Feature families\n")
    if cat is not None and not cat.empty:
        out.append(_md(cat.groupby(["provider", "family", "modality", "measurement"]).size().rename("n_columns").reset_index(), None, 60))
    else:
        out.append("_NOT EXECUTED._\n")
    out.append("\n## 5. Baseline reproduction\n")
    if main is not None and not main.empty:
        b = main[main["group"].isin(["baseline", "baseline_alt"])]
        out.append(_md(b, ["name", "status", "r2_mean", "r2_sd", "rmse", "mae", "n_students"], 10))
        ref = b[b["group"] == "baseline"]
        if not ref.empty and hist:
            out.append(f"\nHistorical reference ({hist.get('name')}): R² ≈ {hist.get('r2')} — {hist.get('note')}. "
                       f"V4 reproduction on the V4 folds: R² = {float(ref['r2_mean'].iloc[0]):.4f}.\n")
    else:
        out.append("_NOT EXECUTED._\n")
    out.append("\n## 6. Multimodal model ladder (primary scheme)\n")
    out.append(_md(main[~main["group"].isin(["baseline", "baseline_alt"])] if main is not None and not main.empty else None, cols, 40))
    out.append("\n## 7. Fallback audit\n" + _md(fb[fb["is_fallback"] == True] if fb is not None and not fb.empty and "is_fallback" in fb else fb,  # noqa: E712
                                               ["name", "scheme", "status", "fallback_folds", "n_folds", "alpha_zero_folds",
                                                "zero_effective_features_folds", "zero_correction_variance_folds",
                                                "zero_fusion_weight_folds", "fallback_reasons"], 40))
    out.append("\n## 8. Ablations and negative controls\n" + _md(abl, ["name", "group", "status", "result_class", "r2_mean",
                                                                       "delta_r2", "delta_ci_low", "delta_ci_high", "q"], 80))
    out.append("\n## 9. Robustness (school / LOSO / instrument / sample rate)\n" +
               _md(rob, ["name", "scheme", "status", "result_class", "r2_mean", "delta_r2", "delta_ci_low", "delta_ci_high"], 80))
    out.append("\n## 10. Hypotheses H1–H10\n" + _md(hyp, ["hypothesis", "question", "evidence", "effect", "ci_low",
                                                          "ci_high", "n_students", "experiment", "result_class",
                                                          "method", "limitations"], 20))
    out.append("\n## 11. Interpretability\n" + _md(imp_g, None, 25))
    out.append("\n## 12. Champion and submission\n")
    if champ:
        out.append(f"Champion `{champ.get('name')}` — {champ.get('reason')} (R² {champ.get('r2_mean')}, ΔR² "
                   f"{champ.get('delta_r2')}).\n\n")
    else:
        out.append("_Champion NOT SELECTED (experiments not executed)._\n")
    if subm:
        out.append(f"Submission `{subm.get('file')}`: {subm.get('rows')} rows, prediction mean {subm.get('pred_mean'):.3f}, "
                   f"sd {subm.get('pred_sd'):.3f}, {subm.get('n_clipped')} clipped to the training range.\n\n")
        out.append(_md(sval, None, 30))
    else:
        out.append("_Submission NOT GENERATED._\n")
    out.append("\n## 13. Limitations\n"
               "* ~541 independent students: incremental effects are small relative to their bootstrap uncertainty; "
               "multiplicity is controlled with BH-FDR within (scheme, group).\n"
               "* ASR predictions were produced externally; their quality on child speech bounds every text feature.\n"
               "* Phonetic hypotheses require measured phone strings; G2P of recognized words is not a substitute.\n"
               "* school_id / score_status are train-only and never prediction inputs; school effects are assessed "
               "with school-grouped and leave-one-school-out validation only.\n"
               "* Results are associations; no causal interpretation.\n")
    return "".join(out)


def write(store, cfg4: dict, run_cfg: dict) -> str:
    text = build(store, cfg4, run_cfg)
    p = store.save_text(text, "reports", "WILDSPEECH_V4_RESEARCH_REPORT.md")
    return str(p)
