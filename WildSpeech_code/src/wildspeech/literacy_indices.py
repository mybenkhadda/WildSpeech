"""Section 19 - experimental literacy index discovery.

Each index is an interpretable composite of validated component features with
pre-declared directions, robust train-fitted scaling, missing-component
handling and an equal-weight versus regularised-weight comparison. These are
EXPERIMENTAL research constructs, not established educational assessments,
and an index is retained only if the evidence supports it.

Index scaling is always refitted on training folds. Components derived from
item statistics (irt_/ia_) use the all-training EDA versions in this stage;
the fully fold-fitted evaluation of the indices is experiment A14 in Section 20.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .stats_utils import bootstrap_ci, safe_corr, spearman_stat

# direction +1: higher component = more of the construct
INDEX_SPECS = {
    "L01_PHONOLOGICAL_PRECISION": {"motivation": "Expected phoneme alignment confidence and sequence preservation.",
                                    "components": [(r"^ph_.*_seq_confidence__mean$", 1), (r"^ph_.*_phone_order_preservation__mean$", 1),
                                                   (r"^ph_.*_phone_er__mean$", -1)]},
    "L02_PHONOLOGICAL_STABILITY": {"motivation": "Within-student consistency of phoneme confidence across comparable items.",
                                    "components": [(r"^ph_.*_seq_confidence__std$", -1), (r"^ia_ph_.*_seq_confidence__std$", -1)]},
    "L03_ARTICULATION_CONSISTENCY": {"motivation": "Item-conditioned acoustic transition consistency.",
                                      "components": [(r"^ia_m_sp_transition_rate__std$", -1), (r"^ia_m_sp_cepstral_variability__std$", -1),
                                                     (r"^m_sp_transition_rate__std$", -1)]},
    "L04_FLUENCY_DISRUPTION": {"motivation": "Internal pauses, discontinuities and restarts.",
                                "components": [(r"^t_n_pauses__mean$", 1), (r"^t_pause_speech_ratio__mean$", 1), (r"^t_internal_sil__mean$", 1),
                                               (r"^t_n_long_pauses__mean$", 1), (r"^ph_.*_self_corrections__mean$", 1)]},
    "L05_COMPLEXITY_SENSITIVITY": {"motivation": "Change of performance and acoustics with item complexity.",
                                    "components": [(r"^cx_t_active_dur__vs_.*__all$", 1), (r"^cx_correct__vs_.*__all$", -1),
                                                   (r"^cx_t_n_pauses__vs_.*__all$", 1)]},
    "L06_EXPECTED_TEXT_DEVIATION": {"motivation": "Expected-versus-recognised speech differences.",
                                     "components": [(r"^ph_.*_word_er__mean$", 1), (r"^ph_.*_char_er__mean$", 1), (r"^ph_.*_phone_er__mean$", 1)]},
    "L07_VERBAL_SEQUENCING": {"motivation": "Sequence completeness and order preservation.",
                               "components": [(r"^ph_.*_word_order_preservation__mean$", 1), (r"^ph_.*_word_completeness__mean$", 1),
                                              (r"^ph_.*_phone_completeness__mean$", 1)]},
    "L08_ACOUSTIC_REGULARITY": {"motivation": "Temporal and spectral consistency.",
                                 "components": [(r"^t_rhythm_regularity__mean$", 1), (r"^s_sp_stationarity__mean$", -1),
                                                (r"^s_sp_entropy_mean__std$", -1), (r"^t_npvi__mean$", -1)]},
    "L09_RESPONSE_CONSISTENCY": {"motivation": "Variability across comparable responses.",
                                  "components": [(r"^ia_t_active_dur__std$", -1), (r"^ia_item_adjusted_acoustic_variability$", -1),
                                                 (r"^t_active_dur__cv$", -1)]},
    "L10_UNEXPECTED_ERROR": {"motivation": "Failures on items expected to be easy given estimated ability.",
                              "components": [(r"^irt_.*_unexpected_easy_item_error_rate$", 1), (r"^irt_.*_response_residual_mean$", -1)]},
    "L11_CROSS_TASK_ASYMMETRY": {"motivation": "Differences between task-specific performance and speech behaviour.",
                                  "components": [(r"^xt_skill_asymmetry_accuracy_range$", 1), (r"^xt_acoustic_asymmetry$", 1),
                                                 (r"^xt_accuracy_between_task_std$", 1)]},
    "L12_ERROR_RECOVERY": {"motivation": "Recovery from unsuccessful responses (valid order required).",
                            "components": [(r"^traj_all_error_recovery_rate$", 1), (r"^traj_all_longest_error_streak$", -1)]},
    "L13_SPECTROTEMPORAL_COMPLEXITY": {"motivation": "Complementary multiscale descriptors.",
                                        "components": [(r"^s_sp_tf_entropy__mean$", 1), (r"^w_.*_entropy__mean$", 1), (r"^w_wp_entropy__mean$", 1),
                                                       (r"^m_sp_cepstral_traj_length_per_s__mean$", 1)]},
    "L14_RESPONSE_EFFICIENCY": {"motivation": "Performance relative to item-conditioned active speech duration.",
                                 "components": [(r"^acc_all$", 1), (r"^ia_item_adjusted_duration$", -1)]},
    "L15_LITERACY_PROFILE_IMBALANCE": {"motivation": "Imbalance among initial literacy skills.",
                                        "components": [(r"^theta_skill_imbalance$", 1), (r"^theta_range$", 1), (r"^theta_std$", 1)]},
}


def resolve_components(columns, spec: dict) -> list:
    comps = []
    for pattern, direction in spec["components"]:
        rx = re.compile(pattern)
        matches = [c for c in columns if rx.search(c)]
        comps += [(m, direction) for m in matches[:3]]
    return comps


def compute_indices(base: pd.DataFrame, fit_ids, cfg: dict, extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Equal-weight indices with robust scaling fitted on fit_ids only.

    EDA-only fold-sensitive columns (irt_/ia_) in ``base`` are ignored; pass
    fold-fitted versions through ``extra``.
    """
    B = base[[c for c in base.columns if not c.startswith(("irt_", "ia_", "idx_", "clu_"))]]
    if extra is not None and not extra.empty:
        B = B.join(extra[[c for c in extra.columns if c not in B.columns]])
    fit_ids = [i for i in fit_ids if i in B.index]
    out = pd.DataFrame(index=B.index)
    report = {}
    min_c = int(cfg["index_min_components"])
    for name, spec in INDEX_SPECS.items():
        comps = resolve_components(B.columns, spec)
        if len(comps) < min_c and not (len(comps) == 1 and min_c <= 1):
            report[name] = {"status": "UNAVAILABLE", "n_components": len(comps)}
            continue
        Z = []
        for col, direction in comps:
            x = B[col]
            ref = x.loc[fit_ids]
            med = ref.median()
            iqr = ref.quantile(0.75) - ref.quantile(0.25)
            if not np.isfinite(iqr) or iqr == 0:
                continue
            Z.append(direction * ((x - med) / iqr).clip(-5, 5))
        if len(Z) < min_c:
            report[name] = {"status": "UNAVAILABLE", "n_components": len(Z)}
            continue
        Zdf = pd.concat(Z, axis=1)
        n_avail = Zdf.notna().sum(axis=1)
        out[f"idx_{name}"] = Zdf.mean(axis=1).where(n_avail >= min_c)
        out[f"idx_{name}_n_components"] = n_avail
        report[name] = {"status": "OK", "components": [c for c, _ in comps], "directions": [d for _, d in comps]}
    return out, report


def cronbach_alpha(Z: pd.DataFrame) -> float:
    Z = Z.dropna()
    k = Z.shape[1]
    if k < 2 or len(Z) < 10:
        return np.nan
    var_items = Z.var(ddof=1).sum()
    var_total = Z.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - var_items / var_total)) if var_total > 0 else np.nan


def stage_literacy_indices(ctx) -> dict:
    from .ablation import ColumnSetBuilder, cv_evaluate
    from .validation import paired_bootstrap_delta, pooled_predictions

    cfg = ctx.config
    S = da.student_features(ctx, "train")
    meta = da.student_meta(ctx, "train")
    y = meta["target"].dropna().astype(float)
    S = S.reindex(y.index)
    res = da.residuals(ctx).reindex(y.index)
    assign = ctx.load_df("cv_assignments")
    numeric = S[[c for c in S.columns if pd.api.types.is_numeric_dtype(S[c])]]
    eda_extra = numeric[[c for c in numeric.columns if c.startswith(("irt_", "ia_"))]]
    idx_all, report = compute_indices(numeric, list(numeric.index), cfg, extra=eda_extra)
    ctx.save_df(idx_all.reset_index().rename(columns={"index": "student_id"}), "literacy_indices")
    screening = da.raw_theta_columns(numeric)
    schools = meta["school_id"].reindex(y.index)
    groups = schools if schools.notna().any() else None
    base_res = cv_evaluate(ColumnSetBuilder(numeric, screening), None, y, assign, "CV_A_REPEATED_STUDENT_KFOLD", cfg, repeats=[0])
    p0 = pooled_predictions(base_res.oof)
    rows = []
    for name, info in report.items():
        row = {"index": name, "motivation": INDEX_SPECS[name]["motivation"], "status": info["status"]}
        if info["status"] != "OK":
            rows.append(row)
            continue
        col = f"idx_{name}"
        comps = info["components"]
        row["components"] = ";".join(comps)
        row["directions"] = ";".join(map(str, info["directions"]))
        Z = pd.concat([d * numeric.get(c, eda_extra.get(c)) for c, d in zip(comps, info["directions"])], axis=1)
        row["cronbach_alpha"] = cronbach_alpha(Z)
        row["missing_rate"] = float(idx_all[col].isna().mean())
        row["target_rho"], row["target_p"], row["n"] = safe_corr(idx_all[col], y, "spearman", 20)
        row["residual_rho"], row["residual_p"], _ = safe_corr(idx_all[col], res, "spearman", 20)
        comp_rhos = [abs(safe_corr(Z.iloc[:, j], res, "spearman", 20)[0]) for j in range(Z.shape[1])]
        row["best_component_abs_residual_rho"] = float(np.nanmax(comp_rhos)) if np.isfinite(comp_rhos).any() else np.nan
        sub = pd.DataFrame({"i": idx_all[col], "r": res}).dropna()
        ci = bootstrap_ci(spearman_stat, sub["i"].to_numpy(), sub["r"].to_numpy(), n_boot=int(cfg["bootstrap_iterations"]),
                          seed=int(cfg["seed"]), groups=groups.reindex(sub.index).to_numpy() if groups is not None else None)
        row["residual_rho_ci_lower"], row["residual_rho_ci_upper"] = ci["lower"], ci["upper"]
        signs = []
        for s, ids in schools.dropna().groupby(schools.dropna()).groups.items():
            if len(ids) >= int(cfg["min_group_size"]):
                r = safe_corr(idx_all.loc[ids, col], res.reindex(ids), "spearman", int(cfg["min_group_size"]))[0]
                if np.isfinite(r) and np.isfinite(row["residual_rho"]):
                    signs.append(np.sign(r) == np.sign(row["residual_rho"]))
        row["cross_school_sign_consistency"] = float(np.mean(signs)) if signs else np.nan
        # incremental contribution: fold-fitted equal-weight index vs regularised components, both with screening
        def eq_builder():
            class _B:
                columns = screening

                def build(self, tr, va, families=None, key=None):
                    idx, _ = compute_indices(numeric, tr, cfg, extra=eda_extra)
                    X = numeric[screening].join(idx[[col]])
                    return X.loc[list(tr)], X.loc[list(va)]
            return _B()
        r_eq = cv_evaluate(eq_builder(), None, y, assign, "CV_A_REPEATED_STUDENT_KFOLD", cfg, repeats=[0])
        comp_cols = [c for c in comps if c in numeric.columns]
        r_reg = cv_evaluate(ColumnSetBuilder(numeric, screening + comp_cols), None, y, assign, "CV_A_REPEATED_STUDENT_KFOLD",
                            cfg, repeats=[0]) if comp_cols else None
        pe = pooled_predictions(r_eq.oof)
        d = paired_bootstrap_delta(pe["y"], pe["pred"], p0.reindex(pe.index)["pred"], cfg, "r2")
        row.update({"delta_r2_equal_weight": d["delta"], "delta_r2_equal_weight_ci_lower": d["ci_lower"],
                    "delta_r2_equal_weight_ci_upper": d["ci_upper"]})
        if r_reg is not None:
            pr = pooled_predictions(r_reg.oof)
            d2 = paired_bootstrap_delta(pr["y"], pr["pred"], p0.reindex(pr.index)["pred"], cfg, "r2")
            row.update({"delta_r2_regularised_components": d2["delta"], "delta_r2_regularised_ci_lower": d2["ci_lower"]})
        retain = (np.isfinite(row.get("delta_r2_equal_weight_ci_lower", np.nan)) and row["delta_r2_equal_weight_ci_lower"] > 0
                  and (row["cross_school_sign_consistency"] if np.isfinite(row["cross_school_sign_consistency"]) else 0) >= 0.7)
        row["retention"] = "RETAIN" if retain else ("INVESTIGATE" if np.isfinite(row["residual_rho_ci_lower"]) and
                                                    (row["residual_rho_ci_lower"] > 0 or row["residual_rho_ci_upper"] < 0) else "DO_NOT_RETAIN")
        rows.append(row)
    rep = pd.DataFrame(rows)
    ctx.save_df(rep, "literacy_index_report", also_csv=True)
    ok = rep[rep["status"] == "OK"] if "status" in rep else rep
    if len(ok):
        o = ok.sort_values("residual_rho")
        fig, axes = viz.new_figure(1, 1, (8, 0.35 * len(o) + 2))
        viz.forest(axes[0, 0], o, "index", "residual_rho", "residual_rho_ci_lower", "residual_rho_ci_upper",
                   "Experimental literacy indices: association with OOF screening residual", "Spearman rho (cluster bootstrap 95% CI)")
        ctx.figure(fig, "literacy_indices_forest", "Experimental literacy indices", "Which composite indices are worth retaining?",
                   "literacy_index_report", int(len(y)), "Experimental constructs, not established assessments.")
    return {"outputs": ["literacy_indices", "literacy_index_report"],
            "summary": rep[["index", "status"] + [c for c in ("retention",) if c in rep]].to_dict("records")}
