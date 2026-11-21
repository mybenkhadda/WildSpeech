"""Section 20 - predictive feature-family ablation.

A controlled experiment matrix (A00-A23) evaluated with student-disjoint CV
(repeated student K-fold and school-grouped K-fold), conservative model
settings, nested supervised selection inside the model pipeline and paired
bootstrap comparisons. Unavailable modalities are reported as skipped and
never replaced with invented data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .baseline_models import ModelUnavailable, make_fit_predict, model_available
from .families import AUDIO_FAMILIES, TABULAR_FAMILIES, modeling_columns
from .validation import available_schemes, paired_bootstrap_delta, pooled_predictions, run_cv

TAB = list(TABULAR_FAMILIES)
AUDIO_CORE = ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC", "ITEM_ADJUSTED_AUDIO",
              "COMPLEXITY_SENSITIVITY", "CORRECTNESS_CONTRAST", "CROSS_TASK", "EMBEDDINGS"]
EXPERIMENTS = {
    "A00_TARGET_MEAN": [],
    "A01_THETA_ONLY": ["SCREENING"],
    "A02_THETA_CORRECTNESS": ["SCREENING", "CORRECTNESS"],
    "A03_THETA_ITEM_DIFFICULTY": ["SCREENING", "ITEM_DIFFICULTY"],
    "A04_THETA_ADAPTIVE_BEHAVIOR": ["SCREENING", "ADAPTIVE"],
    "A05_COMPLETE_TABULAR": TAB,
    "A06_TABULAR_TEMPORAL": TAB + ["TEMPORAL"],
    "A07_TABULAR_SPECTRAL": TAB + ["SPECTRAL"],
    "A08_TABULAR_MFCC": TAB + ["MFCC"],
    "A09_TABULAR_WAVELETS": TAB + ["WAVELETS"],
    "A10_TABULAR_PROSODY": TAB + ["PROSODY"],
    "A11_TABULAR_PHONETIC": TAB + ["PHONETIC"],
    "A12_TABULAR_ITEM_ADJUSTED_AUDIO": TAB + ["ITEM_ADJUSTED_AUDIO", "COMPLEXITY_SENSITIVITY", "CORRECTNESS_CONTRAST"],
    "A13_TABULAR_MULTIMODAL": TAB + AUDIO_CORE,
    "A14_TABULAR_LITERACY_INDICES": TAB + ["LITERACY_INDICES"],
    "A15_TABULAR_CLUSTER_FEATURES": TAB + ["CLUSTER"],
    "A16_AUDIO_ONLY": ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "ITEM_ADJUSTED_AUDIO", "CORRECTNESS_CONTRAST",
                       "COMPLEXITY_SENSITIVITY", "CROSS_TASK"],
    "A17_PHONETIC_ONLY": ["PHONETIC"],
    "A18_TEMPORAL_ONLY": ["TEMPORAL"],
    "A19_SPECTRAL_ONLY": ["SPECTRAL", "MFCC"],
    "A20_WAVELETS_ONLY": ["WAVELETS"],
    "A21_BACKGROUND_ONLY": ["BACKGROUND"],
    "A22_ALL_VALIDATED_FEATURES": TAB + AUDIO_CORE + ["LITERACY_INDICES"],
    "A23_RESIDUAL_MULTIMODAL": TAB + AUDIO_CORE,
}
# Families an experiment is about; a single-purpose experiment is skipped when any is unavailable,
# a composite experiment only when all are unavailable.
KEY_FAMILIES = {name: [f for f in fams if f not in (["SCREENING"] if name[:3] in {"A02", "A03", "A04"} else TAB)]
                for name, fams in EXPERIMENTS.items()}
KEY_FAMILIES["A05_COMPLETE_TABULAR"] = ["CORRECTNESS", "ITEM_DIFFICULTY", "ADAPTIVE", "LINGUISTIC"]
for _single in ("A17_PHONETIC_ONLY", "A18_TEMPORAL_ONLY", "A19_SPECTRAL_ONLY", "A20_WAVELETS_ONLY", "A21_BACKGROUND_ONLY", "A16_AUDIO_ONLY"):
    KEY_FAMILIES[_single] = list(EXPERIMENTS[_single])
COMPOSITE_EXPERIMENTS = {"A05_COMPLETE_TABULAR", "A13_TABULAR_MULTIMODAL", "A16_AUDIO_ONLY", "A19_SPECTRAL_ONLY",
                         "A22_ALL_VALIDATED_FEATURES", "A23_RESIDUAL_MULTIMODAL"}
MULTI_MODEL_EXPERIMENTS = ["A01_THETA_ONLY", "A05_COMPLETE_TABULAR", "A13_TABULAR_MULTIMODAL", "A22_ALL_VALIDATED_FEATURES"]


class ColumnSetBuilder:
    """Builder over a fixed numeric table (no fold-fitted features)."""

    def __init__(self, base: pd.DataFrame, columns: list):
        self.base = base
        self.columns = [c for c in columns if c in base.columns]

    def build(self, train_ids, val_ids, families=None, key=None):
        X = self.base[self.columns]
        X = X.loc[:, X.loc[list(train_ids)].notna().any()]
        return X.loc[list(train_ids)], X.loc[list(val_ids)]

    def columns_for(self, families):
        return self.columns


def family_available(fam: str, builder, ctx) -> bool:
    if fam in {"ITEM_DIFFICULTY"}:
        cm = da.clips(ctx, "train")
        return bool(ctx.config.get("enable_irt")) and "correct" in cm.columns and cm["correct"].notna().any()
    if fam == "ITEM_ADJUSTED_AUDIO":
        return ctx.stage_available("06") or bool(modeling_columns(builder.base.columns, [fam]))
    if fam == "LITERACY_INDICES":
        from .literacy_indices import compute_indices

        idx, _ = compute_indices(builder.base, list(builder.base.index), ctx.config)
        return idx.shape[1] > 0
    if fam == "CLUSTER":
        from .clustering import prepare_space

        Z, _ = prepare_space(builder.base, "C09_SCREENING_ADJUSTED_AUDIO")
        return Z is not None
    return bool(builder.columns_for({fam}))


def cv_evaluate(builder, families, y: pd.Series, assign: pd.DataFrame, scheme: str, cfg: dict, model: str = "RIDGE",
                repeats=None, groups=None, label: str = ""):
    fp = make_fit_predict(builder, set(families or []), model, cfg, y)
    return run_cv(y, assign, scheme, fp, cfg, repeats=repeats, groups=groups, label=label)


def _record(rows, oofs, name, model, scheme, res, families):
    rows.append({"experiment": name, "model": model, "scheme": scheme, "families": "+".join(families), **res.summary})
    oofs.append(res.oof.assign(experiment=name, model=model, scheme=scheme))


def stage_predictive_ablation(ctx) -> dict:
    from .fold_features import FoldFeatureBuilder
    from .residual_models import nested_two_stage_cv

    cfg = ctx.config
    meta = da.student_meta(ctx, "train")
    y = meta["target"].dropna().astype(float)
    assign = ctx.load_df("cv_assignments")
    builder = FoldFeatureBuilder(ctx)
    builder.base = builder.base.reindex(y.index)
    schemes = [s for s in ("CV_A_REPEATED_STUDENT_KFOLD", "CV_B_SCHOOL_GROUP_KFOLD") if s in available_schemes(assign)]
    groups = meta["school_id"] if meta["school_id"].notna().any() else None
    primary = cfg["ablation_primary_model"]
    avail_cache: dict = {}

    def avail(fam):
        if fam not in avail_cache:
            avail_cache[fam] = family_available(fam, builder, ctx)
        return avail_cache[fam]

    rows, oofs, status = [], [], []
    for name, fams in EXPERIMENTS.items():
        if name == "A23_RESIDUAL_MULTIMODAL":
            continue
        present = [f for f in fams if avail(f)]
        keys = KEY_FAMILIES.get(name, [])
        missing = [f for f in keys if not avail(f)]
        if name in COMPOSITE_EXPERIMENTS:
            skip = bool(keys) and len(missing) == len(keys)
        else:
            skip = bool(missing)
        if name != "A00_TARGET_MEAN" and (skip or not present):
            status.append({"experiment": name, "status": "SKIPPED_UNAVAILABLE", "missing_families": ",".join(missing)})
            continue
        model = "MEAN" if name == "A00_TARGET_MEAN" else primary
        for scheme in schemes:
            res = cv_evaluate(builder, present, y, assign, scheme, cfg, model, groups=groups, label=name)
            _record(rows, oofs, name, model, scheme, res, present)
        status.append({"experiment": name, "status": "COMPLETE", "families_used": "+".join(present),
                       "missing_families": ",".join(missing)})
        ctx.logger.info("Ablation %s complete", name)
    for name in MULTI_MODEL_EXPERIMENTS:
        done = [s for s in status if s["experiment"] == name and s["status"] == "COMPLETE"]
        if not done:
            continue
        fams = done[0]["families_used"].split("+")
        for m in cfg["ablation_models"]:
            if m == primary:
                continue
            ok, reason = model_available(m, cfg)
            if not ok:
                status.append({"experiment": name, "model": m, "status": "MODEL_UNAVAILABLE", "reason": reason})
                continue
            res = cv_evaluate(builder, fams, y, assign, schemes[0], cfg, m, repeats=[0], groups=groups, label=name)
            _record(rows, oofs, name, m, schemes[0], res, fams)
    a23 = [f for f in EXPERIMENTS["A23_RESIDUAL_MULTIMODAL"] if avail(f)]
    if any(f in a23 for f in AUDIO_CORE):
        for scheme in schemes:
            res = nested_two_stage_cv(builder, y, assign, scheme, cfg, stage1_families=["SCREENING"] + [f for f in TAB if avail(f) and f != "SCREENING"],
                                      stage2_families=[f for f in AUDIO_CORE if avail(f)], groups=groups)
            _record(rows, oofs, "A23_RESIDUAL_MULTIMODAL", f"TWO_STAGE_{primary}", scheme, res, a23)
        status.append({"experiment": "A23_RESIDUAL_MULTIMODAL", "status": "COMPLETE"})
    else:
        status.append({"experiment": "A23_RESIDUAL_MULTIMODAL", "status": "SKIPPED_UNAVAILABLE", "missing_families": "audio"})
    results = pd.DataFrame(rows)
    oof = pd.concat(oofs, ignore_index=True) if oofs else pd.DataFrame()
    results = add_comparisons(results, oof, cfg)
    loo = leave_one_family_out(builder, y, assign, schemes[0], cfg, primary, avail, groups)
    pair = pairwise_families(builder, y, assign, schemes[0], cfg, primary, avail, groups) if cfg.get("ablation_pairwise_families") else pd.DataFrame()
    fwd = nested_forward_selection(builder, y, assign, schemes[0], cfg, primary, avail)
    red = redundancy_analysis(oof, results)
    ctx.save_df(results, "ablation_results", also_csv=True)
    ctx.save_df(oof, "ablation_oof_predictions")
    ctx.save_df(pd.DataFrame(status), "ablation_status", fmt="csv")
    ctx.save_df(loo, "ablation_leave_one_family_out", fmt="csv")
    ctx.save_df(pair, "ablation_pairwise_families", fmt="csv")
    ctx.save_df(fwd, "ablation_forward_selection", fmt="csv")
    ctx.save_df(red, "ablation_redundancy", fmt="csv")
    _ablation_figures(ctx, results, loo, pair)
    return {"outputs": ["ablation_results", "ablation_oof_predictions", "ablation_status", "ablation_leave_one_family_out",
                        "ablation_pairwise_families", "ablation_forward_selection", "ablation_redundancy"],
            "summary": results[(results["model"] == primary) & (results["scheme"] == schemes[0])][
                ["experiment", "pooled_r2", "delta_r2_vs_screening", "delta_r2_vs_full_tabular"]].to_dict("records")
            if len(results) else {}}


def add_comparisons(results: pd.DataFrame, oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy()
    for col in ("delta_r2_vs_screening", "delta_r2_vs_screening_ci_lower", "delta_r2_vs_screening_ci_upper",
                "delta_r2_vs_full_tabular", "delta_r2_vs_full_tabular_ci_lower", "delta_r2_vs_full_tabular_ci_upper",
                "school_generalization_gap", "confidence_interval_lower", "confidence_interval_upper", "fold_std"):
        out[col] = np.nan
    pooled = {k: pooled_predictions(g) for k, g in oof.groupby(["experiment", "model", "scheme"])}
    primary = cfg["ablation_primary_model"]
    for i, r in out.iterrows():
        key = (r["experiment"], r["model"], r["scheme"])
        if key not in pooled:
            continue
        p = pooled[key]
        for ref, label in (("A01_THETA_ONLY", "screening"), ("A05_COMPLETE_TABULAR", "full_tabular")):
            rk = (ref, primary, r["scheme"])
            if rk in pooled and ref != r["experiment"]:
                common = p.index.intersection(pooled[rk].index)
                d = paired_bootstrap_delta(p.loc[common, "y"], p.loc[common, "pred"], pooled[rk].loc[common, "pred"], cfg, "r2")
                out.loc[i, f"delta_r2_vs_{label}"] = d["delta"]
                out.loc[i, f"delta_r2_vs_{label}_ci_lower"] = d["ci_lower"]
                out.loc[i, f"delta_r2_vs_{label}_ci_upper"] = d["ci_upper"]
        out.loc[i, "confidence_interval_lower"] = r.get("pooled_r2_ci_lower")
        out.loc[i, "confidence_interval_upper"] = r.get("pooled_r2_ci_upper")
        out.loc[i, "fold_std"] = r.get("std_fold_r2")
    for (exp, model), g in out.groupby(["experiment", "model"]):
        a = g[g["scheme"] == "CV_A_REPEATED_STUDENT_KFOLD"]["pooled_r2"]
        b = g[g["scheme"] == "CV_B_SCHOOL_GROUP_KFOLD"]["pooled_r2"]
        if len(a) and len(b):
            out.loc[g.index, "school_generalization_gap"] = float(a.iloc[0] - b.iloc[0])
    out = out.rename(columns={"pooled_r2": "r2", "pooled_rmse": "rmse", "pooled_mae": "mae"})
    out["pooled_r2"] = out["r2"]
    return out


def leave_one_family_out(builder, y, assign, scheme, cfg, model, avail, groups) -> pd.DataFrame:
    fams = [f for f in EXPERIMENTS["A13_TABULAR_MULTIMODAL"] if avail(f)]
    if len(fams) < 2:
        return pd.DataFrame()
    full = cv_evaluate(builder, fams, y, assign, scheme, cfg, model, repeats=[0], groups=groups)
    pf = pooled_predictions(full.oof)
    rows = []
    for f in fams:
        sub = [x for x in fams if x != f]
        res = cv_evaluate(builder, sub, y, assign, scheme, cfg, model, repeats=[0], groups=groups)
        ps = pooled_predictions(res.oof)
        d = paired_bootstrap_delta(pf["y"], pf["pred"], ps.reindex(pf.index)["pred"], cfg, "r2")
        rows.append({"dropped_family": f, "r2_without": res.summary.get("pooled_r2"), "r2_full": full.summary.get("pooled_r2"),
                     "loss_when_dropped": d["delta"], "ci_lower": d["ci_lower"], "ci_upper": d["ci_upper"]})
    return pd.DataFrame(rows)


def pairwise_families(builder, y, assign, scheme, cfg, model, avail, groups) -> pd.DataFrame:
    audio = [f for f in ("TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC", "ITEM_ADJUSTED_AUDIO") if avail(f)]
    tab = [f for f in TAB if avail(f)]
    base = cv_evaluate(builder, tab, y, assign, scheme, cfg, model, repeats=[0], groups=groups)
    r_base = base.summary.get("pooled_r2", np.nan)
    single = {}
    for f in audio:
        single[f] = cv_evaluate(builder, tab + [f], y, assign, scheme, cfg, model, repeats=[0], groups=groups).summary.get("pooled_r2", np.nan) - r_base
    rows = []
    for i, a in enumerate(audio):
        for b in audio[i + 1:]:
            r = cv_evaluate(builder, tab + [a, b], y, assign, scheme, cfg, model, repeats=[0], groups=groups).summary.get("pooled_r2", np.nan)
            gain = r - r_base
            rows.append({"family_a": a, "family_b": b, "delta_r2_pair": gain, "delta_r2_a": single[a], "delta_r2_b": single[b],
                         "complementarity": gain - single[a] - single[b]})
    return pd.DataFrame(rows)


def nested_forward_selection(builder, y, assign, scheme, cfg, model, avail) -> pd.DataFrame:
    """Greedy family selection chosen on inner folds of each outer training set (nested)."""
    from sklearn.model_selection import KFold

    from .validation import iter_folds, regression_metrics

    candidates = [f for f in ("TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC", "CROSS_TASK",
                              "CORRECTNESS_CONTRAST", "COMPLEXITY_SENSITIVITY") if avail(f)]
    tab = [f for f in TAB if avail(f) and f not in {"ITEM_DIFFICULTY"}]
    cols_base = builder.columns_for(set(tab))
    simple = ColumnSetBuilder(builder.base, cols_base)
    rows = []
    for rep, fold, tr, va in iter_folds(assign, scheme, repeats=[0]):
        tr = np.array([s for s in tr if s in y.index])
        va = np.array([s for s in va if s in y.index])
        chosen = []
        inner = KFold(int(cfg["inner_cv_folds"]), shuffle=True, random_state=int(cfg["seed"]) + fold)

        def inner_score(fams):
            cols = cols_base + builder.columns_for(set(fams))
            sb = ColumnSetBuilder(builder.base, cols)
            fp = make_fit_predict(sb, None, model, cfg, y)
            preds = pd.Series(np.nan, index=tr)
            for itr, iva in inner.split(tr):
                preds.iloc[iva] = fp(tr[itr], tr[iva], None)
            return regression_metrics(y.loc[tr], preds)["r2"]

        current = inner_score([])
        for step in range(int(cfg["ablation_forward_max_steps"])):
            scores = {f: inner_score(chosen + [f]) for f in candidates if f not in chosen}
            if not scores:
                break
            best = max(scores, key=scores.get)
            if scores[best] <= current + 0.002:
                break
            chosen.append(best)
            current = scores[best]
        sb = ColumnSetBuilder(builder.base, cols_base + builder.columns_for(set(chosen)))
        pred = make_fit_predict(sb, None, model, cfg, y)(tr, va, None)
        outer = regression_metrics(y.loc[va], pred)
        base_pred = make_fit_predict(simple, None, model, cfg, y)(tr, va, None)
        rows.append({"repeat": rep, "fold": fold, "selected_families": "+".join(chosen), "inner_r2": current,
                     "outer_r2": outer["r2"], "outer_r2_tabular_only": regression_metrics(y.loc[va], base_pred)["r2"]})
    return pd.DataFrame(rows)


def redundancy_analysis(oof: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if oof.empty:
        return pd.DataFrame()
    single = ["A17_PHONETIC_ONLY", "A18_TEMPORAL_ONLY", "A19_SPECTRAL_ONLY", "A20_WAVELETS_ONLY", "A21_BACKGROUND_ONLY",
              "A16_AUDIO_ONLY", "A01_THETA_ONLY"]
    sub = oof[oof["experiment"].isin(single) & (oof["scheme"] == "CV_A_REPEATED_STUDENT_KFOLD")]
    if sub.empty:
        return pd.DataFrame()
    P = sub.groupby(["student_id", "experiment"])["pred"].mean().unstack()
    C = P.corr(method="spearman")
    return C.reset_index().rename(columns={"experiment": "experiment_a"})


def _ablation_figures(ctx, results: pd.DataFrame, loo: pd.DataFrame, pair: pd.DataFrame) -> None:
    if results.empty:
        return
    primary = ctx.config["ablation_primary_model"]
    for scheme, g in results[results["model"].isin([primary, "MEAN", f"TWO_STAGE_{primary}"])].groupby("scheme"):
        g = g.sort_values("experiment")
        fig, axes = viz.new_figure(1, 2, (7, 0.3 * len(g) + 2))
        ax = axes[0, 0]
        lo = (g["r2"] - g["confidence_interval_lower"]).clip(lower=0).fillna(0)
        hi = (g["confidence_interval_upper"] - g["r2"]).clip(lower=0).fillna(0)
        ax.errorbar(g["r2"], g["experiment"], xerr=[lo, hi], fmt="o", capsize=2)
        viz.finalize_axis(ax, f"Pooled OOF R² ({scheme})", "R² (95% CI)", "", int(g["n_students"].max()))
        ax = axes[0, 1]
        d = g.dropna(subset=["delta_r2_vs_screening"])
        ax.errorbar(d["delta_r2_vs_screening"], d["experiment"],
                    xerr=[(d["delta_r2_vs_screening"] - d["delta_r2_vs_screening_ci_lower"]).clip(lower=0).fillna(0),
                          (d["delta_r2_vs_screening_ci_upper"] - d["delta_r2_vs_screening"]).clip(lower=0).fillna(0)], fmt="o", capsize=2)
        ax.axvline(0, color="grey")
        viz.finalize_axis(ax, "Incremental R² vs screening (paired bootstrap)", "delta R²", "", int(g["n_students"].max()))
        ctx.figure(fig, f"ablation_{scheme}", f"Feature-family ablation ({scheme})", "Which modalities add predictive value?",
                   "ablation_results", int(g["n_students"].max()),
                   f"Best experiment: {g.loc[g['r2'].idxmax(), 'experiment']} (R²={g['r2'].max():.3f}).")
    if len(loo):
        fig, axes = viz.new_figure(1, 1)
        l = loo.sort_values("loss_when_dropped")
        viz.forest(axes[0, 0], l, "dropped_family", "loss_when_dropped", "ci_lower", "ci_upper",
                   "Leave-one-family-out: R² lost when dropping a family", "delta R²")
        ctx.figure(fig, "ablation_leave_one_out", "Leave-one-family-out ablation", "Which families are indispensable?",
                   "ablation_leave_one_family_out", None, "Positive values: the family contributes uniquely.")
    if len(pair):
        fams = sorted(set(pair["family_a"]) | set(pair["family_b"]))
        M = pd.DataFrame(np.nan, index=fams, columns=fams)
        for _, r in pair.iterrows():
            M.loc[r["family_a"], r["family_b"]] = M.loc[r["family_b"], r["family_a"]] = r["complementarity"]
        fig, axes = viz.new_figure(1, 1, (6, 5))
        viz.heatmap(axes[0, 0], M, "Family complementarity (pair gain - single gains)", annotate=True, cbar_label="delta R²")
        ctx.figure(fig, "ablation_complementarity", "Feature-family complementarity", "Do families add complementary information?",
                   "ablation_pairwise_families", None, "Positive: synergy; negative: redundancy.")
