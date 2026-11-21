"""Section 21 - properly nested two-stage residual learning.

prediction = baseline_prediction + alpha * residual_prediction

Within every outer fold:
1. inner-fold screening/tabular predictions on outer-training students;
2. inner out-of-fold residual targets;
3. acoustic residual model fitted on outer-training data only;
4. fusion coefficient alpha selected on inner validation predictions;
5. components refitted on all outer-training students;
6. outer-validation students predicted;
7. evaluation on the original target scale.

Outer-validation outcomes are never used for alpha selection. Within the inner
loop, residual targets of inner-training students come from inner OOF
predictions (standard stacking); the outer fold remains fully untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import data_access as da
from . import visualization as viz
from .baseline_models import make_fit_predict
from .families import TABULAR_FAMILIES
from .validation import (available_schemes, iter_folds, official_score, paired_bootstrap_delta, pooled_predictions,
                         regression_metrics, summarize_cv, CVResult)

AUDIO_RESIDUAL_FAMILIES = ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC", "ITEM_ADJUSTED_AUDIO",
                           "COMPLEXITY_SENSITIVITY", "CORRECTNESS_CONTRAST", "CROSS_TASK", "EMBEDDINGS"]


def _inner_oof(builder, fams, model, cfg, target: pd.Series, ids: np.ndarray, splitter, key_prefix) -> pd.Series:
    out = pd.Series(np.nan, index=ids)
    fp = make_fit_predict(builder, set(fams), model, cfg, target)
    for k, (itr, iva) in enumerate(splitter.split(ids)):
        out.iloc[iva] = fp(ids[itr], ids[iva], key_prefix + ("inner", k))
    return out


def nested_two_stage_cv(builder, y: pd.Series, assign: pd.DataFrame, scheme: str, cfg: dict, stage1_families: list,
                        stage2_families: list, groups=None, model: str | None = None, repeats=None) -> CVResult:
    model = model or cfg["ablation_primary_model"]
    alphas = [float(a) for a in cfg["residual_fusion_alphas"]]
    rows, fm = [], []
    for rep, fold, tr, va in iter_folds(assign, scheme, repeats):
        tr = np.array([s for s in tr if s in y.index])
        va = np.array([s for s in va if s in y.index])
        if len(tr) < 20 or len(va) == 0:
            continue
        inner = KFold(int(cfg["inner_cv_folds"]), shuffle=True, random_state=int(cfg["seed"]) + 97 * fold + rep)
        key = (scheme, rep, fold)
        base_oof = _inner_oof(builder, stage1_families, model, cfg, y, tr, inner, key)
        resid = (y.loc[tr] - base_oof).rename("residual")
        res_oof = _inner_oof(builder, stage2_families, model, cfg, resid, tr, inner, key)
        losses = {a: float(np.nanmean((y.loc[tr] - (base_oof + a * res_oof)) ** 2)) for a in alphas}
        alpha = min(losses, key=losses.get)
        base_pred = make_fit_predict(builder, set(stage1_families), model, cfg, y)(tr, va, key + ("outer",))
        res_pred = make_fit_predict(builder, set(stage2_families), model, cfg, resid)(tr, va, key + ("outer",))
        pred = base_pred + alpha * res_pred
        yv = y.loc[va].to_numpy()
        rows.append(pd.DataFrame({"student_id": va, "repeat": rep, "fold": fold, "y": yv, "pred": pred,
                                  "baseline_pred": base_pred, "residual_pred": res_pred, "alpha": alpha}))
        m = regression_metrics(yv, pred)
        m.update({"repeat": rep, "fold": fold, "alpha": alpha, "official": official_score(yv, pred, cfg),
                  "inner_loss_alpha0": losses[min(alphas)], "inner_loss_selected": losses[alpha]})
        fm.append(m)
    oof = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["student_id", "repeat", "fold", "y", "pred"])
    fold_metrics = pd.DataFrame(fm)
    res = CVResult(scheme, oof, fold_metrics)
    res.summary = summarize_cv(oof, fold_metrics, cfg, groups)
    res.summary["alpha_mean"] = float(fold_metrics["alpha"].mean()) if len(fold_metrics) else np.nan
    res.summary["alpha_values"] = fold_metrics["alpha"].tolist() if len(fold_metrics) else []
    return res


def nested_blend_cv(builder, y, assign, scheme, cfg, fams_a, fams_b, groups=None, model=None, repeats=None) -> CVResult:
    """Simple prediction blending w*A + (1-w)*B with w chosen on inner OOF predictions."""
    model = model or cfg["ablation_primary_model"]
    grid = np.linspace(0, 1, 11)
    rows, fm = [], []
    for rep, fold, tr, va in iter_folds(assign, scheme, repeats):
        tr = np.array([s for s in tr if s in y.index])
        va = np.array([s for s in va if s in y.index])
        if len(tr) < 20 or len(va) == 0:
            continue
        inner = KFold(int(cfg["inner_cv_folds"]), shuffle=True, random_state=int(cfg["seed"]) + 131 * fold + rep)
        key = (scheme, rep, fold)
        oa = _inner_oof(builder, fams_a, model, cfg, y, tr, inner, key)
        ob = _inner_oof(builder, fams_b, model, cfg, y, tr, inner, key)
        w = grid[int(np.argmin([np.nanmean((y.loc[tr] - (g * oa + (1 - g) * ob)) ** 2) for g in grid]))]
        pa = make_fit_predict(builder, set(fams_a), model, cfg, y)(tr, va, key + ("outer",))
        pb = make_fit_predict(builder, set(fams_b), model, cfg, y)(tr, va, key + ("outer",))
        pred = w * pa + (1 - w) * pb
        yv = y.loc[va].to_numpy()
        rows.append(pd.DataFrame({"student_id": va, "repeat": rep, "fold": fold, "y": yv, "pred": pred, "weight_a": w}))
        m = regression_metrics(yv, pred)
        m.update({"repeat": rep, "fold": fold, "weight_a": w, "official": official_score(yv, pred, cfg)})
        fm.append(m)
    oof = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["student_id", "repeat", "fold", "y", "pred"])
    res = CVResult(scheme, oof, pd.DataFrame(fm))
    res.summary = summarize_cv(oof, res.fold_metrics, cfg, groups)
    return res


def stage_residual_learning(ctx) -> dict:
    from .ablation import cv_evaluate, family_available
    from .fold_features import FoldFeatureBuilder

    cfg = ctx.config
    meta = da.student_meta(ctx, "train")
    y = meta["target"].dropna().astype(float)
    assign = ctx.load_df("cv_assignments")
    builder = FoldFeatureBuilder(ctx)
    builder.base = builder.base.reindex(y.index)
    groups = meta["school_id"] if meta["school_id"].notna().any() else None
    s1 = [f for f in TABULAR_FAMILIES if family_available(f, builder, ctx)]
    s2 = [f for f in AUDIO_RESIDUAL_FAMILIES if family_available(f, builder, ctx)]
    if not s2:
        ctx.save_json({"status": "SKIPPED_UNAVAILABLE", "reason": "no acoustic/phonetic families available"}, "residual_learning_status")
        return {"outputs": ["residual_learning_status"], "summary": {"status": "SKIPPED_UNAVAILABLE"}}
    schemes = [s for s in ("CV_A_REPEATED_STUDENT_KFOLD", "CV_B_SCHOOL_GROUP_KFOLD") if s in available_schemes(assign)]
    rows, oofs = [], []
    for scheme in schemes:
        repeats = [0] if scheme == "CV_A_REPEATED_STUDENT_KFOLD" and int(cfg["cv_repeats"]) > 1 else None
        runs = {
            "BASELINE_TABULAR": cv_evaluate(builder, s1, y, assign, scheme, cfg, repeats=repeats, groups=groups),
            "DIRECT_MULTIMODAL": cv_evaluate(builder, s1 + s2, y, assign, scheme, cfg, repeats=repeats, groups=groups),
            "TWO_STAGE_RESIDUAL": nested_two_stage_cv(builder, y, assign, scheme, cfg, s1, s2, groups, repeats=repeats),
            "NESTED_BLEND": nested_blend_cv(builder, y, assign, scheme, cfg, s1, s1 + s2, groups, repeats=repeats),
        }
        pooled = {k: pooled_predictions(v.oof) for k, v in runs.items()}
        for name, res in runs.items():
            row = {"approach": name, "scheme": scheme, **{k: v for k, v in res.summary.items() if not isinstance(v, list)}}
            for ref in ("BASELINE_TABULAR", "DIRECT_MULTIMODAL"):
                if ref != name:
                    common = pooled[name].index.intersection(pooled[ref].index)
                    d = paired_bootstrap_delta(pooled[name].loc[common, "y"], pooled[name].loc[common, "pred"],
                                               pooled[ref].loc[common, "pred"], cfg, "r2")
                    row[f"delta_r2_vs_{ref.lower()}"] = d["delta"]
                    row[f"delta_r2_vs_{ref.lower()}_ci_lower"] = d["ci_lower"]
                    row[f"delta_r2_vs_{ref.lower()}_ci_upper"] = d["ci_upper"]
            rows.append(row)
            oofs.append(res.oof.assign(approach=name, scheme=scheme))
    results = pd.DataFrame(rows)
    oof = pd.concat(oofs, ignore_index=True)
    ctx.save_df(results, "residual_learning_results", also_csv=True)
    ctx.save_df(oof, "residual_learning_oof")
    fig, axes = viz.new_figure(1, len(schemes), (6, 3.6))
    for ax, scheme in zip(axes.ravel(), schemes):
        g = results[results["scheme"] == scheme]
        ax.errorbar(g["pooled_r2"], g["approach"], xerr=[(g["pooled_r2"] - g["pooled_r2_ci_lower"]).clip(lower=0).fillna(0),
                                                       (g["pooled_r2_ci_upper"] - g["pooled_r2"]).clip(lower=0).fillna(0)],
                    fmt="o", capsize=3)
        viz.finalize_axis(ax, f"Residual learning ({scheme})", "pooled OOF R² (95% CI)", "", int(g["n_students"].max()))
    ctx.figure(fig, "residual_learning_comparison", "Two-stage residual learning vs direct regression and blending",
               "Does residual learning improve upon direct multimodal regression?", "residual_learning_results",
               int(results["n_students"].max()), "Alpha and blend weights chosen on inner folds only.")
    two = oof[oof["approach"] == "TWO_STAGE_RESIDUAL"]
    if "alpha" in two.columns:
        fig, axes = viz.new_figure(1, 1)
        two.groupby(["scheme", "repeat", "fold"])["alpha"].first().plot(kind="hist", bins=len(cfg["residual_fusion_alphas"]) * 2, ax=axes[0, 0])
        viz.finalize_axis(axes[0, 0], "Selected fusion coefficients alpha", "alpha", "folds", None)
        ctx.figure(fig, "residual_alpha_distribution", "Selected residual fusion coefficients", "How much residual correction is chosen?",
                   "residual_learning_oof", None, f"Grid {cfg['residual_fusion_alphas']}.")
    return {"outputs": ["residual_learning_results", "residual_learning_oof"],
            "summary": results[["approach", "scheme", "pooled_r2"]].to_dict("records")}
