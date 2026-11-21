"""Section 18 - feature interactions and conditional relevance.

Only a predefined, research-driven candidate set of interactions is tested
(no unrestricted all-pairs search). Representative variables are built inside
each training fold (PC1 of a family, standardised on training students) and
interaction gains are evaluated out of fold with paired bootstrap CIs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .families import modeling_columns
from .stats_utils import mutual_information, safe_corr
from .validation import iter_folds, paired_bootstrap_delta, regression_metrics

REPRESENTATIVES = {
    "SCREENING": {"columns": ["theta_mean"]},
    "TEMPORAL": {"families": ["TEMPORAL"]},
    "SPECTRAL": {"families": ["SPECTRAL"]},
    "PHONETIC": {"families": ["PHONETIC"]},
    "ACOUSTIC": {"families": ["TEMPORAL", "SPECTRAL", "MFCC"]},
    "FLUENCY": {"prefixes": ["t_pause_speech_ratio__", "t_n_pauses__", "t_internal_sil__", "t_rhythm_regularity__"]},
    "DURATION": {"columns": ["t_active_dur__mean", "q_duration_s__mean"]},
    "PHONETIC_CONFIDENCE": {"suffix": "seq_confidence__mean"},
    "SPECTRAL_ENTROPY": {"columns": ["s_sp_entropy_mean__mean"]},
    "CORRECTNESS": {"columns": ["acc_all"]},
    "DIFFICULTY": {"columns": ["l_n_phonemes__mean", "l_n_chars__mean", "irt_all_theta"]},
    "ACOUSTIC_CONSISTENCY": {"columns": ["ia_item_adjusted_acoustic_variability", "m_sp_cepstral_variability__std"]},
    "COMPLEXITY": {"prefixes": ["cx_t_active_dur__"]},
}


def representative_columns(S: pd.DataFrame, name: str) -> list:
    spec = REPRESENTATIVES[name]
    if "columns" in spec:
        cols = [c for c in spec["columns"] if c in S.columns]
        return cols[:1]
    if "families" in spec:
        return modeling_columns(S.columns, spec["families"])
    if "prefixes" in spec:
        return [c for c in S.columns if c.startswith(tuple(spec["prefixes"])) and not c.endswith(("_reliable", "_n"))]
    if "suffix" in spec:
        return [c for c in S.columns if c.endswith(spec["suffix"])][:1]
    return []


def fold_representative(S: pd.DataFrame, cols: list, train_ids) -> pd.Series:
    """Single column standardised on training rows, or PC1 of several columns fitted on training rows."""
    X = S[cols]
    tr = X.loc[train_ids]
    med = tr.median()
    sd = tr.std().replace(0, np.nan)
    Z = ((X - med) / sd).loc[:, sd.notna()].fillna(0.0)
    if Z.shape[1] == 0:
        return pd.Series(np.nan, index=S.index)
    if Z.shape[1] == 1:
        return Z.iloc[:, 0]
    Ztr = Z.loc[train_ids].to_numpy()
    mu = Ztr.mean(0)
    _, _, vt = np.linalg.svd(Ztr - mu, full_matrices=False)
    pc = (Z.to_numpy() - mu) @ vt[0]
    s = pd.Series(pc, index=S.index)
    return (s - s.loc[train_ids].mean()) / (s.loc[train_ids].std() or 1.0)


def _ridge(Xtr, ytr, Xva):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30)))
    m.fit(Xtr, ytr)
    return m.predict(Xva)


def h_statistic(model, X: pd.DataFrame, a: str, b: str, max_n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pts = X.iloc[rng.choice(len(X), min(len(X), max_n // 3), replace=False)]
    bg = X.iloc[rng.choice(len(X), min(len(X), 100), replace=False)]

    def pd_val(vals: dict) -> float:
        Xb = bg.copy()
        for k, v in vals.items():
            Xb[k] = v
        return float(model.predict(Xb).mean())
    fab, fa, fb = [], [], []
    for _, r in pts.iterrows():
        fab.append(pd_val({a: r[a], b: r[b]}))
        fa.append(pd_val({a: r[a]}))
        fb.append(pd_val({b: r[b]}))
    fab, fa, fb = (np.asarray(v) - np.mean(v) for v in (fab, fa, fb))
    denom = (fab ** 2).sum()
    return float(np.sqrt(((fab - fa - fb) ** 2).sum() / denom)) if denom > 0 else np.nan


def stage_feature_interactions(ctx) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    cfg = ctx.config
    S = da.student_features(ctx, "train")
    meta = da.student_meta(ctx, "train")
    y = meta["target"].dropna().astype(float)
    S = S.reindex(y.index)
    assign = ctx.load_df("cv_assignments")
    screening = [c for c in da.raw_theta_columns(S)]
    rows, fold_rows, surfaces = [], [], []
    for pair in cfg["interaction_candidates"]:
        a, b = pair
        label = f"{a} x {b}"
        if "TASK" in (a, b) or "INSTRUMENT" in (a, b):
            rows.append(_special_interaction(ctx, S, y, meta, assign, a, b, screening))
            continue
        ca, cb = representative_columns(S, a), representative_columns(S, b)
        if not ca or not cb:
            rows.append({"interaction": label, "status": "UNAVAILABLE", "reason": f"no columns for {a if not ca else b}"})
            continue
        preds_main, preds_int, ids_all = [], [], []
        for rep, fold, tr, va in iter_folds(assign, "CV_A_REPEATED_STUDENT_KFOLD", repeats=[0]):
            tr = [t for t in tr if t in S.index]
            va = [v for v in va if v in S.index]
            ra, rb = fold_representative(S, ca, tr), fold_representative(S, cb, tr)
            base = S[screening].copy()
            base["rep_a"], base["rep_b"] = ra, rb
            inter = base.copy()
            inter["rep_axb"] = ra * rb
            pm = _ridge(base.loc[tr], y.loc[tr], base.loc[va])
            pi = _ridge(inter.loc[tr], y.loc[tr], inter.loc[va])
            preds_main.append(pd.Series(pm, index=va))
            preds_int.append(pd.Series(pi, index=va))
            fold_rows.append({"interaction": label, "fold": fold, "delta_r2": regression_metrics(y.loc[va], pi)["r2"] - regression_metrics(y.loc[va], pm)["r2"]})
        pmain, pint = pd.concat(preds_main), pd.concat(preds_int)
        d = paired_bootstrap_delta(y.loc[pint.index], pint, pmain.reindex(pint.index), cfg, "r2")
        ra_all, rb_all = fold_representative(S, ca, list(S.index)), fold_representative(S, cb, list(S.index))
        prod = (ra_all * rb_all).rename("prod")
        mi = mutual_information(prod.to_frame(), y.reindex(prod.index), int(cfg["seed"]))["prod"]
        terc = pd.qcut(ra_all, 3, labels=["low", "mid", "high"], duplicates="drop")
        cond = {f"rho_b_given_a_{lvl}": safe_corr(rb_all[terc == lvl], y.reindex(rb_all.index)[terc == lvl], "spearman", 15)[0]
                for lvl in terc.dropna().unique()}
        sub = pd.DataFrame({"a": ra_all, "b": rb_all, "theta": S[screening].mean(axis=1)}).dropna()
        h = np.nan
        if len(sub) >= 50:
            gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=int(cfg["seed"]))
            gbm.fit(sub, y.loc[sub.index])
            h = h_statistic(gbm, sub, "a", "b", int(cfg["interaction_h_statistic_max_n"]), int(cfg["seed"]))
            surfaces.append((label, gbm, sub))
        fr = pd.DataFrame(fold_rows)
        fr = fr[fr["interaction"] == label]
        rows.append({"interaction": label, "status": "OK", "n_columns_a": len(ca), "n_columns_b": len(cb),
                     "delta_r2_interaction": d["delta"], "ci_lower": d["ci_lower"], "ci_upper": d["ci_upper"],
                     "fold_sign_consistency": float((fr["delta_r2"] > 0).mean()) if len(fr) else np.nan,
                     "mutual_information_product": mi, "friedman_h": h, **cond})
    res = pd.DataFrame(rows)
    ctx.save_df(res, "interaction_results", also_csv=True)
    ctx.save_df(pd.DataFrame(fold_rows), "interaction_fold_stability", fmt="csv")
    shap_summary = _shap_interactions(ctx, surfaces)
    ctx.save_json(shap_summary, "interaction_shap_summary")
    _interaction_figures(ctx, res, surfaces, y)
    return {"outputs": ["interaction_results", "interaction_fold_stability", "interaction_shap_summary"],
            "summary": res[["interaction", "status"] + [c for c in ("delta_r2_interaction", "friedman_h") if c in res]].to_dict("records")}


def _special_interaction(ctx, S, y, meta, assign, a, b, screening) -> dict:
    cfg = ctx.config
    label = f"{a} x {b}"
    if "INSTRUMENT" in (a, b):
        if meta["instrument"].nunique() < 2:
            return {"interaction": label, "status": "UNAVAILABLE", "reason": "single instrument"}
        dummies = pd.get_dummies(meta["instrument"].reindex(S.index).astype(str), drop_first=True, dtype=float)
        base = S[screening].join(dummies)
        inter = base.copy()
        for c in dummies.columns:
            inter[f"theta_mean_x_{c}"] = S["theta_mean"] * dummies[c] if "theta_mean" in S else S[screening].mean(axis=1) * dummies[c]
    else:
        other = b if a == "TASK" else a
        if other == "COMPLEXITY":
            glob = [c for c in S.columns if c.startswith("cx_") and c.endswith("__all") and "_reliable" not in c]
            task = [c for c in S.columns if c.startswith("cx_") and not c.endswith(("__all", "_reliable", "_n"))]
        else:
            fam_cols = modeling_columns(S.columns, ["TEMPORAL", "SPECTRAL", "MFCC"])
            glob = [c for c in fam_cols if "__task=" not in c and c.endswith("__mean")][:200]
            task = [c for c in fam_cols if "__task=" in c and c.endswith("__mean")][:400]
        if not glob or not task:
            return {"interaction": label, "status": "UNAVAILABLE", "reason": "no task-conditioned features"}
        base = S[screening + glob]
        inter = S[screening + glob + task]
    pm, pi = [], []
    for rep, fold, tr, va in iter_folds(assign, "CV_A_REPEATED_STUDENT_KFOLD", repeats=[0]):
        tr = [t for t in tr if t in S.index]
        va = [v for v in va if v in S.index]
        pm.append(pd.Series(_ridge(base.loc[tr], y.loc[tr], base.loc[va]), index=va))
        pi.append(pd.Series(_ridge(inter.loc[tr], y.loc[tr], inter.loc[va]), index=va))
    pm, pi = pd.concat(pm), pd.concat(pi)
    d = paired_bootstrap_delta(y.loc[pi.index], pi, pm.reindex(pi.index), cfg, "r2")
    return {"interaction": label, "status": "OK", "delta_r2_interaction": d["delta"], "ci_lower": d["ci_lower"],
            "ci_upper": d["ci_upper"], "note": "task-specific vs pooled representation" if "TASK" in (a, b) else "instrument-specific screening slopes"}


def _shap_interactions(ctx, surfaces) -> dict:
    if not surfaces:
        return {"status": "NO_MODELS"}
    try:
        import shap  # type: ignore
        from xgboost import XGBRegressor  # type: ignore
    except ImportError:
        return {"status": "UNAVAILABLE", "reason": "shap and xgboost required"}
    out = {}
    for label, _, sub in surfaces[:5]:
        y = da.target(ctx).reindex(sub.index)
        m = XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05, random_state=int(ctx.config["seed"])).fit(sub, y)
        vals = shap.TreeExplainer(m).shap_interaction_values(sub)
        out[label] = {"mean_abs_interaction_a_b": float(np.abs(vals[:, 0, 1]).mean()),
                      "mean_abs_main_a": float(np.abs(vals[:, 0, 0]).mean()), "mean_abs_main_b": float(np.abs(vals[:, 1, 1]).mean())}
    return {"status": "OK", "results": out}


def _interaction_figures(ctx, res: pd.DataFrame, surfaces, y) -> None:
    ok = res[res["status"] == "OK"].dropna(subset=["delta_r2_interaction"])
    if len(ok):
        ok = ok.sort_values("delta_r2_interaction")
        fig, axes = viz.new_figure(1, 1, (8, 0.35 * len(ok) + 2))
        viz.forest(axes[0, 0], ok, "interaction", "delta_r2_interaction", "ci_lower", "ci_upper",
                   "Out-of-fold R² gain from adding the interaction", "delta R² (paired bootstrap 95% CI)")
        ctx.figure(fig, "interaction_gains", "Interaction stability report", "Do features matter only for particular profiles?",
                   "interaction_results", int(len(y)), "Predefined candidate set; representatives fitted within folds.")
    for label, gbm, sub in surfaces[:4]:
        ga = np.quantile(sub["a"], np.linspace(0.05, 0.95, 12))
        gb = np.quantile(sub["b"], np.linspace(0.05, 0.95, 12))
        M = np.zeros((len(gb), len(ga)))
        bg = sub.sample(min(len(sub), 200), random_state=0)
        for i, vb in enumerate(gb):
            for j, va in enumerate(ga):
                X = bg.copy()
                X["a"], X["b"] = va, vb
                M[i, j] = gbm.predict(X).mean()
        fig, axes = viz.new_figure(1, 1, (6, 5))
        viz.heatmap(axes[0, 0], pd.DataFrame(M, index=np.round(gb, 2), columns=np.round(ga, 2)),
                    f"2-D partial dependence: {label} (a=x, b=y)", cmap="viridis", cbar_label="predicted outcome")
        ctx.figure(fig, f"interaction_surface_{label.replace(' ', '')}", f"Two-dimensional response surface: {label}",
                   "How do the two factors combine?", "interaction_results", int(len(sub)),
                   "Gradient-boosting partial dependence over representative variables (descriptive).")
