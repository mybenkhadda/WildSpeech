"""Section 14 - cross-validation partitions and screening-only baselines.

Models: M00_MEAN, M01_RIDGE, M02_ELASTIC_NET, M03_GAM_OR_SPLINE_RIDGE,
M04_CATBOOST, M05_SHALLOW_XGBOOST, M06_PLS. Optional libraries that are not
installed are reported as MODEL_UNAVAILABLE (never silently substituted).
All preprocessing (imputation, scaling, spline bases, PLS components,
supervised feature selection) is inside the model pipeline, i.e. fitted on
training folds only. Hyper-parameters are tuned by internal CV within the
training fold (RidgeCV / ElasticNetCV / GridSearchCV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import data_access as da
from . import visualization as viz
from .validation import (SCHEMES, available_schemes, group_performance, make_cv_assignments, pooled_predictions,
                         quantile_performance, run_cv)

BASELINE_MODELS = {"M00_MEAN": "MEAN", "M01_RIDGE": "RIDGE", "M02_ELASTIC_NET": "ELASTIC_NET",
                   "M03_GAM_OR_SPLINE_RIDGE": "SPLINE_RIDGE", "M04_CATBOOST": "CATBOOST",
                   "M05_SHALLOW_XGBOOST": "XGBOOST", "M06_PLS": "PLS"}


class ModelUnavailable(ImportError):
    pass


class NanSafeKBest(BaseEstimator, TransformerMixin):
    """Keep the k columns with the largest |Spearman| with y (training fold only)."""

    def __init__(self, k: int = 300):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[1] <= self.k:
            self.support_ = np.arange(X.shape[1])
            return self
        Xr = pd.DataFrame(X).rank().to_numpy()
        yr = pd.Series(y).rank().to_numpy()
        scores = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            m = np.isfinite(Xr[:, j])
            if m.sum() > 10 and np.std(Xr[m, j]) > 0:
                scores[j] = abs(np.corrcoef(Xr[m, j], yr[m])[0, 1])
        self.support_ = np.sort(np.argsort(-np.nan_to_num(scores))[: self.k])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


class PLS1D(BaseEstimator):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression

        self.model_ = PLSRegression(n_components=max(1, min(self.n_components, X.shape[1], X.shape[0] - 1)))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).ravel()


def make_model(kind: str, cfg: dict, n_features: int = 1, n_samples: int = 100):
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    seed = int(cfg["seed"])
    k = int(cfg.get("ablation_max_features", 300))
    inner = KFold(int(cfg["inner_cv_folds"]), shuffle=True, random_state=seed)
    impute = ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=False))
    select = [("select", NanSafeKBest(k))] if n_features > k else []
    kind = kind.upper()
    if kind == "MEAN":
        return DummyRegressor(strategy="mean")
    if kind == "RIDGE":
        return Pipeline(select + [impute, ("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 4, 30)))])
    if kind == "ELASTIC_NET":
        return Pipeline(select + [impute, ("scale", StandardScaler()),
                                  ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=30, cv=inner, max_iter=5000,
                                                         random_state=seed))])
    if kind == "SPLINE_RIDGE":
        return Pipeline(select + [impute, ("scale", StandardScaler()), ("spline", SplineTransformer(n_knots=5, degree=3)),
                                  ("model", RidgeCV(alphas=np.logspace(-2, 5, 30)))])
    if kind == "PLS":
        grid = [c for c in (1, 2, 3, 5, 8) if c <= max(1, min(n_features, 8))]
        return Pipeline(select + [impute, ("scale", StandardScaler()),
                                  ("model", GridSearchCV(PLS1D(), {"n_components": grid}, cv=inner, scoring="neg_mean_squared_error"))])
    if kind == "CATBOOST":
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except ImportError as exc:
            raise ModelUnavailable("catboost not installed") from exc
        return Pipeline(select + [("model", CatBoostRegressor(depth=4, iterations=600, learning_rate=0.03, l2_leaf_reg=6,
                                                               loss_function="RMSE", random_seed=seed, verbose=False,
                                                               thread_count=max(1, int(cfg.get("n_jobs_model", 4)))))])
    if kind == "XGBOOST":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except ImportError as exc:
            raise ModelUnavailable("xgboost not installed") from exc
        return Pipeline(select + [("model", XGBRegressor(max_depth=2, n_estimators=400, learning_rate=0.03, subsample=0.8,
                                                          colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=5,
                                                          tree_method="hist", random_state=seed, n_jobs=4))])
    if kind == "LIGHTGBM":
        try:
            from lightgbm import LGBMRegressor  # type: ignore
        except ImportError as exc:
            raise ModelUnavailable("lightgbm not installed") from exc
        return Pipeline(select + [("model", LGBMRegressor(num_leaves=7, max_depth=3, n_estimators=400, learning_rate=0.03,
                                                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                                                           colsample_bytree=0.8, reg_lambda=5.0, random_state=seed,
                                                           verbose=-1, n_jobs=4))])
    raise ValueError(f"unknown model kind {kind}")


def model_available(kind: str, cfg: dict) -> tuple[bool, str]:
    try:
        make_model(kind, cfg)
        return True, ""
    except ModelUnavailable as exc:
        return False, str(exc)


def make_fit_predict(builder, families, kind: str, cfg: dict, y: pd.Series):
    def fit_predict(train_ids, val_ids, key):
        Xtr, Xva = builder.build(train_ids, val_ids, families, key)
        model = make_model(kind, cfg, n_features=Xtr.shape[1], n_samples=len(Xtr))
        if Xtr.shape[1] == 0:
            return np.full(len(Xva), float(y.loc[train_ids].mean()))
        model.fit(Xtr, y.loc[train_ids].to_numpy())
        return model.predict(Xva)
    return fit_predict


def residual_groups(res: pd.Series, quantiles) -> pd.Series:
    lo, hi = res.quantile(quantiles[0]), res.quantile(quantiles[1])
    return pd.Series(np.where(res <= lo, "NEGATIVE_RESIDUAL", np.where(res >= hi, "POSITIVE_RESIDUAL", "EXPECTED_OUTCOME")),
                     index=res.index)


def stage_validation_and_baseline(ctx) -> dict:
    from .fold_features import FoldFeatureBuilder

    cfg = ctx.config
    meta = da.student_meta(ctx, "train")
    meta = meta[meta["target"].notna()]
    y = meta["target"].astype(float)
    assign, status = make_cv_assignments(meta, cfg)
    ctx.save_df(assign, "cv_assignments")
    ctx.save_json(status, "cv_scheme_status")
    scr = da.screening(ctx)
    base = scr.loc[scr.index.intersection(y.index)]
    builder = FoldFeatureBuilder(ctx, base=base)
    schools = meta["school_id"] if meta["school_id"].notna().any() else None
    results, oofs, registry = [], [], []
    primary = cfg["primary_baseline"]
    primary_res = None
    for name, kind in BASELINE_MODELS.items():
        ok, reason = model_available(kind, cfg)
        registry.append({"model_id": name, "kind": kind, "available": ok, "reason": reason, "families": "SCREENING",
                         "features": ",".join(builder.columns_for({"SCREENING"}) + ["theta_common_factor (fold-fitted)"])})
        if not ok:
            results.append({"model_id": name, "scheme": "ALL", "status": "MODEL_UNAVAILABLE", "reason": reason})
            continue
        fp = make_fit_predict(builder, {"SCREENING"}, kind, cfg, y)
        for scheme in available_schemes(assign):
            res = run_cv(y, assign, scheme, fp, cfg, groups=schools, label=name)
            results.append({"model_id": name, "scheme": scheme, **res.summary})
            oofs.append(res.oof.assign(model_id=name, scheme=scheme))
            if name == primary and scheme == cfg["primary_cv_scheme"]:
                primary_res = res
        ctx.logger.info("Baseline %s done", name)
    results_df = pd.DataFrame(results)
    oof_df = pd.concat(oofs, ignore_index=True) if oofs else pd.DataFrame()
    ctx.save_df(results_df, "baseline_results", fmt="csv")
    ctx.save_df(oof_df, "baseline_oof_predictions")
    ctx.save_df(pd.DataFrame(registry), "baseline_model_registry", fmt="csv")
    if primary_res is None:
        raise RuntimeError(f"primary baseline {primary} under {cfg['primary_cv_scheme']} unavailable")
    pooled = pooled_predictions(primary_res.oof)
    residual = (pooled["y"] - pooled["pred"]).rename("residual")
    res_df = pd.DataFrame({"student_id": pooled.index, "y": pooled["y"].to_numpy(), "oof_prediction": pooled["pred"].to_numpy(),
                           "residual": residual.to_numpy(), "n_oof_predictions": pooled["n_preds"].to_numpy(),
                           "residual_group": residual_groups(residual, cfg["residual_group_quantiles"]).to_numpy(),
                           "model_id": primary, "scheme": cfg["primary_cv_scheme"]})
    ctx.save_df(res_df, "baseline_residuals", also_csv=True)
    qp = quantile_performance(pooled)
    ctx.save_df(qp, "baseline_quantile_performance", fmt="csv")
    outputs = ["cv_assignments", "cv_scheme_status", "baseline_results", "baseline_oof_predictions",
               "baseline_model_registry", "baseline_residuals", "baseline_quantile_performance"]
    if schools is not None:
        gp = group_performance(pooled, schools, int(cfg["min_group_size"]))
        ctx.save_df(gp, "baseline_school_performance", fmt="csv")
        outputs.append("baseline_school_performance")
    _baseline_figures(ctx, results_df, pooled, residual, qp)
    best = results_df[results_df["scheme"] == cfg["primary_cv_scheme"]].sort_values("pooled_official", ascending=False).head(3) \
        if "pooled_official" in results_df.columns else results_df.head(0)
    return {"outputs": outputs,
            "summary": {"primary_pooled_r2": primary_res.summary.get("pooled_r2"),
                        "primary_mean_fold_r2": primary_res.summary.get("mean_fold_r2"),
                        "scheme_status": {k: v["status"] for k, v in status.items()},
                        "best_models": best[["model_id", "pooled_official"]].to_dict("records") if len(best) else []}}


def _baseline_figures(ctx, results: pd.DataFrame, pooled: pd.DataFrame, residual: pd.Series, qp: pd.DataFrame) -> None:
    cfg = ctx.config
    r = results[results.get("status", "OK") == "OK"] if "status" in results.columns else results
    if "pooled_r2" in r.columns and len(r):
        for scheme, g in r.groupby("scheme"):
            g = g.sort_values("pooled_r2")
            fig, axes = viz.new_figure(1, 1, (7, 0.4 * len(g) + 2))
            ax = axes[0, 0]
            lo = (g["pooled_r2"] - g["pooled_r2_ci_lower"]).clip(lower=0).fillna(0)
            hi = (g["pooled_r2_ci_upper"] - g["pooled_r2"]).clip(lower=0).fillna(0)
            ax.errorbar(g["pooled_r2"], g["model_id"], xerr=[lo, hi], fmt="o", capsize=3, label="pooled OOF R² (95% CI)")
            ax.scatter(g["mean_fold_r2"], g["model_id"], marker="x", color="#C44E52", label="mean fold R²")
            ax.legend(fontsize=7)
            viz.finalize_axis(ax, f"Screening-only baselines: {scheme}", "R²", "", int(g["n_students"].max()))
            ctx.figure(fig, f"baseline_comparison_{scheme}", f"Baseline comparison ({scheme})",
                       "How much variation does conventional screening explain?", "baseline_results",
                       int(g["n_students"].max()), f"Best pooled R²={g['pooled_r2'].max():.3f}; pooled and mean-fold R² differ by design.")
    fig, axes = viz.new_figure(1, 3, (4.8, 4))
    viz.scatter_fit(axes[0, 0], pooled["pred"], pooled["y"], "OOF prediction", "observed outcome", "actual vs predicted")
    lim = [np.nanmin([pooled["pred"].min(), pooled["y"].min()]), np.nanmax([pooled["pred"].max(), pooled["y"].max()])]
    axes[0, 0].plot(lim, lim, "--", color="grey")
    viz.hist_kde(axes[0, 1], residual, "OOF residual")
    bins = pd.qcut(pooled["pred"].rank(method="first"), 10, labels=False)
    cal = pooled.groupby(bins).agg(pred=("pred", "mean"), y=("y", "mean"), n=("y", "size"))
    axes[0, 2].plot(cal["pred"], cal["y"], "o-")
    axes[0, 2].plot(lim, lim, "--", color="grey")
    viz.finalize_axis(axes[0, 2], "Calibration (prediction deciles)", "mean prediction", "mean observed", int(len(pooled)))
    ctx.figure(fig, "baseline_primary_diagnostics", f"Primary baseline {cfg['primary_baseline']} diagnostics",
               "Is the screening baseline well calibrated?", "baseline_residuals", int(len(pooled)),
               f"Residual SD={residual.std():.3g}; quantile RMSE range {qp['rmse'].min():.3g}-{qp['rmse'].max():.3g}.")
