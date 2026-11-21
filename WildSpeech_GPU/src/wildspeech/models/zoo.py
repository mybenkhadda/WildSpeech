"""Model factory. Optional libraries degrade to status 'unavailable' instead of failing the stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

ALPHAS = np.logspace(-2, 4, 31)


class ModelUnavailable(RuntimeError):
    pass


class PLSCV(BaseEstimator, RegressorMixin):
    """PLS with the number of components chosen by inner CV on the training fold."""

    def __init__(self, max_components: int = 8, inner_splits: int = 4, seed: int = 0):
        self.max_components, self.inner_splits, self.seed = max_components, inner_splits, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        best, best_err = 1, np.inf
        kf = KFold(self.inner_splits, shuffle=True, random_state=self.seed)
        for k in range(1, min(self.max_components, X.shape[1], len(y) // 3) + 1):
            p = cross_val_predict(PLSRegression(n_components=k, scale=False), X, y, cv=kf).ravel()
            err = np.mean((y - p) ** 2)
            if err < best_err:
                best, best_err = k, err
        self.n_components_ = best
        self.model_ = PLSRegression(n_components=best, scale=False).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float)).ravel()


class SplineGAM(BaseEstimator, RegressorMixin):
    """Additive spline model (GAM approximation): per-feature cubic B-splines + ridge with CV penalty.
    Uses pygam.LinearGAM when available and the feature count is small."""

    def __init__(self, n_knots: int = 5, max_features: int = 30, use_pygam: bool = True):
        self.n_knots, self.max_features, self.use_pygam = n_knots, max_features, use_pygam

    def fit(self, X, y):
        X = pd.DataFrame(X)
        if X.shape[1] > self.max_features:
            r = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]) if c.std() > 0 else 0.0)
            self.cols_ = list(r.sort_values(ascending=False).index[: self.max_features])
        else:
            self.cols_ = list(X.columns)
        Xs = X[self.cols_].to_numpy(float)
        self.backend_ = "spline_ridge"
        if self.use_pygam and Xs.shape[1] <= 15:
            try:
                from pygam import LinearGAM

                self.model_ = LinearGAM(n_splines=max(4, self.n_knots + 3)).gridsearch(Xs, np.asarray(y, float), progress=False)
                self.backend_ = "pygam"
                return self
            except Exception:
                pass
        self.model_ = make_pipeline(SplineTransformer(n_knots=self.n_knots, degree=3, extrapolation="linear"),
                                    RidgeCV(alphas=ALPHAS)).fit(Xs, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(pd.DataFrame(X)[self.cols_].to_numpy(float)), float).ravel()


def make_model(name: str, cfg, seed: int = 0, params: dict | None = None):
    m = cfg.section("models")
    params = dict(params or {})
    threads = int(m.get("gbm_threads", 6))
    gpu = str(m.get("gbm_device", "cpu")) == "cuda"
    inner = int(cfg.get_path("validation.inner_splits", 4))
    name = name.upper()
    if name == "MEAN":
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy="mean")
    if name == "RIDGE":
        return RidgeCV(alphas=params.pop("alphas", ALPHAS))
    if name == "ELASTIC_NET":
        return ElasticNetCV(l1_ratio=params.pop("l1_ratio", [0.1, 0.5, 0.9]), n_alphas=40, cv=inner, max_iter=5000,
                            random_state=seed, n_jobs=1)
    if name == "PLS":
        return PLSCV(max_components=params.pop("max_components", 8), inner_splits=inner, seed=seed)
    if name == "GAM":
        return SplineGAM(**params)
    if name == "LIGHTGBM":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ModelUnavailable("lightgbm") from e
        base = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, random_state=seed, n_jobs=threads, verbose=-1)
        if gpu:
            base["device_type"] = "gpu"
        return LGBMRegressor(**{**base, **params})
    if name == "XGBOOST":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ModelUnavailable("xgboost") from e
        base = dict(n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.5,
                    reg_lambda=5.0, min_child_weight=3, tree_method="hist", random_state=seed, n_jobs=threads,
                    device="cuda" if gpu else "cpu")
        return XGBRegressor(**{**base, **params})
    if name == "CATBOOST":
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ModelUnavailable("catboost") from e
        base = dict(iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=5.0, loss_function="RMSE",
                    random_seed=seed, verbose=0, thread_count=threads, allow_writing_files=False)
        if gpu:
            base["task_type"] = "GPU"
        return CatBoostRegressor(**{**base, **params})
    if name == "MOE":
        from .neural import MoERegressor

        return MoERegressor(cfg=cfg, seed=seed, **params)
    if name in ("LATE_FUSION", "RESIDUAL_STACK", "TASK_ENSEMBLE"):
        from .fusion import make_fusion

        return make_fusion(name, cfg, seed, params)
    raise ValueError(f"unknown model {name}")


def needs_scaling(name: str) -> bool:
    return name.upper() not in ("LIGHTGBM", "XGBOOST", "CATBOOST")


def clone_model(est):
    try:
        return clone(est)
    except Exception:
        import copy

        return copy.deepcopy(est)
