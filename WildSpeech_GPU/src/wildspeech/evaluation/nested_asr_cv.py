"""Traced, fallback-aware student-level cross-validation for V3 (and the P23 re-audit of V2).

V3Spec
    name, model, providers (feature families), budget, method (v2_family | select | pca | pls | rrr | raw | none;
    raw = task-scoped columns imputed/scaled only, for the task-pooled ensemble),
    fusion (early | residual | late | task_pool | stack), alphas (residual shrinkage grid), control
    (None | permuted_features | shuffled_outcome), base_families / base_model
V3FoldModel (fitted on the training fold only)
    baseline columns  -> TabularPreprocessor (impute + indicators + scale)
    added columns     -> per-spec reduction:
                           v2_family : V2 FamilyReducer (PCA capped at 10/family, IN-SAMPLE ridge residual target)
                                       — reproduces V2 P21 exactly, used only by the P23 audit
                           select/pca/pls/rrr : CompactReducer on the training-fold INNER-OOF baseline residual
    fusion            -> V2 estimators (NestedResidualRegressor, ConvexLateFusion, RobustTaskEnsemble, ...)
    trace()           -> what the fold actually used: added columns (raw / non-missing / variance), reduced
                         dimensions, residual shrinkage alpha + inner MSE per alpha, stage-2 prediction SD,
                         fusion weights, fallback flag + reason
V3Evaluator.evaluate(spec, scheme, baseline=...)
    for every registry fold: fold frames (V1 fold features + provider tables fitted on the training ids)
    -> V3FoldModel -> validation predictions; id alignment is asserted (validation ids == fold ids, y aligned);
    per-fold trace + max |pred - baseline pred| on the same students.
    status:  ok                    fitted model whose predictions differ from the baseline
             fallback_baseline     every fold returned the baseline prediction (alpha = 0 / no added columns /
                                   zero fusion weight) -> ΔR² = 0 is a FALLBACK, never reported as a result
             partial_fallback      some folds fell back
             no_added_features     providers produced no usable columns
             unavailable / failed  model library missing / exception (traceback stored)
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .. import ID
from ..features.builder import family_of
from ..io_utils import atomic_write_parquet, read_json, stable_hash, write_json
from ..models.compact_regression import StudentLevelEvaluator
from ..models.preprocessing import TabularPreprocessor
from ..models.zoo import ModelUnavailable, make_model, needs_scaling
from ..validation.metrics import metric_row

DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class V3Spec:
    name: str
    model: str = "RIDGE"
    providers: list[str] = field(default_factory=list)
    budget: int | None = 20
    method: str = "pca"
    fusion: str = "early"
    alphas: tuple | None = None
    control: str | None = None
    base_families: list[str] | None = None
    base_model: str | None = None
    params: dict = field(default_factory=dict)
    group: str = ""
    experiment: str = ""
    stack_models: tuple = ("RIDGE", "GAM", "LIGHTGBM")

    def key(self) -> str:
        d = asdict(self)
        for k in ("name", "group", "experiment"):
            d.pop(k)
        return stable_hash(d)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class V3Result:
    spec: V3Spec
    scheme: str
    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    trace: pd.DataFrame
    summary: dict
    status: str


def _inner_oof_residual(Xb: pd.DataFrame, y: np.ndarray, seed: int, splits: int = 4) -> np.ndarray:
    """Training-fold residual of a ridge on the baseline columns, predicted out-of-fold (no in-sample fit)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    if Xb.shape[1] == 0:
        return y - y.mean()
    oof = np.zeros(len(y))
    for tr, va in KFold(splits, shuffle=True, random_state=seed).split(Xb):
        oof[va] = RidgeCV(alphas=np.logspace(-2, 4, 31)).fit(Xb.iloc[tr], y[tr]).predict(Xb.iloc[va])
    return y - oof


class V3FoldModel:
    def __init__(self, spec: V3Spec, cfg, seed: int, base_cols: list[str], groups: dict[str, list[str]],
                 tasks: list[str], baseline_model: str):
        self.spec, self.cfg, self.seed = spec, cfg, seed
        self.base_cols, self.groups, self.tasks = list(base_cols), {k: list(v) for k, v in groups.items()}, list(tasks)
        self.baseline_model = baseline_model
        self.inner = int(cfg.get_path("validation.inner_splits", 4))
        self.mmf = float(cfg.get_path("models.max_missing_frac", 0.6))
        r = cfg.get_path("v3.representations", {}) or {}
        self.rank_grid, self.rrr_alphas = tuple(r.get("rrr_rank_grid", [2, 4, 8])), tuple(r.get("rrr_alphas", [1, 10, 100, 1000]))

    # ---------------------------------------------------------------- estimator
    def _estimator(self, Xb_cols: list[str]):
        from ..models.residual_fusion import ConvexLateFusion, NestedResidualRegressor, RegularizedStack, RobustTaskEnsemble

        s = self.spec
        if s.fusion == "early":
            return make_model(s.model, self.cfg, seed=self.seed, params=dict(s.params))
        if s.fusion == "residual":
            return NestedResidualRegressor(stage1=make_model(s.base_model or self.baseline_model, self.cfg, seed=self.seed),
                                           stage2=make_model(s.model, self.cfg, seed=self.seed, params=dict(s.params)),
                                           stage1_cols=Xb_cols, alphas=tuple(s.alphas or DEFAULT_ALPHAS),
                                           inner_splits=self.inner, seed=self.seed)
        if s.fusion == "late":
            return ConvexLateFusion(base_cols=Xb_cols, base=make_model(s.model, self.cfg, seed=self.seed),
                                    inner_splits=self.inner, seed=self.seed)
        if s.fusion == "task_pool":
            return RobustTaskEnsemble(base_cols=Xb_cols, tasks=tuple(self.tasks), inner_splits=self.inner, seed=self.seed)
        if s.fusion == "stack":
            return RegularizedStack([make_model(m, self.cfg, seed=self.seed) for m in s.stack_models],
                                    inner_splits=self.inner, seed=self.seed)
        raise ValueError(f"unknown fusion {s.fusion}")

    def _avail(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for t in self.tasks:
            c = f"BEHAVIOR__{t}__n_items__value"
            out[f"__avail__{t}"] = (X[c].fillna(0).to_numpy(float) > 0).astype(float) if c in X else 1.0
        return out

    # ---------------------------------------------------------------- reduction
    def _reduce_fit(self, X: pd.DataFrame, Xb: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        added = [c for cols in self.groups.values() for c in cols if c in X.columns]
        self.added_raw_ = added
        A_raw = X[added] if added else pd.DataFrame(index=X.index)
        self.added_stats_ = {"n_added_raw": len(added),
                             "n_added_nonmissing": int((A_raw.notna().mean() > 0).sum()) if added else 0,
                             "added_missing_frac": float(A_raw.isna().to_numpy().mean()) if added else np.nan,
                             "added_var_mean": float(np.nanmean(A_raw.var().to_numpy())) if added else np.nan,
                             "added_zero_var": int((A_raw.var().fillna(0) <= 1e-12).sum()) if added else 0}
        self.red_, self.cred_, self.raw_ = None, {}, None
        if not added or self.spec.method == "none":
            return pd.DataFrame(index=X.index)
        if self.spec.method == "raw":
            # task-scoped columns kept as they are (impute + scale fitted on the training fold) so the task-pooled
            # ensemble can form per-task members; used only with fusion='task_pool'
            self.raw_ = (added, TabularPreprocessor(max_missing_frac=self.mmf, add_indicators=False, scale=True).fit(X[added]))
            return self.raw_[1].transform(X[added])
        if self.spec.method == "v2_family":
            from sklearn.linear_model import RidgeCV

            from ..features.compact_representation import FamilyReducer

            y_sel = y - RidgeCV(alphas=np.logspace(-2, 4, 31)).fit(Xb, y).predict(Xb) if Xb.shape[1] else y
            k = int(self.cfg.get_path("v2.evaluation.family_components", 10))
            self.red_ = FamilyReducer(self.groups, self.spec.budget, k, "pca_then_supervised", self.seed, self.mmf).fit(X, y_sel)
            return self.red_.transform(X)
        from ..representations.compact_features import CompactReducer

        y_sel = _inner_oof_residual(Xb, y, self.seed, self.inner)
        parts = []
        fams = list(self.groups)
        per = max(1, int(np.ceil((self.spec.budget or 20) / max(len(fams), 1))))
        for fam in fams:
            cols = [c for c in self.groups[fam] if c in X.columns]
            if not cols:
                continue
            r = CompactReducer(self.spec.method, per if len(fams) > 1 else int(self.spec.budget or 20), self.seed, self.mmf,
                               self.rank_grid, self.rrr_alphas).fit(X[cols], y_sel)
            self.cred_[fam] = (cols, r)
            parts.append(r.transform(X[cols], prefix=fam))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

    def _reduce_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.red_ is not None:
            return self.red_.transform(X)
        if getattr(self, "raw_", None) is not None:
            return self.raw_[1].transform(X.reindex(columns=self.raw_[0]))
        parts = [r.transform(X.reindex(columns=cols), prefix=fam) for fam, (cols, r) in self.cred_.items()]
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

    # ---------------------------------------------------------------- fit / predict
    def fit(self, X: pd.DataFrame, y) -> "V3FoldModel":
        y = np.asarray(y, float)
        scale = needs_scaling(self.spec.model) or self.spec.fusion != "early"
        self.prep_ = TabularPreprocessor(max_missing_frac=self.mmf, scale=scale).fit(X[self.base_cols])
        Xb = self.prep_.transform(X[self.base_cols])
        A = self._reduce_fit(X, Xb, y)
        M = pd.concat([Xb, A], axis=1)
        if self.spec.fusion == "task_pool":
            M = pd.concat([M, self._avail(X)], axis=1)
        self.est_ = self._estimator(list(Xb.columns))
        self.est_.fit(M, y)
        self.columns_, self.n_added_, self.base_out_cols_ = list(M.columns), int(A.shape[1]), list(Xb.columns)
        return self

    def matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        Xb = self.prep_.transform(X.reindex(columns=self.base_cols))
        A = self._reduce_transform(X)
        M = pd.concat([Xb, A], axis=1)
        if self.spec.fusion == "task_pool":
            M = pd.concat([M, self._avail(X)], axis=1)
        return M.reindex(columns=self.columns_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est_.predict(self.matrix(X)), float).ravel()

    # ---------------------------------------------------------------- trace
    def trace(self, Xva: pd.DataFrame) -> dict:
        e = self.est_
        t = {"n_base": len(self.base_out_cols_), "n_added": self.n_added_, **self.added_stats_,
             "estimator": type(e).__name__, "alpha": np.nan, "inner_mse_alpha0": np.nan, "inner_mse_best": np.nan,
             "stage2_pred_sd": np.nan, "weights": "", "fallback": False, "fallback_reason": ""}
        if self.n_added_ == 0:
            t.update({"fallback": True, "fallback_reason": "no added columns after reduction"})
        if type(e).__name__ == "NestedResidualRegressor":
            t["alpha"] = float(e.alpha_)
            t["inner_mse_alpha0"] = float(e.inner_mse_.get(0.0, np.nan))
            t["inner_mse_best"] = float(min(e.inner_mse_.values()))
            t["inner_mse"] = ";".join(f"{a:g}:{v:.4f}" for a, v in sorted(e.inner_mse_.items()))
            if e.m2_ is not None:
                M = self.matrix(Xva)
                t["stage2_pred_sd"] = float(np.std(np.asarray(e.m2_.predict(M[e.c2_]), float)))
            else:
                t["stage2_pred_sd"] = 0.0
                if not t["fallback"]:
                    t.update({"fallback": True, "fallback_reason": "residual shrinkage alpha=0 chosen by inner CV"})
        for attr in ("weights_",):
            w = getattr(e, attr, None)
            if w is not None:
                ws = pd.Series(w) if not isinstance(w, pd.Series) else w
                t["weights"] = ";".join(f"{k}:{float(v):.3f}" for k, v in ws.items())
                if type(e).__name__ in ("ConvexLateFusion", "RobustTaskEnsemble"):
                    nb = float(ws.drop(labels=["BASE"], errors="ignore").sum())
                    if nb <= 1e-6 and not t["fallback"]:
                        t.update({"fallback": True, "fallback_reason": "zero weight on every added-family member"})
        return t


class V3Evaluator(StudentLevelEvaluator):
    """StudentLevelEvaluator with V3 specs, per-fold tracing, explicit fallback status, controls and V3 caches.

    `fold_model_cls` is the per-fold model class (V3FoldModel). Subclasses (V4) may set a V3FoldModel subclass that
    adds fusion strategies; the fitting / tracing / caching protocol is unchanged."""

    fold_model_cls = V3FoldModel

    def __init__(self, ctx, stage: str, providers: dict | None = None):
        super().__init__(ctx, stage, providers)
        self.cache_dir = ctx.paths.v3_cache / "eval"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.oof_dir = ctx.stage_dir(stage) / "oof"
        self.oof_dir.mkdir(parents=True, exist_ok=True)
        from ..v3.registry import v3_registries

        self.v3reg = v3_registries(ctx)
        self.tol = float(ctx.cfg.get_path("v3.audit.zero_delta_tol", 1e-12))

    def baseline_v3(self, name: str = "ASR00_TABULAR_GAM") -> V3Spec:
        return V3Spec(name, model=self.baseline["model"], providers=[], budget=None, method="none", fusion="early",
                      base_families=list(self.baseline["families"]), group="baseline", experiment="ASR00")

    def _folds(self, scheme: str):
        from ..v3.common import folds

        return folds(self.ctx, scheme)

    def _key(self, spec: V3Spec, scheme: str) -> str:
        fps = {p: self.providers[p].fingerprint() for p in spec.providers if p in self.providers}
        return stable_hash({"v3spec": spec.key(), "base": self._base_families(spec), "bm": self.baseline["model"],
                            "scheme": scheme, "fps": fps, "reg": self.reg.data_hash(), "tag": self.base_runner.tag,
                            "folds": [(f.repeat, f.fold) for f in self._folds(scheme)],
                            "cfg": self.ctx.cfg.fingerprint(["features", "validation.inner_splits", "models",
                                                            "v2.evaluation.family_components", "v3.representations"])})

    def _control(self, spec: V3Spec, Xtr, Xva, groups, ytr, seed):
        rng = np.random.default_rng(seed)
        added = [c for cols in groups.values() for c in cols]
        if spec.control == "permuted_features" and added:
            Xtr, Xva = Xtr.copy(), Xva.copy()
            Xtr[added] = Xtr[added].to_numpy()[rng.permutation(len(Xtr))]
            Xva[added] = Xva[added].to_numpy()[rng.permutation(len(Xva))]
        if spec.control == "shuffled_outcome":
            ytr = ytr[rng.permutation(len(ytr))]
        return Xtr, Xva, ytr

    def evaluate(self, spec: V3Spec, scheme: str = "repeated_kfold", baseline: V3Result | None = None,
                 log: bool = True) -> V3Result:
        empty = pd.DataFrame()
        missing = [p for p in spec.providers if p not in self.providers or not self.providers[p].available()]
        if missing:
            return V3Result(spec, scheme, empty, empty, empty, {"reason": f"unavailable providers {missing}"}, "unavailable")
        fl = self._folds(scheme)
        if not fl:
            return V3Result(spec, scheme, empty, empty, empty, {"reason": "no folds"}, "no_folds")
        key = self._key(spec, scheme)
        p_oof, p_tr, p_meta = self.cache_dir / f"{key}_oof.parquet", self.cache_dir / f"{key}_trace.parquet", self.cache_dir / f"{key}.json"
        if p_oof.exists() and p_meta.exists() and p_tr.exists():
            meta = read_json(p_meta)
            res = V3Result(spec, scheme, pd.read_parquet(p_oof), pd.DataFrame(meta["fold_metrics"]), pd.read_parquet(p_tr),
                           meta["summary"], meta["status"])
            return self._finish(res, baseline, log, cached=True)
        from ..diagnostics.scientific_validation import full_metrics

        rows, fm, tr_rows = [], [], []
        t0 = time.perf_counter()
        try:
            for f in fl:
                Xtr, Xva, base_cols, groups = self.fold_frames(f, spec)
                va_ids, tr_ids = set(map(str, f.valid_ids)), set(map(str, f.train_ids))
                if (set(Xva.index) - va_ids) or (set(Xtr.index) - tr_ids) or (set(Xva.index) & set(Xtr.index)) or Xva.empty:
                    raise AssertionError(f"student id misalignment in fold {f.repeat}/{f.fold}")
                n_missing_valid = len(va_ids - set(Xva.index))
                ytr = self.y.reindex(Xtr.index).to_numpy(float)
                yva = self.y.reindex(Xva.index).to_numpy(float)
                if not (np.isfinite(ytr).all() and np.isfinite(yva).all()):
                    raise AssertionError(f"missing targets after alignment in fold {f.repeat}/{f.fold}")
                seed = self.ctx.seed + f.repeat * 31 + f.fold
                Xtr, Xva, ytr_fit = self._control(spec, Xtr, Xva, groups, ytr, seed)
                fmod = self.fold_model_cls(spec, self.ctx.cfg, seed, base_cols, groups, self.tasks,
                                           self.baseline["model"]).fit(Xtr, ytr_fit)
                p = fmod.predict(Xva)
                rows.append(pd.DataFrame({ID: Xva.index, "repeat": f.repeat, "fold": f.fold, "y_true": yva, "y_pred": p,
                                          "fold_label": f.label}))
                mr = metric_row(yva, p)
                fm.append({"repeat": f.repeat, "fold": f.fold, "label": f.label, "n_base": len(base_cols),
                           "n_added": fmod.n_added_, "n_features": len(fmod.columns_), **mr})
                tr_rows.append({"scheme": scheme, "repeat": f.repeat, "fold": f.fold, "n_train": len(Xtr), "n_valid": len(Xva),
                                "pred_sd": float(np.std(p)), "r2_fold": mr["r2"], "n_missing_valid": n_missing_valid,
                                "id_alignment_ok": n_missing_valid == 0, **fmod.trace(Xva)})
        except ModelUnavailable as e:
            return V3Result(spec, scheme, empty, empty, empty, {"reason": f"unavailable: {e}"}, "unavailable")
        except Exception as e:  # noqa: BLE001 - recorded, never replaced by baseline predictions
            return V3Result(spec, scheme, empty, pd.DataFrame(fm), pd.DataFrame(tr_rows),
                            {"reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, "failed")
        oof = pd.concat(rows, ignore_index=True)
        summ = full_metrics(oof) if scheme == "repeated_kfold" else {**full_metrics(oof), "r2_mean": metric_row(oof["y_true"], oof["y_pred"])["r2"]}
        fmd, trd = pd.DataFrame(fm), pd.DataFrame(tr_rows)
        summ.update({"seconds": time.perf_counter() - t0, "n_added_mean": float(fmd["n_added"].mean()),
                     "n_features_mean": float(fmd["n_features"].mean()), "fold_r2_min": float(fmd["r2"].min()),
                     "n_students": int(oof[ID].nunique()), "fallback_folds": int(trd["fallback"].sum()), "n_folds": len(trd)})
        status = "ok"
        if spec.providers:
            if int(trd["n_added_raw"].sum()) == 0:
                status = "no_added_features"
            elif bool(trd["fallback"].all()):
                status = "fallback_baseline"
            elif bool(trd["fallback"].any()):
                status = "partial_fallback"
        res = V3Result(spec, scheme, oof, fmd, trd, summ, status)
        atomic_write_parquet(oof, p_oof)
        atomic_write_parquet(trd.astype({c: str for c in trd.columns if trd[c].dtype == object}), p_tr)
        write_json(p_meta, {"summary": summ, "fold_metrics": fm, "status": status, "spec": spec.as_dict(), "scheme": scheme})
        return self._finish(res, baseline, log)

    def _finish(self, res: V3Result, baseline: V3Result | None, log: bool, cached: bool = False) -> V3Result:
        if baseline is not None and not baseline.oof.empty and not res.oof.empty:
            j = res.oof.merge(baseline.oof, on=[ID, "repeat", "fold"], suffixes=("", "_b"))
            d = (j["y_pred"] - j["y_pred_b"]).abs()
            res.summary["max_abs_diff_vs_baseline"] = float(d.max())
            res.summary["mean_abs_diff_vs_baseline"] = float(d.mean())
            res.summary["identical_to_baseline"] = bool(d.max() <= self.tol)
            if res.spec.providers and res.status == "ok" and d.max() <= self.tol:
                res.status = "fallback_baseline"
        if not res.oof.empty:
            atomic_write_parquet(res.oof, self.oof_dir / f"{res.spec.name}__{res.scheme}.parquet")
        if log:
            s = res.summary
            self.v3reg.log_experiment(f"{self.stage}:{res.spec.name}:{res.scheme}:{res.spec.key()}", stage=self.stage,
                                      name=res.spec.name, experiment=res.spec.experiment, group=res.spec.group,
                                      model=res.spec.model, fusion=res.spec.fusion, method=res.spec.method,
                                      providers="+".join(res.spec.providers), budget=res.spec.budget,
                                      control=res.spec.control, scheme=res.scheme, status=res.status,
                                      r2_mean=s.get("r2_mean"), r2_sd=s.get("r2_sd"), r2_pooled=s.get("r2_pooled"),
                                      r2_fold_mean=s.get("r2_fold_mean"), rmse=s.get("rmse_mean"), mae=s.get("mae_mean"),
                                      n_added=s.get("n_added_mean"), fallback_folds=s.get("fallback_folds"),
                                      identical_to_baseline=s.get("identical_to_baseline"), reason=s.get("reason"),
                                      cached=cached)
        return res

    def compare(self, base: V3Result, new: V3Result, n_boot: int | None = None) -> dict:
        if base.oof.empty or new.oof.empty:
            return {}
        from ..validation.metrics import incremental_r2

        nb = int(n_boot or self.ctx.cfg.get_path("v3.ablation.bootstrap", 1000))
        if base.scheme == "repeated_kfold":
            return incremental_r2(base.oof, new.oof, nb, 0)
        from ..stats import delta_r2_bootstrap

        j = base.oof.merge(new.oof, on=[ID, "repeat", "fold"], suffixes=("_b", "_n"))
        est, lo, hi = delta_r2_bootstrap(j["y_true_b"], j["y_pred_b"], j["y_pred_n"], nb, 0)
        return {"delta_r2": est, "delta_ci_low": lo, "delta_ci_high": hi}

    def fit_full_v3(self, spec: V3Spec, target_ids: list[str]):
        """Fit on ALL training students (submission). Returns (V3FoldModel, Xtr, Xte)."""
        from ..features.assemble import make_builder
        from ..features.builder import select_families

        builder = make_builder(self.ctx, None, with_audio=False)
        builder.fit(self.all_train)
        Xtr_b = builder.transform(self.all_train)
        Xte_b = builder.transform(target_ids).reindex(columns=Xtr_b.columns)
        base_cols = select_families(Xtr_b.columns, self._base_families(spec))
        keep = list(dict.fromkeys(base_cols + [f"BEHAVIOR__{t}__n_items__value" for t in self.tasks if f"BEHAVIOR__{t}__n_items__value" in Xtr_b]))
        tr, te, groups = [Xtr_b[keep]], [Xte_b[keep]], {}
        for pname in spec.providers:
            T = self.providers[pname].table(self.all_train, sorted(set(self.all_train) | set(target_ids)))
            T = T.loc[:, ~T.columns.isin(keep)]
            tr.append(T.reindex(Xtr_b.index))
            te.append(T.reindex(Xte_b.index))
            for c in T.columns:
                groups.setdefault(family_of(c), []).append(c)
        Xtr = pd.concat(tr, axis=1)
        Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
        Xte = pd.concat(te, axis=1)
        Xte = Xte.loc[:, ~Xte.columns.duplicated()].reindex(columns=Xtr.columns)
        y = self.y.reindex(Xtr.index).to_numpy(float)
        fmod = self.fold_model_cls(spec, self.ctx.cfg, self.ctx.seed, base_cols, groups, self.tasks,
                                   self.baseline["model"]).fit(Xtr, y)
        return fmod, Xtr, Xte


def result_row(res: V3Result, delta: dict | None = None) -> dict:
    s = res.summary or {}
    sp = res.spec
    return {"name": sp.name, "experiment": sp.experiment, "group": sp.group, "scheme": res.scheme, "status": res.status,
            "model": sp.model, "fusion": sp.fusion, "method": sp.method, "providers": "+".join(sp.providers),
            "budget": sp.budget, "control": sp.control, "r2_mean": s.get("r2_mean"), "r2_sd": s.get("r2_sd", 0.0),
            "r2_pooled": s.get("r2_pooled"), "r2_fold_mean": s.get("r2_fold_mean"),
            "r2_repeat_avg_pred": s.get("r2_repeat_avg_pred"), "rmse": s.get("rmse_mean"), "mae": s.get("mae_mean"),
            "fold_r2_min": s.get("fold_r2_min"), "n_added": s.get("n_added_mean"), "n_features": s.get("n_features_mean"),
            "fallback_folds": s.get("fallback_folds"), "n_folds": s.get("n_folds"),
            "identical_to_baseline": s.get("identical_to_baseline"), "max_abs_diff_vs_baseline": s.get("max_abs_diff_vs_baseline"),
            "reason": s.get("reason"), **(delta or {})}


class InnerResidualOracle:
    """Training-fold baseline residuals WITHOUT the target students: V1 tabular features fitted on the training ids,
    baseline model predicted out-of-fold inside the training ids (for nested MIL)."""

    def __init__(self, ev: V3Evaluator):
        self.ev = ev
        self._mem: dict[str, pd.Series] = {}

    def __call__(self, train_ids: list[str]) -> pd.Series:
        from sklearn.model_selection import KFold

        from ..features.assemble import make_builder
        from ..features.builder import select_families

        k = stable_hash(sorted(train_ids))
        if k in self._mem:
            return self._mem[k]
        b = make_builder(self.ev.ctx, None, with_audio=False)
        b.fit(list(train_ids))
        X = b.transform(list(train_ids))
        X.index = X.index.astype(str)
        cols = select_families(X.columns, self.ev.baseline["families"])
        y = self.ev.y.reindex(X.index).to_numpy(float)
        oof = np.zeros(len(y))
        for tr, va in KFold(4, shuffle=True, random_state=0).split(X):
            prep = TabularPreprocessor().fit(X.iloc[tr][cols])
            m = make_model(self.ev.baseline["model"], self.ev.ctx.cfg, seed=0).fit(prep.transform(X.iloc[tr][cols]), y[tr])
            oof[va] = np.asarray(m.predict(prep.transform(X.iloc[va][cols])), float).ravel()
        r = pd.Series(y - oof, index=X.index)
        self._mem[k] = r
        return r
