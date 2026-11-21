"""Compact multimodal student-level regression on the frozen V2 validation registry.

StudentLevelEvaluator.evaluate(spec, scheme):
  for every fold of the registry:
    baseline columns   <- V1 tabular StudentFeatureBuilder fitted on the training fold (V1 fold cache reused)
    provider columns   <- each CachedProvider fitted on the training fold (nested where labels are involved)
    BudgetedFoldModel  <- fitted on the training fold only:
        baseline preprocessing (impute + indicators + scale) | per-family PCA + budgeted supervised selection
        (ranked against the baseline residual) or fold-validated literacy indices | fusion estimator
    predict the validation students
  -> OOF table, fold metrics, full metrics; cached on disk by a content key so an interrupted stage resumes.
The comparator for every incremental-R2 statement is the strongest validated tabular baseline (v2.baseline).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .. import ID
from ..features.builder import family_of, select_families
from ..features.compact_representation import FamilyReducer, IndexTransformer
from ..io_utils import atomic_write_parquet, read_json, stable_hash, write_json
from ..validation.metrics import incremental_r2, metric_row
from ..validation.runner import CVResult
from .preprocessing import TabularPreprocessor
from .residual_fusion import (ConvexLateFusion, DomainPartialPooling, NestedResidualRegressor, RegularizedStack,
                              RobustTaskEnsemble)
from .zoo import ModelUnavailable, make_model, needs_scaling

FUSIONS = ("early", "residual", "late", "task_pool", "stack", "moe")


@dataclass
class CompactSpec:
    name: str
    model: str = "RIDGE"
    base_families: list[str] | None = None        # None -> strongest tabular baseline families
    base_model: str | None = None                 # stage-1 model for fusion='residual' (None -> baseline model)
    providers: list[str] = field(default_factory=list)
    budget: int | None = None
    reducer: str = "pca_then_supervised"          # none | pca_family | supervised | pca_then_supervised
    family_components: int | None = None
    fusion: str = "early"
    params: dict = field(default_factory=dict)
    indices: bool = False
    stack_models: tuple = ("RIDGE", "GAM", "LIGHTGBM")

    def key(self) -> str:
        d = asdict(self)
        d.pop("name")
        return stable_hash(d)

    def as_dict(self) -> dict:
        return asdict(self)


class BudgetedFoldModel:
    def __init__(self, spec: CompactSpec, cfg, seed: int, base_cols: list[str], groups: dict[str, list[str]],
                 tasks: list[str], baseline_model: str):
        self.spec, self.cfg, self.seed = spec, cfg, seed
        self.base_cols, self.groups, self.tasks = list(base_cols), {k: list(v) for k, v in groups.items()}, list(tasks)
        self.baseline_model = baseline_model
        ev = cfg.get_path("v2.evaluation", {}) or {}
        self.k = int(spec.family_components or ev.get("family_components", 10))
        self.select_on_residual = bool(ev.get("select_on_residual", True))
        self.inner = int(cfg.get_path("validation.inner_splits", 4))
        self.mmf = float(cfg.get_path("models.max_missing_frac", 0.6))

    def _avail(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for t in self.tasks:
            c = f"BEHAVIOR__{t}__n_items__value"
            out[f"__avail__{t}"] = (X[c].fillna(0).to_numpy(float) > 0).astype(float) if c in X else 1.0
        return out

    def _estimator(self, Xb_cols: list[str], domain_cols: list[str]):
        s = self.spec
        if s.fusion == "early":
            return make_model(s.model, self.cfg, seed=self.seed, params=dict(s.params))
        if s.fusion == "residual":
            return NestedResidualRegressor(stage1=make_model(s.base_model or self.baseline_model, self.cfg, seed=self.seed),
                                           stage2=make_model(s.model, self.cfg, seed=self.seed, params=dict(s.params)),
                                           stage1_cols=Xb_cols, inner_splits=self.inner, seed=self.seed)
        if s.fusion == "late":
            return ConvexLateFusion(base_cols=Xb_cols, base=make_model(s.model, self.cfg, seed=self.seed),
                                    inner_splits=self.inner, seed=self.seed)
        if s.fusion == "task_pool":
            return RobustTaskEnsemble(base_cols=Xb_cols, tasks=tuple(self.tasks), inner_splits=self.inner, seed=self.seed)
        if s.fusion == "stack":
            return RegularizedStack([make_model(m, self.cfg, seed=self.seed) for m in s.stack_models],
                                    inner_splits=self.inner, seed=self.seed)
        if s.fusion == "moe":
            return DomainPartialPooling(domain_cols, int(self.cfg.get_path("v2.multimodal.moe_min_domain_size", 40)),
                                        inner_splits=self.inner, seed=self.seed)
        raise ValueError(f"unknown fusion {s.fusion}")

    def fit(self, X: pd.DataFrame, y) -> "BudgetedFoldModel":
        y = np.asarray(y, float)
        scale = needs_scaling(self.spec.model) or self.spec.fusion != "early"
        self.prep_ = TabularPreprocessor(max_missing_frac=self.mmf, scale=scale).fit(X[self.base_cols])
        Xb = self.prep_.transform(X[self.base_cols])
        added = [c for cols in self.groups.values() for c in cols]
        self.red_, self.idx_ = None, None
        if added:
            if self.select_on_residual and Xb.shape[1]:
                from sklearn.linear_model import RidgeCV

                y_sel = y - RidgeCV(alphas=np.logspace(-2, 4, 31)).fit(Xb, y).predict(Xb)
            else:
                y_sel = y
            if self.spec.indices:
                recipes = self.cfg.get_path("v2.multimodal.indices.recipes", {}) or {}
                self.idx_ = IndexTransformer(recipes, int(self.cfg.get_path("v2.multimodal.indices.min_components", 2))).fit(X[added], y)
                A = self.idx_.transform(X[added])
            else:
                self.red_ = FamilyReducer(self.groups, self.spec.budget, self.k, self.spec.reducer, self.seed, self.mmf).fit(X, y_sel)
                A = self.red_.transform(X)
        else:
            A = pd.DataFrame(index=X.index)
        M = pd.concat([Xb, A], axis=1)
        if self.spec.fusion == "task_pool":
            M = pd.concat([M, self._avail(X)], axis=1)
        dom = [c for c in M.columns if c.startswith("RECORDING__orig_sr_low") or c.startswith("SCREENING__instrument_")]
        self.est_ = self._estimator(list(Xb.columns), dom[:2])
        self.est_.fit(M, y)
        self.columns_ = list(M.columns)
        self.n_added_ = int(A.shape[1])
        return self

    def transform_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        Xb = self.prep_.transform(X.reindex(columns=self.base_cols))
        if self.idx_ is not None:
            added = [c for cols in self.groups.values() for c in cols]
            A = self.idx_.transform(X.reindex(columns=added))
        elif self.red_ is not None:
            A = self.red_.transform(X)
        else:
            A = pd.DataFrame(index=X.index)
        M = pd.concat([Xb, A], axis=1)
        if self.spec.fusion == "task_pool":
            M = pd.concat([M, self._avail(X)], axis=1)
        return M.reindex(columns=self.columns_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est_.predict(self.transform_matrix(X)), float).ravel()


def equal_n_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired transfer vs within-domain R2 at identical training size and identical validation students."""
    if fold_metrics.empty or "label" not in fold_metrics:
        return pd.DataFrame()
    fm = fold_metrics.copy()
    fm["mode"] = fm["label"].str.split(":").str[0]
    fm["target"] = fm["label"].str.split("->").str[1].str.split("|").str[0]
    piv = fm.pivot_table(index=["repeat", "target"], columns="mode", values="r2").reset_index()
    if not {"transfer", "within"} <= set(piv.columns):
        return pd.DataFrame()
    piv["gap_within_minus_transfer"] = piv["within"] - piv["transfer"]
    return piv.groupby("target").agg(r2_within=("within", "mean"), r2_transfer=("transfer", "mean"),
                                     gap_mean=("gap_within_minus_transfer", "mean"), gap_sd=("gap_within_minus_transfer", "std"),
                                     n_repeats=("repeat", "nunique")).reset_index()


class StudentLevelEvaluator:
    def __init__(self, ctx, stage: str, providers: dict | None = None):
        from ..stages._common import get_runner
        from ..v2.baseline import strongest_tabular
        from ..v2.registry import v2_registries
        from ..v2.validation_registry import get_registry

        self.ctx, self.stage = ctx, stage
        self.reg = get_registry(ctx)
        self.base_runner = get_runner(ctx, with_audio=False)
        self.y = ctx.bundle().y.copy()
        self.y.index = self.y.index.astype(str)
        self.all_train = sorted(self.y.index)
        self.providers = dict(providers or {})
        self.baseline = strongest_tabular(ctx)
        self.tasks = sorted(ctx.bundle().all_clips["task"].unique())
        self.v2reg = v2_registries(ctx)
        self.cache_dir = ctx.paths.v2_cache / "eval"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.oof_dir = ctx.stage_dir(stage) / "oof"
        self.oof_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ specs
    def baseline_spec(self, name: str = "V2_BASELINE") -> CompactSpec:
        return CompactSpec(name, model=self.baseline["model"], base_families=list(self.baseline["families"]))

    def add_provider(self, key: str, cp) -> None:
        self.providers[key] = cp

    def _base_families(self, spec: CompactSpec) -> list[str]:
        return list(spec.base_families or self.baseline["families"])

    def _cache_key(self, spec: CompactSpec, scheme: str) -> str:
        fps = {p: self.providers[p].fingerprint() for p in spec.providers if p in self.providers}
        return stable_hash({"spec": spec.key(), "base": self._base_families(spec), "bm": self.baseline["model"],
                            "scheme": scheme, "fps": fps, "reg": self.reg.data_hash(), "tag": self.base_runner.tag,
                            "folds": [(f.repeat, f.fold) for f in self.reg.folds(scheme)],
                            "cfg": self.ctx.cfg.fingerprint(["features", "validation.inner_splits", "models", "v2.evaluation",
                                                            "v2.multimodal.indices", "v2.multimodal.moe_min_domain_size"])})

    # ------------------------------------------------------------------ frames
    def fold_frames(self, fold, spec: CompactSpec):
        Xtr_b, Xva_b = self.base_runner.fold_data(fold)
        base_cols = select_families(Xtr_b.columns, self._base_families(spec))
        keep = list(dict.fromkeys(base_cols + [f"BEHAVIOR__{t}__n_items__value" for t in self.tasks if f"BEHAVIOR__{t}__n_items__value" in Xtr_b]))
        tr, va = [Xtr_b[keep]], [Xva_b[keep]]
        groups: dict[str, list[str]] = {}
        for pname in spec.providers:
            cp = self.providers[pname]
            T = cp.table(fold.train_ids, self.all_train)
            T = T.loc[:, ~T.columns.isin(keep)]
            tr.append(T.reindex(Xtr_b.index))
            va.append(T.reindex(Xva_b.index))
            for c in T.columns:
                groups.setdefault(family_of(c), []).append(c)
        Xtr = pd.concat(tr, axis=1)
        Xva = pd.concat(va, axis=1)
        Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
        Xva = Xva.loc[:, ~Xva.columns.duplicated()].reindex(columns=Xtr.columns)
        return Xtr, Xva, base_cols, groups

    # --------------------------------------------------------------- evaluate
    def evaluate(self, spec: CompactSpec, scheme: str = "repeated_kfold", log: bool = True) -> CVResult:
        missing = [p for p in spec.providers if p not in self.providers or not self.providers[p].available()]
        if missing:
            return CVResult(spec, scheme, pd.DataFrame(), pd.DataFrame(), {"status": f"unavailable providers {missing}"}, "unavailable")
        folds = self.reg.folds(scheme)
        if not folds:
            return CVResult(spec, scheme, pd.DataFrame(), pd.DataFrame(), {"status": "no folds"}, "no_folds")
        key = self._cache_key(spec, scheme)
        p_oof, p_meta = self.cache_dir / f"{key}_oof.parquet", self.cache_dir / f"{key}.json"
        if p_oof.exists() and p_meta.exists():
            meta = read_json(p_meta)
            res = CVResult(spec, scheme, pd.read_parquet(p_oof), pd.DataFrame(meta["fold_metrics"]), meta["summary"], meta["status"])
            self._save(res, log)
            return res
        from ..diagnostics.scientific_validation import full_metrics

        rows, fm = [], []
        t0 = time.perf_counter()
        try:
            for f in folds:
                Xtr, Xva, base_cols, groups = self.fold_frames(f, spec)
                ytr = self.y.reindex(Xtr.index).to_numpy(float)
                fmod = BudgetedFoldModel(spec, self.ctx.cfg, self.ctx.seed + f.repeat * 31 + f.fold, base_cols, groups,
                                         self.tasks, self.baseline["model"]).fit(Xtr, ytr)
                p = fmod.predict(Xva)
                yva = self.y.reindex(Xva.index).to_numpy(float)
                rows.append(pd.DataFrame({ID: Xva.index, "repeat": f.repeat, "fold": f.fold, "y_true": yva, "y_pred": p,
                                          "fold_label": f.label}))
                fm.append({"repeat": f.repeat, "fold": f.fold, "label": f.label, "n_base": len(base_cols),
                           "n_added": fmod.n_added_, "n_features": len(fmod.columns_), **metric_row(yva, p)})
        except ModelUnavailable as e:
            return CVResult(spec, scheme, pd.DataFrame(), pd.DataFrame(), {"status": f"unavailable: {e}"}, "unavailable")
        oof = pd.concat(rows, ignore_index=True)
        summ = full_metrics(oof) if scheme in ("repeated_kfold",) else {**full_metrics(oof), "r2_mean": metric_row(oof["y_true"], oof["y_pred"])["r2"]}
        fmd = pd.DataFrame(fm)
        summ.update({"seconds": time.perf_counter() - t0, "n_added_mean": float(fmd["n_added"].mean()),
                     "n_features_mean": float(fmd["n_features"].mean()), "fold_r2_min": float(fmd["r2"].min()),
                     "n_students": int(oof[ID].nunique())})
        res = CVResult(spec, scheme, oof, fmd, summ, "ok")
        atomic_write_parquet(oof, p_oof)
        write_json(p_meta, {"summary": summ, "fold_metrics": fm, "status": "ok", "spec": spec.as_dict(), "scheme": scheme})
        self._save(res, log)
        return res

    def _save(self, res: CVResult, log: bool) -> None:
        if res.oof.empty:
            return
        atomic_write_parquet(res.oof, self.oof_dir / f"{res.spec.name}__{res.scheme}.parquet")
        if log:
            s = res.summary
            self.v2reg.log_experiment(f"{self.stage}:{res.spec.name}:{res.scheme}:{res.spec.key()}", stage=self.stage,
                                      name=res.spec.name, model=res.spec.model, fusion=res.spec.fusion,
                                      providers="+".join(res.spec.providers), budget=res.spec.budget,
                                      base_families="+".join(self._base_families(res.spec)), scheme=res.scheme,
                                      r2_mean=s.get("r2_mean"), r2_sd=s.get("r2_sd"), r2_pooled=s.get("r2_pooled"),
                                      r2_fold_mean=s.get("r2_fold_mean"), r2_repeat_avg_pred=s.get("r2_repeat_avg_pred"),
                                      rmse=s.get("rmse_mean"), mae=s.get("mae_mean"), n_added=s.get("n_added_mean"),
                                      n_features=s.get("n_features_mean"), seconds=s.get("seconds"))

    def compare(self, base: CVResult, new: CVResult, n_boot: int | None = None) -> dict:
        if base.oof.empty or new.oof.empty:
            return {}
        nb = int(n_boot or self.ctx.cfg.get_path("v2.evaluation.bootstrap", 1000))
        if base.scheme == "repeated_kfold" and new.scheme == "repeated_kfold":
            return incremental_r2(base.oof, new.oof, nb, 0)
        j = base.oof.merge(new.oof, on=[ID, "repeat", "fold"], suffixes=("_b", "_n"))
        from ..stats import delta_r2_bootstrap

        est, lo, hi = delta_r2_bootstrap(j["y_true_b"], j["y_pred_b"], j["y_pred_n"], nb, 0)
        return {"delta_r2": est, "delta_ci_low": lo, "delta_ci_high": hi}

    def fit_full(self, spec: CompactSpec, target_ids: list[str]):
        """Fit on ALL training students; return (fitted BudgetedFoldModel, design for target_ids)."""
        from ..features.assemble import make_builder

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
        fmod = BudgetedFoldModel(spec, self.ctx.cfg, self.ctx.seed, base_cols, groups, self.tasks, self.baseline["model"]).fit(Xtr, y)
        return fmod, Xtr, Xte
