"""Fold-aware cross-validation runner.

For every outer fold:  builder.fit(train students) -> transform(train, valid) -> [family column selection]
-> FoldModel(preprocessor -> supervised selector -> estimator).fit(train) -> predict(valid).
Fold feature matrices are cached (memory + Parquet) and shared by every model evaluated on the same fold.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import ID
from ..features.builder import select_families
from ..io_utils import atomic_write_parquet, stable_hash
from ..models.preprocessing import FoldModel
from ..models.zoo import ModelUnavailable, make_model, needs_scaling
from .metrics import metric_row, summarize_oof
from .splits import Fold, build_folds


@dataclass
class ModelSpec:
    name: str
    model: str
    families: list[str]
    params: dict = field(default_factory=dict)
    selector_k: int | None = None
    columns: list[str] | None = None        # explicit column list (overrides families)

    def key(self) -> str:
        return stable_hash({"m": self.model, "f": sorted(self.families), "p": self.params, "k": self.selector_k,
                            "c": sorted(self.columns) if self.columns else None})

    def as_dict(self) -> dict:
        return {"name": self.name, "model": self.model, "families": list(self.families), "params": dict(self.params),
                "selector_k": self.selector_k, "columns": self.columns}


@dataclass
class CVResult:
    spec: ModelSpec
    scheme: str
    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: dict
    status: str = "ok"


class CVRunner:
    def __init__(self, ctx, builder, tag: str = "default", disk_cache: bool = True):
        self.ctx, self.builder, self.tag = ctx, builder, tag
        b = ctx.bundle()
        self.train = b.train
        self.y = b.y
        self.y.index = self.y.index.astype(str)
        self._folds: dict[str, list[Fold]] = {}
        self._mem: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.disk_cache = disk_cache
        self.cache_dir = ctx.paths.cache / "fold_features" / tag
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_boot = int(ctx.cfg.get_path("validation.bootstrap", 1000))

    def folds(self, scheme: str) -> list[Fold]:
        if scheme not in self._folds:
            self._folds[scheme] = build_folds(scheme, self.train, self.y, self.ctx.cfg, self.ctx.seed)
        return self._folds[scheme]

    def fold_data(self, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = stable_hash({"tr": sorted(fold.train_ids), "va": sorted(fold.valid_ids), "tag": self.tag,
                           "cfg": self.ctx.cfg.fingerprint(["features", "embeddings.pca_components"])})
        if key in self._mem:
            return self._mem[key]
        ptr, pva = self.cache_dir / f"{key}_tr.parquet", self.cache_dir / f"{key}_va.parquet"
        if self.disk_cache and ptr.exists() and pva.exists():
            Xtr, Xva = pd.read_parquet(ptr), pd.read_parquet(pva)
        else:
            self.builder.fit(fold.train_ids)
            Xtr = self.builder.transform(fold.train_ids)
            Xva = self.builder.transform(fold.valid_ids)
            Xva = Xva.reindex(columns=Xtr.columns)
            if self.disk_cache:
                atomic_write_parquet(Xtr, ptr, index=True)
                atomic_write_parquet(Xva, pva, index=True)
        Xtr.index, Xva.index = Xtr.index.astype(str), Xva.index.astype(str)
        if len(self._mem) > 64:
            self._mem.pop(next(iter(self._mem)))
        self._mem[key] = (Xtr, Xva)
        return Xtr, Xva

    def _columns(self, spec: ModelSpec, cols) -> list[str]:
        if spec.columns:
            return [c for c in spec.columns if c in set(cols)]
        return select_families(cols, spec.families)

    def evaluate(self, spec: ModelSpec, scheme: str = "repeated_kfold", log: bool = True, stage: str = "") -> CVResult:
        folds = self.folds(scheme)
        rows, fm = [], []
        t0 = time.perf_counter()
        sel_k = spec.selector_k if spec.selector_k is not None else self.ctx.cfg.get_path("models.selector_k")
        try:
            for f in folds:
                Xtr, Xva = self.fold_data(f)
                cols = self._columns(spec, Xtr.columns)
                if not cols:
                    raise ValueError(f"no columns for families {spec.families}")
                ytr = self.y.reindex(Xtr.index).values
                est = make_model(spec.model, self.ctx.cfg, seed=self.ctx.seed + f.repeat * 31 + f.fold, params=dict(spec.params))
                fmod = FoldModel(est, sel_k, float(self.ctx.cfg.get_path("models.max_missing_frac", 0.6)), needs_scaling(spec.model))
                fmod.fit(Xtr[cols], ytr)
                p = fmod.predict(Xva[cols])
                yva = self.y.reindex(Xva.index).values
                rows.append(pd.DataFrame({ID: Xva.index, "repeat": f.repeat, "fold": f.fold, "y_true": yva, "y_pred": p,
                                          "fold_label": f.label}))
                fm.append({"repeat": f.repeat, "fold": f.fold, "label": f.label, "n_features": len(fmod.columns_),
                           **metric_row(yva, p)})
        except ModelUnavailable as e:
            return CVResult(spec, scheme, pd.DataFrame(), pd.DataFrame(), {"status": f"unavailable: {e}"}, "unavailable")
        oof = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if oof.empty:
            return CVResult(spec, scheme, oof, pd.DataFrame(fm), {"status": "no folds"}, "no_folds")
        if scheme == "repeated_kfold":
            summ = summarize_oof(oof)
        else:  # single-pass schemes: pooled OOF R2 + mean fold R2
            pooled = metric_row(oof["y_true"], oof["y_pred"])
            fmd = pd.DataFrame(fm)
            summ = {"r2_mean": pooled["r2"], "r2_sd": float(fmd["r2"].std(ddof=0)), "rmse_mean": pooled["rmse"],
                    "mae_mean": pooled["mae"], "fold_r2_mean": float(fmd["r2"].mean()), "n_students": int(oof[ID].nunique()),
                    "n_repeats": 1}
        summ.update({"seconds": time.perf_counter() - t0, "n_features_mean": float(np.mean([r["n_features"] for r in fm]))})
        res = CVResult(spec, scheme, oof, pd.DataFrame(fm), summ)
        if log:
            self.ctx.registries.log_experiment(f"{stage}:{spec.name}:{scheme}:{spec.key()}", stage=stage, model=spec.model,
                                               name=spec.name, families="+".join(spec.families), scheme=scheme,
                                               r2_mean=summ["r2_mean"], r2_sd=summ["r2_sd"], rmse=summ["rmse_mean"],
                                               mae=summ["mae_mean"], n_features=summ["n_features_mean"],
                                               config_hash=self.ctx.cfg.fingerprint(["features", "validation", "models"]))
        return res

    def save_oof(self, res: CVResult, stage: str) -> None:
        if res.oof.empty:
            return
        d = self.ctx.stage_dir(stage) / "oof"
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(res.oof, d / f"{res.spec.name}__{res.scheme}.parquet")


def load_oof(ctx, stage: str, name: str, scheme: str = "repeated_kfold") -> pd.DataFrame | None:
    p = ctx.stage_dir(stage) / "oof" / f"{name}__{scheme}.parquet"
    return pd.read_parquet(p) if p.exists() else None
