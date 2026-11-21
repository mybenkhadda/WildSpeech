"""Fold-aware feature construction for predictive evaluation.

Everything fitted on data (common screening factor, IRT item parameters,
item reference statistics, index scaling, clustering) is fitted on the
training students of the fold only and then applied to validation (or test)
students. Imputation, scaling, PCA and supervised selection happen inside
scikit-learn pipelines and are therefore fold-aware as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from .families import FOLD_FITTED_FAMILIES, modeling_columns


class FoldFeatureBuilder:
    def __init__(self, ctx, base: pd.DataFrame | None = None, include_test: bool = False):
        self.ctx = ctx
        self.cfg = ctx.config
        if base is None:
            base = da.student_features(ctx, "train")
            if include_test and ctx.has("student_features_test"):
                base = pd.concat([base, da.student_features(ctx, "test")])
        numeric = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
        self.base = base[numeric].replace([np.inf, -np.inf], np.nan)
        self.include_test = include_test
        self._clips = None
        self._cache: dict = {}

    def clips(self) -> pd.DataFrame:
        if self._clips is None:
            parts = [da.clip_table(self.ctx, "train", with_audio=self.ctx.stage_available("06"))]
            if self.include_test:
                try:
                    parts.append(da.clip_table(self.ctx, "test", with_audio=self.ctx.has("clip_features_test")))
                except Exception:
                    pass
            self._clips = pd.concat(parts, ignore_index=True)
        return self._clips

    def columns_for(self, families) -> list:
        return modeling_columns(self.base.columns, families)

    def _cached(self, name: str, key, fn):
        k = (name, key)
        if key is not None and k in self._cache:
            return self._cache[k]
        value = fn()
        if key is not None:
            self._cache[k] = value
        return value

    def build(self, train_ids, val_ids, families, key=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        families = set(families)
        train_ids = list(train_ids)
        ids = train_ids + [v for v in val_ids if v not in set(train_ids)]
        X = self.base.reindex(ids)[self.columns_for(families)].copy()
        if "SCREENING" in families:
            X = X.join(self._cached("theta_cf", key, lambda: self._theta_factor(train_ids)).reindex(ids))
        if "ITEM_DIFFICULTY" in families:
            irt = self._irt(key, train_ids)
            if not irt.empty:
                X = X.join(irt.reindex(ids))
        if "ITEM_ADJUSTED_AUDIO" in families:
            ia = self._ia(key, train_ids)
            if not ia.empty:
                X = X.join(ia.reindex(ids), rsuffix="_fold")
        if "LITERACY_INDICES" in families:
            from .literacy_indices import compute_indices

            def _idx():
                extra = pd.concat([self._irt(key, train_ids), self._ia(key, train_ids)], axis=1)
                return compute_indices(self.base.reindex(ids), train_ids, self.cfg,
                                       extra=extra.reindex(ids) if not extra.empty else None)[0]
            idx = self._cached("idx", key, _idx)
            X = X.join(idx.reindex(ids), rsuffix="_fold")
        if "CLUSTER" in families:
            from .clustering import fold_cluster_features

            clu = self._cached("clu", key, lambda: fold_cluster_features(self.base.reindex(ids), train_ids, self.cfg))
            X = X.join(clu.reindex(ids), rsuffix="_fold")
        X = X.loc[:, ~X.columns.duplicated()]
        X = X.loc[:, X.loc[train_ids].notna().any()]
        return X.loc[train_ids], X.loc[list(val_ids)]

    def _irt(self, key, train_ids) -> pd.DataFrame:
        if not self.cfg.get("enable_irt"):
            return pd.DataFrame()
        from .irt import fold_irt_features

        clips = self.clips()
        if "correct" not in clips.columns or clips["correct"].notna().sum() == 0:
            return pd.DataFrame()
        return self._cached("irt", key, lambda: fold_irt_features(clips, set(train_ids), self.cfg))

    def _ia(self, key, train_ids) -> pd.DataFrame:
        from .student_aggregation import core_features, item_adjusted_features, numeric_clip_features

        def _compute():
            c = self.clips()
            return item_adjusted_features(c, core_features(numeric_clip_features(c), self.cfg), train_ids,
                                          float(self.cfg["item_shrinkage_k"]))
        return self._cached("ia", key, _compute)

    def _theta_factor(self, train_ids) -> pd.DataFrame:
        from .screening_eda import derive_screening_features

        raw = da.raw_theta_columns(self.base)
        if len(raw) < 2:
            return pd.DataFrame(index=self.base.index)
        _, params = derive_screening_features(self.base.loc[train_ids, raw])
        feats, _ = derive_screening_features(self.base[raw], params)
        return feats[["theta_common_factor"]]


def requires_fold_fitting(families) -> bool:
    return bool(set(families) & (FOLD_FITTED_FAMILIES | {"SCREENING"}))
