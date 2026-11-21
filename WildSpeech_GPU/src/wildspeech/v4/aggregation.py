"""Fold-fitted clip -> student aggregation for V4 families (FoldFeatureProvider protocol of V2/V3).

V4ClipAggregateProvider.fit_table(train_ids, target_ids)
  1. clip frame of the requested students (v3_clip_frame: item, task, domain, log duration, complexity, item_score,
     screening ability) left-joined to the clip feature table -> clips without features stay (coverage)
  2. ItemConditionedResidualizer fitted on TRAINING-FOLD clips only (controls: item / domain / duration /
     complexity): item-, recording-domain- and length-conditioned z-scores — the existing harmonization step
  3. ItemReference fitted on training-fold clips -> item difficulty; 'hard' = below the task median p(correct)
  4. per student x task (+ 'all'):
       mean, median, sd (consistency), p10, p90       robust location / spread
       correct_mean                                   officially correct responses only
       hard_mean                                      item-difficulty-conditioned (hard items)
       wmean                                          reliability-weighted mean (weights = clip ASR reliability)
       coverage, n_clips                              share of the student's clips with features (missingness)
     cross-task contrasts (task mean differences) of the first `contrast_n` features
  Statistics with fewer than `min_clips` finite clips are NaN (never 0). A hard cap on raw columns keeps the
  family small relative to ~541 independent students; the fold reducer (V3FoldModel) then compresses to the
  budget (PCA / PLS / selection fitted on the training fold).
Nothing here sees the outcome; item_score is official metadata present in both splits.

ReliabilityProvider (ASRREL): label-free per-student ASR coverage / agreement / confidence summaries used as the
gate inputs of the reliability-gated mixture of experts (M7) and as a missingness description.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.builder import sanitize
from ..features.conditional_aggregation import CachedProvider, FoldFeatureProvider, ItemConditionedResidualizer
from ..io_utils import stable_hash

STAT_ORDER = ("mean", "coverage", "sd", "correct_mean", "hard_mean", "wmean", "median", "p10", "p90")


def _short(col: str) -> str:
    return sanitize(col.replace("::", "_").lower())


class V4ClipAggregateProvider(FoldFeatureProvider):
    def __init__(self, ctx, name: str, family: str, clip_table: pd.DataFrame, cols: list[str],
                 controls=("item", "domain", "duration", "complexity"), stats=STAT_ORDER, shrinkage_k: float = 10.0,
                 reliability: pd.Series | None = None, correct_only: bool = False, tasks: list[str] | None = None,
                 contrast_n: int = 3, min_clips: int = 2, max_columns: int = 400, version: str = "",
                 description: str = ""):
        super().__init__(ctx, name, description)
        self.family = family
        self.cols = [c for c in cols if c in clip_table.columns]
        self.cf = clip_table[[FILE] + self.cols].drop_duplicates(FILE).reset_index(drop=True)
        self.controls, self.k = tuple(controls), float(shrinkage_k)
        self.stats = tuple(s for s in STAT_ORDER if s in set(stats))
        self.rel = reliability.astype(float).clip(0.0, 1.0) if reliability is not None else None
        self.correct_only, self.tasks = bool(correct_only), (sorted(tasks) if tasks else None)
        self.contrast_n, self.min_clips, self.max_columns = int(contrast_n), int(min_clips), int(max_columns)
        self.version = version
        from ..discovery.item_conditioning import v3_clip_frame

        cfr = v3_clip_frame(ctx)
        tr_files = set(cfr.loc[cfr["split"] == "train", FILE].astype(str))
        has = self.cf[self.cols].notna().any(axis=1) if self.cols else pd.Series(False, index=self.cf.index)
        self._train_files = sorted(set(self.cf.loc[has, FILE].astype(str)) & tr_files)

    def available(self) -> bool:
        return bool(self.cols) and len(self._train_files) > 0

    def fingerprint_payload(self) -> dict:
        # training clips only: adding test features later keeps every CV cache valid (training rows never use them)
        return {"cols": self.cols, "controls": self.controls, "stats": self.stats, "k": self.k, "tasks": self.tasks,
                "correct_only": self.correct_only, "contrast_n": self.contrast_n, "min_clips": self.min_clips,
                "max_columns": self.max_columns, "rel": self.rel is not None, "v": self.version,
                "files": stable_hash(self._train_files), "n": len(self._train_files)}

    # ------------------------------------------------------------------ fit
    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from ..discovery.item_conditioning import ItemReference, v3_clip_frame

        train_ids = set(map(str, train_ids))
        ids = train_ids | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].astype(str).isin(ids)]
        if self.tasks:
            c = c[c["task"].isin(self.tasks)]
        c = c.merge(self.cf, on=FILE, how="left").reset_index(drop=True)
        empty = pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        if c.empty:
            return empty
        F = c[self.cols].to_numpy(float)
        tr = c[ID].astype(str).isin(train_ids).to_numpy()
        has = np.isfinite(F).any(axis=1)
        if not (tr & has).any():
            return empty
        res = ItemConditionedResidualizer(self.controls, self.k).fit(c[tr & has], F[tr & has])
        Z = res.transform(c, F)
        ref = ItemReference(("item",), self.k).fit(c[tr])
        diff = ref.difficulty(c)                                   # shrunk p(correct): lower = harder
        tmed = pd.Series(diff[tr]).groupby(c.loc[tr, "task"].to_numpy()).median()
        hard = diff < c["task"].map(tmed).fillna(np.nanmedian(diff)).to_numpy()
        w = c[FILE].map(self.rel).fillna(1.0).to_numpy(float) if self.rel is not None else np.ones(len(c))
        score = c["item_score"].astype(float).to_numpy()
        keep = (score == 1.0) if self.correct_only else np.ones(len(c), bool)
        names = [_short(x) for x in self.cols]
        df = pd.DataFrame(np.where(keep[:, None], Z, np.nan), columns=names)
        df[ID] = c[ID].astype(str).to_numpy()
        df["task"] = c["task"].to_numpy()
        out = self._aggregate(df, names, has & keep, hard, score, w)
        out.index = out.index.astype(str)
        out.index.name = ID
        return out.reindex(sorted(ids)).replace([np.inf, -np.inf], np.nan)

    # -------------------------------------------------------------- aggregate
    def _aggregate(self, df: pd.DataFrame, names: list[str], has: np.ndarray, hard: np.ndarray, score: np.ndarray,
                   w: np.ndarray) -> pd.DataFrame:
        fam = self.family
        keys = [ID, "task"]
        g = df.groupby(keys)[names]
        cnt = g.count()
        ok = cnt >= self.min_clips
        blocks: dict[str, pd.DataFrame] = {}

        def put(stat: str, frame: pd.DataFrame, scope_task: bool = True):
            if frame is None or frame.empty:
                return
            if scope_task:
                wv = frame.unstack("task")
                wv.columns = [f"{fam}__{t}__{f}__{stat}" for f, t in wv.columns]
            else:
                wv = frame.copy()
                wv.columns = [f"{fam}__all__{f}__{stat}" for f in wv.columns]
            blocks[stat] = pd.concat([blocks[stat], wv], axis=1) if stat in blocks else wv

        if "mean" in self.stats:
            put("mean", g.mean().where(ok))
            ga = df.groupby(ID)[names]
            put("mean", ga.mean().where(ga.count() >= self.min_clips), scope_task=False)
        if "median" in self.stats:
            put("median", g.median().where(ok))
        if "sd" in self.stats:
            put("sd", g.std(ddof=0).where(cnt >= max(self.min_clips, 2)))
        if "p10" in self.stats:
            put("p10", g.quantile(0.1).where(ok))
        if "p90" in self.stats:
            put("p90", g.quantile(0.9).where(ok))
        if "correct_mean" in self.stats and not self.correct_only:
            d = df[score == 1.0]
            gc = d.groupby(keys)[names]
            put("correct_mean", gc.mean().where(gc.count() >= self.min_clips))
        if "hard_mean" in self.stats:
            d = df[hard]
            gh = d.groupby(keys)[names]
            put("hard_mean", gh.mean().where(gh.count() >= self.min_clips))
        if "wmean" in self.stats:
            V = df[names].to_numpy(float)
            W = np.where(np.isfinite(V), w[:, None], 0.0)
            num = pd.DataFrame(np.nan_to_num(V) * W, columns=names).assign(**{ID: df[ID].values, "task": df["task"].values})
            den = pd.DataFrame(W, columns=names).assign(**{ID: df[ID].values, "task": df["task"].values})
            sn, sd_ = num.groupby(keys)[names].sum(), den.groupby(keys)[names].sum()
            put("wmean", (sn / sd_.where(sd_ > 1e-9)).where(ok))
        if "coverage" in self.stats:
            cov = pd.DataFrame({ID: df[ID].values, "task": df["task"].values, "coverage": has.astype(float), "n_clips": 1.0})
            gcov = cov.groupby(keys).agg(coverage=("coverage", "mean"), n_clips=("n_clips", "sum"))
            wv = gcov.unstack("task")
            wv.columns = [f"{fam}__{t}__{f}__value" for f, t in wv.columns]
            ca = cov.groupby(ID)["coverage"].mean().rename(f"{fam}__all__coverage__value").to_frame()
            blocks["coverage"] = pd.concat([wv, ca], axis=1)
        # cross-task contrasts of the leading features
        if "mean" in blocks and self.contrast_n > 0:
            tasks = sorted(df["task"].unique())
            m = g.mean().where(ok)
            con = {}
            for f in names[: self.contrast_n]:
                if f not in m:
                    continue
                piv = m[f].unstack("task")
                for a, b in itertools.combinations(tasks, 2):
                    if a in piv and b in piv:
                        con[f"{fam}__all__{f}_{a}_minus_{b}__contrast"] = piv[a] - piv[b]
            if con:
                blocks["contrast"] = pd.DataFrame(con)
        ordered = [blocks[s] for s in (*STAT_ORDER, "contrast") if s in blocks and not blocks[s].empty]
        if not ordered:
            return pd.DataFrame(index=pd.Index([], name=ID))
        out = pd.concat(ordered, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        out = out.loc[:, out.notna().any()]
        if out.shape[1] > self.max_columns:
            out = out.iloc[:, : self.max_columns]
        return out


class ReliabilityProvider(FoldFeatureProvider):
    """Per-student ASR reliability (label-free, nothing fitted): coverage, agreement, confidence by task."""

    family = "ASRREL"

    def __init__(self, ctx, name: str, rel_table: pd.DataFrame, version: str = ""):
        super().__init__(ctx, name, "ASR reliability / coverage (gate inputs)")
        self.cols = [c for c in ("rel_frac_ok", "rel_any_ok", "rel_agree", "rel_conf") if c in rel_table.columns]
        self.t = rel_table[[FILE] + self.cols].drop_duplicates(FILE)
        self.version = version

    def available(self) -> bool:
        return bool(self.cols) and len(self.t) > 0

    def fingerprint_payload(self) -> dict:
        return {"cols": self.cols, "v": self.version, "n": len(self.t)}

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from ..discovery.item_conditioning import v3_clip_frame

        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].astype(str).isin(ids)][[FILE, ID, "task"]].merge(self.t, on=FILE, how="left")
        for col in ("rel_frac_ok", "rel_any_ok"):
            if col in c:
                c[col] = c[col].fillna(0.0)                       # no prediction = not reliable (explicit)
        g = c.groupby([ID, "task"])[self.cols].mean().unstack("task")
        g.columns = [f"{self.family}__{t}__{f}__mean" for f, t in g.columns]
        a = c.groupby(ID)[self.cols].mean()
        a.columns = [f"{self.family}__all__{f}__mean" for f in a.columns]
        out = pd.concat([g, a], axis=1)
        out.index = out.index.astype(str)
        out.index.name = ID
        return out.reindex(sorted(ids))


def cached(ctx, provider: FoldFeatureProvider) -> CachedProvider:
    return CachedProvider(provider, root=ctx.paths.cache / "v4" / "fold_providers")
