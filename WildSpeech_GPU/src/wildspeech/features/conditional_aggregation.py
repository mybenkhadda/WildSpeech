"""Item-conditioned residualization + fold-aware student-level feature providers (V2).

ItemConditionedResidualizer (fitted on TRAINING-FOLD clips only)
    z = (f - mu_item - Z·B) / sd_task
    mu_item  item mean shrunk toward the task mean (n/(n+k)), unseen items -> task mean
    Z        configurable controls: recording domain (low original rate, instrument), log duration,
             linguistic complexity (phonemes, syllables), official correctness (+ unscored indicator)
    B        per-feature ridge regression of the item-centred feature on Z (NaN-aware)
    sd_task  pooled residual SD per task
Controls are chosen per use: student-level aggregation controls correctness; correctness *classifiers* never do.

FoldFeatureProvider protocol
    fit_table(train_ids, target_ids) -> DataFrame indexed by subject_id for every target id, using only clips of
    training students to fit anything. Providers that use labels (clip correctness models, templates) are nested
    so that a training student's row never comes from a model that saw that student.
CachedProvider
    shares one fitted table per (provider fingerprint, training-id set) across schemes, models and stages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..io_utils import atomic_write_parquet, stable_hash
from .builder import sanitize

COMPLEXITY_COLS = ("n_phonemes", "n_syllables", "n_words", "max_cluster", "n_chars")


# ============================================================================== clip frame
def clip_frame(ctx) -> pd.DataFrame:
    """All clips (train + test) with item, domain and complexity covariates (label-free except item_score,
    which is official metadata present in both splits)."""
    key = "v2_clip_frame"
    if key in ctx._cache:
        return ctx._cache[key]
    from ..phonetic.text_features import item_text_table

    b = ctx.bundle()
    c = b.all_clips[[FILE, ID, "task", "item_id", "item_score", "is_scored", "duration_sec", "split",
                     "expected_text", "expected_text_norm"]].drop_duplicates(FILE).reset_index(drop=True)
    st = b.all_students.drop_duplicates(ID).set_index(ID)
    c["instrument"] = c[ID].map(st["instrument"]).fillna("unknown").astype(str) if "instrument" in st else "unknown"
    if "original_sample_rate" in st:
        c["orig_sr_low"] = (c[ID].map(st["original_sample_rate"]).astype(float) < 16000).astype(float)
    else:
        c["orig_sr_low"] = 0.0
    c["log_dur"] = np.log(c["duration_sec"].astype(float).clip(lower=0.05))
    txt = item_text_table(b.all_clips, ctx.cfg.get_path("v2.phonetic.g2p_backend", "auto"))
    keep = ["item_id"] + [k for k in COMPLEXITY_COLS if k in txt.columns]
    c = c.merge(txt[keep], on="item_id", how="left")
    ctx._cache[key] = c
    return c


# ====================================================================== residualizer
class ItemConditionedResidualizer:
    def __init__(self, controls=("item", "domain", "duration", "complexity", "correctness"), shrinkage_k: float = 10.0,
                 ridge: float = 1.0, clip_z: float = 8.0):
        self.controls = tuple(controls)
        self.k, self.ridge, self.clip_z = float(shrinkage_k), float(ridge), float(clip_z)

    # ----------------------------------------------------------------- design
    def _design(self, c: pd.DataFrame, fit: bool) -> np.ndarray:
        cols = []
        if "domain" in self.controls:
            cols.append(c["orig_sr_low"].astype(float).to_numpy())
            if fit:
                self.instruments_ = sorted(c["instrument"].astype(str).unique())[1:]
            for ins in getattr(self, "instruments_", []):
                cols.append((c["instrument"].astype(str) == ins).astype(float).to_numpy())
        if "duration" in self.controls:
            cols.append(c["log_dur"].astype(float).to_numpy())
        if "complexity" in self.controls:
            for k in ("n_phonemes", "n_syllables"):
                if k in c:
                    cols.append(np.log1p(c[k].astype(float).to_numpy()))
        if "correctness" in self.controls:
            s = c["item_score"].astype(float).to_numpy()
            cols.append(np.where(np.isfinite(s), s, 0.0))
            cols.append((~np.isfinite(s)).astype(float))
        if not cols:
            return np.zeros((len(c), 0))
        Z = np.column_stack(cols).astype(float)
        if fit:
            self.z_mean_ = np.nanmean(Z, 0)
            sd = np.nanstd(Z, 0)
            self.z_sd_ = np.where(sd > 1e-9, sd, 1.0)
        Z = (Z - self.z_mean_) / self.z_sd_
        return np.where(np.isfinite(Z), Z, 0.0)

    def _item_mu(self, c: pd.DataFrame) -> np.ndarray:
        if "item" not in self.controls:
            return np.broadcast_to(self.global_mean_, (len(c), len(self.global_mean_)))
        mu = self.item_mu_.reindex(c["item_id"].to_numpy()).to_numpy(float)
        mt = self.task_mu_.reindex(c["task"].to_numpy()).to_numpy(float)
        mu = np.where(np.isfinite(mu), mu, mt)
        return np.where(np.isfinite(mu), mu, self.global_mean_[None, :])

    # -------------------------------------------------------------------- fit
    def fit(self, c: pd.DataFrame, F: np.ndarray) -> "ItemConditionedResidualizer":
        F = np.asarray(F, float)
        self.global_mean_ = np.nan_to_num(np.nanmean(F, 0)) if len(F) else np.zeros(F.shape[1])
        df = pd.DataFrame(F)
        if "item" in self.controls:
            gi = df.groupby(c["item_id"].to_numpy())
            gt = df.groupby(c["task"].to_numpy())
            m_i, n_i, m_t = gi.mean(), gi.count(), gt.mean()
            task_of = c.drop_duplicates("item_id").set_index("item_id")["task"].reindex(m_i.index)
            mt_i = m_t.reindex(task_of.to_numpy()).set_index(m_i.index)
            w = n_i / (n_i + self.k)
            self.item_mu_ = w * m_i.fillna(mt_i) + (1 - w) * mt_i
            self.task_mu_ = m_t
        R = F - self._item_mu(c)
        Z = self._design(c, fit=True)
        q = Z.shape[1]
        self.B_ = np.zeros((q + 1, F.shape[1]))
        Zi = np.column_stack([np.ones(len(Z)), Z])
        lam = self.ridge * np.eye(q + 1)
        lam[0, 0] = 0.0
        for j in range(F.shape[1]):
            m = np.isfinite(R[:, j])
            if m.sum() < q + 5:
                continue
            A = Zi[m]
            self.B_[:, j] = np.linalg.solve(A.T @ A + lam, A.T @ R[m, j])
        E = R - Zi @ self.B_
        e = pd.DataFrame(E).groupby(c["task"].to_numpy()).std(ddof=0)
        glob = np.nanstd(E, 0)
        self.task_sd_ = e.where(e > 1e-9)
        self.global_sd_ = np.where(glob > 1e-9, glob, 1.0)
        return self

    def transform(self, c: pd.DataFrame, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, float)
        R = F - self._item_mu(c)
        Zi = np.column_stack([np.ones(len(c)), self._design(c, fit=False)])
        E = R - Zi @ self.B_
        sd = self.task_sd_.reindex(c["task"].to_numpy()).to_numpy(float)
        sd = np.where(np.isfinite(sd), sd, self.global_sd_[None, :])
        Zs = E / sd
        return np.where(np.isfinite(Zs), np.clip(Zs, -self.clip_z, self.clip_z), np.nan)


# ======================================================================= aggregation
def aggregate_student_task(c: pd.DataFrame, V: np.ndarray, names: list[str], prefix: str,
                           stats=("mean", "sd", "correct_mean", "incorrect_mean"), include_all: bool = True) -> pd.DataFrame:
    """Clip-level values -> one row per student: PREFIX__task__feature__stat (+ PREFIX__all__feature__mean)."""
    names = [sanitize(n) for n in names]
    df = pd.DataFrame(np.asarray(V, float), columns=names, index=c.index)
    df[ID] = c[ID].astype(str).to_numpy()
    df["task"] = c["task"].to_numpy()
    s = c["item_score"].astype(float).to_numpy()
    blocks = []

    def wide(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        w = frame.unstack("task")
        w.columns = [f"{prefix}__{t}__{f}__{stat}" for f, t in w.columns]
        return w

    g = df.groupby([ID, "task"])[names]
    if "mean" in stats:
        blocks.append(wide(g.mean(), "mean"))
    if "sd" in stats:
        blocks.append(wide(g.std(ddof=0).where(g.count() >= 2), "sd"))
    if "median" in stats:
        blocks.append(wide(g.median(), "median"))
    if "correct_mean" in stats:
        blocks.append(wide(df[s == 1].groupby([ID, "task"])[names].mean(), "correct_mean"))
    if "incorrect_mean" in stats:
        blocks.append(wide(df[s == 0].groupby([ID, "task"])[names].mean(), "incorrect_mean"))
    if include_all:
        a = df.groupby(ID)[names].mean()
        a.columns = [f"{prefix}__all__{f}__mean" for f in a.columns]
        blocks.append(a)
    blocks = [b for b in blocks if b is not None and not b.empty]
    out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=pd.Index([], name=ID))
    out.index = out.index.astype(str)
    out.index.name = ID
    return out.replace([np.inf, -np.inf], np.nan)


# ========================================================================== providers
class FoldFeatureProvider:
    """Base class. Subclasses implement fit_table and fingerprint_payload."""

    family = "V2"

    def __init__(self, ctx, name: str, description: str = ""):
        self.ctx, self.name, self.description = ctx, name, description

    def available(self) -> bool:
        return True

    def fingerprint_payload(self) -> dict:
        return {}

    def fingerprint(self) -> str:
        return stable_hash({"provider": type(self).__name__, "name": self.name, "family": self.family,
                            **self.fingerprint_payload()})

    def fit_table(self, train_ids: list[str], target_ids: list[str]) -> pd.DataFrame:
        raise NotImplementedError


class CachedProvider:
    """Memory + Parquet cache of fitted tables keyed by (provider fingerprint, sorted training ids)."""

    def __init__(self, provider: FoldFeatureProvider, root=None):
        self.p = provider
        self.root = (root or provider.ctx.paths.v2_cache / "fold_providers") / sanitize(provider.name)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, pd.DataFrame] = {}
        self._fp: str | None = None

    def fingerprint(self) -> str:
        if self._fp is None:                     # providers are immutable once built -> compute once
            self._fp = self.p.fingerprint()
        return self._fp

    @property
    def name(self) -> str:
        return self.p.name

    @property
    def family(self) -> str:
        return self.p.family

    def available(self) -> bool:
        return self.p.available()

    def table(self, train_ids, target_ids) -> pd.DataFrame:
        train_ids = sorted(map(str, train_ids))
        target_ids = [str(i) for i in target_ids]
        key = stable_hash({"fp": self.fingerprint(), "tr": train_ids})
        if key in self._mem and set(target_ids) <= set(self._mem[key].index):
            return self._mem[key].reindex(target_ids)
        path = self.root / f"{key}.parquet"
        T = None
        if path.exists():
            T = pd.read_parquet(path)
            T.index = T.index.astype(str)
            if not set(target_ids) <= set(T.index):
                T = None
        if T is None:
            want = sorted(set(train_ids) | set(target_ids))
            T = self.p.fit_table(train_ids, want)
            T.index = T.index.astype(str)
            T.index.name = ID
            atomic_write_parquet(T, path, index=True)
        if len(self._mem) > 48:
            self._mem.pop(next(iter(self._mem)))
        self._mem[key] = T
        return T.reindex(target_ids)


class ConditionalAcousticProvider(FoldFeatureProvider):
    """Item-conditioned residualized clip features aggregated per student x task (no labels used)."""

    family = "HARM"

    def __init__(self, ctx, name: str, clip_features: pd.DataFrame, feature_cols: list[str], prefix: str = "HARM",
                 controls=("item", "domain", "duration", "complexity", "correctness"),
                 stats=("mean", "sd", "correct_mean", "incorrect_mean"), shrinkage_k: float = 10.0,
                 description: str = "", version: str = ""):
        super().__init__(ctx, name, description)
        self.family = prefix
        self.cf = clip_features[[FILE] + [c for c in feature_cols if c in clip_features.columns]].drop_duplicates(FILE)
        self.cols = [c for c in feature_cols if c in self.cf.columns]
        self.controls, self.stats, self.k, self.version = tuple(controls), tuple(stats), float(shrinkage_k), version
        cfr = clip_frame(ctx)
        # rows of TRAINING students never depend on test clips (everything is fitted on training clips), so the
        # cache identity uses the training clips only: adding test features later keeps CV caches valid.
        self._train_files = sorted(set(self.cf[FILE].astype(str)) & set(cfr.loc[cfr["split"] == "train", FILE].astype(str)))

    def available(self) -> bool:
        return bool(self.cols) and len(self.cf) > 0

    def fingerprint_payload(self) -> dict:
        return {"cols": self.cols, "controls": self.controls, "stats": self.stats, "k": self.k, "v": self.version,
                "n_train_clips": len(self._train_files), "files_hash": stable_hash(self._train_files)}

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        c = clip_frame(self.ctx)
        ids = set(target_ids) | set(train_ids)
        c = c[c[ID].isin(ids)].merge(self.cf, on=FILE, how="left").reset_index(drop=True)
        F = c[self.cols].to_numpy(float)
        tr = c[ID].isin(set(train_ids)).to_numpy()
        res = ItemConditionedResidualizer(self.controls, self.k).fit(c[tr], F[tr])
        Z = res.transform(c, F)
        names = [col.replace("::", "_").lower() for col in self.cols]
        out = aggregate_student_task(c, Z, names, self.family, self.stats)
        return out.reindex(sorted(ids))
