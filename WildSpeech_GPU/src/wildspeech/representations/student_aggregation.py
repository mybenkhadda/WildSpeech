"""Student-level aggregation of V3 clip features.

phonetic_family_provider(ctx, fam, wide, cols)   ConditionalAcousticProvider (V2) on P25 clip features:
        item-conditioned z-scores (item mean shrunk to task mean, domain / duration / complexity regression,
        task SD) fitted on training-fold clips -> per student x task mean / sd / correct_mean / incorrect_mean
ConsistencyProvider (ASRS)   cross-item stability of key phonetic measures: within-student SD of item-adjusted z,
        mean absolute successive difference (ordered by item id), share of items within ±1 SD of the student's mean
NestedSoftCorrectnessProvider (ASRN)   P(correct | ASR clip features, task, item difficulty) — SUPERVISED:
        for training students the clip predictions come from inner student-disjoint folds (the student's clips
        never trained the model that scores them); validation students are scored by a model fitted on all
        training-fold students. Student features: mean p, near-correct share among incorrect, discrepancy
        (official - p), difficulty-weighted mastery.
ConfusionEmbeddingProvider (ASRX)   per-student phoneme confusion profile (rate of each ref->hyp substitution /
        deletion among the student's expected phones) -> TruncatedSVD fitted on training-fold students.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import (ConditionalAcousticProvider, FoldFeatureProvider,
                                                aggregate_student_task)
from ..io_utils import stable_hash
from ..v2.validation_registry import inner_student_folds


def phonetic_family_provider(ctx, fam: str, wide: pd.DataFrame, cols: list[str], name: str | None = None,
                             controls=("item", "domain", "duration", "complexity")):
    from ..features.conditional_aggregation import CachedProvider

    cols = [c for c in cols if c in wide.columns and wide[c].notna().mean() > 0.05]
    if not cols:
        return None
    p = ConditionalAcousticProvider(ctx, name or fam, wide, cols, prefix=fam, controls=tuple(controls),
                                    stats=("mean", "sd", "correct_mean", "incorrect_mean"),
                                    shrinkage_k=float(ctx.cfg.get_path("v3.discovery.shrinkage_k", 10.0)),
                                    description=f"V3 {fam}", version="v3-fam-1")
    return CachedProvider(p, root=ctx.paths.v3_cache / "fold_providers")


# ================================================================================ consistency
class ConsistencyProvider(FoldFeatureProvider):
    family = "ASRS"

    def __init__(self, ctx, name: str, wide: pd.DataFrame, cols: list[str]):
        super().__init__(ctx, name, "cross-item phonetic consistency")
        self.cols = [c for c in cols if c in wide.columns]
        self.wide = wide[[FILE] + self.cols].drop_duplicates(FILE)

    def available(self) -> bool:
        return bool(self.cols)

    def fingerprint_payload(self) -> dict:
        return {"cols": self.cols, "files": stable_hash(sorted(self.wide[FILE].astype(str))), "v": "asrs-1"}

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from ..discovery.item_conditioning import ItemReference, v3_clip_frame

        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].isin(ids)].merge(self.wide, on=FILE, how="inner").sort_values([ID, "task", "item_id"]).reset_index(drop=True)
        F = c[self.cols].to_numpy(float)
        tr = c[ID].isin(set(map(str, train_ids))).to_numpy()
        ref = ItemReference(("item", "domain", "duration", "complexity")).fit(c[tr], F[tr])
        Z, _, _ = ref.transform(c, F)
        names = [x.replace("::", "_").lower() for x in self.cols]
        Zd = pd.DataFrame(Z, columns=names)
        Zd[ID], Zd["task"] = c[ID].astype(str).to_numpy(), c["task"].to_numpy()
        rows = []
        for (sid, task), g in Zd.groupby([ID, "task"]):
            r = {ID: sid, "task": task}
            for n in names:
                v = g[n].dropna().to_numpy()
                r[f"{n}__within_sd"] = float(v.std()) if len(v) >= 3 else np.nan
                r[f"{n}__succ_diff"] = float(np.mean(np.abs(np.diff(v)))) if len(v) >= 3 else np.nan
                r[f"{n}__within1sd"] = float(np.mean(np.abs(v - v.mean()) <= 1.0)) if len(v) >= 3 else np.nan
            rows.append(r)
        T = pd.DataFrame(rows)
        if T.empty:
            return pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        W = T.set_index([ID, "task"]).unstack("task")
        W.columns = [f"ASRS__{t}__{f}__value" for f, t in W.columns]
        W = W.loc[:, W.notna().any()]
        W.index = W.index.astype(str)
        W.index.name = ID
        return W.reindex(sorted(ids))


# ============================================================================ nested soft correctness
class NestedSoftCorrectnessProvider(FoldFeatureProvider):
    family = "ASRN"

    def __init__(self, ctx, name: str, wide: pd.DataFrame, cols: list[str], model: str = "LOGISTIC",
                 inner_splits: int = 4):
        super().__init__(ctx, name, "nested ASR soft correctness")
        self.cols = [c for c in cols if c in wide.columns]
        self.wide = wide[[FILE] + self.cols].drop_duplicates(FILE)
        self.model, self.inner = str(model).upper(), int(inner_splits)

    def available(self) -> bool:
        return bool(self.cols)

    def fingerprint_payload(self) -> dict:
        return {"cols": self.cols, "model": self.model, "inner": self.inner,
                "files": stable_hash(sorted(self.wide[FILE].astype(str))), "v": "asrn-1"}

    def _est(self, seed: int):
        from ..discovery.correct_response import ClipCorrectnessIncrement

        return ClipCorrectnessIncrement._estimator(self.model, seed)

    def _design(self, c: pd.DataFrame, ref) -> np.ndarray:
        tasks = sorted(c["task"].unique())
        extra = np.column_stack([ref.difficulty(c), c["log_dur"].to_numpy(float)] +
                                [(c["task"] == t).to_numpy(float) for t in tasks])
        return np.hstack([c[self.cols].to_numpy(float), extra])

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from ..discovery.item_conditioning import ItemReference, v3_clip_frame

        train_ids = sorted(map(str, train_ids))
        ids = set(train_ids) | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].isin(ids) & (c["is_scored"] == 1)].merge(self.wide, on=FILE, how="inner").reset_index(drop=True)
        if c.empty:
            return pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        sid = c[ID].astype(str).to_numpy()
        y = c["item_score"].astype(int).to_numpy()
        p = np.full(len(c), np.nan)
        tr_mask = np.isin(sid, train_ids)
        seed = int(int(stable_hash(train_ids), 16) % (2 ** 31 - 1))
        # (1) training students: inner student-disjoint cross-fitting (item difficulty fitted inside the inner split)
        for in_tr, in_va in inner_student_folds(train_ids, self.inner, seed):
            a, b = np.isin(sid, in_tr), np.isin(sid, in_va)
            if a.sum() < 50 or b.sum() == 0 or len(np.unique(y[a])) < 2:
                continue
            ref = ItemReference(("item",)).fit(c[a])
            est = self._est(seed).fit(self._design(c[a], ref), y[a])
            p[b] = est.predict_proba(self._design(c[b], ref))[:, 1]
        # (2) target (validation / test) students: model fitted on all training-fold students
        out_m = ~tr_mask
        if out_m.any() and len(np.unique(y[tr_mask])) == 2:
            ref = ItemReference(("item",)).fit(c[tr_mask])
            est = self._est(seed).fit(self._design(c[tr_mask], ref), y[tr_mask])
            p[out_m] = est.predict_proba(self._design(c[out_m], ref))[:, 1]
        ref_all = ItemReference(("item",)).fit(c[tr_mask])
        diff = ref_all.difficulty(c)
        V = np.column_stack([p, (y - p), np.where(y == 0, (p >= 0.5).astype(float), np.nan),
                             p * (1 - diff)])
        return aggregate_student_task(c, V, ["p", "discrepancy", "near_correct_incorrect", "difficulty_weighted"],
                                      "ASRN", stats=("mean", "sd")).reindex(sorted(ids))


# ============================================================================== confusion embeddings
class ConfusionEmbeddingProvider(FoldFeatureProvider):
    family = "ASRX"

    def __init__(self, ctx, name: str, details: pd.DataFrame, model_key: str, n_components: int = 8, min_count: int = 30):
        super().__init__(ctx, name, f"phoneme confusion embedding ({model_key})")
        d = details[(details["model_key"] == model_key) & details["ops"].fillna("").ne("")] if not details.empty else details
        self.d = d[[FILE, "ops"]].drop_duplicates(FILE) if not d.empty else pd.DataFrame(columns=[FILE, "ops"])
        self.model_key, self.k, self.min_count = model_key, int(n_components), int(min_count)

    def available(self) -> bool:
        return len(self.d) > 100

    def fingerprint_payload(self) -> dict:
        return {"model": self.model_key, "k": self.k, "min": self.min_count,
                "files": stable_hash(sorted(self.d[FILE].astype(str))), "v": "asrx-1"}

    def _profiles(self, c: pd.DataFrame) -> pd.DataFrame:
        from ..phonetics.alignment import ops_from_string

        rows = []
        for sid, ops in zip(c[ID].astype(str), c["ops"]):
            for o in ops_from_string(ops):
                if o.op in ("S", "D", "M"):
                    rows.append((sid, o.ref, "∅" if o.op == "D" else o.hyp))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=[ID, "ref", "hyp"])
        df = df[df["ref"] != df["hyp"]].assign(pair=lambda x: x["ref"] + ">" + x["hyp"])
        tot = pd.DataFrame(rows, columns=[ID, "ref", "hyp"]).groupby(ID).size()
        cnt = df.groupby([ID, "pair"]).size().unstack(fill_value=0)
        return cnt.div(tot.reindex(cnt.index), axis=0)

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from sklearn.decomposition import TruncatedSVD

        from ..discovery.item_conditioning import v3_clip_frame

        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].isin(ids)].merge(self.d, on=FILE, how="inner")
        P = self._profiles(c)
        if P.empty:
            return pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        trn = [i for i in P.index if i in set(map(str, train_ids))]
        keep = (P.loc[trn] > 0).sum(0)
        keep = keep[keep >= max(3, self.min_count // 10)].index      # pairs observed in enough TRAINING students
        P = P[keep].fillna(0.0)
        k = int(min(self.k, max(1, P.shape[1] - 1), max(1, len(trn) - 1)))
        if P.shape[1] < 2 or len(trn) < 10:
            return pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        mu = P.loc[trn].mean(0)
        svd = TruncatedSVD(n_components=k, random_state=0).fit((P.loc[trn] - mu).to_numpy())
        E = svd.transform((P - mu).to_numpy())
        out = pd.DataFrame(E, index=P.index, columns=[f"ASRX__all__svd{j}__value" for j in range(k)])
        out["ASRX__all__error_mass__value"] = P.sum(1)
        out.index.name = ID
        return out.reindex(sorted(ids))
