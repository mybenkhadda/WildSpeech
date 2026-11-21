"""Fold-fitted compression of clip embeddings and layer-wise discovery.

linear_cka           representation similarity between layers / models (centred linear CKA)
LayerEvaluator       per (model, layer, pooling) on a fixed clip sample, all GroupKFold by student:
                       correct_auc   correctness decodability (PCA -> logistic, + task one-hot)
                       content_acc   phonetic-content preservation: kNN (cosine) item-identity accuracy among
                                     correct responses within task, vs chance
                       sr_auc        recording-domain leakage (low original rate) — lower is better
                       pls_auc       supervised PLS (2 comps, fitted in fold) correctness AUROC
FoundationProvider   student-level compact features for one (model, layer, pooling):
                       FM__task__pc{j}__mean       per-student task means of fold-fitted PCA components
                       FM__task__cent_sim__*       cosine to fold-fitted item centroids of correct responses
                                                   (nested: training students use inner-fold centroids)
                       FM__task__bottleneck__*     nested cross-fitted P(correct | embedding PCs, task) summaries
Never concatenates raw embedding statistics into the student-level model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..diagnostics.scientific_validation import auc_fast
from ..features.conditional_aggregation import FoldFeatureProvider, aggregate_student_task, clip_frame
from ..io_utils import stable_hash
from ..v2.validation_registry import inner_student_folds


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / den) if den > 0 else np.nan


class FoldPCA:
    def __init__(self, k: int, seed: int = 0):
        self.k, self.seed = int(k), seed

    def fit(self, X: np.ndarray) -> "FoldPCA":
        from sklearn.decomposition import PCA

        ok = np.isfinite(X).all(1)
        self.mu_, sd = X[ok].mean(0), X[ok].std(0)
        self.sd_ = np.where(sd > 1e-6, sd, 1.0)
        self.pca_ = PCA(n_components=int(min(self.k, ok.sum() - 1, X.shape[1])), random_state=self.seed).fit((X[ok] - self.mu_) / self.sd_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        ok = np.isfinite(X).all(1)
        out = np.full((len(X), self.pca_.n_components_), np.nan)
        if ok.any():
            out[ok] = self.pca_.transform((X[ok] - self.mu_) / self.sd_)
        return out


def _gkf(groups, n_splits=5, seed=0):
    from ..correctness.crossfit import student_groups_kfold

    return student_groups_kfold(np.asarray(groups).astype(str), n_splits, seed)


class LayerEvaluator:
    def __init__(self, ctx, n_pca: int = 64, knn_k: int = 5, seed: int = 0):
        self.ctx, self.k, self.knn, self.seed = ctx, int(n_pca), int(knn_k), seed

    def evaluate(self, c: pd.DataFrame, X: np.ndarray) -> dict:
        """c: clip rows aligned with X (N, D)."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import LogisticRegression

        y = c["item_score"].to_numpy(float)
        sr = c["orig_sr_low"].to_numpy(float)
        tasks = sorted(c["task"].unique())
        T = np.column_stack([(c["task"] == t).astype(float).to_numpy() for t in tasks])
        p_cor, p_sr, p_pls = np.full(len(c), np.nan), np.full(len(c), np.nan), np.full(len(c), np.nan)
        hits, tot = 0, 0
        for tr, va in _gkf(c[ID].to_numpy(), 5, self.seed):
            pca = FoldPCA(self.k, self.seed).fit(X[tr])
            Ztr, Zva = np.nan_to_num(pca.transform(X[tr])), np.nan_to_num(pca.transform(X[va]))
            mtr = np.isfinite(y[tr])
            if len(np.unique(y[tr][mtr])) == 2:
                m = LogisticRegression(C=0.5, max_iter=3000).fit(np.column_stack([Ztr, T[tr]])[mtr], y[tr][mtr].astype(int))
                p_cor[va] = m.predict_proba(np.column_stack([Zva, T[va]]))[:, 1]
                pls = PLSRegression(n_components=2, scale=False).fit(Ztr[mtr], y[tr][mtr])
                p_pls[va] = np.asarray(pls.predict(Zva)).ravel()
            if len(np.unique(sr[tr])) == 2:
                ms = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(Ztr, sr[tr].astype(int))
                p_sr[va] = ms.predict_proba(Zva)[:, 1]
            # content: kNN item identity among correct responses within task
            for t in tasks:
                a = tr[(c["task"].to_numpy()[tr] == t) & (y[tr] == 1)]
                b = va[(c["task"].to_numpy()[va] == t) & (y[va] == 1)]
                if len(a) < self.knn + 1 or len(b) == 0:
                    continue
                A = Ztr[np.searchsorted(tr, a)]
                Bm = Zva[np.searchsorted(va, b)]
                A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
                Bm = Bm / (np.linalg.norm(Bm, axis=1, keepdims=True) + 1e-9)
                S = Bm @ A.T
                nn = np.argsort(-S, axis=1)[:, : self.knn]
                lab_a = c["item_id"].to_numpy()[a]
                lab_b = c["item_id"].to_numpy()[b]
                for i in range(len(b)):
                    vals, cnt = np.unique(lab_a[nn[i]], return_counts=True)
                    hits += int(vals[np.argmax(cnt)] == lab_b[i])
                    tot += 1
        cc = c[c["item_score"] == 1]
        n_items = cc.groupby("task")["item_id"].nunique()
        chance = float((cc["task"].map(1.0 / n_items)).mean()) if len(cc) else np.nan
        return {"correct_auc": auc_fast(y, p_cor), "pls_auc": auc_fast(y, p_pls), "sr_auc": auc_fast(sr, p_sr),
                "content_acc": hits / tot if tot else np.nan, "content_chance": chance, "n_clips": int(len(c))}


class FoundationProvider(FoldFeatureProvider):
    family = "FM"

    def __init__(self, ctx, name: str, emb: pd.DataFrame, n_pcs: int = 8, bottleneck_pca: int = 32,
                 inner_splits: int | None = None, store_fp: str = "", prefix: str = "FM"):
        super().__init__(ctx, name, "compact foundation-model representation")
        self.family = prefix
        self.emb = emb.drop_duplicates(FILE).reset_index(drop=True)
        self.cols = [c for c in self.emb.columns if c != FILE]
        self.k, self.kb = int(n_pcs), int(bottleneck_pca)
        self.inner = int(inner_splits or ctx.cfg.get_path("v2.validation.inner_splits", 4))
        self.store_fp = store_fp
        cfr = clip_frame(ctx)
        self._n_train = int(self.emb[FILE].isin(set(cfr.loc[cfr["split"] == "train", FILE])).sum())

    def available(self) -> bool:
        return len(self.emb) > 0 and bool(self.cols)

    def fingerprint_payload(self) -> dict:
        return {"store": self.store_fp, "k": self.k, "kb": self.kb, "inner": self.inner, "n_train": self._n_train}

    def _centroid_sim(self, c, Z, fit_rows, pred_rows) -> np.ndarray:
        y = c["item_score"].to_numpy(float)
        items = c["item_id"].to_numpy()
        sel = fit_rows[(y[fit_rows] == 1) & np.isfinite(Z[fit_rows]).all(1)]
        cent = pd.DataFrame(Z[sel]).groupby(items[sel]).mean()
        out = np.full(len(pred_rows), np.nan)
        C = cent.reindex(items[pred_rows]).to_numpy(float)
        V = Z[pred_rows]
        num = (C * V).sum(1)
        den = np.linalg.norm(C, axis=1) * np.linalg.norm(V, axis=1)
        ok = np.isfinite(num) & (den > 0)
        out[ok] = num[ok] / den[ok]
        return out

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from sklearn.linear_model import LogisticRegression

        c = clip_frame(self.ctx)
        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = c[c[ID].isin(ids)].merge(self.emb, on=FILE, how="inner").reset_index(drop=True)
        X = c[self.cols].to_numpy(np.float32)
        sid = c[ID].astype(str).to_numpy()
        is_tr = np.isin(sid, list(map(str, train_ids)))
        y = c["item_score"].to_numpy(float)
        pca = FoldPCA(max(self.k, self.kb), self.ctx.seed).fit(X[is_tr])
        Z = pca.transform(X)
        tasks = sorted(c["task"].unique())
        T = np.column_stack([(c["task"] == t).astype(float).to_numpy() for t in tasks])
        sim = np.full(len(c), np.nan)
        bott = np.full(len(c), np.nan)
        seed = int(int(stable_hash(sorted(map(str, train_ids))), 16) % (2 ** 31 - 1))
        tr_ids = sorted(set(sid[is_tr]))

        def fit_pred(fit_rows, pred_rows):
            sim[pred_rows] = self._centroid_sim(c, Z, fit_rows, pred_rows)
            fr = fit_rows[np.isfinite(y[fit_rows]) & np.isfinite(Z[fit_rows]).all(1)]
            if len(np.unique(y[fr])) == 2:
                Zb = np.nan_to_num(Z[:, : self.kb])
                m = LogisticRegression(C=0.5, max_iter=3000).fit(np.column_stack([Zb[fr], T[fr]]), y[fr].astype(int))
                bott[pred_rows] = m.predict_proba(np.column_stack([Zb[pred_rows], T[pred_rows]]))[:, 1]

        for in_tr, in_va in inner_student_folds(tr_ids, self.inner, seed):
            fit_pred(np.flatnonzero(np.isin(sid, in_tr)), np.flatnonzero(np.isin(sid, in_va)))
        outer = np.flatnonzero(~is_tr)
        if len(outer):
            fit_pred(np.flatnonzero(is_tr), outer)
        pcs = Z[:, : self.k]
        names = [f"pc{j}" for j in range(pcs.shape[1])]
        a = aggregate_student_task(c, pcs, names, self.family, ("mean",), include_all=False)
        extra = np.column_stack([sim, bott])
        b = aggregate_student_task(c, extra, ["cent_sim", "bottleneck"], self.family, ("mean", "correct_mean", "incorrect_mean"))
        return a.join(b, how="outer").reindex(sorted(ids))
