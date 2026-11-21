"""Clip-level correctness classifiers and their fold-fitted input pipeline.

ClipFeatureBank    one wide clip table (train + test where cached) with blocks:
                   TEMPORAL / SPECTRAL / QUALITY / WAVELET_PITCH (V1 caches), HARMONIZED (P16 selected variant),
                   PHONETIC (P18), EMBEDDING (P20 selected layer). Missing blocks are simply unavailable.
FEATURE_SETS       which blocks each experiment uses. Every set contains META (task + fold-fitted item
                   difficulty), so "audio-specific information" = gain of a set over META.
ItemEncoder        shrunk item p-correct fitted on the classifier's training clips; leave-one-student-out for the
                   training rows themselves (target-encoding leakage guard); logit used as a feature.
ClipPipeline       ItemEncoder + per-block item/domain residualization (never conditioned on correctness)
                   + optional fold-fitted PCA (embeddings) + median imputation + scaling + classifier.
make_classifier    LOGISTIC | LIGHTGBM | XGBOOST | CATBOOST | MLP (torch, student-disjoint early stopping).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import COMPLEXITY_COLS, ItemConditionedResidualizer, clip_frame
from ..models.zoo import ModelUnavailable

FEATURE_SETS: dict[str, dict] = {
    "META": {"blocks": [], "complexity": False, "duration": False},
    "LINGUISTIC": {"blocks": [], "complexity": True, "duration": False},
    "TEMPORAL": {"blocks": ["TEMPORAL"], "complexity": False, "duration": True},
    "SPECTRAL": {"blocks": ["SPECTRAL"], "complexity": False, "duration": False},
    "WAVELET_PITCH": {"blocks": ["WAVELET_PITCH"], "complexity": False, "duration": False},
    "HARMONIZED": {"blocks": ["HARMONIZED"], "complexity": False, "duration": True},
    "COMBINED": {"blocks": ["TEMPORAL", "SPECTRAL", "QUALITY", "WAVELET_PITCH", "HARMONIZED"], "complexity": True, "duration": True},
    "PHONETIC": {"blocks": ["PHONETIC"], "complexity": True, "duration": True},
    "EMBEDDING": {"blocks": ["EMBEDDING"], "complexity": False, "duration": True},
    "FULL": {"blocks": ["TEMPORAL", "SPECTRAL", "QUALITY", "WAVELET_PITCH", "HARMONIZED", "PHONETIC", "EMBEDDING"],
             "complexity": True, "duration": True},
}
RESIDUALIZE_BLOCKS = ("TEMPORAL", "SPECTRAL", "QUALITY", "WAVELET_PITCH", "HARMONIZED")
PCA_BLOCKS = ("EMBEDDING",)


def logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ======================================================================== feature bank
class ClipFeatureBank:
    """Wide clip table aligned to clip_frame(ctx) rows; columns '<BLOCK>|<feature>'."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.c = clip_frame(ctx)
        self.blocks: dict[str, list[str]] = {}
        self.table = self.c[[FILE]].copy()
        self.sources: dict[str, str] = {}

    def add_block(self, block: str, df: pd.DataFrame, source: str) -> None:
        if df is None or df.empty:
            return
        cols = [c for c in df.columns if c != FILE and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            return
        ren = {c: f"{block}|{c}" for c in cols}
        self.table = self.table.merge(df[[FILE] + cols].drop_duplicates(FILE).rename(columns=ren), on=FILE, how="left")
        self.blocks[block] = self.blocks.get(block, []) + list(ren.values())
        self.sources[block] = source

    def has(self, block: str) -> bool:
        return bool(self.blocks.get(block))

    def set_available(self, name: str) -> bool:
        spec = FEATURE_SETS[name]
        need = [b for b in spec["blocks"] if b in ("PHONETIC", "EMBEDDING", "HARMONIZED")]
        return all(self.has(b) for b in need) and (not spec["blocks"] or any(self.has(b) for b in spec["blocks"]))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"block": b, "n_features": len(c), "source": self.sources.get(b, ""),
                              "coverage": float(self.table[c].notna().any(axis=1).mean())} for b, c in self.blocks.items()])

    def fingerprint(self, blocks: list[str]) -> dict:
        from ..io_utils import stable_hash

        tr = (self.c["split"] == "train").to_numpy()        # cache identity from training clips only
        return {b: stable_hash({"cols": self.blocks.get(b, []), "src": self.sources.get(b, ""),
                                "n": int(self.table.loc[tr, self.blocks[b]].notna().any(axis=1).sum()) if b in self.blocks else 0})
                for b in blocks}


# ========================================================================= item encoder
class ItemEncoder:
    def __init__(self, k: float = 10.0):
        self.k = float(k)

    def fit(self, c: pd.DataFrame) -> "ItemEncoder":
        s = c[c["item_score"].notna()]
        self.task_p_ = s.groupby("task")["item_score"].mean()
        g = s.groupby("item_id")["item_score"]
        self.sum_, self.n_ = g.sum(), g.count()
        own = s.groupby([ID, "item_id"])["item_score"].agg(["sum", "count"])
        self.own_ = own
        self.train_ids_ = set(s[ID].astype(str))
        return self

    def p_item(self, c: pd.DataFrame, loso: bool) -> np.ndarray:
        tp = c["task"].map(self.task_p_).fillna(float(self.task_p_.mean()) if len(self.task_p_) else 0.5).to_numpy(float)
        S = c["item_id"].map(self.sum_).fillna(0.0).to_numpy(float)
        N = c["item_id"].map(self.n_).fillna(0.0).to_numpy(float)
        if loso:
            key = pd.MultiIndex.from_arrays([c[ID].astype(str), c["item_id"]])
            os_ = self.own_["sum"].reindex(key).fillna(0.0).to_numpy(float)
            on_ = self.own_["count"].reindex(key).fillna(0.0).to_numpy(float)
            S, N = S - os_, N - on_
        return (S + self.k * tp) / (N + self.k)

    def difficulty(self, c: pd.DataFrame, loso: bool = False) -> np.ndarray:
        return -logit(self.p_item(c, loso))


# =========================================================================== classifiers
class TorchMLPClassifier:
    """Two-layer MLP with dropout; early stopping on a student-disjoint 10% split of the training clips."""

    def __init__(self, hidden: int = 64, dropout: float = 0.2, lr: float = 2e-3, weight_decay: float = 1e-4,
                 epochs: int = 60, patience: int = 8, batch_size: int = 512, seed: int = 0, device: str = "cpu"):
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, weight_decay
        self.epochs, self.patience, self.bs, self.seed, self.device = epochs, patience, batch_size, seed, device

    def fit(self, X, y, groups=None):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        rng = np.random.default_rng(self.seed)
        if groups is not None:
            g = np.asarray(groups).astype(str)
            uniq = rng.permutation(np.unique(g))
            val_g = set(uniq[: max(1, len(uniq) // 10)])
            vm = np.array([x in val_g for x in g])
        else:
            vm = rng.random(len(y)) < 0.1
        dev = torch.device(self.device)
        self.net_ = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
                                  nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
                                  nn.Linear(self.hidden, 1)).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.BCEWithLogitsLoss()
        Xt, yt = torch.from_numpy(X[~vm]).to(dev), torch.from_numpy(y[~vm]).to(dev)
        Xv, yv = torch.from_numpy(X[vm]).to(dev), torch.from_numpy(y[vm]).to(dev)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                loss = lossf(self.net_(Xt[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vl = float(lossf(self.net_(Xv).squeeze(-1), yv).item()) if len(Xv) else float(loss.item())
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch

        self.net_.eval()
        with torch.no_grad():
            z = self.net_(torch.from_numpy(np.asarray(X, np.float32)).to(next(self.net_.parameters()).device)).squeeze(-1)
            p = torch.sigmoid(z).cpu().numpy().astype(float)
        return np.column_stack([1 - p, p])


def make_classifier(name: str, cfg, seed: int = 0, device: str = "cpu"):
    name = name.upper()
    c = cfg.get_path("v2.correctness", {}) or {}
    threads = int(cfg.get_path("models.gbm_threads", 6))
    if name == "LOGISTIC":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=0.5, max_iter=4000)
    if name == "LIGHTGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ModelUnavailable("lightgbm") from e
        return LGBMClassifier(**{**(c.get("lgbm") or {}), "random_state": seed, "n_jobs": threads, "verbose": -1})
    if name == "XGBOOST":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ModelUnavailable("xgboost") from e
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.6,
                             reg_lambda=5.0, min_child_weight=5, tree_method="hist", eval_metric="logloss",
                             random_state=seed, n_jobs=threads)
    if name == "CATBOOST":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ModelUnavailable("catboost") from e
        return CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=5.0, random_seed=seed,
                                  verbose=0, thread_count=threads, allow_writing_files=False)
    if name == "MLP":
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ModelUnavailable("torch") from e
        return TorchMLPClassifier(**(c.get("mlp") or {}), seed=seed, device=device)
    raise ValueError(f"unknown classifier {name}")


# =========================================================================== pipeline
class ClipPipeline:
    def __init__(self, bank: ClipFeatureBank, feature_set: str, model: str, cfg, seed: int = 0, device: str = "cpu",
                 embedding_pca: int = 32, include_theta: bool = False):
        self.bank, self.set, self.model_name, self.cfg, self.seed, self.device = bank, feature_set, model, cfg, seed, device
        self.spec = FEATURE_SETS[feature_set]
        self.blocks = [b for b in self.spec["blocks"] if bank.has(b)]
        self.emb_k = int(embedding_pca)
        self.include_theta = include_theta

    def _base(self, c: pd.DataFrame, loso: bool) -> pd.DataFrame:
        X = pd.DataFrame(index=c.index)
        for t in self.tasks_:
            X[f"task_{t}"] = (c["task"] == t).astype(float)
        X["item_logit_p"] = logit(self.enc_.p_item(c, loso))
        if self.spec["complexity"]:
            for k in COMPLEXITY_COLS:
                if k in c:
                    X[f"cpx_{k}"] = np.log1p(c[k].astype(float))
        if self.spec["duration"]:
            X["log_dur"] = c["log_dur"].astype(float)
        if self.include_theta:
            st = self.bank.ctx.bundle().all_students.drop_duplicates(ID).set_index(ID)
            for col in [x for x in st.columns if x.endswith("_theta")]:
                X[col] = c[ID].map(st[col]).astype(float).to_numpy()
        return X

    def _blocks(self, c: pd.DataFrame, rows: np.ndarray, fit: bool) -> np.ndarray:
        parts = []
        T = self.bank.table
        for b in self.blocks:
            cols = self.bank.blocks[b]
            F = T.loc[rows, cols].to_numpy(float)
            if b in RESIDUALIZE_BLOCKS:
                if fit:
                    self.res_[b] = ItemConditionedResidualizer(("item", "domain")).fit(c, F)
                F = self.res_[b].transform(c, F)
            elif b in PCA_BLOCKS:
                if fit:
                    from sklearn.decomposition import PCA

                    ok = np.isfinite(F).all(1)
                    mu, sd = np.nanmean(F[ok], 0), np.nanstd(F[ok], 0) + 1e-6
                    k = int(min(self.emb_k, max(1, ok.sum() - 1), F.shape[1]))
                    self.pca_[b] = (mu, sd, PCA(n_components=k, random_state=self.seed).fit((F[ok] - mu) / sd))
                mu, sd, pca = self.pca_[b]
                ok = np.isfinite(F).all(1)
                P = np.full((len(F), pca.n_components_), np.nan)
                if ok.any():
                    P[ok] = pca.transform((F[ok] - mu) / sd)
                F = P
            parts.append(F)
        return np.column_stack(parts) if parts else np.zeros((len(rows), 0))

    def _matrix(self, c: pd.DataFrame, rows: np.ndarray, fit: bool, loso: bool) -> np.ndarray:
        X = np.column_stack([self._base(c, loso).to_numpy(float), self._blocks(c, rows, fit)])
        if fit:
            self.keep_ = np.isfinite(X).mean(0) > 0.05
            self.med_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan)[:, self.keep_], 0)
            self.med_ = np.where(np.isfinite(self.med_), self.med_, 0.0)
        X = X[:, self.keep_]
        X = np.where(np.isfinite(X), X, self.med_[None, :])
        if fit:
            self.mu_, sd = X.mean(0), X.std(0)
            self.sd_ = np.where(sd > 1e-9, sd, 1.0)
        return np.clip((X - self.mu_) / self.sd_, -10, 10)

    def fit(self, rows: np.ndarray) -> "ClipPipeline":
        """rows: integer positions into bank.c / bank.table (training clips with a score)."""
        c = self.bank.c.iloc[rows]
        self.tasks_ = sorted(self.bank.c["task"].unique())
        self.enc_ = ItemEncoder(float(self.cfg.get_path("v2.harmonization.shrinkage_k", 10.0))).fit(c)
        self.res_, self.pca_ = {}, {}
        X = self._matrix(c, rows, fit=True, loso=True)
        y = c["item_score"].to_numpy(float).astype(int)
        self.clf_ = make_classifier(self.model_name, self.cfg, self.seed, self.device)
        if isinstance(self.clf_, TorchMLPClassifier):
            self.clf_.fit(X, y, groups=c[ID].to_numpy())
        else:
            self.clf_.fit(X, y)
        return self

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        c = self.bank.c.iloc[rows]
        X = self._matrix(c, rows, fit=False, loso=False)
        return np.asarray(self.clf_.predict_proba(X))[:, 1]
