"""Frozen encoder embeddings (P24 pooled clip vectors) -> compact student representations.

PooledEmbeddingProvider (ASRE)   clip embeddings -> PCA fitted on TRAINING-fold clips -> per student x task mean /
                                 sd of the components (+ correct-only mean). Never thousands of raw dimensions.
AttentionMIL                     multiple-instance regression: a student = a bag of clip vectors (PCA-reduced,
                                 + task one-hot + correctness); gated attention pooling -> scalar prediction of
                                 the training-fold BASELINE RESIDUAL. Small (hidden 32), weight decay, dropout,
                                 early stopping on an inner split.
MILProvider (ASRE)               nested: training students receive OOF MIL scores from inner student folds;
                                 validation students are scored by the MIL fitted on all training students.
                                 Used only when the training fold has >= mil.min_train_students students
                                 (otherwise the provider is unavailable and recorded as such).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import FoldFeatureProvider, aggregate_student_task
from ..io_utils import stable_hash
from ..v2.validation_registry import inner_student_folds


def embedding_matrix(emb: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    e = emb[[FILE] + cols].drop_duplicates(FILE)
    return e, cols


class PooledEmbeddingProvider(FoldFeatureProvider):
    family = "ASRE"

    def __init__(self, ctx, name: str, emb: pd.DataFrame, n_pcs: int = 16):
        super().__init__(ctx, name, "pooled frozen encoder embeddings")
        self.source_model = str(getattr(emb, "attrs", {}).get("model_key", ""))
        self.emb, self.cols = embedding_matrix(emb)
        self.k = int(n_pcs)

    def available(self) -> bool:
        return bool(self.cols) and len(self.emb) > 100

    def fingerprint_payload(self) -> dict:
        # the source model is part of the identity: two encoders with the same width and clip set must never share
        # a fold cache (ASRE_pooled switches model when Pasketti features are excluded)
        return {"k": self.k, "dim": len(self.cols), "files": stable_hash(sorted(self.emb[FILE].astype(str))),
                "model": self.source_model, "v": "asre-2"}

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from sklearn.decomposition import PCA

        from ..discovery.item_conditioning import v3_clip_frame

        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].isin(ids)].merge(self.emb, on=FILE, how="inner").reset_index(drop=True)
        if c.empty:
            return pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        X = c[self.cols].to_numpy(np.float32)
        tr = c[ID].isin(set(map(str, train_ids))).to_numpy()
        ok = np.isfinite(X).all(1)
        mu = X[tr & ok].mean(0)
        sd = X[tr & ok].std(0) + 1e-6
        k = int(min(self.k, (tr & ok).sum() - 1, X.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(((X[tr & ok] - mu) / sd))
        Z = np.full((len(c), k), np.nan, np.float32)
        Z[ok] = pca.transform((X[ok] - mu) / sd)
        out = aggregate_student_task(c, Z, [f"pc{j}" for j in range(k)], "ASRE", stats=("mean", "sd", "correct_mean"))
        return out.reindex(sorted(ids))


# ==================================================================================== attention MIL
class AttentionMIL:
    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.2, lr: float = 3e-3, weight_decay: float = 1e-3,
                 epochs: int = 80, patience: int = 10, seed: int = 0, device: str = "cpu"):
        self.in_dim, self.hidden, self.dropout, self.lr, self.wd = in_dim, hidden, dropout, lr, weight_decay
        self.epochs, self.patience, self.seed, self.device = epochs, patience, seed, device

    def _net(self):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(s, d, h, p):
                super().__init__()
                s.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p))
                s.V, s.U, s.w = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, 1)
                s.head = nn.Linear(h, 1)

            def forward(s, x, mask):                       # x (B, N, d), mask (B, N) True = valid instance
                h = s.enc(x)
                a = s.w(torch.tanh(s.V(h)) * torch.sigmoid(s.U(h))).squeeze(-1)
                a = a.masked_fill(~mask, -1e9).softmax(-1)
                z = (a.unsqueeze(-1) * h).sum(1)
                return s.head(z).squeeze(-1), a

        return Net(self.in_dim, self.hidden, self.dropout).to(self.device)

    @staticmethod
    def pad(bags: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = max(len(b) for b in bags)
        d = bags[0].shape[1]
        X = np.zeros((len(bags), n, d), np.float32)
        M = np.zeros((len(bags), n), bool)
        for i, b in enumerate(bags):
            X[i, : len(b)] = b
            M[i, : len(b)] = True
        return X, M

    def fit(self, bags: list[np.ndarray], y: np.ndarray) -> "AttentionMIL":
        import torch

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(bags))
        n_va = max(5, len(bags) // 8)
        va, tr = idx[:n_va], idx[n_va:]
        X, M = self.pad(bags)
        Xt, Mt, yt = (torch.tensor(X, device=self.device), torch.tensor(M, device=self.device),
                      torch.tensor(np.asarray(y, np.float32), device=self.device))
        self.y_mu_, self.y_sd_ = float(np.mean(y[tr])), float(np.std(y[tr]) + 1e-6)
        ys = (yt - self.y_mu_) / self.y_sd_
        net = self._net()
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
                opt.zero_grad()
                p, _ = net(Xt[b], Mt[b])
                loss = torch.mean((p - ys[b]) ** 2)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv, _ = net(Xt[va], Mt[va])
                v = float(torch.mean((pv - ys[va]) ** 2))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_, self.val_mse_ = net.eval(), best
        return self

    def predict(self, bags: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        X, M = self.pad(bags)
        with torch.no_grad():
            p, a = self.net_(torch.tensor(X, device=self.device), torch.tensor(M, device=self.device))
        a = a.cpu().numpy()
        ent = np.array([-(ai[m] * np.log(ai[m] + 1e-12)).sum() / max(np.log(m.sum()), 1e-9) for ai, m in zip(a, M)])
        return p.cpu().numpy() * self.y_sd_ + self.y_mu_, ent


class MILProvider(FoldFeatureProvider):
    """Nested attention-MIL score of the BASELINE residual. `residual_fn(train_ids)` must return the training-fold
    residual of the tabular baseline computed WITHOUT using the target students (in-fold OOF residuals)."""

    family = "ASRE"

    def __init__(self, ctx, name: str, emb: pd.DataFrame, residual_fn, cfg_mil: dict, n_pcs: int = 32):
        super().__init__(ctx, name, "attention-MIL pooled embeddings")
        self.emb, self.cols = embedding_matrix(emb)
        self.residual_fn, self.m, self.k = residual_fn, dict(cfg_mil or {}), int(n_pcs)

    def available(self) -> bool:
        return bool(self.m.get("enabled", True)) and bool(self.cols) and len(self.emb) > 500

    def fingerprint_payload(self) -> dict:
        return {"m": self.m, "k": self.k, "files": stable_hash(sorted(self.emb[FILE].astype(str))), "v": "mil-1"}

    def _bags(self, c: pd.DataFrame, Z: np.ndarray, sids: list[str]) -> list[np.ndarray]:
        tasks = sorted(c["task"].unique())
        T = np.column_stack([(c["task"] == t).to_numpy(np.float32) for t in tasks] +
                            [c["item_score"].fillna(0.5).to_numpy(np.float32)])
        F = np.hstack([Z, T]).astype(np.float32)
        by = {s: np.flatnonzero(c[ID].astype(str).to_numpy() == s) for s in sids}
        return [F[by[s]] if len(by[s]) else np.zeros((1, F.shape[1]), np.float32) for s in sids]

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from sklearn.decomposition import PCA

        from ..discovery.item_conditioning import v3_clip_frame

        train_ids = sorted(map(str, train_ids))
        ids = sorted(set(train_ids) | set(map(str, target_ids)))
        if len(train_ids) < int(self.m.get("min_train_students", 200)):
            return pd.DataFrame(index=pd.Index(ids, name=ID))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].isin(set(ids))].merge(self.emb, on=FILE, how="inner").reset_index(drop=True)
        X = np.nan_to_num(c[self.cols].to_numpy(np.float32))
        tr = c[ID].isin(set(train_ids)).to_numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        pca = PCA(n_components=int(min(self.k, tr.sum() - 1, X.shape[1])), random_state=0).fit((X[tr] - mu) / sd)
        Z = pca.transform((X - mu) / sd).astype(np.float32)
        resid = self.residual_fn(train_ids)                          # pd.Series indexed by train student
        seed = int(int(stable_hash(train_ids), 16) % (2 ** 31 - 1))
        kw = {k: self.m[k] for k in ("hidden", "dropout", "lr", "weight_decay", "epochs", "patience") if k in self.m}
        dev = "cuda" if self.ctx.device == "cuda" else "cpu"
        score, ent = pd.Series(np.nan, index=ids), pd.Series(np.nan, index=ids)
        for in_tr, in_va in inner_student_folds(train_ids, int(self.ctx.cfg.get_path("v3.representations.nested_soft.inner_splits", 4)), seed):
            in_tr = [s for s in in_tr if s in resid.index]
            m = AttentionMIL(Z.shape[1] + len(c["task"].unique()) + 1, seed=seed, device=dev, **kw)
            m.fit(self._bags(c, Z, in_tr), resid.reindex(in_tr).to_numpy(float))
            p, e = m.predict(self._bags(c, Z, in_va))
            score[in_va], ent[in_va] = p, e
        others = [s for s in ids if s not in set(train_ids)]
        if others:
            m = AttentionMIL(Z.shape[1] + len(c["task"].unique()) + 1, seed=seed, device=dev, **kw)
            tri = [s for s in train_ids if s in resid.index]
            m.fit(self._bags(c, Z, tri), resid.reindex(tri).to_numpy(float))
            p, e = m.predict(self._bags(c, Z, others))
            score[others], ent[others] = p, e
        out = pd.DataFrame({"ASRE__all__mil_score__value": score, "ASRE__all__mil_attention_entropy__value": ent})
        out.index.name = ID
        return out
