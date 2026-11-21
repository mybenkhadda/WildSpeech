"""Neural multimodal models (PyTorch, single process, Windows-safe).

* MoERegressor        student-level mixture-of-experts: gate on screening scores, linear experts on all features
* StudentMILModel     clips = instances, student = bag. Clip encoder + task embedding, then either gated
                      attention pooling ('attention') or a lightweight Transformer encoder with CLS ('transformer').
                      Predicts the residual over an in-fold screening ridge baseline (stable, skip connection).
All models standardise inputs with training-fold statistics and early-stop on an inner validation split
drawn from training-fold students.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV

from .. import ID

ALPHAS = np.logspace(-2, 4, 31)


def _torch():
    import torch

    return torch


def _device(cfg) -> str:
    try:
        import torch

        pref = cfg.get_path("runtime.device", "auto")
        return "cuda" if pref in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _inner_split(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = max(2, int(round(n * frac)))
    return perm[k:], perm[:k]


# ------------------------------------------------------------------ MoE
class MoERegressor(BaseEstimator, RegressorMixin):
    def __init__(self, cfg=None, seed: int = 0, n_experts: int | None = None, gate_prefix: str = "SCREENING__"):
        self.cfg, self.seed, self.n_experts, self.gate_prefix = cfg, seed, n_experts, gate_prefix

    def fit(self, X: pd.DataFrame, y):
        torch = _torch()
        nn = torch.nn
        nc = self.cfg.section("multimodal").get("neural", {}) if self.cfg is not None else {}
        E = int(self.n_experts or nc.get("n_experts", 3))
        dev = _device(self.cfg) if self.cfg is not None else "cpu"
        X = pd.DataFrame(X)
        self.gate_cols_ = [c for c in X.columns if str(c).startswith(self.gate_prefix)] or list(X.columns[:5])
        Xa = X.to_numpy(np.float32)
        Xg = X[self.gate_cols_].to_numpy(np.float32)
        y = np.asarray(y, np.float32)
        self.ymu_, self.ysd_ = float(y.mean()), float(y.std() + 1e-6)
        yt = (y - self.ymu_) / self.ysd_
        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(s, d, dg):
                super().__init__()
                s.gate = nn.Sequential(nn.Linear(dg, 16), nn.GELU(), nn.Linear(16, E))
                s.experts = nn.Linear(d, E)
                s.drop = nn.Dropout(float(nc.get("dropout", 0.2)))

            def forward(s, xa, xg):
                w = torch.softmax(s.gate(xg), -1)
                return (w * s.experts(s.drop(xa))).sum(-1)

        tr, va = _inner_split(len(y), 0.15, self.seed)
        net = Net(Xa.shape[1], Xg.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=float(nc.get("lr", 1e-3)), weight_decay=float(nc.get("weight_decay", 0.01)) * 10)
        T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        xa, xg, yy = T(Xa), T(Xg), T(yt)
        best, best_state, bad = np.inf, None, 0
        for ep in range(int(nc.get("epochs", 150)) * 2):
            net.train()
            perm = np.random.default_rng(self.seed + ep).permutation(tr)
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss = ((net(xa[b], xg[b]) - yy[b]) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(((net(xa[va], xg[va]) - yy[va]) ** 2).mean())
            if vl < best - 1e-5:
                best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= int(nc.get("patience", 20)):
                    break
        net.load_state_dict(best_state)
        self.net_, self.dev_ = net.eval(), dev
        return self

    def predict(self, X):
        torch = _torch()
        X = pd.DataFrame(X)
        with torch.no_grad():
            xa = torch.tensor(X.to_numpy(np.float32), device=self.dev_)
            xg = torch.tensor(X[self.gate_cols_].to_numpy(np.float32), device=self.dev_)
            return self.net_(xa, xg).cpu().numpy() * self.ysd_ + self.ymu_


# ------------------------------------------------------------------ MIL
def make_bags(clip_df: pd.DataFrame, cols: list[str], ids: list[str], tasks: list[str], max_clips: int,
              mu: np.ndarray, sd: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> X (S, L, D) float32, task index (S, L) int64 (0 = padding), mask (S, L) bool."""
    t_index = {t: i + 1 for i, t in enumerate(tasks)}
    S, D = len(ids), len(cols)
    X = np.zeros((S, max_clips, D), np.float32)
    TI = np.zeros((S, max_clips), np.int64)
    M = np.zeros((S, max_clips), bool)
    groups = clip_df.groupby(ID)
    rng = np.random.default_rng(seed)
    for s, sid in enumerate(ids):
        if sid not in groups.groups:
            continue
        g = groups.get_group(sid)
        if len(g) > max_clips:
            g = g.iloc[np.sort(rng.choice(len(g), max_clips, replace=False))]
        v = (g[cols].to_numpy(np.float32) - mu) / sd
        X[s, :len(g)] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        TI[s, :len(g)] = g["task"].map(t_index).fillna(0).to_numpy(np.int64)
        M[s, :len(g)] = True
    return X, TI, M


def _build_mil_net(arch: str, d_in: int, n_tasks: int, nc: dict):
    torch = _torch()
    nn = torch.nn
    H, drop = int(nc.get("hidden", 64)), float(nc.get("dropout", 0.2))

    class MIL(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(d_in, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H))
            s.task = nn.Embedding(n_tasks + 1, H, padding_idx=0)
            if arch == "transformer":
                layer = nn.TransformerEncoderLayer(H, int(nc.get("heads", 4)), dim_feedforward=2 * H, dropout=drop,
                                                   batch_first=True, norm_first=True)
                s.tr = nn.TransformerEncoder(layer, int(nc.get("layers", 2)))
                s.cls = nn.Parameter(torch.zeros(1, 1, H))
            else:
                s.att_v = nn.Linear(H, H)
                s.att_u = nn.Linear(H, H)
                s.att_w = nn.Linear(H, 1)
            s.head = nn.Sequential(nn.LayerNorm(H), nn.Dropout(drop), nn.Linear(H, 1))

        def forward(s, x, ti, m):
            h = s.enc(x) + s.task(ti)
            if arch == "transformer":
                B = h.shape[0]
                h = torch.cat([s.cls.expand(B, -1, -1), h], 1)
                pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=m.device), ~m], 1)
                z = s.tr(h, src_key_padding_mask=pad)[:, 0]
                att = None
            else:
                a = s.att_w(torch.tanh(s.att_v(h)) * torch.sigmoid(s.att_u(h))).squeeze(-1)
                a = a.masked_fill(~m, -1e4)
                att = torch.softmax(a, -1)
                z = (att.unsqueeze(-1) * h).sum(1)
            return s.head(z).squeeze(-1), att

    return MIL()


class StudentMILModel:
    def __init__(self, arch: str, cfg, tasks: list[str], seed: int = 0):
        self.arch, self.cfg, self.tasks, self.seed = arch, cfg, tasks, seed
        self.nc = cfg.section("multimodal").get("neural", {})

    def fit(self, clip_df: pd.DataFrame, cols: list[str], X_student: pd.DataFrame, y: pd.Series) -> "StudentMILModel":
        torch = _torch()
        ids = list(y.index.astype(str))
        tr_clips = clip_df[clip_df[ID].isin(ids)]
        V = tr_clips[cols].to_numpy(np.float32)
        self.mu_ = np.nanmean(V, 0)
        self.mu_ = np.where(np.isfinite(self.mu_), self.mu_, 0.0).astype(np.float32)
        self.sd_ = np.nanstd(V, 0)
        self.sd_ = np.where(np.isfinite(self.sd_) & (self.sd_ > 1e-6), self.sd_, 1.0).astype(np.float32)
        self.cols_ = cols
        scr = [c for c in X_student.columns if c.startswith("SCREENING__")]
        self.scr_cols_ = scr
        Xs = X_student.reindex(ids)[scr]
        self.scr_med_ = Xs.median()
        self.base_ = RidgeCV(alphas=ALPHAS).fit(Xs.fillna(self.scr_med_), y.values)
        base_pred = self.base_.predict(Xs.fillna(self.scr_med_))
        r = y.values - base_pred
        self.rsd_ = float(r.std() + 1e-6)
        rt = (r / self.rsd_).astype(np.float32)
        Xb, TI, M = make_bags(clip_df, cols, ids, self.tasks, int(self.nc.get("max_clips_per_student", 64)), self.mu_, self.sd_, self.seed)
        dev = _device(self.cfg)
        self.dev_ = dev
        self.nets_ = []
        for s in self.nc.get("seeds", [0, 1, 2]):
            torch.manual_seed(self.seed * 100 + int(s))
            net = _build_mil_net(self.arch, len(cols), len(self.tasks), self.nc).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=float(self.nc.get("lr", 1e-3)), weight_decay=float(self.nc.get("weight_decay", 0.01)))
            tr, va = _inner_split(len(ids), 0.15, self.seed + int(s))
            T = lambda a, dt=None: torch.tensor(a, device=dev) if dt is None else torch.tensor(a, device=dev, dtype=dt)  # noqa: E731
            xb, ti, mm, yy = T(Xb), T(TI), T(M), T(rt)
            best, best_state, bad = np.inf, None, 0
            bs = int(self.nc.get("batch_students", 32))
            for ep in range(int(self.nc.get("epochs", 150))):
                net.train()
                perm = np.random.default_rng(int(s) * 7 + ep).permutation(tr)
                for i in range(0, len(perm), bs):
                    b = torch.tensor(perm[i:i + bs], device=dev)
                    opt.zero_grad()
                    pred, _ = net(xb[b], ti[b], mm[b])
                    loss = ((pred - yy[b]) ** 2).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vb = torch.tensor(va, device=dev)
                    vp, _ = net(xb[vb], ti[vb], mm[vb])
                    vl = float(((vp - yy[vb]) ** 2).mean())
                if vl < best - 1e-5:
                    best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= int(self.nc.get("patience", 20)):
                        break
            net.load_state_dict(best_state)
            self.nets_.append(net.eval())
        return self

    def predict(self, clip_df: pd.DataFrame, X_student: pd.DataFrame, ids: list[str], return_attention: bool = False):
        torch = _torch()
        ids = [str(i) for i in ids]
        Xs = X_student.reindex(ids).reindex(columns=self.scr_cols_).fillna(self.scr_med_)
        base = self.base_.predict(Xs)
        Xb, TI, M = make_bags(clip_df, self.cols_, ids, self.tasks, int(self.nc.get("max_clips_per_student", 64)), self.mu_, self.sd_, self.seed)
        preds, atts = [], []
        with torch.no_grad():
            xb, ti, mm = (torch.tensor(a, device=self.dev_) for a in (Xb, TI, M))
            for net in self.nets_:
                p, a = net(xb, ti, mm)
                preds.append(p.cpu().numpy())
                if a is not None:
                    atts.append(a.cpu().numpy())
        out = base + np.mean(preds, 0) * self.rsd_
        if return_attention:
            return out, (np.mean(atts, 0) if atts else None), M
        return out
