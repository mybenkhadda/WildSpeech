"""V4 fold model + evaluator on top of the V3 traced, fallback-aware cross-validation.

V4FoldModel(V3FoldModel) adds three fusions (everything fitted on the training fold, inner CV only):
  moe_gated   ReliabilityGatedExperts — M7 conditional mixture of experts: stage 1 = baseline model on the
              baseline columns (inner-OOF residual); one ridge expert per added modality (reduced family columns)
              fitted to that residual; expert weights w_j(x) = sigmoid(g(x)·θ_j) gated by task availability,
              recording domain and ASR reliability (ASRREL family, kept raw, never reduced). θ is fitted on the
              inner-OOF expert predictions with an L2 penalty. If the gated combination does not beat the stage-1
              inner-OOF MSE, every weight is set to 0 and the fold is traced as a FALLBACK.
  domain_moe  existing DomainPartialPooling (global + per-domain experts, partial pooling chosen by inner CV)
  hier_net    M8 HierarchicalMultimodalNet — compact modality towers + reliability gate on the stage-1 residual,
              seed ensemble, early stopping on an inner hold-out, shrinkage alpha in {0, .5, 1} chosen on that
              hold-out (alpha = 0 is traced as a fallback). torch required (else ModelUnavailable).
The V3 fusions (early / residual / late / task_pool / stack) are inherited unchanged.

V4Evaluator(V3Evaluator): fold_model_cls = V4FoldModel and extra controls
  permuted_families:<F1+F2>   rows of the named families' columns permuted (train and validation separately)
                              -> shuffled-ASR negative control
  missing_modality:<F1+F2>    the named families are set to NaN for VALIDATION students only (imputed by the
                              training-fold preprocessing) -> robustness to a missing modality at prediction time
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from ..evaluation.nested_asr_cv import V3Evaluator, V3FoldModel
from ..features.builder import family_of
from ..models.preprocessing import TabularPreprocessor
from ..models.zoo import ModelUnavailable, make_model, needs_scaling

ALPHAS = np.logspace(-2, 4, 31)
GATED = ("moe_gated", "hier_net")
V4_FUSIONS = ("moe_gated", "domain_moe", "hier_net")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ================================================================================ M7
class ReliabilityGatedExperts(BaseEstimator, RegressorMixin):
    def __init__(self, base_cols=None, gate_cols=None, base=None, inner_splits: int = 4, seed: int = 0, l2: float = 1.0,
                 min_weight: float = 1e-4):
        self.base_cols, self.gate_cols, self.base = base_cols, gate_cols, base
        self.inner_splits, self.seed, self.l2, self.min_weight = inner_splits, seed, l2, min_weight

    def _m1(self):
        return clone(self.base) if self.base is not None else RidgeCV(alphas=ALPHAS)

    def _gate(self, X: pd.DataFrame) -> np.ndarray:
        G = X[self.g_].to_numpy(float) if self.g_ else np.zeros((len(X), 0))
        G = np.where(np.isfinite(G), G, 0.0)
        return np.column_stack([np.ones(len(X)), G])

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.c1_ = [c for c in (self.base_cols or []) if c in X.columns]
        self.g_ = [c for c in (self.gate_cols or []) if c in X.columns]
        rest = [c for c in X.columns if c not in set(self.c1_) | set(self.g_)]
        self.experts_ = {}
        for c in rest:
            self.experts_.setdefault(family_of(c), []).append(c)
        names = list(self.experts_)
        kf = KFold(self.inner_splits, shuffle=True, random_state=self.seed)
        oof1 = np.zeros(len(y))
        for tr, va in kf.split(X):
            oof1[va] = np.asarray(self._m1().fit(X.iloc[tr][self.c1_], y[tr]).predict(X.iloc[va][self.c1_]), float).ravel()
        r = y - oof1
        S = np.zeros((len(y), len(names)))
        for tr, va in kf.split(X):
            for j, g in enumerate(names):
                S[va, j] = RidgeCV(alphas=ALPHAS).fit(X.iloc[tr][self.experts_[g]], r[tr]).predict(X.iloc[va][self.experts_[g]])
        G = self._gate(X)
        k, q = len(names), G.shape[1]
        self.base_mse_ = float(np.mean(r ** 2))
        if k == 0:
            self.theta_, self.fallback_, self.inner_mse_ = np.zeros((0, q)), True, self.base_mse_
        else:
            from scipy.optimize import minimize

            def loss(t):
                th = t.reshape(k, q)
                W = _sigmoid(G @ th.T)
                e = r - (W * S).sum(1)
                return float(np.mean(e ** 2) + self.l2 * np.sum(th[:, 1:] ** 2) / len(y))

            res = minimize(loss, np.zeros(k * q), method="L-BFGS-B", options={"maxiter": 500})
            self.theta_ = res.x.reshape(k, q)
            self.inner_mse_ = float(np.mean((r - (_sigmoid(G @ self.theta_.T) * S).sum(1)) ** 2))
            self.fallback_ = not (self.inner_mse_ < self.base_mse_)
        self.m1_ = self._m1().fit(X[self.c1_], y)
        self.m2_ = {g: RidgeCV(alphas=ALPHAS).fit(X[self.experts_[g]], r) for g in names} if not self.fallback_ else {}
        W = _sigmoid(G @ self.theta_.T) if k else np.zeros((len(y), 0))
        self.weights_ = pd.Series(0.0 if self.fallback_ else W.mean(0), index=names, dtype=float) if k else pd.Series(dtype=float)
        return self

    def gate_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        names = list(self.experts_)
        if not names or self.fallback_:
            return pd.DataFrame(0.0, index=X.index, columns=names)
        return pd.DataFrame(_sigmoid(self._gate(X) @ self.theta_.T), index=X.index, columns=names)

    def predict(self, X: pd.DataFrame):
        p = np.asarray(self.m1_.predict(X[self.c1_]), float).ravel()
        if self.fallback_ or not self.m2_:
            return p
        W = self.gate_weights(X)
        for g, m in self.m2_.items():
            p = p + W[g].to_numpy() * np.asarray(m.predict(X[self.experts_[g]]), float).ravel()
        return p


# ================================================================================ M8
class HierarchicalMultimodalNet(BaseEstimator, RegressorMixin):
    """Modality towers (one small MLP per added family) -> reliability-gated sum -> residual of a ridge stage 1."""

    def __init__(self, base_cols=None, gate_cols=None, hidden: int = 32, epochs: int = 300, patience: int = 25,
                 lr: float = 3e-3, weight_decay: float = 1e-3, dropout: float = 0.2, seeds=(0, 1, 2), seed: int = 0,
                 holdout: float = 0.2):
        self.base_cols, self.gate_cols, self.hidden, self.epochs, self.patience = base_cols, gate_cols, hidden, epochs, patience
        self.lr, self.weight_decay, self.dropout, self.seeds, self.seed, self.holdout = lr, weight_decay, dropout, seeds, seed, holdout

    def _net(self, dims: list[int], n_gate: int):
        import torch
        from torch import nn

        h, p = int(self.hidden), float(self.dropout)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.towers = nn.ModuleList([nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1)) for d in dims])
                self.gate = nn.Linear(max(n_gate, 1), len(dims))

            def forward(self, parts, g):
                outs = torch.cat([t(x) for t, x in zip(self.towers, parts)], dim=1)
                w = torch.sigmoid(self.gate(g))
                return (w * outs).sum(1)

        return Net()

    def _tensors(self, X: pd.DataFrame):
        import torch

        parts = [torch.tensor(np.nan_to_num(X[c].to_numpy(np.float32)), dtype=torch.float32) for c in self.groups_.values()]
        G = X[self.g_].to_numpy(np.float32) if self.g_ else np.ones((len(X), 1), np.float32)
        return parts, torch.tensor(np.nan_to_num(G), dtype=torch.float32)

    def _train(self, X: pd.DataFrame, r: np.ndarray, seed: int, idx_tr: np.ndarray, idx_va: np.ndarray):
        import torch

        torch.manual_seed(seed)
        net = self._net([len(c) for c in self.groups_.values()], len(self.g_))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        parts, G = self._tensors(X)
        rt = torch.tensor(r, dtype=torch.float32)
        best, best_state, bad = np.inf, None, 0
        for _ in range(int(self.epochs)):
            net.train()
            opt.zero_grad()
            loss = torch.mean((net([p[idx_tr] for p in parts], G[idx_tr]) - rt[idx_tr]) ** 2)
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                v = float(torch.mean((net([p[idx_va] for p in parts], G[idx_va]) - rt[idx_va]) ** 2))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= int(self.patience):
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net

    def _pred(self, nets, X: pd.DataFrame) -> np.ndarray:
        import torch

        parts, G = self._tensors(X)
        with torch.no_grad():
            return np.mean([n(parts, G).numpy() for n in nets], axis=0) * self.r_sd_

    def fit(self, X: pd.DataFrame, y):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ModelUnavailable("torch") from e
        y = np.asarray(y, float)
        self.c1_ = [c for c in (self.base_cols or []) if c in X.columns]
        self.g_ = [c for c in (self.gate_cols or []) if c in X.columns]
        self.groups_ = {}
        for c in X.columns:
            if c not in set(self.c1_) | set(self.g_):
                self.groups_.setdefault(family_of(c), []).append(c)
        oof1 = np.zeros(len(y))
        for tr, va in KFold(4, shuffle=True, random_state=self.seed).split(X):
            oof1[va] = RidgeCV(alphas=ALPHAS).fit(X.iloc[tr][self.c1_], y[tr]).predict(X.iloc[va][self.c1_])
        r = y - oof1
        self.r_sd_ = float(np.std(r)) or 1.0
        self.m1_ = RidgeCV(alphas=ALPHAS).fit(X[self.c1_], y)
        if not self.groups_:
            self.nets_, self.alpha_ = [], 0.0
            return self
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(y))
        n_va = max(10, int(round(self.holdout * len(y))))
        va, tr = perm[:n_va], perm[n_va:]
        nets = [self._train(X, r / self.r_sd_, int(s), tr, va) for s in self.seeds]
        pv = self._pred(nets, X.iloc[va])
        errs = {a: float(np.mean((r[va] - a * pv) ** 2)) for a in (0.0, 0.5, 1.0)}
        self.alpha_ = float(min(errs, key=errs.get))
        self.holdout_mse_ = errs
        self.nets_ = [self._train(X, r / self.r_sd_, int(s), np.arange(len(y)), va) for s in self.seeds] if self.alpha_ > 0 else []
        return self

    def predict(self, X: pd.DataFrame):
        p = np.asarray(self.m1_.predict(X[self.c1_]), float).ravel()
        if self.alpha_ > 0 and self.nets_:
            p = p + self.alpha_ * self._pred(self.nets_, X)
        return p


# ================================================================================ fold model
class V4FoldModel(V3FoldModel):
    def _gate_frame(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if self.spec.fusion not in GATED:
            return pd.DataFrame(index=X.index)
        cols = list(getattr(self, "gate_raw_", []))
        if fit:
            dom = [c for c in X.columns if c.startswith("RECORDING__orig_sr_low")]
            cols = list(dict.fromkeys(cols + dom))
            self.gate_raw_ = cols
        G = X.reindex(columns=cols).astype(float)
        av = self._avail(X)
        G = pd.concat([G, av], axis=1)
        if fit:
            self.gate_prep_ = TabularPreprocessor(max_missing_frac=1.0, add_indicators=False, scale=True).fit(G)
        Gt = self.gate_prep_.transform(G) if hasattr(self, "gate_prep_") else G
        Gt.columns = [f"__gate__{c}" for c in Gt.columns]
        return Gt

    def fit(self, X: pd.DataFrame, y) -> "V4FoldModel":
        y = np.asarray(y, float)
        self.gate_raw_ = []
        if self.spec.fusion in GATED:
            for fam in list(self.spec.params.get("gate_families", [])):
                self.gate_raw_ += [c for c in self.groups.pop(fam, []) if c in X.columns]
        scale = needs_scaling(self.spec.model) or self.spec.fusion != "early"
        self.prep_ = TabularPreprocessor(max_missing_frac=self.mmf, scale=scale).fit(X[self.base_cols])
        Xb = self.prep_.transform(X[self.base_cols])
        A = self._reduce_fit(X, Xb, y)
        G = self._gate_frame(X, fit=True)
        M = pd.concat([Xb, A, G], axis=1)
        if self.spec.fusion == "task_pool":
            M = pd.concat([M, self._avail(X)], axis=1)
        self.gate_out_cols_ = list(G.columns)
        self.est_ = self._estimator(list(Xb.columns))
        self.est_.fit(M, y)
        self.columns_, self.n_added_, self.base_out_cols_ = list(M.columns), int(A.shape[1]), list(Xb.columns)
        return self

    def matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        Xb = self.prep_.transform(X.reindex(columns=self.base_cols))
        A = self._reduce_transform(X)
        G = self._gate_frame(X, fit=False)
        M = pd.concat([Xb, A, G], axis=1)
        if self.spec.fusion == "task_pool":
            M = pd.concat([M, self._avail(X)], axis=1)
        return M.reindex(columns=self.columns_)

    def _estimator(self, Xb_cols: list[str]):
        s = self.spec
        if s.fusion == "moe_gated":
            moe = self.cfg.get_path("v4.moe", {}) or {}
            return ReliabilityGatedExperts(base_cols=Xb_cols, gate_cols=getattr(self, "gate_out_cols_", None),
                                           base=make_model(s.base_model or self.baseline_model, self.cfg, seed=self.seed),
                                           inner_splits=self.inner, seed=self.seed, l2=float(moe.get("l2", 1.0)),
                                           min_weight=float(moe.get("min_expert_weight", 1e-4)))
        if s.fusion == "domain_moe":
            from ..models.residual_fusion import DomainPartialPooling

            dom = [c for c in Xb_cols if c.startswith("RECORDING__orig_sr_low") or c.startswith("SCREENING__instrument_")]
            return DomainPartialPooling(dom[:2], int(self.cfg.get_path("v2.multimodal.moe_min_domain_size", 40)),
                                        inner_splits=self.inner, seed=self.seed)
        if s.fusion == "hier_net":
            h = self.cfg.get_path("v4.hier_net", {}) or {}
            return HierarchicalMultimodalNet(base_cols=Xb_cols, gate_cols=getattr(self, "gate_out_cols_", None),
                                             hidden=int(h.get("hidden", 32)), epochs=int(h.get("epochs", 300)),
                                             patience=int(h.get("patience", 25)), lr=float(h.get("lr", 3e-3)),
                                             weight_decay=float(h.get("weight_decay", 1e-3)),
                                             dropout=float(h.get("dropout", 0.2)), seeds=tuple(h.get("seeds", [0, 1, 2])),
                                             seed=self.seed)
        return super()._estimator(Xb_cols)

    def trace(self, Xva: pd.DataFrame) -> dict:
        t = super().trace(Xva)
        e = self.est_
        name = type(e).__name__
        t["n_gate"] = len(getattr(self, "gate_out_cols_", []))
        if name == "ReliabilityGatedExperts":
            t["inner_mse_alpha0"], t["inner_mse_best"] = e.base_mse_, e.inner_mse_
            if e.fallback_ and not t["fallback"]:
                t.update({"fallback": True, "fallback_reason": "gated experts did not beat the stage-1 inner-OOF MSE "
                                                               "(all expert weights set to 0)"})
            W = e.gate_weights(self.matrix(Xva))
            t["gate_weight_mean"] = float(W.to_numpy().mean()) if W.size else 0.0
            t["gate_weight_sd"] = float(W.to_numpy().std()) if W.size else 0.0
        elif name == "DomainPartialPooling":
            t["alpha"] = float(e.lambda_)
            t["n_domain_experts"] = int(e.n_experts_)
            if (e.lambda_ <= 0 or e.n_experts_ == 0) and not t["fallback"]:
                t.update({"fallback": True, "fallback_reason": "domain pooling lambda=0 or no domain expert"})
        elif name == "HierarchicalMultimodalNet":
            t["alpha"] = float(e.alpha_)
            if e.alpha_ <= 0 and not t["fallback"]:
                t.update({"fallback": True, "fallback_reason": "network shrinkage alpha=0 on the inner hold-out"})
        return t


# ================================================================================ evaluator
class V4Evaluator(V3Evaluator):
    fold_model_cls = V4FoldModel

    def __init__(self, ctx, stage: str = "V4", providers: dict | None = None):
        super().__init__(ctx, stage, providers)
        self.cache_dir = ctx.paths.cache / "v4" / "eval"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _control(self, spec, Xtr, Xva, groups, ytr, seed):
        ctl = str(spec.control or "")
        if ctl.startswith("permuted_families:") or ctl.startswith("missing_modality:"):
            kind, fams = ctl.split(":", 1)
            cols = [c for f in fams.split("+") for c in groups.get(f, []) if c in Xtr.columns]
            Xtr, Xva = Xtr.copy(), Xva.copy()
            if cols and kind == "permuted_families":
                rng = np.random.default_rng(seed)
                Xtr[cols] = Xtr[cols].to_numpy()[rng.permutation(len(Xtr))]
                Xva[cols] = Xva[cols].to_numpy()[rng.permutation(len(Xva))]
            elif cols:
                Xva[cols] = np.nan
            return Xtr, Xva, ytr
        return super()._control(spec, Xtr, Xva, groups, ytr, seed)
