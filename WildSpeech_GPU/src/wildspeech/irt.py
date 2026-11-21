"""Rasch / 2PL IRT by joint MAP (L-BFGS, analytic gradients) + person scoring with fixed item parameters.

Used (a) descriptively in P01 and (b) inside training folds by the feature builder: item parameters are
fitted on training-fold students only; held-out students are scored from their own responses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class IRTModel:
    def __init__(self, model: str = "rasch", theta_sd: float = 1.0, b_sd: float = 2.0, loga_sd: float = 0.5,
                 max_iter: int = 500, min_obs_item: int = 3):
        assert model in ("rasch", "2pl")
        self.model, self.theta_sd, self.b_sd, self.loga_sd = model, theta_sd, b_sd, loga_sd
        self.max_iter, self.min_obs_item = max_iter, min_obs_item
        self.items_: pd.DataFrame | None = None
        self.persons_: pd.DataFrame | None = None

    def fit(self, resp: pd.DataFrame, person: str = "subject_id", item: str = "item_id", score: str = "item_score") -> "IRTModel":
        d = resp[[person, item, score]].dropna()
        cnt = d[item].value_counts()
        d = d[d[item].isin(cnt[cnt >= self.min_obs_item].index)]
        if d.empty:
            self.items_ = pd.DataFrame(columns=["item_id", "b", "a", "n"])
            return self
        pc, pid = pd.factorize(d[person])
        ic, iid = pd.factorize(d[item])
        y = d[score].to_numpy(float)
        P, I = len(pid), len(iid)
        two = self.model == "2pl"

        def unpack(v):
            th, b = v[:P], v[P:P + I]
            la = v[P + I:] if two else np.zeros(I)
            return th, b, la

        def f(v):
            th, b, la = unpack(v)
            a = np.exp(la)
            eta = a[ic] * (th[pc] - b[ic])
            p = _sig(eta)
            ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            lp = -0.5 * (np.sum(th ** 2) / self.theta_sd ** 2 + np.sum(b ** 2) / self.b_sd ** 2 + (np.sum(la ** 2) / self.loga_sd ** 2 if two else 0))
            r = y - p
            g_th = np.bincount(pc, r * a[ic], P) - th / self.theta_sd ** 2
            g_b = -np.bincount(ic, r * a[ic], I) - b / self.b_sd ** 2
            grads = [g_th, g_b]
            if two:
                grads.append(np.bincount(ic, r * (th[pc] - b[ic]) * a[ic], I) - la / self.loga_sd ** 2)
            return -(ll + lp), -np.concatenate(grads)

        v0 = np.zeros(P + I + (I if two else 0))
        pm = d.groupby(ic)[score].mean().reindex(range(I)).to_numpy()
        v0[P:P + I] = -np.log(np.clip(pm, 0.02, 0.98) / (1 - np.clip(pm, 0.02, 0.98)))
        res = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        th, b, la = unpack(res.x)
        a = np.exp(la)
        p = _sig(a[ic] * (th[pc] - b[ic]))
        r = y - p
        var = np.clip(p * (1 - p), 1e-9, None)
        outfit = pd.Series(r ** 2 / var).groupby(ic).mean().reindex(range(I)).to_numpy()
        infit = (np.bincount(ic, r ** 2, I) / np.bincount(ic, var, I))
        self.items_ = pd.DataFrame({"item_id": iid, "b": b, "a": a, "n": np.bincount(ic, minlength=I),
                                    "p_correct": pm, "infit": infit, "outfit": outfit})
        self.persons_ = pd.DataFrame({"subject_id": pid, "theta": th})
        self.converged_ = bool(res.success)
        return self

    def score_persons(self, resp: pd.DataFrame, person: str = "subject_id", item: str = "item_id",
                      score: str = "item_score", n_iter: int = 30) -> pd.DataFrame:
        """MAP theta + SE for each person using fixed item parameters (unknown items ignored)."""
        if self.items_ is None or self.items_.empty:
            return pd.DataFrame(columns=["subject_id", "theta", "theta_se", "n_items"])
        d = resp[[person, item, score]].dropna().merge(self.items_[["item_id", "a", "b"]], left_on=item, right_on="item_id")
        if d.empty:
            return pd.DataFrame(columns=["subject_id", "theta", "theta_se", "n_items"])
        pc, pid = pd.factorize(d[person])
        P = len(pid)
        a, b, y = d["a"].to_numpy(), d["b"].to_numpy(), d[score].to_numpy(float)
        th = np.zeros(P)
        info = np.ones(P)
        for _ in range(n_iter):
            p = _sig(a * (th[pc] - b))
            g = np.bincount(pc, a * (y - p), P) - th / self.theta_sd ** 2
            info = np.bincount(pc, a ** 2 * p * (1 - p), P) + 1 / self.theta_sd ** 2
            step = g / info
            th = th + np.clip(step, -1, 1)
            if np.max(np.abs(step)) < 1e-6:
                break
        return pd.DataFrame({"subject_id": pid, "theta": th, "theta_se": 1 / np.sqrt(info),
                             "n_items": np.bincount(pc, minlength=P)})
