"""Regularised item response theory (Rasch, 1PL, 2PL) by penalised maximum likelihood.

Model: P(y=1) = sigmoid(a_i * (theta_s - b_i)) with Gaussian priors
theta ~ N(0, s_theta^2), b ~ N(0, s_b^2), log a ~ N(0, s_a^2).

* rasch: a_i = 1
* 1pl:   a_i = a (one common discrimination)
* 2pl:   item-specific a_i (only fitted when every item has enough responses)

Only observed responses enter the likelihood: non-attempted items are never
treated as incorrect, which accounts for adaptive exposure and sparsity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class IRTResult:
    model: str
    items: pd.DataFrame
    students: pd.DataFrame
    loglik: float
    n_params: int
    n_obs: int
    converged: bool
    message: str

    @property
    def aic(self) -> float:
        return float(2 * self.n_params - 2 * self.loglik)

    @property
    def bic(self) -> float:
        return float(np.log(max(self.n_obs, 1)) * self.n_params - 2 * self.loglik)


def _encode(df: pd.DataFrame, student_col: str, item_col: str, resp_col: str):
    d = df[[student_col, item_col, resp_col]].dropna()
    d = d[d[resp_col].isin([0, 1, 0.0, 1.0])]
    s_codes, s_index = pd.factorize(d[student_col])
    i_codes, i_index = pd.factorize(d[item_col])
    return s_codes, i_codes, d[resp_col].to_numpy(dtype=float), s_index, i_index


def fit_irt(df: pd.DataFrame, student_col: str = "student_id", item_col: str = "item_key", resp_col: str = "correct",
            model: str = "rasch", prior_theta_sd: float = 1.0, prior_b_sd: float = 2.0, prior_loga_sd: float = 0.5,
            max_iter: int = 500, min_obs_per_item: int = 5) -> IRTResult:
    counts = df.dropna(subset=[resp_col]).groupby(item_col)[resp_col].count()
    keep = counts[counts >= min_obs_per_item].index
    d = df[df[item_col].isin(keep)]
    s, i, y, s_index, i_index = _encode(d, student_col, item_col, resp_col)
    S, I, N = len(s_index), len(i_index), len(y)
    if N == 0 or I == 0:
        raise ValueError("no valid binary responses for IRT")
    n_a = {"rasch": 0, "1pl": 1, "2pl": I}[model]

    def unpack(p):
        theta, b = p[:S], p[S:S + I]
        if model == "rasch":
            loga = np.zeros(I)
        elif model == "1pl":
            loga = np.full(I, p[S + I])
        else:
            loga = p[S + I:]
        return theta, b, loga

    def objective(p):
        theta, b, loga = unpack(p)
        a = np.exp(loga)
        diff = theta[s] - b[i]
        eta = a[i] * diff
        prob = expit(eta)
        ll = np.sum(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12))
        pen = (theta ** 2).sum() / (2 * prior_theta_sd ** 2) + (b ** 2).sum() / (2 * prior_b_sd ** 2)
        r = prob - y
        g_theta = np.bincount(s, weights=r * a[i], minlength=S) + theta / prior_theta_sd ** 2
        g_b = np.bincount(i, weights=-r * a[i], minlength=I) + b / prior_b_sd ** 2
        grads = [g_theta, g_b]
        if model == "1pl":
            pen += loga[0] ** 2 / (2 * prior_loga_sd ** 2)
            grads.append(np.array([np.sum(r * a[i] * diff) + loga[0] / prior_loga_sd ** 2]))
        elif model == "2pl":
            pen += (loga ** 2).sum() / (2 * prior_loga_sd ** 2)
            grads.append(np.bincount(i, weights=r * a[i] * diff, minlength=I) + loga / prior_loga_sd ** 2)
        return -ll + pen, np.concatenate(grads)

    item_p = pd.Series(y).groupby(i).mean().reindex(range(I)).fillna(0.5).clip(0.02, 0.98).to_numpy()
    p0 = np.concatenate([np.zeros(S), -np.log(item_p / (1 - item_p)), np.zeros(n_a)])
    res = minimize(objective, p0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    theta, b, loga = unpack(res.x)
    a = np.exp(loga)
    prob = expit(a[i] * (theta[s] - b[i]))
    info_obs = a[i] ** 2 * prob * (1 - prob)
    b_se = 1 / np.sqrt(np.bincount(i, weights=info_obs, minlength=I) + 1 / prior_b_sd ** 2)
    t_se = 1 / np.sqrt(np.bincount(s, weights=info_obs, minlength=S) + 1 / prior_theta_sd ** 2)
    diff = theta[s] - b[i]
    a_info = np.bincount(i, weights=(a[i] * diff) ** 2 * prob * (1 - prob), minlength=I) + 1 / prior_loga_sd ** 2
    loga_se = 1 / np.sqrt(a_info) if model == "2pl" else np.full(I, np.nan)
    ll = float(np.sum(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12)))
    items = pd.DataFrame({"item_key": i_index, "b": b, "b_se": b_se, "a": a, "loga_se": loga_se,
                          "n_obs": np.bincount(i, minlength=I), "p_correct": pd.Series(y).groupby(i).mean().reindex(range(I)).to_numpy()})
    students = pd.DataFrame({"student_id": s_index, "theta": theta, "theta_se": t_se, "n_obs": np.bincount(s, minlength=S)})
    return IRTResult(model, items, students, ll, S + I + n_a, N, bool(res.success), str(res.message))


def score_students(df: pd.DataFrame, items: pd.DataFrame, student_col: str = "student_id", item_col: str = "item_key",
                   resp_col: str = "correct", prior_theta_sd: float = 1.0, n_iter: int = 40) -> pd.DataFrame:
    """MAP ability for each student given fixed item parameters (Newton iterations)."""
    d = df[[student_col, item_col, resp_col]].dropna()
    d = d.merge(items[["item_key", "a", "b"]].rename(columns={"item_key": item_col}), on=item_col, how="inner")
    if d.empty:
        return pd.DataFrame(columns=[student_col, "theta", "theta_se", "n_obs"])
    s, s_index = pd.factorize(d[student_col])
    a, b, y = d["a"].to_numpy(), d["b"].to_numpy(), d[resp_col].to_numpy(dtype=float)
    S = len(s_index)
    theta = np.zeros(S)
    for _ in range(n_iter):
        p = expit(a * (theta[s] - b))
        g = np.bincount(s, weights=a * (p - y), minlength=S) + theta / prior_theta_sd ** 2
        h = np.bincount(s, weights=a ** 2 * p * (1 - p), minlength=S) + 1 / prior_theta_sd ** 2
        step = g / h
        theta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    p = expit(a * (theta[s] - b))
    h = np.bincount(s, weights=a ** 2 * p * (1 - p), minlength=S) + 1 / prior_theta_sd ** 2
    return pd.DataFrame({student_col: s_index, "theta": theta, "theta_se": 1 / np.sqrt(h), "n_obs": np.bincount(s, minlength=S)})


def item_information(theta_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    p = expit(a * (theta_grid - b))
    return a ** 2 * p * (1 - p)


def difficulty_features(df: pd.DataFrame, items: pd.DataFrame, abilities: pd.DataFrame, prefix: str = "irt_",
                        student_col: str = "student_id", item_col: str = "item_key", resp_col: str = "correct") -> pd.DataFrame:
    """Difficulty-aware student features from fixed item parameters and own responses."""
    d = df[[student_col, item_col, resp_col]].dropna()
    d = d.merge(items[["item_key", "a", "b"]].rename(columns={"item_key": item_col}), on=item_col, how="inner")
    d = d.merge(abilities[[student_col, "theta"]], on=student_col, how="left")
    if d.empty:
        return pd.DataFrame(columns=[student_col])
    b_lo, b_hi = items["b"].quantile(1 / 3), items["b"].quantile(2 / 3)
    d["p_hat"] = expit(d["a"] * (d["theta"] - d["b"]))
    d["w"] = expit(d["b"])
    d["info"] = d["a"] ** 2 * d["p_hat"] * (1 - d["p_hat"])
    d["resid"] = d[resp_col] - d["p_hat"]
    d["easy_expected"] = d["p_hat"] >= 0.8
    d["hard_expected"] = d["p_hat"] <= 0.2
    g = d.groupby(student_col)

    def ratio(num, den):
        return num / den.where(den > 0)

    out = pd.DataFrame({
        f"{prefix}difficulty_weighted_accuracy": ratio(g.apply(lambda x: (x[resp_col] * x["w"]).sum()), g["w"].sum()),
        f"{prefix}mean_attempted_item_difficulty": g["b"].mean(),
        f"{prefix}max_attempted_item_difficulty": g["b"].max(),
        f"{prefix}hard_item_accuracy": d[d["b"] >= b_hi].groupby(student_col)[resp_col].mean(),
        f"{prefix}easy_item_accuracy": d[d["b"] <= b_lo].groupby(student_col)[resp_col].mean(),
        f"{prefix}unexpected_easy_item_error_rate": d[d["easy_expected"]].groupby(student_col)[resp_col].apply(lambda s: 1 - s.mean()),
        f"{prefix}unexpected_hard_item_success_rate": d[d["hard_expected"]].groupby(student_col)[resp_col].mean(),
        f"{prefix}response_residual_mean": g["resid"].mean(),
        f"{prefix}response_residual_std": g["resid"].std(),
        f"{prefix}item_information_weighted_score": ratio(g.apply(lambda x: (x[resp_col] * x["info"]).sum()), g["info"].sum()),
        f"{prefix}theta": abilities.set_index(student_col)["theta"],
        f"{prefix}theta_se": abilities.set_index(student_col)["theta_se"],
        f"{prefix}n_scored": g[resp_col].count(),
    })
    out.index.name = student_col
    return out.reset_index()


def fit_task_irt(clips: pd.DataFrame, cfg: dict, models: list | None = None) -> dict:
    """Fit IRT separately per task family; returns per-task results and selection."""
    results: dict = {}
    for task, g in clips.dropna(subset=["correct", "item_key"]).groupby("task_family"):
        if g["item_key"].nunique() < 3 or g["student_id"].nunique() < 10:
            results[task] = {"status": "INSUFFICIENT_DATA"}
            continue
        fits = {}
        counts = g.groupby("item_key")["correct"].count()
        for m in models or cfg["irt_models"]:
            if m == "2pl" and counts[counts >= int(cfg["irt_min_obs_per_item"])].min() < int(cfg["irt_2pl_min_obs_per_item"]):
                fits[m] = None
                continue
            fits[m] = fit_irt(g, model=m, prior_theta_sd=float(cfg["irt_prior_theta_sd"]),
                              prior_b_sd=float(cfg["irt_prior_b_sd"]), prior_loga_sd=float(cfg["irt_prior_loga_sd"]),
                              max_iter=int(cfg["irt_max_iter"]), min_obs_per_item=int(cfg["irt_min_obs_per_item"]))
        valid = {m: f for m, f in fits.items() if f is not None and f.converged}
        if not valid:
            valid = {m: f for m, f in fits.items() if f is not None}
        selected = min(valid, key=lambda m: valid[m].bic) if valid else None
        results[task] = {"status": "OK" if selected else "FAILED", "fits": fits, "selected": selected}
    return results


def fold_irt_features(clips: pd.DataFrame, train_students: set, cfg: dict, prefix: str = "irt_") -> pd.DataFrame:
    """Fit item parameters on training-fold students only, then score every student."""
    frames = []
    train = clips[clips["student_id"].isin(train_students)]
    for task, g_train in train.dropna(subset=["correct", "item_key"]).groupby("task_family"):
        if g_train["item_key"].nunique() < 3 or g_train["student_id"].nunique() < 10:
            continue
        fit = fit_irt(g_train, model="rasch", prior_theta_sd=float(cfg["irt_prior_theta_sd"]),
                      prior_b_sd=float(cfg["irt_prior_b_sd"]), max_iter=int(cfg["irt_max_iter"]),
                      min_obs_per_item=int(cfg["irt_min_obs_per_item"]))
        g_all = clips[clips["task_family"] == task]
        ab = score_students(g_all, fit.items, prior_theta_sd=float(cfg["irt_prior_theta_sd"]))
        feats = difficulty_features(g_all, fit.items, ab, prefix=f"{prefix}{task}_")
        frames.append(feats.set_index("student_id"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    for stat in ("difficulty_weighted_accuracy", "response_residual_mean", "theta", "unexpected_easy_item_error_rate"):
        cols = [c for c in out.columns if c.endswith(stat) and c.startswith(prefix)]
        if cols:
            out[f"{prefix}all_{stat}"] = out[cols].mean(axis=1)
    return out
