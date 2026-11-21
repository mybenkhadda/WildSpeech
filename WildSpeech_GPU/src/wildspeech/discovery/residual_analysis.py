"""Grouping-aware statistics for V3 discovery.

student_block_bootstrap(stat, frame, group, n_boot)   resample STUDENTS (all their clips together) -> CI
auc / auc_diff_block_bootstrap                        clip AUROC and paired AUROC differences with student blocks
stratified_permutation_p                              permutation of the student-level feature within strata
                                                      (school when available, else ability stratum) -> p value
residual_associations                                 discovery half screen -> confirmation half test of the
                                                      top-k (Spearman, stratified permutation p, partial correlation
                                                      controlling for screening ability), BH-FDR per tier
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .. import ID


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    y, s = np.asarray(y, float), np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return np.nan
    return float(roc_auc_score(y[m], s[m]))


def student_block_bootstrap(stat: Callable[[pd.DataFrame], float], frame: pd.DataFrame, group: str = ID,
                            n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    est = stat(frame)
    groups = frame[group].astype(str).to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[g] for g in pick])
        v = stat(frame.iloc[rows])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return est, np.nan, np.nan
    return float(est), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def auc_diff_block_bootstrap(frame: pd.DataFrame, y: str, s_new: str, s_base: str, n_boot: int = 1000,
                             seed: int = 0) -> dict:
    def d(f):
        return auc(f[y], f[s_new]) - auc(f[y], f[s_base])

    est, lo, hi = student_block_bootstrap(d, frame, ID, n_boot, seed)
    return {"auc_new": auc(frame[y], frame[s_new]), "auc_base": auc(frame[y], frame[s_base]), "delta_auc": est,
            "ci_low": lo, "ci_high": hi}


def stratified_permutation_p(x: pd.Series, y: pd.Series, strata: pd.Series | None, n_perm: int = 2000,
                             seed: int = 0) -> float:
    """Two-sided permutation p of Spearman(x, y), permuting x within strata (students are the units)."""
    from scipy.stats import rankdata

    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if strata is not None:
        j["s"] = strata.reindex(j.index).fillna("NA").astype(str)
    else:
        j["s"] = "all"
    if len(j) < 10:
        return np.nan
    xr, yr = rankdata(j["x"]), rankdata(j["y"])
    xr, yr = (xr - xr.mean()) / (xr.std() + 1e-12), (yr - yr.mean()) / (yr.std() + 1e-12)
    obs = abs(float(np.mean(xr * yr)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(j["s"].to_numpy() == g) for g in j["s"].unique()]
    hits = 0
    for _ in range(int(n_perm)):
        xp = xr.copy()
        for g in groups:
            xp[g] = xp[rng.permutation(g)]
        hits += abs(float(np.mean(xp * yr))) >= obs - 1e-12
    return float((hits + 1) / (n_perm + 1))


def residual_associations(X: pd.DataFrame, resid: pd.Series, ability: pd.Series, ids_disc: list[str],
                          ids_conf: list[str], strata: pd.Series | None, n_perm: int, seed: int, top_k: int = 25) -> pd.DataFrame:
    from ..stats import bh_fdr, partial_corr, spearman

    rows = []
    for c in X.columns:
        r, p, n = spearman(X[c].reindex(ids_disc), resid.reindex(ids_disc))
        rows.append({"feature": c, "tier": "exploratory", "rho": r, "p": p, "n": n})
    exp = pd.DataFrame(rows)
    if exp.empty:
        return exp
    exp["q"] = bh_fdr(exp["p"].to_numpy(float))
    top = exp.dropna(subset=["p"]).sort_values("p").head(top_k)
    conf = []
    for c, rho_d in zip(top["feature"], top["rho"]):
        x, y = X[c].reindex(ids_conf), resid.reindex(ids_conf)
        r, p, n = spearman(x, y)
        pp = stratified_permutation_p(x, y, strata.reindex(ids_conf) if strata is not None else None, n_perm, seed)
        j = pd.concat([x.rename("x"), y.rename("y"), ability.reindex(ids_conf).rename("a")], axis=1).dropna()
        pr, ppr, _ = partial_corr(j["x"].to_numpy(), j["y"].to_numpy(), j[["a"]].to_numpy()) if len(j) > 10 else (np.nan, np.nan, 0)
        conf.append({"feature": c, "tier": "confirmatory", "rho": r, "p": pp, "p_spearman": p, "n": n,
                     "partial_rho_ability": pr, "partial_p": ppr,
                     "same_sign": bool(np.isfinite(r) and np.sign(r) == np.sign(rho_d))})
    conf = pd.DataFrame(conf)
    if not conf.empty:
        conf["q"] = bh_fdr(conf["p"].to_numpy(float))
    return pd.concat([exp, conf], ignore_index=True)
