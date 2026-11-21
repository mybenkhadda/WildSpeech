"""P08 — are the discovered signals genuine?

* School effects: eta^2 / ICC of features by school; school-demeaned residual associations.
* Recording artifacts: correlation of features with quality/recording covariates; artifact-adjusted associations.
* Anonymisation / perturbation sensitivity: recompute clip features after controlled perturbations (pitch shift,
  noise, low-pass, gain) and measure stability (Spearman across clips, relative change).
* Domain generalisation: R2 under school K-fold, LOSO, instrument and sample-rate transfer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.builder import family_of
from ..stats import bh_fdr, eta_squared, icc1, partial_corr, spearman

ARTIFACT_PATTERNS = ("QUALITY__all__snr_db", "QUALITY__all__noise_floor_db", "QUALITY__all__bandlimit_hz",
                     "QUALITY__all__clipping_ratio", "RECORDING__orig_sr_low")


def school_effects(X: pd.DataFrame, schools: pd.Series, feats: list[str], flag: float) -> pd.DataFrame:
    s = schools.reindex(X.index)
    rows = []
    for c in feats:
        e, p = eta_squared(X[c], s)
        rows.append({"feature": c, "family": family_of(c), "school_eta2": e, "kruskal_p": p, "icc1": icc1(X[c], s)})
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["kruskal_p"])
    out["school_flag"] = out["school_eta2"] > flag
    return out.sort_values("school_eta2", ascending=False)


def school_adjusted_associations(X: pd.DataFrame, resid: pd.Series, schools: pd.Series, feats: list[str]) -> pd.DataFrame:
    s = schools.reindex(X.index)
    r_dm = resid - resid.groupby(s).transform("mean")
    rows = []
    for c in feats:
        x_dm = X[c] - X[c].groupby(s).transform("mean")
        r0, p0, _ = spearman(X[c], resid)
        r1, p1, n = spearman(x_dm, r_dm)
        rows.append({"feature": c, "family": family_of(c), "rho_raw": r0, "p_raw": p0, "rho_school_demeaned": r1,
                     "p_school_demeaned": p1, "n": n, "sign_kept": bool(np.sign(r0) == np.sign(r1)) if np.isfinite(r0) and np.isfinite(r1) else False})
    out = pd.DataFrame(rows)
    out["q_raw"] = bh_fdr(out["p_raw"])
    out["q_school_demeaned"] = bh_fdr(out["p_school_demeaned"])
    return out.sort_values("p_school_demeaned")


def artifact_covariates(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ARTIFACT_PATTERNS if c in X.columns]
    cols += [c for c in X.columns if c.startswith("QUALITY__all__") and c.endswith("__mean") and c not in cols][:6]
    return X[cols]


def artifact_correlations(X: pd.DataFrame, feats: list[str], flag: float) -> pd.DataFrame:
    A = artifact_covariates(X)
    rows = []
    for c in feats:
        if c in A.columns:
            continue
        rs = {a: spearman(X[c], A[a])[0] for a in A.columns}
        worst = max(rs, key=lambda k: abs(rs[k]) if np.isfinite(rs[k]) else -1) if rs else None
        rows.append({"feature": c, "family": family_of(c), "max_abs_artifact_rho": abs(rs[worst]) if worst and np.isfinite(rs[worst]) else np.nan,
                     "worst_artifact": worst, **{f"rho__{a}": v for a, v in rs.items()}})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["artifact_flag"] = out["max_abs_artifact_rho"] > flag
    return out.sort_values("max_abs_artifact_rho", ascending=False) if not out.empty else out


def artifact_adjusted_associations(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> pd.DataFrame:
    """Rank-partial correlation with the outcome controlling screening + artifact covariates."""
    from .residuals import screening_controls

    Z = screening_controls(X).join(artifact_covariates(X), how="left")
    Z = Z.loc[:, ~Z.columns.duplicated()].fillna(Z.median())
    rows = []
    for c in feats:
        if c in Z.columns:
            continue
        r, p, n = partial_corr(X[c].values, y.reindex(X.index).values, Z.values)
        rows.append({"feature": c, "family": family_of(c), "partial_rho_adj": r, "p_adj": p, "n": n})
    out = pd.DataFrame(rows)
    out["q_adj"] = bh_fdr(out["p_adj"])
    return out.sort_values("p_adj")


def perturbation_stability(orig: pd.DataFrame, pert: dict[str, pd.DataFrame], flag: float) -> pd.DataFrame:
    """orig / pert[name]: clip x feature frames with identical index. Stability = Spearman(orig, perturbed)."""
    rows = []
    for name, P in pert.items():
        for c in orig.columns:
            if c not in P:
                continue
            r, _, n = spearman(orig[c], P[c])
            rel = (P[c] - orig[c]).abs() / (orig[c].abs() + orig[c].std() * 0.1 + 1e-12)
            rows.append({"perturbation": name, "feature": c, "family": c.split("::")[0], "stability_rho": r,
                         "median_rel_change": float(rel.median()), "n": n})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["unstable"] = out["stability_rho"] < flag
    return out


def stability_summary(stab: pd.DataFrame) -> pd.DataFrame:
    if stab.empty:
        return stab
    piv = stab.pivot_table(index="feature", columns="perturbation", values="stability_rho")
    piv["min_stability"] = piv.min(axis=1)
    piv["family"] = [c.split("::")[0] for c in piv.index]
    return piv.sort_values("min_stability").reset_index()
