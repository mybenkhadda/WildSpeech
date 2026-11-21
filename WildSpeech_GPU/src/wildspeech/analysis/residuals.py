"""P07 — residual discovery: what explains screening-only prediction errors?

1. Screening-only OOF predictions (repeated CV) -> residual per student (mean over repeats).
2. Associations of cross-fitted student features with the residual: Spearman, rank-partial controlling
   for screening + instrument, bootstrap CI, BH-FDR, permutation p for top hits.
3. Split-sample replication: discovery on half the students, replication on the other half, repeated.
4. Matched comparisons: students with similar screening profiles (caliper on a screening score) but in
   opposite residual tails, paired differences with Wilcoxon tests and effect sizes.
5. Adjusted associations: OLS outcome ~ screening + feature with HC3 robust SE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..features.builder import family_of
from ..stats import bh_fdr, bootstrap_ci, cohens_d, ols_hc3, partial_corr, permutation_pvalue_corr, spearman


def candidate_features(X: pd.DataFrame, exclude=("SCREENING", "RECORDING"), max_missing: float = 0.4) -> list[str]:
    cols = [c for c in X.columns if family_of(c) not in exclude]
    ok = X[cols].isna().mean() <= max_missing
    std = X[cols].std()
    return [c for c in cols if ok[c] and std[c] > 1e-12]


def screening_controls(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in X.columns if family_of(c) == "SCREENING" and not c.endswith("_missing")]
    Z = X[cols].copy()
    return Z.fillna(Z.median())


def residual_associations(X: pd.DataFrame, resid: pd.Series, y: pd.Series, feats: list[str], n_boot: int, seed: int,
                          extra_controls: pd.DataFrame | None = None) -> pd.DataFrame:
    Z = screening_controls(X)
    if extra_controls is not None:
        Z = Z.join(extra_controls.fillna(extra_controls.median()), how="left")
    rows = []
    for c in feats:
        x = X[c]
        r, p, n = spearman(x, resid)
        pr, pp, _ = partial_corr(x.values, y.reindex(X.index).values, Z.values)
        rows.append({"feature": c, "family": family_of(c), "rho_resid": r, "p_resid": p, "n": n,
                     "partial_rho_outcome": pr, "p_partial": pp})
    out = pd.DataFrame(rows)
    out["q_resid"] = bh_fdr(out["p_resid"])
    out["q_partial"] = bh_fdr(out["p_partial"])
    out = out.sort_values("p_resid").reset_index(drop=True)
    top = out.head(min(len(out), 60))
    cis = []
    for c in top["feature"]:
        ok = X[c].notna() & resid.notna()
        _, lo, hi = bootstrap_ci(lambda a, b: spearman(a, b)[0], X.loc[ok, c].to_numpy(), resid[ok].to_numpy(), n_boot=n_boot, seed=seed)
        cis.append((lo, hi))
    out.loc[top.index, "rho_ci_low"] = [a for a, _ in cis]
    out.loc[top.index, "rho_ci_high"] = [b for _, b in cis]
    return out


def permutation_check(X: pd.DataFrame, resid: pd.Series, feats: list[str], n_perm: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame([{"feature": c, "perm_p": permutation_pvalue_corr(X[c].values, resid.values, n_perm, seed)} for c in feats])


def split_replication(X: pd.DataFrame, resid: pd.Series, feats: list[str], repeats: int, top_k: int, alpha: float, seed: int) -> pd.DataFrame:
    """Fraction of repeats in which a discovery-half top-k hit replicates (same sign, p<alpha) in the other half."""
    rng = np.random.default_rng(seed)
    ids = np.array(X.index)
    hits = {c: [0, 0] for c in feats}
    for r in range(repeats):
        perm = rng.permutation(ids)
        a, b = perm[: len(ids) // 2], perm[len(ids) // 2:]
        disc = []
        for c in feats:
            rho, p, _ = spearman(X.loc[a, c], resid.loc[a])
            disc.append((c, rho, p))
        disc = sorted([d for d in disc if np.isfinite(d[2])], key=lambda t: t[2])[:top_k]
        for c, rho, _ in disc:
            rho2, p2, _ = spearman(X.loc[b, c], resid.loc[b])
            hits[c][0] += 1
            hits[c][1] += int(np.isfinite(rho2) and np.sign(rho2) == np.sign(rho) and p2 < alpha)
    rows = [{"feature": c, "times_discovered": v[0], "times_replicated": v[1],
             "replication_rate": v[1] / v[0] if v[0] else np.nan} for c, v in hits.items() if v[0]]
    return pd.DataFrame(rows).sort_values(["times_replicated", "replication_rate"], ascending=False)


def matched_comparison(X: pd.DataFrame, resid: pd.Series, feats: list[str], quantiles: tuple[float, float], caliper_sd: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair each 'over-achiever' (high residual) with the closest 'under-achiever' on the screening profile."""
    Z = screening_controls(X)
    Zs = ((Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)).fillna(0)
    lo_t, hi_t = resid.quantile(quantiles[0]), resid.quantile(quantiles[1])
    hi_ids, lo_ids = list(resid[resid >= hi_t].index), list(resid[resid <= lo_t].index)
    used, pairs = set(), []
    caliper = caliper_sd * np.sqrt(Zs.shape[1])
    for h in hi_ids:
        cands = [l for l in lo_ids if l not in used]
        if not cands:
            break
        d = np.sqrt(((Zs.loc[cands] - Zs.loc[h]) ** 2).sum(1))
        best = d.idxmin()
        if d[best] <= caliper:
            pairs.append((h, best, float(d[best])))
            used.add(best)
    pairs_df = pd.DataFrame(pairs, columns=["high_resid_id", "low_resid_id", "distance"])
    rows = []
    if len(pairs_df) >= 8:
        for c in feats:
            a, b = X.loc[pairs_df["high_resid_id"], c].to_numpy(float), X.loc[pairs_df["low_resid_id"], c].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 8 or np.allclose(a[ok], b[ok]):
                continue
            p = wilcoxon(a[ok], b[ok]).pvalue
            rows.append({"feature": c, "family": family_of(c), "n_pairs": int(ok.sum()), "mean_diff": float((a[ok] - b[ok]).mean()),
                         "cohens_d": cohens_d(a[ok], b[ok]), "p": float(p)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["q"] = bh_fdr(res["p"])
        res = res.sort_values("p")
    return pairs_df, res


def adjusted_associations(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> pd.DataFrame:
    Z = screening_controls(X)
    rows = []
    for c in feats:
        x = X[c]
        xs = (x - x.mean()) / (x.std() or 1.0)
        tab = ols_hc3(y.reindex(X.index).values, np.column_stack([Z.values, xs.values]), list(Z.columns) + [c])
        r = tab.iloc[-1]
        rows.append({"feature": c, "family": family_of(c), "beta_per_sd": r["coef"], "se_hc3": r["se_hc3"], "p": r["p"], "n": r["n"]})
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"])
    return out.sort_values("p")


def family_summary(assoc: pd.DataFrame, alpha: float) -> pd.DataFrame:
    g = assoc.groupby("family")
    return pd.DataFrame({"n_features": g.size(), "n_q_resid_sig": g["q_resid"].apply(lambda s: int((s < alpha).sum())),
                         "max_abs_rho": g["rho_resid"].apply(lambda s: float(s.abs().max())),
                         "median_abs_rho": g["rho_resid"].apply(lambda s: float(s.abs().median()))}).sort_values("max_abs_rho", ascending=False).reset_index()
