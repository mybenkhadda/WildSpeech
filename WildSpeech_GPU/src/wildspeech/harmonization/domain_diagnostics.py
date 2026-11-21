"""Recording-domain diagnostics for harmonization variants (all student-grouped, fold-fitted).

sample_rate_separability   clip-level logistic AUROC predicting low-original-rate recordings from (item-centred)
                           features, GroupKFold by student, + student-block bootstrap CI. Lower = better.
school_separability        student-level multinomial logistic, balanced accuracy vs chance -> separability index.
distribution_shift         per feature x task standardised mean difference, KS and Wasserstein distance between
                           original-rate groups (item-centred values).
correctness_discrimination clip-level logistic AUROC for official correctness (+ univariate AUROC per feature).
robustness_ranking         per-feature combination of stability, domain shift, school effect and correctness signal.
Every residualizer/scaler/classifier is fitted on the training students of each split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import ID
from ..diagnostics.scientific_validation import auc_columns, auc_fast, block_bootstrap
from ..features.conditional_aggregation import ItemConditionedResidualizer
from ..stats import eta_squared


def _group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    g = np.asarray(groups).astype(str)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = {u: i for i, u in enumerate(rng.permutation(uniq))}
    gi = np.array([perm[x] for x in g])
    return list(GroupKFold(n_splits=min(n_splits, len(uniq))).split(np.zeros(len(g)), groups=gi))


def _logit_pipeline(C: float = 0.5, multinomial: bool = False):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def grouped_oof_scores(c: pd.DataFrame, F: np.ndarray, target: np.ndarray, controls: tuple, n_splits: int = 5,
                       seed: int = 0, residualize: bool = True) -> np.ndarray:
    """OOF probability of `target` per clip; residualizer + model fitted on training students only."""
    p = np.full(len(c), np.nan)
    t = np.asarray(target, float)
    for tr, va in _group_folds(c[ID].to_numpy(), n_splits, seed):
        trm = tr[np.isfinite(t[tr])]
        if len(np.unique(t[trm])) < 2:
            continue
        if residualize:
            res = ItemConditionedResidualizer(controls).fit(c.iloc[trm], F[trm])
            Xtr, Xva = res.transform(c.iloc[trm], F[trm]), res.transform(c.iloc[va], F[va])
        else:
            Xtr, Xva = F[trm], F[va]
        keep = np.isfinite(Xtr).mean(0) > 0.5
        if not keep.any():
            continue
        m = _logit_pipeline().fit(Xtr[:, keep], t[trm].astype(int))
        p[va] = m.predict_proba(Xva[:, keep])[:, 1]
    return p


def sample_rate_separability(c: pd.DataFrame, F: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """Two views: item-centred only (does the representation carry the domain?) and after domain
    residualization fitted out-of-sample (does harmonization-by-regression generalise to new students?)."""
    y = c["orig_sr_low"].to_numpy(float)
    out = {}
    for tag, controls in (("item_centred", ("item",)), ("domain_residualized", ("item", "domain"))):
        p = grouped_oof_scores(c, F, y, controls, seed=seed)
        d = pd.DataFrame({ID: c[ID].to_numpy(), "y": y, "p": p}).dropna()
        est, lo, hi = block_bootstrap(lambda x: auc_fast(x["y"], x["p"]), d, ID, n_boot, seed)
        out[f"sr_auc_{tag}"], out[f"sr_auc_{tag}_lo"], out[f"sr_auc_{tag}_hi"] = est, lo, hi
    return out


def correctness_discrimination(c: pd.DataFrame, F: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    y = c["item_score"].to_numpy(float)
    p = grouped_oof_scores(c, F, y, ("item", "domain", "duration"), seed=seed)
    d = pd.DataFrame({ID: c[ID].to_numpy(), "y": y, "p": p}).dropna()
    est, lo, hi = block_bootstrap(lambda x: auc_fast(x["y"], x["p"]), d, ID, n_boot, seed)
    return {"correct_auc": est, "correct_auc_lo": lo, "correct_auc_hi": hi, "n_clips": int(len(d))}


def school_separability(Xs: pd.DataFrame, school: pd.Series, min_size: int = 8, seed: int = 0) -> dict:
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    s = school.reindex(Xs.index).astype(str)
    vc = s.value_counts()
    keep = s.isin(vc[vc >= min_size].index)
    X, s = Xs[keep].to_numpy(float), s[keep].to_numpy()
    if len(np.unique(s)) < 2 or len(s) < 20:
        return {"school_bacc": np.nan, "school_chance": np.nan, "school_sep_index": np.nan, "n_schools": int(len(np.unique(s)))}
    n_splits = int(min(5, pd.Series(s).value_counts().min()))
    pred = np.empty(len(s), dtype=object)
    for tr, va in StratifiedKFold(max(2, n_splits), shuffle=True, random_state=seed).split(X, s):
        m = _logit_pipeline(C=0.2).fit(X[tr], s[tr])
        pred[va] = m.predict(X[va])
    bacc = float(balanced_accuracy_score(s, pred))
    chance = 1.0 / len(np.unique(s))
    return {"school_bacc": bacc, "school_chance": chance, "school_sep_index": (bacc - chance) / (1 - chance),
            "n_schools": int(len(np.unique(s)))}


def distribution_shift(c: pd.DataFrame, Z: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Z: item-centred values. Per feature x task: SMD, KS, Wasserstein between original-rate groups."""
    low = c["orig_sr_low"].to_numpy(float) == 1
    rows = []
    for t in sorted(c["task"].unique()):
        mt = (c["task"] == t).to_numpy()
        for j, f in enumerate(names):
            a, b = Z[mt & low, j], Z[mt & ~low, j]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                continue
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            rows.append({"task": t, "feature": f, "smd": float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan,
                         "ks": float(ss.ks_2samp(a, b).statistic), "wasserstein": float(ss.wasserstein_distance(a, b)),
                         "n_low": len(a), "n_16k": len(b)})
    return pd.DataFrame(rows)


def feature_school_eta2(Xs: pd.DataFrame, school: pd.Series) -> pd.Series:
    s = school.reindex(Xs.index)
    return pd.Series({c: eta_squared(Xs[c].to_numpy(float), s.to_numpy())[0] for c in Xs.columns})


def univariate_correctness_auc(c: pd.DataFrame, Z: np.ndarray, names: list[str]) -> pd.Series:
    y = c["item_score"].to_numpy(float)
    m = np.isfinite(y)
    a = auc_columns(y[m], Z[m])
    return pd.Series(np.abs(a - 0.5) * 2, index=names)    # 0 = no signal, 1 = perfect separation


def robustness_ranking(shift: pd.DataFrame, stability: pd.Series, corr_signal: pd.Series, school_eta2: pd.Series | None) -> pd.DataFrame:
    """Rank-average of: stability (high), |SMD| (low), school eta2 (low), correctness signal (high)."""
    feats = sorted(set(shift["feature"]) if not shift.empty else set(corr_signal.index))
    d = pd.DataFrame(index=feats)
    d["abs_smd"] = shift.groupby("feature")["smd"].apply(lambda s: float(np.nanmean(np.abs(s)))) if not shift.empty else np.nan
    d["stability"] = stability.reindex(feats)
    d["correct_signal"] = corr_signal.reindex(feats)
    if school_eta2 is not None and not school_eta2.empty:
        d["school_eta2"] = school_eta2.reindex(feats)
    ranks = []
    if d["stability"].notna().any():
        ranks.append(d["stability"].rank(pct=True))
    if d["abs_smd"].notna().any():
        ranks.append(1 - d["abs_smd"].rank(pct=True))
    if d["correct_signal"].notna().any():
        ranks.append(d["correct_signal"].rank(pct=True))
    if "school_eta2" in d and d["school_eta2"].notna().any():
        ranks.append(1 - d["school_eta2"].rank(pct=True))
    d["robustness_score"] = pd.concat(ranks, axis=1).mean(axis=1) if ranks else np.nan
    return d.sort_values("robustness_score", ascending=False).reset_index().rename(columns={"index": "feature"})


def ltas_by_domain(c: pd.DataFrame, mel_table: pd.DataFrame, prefix: str = "MEL::melband") -> pd.DataFrame:
    """Long-term average mel-band level per original-rate group (from per-clip mel band means)."""
    cols = sorted([x for x in mel_table.columns if x.startswith(prefix) and x.endswith("_mean")],
                  key=lambda s: int(s.replace(prefix, "").split("_")[0]))
    if not cols:
        return pd.DataFrame()
    j = c[["filename", "orig_sr_low", "task"]].merge(mel_table[["filename"] + cols], on="filename", how="inner")
    out = j.groupby("orig_sr_low")[cols].mean().T
    out.index = [int(s.replace(prefix, "").split("_")[0]) for s in out.index]
    out.index.name = "mel_band"
    out.columns = ["16k" if v == 0 else "low_rate" for v in out.columns]
    return out.reset_index()
