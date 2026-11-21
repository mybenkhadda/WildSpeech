"""Student-disjoint cross-validation schemes, metrics and CV runner.

Schemes:
* CV_A_REPEATED_STUDENT_KFOLD - repeated K-fold over students, stratified by target quantile
* CV_B_SCHOOL_GROUP_KFOLD    - grouped by school (stratified-group K-fold)
* CV_C_LEAVE_ONE_SCHOOL_OUT  - leave one school out
* CV_D_INSTRUMENT_STRATIFIED - stratified by instrument x target bin
* CV_E_CROSS_INSTRUMENT_TRANSFER - train on other instruments, validate on one

Every fold partitions students, so all clips of a student stay together
(modelling happens at student level). Schemes whose metadata is missing are
reported as unavailable instead of being silently replaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps

SCHEMES = ["CV_A_REPEATED_STUDENT_KFOLD", "CV_B_SCHOOL_GROUP_KFOLD", "CV_C_LEAVE_ONE_SCHOOL_OUT",
           "CV_D_INSTRUMENT_STRATIFIED", "CV_E_CROSS_INSTRUMENT_TRANSFER"]
GREATER_IS_BETTER = {"r2": True, "rmse": False, "mae": False, "spearman": True, "pearson": True}


def _ybins(y: pd.Series, q: int) -> np.ndarray:
    ranks = y.rank(method="first")
    return pd.qcut(ranks, min(q, max(2, len(y) // 10)), labels=False, duplicates="drop").to_numpy()


def make_cv_assignments(meta: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """meta: index student_id with columns target, school_id, instrument."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    seed = int(cfg["seed"])
    k = int(cfg["cv_folds"])
    meta = meta[meta["target"].notna()]
    ids = meta.index.to_numpy()
    yb = _ybins(meta["target"], int(cfg["target_bins_for_stratification"]))
    rows, status = [], {}
    for r in range(int(cfg["cv_repeats"])):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for f, (_, va) in enumerate(skf.split(ids, yb)):
            rows += [("CV_A_REPEATED_STUDENT_KFOLD", r, f, sid) for sid in ids[va]]
    status["CV_A_REPEATED_STUDENT_KFOLD"] = {"status": "OK", "repeats": int(cfg["cv_repeats"]), "folds": k}
    schools = meta["school_id"] if "school_id" in meta.columns else pd.Series(np.nan, index=meta.index)
    n_schools = int(schools.nunique())
    if n_schools >= 2 and schools.notna().all():
        kk = min(k, n_schools)
        sgk = StratifiedGroupKFold(n_splits=kk, shuffle=True, random_state=seed)
        for f, (_, va) in enumerate(sgk.split(ids, yb, groups=schools.to_numpy())):
            rows += [("CV_B_SCHOOL_GROUP_KFOLD", 0, f, sid) for sid in ids[va]]
        status["CV_B_SCHOOL_GROUP_KFOLD"] = {"status": "OK", "folds": kk, "n_schools": n_schools}
    else:
        status["CV_B_SCHOOL_GROUP_KFOLD"] = {"status": "UNAVAILABLE_MISSING_METADATA",
                                             "reason": f"{n_schools} schools; school_id missing for {int(schools.isna().sum())} students"}
    if 3 <= n_schools <= int(cfg["max_schools_for_loso"]) and schools.notna().all():
        codes = {s: i for i, s in enumerate(sorted(schools.unique(), key=str))}
        rows += [("CV_C_LEAVE_ONE_SCHOOL_OUT", 0, codes[schools[sid]], sid) for sid in ids]
        status["CV_C_LEAVE_ONE_SCHOOL_OUT"] = {"status": "OK", "folds": n_schools}
    else:
        status["CV_C_LEAVE_ONE_SCHOOL_OUT"] = {"status": "UNAVAILABLE", "reason": f"{n_schools} schools (need 3..{cfg['max_schools_for_loso']})"}
    inst = meta["instrument"] if "instrument" in meta.columns else pd.Series(np.nan, index=meta.index)
    n_inst = int(inst.nunique())
    if n_inst >= 1 and inst.notna().all():
        label = inst.astype(str) + "_" + pd.Series(yb, index=meta.index).astype(str)
        counts = label.value_counts()
        label = label.where(label.map(counts) >= k, inst.astype(str))
        if label.value_counts().min() >= k:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + 1000)
            for f, (_, va) in enumerate(skf.split(ids, label.to_numpy())):
                rows += [("CV_D_INSTRUMENT_STRATIFIED", 0, f, sid) for sid in ids[va]]
            status["CV_D_INSTRUMENT_STRATIFIED"] = {"status": "OK", "folds": k, "n_instruments": n_inst}
        else:
            status["CV_D_INSTRUMENT_STRATIFIED"] = {"status": "UNAVAILABLE", "reason": "instrument strata smaller than cv_folds"}
    else:
        status["CV_D_INSTRUMENT_STRATIFIED"] = {"status": "UNAVAILABLE_MISSING_METADATA", "reason": "instrument missing"}
    if n_inst >= 2 and inst.notna().all():
        codes = {s: i for i, s in enumerate(sorted(inst.unique(), key=str))}
        rows += [("CV_E_CROSS_INSTRUMENT_TRANSFER", 0, codes[inst[sid]], sid) for sid in ids]
        status["CV_E_CROSS_INSTRUMENT_TRANSFER"] = {"status": "OK", "folds": n_inst, "instrument_codes": {str(k_): v for k_, v in codes.items()},
                                                    "note": "Targets of different instruments may not share a scale."}
    else:
        status["CV_E_CROSS_INSTRUMENT_TRANSFER"] = {"status": "UNAVAILABLE", "reason": f"{n_inst} instruments"}
    assign = pd.DataFrame(rows, columns=["scheme", "repeat", "fold", "student_id"])
    assert_student_disjoint(assign)
    return assign, status


def assert_student_disjoint(assign: pd.DataFrame) -> None:
    dup = assign.duplicated(["scheme", "repeat", "student_id"])
    if dup.any():
        raise AssertionError("a student appears in more than one validation fold within a scheme repeat")


def available_schemes(assign: pd.DataFrame) -> list:
    return [s for s in SCHEMES if s in set(assign["scheme"])]


def iter_folds(assign: pd.DataFrame, scheme: str, repeats: list | None = None):
    sub = assign[assign["scheme"] == scheme]
    for rep in sorted(sub["repeat"].unique()):
        if repeats is not None and rep not in repeats:
            continue
        rs = sub[sub["repeat"] == rep]
        all_ids = rs["student_id"].to_numpy()
        for fold in sorted(rs["fold"].unique()):
            val = rs.loc[rs["fold"] == fold, "student_id"].to_numpy()
            train = np.setdiff1d(all_ids, val)
            if len(np.intersect1d(train, val)):
                raise AssertionError("train/validation overlap")
            yield int(rep), int(fold), train, val


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n = len(y)
    if n < 3:
        return {"n": n, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "medae": np.nan, "pearson": np.nan, "spearman": np.nan}
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": n, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "rmse": float(np.sqrt(ss_res / n)),
            "mae": float(np.abs(y - p).mean()), "medae": float(np.median(np.abs(y - p))),
            "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan,
            "spearman": float(sps.spearmanr(y, p)[0]) if np.std(p) > 0 else np.nan}


def official_score(y, p, cfg: dict) -> float:
    return regression_metrics(y, p)[cfg.get("official_metric", "r2")]


@dataclass
class CVResult:
    scheme: str
    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_cv(y: pd.Series, assign: pd.DataFrame, scheme: str, fit_predict: Callable, cfg: dict,
           repeats: list | None = None, groups: pd.Series | None = None, label: str = "") -> CVResult:
    """fit_predict(train_ids, val_ids, fold_key) -> predictions for val_ids (original target scale)."""
    oof_rows, fm = [], []
    for rep, fold, tr, va in iter_folds(assign, scheme, repeats):
        tr = np.array([s for s in tr if s in y.index])
        va = np.array([s for s in va if s in y.index])
        if len(va) == 0 or len(tr) < 5:
            continue
        pred = np.asarray(fit_predict(tr, va, (scheme, rep, fold)), dtype=float)
        yv = y.loc[va].to_numpy()
        oof_rows.append(pd.DataFrame({"student_id": va, "repeat": rep, "fold": fold, "y": yv, "pred": pred}))
        m = regression_metrics(yv, pred)
        m.update({"repeat": rep, "fold": fold, "n_train": len(tr), "official": official_score(yv, pred, cfg)})
        fm.append(m)
    oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(columns=["student_id", "repeat", "fold", "y", "pred"])
    fold_metrics = pd.DataFrame(fm)
    res = CVResult(scheme, oof, fold_metrics)
    res.summary = summarize_cv(oof, fold_metrics, cfg, groups)
    res.summary["label"] = label
    return res


def pooled_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    return oof.groupby("student_id").agg(y=("y", "first"), pred=("pred", "mean"), n_preds=("pred", "count"))


def summarize_cv(oof: pd.DataFrame, fold_metrics: pd.DataFrame, cfg: dict, groups: pd.Series | None = None) -> dict:
    from .stats_utils import bootstrap_ci

    if oof.empty:
        return {"status": "NO_FOLDS"}
    pooled = pooled_predictions(oof)
    metric = cfg.get("official_metric", "r2")
    out = {"status": "OK", "n_students": int(len(pooled)), "n_folds": int(len(fold_metrics))}
    pm = regression_metrics(pooled["y"], pooled["pred"])
    out.update({f"pooled_{k}": v for k, v in pm.items() if k != "n"})
    out["pooled_official"] = pm[metric]
    for k in ("r2", "rmse", "mae", "spearman", "official"):
        if k in fold_metrics.columns:
            out[f"mean_fold_{k}"] = float(fold_metrics[k].mean())
            out[f"std_fold_{k}"] = float(fold_metrics[k].std(ddof=1)) if len(fold_metrics) > 1 else np.nan
    yv, pv = pooled["y"].to_numpy(), pooled["pred"].to_numpy()
    n_boot = int(cfg["bootstrap_iterations"])
    for k in ("r2", "rmse"):
        ci = bootstrap_ci(lambda a, b, k=k: regression_metrics(a, b)[k], yv, pv, n_boot=n_boot, seed=int(cfg["seed"]))
        out[f"pooled_{k}_ci_lower"], out[f"pooled_{k}_ci_upper"] = ci["lower"], ci["upper"]
    if groups is not None:
        g = groups.reindex(pooled.index)
        gm = []
        for level, idx in g.dropna().groupby(g.dropna()).groups.items():
            if len(idx) >= int(cfg["min_group_size"]):
                m = regression_metrics(pooled.loc[idx, "y"], pooled.loc[idx, "pred"])
                gm.append(m["r2"])
        out["group_r2_median"] = float(np.median(gm)) if gm else np.nan
        out["group_r2_min"] = float(np.min(gm)) if gm else np.nan
        out["n_groups_evaluated"] = len(gm)
    return out


def quantile_performance(pooled: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    bins = pd.qcut(pooled["y"].rank(method="first"), q, labels=[f"Q{i + 1}" for i in range(q)])
    rows = []
    for b, g in pooled.groupby(bins, observed=True):
        err = g["pred"] - g["y"]
        rows.append({"target_quantile": b, "n": len(g), "rmse": float(np.sqrt((err ** 2).mean())),
                     "mae": float(err.abs().mean()), "mean_signed_error": float(err.mean())})
    return pd.DataFrame(rows)


def group_performance(pooled: pd.DataFrame, groups: pd.Series, min_n: int) -> pd.DataFrame:
    g = groups.reindex(pooled.index)
    rows = []
    for level, idx in g.dropna().groupby(g.dropna()).groups.items():
        m = regression_metrics(pooled.loc[idx, "y"], pooled.loc[idx, "pred"])
        m.update({"group": level, "reliable": len(idx) >= min_n})
        rows.append(m)
    return pd.DataFrame(rows)


def paired_bootstrap_delta(y, p_new, p_ref, cfg: dict, metric: str = "r2") -> dict:
    """Bootstrap CI of metric(new) - metric(ref) on the same students."""
    from .stats_utils import bootstrap_ci

    y, a, b = (np.asarray(v, dtype=float) for v in (y, p_new, p_ref))
    sign = 1.0 if GREATER_IS_BETTER.get(metric, True) else -1.0
    ci = bootstrap_ci(lambda yy, aa, bb: sign * (regression_metrics(yy, aa)[metric] - regression_metrics(yy, bb)[metric]),
                      y, a, b, n_boot=int(cfg["bootstrap_iterations"]), seed=int(cfg["seed"]))
    return {"delta": ci["estimate"], "ci_lower": ci["lower"], "ci_upper": ci["upper"],
            "p_le_zero": np.nan if not np.isfinite(ci["se"]) or ci["se"] == 0 else float(sps.norm.cdf(-ci["estimate"] / ci["se"]))}
