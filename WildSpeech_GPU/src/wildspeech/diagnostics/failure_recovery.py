"""Failure populations and which modality recovers them.

define_populations   students flagged (from the BASELINE's OOF predictions and label-free metadata):
  large_error, unexpected_high (resid > q), unexpected_low (resid < 1-q), extreme_score (target tails),
  skill_imbalance (spread of the three screening thetas), missing_task (fewer tasks / missing theta),
  complexity_sensitive (strong accuracy-vs-complexity slope, CPX), phonetic_inconsistent (high SD of PHON
  error rates, if available), low_rate_recording, model_disagreement (SD across candidate OOFs),
  v1_unexplained (P12 failure clusters / is_failure, if available).
recovery_matrix      per population x model: baseline MAE - model MAE on the population with a paired student
                     bootstrap CI; recovered = CI > 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID
from ..validation.metrics import mean_oof


def define_populations(ctx, base_oof: pd.DataFrame, model_oofs: dict[str, pd.DataFrame], extra: pd.DataFrame | None,
                       q: float, extreme_q: float, disagreement_q: float) -> pd.DataFrame:
    b = ctx.bundle()
    m = mean_oof(base_oof)
    m.index = m.index.astype(str)
    resid = m["y_true"] - m["y_pred"]
    P = pd.DataFrame(index=m.index)
    P["large_error"] = resid.abs() >= resid.abs().quantile(q)
    P["unexpected_high"] = resid >= resid.quantile(q)
    P["unexpected_low"] = resid <= resid.quantile(1 - q)
    P["extreme_score"] = (m["y_true"] <= m["y_true"].quantile(extreme_q)) | (m["y_true"] >= m["y_true"].quantile(1 - extreme_q))
    st = b.train_students.drop_duplicates(ID).set_index(ID)
    st.index = st.index.astype(str)
    th = [c for c in b.theta_cols if c in st]
    if th:
        spread = st[th].std(axis=1)
        P["skill_imbalance"] = (spread >= spread.quantile(q)).reindex(P.index).fillna(False)
        P["missing_task"] = st[th].isna().any(axis=1).reindex(P.index).fillna(False)
    clips = b.train_clips
    ntask = clips.groupby(ID)["task"].nunique()
    ntask.index = ntask.index.astype(str)
    fewer = ntask.reindex(P.index).fillna(0) < len(b.tasks)
    P["missing_task"] = (P["missing_task"] | fewer) if "missing_task" in P else fewer
    if "original_sample_rate" in st:
        P["low_rate_recording"] = (st["original_sample_rate"].astype(float) < 16000).reindex(P.index).fillna(False)
    if extra is not None and not extra.empty:
        cpx = [c for c in extra.columns if c.startswith("CPX__") and "accuracy__cpx_slope" in c]
        if cpx:
            s = extra[cpx].abs().mean(axis=1).reindex(P.index)
            P["complexity_sensitive"] = (s >= s.quantile(q)).fillna(False)
        ph = [c for c in extra.columns if c.startswith("PHON__") and ("per" in c or "_er" in c) and c.endswith("__sd")]
        if ph:
            s = extra[ph].mean(axis=1).reindex(P.index)
            P["phonetic_inconsistent"] = (s >= s.quantile(q)).fillna(False)
    if len(model_oofs) >= 2:
        preds = pd.concat({k: mean_oof(v)["y_pred"].rename(k) for k, v in model_oofs.items()}, axis=1)
        preds.index = preds.index.astype(str)
        sd = preds.std(axis=1).reindex(P.index)
        P["model_disagreement"] = (sd >= sd.quantile(disagreement_q)).fillna(False)
    v1 = ctx.read_stage_table("P12", "student_errors_and_failure_clusters")
    if v1 is not None and "is_failure" in v1:
        f = v1.set_index(v1[ID].astype(str))["is_failure"].astype(bool)
        P["v1_unexplained"] = f.reindex(P.index).fillna(False)
    return P.astype(bool)


def recovery_matrix(P: pd.DataFrame, base_oof: pd.DataFrame, model_oofs: dict[str, pd.DataFrame], n_boot: int = 1000,
                    seed: int = 0) -> pd.DataFrame:
    base = mean_oof(base_oof)
    base.index = base.index.astype(str)
    rng = np.random.default_rng(seed)
    rows = []
    for name, oof in model_oofs.items():
        m = mean_oof(oof)
        m.index = m.index.astype(str)
        j = base.join(m["y_pred"].rename("p_new"), how="inner")
        e_b = (j["y_true"] - j["y_pred"]).abs()
        e_n = (j["y_true"] - j["p_new"]).abs()
        for pop in P.columns:
            ids = P.index[P[pop]].intersection(j.index)
            if len(ids) < 8:
                continue
            d = (e_b.loc[ids] - e_n.loc[ids]).to_numpy()
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
            lo, hi = np.quantile(boots, [0.025, 0.975])
            rows.append({"population": pop, "model": name, "n": int(len(ids)), "mae_base": float(e_b.loc[ids].mean()),
                         "mae_model": float(e_n.loc[ids].mean()), "mae_reduction": float(d.mean()), "ci_low": float(lo),
                         "ci_high": float(hi), "recovered": bool(lo > 0), "worsened": bool(hi < 0)})
    return pd.DataFrame(rows)


def population_profiles(P: pd.DataFrame, X: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Standardised mean difference of student features between each population and the rest."""
    rows = []
    Xs = X.reindex(P.index)
    num = Xs.select_dtypes(include=[np.number])
    sd = num.std(ddof=0).replace(0, np.nan)
    for pop in P.columns:
        inp = P[pop]
        if inp.sum() < 8:
            continue
        smd = ((num[inp].mean() - num[~inp].mean()) / sd).dropna()
        for f, v in smd.abs().sort_values(ascending=False).head(top).items():
            rows.append({"population": pop, "feature": f, "smd": float(smd[f]), "n": int(inp.sum())})
    return pd.DataFrame(rows)
