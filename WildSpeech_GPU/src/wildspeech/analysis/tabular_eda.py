"""P01 — tabular EDA: targets, screening scores, items, IRT, adaptive-assessment behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID, TARGET
from ..irt import IRTModel
from ..stats import bh_fdr, bootstrap_ci, eta_squared, icc1, spearman


def target_summary(train: pd.DataFrame) -> pd.DataFrame:
    y = train[TARGET]
    rows = [{"group_col": "all", "group": "all", "n": len(y), "mean": y.mean(), "sd": y.std(), "median": y.median(),
             "p05": y.quantile(.05), "p95": y.quantile(.95), "skew": y.skew(), "kurtosis": y.kurt(), "min": y.min(), "max": y.max()}]
    for col in ("instrument", "original_sample_rate", "school_id"):
        if col in train:
            for g, d in train.groupby(col):
                rows.append({"group_col": col, "group": str(g), "n": len(d), "mean": d[TARGET].mean(), "sd": d[TARGET].std(),
                             "median": d[TARGET].median(), "p05": d[TARGET].quantile(.05), "p95": d[TARGET].quantile(.95),
                             "skew": d[TARGET].skew(), "kurtosis": d[TARGET].kurt(), "min": d[TARGET].min(), "max": d[TARGET].max()})
    return pd.DataFrame(rows)


def target_group_effects(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("instrument", "original_sample_rate", "school_id"):
        if col in train:
            e, p = eta_squared(train[TARGET], train[col])
            rows.append({"factor": col, "eta2": e, "kruskal_p": p, "icc1": icc1(train[TARGET], train[col]),
                         "n_groups": train[col].nunique()})
    return pd.DataFrame(rows)


def screening_associations(train: pd.DataFrame, theta_cols: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for c in theta_cols + (["theta_mean"] if theta_cols else []):
        x = train[theta_cols].mean(axis=1) if c == "theta_mean" else train[c]
        for scope, d in [("all", train)] + ([(f"instrument={g}", dd) for g, dd in train.groupby("instrument")] if "instrument" in train else []):
            xs = x.loc[d.index]
            r, p, n = spearman(xs, d[TARGET])
            ok = xs.notna() & d[TARGET].notna()
            est, lo, hi = bootstrap_ci(lambda a, b: spearman(a, b)[0], xs[ok].to_numpy(), d.loc[ok, TARGET].to_numpy(),
                                       n_boot=n_boot, seed=seed) if ok.sum() > 10 else (r, np.nan, np.nan)
            rows.append({"screening": c, "scope": scope, "spearman": r, "ci_low": lo, "ci_high": hi, "p": p, "n": n,
                         "missing_frac": float(x.loc[d.index].isna().mean())})
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"])
    return out


def missingness_patterns(students: pd.DataFrame, theta_cols: list[str]) -> pd.DataFrame:
    m = students[theta_cols].isna()
    pat = m.apply(lambda r: "".join("M" if v else "o" for v in r), axis=1)
    df = pd.DataFrame({"pattern": pat, "instrument": students.get("instrument", "na"), "split": students["split"]})
    return df.groupby(["split", "instrument", "pattern"]).size().rename("n").reset_index()


def clip_behaviour(clips: pd.DataFrame) -> pd.DataFrame:
    g = clips.groupby([ID, "task"])
    return pd.DataFrame({"n_items": g.size(), "n_scored": g["is_scored"].sum(), "accuracy": g["item_score"].mean(),
                         "mean_duration": g["duration_sec"].mean(), "total_duration": g["duration_sec"].sum()}).reset_index()


def score_status_table(train_clips: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Train-only descriptive: how unscored statuses relate to outcome (never a model feature)."""
    if "score_status" not in train_clips:
        return pd.DataFrame()
    st = train_clips.groupby([ID, "score_status"]).size().unstack(fill_value=0)
    st = st.join(train.set_index(ID)[TARGET], how="inner")
    rows = []
    for c in st.columns.drop(TARGET):
        r, p, n = spearman(st[c], st[TARGET])
        rows.append({"score_status": c, "clips": int(st[c].sum()), "students_with_any": int((st[c] > 0).sum()),
                     "spearman_count_vs_outcome": r, "p": p})
    return pd.DataFrame(rows)


def item_table(train_clips: pd.DataFrame, train: pd.DataFrame, text_table: pd.DataFrame | None) -> pd.DataFrame:
    c = train_clips.merge(train[[ID, TARGET]], on=ID, how="left")
    sc = c[c["item_score"].notna()]
    g = sc.groupby("item_id")
    items = pd.DataFrame({"task": g["task"].first(), "expected_text": g["expected_text"].first(), "n": g.size(),
                          "p_correct": g["item_score"].mean(), "mean_duration": g["duration_sec"].mean(),
                          "mean_outcome_of_takers": g[TARGET].mean()})
    exposure = c.groupby("item_id")[ID].nunique().rename("n_students")
    items = items.join(exposure)
    # point-biserial (item score vs outcome) among takers
    pb = sc.groupby("item_id").apply(lambda d: spearman(d["item_score"], d[TARGET])[0] if len(d) >= 10 else np.nan)
    items["discrimination_vs_outcome"] = pb
    if text_table is not None and not text_table.empty:
        items = items.join(text_table.set_index("item_id"), how="left")
    return items.reset_index()


def irt_analysis(train_clips: pd.DataFrame, train: pd.DataFrame, theta_cols: list[str], cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    items_all, persons_all = [], []
    for t in sorted(train_clips["task"].unique()):
        d = train_clips[(train_clips["task"] == t) & train_clips["item_score"].notna()]
        for model in ("rasch", "2pl"):
            if model == "2pl" and d["item_id"].value_counts().median() < 30:
                continue
            m = IRTModel(model).fit(d)
            if m.items_ is None or m.items_.empty:
                continue
            it = m.items_.assign(task=t, model=model)
            items_all.append(it)
            pr = m.score_persons(d).assign(task=t, model=model)
            persons_all.append(pr)
    items = pd.concat(items_all, ignore_index=True) if items_all else pd.DataFrame()
    persons = pd.concat(persons_all, ignore_index=True) if persons_all else pd.DataFrame()
    if not persons.empty:
        persons = persons.merge(train[[ID, TARGET] + theta_cols], on=ID, how="left")
    return items, persons


def irt_vs_provided(persons: pd.DataFrame, theta_cols: list[str]) -> pd.DataFrame:
    rows = []
    if persons.empty:
        return pd.DataFrame()
    for (t, model), d in persons.groupby(["task", "model"]):
        prov = f"{t}_theta"
        r_prov = spearman(d["theta"], d[prov])[0] if prov in d else np.nan
        r_out, p_out, n = spearman(d["theta"], d[TARGET])
        rows.append({"task": t, "model": model, "spearman_vs_provided_theta": r_prov, "spearman_vs_outcome": r_out,
                     "p_vs_outcome": p_out, "n": n,
                     "provided_theta_vs_outcome": spearman(d[prov], d[TARGET])[0] if prov in d else np.nan})
    return pd.DataFrame(rows)


def adaptive_behaviour(beh: pd.DataFrame, train: pd.DataFrame, theta_cols: list[str]) -> pd.DataFrame:
    """Does the number of administered items track ability (adaptive stopping rules)?"""
    d = beh.merge(train[[ID, TARGET] + theta_cols], on=ID, how="inner")
    rows = []
    for t, g in d.groupby("task"):
        th = f"{t}_theta"
        for var in ("n_items", "accuracy", "mean_duration"):
            r1, p1, n1 = spearman(g[var], g[TARGET])
            r2, p2, _ = spearman(g[var], g[th]) if th in g else (np.nan, np.nan, 0)
            rows.append({"task": t, "variable": var, "rho_outcome": r1, "p_outcome": p1, "rho_task_theta": r2,
                         "p_task_theta": p2, "n": n1})
    out = pd.DataFrame(rows)
    out["q_outcome"] = bh_fdr(out["p_outcome"])
    return out
