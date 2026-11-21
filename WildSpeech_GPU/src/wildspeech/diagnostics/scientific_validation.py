"""Statistics for V2 scientific claims.

Recordings are repeated measures of students, so every clip-level inference resamples or permutes STUDENTS:
  block_bootstrap / block_bootstrap_contrasts   student-block bootstrap (vectorised multinomial weights)
  permutation_within / grouped_permutation_test permutations that respect items or students
  cluster_robust_ols                            OLS with CR1 cluster-robust (student) standard errors
  two_way_demean                                student + item fixed effects by alternating projections
Model-level metrics: full_metrics (mean repeat R2, pooled and fold-averaged OOF R2, averaged-prediction R2,
RMSE, MAE) and paired incremental R2 (validation.metrics.incremental_r2).
PreviousFindingsAudit re-derives every claim of the first experiment from the stored V1 artefacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import ID
from ..stats import bh_fdr, r2
from ..validation.metrics import metric_row, summarize_oof


# ============================================================================ model metrics
def full_metrics(oof: pd.DataFrame) -> dict:
    if oof is None or oof.empty:
        return {}
    out = {}
    if oof["repeat"].nunique() > 1 or "fold" in oof:
        try:
            out.update(summarize_oof(oof))
        except Exception:  # noqa: BLE001 - degenerate OOF (single row per repeat)
            pass
    pooled = metric_row(oof["y_true"], oof["y_pred"])
    out["r2_pooled"] = pooled["r2"]
    fr = oof.groupby(["repeat", "fold"]).apply(lambda d: r2(d["y_true"], d["y_pred"]) if len(d) > 2 else np.nan)
    out["r2_fold_mean"] = float(np.nanmean(fr)) if len(fr) else np.nan
    out["r2_fold_sd"] = float(np.nanstd(fr)) if len(fr) else np.nan
    out["r2_fold_min"] = float(np.nanmin(fr)) if len(fr) and np.isfinite(fr).any() else np.nan
    out.setdefault("r2_mean", pooled["r2"])
    out.setdefault("rmse_mean", pooled["rmse"])
    out.setdefault("mae_mean", pooled["mae"])
    out["n_rows"] = int(len(oof))
    return out


# ================================================================================== AUROC
def auc_fast(y, s) -> float:
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = ss.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_columns(y, S: np.ndarray) -> np.ndarray:
    """Univariate AUROC of every column of S for binary y (NaN rows handled per column)."""
    S = np.asarray(S, float)
    return np.array([auc_fast(y, S[:, j]) for j in range(S.shape[1])])


# ============================================================================== bootstrap
def block_weights(groups, n_boot: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique groups, row->group index, W[n_boot, n_groups]) with multinomial student counts."""
    g = np.asarray(groups).astype(str)
    uniq, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot).astype(float)
    return uniq, inv, W


def block_bootstrap(fn, df: pd.DataFrame, group_col: str, n_boot: int = 500, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float, float, float]:
    """Generic student-block bootstrap of fn(df) (resample whole students with replacement)."""
    est = float(fn(df))
    if n_boot <= 0:
        return est, np.nan, np.nan
    groups = df[group_col].astype(str).to_numpy()
    uniq = np.unique(groups)
    pos = pd.Series(np.arange(len(df))).groupby(groups).apply(lambda s: s.to_numpy()).to_dict()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos[u] for u in pick])
        try:
            vals.append(float(fn(df.iloc[idx])))
        except Exception:  # noqa: BLE001
            vals.append(np.nan)
    v = np.asarray(vals)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return est, np.nan, np.nan
    return est, float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def block_bootstrap_contrasts(V: np.ndarray, label: np.ndarray, groups, n_boot: int = 1000, seed: int = 0,
                              alpha: float = 0.05) -> pd.DataFrame:
    """Mean(V | label=1) - mean(V | label=0) for every column with a vectorised student-block bootstrap CI."""
    V = np.asarray(V, float)
    lab = np.asarray(label, float)
    uniq, inv, W = block_weights(groups, n_boot, seed)
    G = len(uniq)
    out = []
    M = np.isfinite(V)
    V0 = np.where(M, V, 0.0)
    sums = {}
    for k in (0.0, 1.0):
        sel = (lab == k)[:, None] & M
        S = np.zeros((G, V.shape[1]))
        N = np.zeros((G, V.shape[1]))
        np.add.at(S, inv, np.where(sel, V0, 0.0))
        np.add.at(N, inv, sel.astype(float))
        sums[k] = (S, N)
    est = sums[1.0][0].sum(0) / np.maximum(sums[1.0][1].sum(0), 1) - sums[0.0][0].sum(0) / np.maximum(sums[0.0][1].sum(0), 1)
    b1 = (W @ sums[1.0][0]) / np.maximum(W @ sums[1.0][1], 1)
    b0 = (W @ sums[0.0][0]) / np.maximum(W @ sums[0.0][1], 1)
    bd = b1 - b0
    lo, hi = np.nanquantile(bd, alpha / 2, axis=0), np.nanquantile(bd, 1 - alpha / 2, axis=0)
    se = np.nanstd(bd, axis=0)
    z = np.where(se > 0, est / se, np.nan)
    p = 2 * ss.norm.sf(np.abs(z))
    for j in range(V.shape[1]):
        out.append({"diff": est[j], "ci_low": lo[j], "ci_high": hi[j], "boot_se": se[j], "p_boot": p[j],
                    "n1": int(sums[1.0][1][:, j].sum()), "n0": int(sums[0.0][1][:, j].sum())})
    return pd.DataFrame(out)


# =========================================================================== permutations
def permutation_within(labels: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    """Shuffle labels within strata (e.g. items), preserving per-item difficulty."""
    labels = np.asarray(labels).copy()
    s = pd.Series(np.arange(len(labels))).groupby(np.asarray(strata)).apply(lambda x: x.to_numpy())
    for idx in s:
        labels[idx] = labels[rng.permutation(idx)]
    return labels


def grouped_permutation_test(stat_fn, x: pd.Series, y: pd.Series, n_perm: int = 1000, seed: int = 0) -> float:
    """Student-level permutation (rows are students): p = P(|stat(perm)| >= |stat(obs)|)."""
    rng = np.random.default_rng(seed)
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(j) < 8:
        return np.nan
    obs = abs(stat_fn(j["x"].to_numpy(), j["y"].to_numpy()))
    yy = j["y"].to_numpy()
    cnt = sum(abs(stat_fn(j["x"].to_numpy(), rng.permutation(yy))) >= obs for _ in range(n_perm))
    return float((cnt + 1) / (n_perm + 1))


def spearman_stat(a, b) -> float:
    return float(ss.spearmanr(a, b)[0])


# ======================================================================= regression (CR1)
def cluster_robust_ols(y, X, groups, names: list[str] | None = None) -> pd.DataFrame:
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    g = np.asarray(groups).astype(str)
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    y, X, g = y[ok], X[ok], g[ok]
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    uniq, inv = np.unique(g, return_inverse=True)
    G = len(uniq)
    S = np.zeros((G, k))
    np.add.at(S, inv, X * e[:, None])
    meat = S.T @ S
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = c * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = beta / np.where(se > 0, se, np.nan)
    p = 2 * ss.t.sf(np.abs(t), max(G - 1, 1))
    names = ["intercept"] + (names or [f"x{i}" for i in range(k - 1)])
    return pd.DataFrame({"term": names, "coef": beta, "se_cr1": se, "t": t, "p": p, "n": n, "clusters": G})


def two_way_demean(v: np.ndarray, a, b, iters: int = 20, tol: float = 1e-8) -> np.ndarray:
    """Remove student and item fixed effects (alternating projections, NaN-aware)."""
    v = np.asarray(v, float).copy()
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(v)
    for _ in range(iters):
        before = v.copy()
        for g in (a, b):
            s = pd.Series(np.where(m, v, np.nan)).groupby(g).transform("mean").to_numpy()
            v = np.where(m, v - np.nan_to_num(s), np.nan)
        if np.nanmax(np.abs(v - before)) < tol:
            break
    return v


def add_fdr(df: pd.DataFrame, p_col: str = "p", q_col: str = "q", by: str | None = None) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df[q_col] = []
        return df
    if by is None:
        df[q_col] = bh_fdr(df[p_col].to_numpy(float))
    else:
        df[q_col] = np.nan
        for _, idx in df.groupby(by).groups.items():
            df.loc[idx, q_col] = bh_fdr(df.loc[idx, p_col].to_numpy(float))
    return df


# ================================================================= previous experiment audit
class PreviousFindingsAudit:
    """Re-derive the first experiment's findings from stored artefacts (never hard-coded)."""

    def __init__(self, ctx):
        self.ctx = ctx

    def _t(self, sid, name):
        return self.ctx.read_stage_table(sid, name)

    def run(self) -> pd.DataFrame:
        rows = []
        b = self.ctx.bundle()

        def add(claim, value, verdict, source, detail=""):
            rows.append({"claim": claim, "value": value, "verdict": verdict, "source": source, "detail": detail})

        add("independent training students", len(b.train), "measured", "data", "")
        add("training recordings", len(b.train_clips), "measured", "data", "")
        bench = self._t("P10", "model_benchmark")
        if bench is not None and not bench.empty:
            ok = bench[bench["status"] == "ok"]
            scr = ok[ok["families"] == "SCREENING"]
            if not scr.empty:
                add("screening-only R2 (best model, repeated K-fold)", float(scr["r2_mean"].max()), "measured", "P10")
            gam = ok[ok["model"] == "GAM"]
            if not gam.empty:
                g = gam.sort_values("r2_mean").iloc[-1]
                add("best tabular GAM R2", float(g["r2_mean"]), "measured", "P10", str(g["families"]))
            sbi = ok[ok["families"] == "SCREENING+BEHAVIOR+ITEM"]["r2_mean"].max()
            sbit = ok[ok["families"] == "SCREENING+BEHAVIOR+ITEM+TEMPORAL"]["r2_mean"].max()
            allf = ok[ok["families"] == "ALL"]["r2_mean"].max()
            if np.isfinite(sbi) and np.isfinite(allf):
                add("adding all acoustic features reduces R2", float(allf - sbi), "verified" if allf < sbi else "not_verified",
                    "P10", f"ALL={allf:.4f} vs SCREENING+BEHAVIOR+ITEM={sbi:.4f}")
            if np.isfinite(sbi) and np.isfinite(sbit):
                add("behaviour/item outperform generic acoustic aggregation", float(sbi - sbit),
                    "verified" if sbi > sbit else "not_verified", "P10", f"+TEMPORAL={sbit:.4f}")
        else:
            add("P10 benchmark", None, "artifact_missing", "P10")
        ra = self._t("P07", "residual_associations")
        if ra is not None and not ra.empty and "q_resid" in ra:
            alpha = float(self.ctx.cfg.get_path("validation.fdr_alpha", 0.05))
            n = int((ra["q_resid"] < alpha).sum())
            add("univariate residual associations surviving FDR", n, "verified" if n == 0 else "not_verified", "P07",
                f"{len(ra)} features tested, min q={ra['q_resid'].min():.3g}")
        cc = self._t("P02", "correctness_contrasts_item_centred")
        if cc is not None and not cc.empty:
            qcol = next((c for c in cc.columns if c.startswith("q")), None)
            if qcol:
                n = int((cc[qcol] < 0.05).sum())
                add("acoustic correct-vs-incorrect contrasts (q<0.05)", n, "verified" if n > 0 else "not_verified", "P02")
        dg = self._t("P08", "domain_generalisation")
        if dg is not None and not dg.empty:
            r = dg[(dg["scheme"] == "sample_rate_transfer")]
            for rr in r.itertuples():
                add(f"sample-rate transfer R2 ({rr.name})", float(rr.r2_mean), "measured", "P08", str(rr.families))
            for sc in ("loso", "school_kfold"):
                r = dg[dg["scheme"] == sc]
                for rr in r.itertuples():
                    add(f"{sc} R2 ({rr.name})", float(rr.r2_mean), "measured", "P08")
        ge = self._t("P02", "clip_feature_group_effects")
        if ge is not None and not ge.empty and {"factor", "eta2"} <= set(ge.columns):
            for fac, d in ge.groupby("factor"):
                add(f"median eta2 of clip features by {fac}", float(d["eta2"].median()), "measured", "P02", "confounding screen")
        fc = self._t("P05", "feature_family_summary")
        if fc is not None and not fc.empty:
            ncol = next((c for c in fc.columns if c.startswith("n_")), None)
            if ncol:
                add("student-level features vs students", float(fc[ncol].sum()), "measured", "P05",
                    f"{int(fc[ncol].sum())} features for {len(b.train)} students")
        iv = self._t("P09", "index_validation")
        if iv is not None and not iv.empty:
            lo = next((c for c in iv.columns if "ci_low" in c), None)
            if lo:
                n = int((iv[lo] > 0).sum())
                add("composite indices with incremental R2 CI > 0", n, "verified_weak" if n == 0 else "not_verified", "P09")
        mm = self._t("P11", "multimodal_results")
        if mm is not None and not mm.empty:
            te = mm[mm["name"].astype(str).str.contains("TASK_ENSEMBLE")]
            for rr in te.itertuples():
                add("task-ensemble R2", float(rr.r2_mean), "verified_catastrophic" if rr.r2_mean < -1 else "not_catastrophic",
                    "P11", f"rmse={rr.rmse_mean:.2f}")
        for sid in ("P03", "P04"):
            st = self.ctx.read_stage_json(sid, "_status", {}) or {}
            add(f"{sid} status", st.get("status", "never_run"), "measured", sid, st.get("note", ""))
        return pd.DataFrame(rows)
