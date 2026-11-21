"""Feature importance that respects the fold structure.

grouped_permutation_importance  for folds of the first `repeats` outer repeats: fit the spec's V4FoldModel on the
                                training fold, then permute the RAW columns of one family (baseline families by
                                family prefix, every provider family) in the VALIDATION fold and record the R² drop.
                                Nothing is refitted on validation students.
model_importance                the champion refitted on all labelled students: SHAP (TreeExplainer) for boosted
                                trees when `shap` is installed, else |coefficient| x SD for linear models; each
                                column is mapped to its family (reduced columns keep the family prefix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.builder import family_of
from ..stats import r2


def grouped_permutation_importance(ev, spec, scheme: str = "repeated_kfold", repeats: int = 1, n_perm: int = 5,
                                   seed: int = 0) -> pd.DataFrame:
    from ..v3.common import folds

    rows = []
    rng = np.random.default_rng(seed)
    for f in [f for f in folds(ev.ctx, scheme) if f.repeat < int(repeats)]:
        Xtr, Xva, base_cols, groups = ev.fold_frames(f, spec)
        ytr = ev.y.reindex(Xtr.index).to_numpy(float)
        yva = ev.y.reindex(Xva.index).to_numpy(float)
        fm = ev.fold_model_cls(spec, ev.ctx.cfg, ev.ctx.seed + f.repeat * 31 + f.fold, base_cols, groups, ev.tasks,
                               ev.baseline["model"]).fit(Xtr, ytr)
        ref = r2(yva, fm.predict(Xva))
        fams: dict[str, list[str]] = {}
        for c in base_cols:
            fams.setdefault(family_of(c), []).append(c)
        for k, v in groups.items():
            fams.setdefault(k, []).extend(v)
        for fam, cols in fams.items():
            cols = [c for c in cols if c in Xva.columns]
            if not cols:
                continue
            drops = []
            for _ in range(int(n_perm)):
                Xp = Xva.copy()
                Xp[cols] = Xp[cols].to_numpy()[rng.permutation(len(Xp))]
                drops.append(ref - r2(yva, fm.predict(Xp)))
            rows.append({"name": spec.name, "scheme": scheme, "repeat": f.repeat, "fold": f.fold, "family": fam,
                         "n_columns": len(cols), "r2_fold": ref, "r2_drop_mean": float(np.mean(drops)),
                         "r2_drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows)


def summarize_grouped(imp: pd.DataFrame) -> pd.DataFrame:
    if imp is None or imp.empty:
        return pd.DataFrame()
    g = imp.groupby("family")["r2_drop_mean"]
    out = pd.DataFrame({"importance_mean": g.mean(), "importance_sd": g.std(ddof=0), "n_folds": g.size()})
    out["ci_low"] = out["importance_mean"] - 1.96 * out["importance_sd"] / np.sqrt(out["n_folds"].clip(lower=1))
    out["ci_high"] = out["importance_mean"] + 1.96 * out["importance_sd"] / np.sqrt(out["n_folds"].clip(lower=1))
    return out.reset_index().sort_values("importance_mean", ascending=False)


def model_importance(fmod, Xtr: pd.DataFrame) -> pd.DataFrame:
    M = fmod.matrix(Xtr)
    est = fmod.est_
    name = type(est).__name__
    vals, method = None, None
    if name in ("CatBoostRegressor", "LGBMRegressor", "XGBRegressor"):
        try:
            import shap

            sv = shap.TreeExplainer(est).shap_values(M)
            vals, method = np.abs(np.asarray(sv)).mean(0), "shap_tree_mean_abs"
        except Exception:  # noqa: BLE001 - shap missing / unsupported -> native importance
            imp = getattr(est, "feature_importances_", None)
            if imp is not None:
                vals, method = np.asarray(imp, float), "native_gain"
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, float).ravel()
        if len(coef) == M.shape[1]:
            vals, method = np.abs(coef) * M.std(ddof=0).to_numpy(float), "abs_coef_x_sd"
    if vals is None:
        return pd.DataFrame(columns=["feature", "family", "importance", "method"])
    out = pd.DataFrame({"feature": list(M.columns), "importance": vals[: M.shape[1]], "method": method})
    out["family"] = out["feature"].map(lambda c: family_of(c.replace("__gate__", "GATE__")))
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
