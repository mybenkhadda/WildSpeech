"""Shared helpers for the V2 stages P16-P22."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID


def begin(ctx, sid: str):
    """Lineage check + frozen validation registry + V2 registries; writes the registry summary for the stage."""
    from ..v2.lineage import require_fresh
    from ..v2.registry import v2_registries
    from ..v2.validation_registry import get_registry

    require_fresh(ctx, sid)
    reg = get_registry(ctx)
    ctx.save_table(reg.summary(), sid, "validation_registry_summary")
    return reg, v2_registries(ctx)


def schemes(ctx, key: str = "v2.evaluation.student_schemes") -> list[str]:
    from ..v2.validation_registry import get_registry

    avail = set(get_registry(ctx).schemes())
    return [s for s in ctx.cfg.get_path(key, ["repeated_kfold"]) if s in avail]


def result_row(res, extra: dict | None = None) -> dict:
    s = res.summary or {}
    return {"name": res.spec.name, "scheme": res.scheme, "status": res.status, "model": res.spec.model,
            "fusion": res.spec.fusion, "providers": "+".join(res.spec.providers), "budget": res.spec.budget,
            "r2_mean": s.get("r2_mean"), "r2_sd": s.get("r2_sd", 0.0), "r2_pooled": s.get("r2_pooled"),
            "r2_fold_mean": s.get("r2_fold_mean"), "r2_repeat_avg_pred": s.get("r2_repeat_avg_pred"),
            "rmse": s.get("rmse_mean"), "mae": s.get("mae_mean"), "fold_r2_min": s.get("fold_r2_min"),
            "n_added": s.get("n_added_mean"), "n_features": s.get("n_features_mean"), **(extra or {})}


def evaluate_against(ev, spec, base_results: dict, scheme_list: list[str]) -> tuple[list[dict], dict]:
    """Evaluate spec on each scheme; paired delta vs the comparator on the same scheme."""
    rows, res_by = [], {}
    for sc in scheme_list:
        r = ev.evaluate(spec, sc)
        res_by[sc] = r
        d = ev.compare(base_results[sc], r) if (r.status == "ok" and sc in base_results and base_results[sc].status == "ok") else {}
        rows.append(result_row(r, d))
    return rows, res_by


def transfer_breakdown(res) -> pd.DataFrame:
    """Per-direction metrics of a transfer scheme (fold labels name the validation domain)."""
    if res is None or res.fold_metrics is None or res.fold_metrics.empty:
        return pd.DataFrame()
    return res.fold_metrics[["label", "r2", "rmse", "mae", "n"]].assign(name=res.spec.name, scheme=res.scheme)


def student_meta(ctx) -> pd.DataFrame:
    b = ctx.bundle()
    cols = [ID] + [c for c in ("instrument", "original_sample_rate", "school_id") if c in b.train_students]
    m = b.train_students[cols].copy()
    m[ID] = m[ID].astype(str)
    return m.set_index(ID)


def baseline_residuals(ev, base_res) -> pd.Series:
    from ..validation.metrics import mean_oof

    m = mean_oof(base_res.oof)
    m.index = m.index.astype(str)
    return (m["y_true"] - m["y_pred"]).rename("resid")


def association_table(X: pd.DataFrame, resid: pd.Series, ids_disc: list[str], ids_conf: list[str], n_perm: int,
                      seed: int, top_k: int = 30) -> pd.DataFrame:
    """Exploratory screen on the discovery half, confirmatory test of the top-k on the confirmation half
    (Spearman + student-level permutation p, FDR within tier)."""
    from ..diagnostics.scientific_validation import add_fdr, grouped_permutation_test, spearman_stat
    from ..stats import spearman

    rows = []
    for c in X.columns:
        r, p, n = spearman(X[c].reindex(ids_disc), resid.reindex(ids_disc))
        rows.append({"feature": c, "tier": "exploratory", "rho": r, "p": p, "n": n})
    exp = add_fdr(pd.DataFrame(rows))
    top = exp.dropna(subset=["p"]).sort_values("p").head(top_k)["feature"].tolist()
    conf = []
    for c in top:
        r, p, n = spearman(X[c].reindex(ids_conf), resid.reindex(ids_conf))
        sign = np.sign(exp.set_index("feature").loc[c, "rho"])
        perm = grouped_permutation_test(spearman_stat, X[c].reindex(ids_conf), resid.reindex(ids_conf), n_perm, seed)
        conf.append({"feature": c, "tier": "confirmatory", "rho": r, "p": perm, "p_spearman": p, "n": n,
                     "same_sign": bool(np.sign(r) == sign) if np.isfinite(r) else False})
    conf = add_fdr(pd.DataFrame(conf)) if conf else pd.DataFrame(columns=["feature", "tier", "rho", "p", "q"])
    return pd.concat([exp, conf], ignore_index=True)
