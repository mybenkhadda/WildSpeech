"""P23 — audit of the V2 fusion system before any new ASR experiment.

Observed in outputs/V2/P21_MULTIMODAL/tables/all_results.csv (read at runtime, never hard-coded): most
residual-fusion rows report r2 IDENTICAL to the GAM comparator to ~15 digits and ΔR² = 0 with a zero-width CI.
Mechanism under test: NestedResidualRegressor picks the residual shrinkage alpha from {0, .25, .5, .75, 1} by
inner CV; alpha = 0 drops the stage-2 model and the prediction IS the stage-1 GAM = the baseline. That is a
fallback, not an estimate of incremental value.

Components
  classify_v2_results(all_results)      zero-delta rows, identical-to-baseline rows, budget cap (n_added constant
                                        across budgets because V2 PCA is capped at family_components)
  reaudit(ev, specs, providers, base)   re-runs V2 specs with V3Evaluator(method='v2_family') -> per-fold trace
                                        (inputs, valid counts, variance, missingness, alpha, inner MSE per alpha,
                                        stage-2 prediction SD, prediction differences vs baseline, fallback reason,
                                        id alignment) + reproduction check against the V2 table
  diagnostic variants                   forced alpha (default 1.0), continuous alpha grid, V3 reducer (budget-
                                        following PCA on inner-OOF residuals) -> separates "no signal" from
                                        "signal discarded by the gate"
  verdict(...)                          genuine_negative | silent_fallback | unused_features | evaluation_error |
                                        positive_signal_discarded | fitted_positive
  lineage_audit(ctx)                    stale V1/V2 stage fingerprints + recorded-vs-current upstream fingerprints
  TaskEnsembleAudit                     V1 P11 TASK_ENSEMBLE re-run with prediction tracing (NNLS weight sum,
                                        per-task member predictions, extreme predictions, students with missing
                                        task data) vs the repaired RobustTaskEnsemble on identical folds/features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID
from .nested_asr_cv import V3Evaluator, V3Result, V3Spec, result_row


def classify_v2_results(res: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    if res is None or res.empty:
        return pd.DataFrame()
    r = res.copy()
    base = r[r["name"] == "V2_BASELINE"].set_index("scheme")["r2_mean"]
    r["baseline_r2"] = r["scheme"].map(base)
    r["identical_r2"] = (r["r2_mean"] - r["baseline_r2"]).abs() <= tol
    r["zero_delta"] = r["delta_r2"].abs().fillna(np.inf) <= tol
    r["zero_width_ci"] = (r["delta_ci_high"] - r["delta_ci_low"]).abs().fillna(np.inf) <= tol
    r["suspect_fallback"] = r["identical_r2"] & r["zero_delta"] & (r["name"] != "V2_BASELINE") & (r["group"] != "comparator")
    bud = r[r["group"] == "budget"].copy()
    if not bud.empty:
        bud["family"] = bud["name"].str.split("_").str[2]
        cap = bud.groupby("family").apply(lambda d: pd.Series({"budgets": ",".join(map(str, sorted(d["budget"].astype(int)))),
                                                               "n_added_values": ",".join(map(str, sorted(d["n_added"].unique()))),
                                                               "max_n_added": float(d["n_added"].max()),
                                                               "capped": bool(d["n_added"].max() < d["budget"].max())}))
        r.attrs["budget_cap"] = cap.reset_index()
    return r


def budget_cap_table(classified: pd.DataFrame) -> pd.DataFrame:
    return classified.attrs.get("budget_cap", pd.DataFrame()) if classified is not None else pd.DataFrame()


def v2_spec_to_v3(d: dict, method: str = "v2_family", **over) -> V3Spec:
    s = V3Spec(name=d["name"], model=d.get("model", "RIDGE"), providers=list(d.get("providers") or []),
               budget=d.get("budget"), method=method, fusion=d.get("fusion", "early"),
               base_families=d.get("base_families"), base_model=d.get("base_model"), params=dict(d.get("params") or {}),
               group="audit", experiment="P21")
    for k, v in over.items():
        setattr(s, k, v)
    return s


def reaudit(ev: V3Evaluator, v2_specs: dict, names: list[str], schemes: list[str], base: dict[str, V3Result],
            v2_table: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = cfg.get_path("v3.audit", {}) or {}
    forced = float(a.get("forced_alpha", 1.0))
    grid = tuple(float(x) for x in a.get("continuous_alphas", [0, 0.1, 0.25, 0.5, 1.0]))
    rows, traces = [], []
    for n in names:
        if n not in v2_specs:
            rows.append({"name": n, "variant": "reproduce", "status": "spec_not_found"})
            continue
        d = v2_specs[n]
        variants = [("reproduce", v2_spec_to_v3(d))]
        if d.get("fusion") == "residual":
            variants += [(f"forced_alpha_{forced:g}", v2_spec_to_v3(d, name=f"{n}__ALPHA{forced:g}", alphas=(forced,))),
                         ("continuous_alpha", v2_spec_to_v3(d, name=f"{n}__ALPHAGRID", alphas=grid))]
        variants += [("v3_reducer", v2_spec_to_v3(d, method="pca", name=f"{n}__V3PCA"))]
        for vname, sp in variants:
            for sc in schemes:
                if sc not in base:
                    continue
                r = ev.evaluate(sp, sc, baseline=base[sc])
                dlt = ev.compare(base[sc], r) if r.status not in ("failed", "unavailable", "no_folds") else {}
                row = {**result_row(r, dlt), "audited": n, "variant": vname}
                if vname == "reproduce" and v2_table is not None and not v2_table.empty:
                    v = v2_table[(v2_table["name"] == n) & (v2_table["scheme"] == sc)]
                    if not v.empty:
                        row["v2_r2_mean"] = float(v.iloc[0]["r2_mean"])
                        row["v2_delta_r2"] = float(v.iloc[0]["delta_r2"]) if pd.notna(v.iloc[0]["delta_r2"]) else np.nan
                        row["reproduced"] = bool(abs(row["v2_r2_mean"] - (row["r2_mean"] or np.nan)) < 1e-6)
                rows.append(row)
                if not r.trace.empty:
                    traces.append(r.trace.assign(audited=n, variant=vname, name=sp.name))
    return pd.DataFrame(rows), (pd.concat(traces, ignore_index=True) if traces else pd.DataFrame())


def verdicts(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "variant" not in audit:
        return pd.DataFrame()
    out = []
    for (n, sc), g in audit.groupby(["audited", "scheme"]):
        rep = g[g["variant"] == "reproduce"]
        if rep.empty:
            continue
        rep = rep.iloc[0]
        forced = g[g["variant"].str.startswith("forced_alpha")]
        v3r = g[g["variant"] == "v3_reducer"]
        fd = float(forced.iloc[0]["delta_r2"]) if not forced.empty and pd.notna(forced.iloc[0].get("delta_r2")) else np.nan
        vd = float(v3r.iloc[0]["delta_r2"]) if not v3r.empty and pd.notna(v3r.iloc[0].get("delta_r2")) else np.nan
        st = rep["status"]
        if st in ("failed", "unavailable"):
            v = "evaluation_error"
        elif st == "no_added_features":
            v = "unused_features"
        elif st == "fallback_baseline":
            v = "positive_signal_discarded" if (np.isfinite(fd) and fd > 0) or (np.isfinite(vd) and vd > 0) else "silent_fallback"
        elif pd.notna(rep.get("delta_r2")) and rep["delta_r2"] > 0:
            v = "fitted_positive"
        else:
            v = "genuine_negative"
        out.append({"audited": n, "scheme": sc, "status": st, "verdict": v, "delta_r2_reported": rep.get("v2_delta_r2"),
                    "delta_r2_reproduced": rep.get("delta_r2"), "reproduced": rep.get("reproduced"),
                    "fallback_folds": rep.get("fallback_folds"), "n_folds": rep.get("n_folds"),
                    "delta_r2_forced_alpha": fd, "delta_r2_v3_reducer": vd,
                    "max_abs_diff_vs_baseline": rep.get("max_abs_diff_vs_baseline")})
    return pd.DataFrame(out)


def lineage_audit(ctx) -> pd.DataFrame:
    from ..orchestrator import STAGES, V2_STAGES, Pipeline

    pipe = Pipeline(ctx)
    rows = []
    for sid in list(V2_STAGES) + ["P10", "P11"]:
        st = pipe.status(sid)
        if not st:
            rows.append({"stage": sid, "status": "never_run"})
            continue
        cur = pipe.fingerprint(sid) if st.get("status") == "completed" else None
        rec_up = st.get("upstream", {}) or {}
        mism = [f"{u}:{fp}->{pipe.status(u).get('fingerprint')}" for u, fp in rec_up.items() if pipe.status(u).get("fingerprint") != fp]
        spec = STAGES[sid]
        rows.append({"stage": sid, "status": st.get("status"), "profile": st.get("profile"), "finished": st.get("finished"),
                     "recorded_fp": st.get("fingerprint"), "current_fp": cur,
                     "stale_under_current_config": (cur != st.get("fingerprint")) if cur else None,
                     "upstream_changed_since_run": ";".join(mism), "n_upstream_changed": len(mism),
                     "hard_deps": ",".join(spec.deps), "soft_deps": ",".join(spec.soft), "note": st.get("note", "")})
    return pd.DataFrame(rows)


# ========================================================================== V1 task ensemble audit
class TaskEnsembleAudit:
    """Re-run V1 P11 F_TASK_ENSEMBLE_RIDGE (LateFusionRegressor grouping='task', features ALL) with tracing, and the
    repaired V2 RobustTaskEnsemble on the same folds and features."""

    def __init__(self, ctx, max_repeats: int | None = None):
        self.ctx, self.max_repeats = ctx, max_repeats

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from ..models.fusion import LateFusionRegressor
        from ..models.preprocessing import FoldModel
        from ..models.residual_fusion import RobustTaskEnsemble
        from ..stages._common import get_runner

        ctx = self.ctx
        z_ext = float(ctx.cfg.get_path("v3.audit.extreme_z", 6.0))
        runner = get_runner(ctx, with_audio=True)
        tasks = sorted(ctx.bundle().all_clips["task"].unique())
        folds = runner.folds("repeated_kfold")
        if self.max_repeats:
            folds = [f for f in folds if f.repeat < int(self.max_repeats)]
        y_all = runner.y
        oof_rows, fold_rows, mem_rows = [], [], []
        for f in folds:
            Xtr, Xva = runner.fold_data(f)
            ytr, yva = y_all.reindex(Xtr.index).to_numpy(float), y_all.reindex(Xva.index).to_numpy(float)
            mu, sd = float(ytr.mean()), float(ytr.std())
            seed = ctx.seed + f.repeat * 31 + f.fold
            miss = {t: (Xva.get(f"BEHAVIOR__{t}__n_items__value", pd.Series(1.0, index=Xva.index)).fillna(0) <= 0).to_numpy() for t in tasks}
            for variant in ("v1_task_ensemble", "repaired_robust"):
                if variant == "v1_task_ensemble":
                    est = LateFusionRegressor(grouping="task", tasks=tuple(tasks), inner_splits=int(ctx.cfg.get_path("validation.inner_splits", 4)), seed=seed)
                    fm = FoldModel(est, 0, float(ctx.cfg.get_path("models.max_missing_frac", 0.6)), True).fit(Xtr, ytr)
                    p = fm.predict(Xva)
                    Xs = fm.sel.transform(fm.prep.transform(Xva))
                    w = est.weights_
                    members = {g: np.asarray(m.predict(Xs[est.groups_[g]]), float) for g, m in est.models_.items()}
                else:
                    from ..models.preprocessing import TabularPreprocessor

                    prep = TabularPreprocessor(max_missing_frac=float(ctx.cfg.get_path("models.max_missing_frac", 0.6))).fit(Xtr)
                    A = lambda X: pd.concat([prep.transform(X), pd.DataFrame({f"__avail__{t}": (X.get(f"BEHAVIOR__{t}__n_items__value", pd.Series(1.0, index=X.index)).fillna(0) > 0).astype(float) for t in tasks}, index=X.index)], axis=1)  # noqa: E731
                    base = [c for c in prep.transform(Xtr.head(2)).columns if c.split("__")[0] in ("SCREENING", "RECORDING")]
                    est = RobustTaskEnsemble(base_cols=base, tasks=tuple(tasks), seed=seed).fit(A(Xtr), ytr)
                    p = est.predict(A(Xva))
                    w = est.weights_
                    members = {}
                z = np.abs(p - mu) / max(sd, 1e-9)
                ext = z > z_ext
                fold_rows.append({"variant": variant, "repeat": f.repeat, "fold": f.fold, "r2": metric(yva, p),
                                  "weight_sum": float(np.sum(w)), "weights": ";".join(f"{k}:{float(v):.3f}" for k, v in pd.Series(w).items()),
                                  "max_abs_z": float(z.max()), "n_extreme": int(ext.sum()), "extreme_frac": float(ext.mean()),
                                  **{f"extreme_missing_{t}": int((ext & miss[t]).sum()) for t in tasks}})
                for g, mp in members.items():
                    mz = np.abs(mp - mu) / max(sd, 1e-9)
                    mem_rows.append({"variant": variant, "repeat": f.repeat, "fold": f.fold, "member": g,
                                     "weight": float(pd.Series(w).get(g, np.nan)), "max_abs_z": float(mz.max()),
                                     "n_extreme": int((mz > z_ext).sum()), "pred_sd": float(np.std(mp))})
                oof_rows.append(pd.DataFrame({ID: Xva.index, "repeat": f.repeat, "fold": f.fold, "y_true": yva, "y_pred": p,
                                              "variant": variant, "abs_z": z}))
        return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(fold_rows), pd.DataFrame(mem_rows)


def metric(y, p) -> float:
    from ..stats import r2

    return float(r2(y, p))
