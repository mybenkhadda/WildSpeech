"""P12 — failure analysis: clusters of large screening-baseline errors; which modalities/models recover them."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import viz
from ..analysis import failures as Fa
from ._common import load_crossfit, save_fig, student_meta

S = "P12"
BASELINE = "B_RIDGE__SCREENING+RECORDING"


def collect_oofs(ctx, stages=("P10", "P11"), scheme: str = "repeated_kfold") -> dict[str, pd.DataFrame]:
    out = {}
    for st in stages:
        d = ctx.stage_dir(st) / "oof"
        for p in sorted(Path(d).glob(f"*__{scheme}.parquet")):
            out[p.name.replace(f"__{scheme}.parquet", "")] = pd.read_parquet(p)
    return out


def run(ctx) -> dict:
    a = ctx.cfg.section("analysis")
    oofs = collect_oofs(ctx)
    if BASELINE not in oofs:
        return {"status": "baseline OOF missing (run P10)"}
    bench = ctx.read_stage_table("P10", "model_benchmark")
    keep = {BASELINE}
    if bench is not None:
        keep |= set(bench[bench.status == "ok"].sort_values("r2_mean", ascending=False).head(6)["name"])
    keep |= {k for k in oofs if k.startswith(("F_", "N_"))}
    oofs = {k: v for k, v in oofs.items() if k in keep}
    err = Fa.error_table(oofs)
    X = load_crossfit(ctx)
    X.index = X.index.astype(str)
    err.index = err.index.astype(str)
    k_lo, k_hi = a.get("cluster_k_range", [2, 7])
    fc, sel = Fa.failure_clusters(X, err, BASELINE, float(a.get("failure_quantile", 0.8)), (int(k_lo), int(k_hi)), ctx.seed)
    ctx.save_table(fc.rename_axis("subject_id").reset_index(), S, "student_errors_and_failure_clusters")
    ctx.save_table(sel, S, "failure_cluster_selection")
    rec = Fa.recovery_table(fc, list(oofs), BASELINE, int(a.get("bootstrap", 1000)), ctx.seed)
    ctx.save_table(rec, S, "failure_recovery")
    desc = Fa.describe_clusters(fc, X, student_meta(ctx).set_index("subject_id"))
    ctx.save_table(desc, S, "failure_cluster_profiles")
    best = [c for c in err.columns if c.startswith("abserr__") and c != f"abserr__{BASELINE}"]
    if best:
        bcol = min(best, key=lambda c: err[c].mean())
        save_fig(ctx, S, "baseline_vs_best_abs_error", viz.scatter_fit(err[f"abserr__{BASELINE}"], err[bcol], "|error| screening baseline",
                                                                      f"|error| {bcol.split('__', 1)[1]}", "Per-student absolute errors",
                                                                      hue=fc["is_failure"].map({True: "failure", False: "other"})),
                 err[[f"abserr__{BASELINE}", bcol]].reset_index())
    if not rec.empty:
        r = rec.assign(label=lambda d: "cluster " + d["failure_cluster"].astype(str) + " · " + d["model"])
        save_fig(ctx, S, "failure_recovery", viz.forest(r.sort_values("mae_reduction", ascending=False), "mae_reduction", "ci_low", "ci_high",
                                                       "label", "MAE reduction on failure clusters vs screening baseline"), r)
        n_rec = int(rec.groupby("failure_cluster")["recovers"].any().sum())
        ctx.registries.update_hypothesis("H11", "supported" if n_rec else "not_supported", S,
                                         f"{n_rec} failure clusters recovered by >=1 model (MAE reduction CI>0)")
        regs = []
        for _, d in desc.iterrows():
            rr = rec[rec.failure_cluster == d["failure_cluster"]]
            winners = rr[rr["recovers"]].sort_values("mae_reduction", ascending=False)["model"].tolist()
            regs.append({"failure_id": f"P12_C{int(d['failure_cluster'])}", "n_students": int(d["n"]),
                         "frac_underpredicted": d["frac_underpredicted"], "top_features": d["top_features"],
                         "recovered_by": ", ".join(winners) or "none"})
        ctx.registries.failures.upsert(regs)
    return {"n_failures": int(fc["is_failure"].sum()), "n_clusters": int(fc.loc[fc.is_failure, "failure_cluster"].nunique())}
