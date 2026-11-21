"""P10 — model benchmark (models x feature-family sets) + budgeted family ablations, all student-disjoint CV."""
from __future__ import annotations

import pandas as pd

from .. import viz
from ..features.builder import families_in
from ..models.ablation import FamilyAblation
from ..validation.metrics import incremental_r2, subgroup_metrics
from ..validation.runner import ModelSpec
from ._common import get_runner, save_fig, student_meta

S = "P10"
BASELINE = "B_RIDGE__SCREENING+RECORDING"


def run(ctx) -> dict:
    m = ctx.cfg.section("models")
    runner = get_runner(ctx, with_audio=True)
    f0 = runner.folds("repeated_kfold")[0]
    Xtr, _ = runner.fold_data(f0)
    fams = families_in(Xtr.columns)
    ctx.save_json({"available_families": fams}, S, "available_families")
    nb = int(ctx.cfg.get_path("validation.bootstrap", 1000))
    rows, oofs = [], {}
    for model in m.get("benchmark_models", ["RIDGE"]):
        for fs in m.get("benchmark_families", [["SCREENING"], ["ALL"]]):
            fs = [f for f in fs if f == "ALL" or f in fams]
            if not fs:
                continue
            name = f"B_{model}__{'+'.join(fs)}"
            res = runner.evaluate(ModelSpec(name, model, fs), "repeated_kfold", stage=S)
            runner.save_oof(res, S)
            rows.append({"name": name, "model": model, "families": "+".join(fs), "status": res.status, **res.summary})
            if res.status == "ok":
                oofs[name] = res.oof
    if BASELINE not in oofs:
        res = runner.evaluate(ModelSpec(BASELINE, "RIDGE", ["SCREENING", "RECORDING"]), "repeated_kfold", stage=S)
        runner.save_oof(res, S)
        oofs[BASELINE] = res.oof
        rows.append({"name": BASELINE, "model": "RIDGE", "families": "SCREENING+RECORDING", "status": res.status, **res.summary})
    bench = pd.DataFrame(rows)
    deltas = [{"name": n, **incremental_r2(oofs[BASELINE], o, nb, 0)} for n, o in oofs.items() if n != BASELINE]
    if deltas:
        bench = bench.merge(pd.DataFrame(deltas), on="name", how="left")
    ctx.save_table(bench, S, "model_benchmark")
    ok = bench[bench.status == "ok"]
    if not ok.empty:
        piv = ok.pivot_table(index="model", columns="families", values="r2_mean")
        save_fig(ctx, S, "benchmark_heatmap", viz.heatmap(piv, "CV R2 (mean over repeats): model x feature families", diverging=False,
                                                         annotate=True), piv.reset_index())
        best = ok.sort_values("r2_mean", ascending=False).iloc[0]
        ctx.save_table(subgroup_metrics(oofs[best["name"]], student_meta(ctx),
                                        [c for c in ("instrument", "original_sample_rate", "school_id") if c in student_meta(ctx)]),
                       S, "best_model_subgroups")

    # ---- budgeted ablation
    abl = FamilyAblation(runner, m.get("ablation_model", "RIDGE"), list(m.get("ablation_base", ["SCREENING", "RECORDING"])),
                         fams, "repeated_kfold", int(m.get("ablation_max_evals", 80)), float(m.get("ablation_time_budget_s", 3600)),
                         min(nb, 500), S)
    table = abl.run(int(m.get("ablation_max_subset", 3)), int(m.get("ablation_forward_steps", 6)))
    ctx.save_table(table, S, "family_ablation")
    for key, oof in abl.oofs.items():
        from ..io_utils import atomic_write_parquet

        d = ctx.stage_dir(S) / "oof"
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(oof, d / f"ABL_{'+'.join(key)}__repeated_kfold.parquet")
    single = table[table["kind"].isin(["single_add", "base"])].sort_values("r2_mean")
    if "delta_ci_low" in single:
        save_fig(ctx, S, "ablation_single_add", viz.bar_ci(single, "families", "delta_r2", "delta_ci_low", "delta_ci_high",
                                                          "Delta R2 of adding one family to the base (paired bootstrap 95% CI)"), single)
        item = single[single["families"].str.contains("ITEM")]
        if not item.empty:
            r = item.iloc[0]
            ctx.registries.update_hypothesis("H01", "supported" if r["delta_ci_low"] > 0 else "not_supported", S,
                                             f"base+ITEM delta R2={r['delta_r2']:.4f}", effect=float(r["delta_r2"]),
                                             ci_low=float(r["delta_ci_low"]), ci_high=float(r["delta_ci_high"]))
    # registry: family-level ablation contribution
    contrib = [{"feature": c, "ablation_single_add_delta_r2": float(single.loc[single["families"].str.endswith(c.split("__")[0]), "delta_r2"].max())
                if "delta_r2" in single and single["families"].str.endswith(c.split("__")[0]).any() else None}
               for c in Xtr.columns]
    ctx.registries.features.upsert(contrib)
    return {"n_models": len(bench), "best": ok.sort_values("r2_mean").iloc[-1]["name"] if not ok.empty else None,
            "best_r2": float(ok["r2_mean"].max()) if not ok.empty else None, "ablation_evals": int(table["evals_used"].iloc[0])}
