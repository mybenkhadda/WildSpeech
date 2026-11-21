"""P28 — compact phonetic representations (strictly nested, student-disjoint).

For every available V3 family and every reducer (select / PCA / PLS / reduced-rank ridge; attention-MIL for
embeddings when enabled) at 5 / 10 / 20 / 50 / 100 added features: nested residual fusion on top of the GAM
(continuous shrinkage grid, fallback recorded), ΔR² with paired bootstrap CI on the primary scheme.
All reducers are fitted inside each training fold on inner-OOF baseline residuals; provider tables are fitted
on training-fold students only. The best (method, budget) found here is EXPLORATORY: P29's headline rows use the
pre-specified defaults (v3.ablation.default_method / default_budget) so selection never leaks into reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.ablation import FAMILY_EXPERIMENTS, AblationRunner, add_inference, budget_specs
from ..evaluation.nested_asr_cv import V3Evaluator, V3Spec, result_row
from ..v3 import viz as V
from ..v3.common import available_schemes, begin, not_executed
from ..v3.providers import build_all
from ..v3.registry import save_figure_v3

S = "P28"


def run(ctx) -> dict:
    from ..asr.outputs import require_outputs

    reg, v3r = begin(ctx, S)
    ok, _acc, why = require_outputs(ctx, S, "any", 1, ("train",))
    if not ok:
        return not_executed(ctx, S, why)
    r = ctx.cfg.get_path("v3.representations", {}) or {}
    ev = V3Evaluator(ctx, S, providers={})
    provs, avail, pm, models = build_all(ctx, ev)
    ctx.save_table(avail, S, "family_availability")
    for k, p in provs.items():
        ev.add_provider(k, p)
    fams = {exp: f for exp, f in FAMILY_EXPERIMENTS.items() if all(x in provs for x in f)}
    if not fams:
        return not_executed(ctx, S, "no V3 family provider available (P25-P27 outputs missing)")
    prim = available_schemes(ctx, [ctx.cfg.get_path("v3.ablation.primary_scheme", "repeated_kfold")])
    if not prim:
        return not_executed(ctx, S, "primary scheme missing from the validation registry")
    sc = prim[0]
    base = {sc: ev.evaluate(ev.baseline_v3(), sc)}
    alphas = tuple(ctx.cfg.get_path("v3.audit.continuous_alphas", [0, .1, .25, .5, 1]))
    methods = [m for m in r.get("methods", ["select", "pca"]) if m in ("select", "pca", "pls", "rrr")]
    run_ = AblationRunner(ev, base, int(ctx.cfg.get_path("v3.ablation.max_evals", 600)),
                          float(ctx.cfg.get_path("v3.ablation.eval_time_budget_s", 14400)), ctx.logger)
    for sp in budget_specs(fams, [int(b) for b in r.get("budgets", [5, 10, 20, 50, 100])], methods, alphas):
        run_.add(sp, [sc], 10 if sp.budget == 20 else 20)
    if "mil" in r.get("methods", []) and "ASRE_MIL" in provs:
        run_.add(V3Spec("ASR08_MIL_RESIDUAL", "RIDGE", ["ASRE_MIL"], None, "none", "residual", alphas=alphas,
                        group="budget", experiment="ASR08"), [sc], 15)
    rows, sched = run_.run()
    ctx.save_table(sched, S, "schedule")
    rows = add_inference(rows, float(ctx.cfg.get_path("v3.ablation.fdr_alpha", 0.05)))
    rows = pd.concat([pd.DataFrame([result_row(base[sc], {"delta_r2": 0.0})]), rows], ignore_index=True)
    ctx.save_table(rows, S, "representation_results")
    for x in rows.to_dict("records"):
        v3r.log_ablation(f"{S}:{x['name']}:{x['scheme']}", stage=S, kind="representation", **{k: x.get(k) for k in
                         ("experiment", "name", "scheme", "status", "method", "budget", "providers", "delta_r2",
                          "delta_ci_low", "delta_ci_high", "q", "fallback_folds")})
    bud = rows[rows["group"] == "budget"].dropna(subset=["delta_r2"]).copy()
    if not bud.empty:
        bud["family_method"] = bud["experiment"] + " · " + bud["method"]
        for meth, g in bud.groupby("method"):
            save_figure_v3(ctx, S, f"budget_curve_{meth}", V.lines(g, "budget", "delta_r2", "experiment",
                           f"ΔR² vs added features ({meth}; nested residual fusion)", "delta_ci_low", "delta_ci_high",
                           xlabel="added features", ylabel="ΔR² vs GAM", ref=0.0, logx=True), g,
                           f"Representation budget curve — {meth}")
        at20 = bud[bud["budget"] == 20]
        if not at20.empty:
            save_figure_v3(ctx, S, "method_comparison_b20", V.forest(at20.assign(label=at20["experiment"] + " · " + at20["method"]),
                           "delta_r2", "delta_ci_low", "delta_ci_high", "label", "Reducers at 20 added features",
                           group="method"), at20, "Representation method comparison")
        rep = bud[bud["status"] == "ok"]
        if not rep.empty:
            best = rep.sort_values("delta_r2", ascending=False).iloc[0]
            v3r.update_hypothesis("V3H07", "supported" if int(best["budget"]) <= 20 else "not_supported", S,
                                  f"best {best['experiment']} {best['method']} budget={int(best['budget'])} ΔR²={best['delta_r2']:.4f}",
                                  tier="exploratory", effect=float(best["delta_r2"]))
            best_cfg = rep.sort_values("delta_r2", ascending=False).groupby("experiment").head(1)[["experiment", "method", "budget", "delta_r2"]]
            ctx.save_table(best_cfg, S, "exploratory_best_config")
        fb = bud.groupby("experiment")["status"].apply(lambda s: float((s == "fallback_baseline").mean())).reset_index(name="fallback_share")
        ctx.save_table(fb, S, "fallback_share_by_family")
    return {"families": list(fams), "primary_model": pm, "models": models, "n_results": int(len(rows))}
