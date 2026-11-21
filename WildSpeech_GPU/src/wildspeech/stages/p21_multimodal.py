"""P21 — compact multimodal student-level regression (budget-controlled).

Every experiment adds a small, fold-fitted representation to the strongest validated tabular comparator:
  1. single-family incremental value on all student schemes (nested residual fusion, default budget)
  2. feature-budget curves (5 / 10 / 20 / 50 / 100 added features) per family and for all new families
  3. model comparison (Ridge, Elastic Net, GAM, PLS, CatBoost, XGBoost, LightGBM) at the default budget
  4. fusion comparison: early, late (convex), nested residual, task-specific pooling (repaired task ensemble),
     regularised stacking, domain mixture-of-experts (partial pooling)
  5. pairwise family combinations and leave-one-family-out
  6. redesigned literacy indices with fold-validated component directions
  7. metadata-only soft correctness as a control
Outputs: model_comparison_p21, ablation table, family_value (genuine incremental value verdicts), budget curve,
registries (model_comparison.csv, ablation_results.csv), specs.json for P22.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as ss

from ..features.compact_representation import IndexTransformer, match_columns
from ..models.compact_regression import CompactSpec, StudentLevelEvaluator
from ..stats import bh_fdr
from ..v2 import visualization_v2 as V
from ..v2.providers import build_providers
from ..v2.registry import save_figure_v2
from ..v2.scheduler import ExperimentScheduler
from ._v2common import begin, evaluate_against, result_row, schemes

S = "P21"
CONTROL = "SOFTMETA"


def _p_from_ci(est, lo, hi) -> float:
    if not all(np.isfinite([est, lo, hi])) or hi <= lo:
        return np.nan
    se = (hi - lo) / (2 * 1.959964)
    return float(2 * ss.norm.sf(abs(est) / se))


def run(ctx) -> dict:
    reg, v2r = begin(ctx, S)
    m = ctx.cfg.get_path("v2.multimodal", {}) or {}
    ev_cfg = ctx.cfg.get_path("v2.evaluation", {}) or {}
    bud = int(ev_cfg.get("default_budget", 20))
    provs = build_providers(ctx)
    ev = StudentLevelEvaluator(ctx, S, providers=provs)
    ctx.save_table(pd.DataFrame([{"provider": k, "name": p.name, "family": p.family, "available": p.available(),
                                  "fingerprint": p.p.fingerprint()} for k, p in provs.items()]), S, "providers")
    sch = schemes(ctx)
    base = {sc: ev.evaluate(ev.baseline_spec(), sc) for sc in sch}
    # ---- baseline verification against the stored V1 result
    ver = {"comparator_model": ev.baseline["model"], "families": "+".join(ev.baseline["families"]), "source": ev.baseline.get("source"),
           "v1_r2_mean": ev.baseline.get("r2_mean"), "v2_r2_mean": base["repeated_kfold"].summary.get("r2_mean") if "repeated_kfold" in base else None}
    ver["abs_diff"] = abs(ver["v2_r2_mean"] - ver["v1_r2_mean"]) if ver["v1_r2_mean"] is not None and ver["v2_r2_mean"] is not None else None
    ctx.save_table(pd.DataFrame([ver]), S, "baseline_verification")
    rows, specs = [], {}
    from ..v2.baseline import conservative

    cons = conservative(ctx)
    cspec = CompactSpec("V2_CONSERVATIVE", cons["model"], base_families=list(cons["families"]))
    specs["V2_CONSERVATIVE"] = cspec.as_dict()
    specs["V2_BASELINE"] = ev.baseline_spec().as_dict()
    for sc in sch:
        r = ev.evaluate(cspec, sc)
        d = ev.compare(base[sc], r) if r.status == "ok" else {}
        rows.append({**result_row(r, d), "group": "conservative"})
        rows.append({**result_row(base[sc], {"delta_r2": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0}), "group": "comparator"})

    fams = [k for k in ("HARM", "SOFT", "PHON", "REF", "FM", "CPX") if k in provs and provs[k].available()]
    allnew = list(fams)
    sched = ExperimentScheduler(int(ev_cfg.get("max_evals", 400)), float(ev_cfg.get("eval_time_budget_s", 7200)), ctx.logger)

    def add(spec: CompactSpec, scs: list[str], group: str, priority: int):
        specs[spec.name] = spec.as_dict()

        def task():
            rws, _ = evaluate_against(ev, spec, base, scs)
            for r_ in rws:
                r_["group"] = group
            rows.extend(rws)
            return len(rws)
        sched.add(spec.name, task, priority, group, cost=len(scs))

    prim = ["repeated_kfold"] if "repeated_kfold" in sch else sch[:1]
    if CONTROL in provs:
        add(CompactSpec("P21_CONTROL_SOFTMETA", "RIDGE", providers=[CONTROL], budget=bud, fusion="residual"), sch, "control", 5)
    for f in fams:
        add(CompactSpec(f"P21_SINGLE_{f}", "RIDGE", providers=[f], budget=bud, fusion="residual"), sch, "single_family", 10)
    if (m.get("indices", {}) or {}).get("enabled", True) and allnew:
        add(CompactSpec("P21_INDICES", "RIDGE", providers=allnew, indices=True, fusion="residual"), sch, "indices", 15)
    if len(allnew) > 1:
        add(CompactSpec("P21_ALL_RESIDUAL_RIDGE", "RIDGE", providers=allnew, budget=bud, fusion="residual"), sch, "fusion", 18)
    for b in m.get("budgets", [5, 10, 20, 50, 100]):
        for f in fams + (["ALL"] if len(allnew) > 1 else []):
            pv = allnew if f == "ALL" else [f]
            add(CompactSpec(f"P21_BUDGET_{f}_{b}", "RIDGE", providers=pv, budget=int(b), fusion="residual"), prim, "budget", 20)
    if allnew:
        for fu in m.get("fusions", []):
            if fu == "residual":
                continue
            if fu == "moe" and not _moe_justified(ctx, int(m.get("moe_min_domain_size", 40))):
                continue
            add(CompactSpec(f"P21_FUSION_{fu.upper()}", "RIDGE", providers=allnew, budget=bud, fusion=fu), prim + (["school_kfold"] if "school_kfold" in sch else []), "fusion", 25)
        for mod in m.get("models", []):
            add(CompactSpec(f"P21_MODEL_{mod}_EARLY", mod, providers=allnew, budget=bud, fusion="early"), prim, "model", 30)
            add(CompactSpec(f"P21_MODEL_{mod}_RESIDUAL", mod, providers=allnew, budget=bud, fusion="residual"), prim, "model", 31)
    if m.get("leave_one_out", True) and len(allnew) > 2:
        for f in allnew:
            rest = [x for x in allnew if x != f]
            add(CompactSpec(f"P21_LOO_minus_{f}", m.get("ablation_model", "RIDGE"), providers=rest,
                            budget=int(m.get("ablation_budget", bud)), fusion="residual"), prim, "leave_one_out", 35)
    if m.get("pairwise", True) and len(fams) > 1:
        for a, b in itertools.combinations(fams, 2):
            add(CompactSpec(f"P21_PAIR_{a}+{b}", m.get("ablation_model", "RIDGE"), providers=[a, b],
                            budget=int(m.get("ablation_budget", bud)), fusion="residual"), prim, "pairwise", 40)
    ctx.save_table(sched.run(), S, "schedule")
    res = pd.DataFrame(rows)
    ctx.save_table(res, S, "all_results")
    ctx.save_json(specs, S, "specs")
    mc = res[res["scheme"] == "repeated_kfold"].copy() if not res.empty else res
    ctx.save_table(mc, S, "model_comparison_p21")
    for r in mc.to_dict("records"):
        v2r.log_model(f"{S}:{r['name']}", stage=S, name=r["name"], group=r.get("group"), model=r.get("model"), fusion=r.get("fusion"),
                      providers=r.get("providers"), budget=r.get("budget"), r2_mean=r.get("r2_mean"), r2_sd=r.get("r2_sd"),
                      r2_pooled=r.get("r2_pooled"), delta_r2=r.get("delta_r2"), delta_ci_low=r.get("delta_ci_low"),
                      delta_ci_high=r.get("delta_ci_high"), rmse=r.get("rmse"), mae=r.get("mae"),
                      comparator=f"{ev.baseline['model']}:{'+'.join(ev.baseline['families'])}")
    abl = res[res["group"].isin(["single_family", "budget", "pairwise", "leave_one_out", "indices", "control"])] if not res.empty else res
    ctx.save_table(abl, S, "ablations")
    for r in abl.to_dict("records"):
        v2r.log_ablation(f"{S}:{r['name']}:{r['scheme']}", stage=S, kind=r.get("group"), name=r["name"], providers=r.get("providers"),
                         budget=r.get("budget"), scheme=r["scheme"], r2_mean=r.get("r2_mean"), delta_r2=r.get("delta_r2"),
                         delta_ci_low=r.get("delta_ci_low"), delta_ci_high=r.get("delta_ci_high"))

    # ---- family value verdicts
    fv = []
    for f in fams:
        a = res[(res["name"] == f"P21_SINGLE_{f}")]
        if a.empty:
            continue
        rk = a[a["scheme"] == "repeated_kfold"]
        if rk.empty:
            continue
        rk = rk.iloc[0]
        school = a[a["scheme"] == "school_kfold"]
        fv.append({"family": f, "delta_r2": rk.get("delta_r2"), "delta_ci_low": rk.get("delta_ci_low"), "delta_ci_high": rk.get("delta_ci_high"),
                   "p": _p_from_ci(rk.get("delta_r2", np.nan), rk.get("delta_ci_low", np.nan), rk.get("delta_ci_high", np.nan)),
                   "delta_school": float(school.iloc[0]["delta_r2"]) if not school.empty and pd.notna(school.iloc[0].get("delta_r2")) else np.nan,
                   "r2_mean": rk.get("r2_mean")})
    fv = pd.DataFrame(fv)
    if not fv.empty:
        fv["q"] = bh_fdr(fv["p"].to_numpy(float))
        alpha = float(ev_cfg.get("fdr_alpha", 0.05))

        def verdict(r):
            if r["delta_ci_low"] > 0 and r["q"] < alpha and (not np.isfinite(r["delta_school"]) or r["delta_school"] > 0):
                return "genuine_incremental_value"
            if r["delta_r2"] > 0 and (r["delta_ci_low"] > 0 or (np.isfinite(r["delta_school"]) and r["delta_school"] > 0)):
                return "suggestive"
            return "no_incremental_value"
        fv["verdict"] = fv.apply(verdict, axis=1)
    ctx.save_table(fv, S, "family_value")
    if not fv.empty:
        v2r.update_hypothesis("V2H11", "supported" if (fv["verdict"] == "genuine_incremental_value").any() else "not_supported", S,
                              "families with genuine incremental value: " + ",".join(fv.loc[fv["verdict"] == "genuine_incremental_value", "family"]),
                              effect=float(fv["delta_r2"].max()))
        save_figure_v2(ctx, S, "family_value", V.forest(fv.sort_values("delta_r2", ascending=False), "delta_r2", "delta_ci_low", "delta_ci_high",
                       "family", "Incremental R² of each representation over the strongest tabular comparator"), fv,
                       "Family-level incremental value (paired bootstrap 95% CI)")
    # ---- budget curve
    bc = res[(res["group"] == "budget") & (res["scheme"] == "repeated_kfold")].copy() if not res.empty else pd.DataFrame()
    if not bc.empty:
        bc["family"] = bc["name"].str.split("_").str[2]
        save_figure_v2(ctx, S, "budget_curve", V.line_ci(bc, "budget", "delta_r2", "delta_ci_low", "delta_ci_high", "family",
                       "Feature budget vs incremental R²", ref=0.0, xlabel="added features"), bc, "Feature dimensionality vs R²")
        best_b = bc.sort_values("delta_r2", ascending=False).iloc[0]
        v2r.update_hypothesis("V2H12", "supported" if int(best_b["budget"]) <= 20 else "not_supported", S,
                              f"best budget {int(best_b['budget'])} ({best_b['family']}) ΔR²={best_b['delta_r2']:.4f}", tier="exploratory")
    # ---- model / fusion comparison figures
    mm = mc[mc["group"].isin(["model", "fusion", "indices", "comparator", "conservative"])].dropna(subset=["delta_r2"]) if not mc.empty else mc
    if not mm.empty:
        save_figure_v2(ctx, S, "model_comparison", V.forest(mm.sort_values("delta_r2", ascending=False), "delta_r2", "delta_ci_low", "delta_ci_high",
                       "name", "Models and fusions: ΔR² over the comparator"), mm, "Model comparison and uncertainty")
    pw = mc[mc["group"] == "pairwise"] if not mc.empty else mc
    if not pw.empty:
        mat = pd.DataFrame(index=fams, columns=fams, dtype=float)
        for r in pw.to_dict("records"):
            a, b = r["name"].replace("P21_PAIR_", "").split("+")
            mat.loc[a, b] = mat.loc[b, a] = r["delta_r2"]
        for f in fams:
            s = mc[mc["name"] == f"P21_SINGLE_{f}"]
            if not s.empty:
                mat.loc[f, f] = s.iloc[0]["delta_r2"]
        save_figure_v2(ctx, S, "pairwise_ablation", V.heatmap(mat, "Pairwise family ΔR² (diagonal = single family)", annotate=True), mat,
                       "Feature-family ablations")
    loo = mc[mc["group"] == "leave_one_out"] if not mc.empty else mc
    if not loo.empty:
        save_figure_v2(ctx, S, "leave_one_family_out", V.bar_ci(loo.sort_values("delta_r2"), "name", "delta_r2", "delta_ci_low", "delta_ci_high",
                       "Leave-one-family-out ΔR²"), loo, "Leave-one-family-out")
    # ---- index components (descriptive fit on all training students)
    if (m.get("indices", {}) or {}).get("enabled", True) and allnew:
        Xs = pd.concat([provs[f].table(ev.all_train, ev.all_train) for f in allnew], axis=1)
        it = IndexTransformer(m.get("indices", {}).get("recipes", {}), int(m.get("indices", {}).get("min_components", 2))).fit(Xs, ev.y.reindex(Xs.index).to_numpy())
        ctx.save_table(it.report_, S, "index_components")
        ctx.save_table(pd.DataFrame([{"index": k, "patterns": ";".join(v.get("include", [])),
                                      "n_matching_columns": len(match_columns(Xs.columns, v.get("include", [])))}
                                     for k, v in (m.get("indices", {}).get("recipes", {}) or {}).items()]), S, "index_recipes")
    return {"families": fams, "n_results": int(len(res)), "comparator": ev.baseline.get("source")}


def _moe_justified(ctx, min_size: int) -> bool:
    """A domain mixture is evaluated only when at least two recording domains have >= min_size students."""
    st = ctx.bundle().train_students
    if "original_sample_rate" not in st:
        return False
    vc = (st["original_sample_rate"].astype(float) < 16000).value_counts()
    return bool((vc >= min_size).sum() >= 2)
