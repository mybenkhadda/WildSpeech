"""Section 04 - item difficulty and response intelligence.

Scientific question: are some literacy assessment items substantially more
informative than others?

All statistics use observed responses only (non-attempted items are never
counted as incorrect). Item parameters here are fitted on all training
students for exploratory purposes; predictive stages refit them per fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import irt
from . import visualization as viz
from .stats_utils import bh_fdr, partial_corr, safe_corr


def item_registry(clips: pd.DataFrame, students: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = clips.dropna(subset=["item_key"]).copy()
    has_correct = "correct" in d.columns and d["correct"].notna().any()
    g = d.groupby("item_key")
    reg = pd.DataFrame({"task_family": g["task_family"].first(), "n_exposures": g.size(), "n_students": g["student_id"].nunique()})
    for c in ("item_id", "expected_text"):
        if c in d.columns:
            reg[c] = g[c].first()
    n_task_students = d.groupby("task_family")["student_id"].nunique()
    reg["exposure_rate_within_task"] = reg["n_students"] / reg["task_family"].map(n_task_students)
    if not has_correct:
        return reg.reset_index()
    reg["n_scored"] = g["correct"].count()
    reg["missing_correctness_rate"] = 1 - reg["n_scored"] / reg["n_exposures"]
    reg["p_correct"] = g["correct"].mean()
    k = 2.0
    task_p = d.groupby("task_family")["correct"].mean()
    prior = reg["task_family"].map(task_p)
    reg["p_correct_smoothed"] = (reg["p_correct"] * reg["n_scored"] + prior * k) / (reg["n_scored"] + k)
    reg["empirical_difficulty_logit"] = -np.log(reg["p_correct_smoothed"].clip(1e-3, 1 - 1e-3) / (1 - reg["p_correct_smoothed"].clip(1e-3, 1 - 1e-3)))
    reg["rare"] = reg["n_scored"] < int(cfg["item_rare_threshold"])
    d = d.merge(students[["student_id", "target", "school_id"] + [c for c in students.columns if c.startswith("theta_")]],
                on="student_id", how="left")
    task_sum = d.groupby(["student_id", "task_family"])["correct"].agg(["sum", "count"]).rename(columns={"sum": "ts", "count": "tc"})
    d = d.join(task_sum, on=["student_id", "task_family"])
    d["rest_score"] = (d["ts"] - d["correct"]) / (d["tc"] - 1).where(d["tc"] > 1)
    rows = []
    for item, gi in d.groupby("item_key"):
        r = {"item_key": item}
        r["point_biserial_rest"], r["p_rest"], _ = safe_corr(gi["correct"], gi["rest_score"], "pearson", 10)
        r["point_biserial_target"], r["p_target"], r["n_target"] = safe_corr(gi["correct"], gi["target"], "pearson", 10)
        task_theta = f"theta_{gi['task_family'].iloc[0]}"
        if task_theta in gi.columns:
            r["partial_r_target_given_task_theta"], r["p_partial"], _ = partial_corr(gi["correct"], gi["target"], gi[[task_theta]], "pearson", 15)
        if "school_id" in gi.columns:
            sp = gi.groupby("school_id")["correct"].agg(["mean", "count"])
            sp = sp[sp["count"] >= 5]
            r["difficulty_sd_across_schools"] = float(sp["mean"].std()) if len(sp) >= 2 else np.nan
            r["n_schools_ge5"] = int(len(sp))
        if "theta_mean" in gi.columns or task_theta in gi.columns:
            th = gi[task_theta] if task_theta in gi.columns else gi["theta_mean"]
            try:
                grp = pd.qcut(th, 3, labels=["low", "mid", "high"], duplicates="drop")
                ap = gi.groupby(grp, observed=True)["correct"].agg(["mean", "count"])
                ap = ap[ap["count"] >= 5]
                r["difficulty_sd_across_ability"] = float(ap["mean"].std()) if len(ap) >= 2 else np.nan
                for lvl in ap.index:
                    r[f"p_correct_ability_{lvl}"] = float(ap.loc[lvl, "mean"])
            except ValueError:
                pass
        rows.append(r)
    stats = pd.DataFrame(rows).set_index("item_key")
    reg = reg.join(stats)
    for p in ("p_rest", "p_target", "p_partial"):
        if p in reg.columns:
            reg[p.replace("p_", "q_", 1)] = bh_fdr(reg[p])
    return reg.reset_index()


def item_redundancy(clips: pd.DataFrame, cfg: dict, max_items: int = 400) -> pd.DataFrame:
    d = clips.dropna(subset=["item_key", "correct"])
    top = d["item_key"].value_counts().head(max_items).index
    M = d[d["item_key"].isin(top)].pivot_table(index="student_id", columns="item_key", values="correct", aggfunc="mean")
    E = M.notna().astype(float)
    co = E.T @ E
    rows = []
    min_co = int(cfg["item_redundancy_min_coexposure"])
    cols = list(M.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            n = co.loc[a, b]
            if n < min_co:
                continue
            pair = M[[a, b]].dropna()
            if pair[a].std() == 0 or pair[b].std() == 0:
                continue
            rows.append({"item_a": a, "item_b": b, "n_coexposed": int(n), "phi": float(pair[a].corr(pair[b]))})
    return pd.DataFrame(rows), co


def stage_item_intelligence(ctx) -> dict:
    cfg = ctx.config
    clips = da.clips(ctx, "train")
    sm = da.students(ctx, "train")
    scr = da.screening(ctx) if ctx.stage_available("03") else None
    if scr is not None and "theta_mean" in scr.columns:
        sm = sm.merge(scr[["theta_mean"]], left_on="student_id", right_index=True, how="left")
    if clips["item_key"].isna().all():
        ctx.save_json({"status": "NO_ITEM_IDENTIFIERS"}, "item_status")
        return {"outputs": ["item_status"], "summary": {"status": "NO_ITEM_IDENTIFIERS"}}
    reg = item_registry(clips, sm, cfg)
    outputs = []
    has_correct = "correct" in clips.columns and clips["correct"].notna().sum() > 0
    params, studs, comparison, feats = [], [], [], []
    if has_correct and cfg.get("enable_irt"):
        results = irt.fit_task_irt(clips, cfg)
        for task, res in results.items():
            if res.get("status") != "OK":
                comparison.append({"task_family": task, "status": res.get("status")})
                continue
            for m, fit in res["fits"].items():
                if fit is None:
                    comparison.append({"task_family": task, "model": m, "status": "NOT_IDENTIFIABLE_SAMPLE_SIZE"})
                    continue
                comparison.append({"task_family": task, "model": m, "status": "OK", "loglik": fit.loglik, "aic": fit.aic,
                                   "bic": fit.bic, "n_params": fit.n_params, "n_obs": fit.n_obs, "converged": fit.converged,
                                   "selected": m == res["selected"]})
                params.append(fit.items.assign(task_family=task, model=m, selected_model=res["selected"]))
                studs.append(fit.students.assign(task_family=task, model=m, selected_model=res["selected"]))
            sel = res["fits"][res["selected"]]
            g = clips[clips["task_family"] == task]
            feats.append(irt.difficulty_features(g, sel.items, sel.students, prefix=f"irt_{task}_").set_index("student_id"))
        item_params = pd.concat(params, ignore_index=True) if params else pd.DataFrame(columns=["item_key", "b", "a", "model", "selected_model"])
        ctx.save_df(item_params, "irt_item_params", also_csv=True)
        ctx.save_df(pd.concat(studs, ignore_index=True) if studs else pd.DataFrame(), "irt_student_params")
        ctx.save_df(pd.DataFrame(comparison), "irt_model_comparison", fmt="csv")
        outputs += ["irt_item_params", "irt_student_params", "irt_model_comparison"]
        if feats:
            sf = pd.concat(feats, axis=1)
            for stat in ("difficulty_weighted_accuracy", "response_residual_mean", "theta", "unexpected_easy_item_error_rate"):
                cols = [c for c in sf.columns if c.endswith(stat)]
                if cols:
                    sf[f"irt_all_{stat}"] = sf[cols].mean(axis=1)
            ctx.save_df(sf.reset_index(), "irt_student_features")
            outputs.append("irt_student_features")
        sel_params = item_params[item_params["model"] == item_params["selected_model"]]
        reg = reg.merge(sel_params[["item_key", "b", "b_se", "a"]].rename(columns={"b": "irt_b", "b_se": "irt_b_se", "a": "irt_a"}),
                        on="item_key", how="left")
        _irt_figures(ctx, item_params, pd.concat(studs, ignore_index=True) if studs else pd.DataFrame())
    ctx.save_df(reg, "item_registry", also_csv=True)
    outputs.append("item_registry")
    if has_correct:
        red, co = item_redundancy(clips, cfg)
        ctx.save_df(red, "item_redundancy", fmt="csv")
        outputs.append("item_redundancy")
        _item_figures(ctx, reg, red, co, clips)
    top = reg.sort_values("point_biserial_target", ascending=False).head(10) if "point_biserial_target" in reg.columns else reg.head(0)
    return {"outputs": outputs, "summary": {"n_items": int(len(reg)), "n_rare": int(reg.get("rare", pd.Series(dtype=bool)).sum()),
                                            "top_target_items": top[["item_key", "point_biserial_target"]].to_dict("records") if len(top) else []}}


def _irt_figures(ctx, params: pd.DataFrame, studs: pd.DataFrame) -> None:
    grid = np.linspace(-4, 4, 161)
    sel = params[params["model"] == params["selected_model"]]
    for task, g in sel.groupby("task_family"):
        fig, axes = viz.new_figure(1, 3, (5, 4))
        show = g.iloc[np.linspace(0, len(g) - 1, min(12, len(g))).astype(int)] if len(g) else g
        for _, r in show.iterrows():
            p = 1 / (1 + np.exp(-r["a"] * (grid - r["b"])))
            axes[0, 0].plot(grid, p, lw=0.8)
            axes[0, 1].plot(grid, irt.item_information(grid, r["a"], r["b"]), lw=0.8)
        viz.finalize_axis(axes[0, 0], f"{task}: item characteristic curves (subset)", "theta", "P(correct)", len(g))
        viz.finalize_axis(axes[0, 1], f"{task}: item information curves (subset)", "theta", "information", len(g))
        test_info = sum(irt.item_information(grid, r["a"], r["b"]) for _, r in g.iterrows())
        axes[0, 2].plot(grid, test_info)
        if not studs.empty:
            st = studs[(studs["task_family"] == task) & (studs["model"] == studs["selected_model"])]
            ax2 = axes[0, 2].twinx()
            ax2.hist(st["theta"], bins=30, alpha=0.3, color="grey")
            ax2.set_ylabel("students")
        viz.finalize_axis(axes[0, 2], f"{task}: total information and ability distribution", "theta", "information", len(g))
        ctx.figure(fig, f"irt_curves_{task}", f"IRT curves for {task} ({g['selected_model'].iloc[0]})",
                   "Where on the ability scale are items informative?", "irt_item_params", len(g),
                   f"Selected by BIC; median b={g['b'].median():.2f}, median a={g['a'].median():.2f}.")
    fig, axes = viz.new_figure(1, 1)
    ax = axes[0, 0]
    for task, g in sel.groupby("task_family"):
        ax.errorbar(g["b"], g["p_correct"], xerr=1.96 * g["b_se"], fmt="o", ms=3, alpha=0.6, label=task)
    ax.legend(fontsize=7)
    viz.finalize_axis(ax, "IRT difficulty (95% CI) vs observed proportion correct", "b", "p correct", len(sel))
    ctx.figure(fig, "irt_difficulty_uncertainty", "Item difficulty with uncertainty", "How precisely are items calibrated?",
               "irt_item_params", len(sel), "Error bars from the observed information at the MAP estimate.")


def _item_figures(ctx, reg: pd.DataFrame, red: pd.DataFrame, co: pd.DataFrame, clips: pd.DataFrame) -> None:
    fig, axes = viz.new_figure(1, 3, (5, 4))
    viz.hist_kde(axes[0, 0], reg["n_exposures"], "item exposure", "clips")
    viz.hist_kde(axes[0, 1], reg["p_correct"], "proportion correct")
    if "point_biserial_target" in reg.columns:
        viz.scatter_fit(axes[0, 2], reg["empirical_difficulty_logit"], reg["point_biserial_target"],
                        "empirical difficulty (logit)", "point-biserial with outcome", "difficulty vs outcome association")
    ctx.figure(fig, "item_exposure_difficulty", "Item exposure, difficulty and outcome association",
               "Are some items more informative?", "item_registry", int(len(reg)),
               f"{int(reg['rare'].sum())} rare items (< threshold scored responses).")
    if len(red):
        fig, axes = viz.new_figure(1, 1)
        viz.hist_kde(axes[0, 0], red["phi"], "phi between co-exposed items")
        ctx.figure(fig, "item_redundancy", "Item redundancy (phi coefficients)", "Are items redundant?", "item_redundancy",
                   int(len(red)), f"{int((red['phi'] > 0.7).sum())} pairs with phi > 0.7.")
    top = co.iloc[:40, :40]
    if top.shape[0] > 1:
        fig, axes = viz.new_figure(1, 1, (8, 7))
        viz.heatmap(axes[0, 0], top, "Item co-occurrence (students exposed to both)", cmap="viridis", cbar_label="students")
        ctx.figure(fig, "item_cooccurrence", "Item co-occurrence matrix (40 most frequent items)",
                   "Which items are administered together?", "clip_master", int(clips["student_id"].nunique()),
                   "Counts reflect adaptive exposure.")
