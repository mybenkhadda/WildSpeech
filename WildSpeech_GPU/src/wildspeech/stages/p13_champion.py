"""P13 — robust champion selection, regularised stacking, ensemble weights, uncertainty; writes the frozen
champion spec consumed by P14."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..features.assemble import cached_families
from ..io_utils import write_json
from ..models.champion import (candidate_matrix, conformal_intervals, cv_stack, final_q, r2_bootstrap, robust_ranking)
from ..validation.metrics import summarize_oof
from ._common import save_fig, student_meta
from .p12_failures import BASELINE, collect_oofs

S = "P13"


def _spec_lookup(ctx) -> dict[str, dict]:
    """name -> deployable ModelSpec dict (from P10/P11 result tables)."""
    out = {}
    bench = ctx.read_stage_table("P10", "model_benchmark")
    if bench is not None:
        for _, r in bench[bench.status == "ok"].iterrows():
            out[r["name"]] = {"name": r["name"], "model": r["model"], "families": str(r["families"]).split("+"), "params": {}, "selector_k": None}
    abl = ctx.read_stage_table("P10", "family_ablation")
    if abl is not None:
        m = ctx.cfg.get_path("models.ablation_model", "RIDGE")
        for fams in abl["families"].astype(str):
            out[f"ABL_{fams}"] = {"name": f"ABL_{fams}", "model": m, "families": fams.split("+"), "params": {}, "selector_k": None}
    mm = ctx.read_stage_table("P11", "multimodal_results")
    if mm is not None:
        params = {"F_RESIDUAL_STACK_LGBM": {"base": "LIGHTGBM"}}
        sel0 = {"F_LATE_FAMILY_RIDGE", "F_TASK_ENSEMBLE_RIDGE"}
        for _, r in mm[(mm.status == "ok") & (mm.get("deployable", True) == True)].iterrows():  # noqa: E712
            out[r["name"]] = {"name": r["name"], "model": r["model"], "families": ["ALL"], "params": params.get(r["name"], {}),
                              "selector_k": 0 if r["name"] in sel0 else None}
    return out


def run(ctx) -> dict:
    nb = int(ctx.cfg.get_path("validation.bootstrap", 1000))
    oofs = collect_oofs(ctx)
    specs = _spec_lookup(ctx)
    oofs = {k: v for k, v in oofs.items() if k in specs or k.startswith("N_")}
    if BASELINE not in oofs:
        raise RuntimeError("baseline OOF missing — run P10")
    summ = pd.DataFrame([{"name": k, **summarize_oof(v)} for k, v in oofs.items()])
    rank = robust_ranking(summ, oofs, BASELINE, lam=1.0, n_boot=nb)
    rank["deployable"] = rank["name"].isin(specs)
    ctx.save_table(rank, S, "candidate_ranking")
    dep = rank[rank["deployable"] & rank["beats_baseline"]]
    single = dep.iloc[0]["name"] if not dep.empty else BASELINE

    # ---- regularised stacking over the top deployable candidates (weights cross-validated over students)
    top = dep.head(5)["name"].tolist()
    if BASELINE not in top:
        top = [BASELINE] + top
    P, y = candidate_matrix({k: oofs[k] for k in top})
    first = oofs[BASELINE]
    folds_of = first[first["repeat"] == first["repeat"].min()].drop_duplicates("subject_id").set_index("subject_id")["fold"]
    folds_of.index = folds_of.index.astype(str)
    P.index, y.index = P.index.astype(str), y.index.astype(str)
    stack_pred, wtab = cv_stack(P, y, folds_of, "nnls")
    ctx.save_table(wtab, S, "stack_fold_weights")
    single_pred = P[single]
    rows = [{"candidate": f"single:{single}", **r2_bootstrap(y, single_pred, nb)},
            {"candidate": "stack:nnls_cv", **r2_bootstrap(y, stack_pred, nb)},
            {"candidate": f"baseline:{BASELINE}", **r2_bootstrap(y, P[BASELINE], nb)}]
    comp = pd.DataFrame(rows)
    ctx.save_table(comp, S, "champion_comparison")
    # prefer the single model unless stacking is clearly better (> 0.005 R2) — stability over complexity
    use_stack = comp.iloc[1]["r2"] > comp.iloc[0]["r2"] + 0.005 and len(top) > 1
    if use_stack:
        w = wtab[top].mean().clip(lower=0)
        w = w / w.sum() if w.sum() > 0 else w
        intercept = float(y.mean() - (P[top].mean() * w).sum())
        members = [specs[n] for n in top if w[n] > 0]
        weights = [float(w[n]) for n in top if w[n] > 0]
        champ_pred, name = stack_pred, "stack:" + "+".join(m["name"] for m in members)
    else:
        members, weights, intercept, champ_pred, name = [specs[single]], [1.0], 0.0, single_pred, single
    meta = student_meta(ctx).set_index("subject_id")
    groups = meta["instrument"] if "instrument" in meta else None
    conf = conformal_intervals(y, champ_pred, groups, (0.1, 0.2), folds_of)
    ctx.save_table(conf, S, "conformal_coverage")
    spec = {"name": name, "members": members, "weights": weights, "intercept": intercept,
            "audio_families": cached_families(ctx), "cv_r2": float(comp.iloc[1 if use_stack else 0]["r2"]),
            "interval_q90": final_q(y, champ_pred, 0.1), "interval_q80": final_q(y, champ_pred, 0.2),
            "baseline": BASELINE, "selection_rule": "robust score (mean - sd) among candidates whose delta-R2 CI > 0; "
                                                     "stack only if > +0.005 R2"}
    write_json(ctx.paths.models / "champion_spec.json", spec)
    write_json(ctx.paths.models / "champion_config.json", {"profile": ctx.profile, "stages": ["P14"], "config": ctx.cfg.to_dict()})
    ctx.save_json(spec, S, "champion_spec")
    save_fig(ctx, S, "candidate_ranking", viz.bar_ci(rank.head(25).iloc[::-1], "name", "r2_mean", None, None,
                                                    "Top candidates by CV R2 (mean over repeats)"), rank.head(25))
    return {"champion": name, "cv_r2": spec["cv_r2"], "stacked": bool(use_stack), "n_candidates": len(rank)}
