"""P11 — multimodal learning: early/late fusion, nested residual learning, task ensembles, MoE and
(opt-in) neural MIL with attention pooling / lightweight transformer over each student's recordings."""
from __future__ import annotations

import pandas as pd

from .. import ID
from ..io_utils import atomic_write_parquet
from ..models.neural import StudentMILModel
from ..validation.metrics import incremental_r2, metric_row, summarize_oof
from ..validation.runner import ModelSpec, load_oof
from ._common import get_runner

S = "P11"


def _neural_cv(ctx, runner, arch: str) -> pd.DataFrame:
    tasks = sorted(ctx.bundle().all_clips["task"].unique())
    rows = []
    for f in runner.folds("repeated_kfold"):
        Xtr, Xva = runner.fold_data(f)          # fits the builder on this fold's training students
        runner.builder.fit(f.train_ids)          # make sure clip normalisation uses the same training fold
        clips_tr, cols = runner.builder.clip_level(f.train_ids)
        clips_va, _ = runner.builder.clip_level(f.valid_ids)
        clip_df = pd.concat([clips_tr, clips_va], ignore_index=True)
        ytr = runner.y.reindex(Xtr.index)
        model = StudentMILModel(arch, ctx.cfg, tasks, seed=ctx.seed + f.repeat * 31 + f.fold).fit(clip_df, cols, Xtr, ytr)
        p = model.predict(clip_df, Xva, list(Xva.index))
        rows.append(pd.DataFrame({ID: Xva.index, "repeat": f.repeat, "fold": f.fold, "y_true": runner.y.reindex(Xva.index).values,
                                  "y_pred": p, "fold_label": ""}))
    return pd.concat(rows, ignore_index=True)


def run(ctx) -> dict:
    runner = get_runner(ctx, with_audio=True)
    nb = int(ctx.cfg.get_path("validation.bootstrap", 1000))
    bench = ctx.read_stage_table("P10", "model_benchmark")
    best_name = bench[bench.status == "ok"].sort_values("r2_mean").iloc[-1]["name"] if bench is not None and not bench.empty else None
    best_oof = load_oof(ctx, "P10", best_name) if best_name else None
    specs = [ModelSpec("F_EARLY_RIDGE_ALL", "RIDGE", ["ALL"]),
             ModelSpec("F_EARLY_ENET_ALL", "ELASTIC_NET", ["ALL"]),
             ModelSpec("F_LATE_FAMILY_RIDGE", "LATE_FUSION", ["ALL"], selector_k=0),
             ModelSpec("F_TASK_ENSEMBLE_RIDGE", "TASK_ENSEMBLE", ["ALL"], selector_k=0),
             ModelSpec("F_RESIDUAL_STACK_RIDGE", "RESIDUAL_STACK", ["ALL"]),
             ModelSpec("F_RESIDUAL_STACK_LGBM", "RESIDUAL_STACK", ["ALL"], params={"base": "LIGHTGBM"}),
             ModelSpec("F_MOE_ALL", "MOE", ["ALL"])]
    rows, oofs = [], {}
    for sp in specs:
        if sp.model == "MOE" and not ctx.cfg.get_path("multimodal.neural_enabled", False):
            continue
        res = runner.evaluate(sp, "repeated_kfold", stage=S)
        runner.save_oof(res, S)
        rows.append({"name": sp.name, "model": sp.model, "deployable": True, "status": res.status, **res.summary})
        if res.status == "ok":
            oofs[sp.name] = res.oof
    if ctx.cfg.get_path("multimodal.neural_enabled", False):
        for arch in ("attention", "transformer"):
            name = f"N_MIL_{arch.upper()}"
            oof = _neural_cv(ctx, runner, arch)
            d = ctx.stage_dir(S) / "oof"
            d.mkdir(parents=True, exist_ok=True)
            atomic_write_parquet(oof, d / f"{name}__repeated_kfold.parquet")
            oofs[name] = oof
            rows.append({"name": name, "model": f"MIL_{arch}", "deployable": False, "status": "ok", **summarize_oof(oof)})
    tab = pd.DataFrame(rows)
    if best_oof is not None:
        deltas = [{"name": n, **incremental_r2(best_oof, o, nb, 0)} for n, o in oofs.items()]
        if deltas:
            tab = tab.merge(pd.DataFrame(deltas).rename(columns=lambda c: c if c == "name" else f"vs_best_p10_{c}"), on="name", how="left")
    ctx.save_table(tab, S, "multimodal_results")
    if best_oof is not None and not tab.empty and "vs_best_p10_delta_ci_low" in tab:
        top = tab.sort_values("r2_mean", ascending=False).iloc[0]
        ctx.registries.update_hypothesis("H09", "supported" if top["vs_best_p10_delta_ci_low"] > 0 else "not_supported", S,
                                         f"best fusion {top['name']} vs best P10 {best_name}: delta R2={top['vs_best_p10_delta_r2']:.4f}",
                                         effect=float(top["vs_best_p10_delta_r2"]), ci_low=float(top["vs_best_p10_delta_ci_low"]),
                                         ci_high=float(top["vs_best_p10_delta_ci_high"]))
    return {"n_models": len(tab), "best": tab.sort_values("r2_mean").iloc[-1]["name"] if not tab.empty else None}
