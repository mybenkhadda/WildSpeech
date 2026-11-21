"""P09 — experimental composite literacy indices: fold-fitted construction, incremental validity, reliability."""
from __future__ import annotations

import pandas as pd

from .. import viz
from ..analysis.indices import DISCLAIMER, evaluate_index_cv, split_half_reliability
from ..stats import delta_r2_bootstrap
from ._common import get_runner, save_fig

S = "P09"


def run(ctx) -> dict:
    runner = get_runner(ctx, with_audio=True)
    recipes = ctx.cfg.get_path("indices.recipes", {}) or {}
    nb = int(ctx.cfg.get_path("validation.bootstrap", 1000))
    rows, best = [], None
    for name, rec in recipes.items():
        oof, comps = evaluate_index_cv(runner, name, rec["include"])
        if oof.empty:
            rows.append({"index": name, "status": "insufficient components", "description": rec.get("description")})
            continue
        ctx.save_parquet(oof, S, f"index_oof_{name}")
        ctx.save_table(comps, S, f"index_components_{name}")
        m = oof.groupby("subject_id").agg(y=("y_true", "first"), b=("pred_base", "mean"), i=("pred_index", "mean"))
        est, lo, hi = delta_r2_bootstrap(m["y"], m["b"], m["i"], nb, ctx.seed)
        rel = split_half_reliability(runner.builder, list(runner.y.index), runner.y, name, rec["include"])
        sign_cols = [c for c in comps.columns if c.startswith("sign__")]
        sign_stab = float(comps[sign_cols].apply(lambda s: abs(s.mean())).mean()) if sign_cols else float("nan")
        row = {"index": name, "status": "ok", "description": rec.get("description"), "delta_r2": est, "ci_low": lo, "ci_high": hi,
               "cronbach_alpha_mean": float(comps["alpha"].mean()), "n_components_mean": float(comps["n_components"].mean()),
               "sign_stability": sign_stab, **{k: v for k, v in rel.items() if k != "index"}}
        rows.append(row)
        if best is None or est > best["delta_r2"]:
            best = row
    tab = pd.DataFrame(rows)
    ctx.save_table(tab, S, "index_validation")
    ctx.save_json({"disclaimer": DISCLAIMER}, S, "DISCLAIMER")
    ok = tab[tab.get("status", pd.Series(dtype=str)) == "ok"] if not tab.empty else tab
    if not ok.empty:
        save_fig(ctx, S, "index_incremental_r2", viz.forest(ok.sort_values("delta_r2", ascending=False), "delta_r2", "ci_low", "ci_high",
                                                           "index", "Incremental CV R2 over screening (95% bootstrap CI)"), ok)
        n_sup = int((ok["ci_low"] > 0).sum())
        ctx.registries.update_hypothesis("H08", "supported" if n_sup else "not_supported", S,
                                         f"{n_sup}/{len(ok)} indices with delta-R2 CI > 0; best {best['index']}",
                                         effect=float(best["delta_r2"]), ci_low=float(best["ci_low"]), ci_high=float(best["ci_high"]))
    return {"n_indices": len(tab), "best": best["index"] if best else None}
