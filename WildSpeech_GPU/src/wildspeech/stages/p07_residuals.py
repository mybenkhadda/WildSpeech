"""P07 — residual discovery: which signals explain screening-only out-of-fold prediction errors?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..analysis import residuals as R
from ..validation.metrics import summarize_oof
from ._common import load_crossfit, save_fig, screening_residuals

S = "P07"


def _hyp(ctx, hid: str, assoc: pd.DataFrame, family: str, alpha: float) -> None:
    d = assoc[assoc["family"] == family]
    if d.empty:
        ctx.registries.update_hypothesis(hid, "not_tested", S, f"no {family} features available")
        return
    n = int((d["q_resid"] < alpha).sum())
    top = d.sort_values("p_resid").iloc[0]
    ctx.registries.update_hypothesis(hid, "supported" if n else "not_supported", S,
                                     f"{n} {family} features with q<{alpha}; top {top['feature']}", effect=float(top["rho_resid"]),
                                     ci_low=float(top.get("rho_ci_low", np.nan)), ci_high=float(top.get("rho_ci_high", np.nan)),
                                     q_value=float(top["q_resid"]))


def run(ctx) -> dict:
    a = ctx.cfg.section("analysis")
    alpha = float(ctx.cfg.get_path("validation.fdr_alpha", 0.05))
    resid, oof = screening_residuals(ctx, refresh=True)
    ctx.save_json(summarize_oof(oof), S, "screening_only_cv")
    X = load_crossfit(ctx)
    X.index = X.index.astype(str)
    resid = resid.reindex(X.index)
    y = ctx.bundle().y.reindex(X.index)
    ctx.save_table(pd.DataFrame({"subject_id": resid.index, "resid": resid.values, "y": y.values}), S, "screening_residuals")
    feats = R.candidate_features(X)
    if not feats:
        return {"status": "no candidate features"}
    nb = int(a.get("bootstrap", 1000))
    assoc = R.residual_associations(X, resid, y, feats, nb, ctx.seed)
    top = assoc.head(int(a.get("top_k", 40)))["feature"].tolist()
    assoc = assoc.merge(R.permutation_check(X, resid, top, int(a.get("permutations", 1000)), ctx.seed), on="feature", how="left")
    rep = R.split_replication(X, resid, feats, int(a.get("discovery_repeats", 20)), int(a.get("top_k", 40)), alpha, ctx.seed)
    assoc = assoc.merge(rep, on="feature", how="left")
    ctx.save_table(assoc, S, "residual_associations")
    fam = R.family_summary(assoc, alpha)
    ctx.save_table(fam, S, "residual_family_summary")
    save_fig(ctx, S, "residual_family_max_rho", viz.bar_ci(fam.sort_values("max_abs_rho"), "family", "max_abs_rho", None, None,
                                                          "Strongest |rho| with screening residual per family"), fam)
    f30 = assoc.dropna(subset=["rho_ci_low"]).head(30)
    save_fig(ctx, S, "residual_forest_top30", viz.forest(f30, "rho_resid", "rho_ci_low", "rho_ci_high", "feature",
                                                        "Top residual associations (Spearman, bootstrap 95% CI)"), f30)
    pairs, matched = R.matched_comparison(X, resid, top, tuple(a.get("residual_quantiles", [0.2, 0.8])), float(a.get("matching_caliper_sd", 0.25)))
    ctx.save_table(pairs, S, "matched_pairs")
    ctx.save_table(matched, S, "matched_comparisons")
    adj = R.adjusted_associations(X, y, top)
    ctx.save_table(adj, S, "adjusted_associations_hc3")
    ctx.registries.features.upsert(assoc[["feature", "rho_resid", "q_resid", "partial_rho_outcome", "q_partial"]
                                         + [c for c in ("replication_rate", "perm_p") if c in assoc]].to_dict("records"))
    _hyp(ctx, "H02", assoc, "TEMPORAL", alpha)
    _hyp(ctx, "H10", assoc, "ALIGN", alpha)
    return {"n_features_tested": len(feats), "n_q_sig": int((assoc["q_resid"] < alpha).sum()), "n_matched_pairs": len(pairs)}
