"""P26 — correct-response speech intelligence (central V3 experiment).

A. Clip level (training clips): correct vs incorrect profiles; correct-only confidence distributions by outcome
   tercile (descriptive); confidence vs fold-free item difficulty; does phonetics predict official correctness
   beyond META / TEMPORAL models (student-grouped CV, student-block bootstrap ΔAUROC; V3H05)?
B. Residual science: correct-response student features (CorrectResponseProvider fitted on the DISCOVERY half)
   vs the out-of-fold residual of the tabular GAM; top-k confirmed on the CONFIRMATION half with permutation p
   stratified by school (train-only grouping, never a feature), partial correlation with screening ability,
   BH-FDR (V3H04).
C. Student-level incremental R² (nested): ASRO vs metadata-only (SOFTMETA) and temporal-only (SOFT_TEMPORAL)
   correctness controls, all against the same GAM on the same folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..asr.model_registry import registry_for
from ..discovery.correct_response import ClipCorrectnessIncrement, CorrectResponseProvider, pick_role_columns
from ..discovery.item_conditioning import ItemReference, v3_clip_frame
from ..discovery.residual_analysis import auc_diff_block_bootstrap, residual_associations
from ..evaluation.nested_asr_cv import V3Evaluator, V3Spec, result_row
from ..features.conditional_aggregation import CachedProvider
from ..phonetics.builder import feature_columns, load_phonetic_tables
from ..representations.compact_features import FAMILY_FEATURES
from ..v2.validation_registry import discovery_split
from ..v3 import viz as V
from ..v3.common import available_schemes, begin, not_executed, student_meta
from ..v3.registry import save_figure_v3

S = "P26"


def primary_model(ctx, wide: pd.DataFrame) -> str | None:
    """Registry priority (pasketti_phonetic=1, whisper_small_en=2, ...; role order breaks ties) -> the first model
    whose phonetic features include a confidence column."""
    specs, _ = registry_for(ctx)
    role = {"child_phonetic": 0, "child_word": 1, "general_phonetic": 2}
    order = sorted(specs, key=lambda k: (specs[k].priority, role.get(specs[k].role, 3), k))
    for k in order:
        if "conf" in pick_role_columns(wide, k):
            return k
    return None


def run(ctx) -> dict:
    from ..asr.outputs import require_outputs

    reg, v3r = begin(ctx, S)
    ok, _acc, why = require_outputs(ctx, S, "text", 1, ("train",))
    if not ok:
        return not_executed(ctx, S, why)
    d = ctx.cfg.get_path("v3.discovery", {}) or {}
    wide, det = load_phonetic_tables(ctx, ("train",))
    if wide.empty or len(wide.columns) <= 1:
        return not_executed(ctx, S, "no P25 phonetic features")
    pm = primary_model(ctx, wide)
    if pm is None:
        return not_executed(ctx, S, "no model exposes a confidence measure (token probabilities or posteriors)")
    roles = pick_role_columns(wide, pm)
    ctx.save_table(pd.DataFrame([{"primary_model": pm, **roles}]), S, "primary_model_roles")
    c = v3_clip_frame(ctx)
    c = c[c["split"] == "train"].merge(wide, on=FILE, how="inner")
    y = ctx.bundle().y
    y.index = y.index.astype(str)
    conf = roles["conf"]
    # ---- A1 profiles (descriptive)
    ref = ItemReference(("item",)).fit(c)
    c["difficulty"] = ref.difficulty(c)
    c["correct"] = c["item_score"].map({1.0: "correct", 0.0: "incorrect"})
    prof_cols = list(dict.fromkeys(roles.values()))
    prof = c.dropna(subset=["correct"]).groupby(["task", "correct"])[prof_cols].agg(["mean", "std", "count"])
    prof.columns = [f"{a}__{b}" for a, b in prof.columns]
    ctx.save_table(prof.reset_index(), S, "correct_vs_incorrect_profiles")
    save_figure_v3(ctx, S, "correct_vs_incorrect_confidence", V.small_multiple_distributions(
        c.dropna(subset=["correct"]), conf, "task", "correct", f"Confidence ({conf}) for correct vs incorrect responses"),
        c[[FILE, "task", "correct", conf]], "Correct vs incorrect speech profiles")
    cc = c[c["item_score"] == 1].copy()
    cc["outcome_tercile"] = pd.qcut(cc[ID].map(y).rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    save_figure_v3(ctx, S, "correct_only_confidence_by_outcome", V.small_multiple_distributions(
        cc, conf, "task", "outcome_tercile", "Correct responses only: confidence by end-of-year outcome tercile (descriptive)"),
        cc[[FILE, "task", "outcome_tercile", conf]], "Correct-only confidence distributions")
    fig, tab = V.binned_profile(c.dropna(subset=["correct"]).assign(grp=lambda x: x["task"] + " · " + x["correct"]),
                                "difficulty", conf, "grp", "Confidence vs item difficulty (item p-correct)",
                                xlabel="item p(correct)", ylabel=conf)
    save_figure_v3(ctx, S, "confidence_vs_difficulty", fig, tab, "Confidence versus item difficulty")

    # ---- A2 clip-correctness increment over META / TEMPORAL
    phon_cols = [x for f in ("ASRW", "ASRP", "ASRC", "ASRT") for x in feature_columns(wide, pm, FAMILY_FEATURES[f])]
    temporal, tcols = _temporal(ctx)
    inc = ClipCorrectnessIncrement(ctx, int(d.get("clip_cv_splits", 5)), tuple(d.get("clip_models", ["LOGISTIC"])), ctx.seed)
    P, sets = inc.run(wide, phon_cols, temporal, tcols)
    ctx.save_table(sets, S, "clip_feature_sets")
    nb = int(d.get("bootstrap", 1000))
    comp = []
    if not P.empty:
        ctx.save_parquet(P, S, "clip_correctness_oof")
        for mdl in inc.models:
            for new, basek in (("META+PHON", "META"), ("META+TEMPORAL+PHON", "META+TEMPORAL"), ("META+TEMPORAL", "META")):
                a, bcol = f"p__{mdl}__{new}", f"p__{mdl}__{basek}"
                if a in P and bcol in P:
                    comp.append({"model": mdl, "new": new, "base": basek, **auc_diff_block_bootstrap(P, "item_score", a, bcol, nb, ctx.seed)})
    comp = pd.DataFrame(comp)
    ctx.save_table(comp, S, "clip_correctness_increment")
    if not comp.empty:
        k = comp[comp["new"].str.contains("PHON")]
        ok = bool((k["ci_low"] > 0).all()) if not k.empty else False
        v3r.update_hypothesis("V3H05", "supported" if ok else "not_supported", S,
                              "; ".join(f"{r.model} {r.new} vs {r.base}: ΔAUROC={r.delta_auc:.4f} [{r.ci_low:.4f},{r.ci_high:.4f}]" for r in k.itertuples()),
                              effect=float(k["delta_auc"].min()) if not k.empty else None)
        save_figure_v3(ctx, S, "clip_correctness_increment", V.forest(comp.assign(label=comp["model"] + ": " + comp["new"] + " vs " + comp["base"]),
                       "delta_auc", "ci_low", "ci_high", "label", "Clip correctness: phonetic information beyond metadata / temporal models",
                       xlabel="ΔAUROC (student-block bootstrap)"), comp, "Incremental clip-correctness information")

    # ---- B residual science (discovery / confirmation halves)
    ev = V3Evaluator(ctx, S, providers={})
    base = ev.evaluate(ev.baseline_v3(), "repeated_kfold")
    from ..validation.metrics import mean_oof

    mo = mean_oof(base.oof)
    mo.index = mo.index.astype(str)
    resid = (mo["y_true"] - mo["y_pred"]).rename("resid")
    ids = sorted(resid.index)
    disc, conf_ids = discovery_split(ids, float(d.get("discovery_fraction", 0.5)), ctx.seed)
    prov = CorrectResponseProvider(ctx, f"ASRO_{pm}", wide, roles, pm)
    X = prov.fit_table(disc, ids)                             # reference statistics from the discovery half only
    meta = student_meta(ctx)
    ability = c.drop_duplicates(ID).set_index(ID)["ability"].reindex(ids)
    strata = meta["school_id"].astype(str) if "school_id" in meta else None
    assoc = residual_associations(X, resid, ability, disc, conf_ids, strata, int(d.get("n_perm", 2000)), ctx.seed,
                                  int(d.get("top_k_confirm", 25)))
    ctx.save_table(assoc, S, "residual_associations")
    v3r.register_features(list(X.columns), S, pm, "student", True, "correct-response intelligence (ASRO)")
    cf = assoc[assoc["tier"] == "confirmatory"] if not assoc.empty else assoc
    alpha = float(ctx.cfg.get_path("v3.ablation.fdr_alpha", 0.05))
    if not cf.empty:
        hit = cf[(cf["q"] < alpha) & cf["same_sign"]]
        v3r.update_hypothesis("V3H04", "supported" if not hit.empty else "not_supported", S,
                              f"{len(hit)} confirmed of {len(cf)} tested; top: " + ",".join(hit.sort_values("q")["feature"].head(5)),
                              effect=float(cf.loc[cf["p"].idxmin(), "rho"]) if cf["p"].notna().any() else None,
                              q_value=float(cf["q"].min()) if cf["q"].notna().any() else None, n=int(cf["n"].max()))
        top = cf.sort_values("p").iloc[0]["feature"]
        j = pd.concat([X[top].rename("x"), resid], axis=1).reindex(conf_ids).dropna()
        save_figure_v3(ctx, S, "screening_residual_association", V.scatter(j.reset_index(), "x", "resid",
                       f"Confirmation half: {top} vs GAM residual", xlabel=top, ylabel="outcome − GAM OOF prediction"),
                       j.reset_index(), "Screening-residual association (confirmation half)")

    # ---- C student-level incremental R² vs correctness controls
    from ..v2.providers import build_bank, soft_provider

    provs = {"ASRO": CachedProvider(prov, root=ctx.paths.v3_cache / "fold_providers")}
    try:
        bank = build_bank(ctx)
        provs["SOFTMETA"] = soft_provider(ctx, bank, "META", "SOFTMETA")
        if bank.set_available("TEMPORAL"):
            provs["SOFTTEMP"] = soft_provider(ctx, bank, "TEMPORAL", "SOFTTEMP")
    except Exception as e:  # noqa: BLE001 - controls unavailable -> recorded
        ctx.logger.warning("P26: correctness controls unavailable: %s", e)
    ev2 = V3Evaluator(ctx, S, providers=provs)
    sch = available_schemes(ctx, ["repeated_kfold", "school_kfold"])
    rows = []
    for sc in sch:
        b = ev2.evaluate(ev2.baseline_v3(), sc)
        rows.append(result_row(b, {"delta_r2": 0.0}))
        for key in provs:
            for fu, mdl in (("residual", "RIDGE"), ("early", ev2.baseline["model"])):
                sp = V3Spec(f"P26_{key}_{fu.upper()}", mdl, [key], 20, "pca", fu,
                            alphas=tuple(ctx.cfg.get_path("v3.audit.continuous_alphas", [0, .1, .25, .5, 1])) if fu == "residual" else None,
                            group="correct_only", experiment="ASR06" if key == "ASRO" else "CONTROL")
                r = ev2.evaluate(sp, sc, baseline=b)
                rows.append(result_row(r, ev2.compare(b, r) if r.status in ("ok", "partial_fallback", "fallback_baseline") else {}))
    res = pd.DataFrame(rows)
    ctx.save_table(res, S, "student_incremental_r2")
    fr = res[res["scheme"] == "repeated_kfold"].dropna(subset=["delta_r2"]) if not res.empty else res
    if not fr.empty:
        save_figure_v3(ctx, S, "correct_only_incremental_r2", V.forest(fr, "delta_r2", "delta_ci_low", "delta_ci_high", "name",
                       "Correct-response features vs metadata / temporal correctness controls"), fr,
                       "Incremental R² of correct-only features vs controls")
    return {"primary_model": pm, "n_student_features": int(X.shape[1]),
            "n_confirmed": int(((cf["q"] < alpha) & cf["same_sign"]).sum()) if not cf.empty else 0}


def _temporal(ctx) -> tuple[pd.DataFrame | None, list[str]]:
    from ..acoustic.store import FeatureStore

    df = FeatureStore(ctx.paths.cache / "features").load("temporal", "train", prefix=True)
    if df.empty:
        return None, []
    cols = [c for c in df.columns if c != FILE and pd.api.types.is_numeric_dtype(df[c])
            and c.split("::")[-1] not in ("true_duration_s", "decoded_duration_s", "cropped", "file_sr")]
    return df.drop_duplicates(FILE), cols
