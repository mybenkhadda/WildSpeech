"""P08 — robustness: school effects, recording artifacts, perturbation (anonymisation) sensitivity,
noise controls and domain generalisation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, viz
from ..acoustic.extractor import AcousticExtractionRunner, active_families
from ..analysis import robustness as Rb
from ..analysis.residuals import candidate_features
from ..data.audio_io import make_reader
from ..validation.runner import ModelSpec
from ._common import get_runner, load_crossfit, save_fig, screening_residuals, student_meta

S = "P08"


def _perturbation_experiment(ctx) -> pd.DataFrame:
    """Same clip sample and statistics as before, but base/perturbed feature tables are cached on disk
    (cache/v2/perturbation) and perturbations run batched on the GPU, so a re-run of P08 skips this
    model-independent work entirely (it dominated P08's 3,000 s runtime)."""
    from ..harmonization.perturbations import PerturbationFeatureCache, pick_clips
    from ..io_utils import stable_hash

    runner = AcousticExtractionRunner(ctx)
    fams = [f for f in active_families(ctx.cfg) if f != "wavelet"]
    man = runner.manifest("train")
    if man is None or man.empty or make_reader(ctx.paths.files["train_audio"], man) is None:
        return pd.DataFrame()
    n = int(ctx.cfg.get_path("analysis.perturbation_clips", 300))
    clips = pick_clips(man.loc[man["safe"], FILE], n)

    def fn(ws):
        out = runner._gpu_extract(ws, fams)
        return [{f"{f.upper()}::{k}": v for f in fams for k, v in out[f][j].items()} for j in range(len(ws))]

    cache = PerturbationFeatureCache(ctx, fn, stable_hash({f: runner.family_fingerprint(f) for f in fams}), "P08")
    base, _ = cache.get(clips, None, None, 10.0)
    if base.empty:
        return pd.DataFrame()
    base = base.set_index(FILE)
    pert, methods = {}, {}
    for name in ctx.cfg.get_path("analysis.perturbations", []):
        df, methods[name] = cache.get(clips, name, None, 10.0)
        pert[name] = df.set_index(FILE).reindex(base.index)
    stab = Rb.perturbation_stability(base, pert, float(ctx.cfg.get_path("analysis.stability_flag", 0.7)))
    if not stab.empty:
        stab["method"] = stab["perturbation"].map(methods)
    return stab


def run(ctx) -> dict:
    a = ctx.cfg.section("analysis")
    alpha = float(ctx.cfg.get_path("validation.fdr_alpha", 0.05))
    X = load_crossfit(ctx)
    X.index = X.index.astype(str)
    y = ctx.bundle().y.reindex(X.index)
    resid, _ = screening_residuals(ctx)
    resid = resid.reindex(X.index)
    meta = student_meta(ctx).set_index("subject_id")
    feats = candidate_features(X)
    out = {}
    acoustic = [c for c in feats if c.split("__")[0] not in ("BEHAVIOR", "ITEM", "LINGUISTIC")]

    # ---- school effects
    if "school_id" in meta:
        se = Rb.school_effects(X, meta["school_id"], feats, float(a.get("school_eta2_flag", 0.1)))
        ctx.save_table(se, S, "school_effects")
        sa = Rb.school_adjusted_associations(X, resid, meta["school_id"], feats)
        ctx.save_table(sa, S, "school_adjusted_residual_associations")
        ac = sa[sa["feature"].isin(acoustic)]
        n_ok = int(((ac["q_school_demeaned"] < alpha) & ac["sign_kept"]).sum())
        ctx.registries.update_hypothesis("H06", "supported" if n_ok else "not_supported", S,
                                         f"{n_ok} acoustic features keep sign and q<{alpha} after school demeaning")
        out["school_flagged"] = int(se["school_flag"].sum())
    # ---- recording artifacts
    art = Rb.artifact_correlations(X, acoustic, float(a.get("artifact_corr_flag", 0.4)))
    ctx.save_table(art, S, "artifact_correlations")
    if not art.empty and "rho__RECORDING__orig_sr_low" in art:
        frac = float((art["rho__RECORDING__orig_sr_low"].abs() > float(a.get("artifact_corr_flag", 0.4))).mean())
        ctx.registries.update_hypothesis("H07", "supported" if frac >= 0.10 else "not_supported", S,
                                         f"{frac:.1%} of acoustic features |rho| with low original sample rate > flag", effect=frac)
    adj = Rb.artifact_adjusted_associations(X, y, acoustic)
    ctx.save_table(adj, S, "artifact_adjusted_associations")
    spec_fams = adj[adj["family"].isin(["SPECTRAL", "MFCC", "MEL"])]
    if not spec_fams.empty:
        n = int((spec_fams["q_adj"] < alpha).sum())
        ctx.registries.update_hypothesis("H03", "supported" if n else "not_supported", S,
                                         f"{n} SPECTRAL/MFCC/MEL features with artifact-adjusted q<{alpha}",
                                         effect=float(spec_fams.iloc[0]["partial_rho_adj"]), q_value=float(spec_fams.iloc[0]["q_adj"]))
    # ---- perturbation / anonymisation sensitivity
    stab = _perturbation_experiment(ctx)
    if not stab.empty:
        ctx.save_table(stab, S, "perturbation_stability")
        ss = Rb.stability_summary(stab)
        ctx.save_table(ss, S, "perturbation_stability_summary")
        piv = ss.set_index("feature").drop(columns=["family", "min_stability"]).head(60)
        save_fig(ctx, S, "perturbation_stability", viz.heatmap(piv, "Feature stability (Spearman orig vs perturbed)", diverging=False), piv.reset_index())
        out["unstable_features"] = int((ss["min_stability"] < float(a.get("stability_flag", 0.7))).sum())
    # ---- domain generalisation
    runner = get_runner(ctx, with_audio=True)
    abl = ctx.read_stage_table("P10", "family_ablation")
    best_fams = ["ALL"]
    if abl is not None and not abl.empty:
        best_fams = str(abl.sort_values("r2_mean", ascending=False).iloc[0]["families"]).split("+")
    specs = [ModelSpec("DG_SCREENING", "RIDGE", ["SCREENING", "RECORDING"]), ModelSpec("DG_BEST", "RIDGE", best_fams)]
    schemes = ["repeated_kfold"] + list(ctx.cfg.get_path("validation.extended_schemes", [])) + ["school_kfold"]
    rows = []
    for sp in specs:
        for sc in dict.fromkeys(schemes):
            res = runner.evaluate(sp, sc, stage=S)
            if res.status == "ok":
                rows.append({"name": sp.name, "families": "+".join(sp.families), "scheme": sc, **res.summary})
                if sc != "repeated_kfold":
                    ctx.save_table(res.fold_metrics, S, f"dg_folds_{sp.name}_{sc}")
    dg = pd.DataFrame(rows)
    ctx.save_table(dg, S, "domain_generalisation")
    if not dg.empty:
        save_fig(ctx, S, "domain_generalisation", viz.lines(dg, "scheme", "r2_mean", "name", "R2 across validation schemes"), dg)
        try:
            r_cv = dg[(dg.name == "DG_BEST") & (dg.scheme == "repeated_kfold")]["r2_mean"].iloc[0]
            r_lo = dg[(dg.name == "DG_BEST") & (dg.scheme == "loso")]["r2_mean"].iloc[0]
            ctx.registries.update_hypothesis("H12", "supported" if r_lo >= r_cv - 0.05 else "not_supported", S,
                                             f"LOSO R2={r_lo:.3f} vs repeated K-fold R2={r_cv:.3f}", effect=float(r_lo - r_cv))
        except IndexError:
            ctx.registries.update_hypothesis("H12", "not_tested", S, "LOSO not evaluated (enable validation.extended_schemes)")
    out["domain_rows"] = len(dg)
    return out
