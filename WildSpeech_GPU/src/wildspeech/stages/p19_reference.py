"""P19 — reference-conditioned speech modelling.

Frame sequences (CMVN-MFCC + log-mel, speech span) on the P16-selected harmonized variant; exemplar templates of
correct responses fitted on training-fold students only (nested for training students); DTW (GPU, batched),
duration-normalised and localized deviations, temporal consistency, posterior / embedding similarity, CTC
likelihood pass-through. Analyses: clip-level discrimination of correctness, deviation AMONG correct responses
vs the comparator's residuals (exploratory / confirmatory), adjusted association controlling accuracy,
item-level reference atlas, student-level incremental R2 over the strongest tabular comparator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..diagnostics.scientific_validation import auc_fast, block_bootstrap
from ..features.conditional_aggregation import ItemConditionedResidualizer, clip_frame
from ..harmonization.bandwidth import Variant
from ..models.compact_regression import CompactSpec, StudentLevelEvaluator
from ..reference.dtw import dtw_with_path
from ..reference.similarity import REF_FEATURES
from ..reference.templates import ExemplarBank, SequenceExtractionRunner
from ..stats import ols_hc3
from ..v2 import visualization_v2 as V
from ..v2.providers import ref_provider, selected_variant
from ..v2.registry import save_figure_v2
from ..v2.validation_registry import discovery_split
from ._v2common import association_table, baseline_residuals, begin, evaluate_against

S = "P19"


def _variant(ctx) -> Variant:
    r = ctx.cfg.get_path("v2.reference", {}) or {}
    name = r.get("variant", "auto")
    name = selected_variant(ctx) if name in (None, "auto") else name
    vv = (ctx.cfg.get_path("v2.harmonization.variants", {}) or {}).get(name, {})
    return Variant(name, vv.get("lowpass_hz"), vv.get("highpass_hz"), bool(vv.get("amp_norm", False)))


def run(ctx) -> dict:
    reg, v2r = begin(ctx, S)
    r = ctx.cfg.get_path("v2.reference", {}) or {}
    v = _variant(ctx)
    runner = SequenceExtractionRunner(ctx, v)
    ext = [{"split": s, **{k: str(x) for k, x in runner.run(s).items()}} for s in r.get("splits", ["train"])]
    ctx.save_table(pd.DataFrame(ext), S, "sequence_extraction")
    clip_dir = ctx.stage_dir(S) / "clip_deviations"
    clip_dir.mkdir(parents=True, exist_ok=True)
    prov = ref_provider(ctx, ("train",), save_clip_dir=clip_dir)
    if prov is None:
        v2r.update_hypothesis("V2H07", "unavailable", S, "no frame sequences")
        return {"status": "unavailable", "reason": "no frame sequences"}
    ctx.save_table(pd.DataFrame([{"synthetic_enabled": bool((r.get("synthetic") or {}).get("enabled")),
                                  "synthetic_items": len(prov.p.synthetic), "variant": v.name,
                                  "posteriors": prov.p.extra.get("post"), "embeddings": prov.p.extra.get("emb") is not None,
                                  "ctc_likelihood": prov.p.extra.get("ctc")}]), S, "reference_sources")

    ev = StudentLevelEvaluator(ctx, S)
    sch = [s for s in r.get("crossfit_schemes", ["repeated_kfold"]) if s in reg.schemes()]
    base = {sc: ev.evaluate(ev.baseline_spec(), sc) for sc in sch}
    ev.add_provider("REF", prov)
    rws, res_by = evaluate_against(ev, CompactSpec("P19_REF", "RIDGE", providers=["REF"],
                                                   budget=int(ctx.cfg.get_path("v2.evaluation.default_budget", 20)), fusion="residual"), base, sch)
    stu = pd.DataFrame(rws)
    ctx.save_table(stu, S, "student_results")

    # ---- clip level (nested over all training students: every clip scored without its own student's clips)
    clip = prov.p.clip_table(ev.all_train, ev.all_train)
    c = clip_frame(ctx)
    c = c[c["split"] == "train"].merge(clip, on=FILE, how="inner").reset_index(drop=True)
    ctx.save_parquet(c, S, "reference_clip_table")
    feats = [f for f in REF_FEATURES if f in c and c[f].notna().sum() > 50]
    y = c["item_score"].to_numpy(float)
    rows = []
    for f in feats:
        a = auc_fast(y, c[f].to_numpy(float))
        d = pd.DataFrame({ID: c[ID], "y": y, "s": c[f]}).dropna()
        est, lo, hi = block_bootstrap(lambda x: auc_fast(x["y"], x["s"]), d, ID, int(ctx.cfg.get_path("v2.evaluation.bootstrap", 1000)) // 2, ctx.seed)
        rows.append({"feature": f, "auc": a, "auc_ci_low": lo, "auc_ci_high": hi, "auc_signal": abs(a - 0.5) * 2, "n": int(len(d))})
    ra = pd.DataFrame(rows).sort_values("auc_signal", ascending=False) if rows else pd.DataFrame()
    ctx.save_table(ra, S, "reference_clip_auc")
    if feats:
        Z = ItemConditionedResidualizer(("item", "domain")).fit(c, c[feats].to_numpy(float)).transform(c, c[feats].to_numpy(float))
        for f in feats[:4]:
            j = feats.index(f)
            vals = {f"{t} / {'correct' if s == 1 else 'incorrect'}": Z[(c["task"] == t).to_numpy() & (y == s), j]
                    for t in sorted(c["task"].unique()) for s in (0.0, 1.0)}
            save_figure_v2(ctx, S, f"deviation_distribution_{f}", V.distributions_by_group(vals, f"{f} (item-centred) by task and score", f),
                           pd.DataFrame({ID: c[ID], "task": c["task"], "y": y, f: Z[:, j]}), "Reference deviation distributions")

    # ---- deviation beyond binary correctness
    if "repeated_kfold" in base and base["repeated_kfold"].status == "ok":
        resid = baseline_residuals(ev, base["repeated_kfold"])
        T = prov.table(ev.all_train, ev.all_train)
        Tc = T[[x for x in T.columns if x.endswith("__correct_mean")]]
        disc, conf = discovery_split(ev.all_train, float(ctx.cfg.get_path("v2.validation.discovery_fraction", 0.5)), ctx.seed)
        wc = association_table(Tc, resid, disc, conf, int(ctx.cfg.get_path("analysis.permutations", 1000)), ctx.seed)
        ctx.save_table(wc, S, "within_correct_association")
        acc = c.groupby(ID)["item_score"].mean()
        adj = []
        for col in Tc.columns:
            j = pd.concat([resid, acc.rename("acc"), Tc[col].rename("dev")], axis=1).dropna()
            if len(j) < 30:
                continue
            tab = ols_hc3(j["resid"].to_numpy(), j[["acc", "dev"]].to_numpy(), ["accuracy", "deviation"])
            t = tab[tab["term"] == "deviation"].iloc[0]
            adj.append({"feature": col, "coef": t["coef"], "se_hc3": t["se_hc3"], "p": t["p"], "n": int(t["n"])})
        from ..diagnostics.scientific_validation import add_fdr

        adj = add_fdr(pd.DataFrame(adj)) if adj else pd.DataFrame()
        ctx.save_table(adj, S, "adjusted_associations")
        if "q" in wc and "same_sign" in wc:
            ok = wc[(wc["tier"] == "confirmatory") & (wc["q"] < 0.05) & wc["same_sign"].fillna(False).astype(bool)]
            v2r.update_hypothesis("V2H08", "supported" if len(ok) else "not_supported", S,
                                  f"{len(ok)} correct-only deviation features replicate (confirmatory, q<0.05)", n=int(len(ok)))
        top = wc[wc["tier"] == "exploratory"].dropna(subset=["p"]).sort_values("p").head(1)
        if not top.empty:
            f = top.iloc[0]["feature"]
            save_figure_v2(ctx, S, "residual_vs_top_deviation", V.viz.scatter_fit(Tc[f].reindex(resid.index), resid, f, "comparator residual",
                           "Deviation among correct responses vs residual (exploratory)"),
                           pd.DataFrame({ID: resid.index, f: Tc[f].reindex(resid.index).to_numpy(), "resid": resid.to_numpy()}), "Residual association")
        v2r.register_features(list(T.columns), S, "REF", "reference-conditioned deviations")
    if not stu.empty and "delta_ci_low" in stu:
        rr = stu[stu["scheme"] == "repeated_kfold"]
        if not rr.empty and pd.notna(rr.iloc[0].get("delta_ci_low")):
            rr = rr.iloc[0]
            v2r.update_hypothesis("V2H07", "supported" if rr["delta_ci_low"] > 0 else "not_supported", S,
                                  f"REF ΔR²={rr['delta_r2']:.4f}", effect=float(rr["delta_r2"]), ci_low=float(rr["delta_ci_low"]),
                                  ci_high=float(rr["delta_ci_high"]))

    # ---- item-level reference atlas + similarity matrix + alignment plots
    seqs = prov.p.comp.seq
    bank = ExemplarBank(int(r.get("exemplars_per_item", 4)), int(r.get("min_correct_per_item", 5)), ctx.seed).fit(
        clip_frame(ctx)[lambda d: d["split"] == "train"], set(seqs.get("mfcc", {})))
    if "dtw_mfcc_mean" in c:
        atlas = c.groupby(["task", "item_id"]).apply(lambda d: pd.Series({
            "n_clips": len(d), "p_correct": d["item_score"].mean(), "n_exemplars": len(bank.exemplars(d.name[1])),
            "median_exemplar_frames": float(np.median([len(seqs["mfcc"][e]) for e in bank.exemplars(d.name[1])])) if bank.exemplars(d.name[1]) else np.nan,
            "dev_correct": d.loc[d["item_score"] == 1, "dtw_mfcc_mean"].mean(), "dev_incorrect": d.loc[d["item_score"] == 0, "dtw_mfcc_mean"].mean()})).reset_index()
        ctx.save_table(atlas, S, "item_reference_atlas")
        a = atlas.dropna(subset=["dev_correct", "dev_incorrect"])
        if not a.empty:
            save_figure_v2(ctx, S, "item_reference_atlas", V.viz.scatter_fit(a["dev_correct"], a["dev_incorrect"], "mean DTW (correct)",
                           "mean DTW (incorrect)", "Item-level reference atlas", hue=a["task"]), a, "Per-item deviation of correct vs incorrect responses")
        it = c["item_id"].value_counts().index[0]
        d = c[c["item_id"] == it]
        sel = list(d[d["item_score"] == 1][FILE].head(16)) + list(d[d["item_score"] == 0][FILE].head(8))
        sel = [s for s in sel if s in seqs["mfcc"]]
        if len(sel) >= 4:
            M = np.zeros((len(sel), len(sel)))
            for i, a_ in enumerate(sel):
                for j in range(i + 1, len(sel)):
                    M[i, j] = M[j, i] = dtw_with_path(seqs["mfcc"][a_].astype(np.float32), seqs["mfcc"][sel[j]].astype(np.float32), r.get("band_frac", 0.35))[0]["norm_cost"]
            lab = [f"{'C' if s in set(d[d['item_score'] == 1][FILE]) else 'I'}:{s[:6]}" for s in sel]
            mdf = pd.DataFrame(M, index=lab, columns=lab)
            save_figure_v2(ctx, S, "similarity_matrix", V.heatmap(mdf, f"DTW distance matrix — {it} (C=correct, I=incorrect)", diverging=False),
                           mdf, "Pairwise duration-normalised DTW cost")
            ex = bank.exemplars(it)
            if ex:
                panels = []
                cor_q =next((x for x in sel if x in set(d[d["item_score"] == 1][FILE]) and x not in ex), None)
                inc_q = next((x for x in sel if x in set(d[d["item_score"] == 0][FILE])), None)
                for q, tag in ((cor_q, "correct"), (inc_q, "incorrect")):
                    if q is None:
                        continue
                    st_, C, path = dtw_with_path(seqs["mfcc"][q].astype(np.float32), seqs["mfcc"][ex[0]].astype(np.float32), r.get("band_frac", 0.35))
                    panels.append({"label": f"{tag} response (norm cost {st_['norm_cost']:.3f})", "C": C, "path": path})
                if panels:
                    save_figure_v2(ctx, S, "dtw_alignment_examples", V.dtw_alignment(panels, f"MFCC DTW alignment to an exemplar — {it}"),
                                   pd.DataFrame([{"label": p_["label"]} for p_ in panels]), "Expected-versus-observed alignment")
    return {"status": "ok", "variant": v.name, "n_clips_scored": int(len(c)), "features": feats}
