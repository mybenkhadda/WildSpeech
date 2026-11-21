"""P16 — acoustic domain harmonization.

1. CPU/GPU numerical parity of the band-limiting transform and the robust family.
2. Resumable extraction of every active variant (decoded once per clip; existing P02 caches reused for 'orig').
3. Per variant: sample-rate separability (item-centred and after out-of-sample domain residualization), school
   separability, feature distribution shifts (SMD / KS / Wasserstein), perturbation stability (cached),
   correctness discrimination, incremental reading R2 over the strongest tabular comparator (nested residual
   fusion), cross-school and bidirectional sample-rate transfer, equal-training-size transfer controls.
4. Feature robustness ranking and variant selection by a pre-registered rule -> selection.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..diagnostics.scientific_validation import auc_fast
from ..features.conditional_aggregation import ItemConditionedResidualizer, clip_frame
from ..harmonization import domain_diagnostics as DD
from ..harmonization.bandwidth import (BandwidthHarmonizer, HarmonizedExtractionRunner, bandlimit_numpy, configured_variants,
                                       load_variant_features)
from ..harmonization.perturbations import pick_clips, stability_experiment
from ..io_utils import stable_hash
from ..models.compact_regression import CompactSpec, StudentLevelEvaluator, equal_n_summary
from ..stats import eta_squared
from ..v2 import visualization_v2 as V
from ..v2.providers import harm_columns, harm_provider
from ..v2.registry import save_figure_v2
from ._v2common import begin, evaluate_against, schemes, student_meta, transfer_breakdown

S = "P16"


def _parity(ctx, runner: HarmonizedExtractionRunner, variants) -> pd.DataFrame:
    from ..data.audio_io import decode_audio_bytes, make_reader

    if ctx.device != "cuda":
        return pd.DataFrame([{"check": "cpu_gpu_parity", "status": "SKIPPED", "detail": "no CUDA device"}])
    h = ctx.cfg.get_path("v2.harmonization", {}) or {}
    man = runner_manifest(ctx)
    reader = make_reader(ctx.paths.files["train_audio"], man) if man is not None else None
    if reader is None:
        return pd.DataFrame([{"check": "cpu_gpu_parity", "status": "SKIPPED", "detail": "no random-access TAR"}])
    clips = pick_clips(man.loc[man["safe"], FILE], int(h.get("parity_clips", 8)))
    try:
        waves = [decode_audio_bytes(reader.read(f), runner.base.sr, 8.0)[0] for f in clips]
    finally:
        reader.close()
    rows = []
    rtol, atol = float(h.get("parity_rtol", 3e-3)), float(h.get("parity_atol", 2e-3))
    from ..harmonization.normalization import RobustFeatureExtractor
    from ..harmonization.bandwidth import variant_params

    for v in variants:
        g = BandwidthHarmonizer(runner.base.sr, "cuda", float(h.get("transition_hz", 250))).apply(waves, v)
        if v.lowpass_hz or v.highpass_hz:
            c = [bandlimit_numpy(w, runner.base.sr, v.lowpass_hz, v.highpass_hz, float(h.get("transition_hz", 250))) for w in waves]
            err = max(float(np.max(np.abs(a - b))) for a, b in zip(g, c))
            rows.append({"check": f"bandlimit_{v.name}", "max_abs_err": err, "status": "PASS" if err <= atol else "FAIL"})
        p = variant_params(runner.base, v)
        rg = RobustFeatureExtractor(p, "cuda").extract_batch(g)
        rc = RobustFeatureExtractor(p, "cpu").extract_batch(g)
        worst = 0.0
        for a, b in zip(rg, rc):
            for k in a:
                x, y = a[k], b[k]
                if np.isfinite(x) and np.isfinite(y):
                    worst = max(worst, abs(x - y) / (abs(y) + 1.0))
        rows.append({"check": f"robust_family_{v.name}", "max_rel_err": worst, "status": "PASS" if worst <= rtol else "FAIL"})
    return pd.DataFrame(rows)


def runner_manifest(ctx):
    from ..data.tar_manifest import build_tar_manifest

    if "train_audio" not in ctx.paths.files:
        return None
    return build_tar_manifest(ctx.paths.files["train_audio"], ctx.paths.cache / "manifests", logger=ctx.logger)


def _feature_fn(runner: HarmonizedExtractionRunner, v, gpu_fams: list[str]):
    gpu, rob = runner.extractor(v, "gpu"), runner.extractor(v, "robust")

    def fn(ws):
        out = gpu.extract_batch(ws, gpu_fams)
        r = rob.extract_batch(ws)
        return [{**{f"{f.upper()}::{k}": val for f in gpu_fams for k, val in out[f][j].items()},
                 **{f"ROBUST::{k}": val for k, val in r[j].items()}} for j in range(len(ws))]
    return fn


def run(ctx) -> dict:
    reg, v2r = begin(ctx, S)
    h = ctx.cfg.get_path("v2.harmonization", {}) or {}
    nb = int(ctx.cfg.get_path("v2.evaluation.bootstrap", 1000))
    variants = configured_variants(ctx.cfg)
    runner = HarmonizedExtractionRunner(ctx)
    ctx.save_table(_parity(ctx, runner, variants), S, "cpu_gpu_parity")
    ext = runner.run("train", variants, h.get("max_clips"))
    ctx.save_table(pd.DataFrame([{"key": k, "value": str(v)} for k, v in ext.items()]), S, "extraction_summary")

    c_all = clip_frame(ctx)
    ctr = c_all[c_all["split"] == "train"].reset_index(drop=True)
    meta = student_meta(ctx)
    ev = StudentLevelEvaluator(ctx, S)
    sch = schemes(ctx)
    base = {sc: ev.evaluate(ev.baseline_spec(), sc) for sc in sch}
    fams = list(h.get("families", [])) + (["robust"] if h.get("robust_family", True) else [])
    gpu_fams = [f for f in fams if f != "robust"]
    man = runner_manifest(ctx)
    pclips = pick_clips(man.loc[man["safe"], FILE], int(h.get("perturbation_clips", 240))) if man is not None else []
    harm = BandwidthHarmonizer(runner.base.sr, ctx.device, float(h.get("transition_hz", 250)), float(h.get("amp_target_dbfs", -26)))
    summary, student_rows, shifts, stabs, ranks, ltas, transfers, eqn = [], [], [], [], [], [], [], []
    emb_coords = {}
    for v in variants:
        tab = load_variant_features(ctx, v.name, fams, ("train",))
        cols = harm_columns(ctx, tab)
        if not cols:
            summary.append({"variant": v.name, "status": "no_features"})
            continue
        cv = ctr.merge(tab[[FILE] + cols], on=FILE, how="inner").reset_index(drop=True)
        F = cv[cols].to_numpy(float)
        row = {"variant": v.name, "status": "ok", "n_clips": int(len(cv)), "n_features": len(cols), **v.as_dict()}
        row.update(DD.sample_rate_separability(cv, F, nb, ctx.seed))
        row.update(DD.correctness_discrimination(cv, F, nb, ctx.seed))
        Zc = ItemConditionedResidualizer(("item",)).fit(cv, F).transform(cv, F)
        sh = DD.distribution_shift(cv, Zc, cols).assign(variant=v.name)
        shifts.append(sh)
        row["median_abs_smd"] = float(sh["smd"].abs().median()) if not sh.empty else np.nan
        row["median_ks"] = float(sh["ks"].median()) if not sh.empty else np.nan
        stab_feat = pd.Series(dtype=float)
        if pclips and h.get("perturbations"):
            fp = stable_hash({"v": runner.fingerprint(v, "robust"), "g": {f: runner.fingerprint(v, f) for f in gpu_fams}})
            st = stability_experiment(ctx, _feature_fn(runner, v, gpu_fams), fp, f"P16_{v.name}", pclips,
                                      list(h.get("perturbations", [])), variant_apply=lambda ws, v=v: harm.apply(ws, v))
            st = st[st["feature"].isin(cols)].assign(variant=v.name)
            stabs.append(st)
            stab_feat = st.groupby("feature")["stability_rho"].min()
            row["stability_median"] = float(st["stability_rho"].median())
            row["stability_min_median"] = float(stab_feat.median())
        corr_sig = DD.univariate_correctness_auc(cv, Zc, cols)
        if "school_id" in meta:
            sch_id = cv[ID].map(meta["school_id"]).to_numpy()
            sch_eta = pd.Series({col: eta_squared(Zc[:, j], sch_id)[0] for j, col in enumerate(cols)})
        else:
            sch_eta = None
        ranks.append(DD.robustness_ranking(sh, stab_feat, corr_sig, sch_eta).assign(variant=v.name))
        mel = [x for x in tab.columns if x.startswith("MEL::melband") and x.endswith("_mean")]
        if mel:
            lt = DD.ltas_by_domain(cv, tab[[FILE] + mel])
            if not lt.empty:
                ltas.append(lt.assign(variant=v.name))
        # ---- student level
        key = f"HARM_{v.name}"
        prov = harm_provider(ctx, v.name)
        if prov is None:
            summary.append(row)
            continue
        ev.add_provider(key, prov)
        spec = CompactSpec(f"P16_{key}", model="RIDGE", providers=[key], budget=int(h.get("student_budget", 20)), fusion="residual")
        rws, res_by = evaluate_against(ev, spec, base, sch)
        for r in rws:
            r["variant"] = v.name
        student_rows += rws
        prim = next((r for r in rws if r["scheme"] == "repeated_kfold"), {})
        row.update({"delta_r2": prim.get("delta_r2"), "delta_ci_low": prim.get("delta_ci_low"),
                    "delta_ci_high": prim.get("delta_ci_high"), "r2_cv": prim.get("r2_mean")})
        for sc in ("sample_rate_transfer", "school_kfold", "loso"):
            if sc in res_by and res_by[sc].status == "ok":
                row[f"r2_{sc}"] = res_by[sc].summary.get("r2_mean")
                tb = transfer_breakdown(res_by[sc])
                if not tb.empty:
                    transfers.append(tb.assign(variant=v.name))
        if "sample_rate_equal_n" in res_by and res_by["sample_rate_equal_n"].status == "ok":
            es = equal_n_summary(res_by["sample_rate_equal_n"].fold_metrics)
            if not es.empty:
                eqn.append(es.assign(variant=v.name))
        T = prov.table(ev.all_train, ev.all_train)
        Tn = T.fillna(T.median()).dropna(axis=1, how="all")
        if "school_id" in meta and Tn.shape[1] > 1:
            row.update(DD.school_separability(Tn, meta["school_id"], int(h.get("min_school_size", 8)), ctx.seed))
        if Tn.shape[1] > 2:
            from sklearn.decomposition import PCA

            Zs = (Tn - Tn.mean()) / Tn.std(ddof=0).replace(0, 1)
            emb_coords[v.name] = (PCA(2, random_state=0).fit_transform(Zs.to_numpy(float)), Tn.index)
            lab = (meta["original_sample_rate"].reindex(Tn.index) < 16000).map({True: "low_rate", False: "16k"}) if "original_sample_rate" in meta else None
            if lab is not None:
                y_sr = (lab == "low_rate").astype(float).to_numpy()
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_predict

                pp = cross_val_predict(LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced"), Zs.to_numpy(float),
                                       y_sr, cv=5, method="predict_proba")[:, 1]
                row["sr_auc_student_level"] = auc_fast(y_sr, pp)
        v2r.register_features(list(T.columns), S, key, f"item-conditioned harmonized acoustics ({v.name})")
        summary.append(row)

    vs = pd.DataFrame(summary)
    ctx.save_table(vs, S, "variant_summary")
    stu = pd.DataFrame(student_rows)
    ctx.save_table(stu, S, "student_results")
    sh_all = pd.concat(shifts, ignore_index=True) if shifts else pd.DataFrame()
    ctx.save_table(sh_all, S, "distribution_shift", parquet=True)
    st_all = pd.concat(stabs, ignore_index=True) if stabs else pd.DataFrame()
    ctx.save_table(st_all, S, "perturbation_stability")
    rk = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()
    ctx.save_table(rk, S, "feature_robustness_ranking")
    tr = pd.concat(transfers, ignore_index=True) if transfers else pd.DataFrame()
    ctx.save_table(tr, S, "transfer_breakdown")
    en = pd.concat(eqn, ignore_index=True) if eqn else pd.DataFrame()
    ctx.save_table(en, S, "equal_n_transfer")

    # ---- selection (pre-registered rule)
    sel = {"selected_variant": "orig", "rule": h.get("selection_rule", "")}
    ok = vs[vs.get("status", "") == "ok"] if not vs.empty else vs
    if not ok.empty and "sr_auc_item_centred" in ok:
        ref = ok.loc[ok["variant"] == "orig", "sr_auc_item_centred"]
        ref_auc = float(ref.iloc[0]) if len(ref) else np.inf
        cand = ok[(ok["sr_auc_item_centred"] <= ref_auc + 1e-9) & ok["delta_r2"].notna()] if "delta_r2" in ok else ok.iloc[0:0]
        if not cand.empty:
            cand = cand.assign(_d=cand["delta_r2"].round(3), _s=cand.get("stability_median", pd.Series(0, index=cand.index)).fillna(0))
            best = cand.sort_values(["_d", "_s"], ascending=False).iloc[0]
            sel.update({"selected_variant": str(best["variant"]), "delta_r2": float(best["delta_r2"]),
                        "sr_auc_item_centred": float(best["sr_auc_item_centred"]), "orig_sr_auc": ref_auc})
    ctx.save_json(sel, S, "selection")

    # ---- figures
    if ltas:
        lt = pd.concat(ltas, ignore_index=True)
        save_figure_v2(ctx, S, "spectra_by_domain", V.spectra_small_multiples(lt, "Long-term mel spectrum by original rate"), lt,
                       "Before/after harmonization: mean mel-band level for 16 kHz vs low-rate-origin recordings")
    if not vs.empty and "sr_auc_item_centred" in vs:
        d = vs.dropna(subset=["sr_auc_item_centred"])
        save_figure_v2(ctx, S, "sample_rate_separability", V.bar_ci(d, "variant", "sr_auc_item_centred", "sr_auc_item_centred_lo",
                       "sr_auc_item_centred_hi", "Sample-rate AUROC by variant (lower = harmonized)"), d, "Recording-domain separability")
    if not sh_all.empty:
        piv = sh_all.assign(abs_smd=sh_all["smd"].abs()).pivot_table(index="feature", columns="variant", values="abs_smd", aggfunc="mean")
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index[:60]]
        save_figure_v2(ctx, S, "distribution_shift_heatmap", V.heatmap(piv, "|SMD| low-rate vs 16k (item-centred)", diverging=False),
                       piv, "Feature distribution shifts by variant")
    if not st_all.empty:
        for vn, d in st_all.groupby("variant"):
            piv = d.pivot_table(index="feature", columns="perturbation", values="stability_rho")
            piv = piv.loc[piv.min(axis=1).sort_values().index[:60]]
            save_figure_v2(ctx, S, f"stability_matrix_{vn}", V.heatmap(piv, f"Perturbation stability ({vn})", diverging=False),
                           piv, f"Spearman original vs perturbed, variant {vn}")
    if not rk.empty:
        d = rk[rk["variant"] == sel["selected_variant"]].head(30)
        if not d.empty:
            save_figure_v2(ctx, S, "feature_robustness_ranking", V.bar_ci(d, "feature", "robustness_score", None, None,
                           f"Feature robustness ranking ({sel['selected_variant']})"), d, "Rank-average of stability, shift, school effect, correctness signal")
    for vn, (xy, idx) in emb_coords.items():
        lab = (meta["original_sample_rate"].reindex(idx) < 16000).map({True: "low_rate", False: "16k"}).to_numpy() if "original_sample_rate" in meta else np.array(["all"] * len(idx))
        save_figure_v2(ctx, S, f"domain_embedding_{vn}", V.embedding_scatter(xy, lab, f"Student-level harmonized features ({vn})"),
                       pd.DataFrame({ID: idx, "pc1": xy[:, 0], "pc2": xy[:, 1], "domain": lab}), "Domain embedding (PCA)")
    if not stu.empty and "delta_r2" in stu:
        d = stu[stu["scheme"] == "repeated_kfold"].dropna(subset=["delta_r2"])
        if not d.empty:
            save_figure_v2(ctx, S, "incremental_r2_by_variant", V.forest(d.sort_values("delta_r2", ascending=False), "delta_r2",
                           "delta_ci_low", "delta_ci_high", "variant", "Incremental R² over the strongest tabular comparator"), d,
                           "Paired bootstrap CI of ΔR² per variant")
    if not tr.empty:
        save_figure_v2(ctx, S, "transfer_by_direction", V.lines(tr.assign(direction=tr["label"].astype(str)), "variant", "r2", "direction",
                       "R² by validation direction / school fold"), tr, "Bidirectional sample-rate transfer and cross-school R²")
    if not en.empty:
        save_figure_v2(ctx, S, "equal_n_gap", V.bar_ci(en.assign(lbl=en["variant"] + " -> " + en["target"]), "lbl", "gap_mean", None, None,
                       "Within-domain minus transfer R² at equal training size"), en, "Equal-size transfer controls")
    # ---- hypotheses
    if not ok.empty and "sr_auc_item_centred_hi" in ok and (ok["variant"] == "orig").any():
        o = ok[ok["variant"] == "orig"].iloc[0]
        s_ = ok[ok["variant"] == sel["selected_variant"]].iloc[0]
        status = "supported" if s_["sr_auc_item_centred_hi"] < o["sr_auc_item_centred_lo"] else "not_supported"
        v2r.update_hypothesis("V2H01", status, S, f"{sel['selected_variant']} sr AUROC {s_['sr_auc_item_centred']:.3f} vs orig {o['sr_auc_item_centred']:.3f}",
                              effect=float(s_["sr_auc_item_centred"] - o["sr_auc_item_centred"]))
    if "delta_ci_low" in vs and vs["delta_ci_low"].notna().any():
        b = vs.dropna(subset=["delta_ci_low"]).sort_values("delta_r2", ascending=False).iloc[0]
        v2r.update_hypothesis("V2H02", "supported" if b["delta_ci_low"] > 0 else "not_supported", S,
                              f"best variant {b['variant']} ΔR²={b['delta_r2']:.4f}", effect=float(b["delta_r2"]),
                              ci_low=float(b["delta_ci_low"]), ci_high=float(b["delta_ci_high"]))
    return {"variants": len(variants), "selected": sel["selected_variant"], "baseline": ev.baseline.get("source"),
            "baseline_r2": base.get("repeated_kfold").summary.get("r2_mean") if "repeated_kfold" in base else None}
