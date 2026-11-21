"""P18 — phonetic and linguistic intelligence (eligible local models only).

Runs every verified adapter (Whisper / wav2vec2 CTC characters / CTC phonemes / child models), then:
  * correctness-conditioned contrasts per task (item-centred, student-block bootstrap, FDR)
  * variation AMONG OFFICIALLY CORRECT responses and its association with the comparator's residuals
    (discovery half = exploratory, confirmation half = confirmatory)
  * phoneme confusion matrices, error-position heatmaps, confidence distributions, alignment visualisations,
    task-specific phonetic profiles, linguistic-complexity sensitivity
  * student-level incremental R2 of PHON (+ CPX) over the strongest tabular comparator
If no model is eligible the stage completes with an explicit 'unavailable' record (nothing is fabricated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..diagnostics.scientific_validation import add_fdr, block_bootstrap_contrasts
from ..features.conditional_aggregation import ItemConditionedResidualizer, clip_frame
from ..models.compact_regression import CompactSpec, StudentLevelEvaluator
from ..phonetic.error_features import confusion_counts, ops_from_string
from ..phonetic.runner_v2 import PhoneticV2Runner, load_phonetic_details, load_phonetic_features, model_jobs
from ..v2 import visualization_v2 as V
from ..v2.providers import cpx_provider, phon_provider
from ..v2.registry import save_figure_v2
from ..v2.validation_registry import discovery_split
from ._v2common import association_table, baseline_residuals, begin, evaluate_against, schemes

S = "P18"


def run(ctx) -> dict:
    reg, v2r = begin(ctx, S)
    ph_cfg = ctx.cfg.get_path("v2.phonetic", {}) or {}
    jobs, elig = model_jobs(ctx)
    ctx.save_table(elig, S, "model_eligibility")
    runner = PhoneticV2Runner(ctx)
    ctx.save_table(pd.DataFrame([{"g2p_backend": runner.g2p.backend, "exact": runner.g2p.exact,
                                  "allow_approximate": runner.allow_approx}]), S, "g2p_backend")
    if ph_cfg.get("require_verification", True) and not jobs:
        for h in ("V2H05", "V2H06"):
            v2r.update_hypothesis(h, "unavailable", S, "no eligible (license + overlap verified) local phonetic model")
        return {"status": "unavailable", "reason": "no eligible phonetic model; see model_eligibility"}
    runs = []
    for split in ph_cfg.get("splits", ["train"]):
        s = runner.run(split)
        runs.append({"split": split, **{k: str(v) for k, v in s.items() if k != "eligibility"}})
    ctx.save_table(pd.DataFrame(runs), S, "phonetic_run_summary")

    ph = load_phonetic_features(ctx, ("train",))
    if ph.empty:
        return {"status": "no phonetic features produced"}
    c = clip_frame(ctx)
    c = c[c["split"] == "train"].merge(ph, on=FILE, how="inner").reset_index(drop=True)
    cols = [x for x in ph.columns if x != FILE and "::post_" not in x]
    F = c[cols].to_numpy(float)
    Zc = ItemConditionedResidualizer(("item",)).fit(c, F).transform(c, F)

    # ---- correct vs incorrect (item-centred, per task, student-block bootstrap)
    rows = []
    for t in sorted(c["task"].unique()):
        m = (c["task"] == t).to_numpy() & c["item_score"].notna().to_numpy()
        if m.sum() < 30:
            continue
        r = block_bootstrap_contrasts(Zc[m], c.loc[m, "item_score"].to_numpy(float), c.loc[m, ID].to_numpy(),
                                      int(ctx.cfg.get_path("v2.evaluation.bootstrap", 1000)), ctx.seed)
        rows.append(r.assign(task=t, feature=cols))
    cvi = add_fdr(pd.concat(rows, ignore_index=True), "p_boot", "q") if rows else pd.DataFrame()
    ctx.save_table(cvi, S, "correct_vs_incorrect")
    if not cvi.empty:
        piv = cvi.pivot_table(index="feature", columns="task", values="diff")
        save_figure_v2(ctx, S, "correctness_conditioned_differences", V.heatmap(piv, "Correct − incorrect (item-centred SD units)"),
                       piv, "Correctness-conditioned phonetic differences per task")

    # ---- among officially correct responses
    cor = c[c["item_score"] == 1]
    prof = []
    for t, d in cor.groupby("task"):
        row = {"task": t, "n_correct_clips": len(d)}
        for col in cols:
            v = d[col].astype(float)
            if v.notna().sum() < 10:
                continue
            short = col.split("::")[1]
            row[f"{short}__median"] = float(v.median())
            if short.endswith(("_per", "_er", "w_er", "c_er")):
                row[f"{short}__frac_nonzero"] = float((v > 0).mean())
        prof.append(row)
    ctx.save_table(pd.DataFrame(prof), S, "within_correct_profile")

    ev = StudentLevelEvaluator(ctx, S)
    sch = schemes(ctx)
    base = {sc: ev.evaluate(ev.baseline_spec(), sc) for sc in sch}
    pp = phon_provider(ctx)
    cp = cpx_provider(ctx)
    stu = []
    if pp is not None:
        ev.add_provider("PHON", pp)
        ev.add_provider("CPX", cp)
        bud = int(ctx.cfg.get_path("v2.evaluation.default_budget", 20))
        for name, provs in (("P18_PHON", ["PHON"]), ("P18_PHON_CPX", ["PHON", "CPX"])):
            rws, _ = evaluate_against(ev, CompactSpec(name, "RIDGE", providers=provs, budget=bud, fusion="residual"), base, sch)
            stu += rws
        if "repeated_kfold" in base and base["repeated_kfold"].status == "ok":
            T = pp.table(ev.all_train, ev.all_train)
            Tc = T[[x for x in T.columns if x.endswith("__correct_mean")]]
            resid = baseline_residuals(ev, base["repeated_kfold"])
            disc, conf = discovery_split(ev.all_train, float(ctx.cfg.get_path("v2.validation.discovery_fraction", 0.5)), ctx.seed)
            wc = association_table(Tc, resid, disc, conf, int(ctx.cfg.get_path("analysis.permutations", 1000)), ctx.seed)
            ctx.save_table(wc, S, "within_correct_association")
            if "q" in wc and "same_sign" in wc:
                conf_ok = wc[(wc["tier"] == "confirmatory") & (wc["q"] < 0.05) & wc["same_sign"].fillna(False).astype(bool)]
            else:
                conf_ok = wc.iloc[0:0]
            v2r.update_hypothesis("V2H06", "supported" if len(conf_ok) else "not_supported", S,
                                  f"{len(conf_ok)} correct-only phonetic features replicate (confirmatory, q<0.05)",
                                  tier="confirmatory", n=int(len(conf_ok)))
            v2r.register_features(list(T.columns), S, "PHON", "phonetic conditional aggregation")
            Ct = cp.table(ev.all_train, ev.all_train)
            ctx.save_table(Ct.describe().T.reset_index().rename(columns={"index": "feature"}), S, "complexity_sensitivity_summary")
    stu = pd.DataFrame(stu)
    ctx.save_table(stu, S, "student_results")
    if not stu.empty and "delta_ci_low" in stu:
        r = stu[(stu["name"] == "P18_PHON") & (stu["scheme"] == "repeated_kfold")]
        if not r.empty and pd.notna(r.iloc[0]["delta_ci_low"]):
            r = r.iloc[0]
            v2r.update_hypothesis("V2H05", "supported" if r["delta_ci_low"] > 0 else "not_supported", S,
                                  f"PHON ΔR²={r['delta_r2']:.4f}", effect=float(r["delta_r2"]), ci_low=float(r["delta_ci_low"]),
                                  ci_high=float(r["delta_ci_high"]))

    # ---- figures from alignment details
    det = load_phonetic_details(ctx, ("train",))
    if not det.empty:
        det = det.merge(clip_frame(ctx)[[FILE, "task", "item_score"]], on=FILE, how="left")
        for model, d in det.groupby("model"):
            d = d[d["ops"].astype(str).str.len() > 0]
            if d.empty:
                continue
            mat = confusion_counts(d["ops"], top=30)
            if not mat.empty:
                save_figure_v2(ctx, S, f"phoneme_confusion_{model}", V.heatmap(np.log1p(mat), f"Phoneme confusion (log counts) — {model}", diverging=False),
                               mat, "Rows = expected phone, columns = recognised phone (∅ = deletion/insertion)")
            bins = int(ph_cfg.get("position_bins", 3))
            pos = []
            for t, dt in d.groupby("task"):
                err, tot = np.zeros(bins), np.zeros(bins)
                for s, ref in zip(dt["ops"], dt["ref_phones"]):
                    n = max(len(str(ref).split()), 1)
                    for k, _, _, p_ in ops_from_string(s):
                        b = min(bins - 1, int(bins * p_ / n))
                        if k in ("M", "S", "D"):
                            tot[b] += 1
                        if k != "M":
                            err[b] += 1
                pos.append(pd.Series(err / np.maximum(tot, 1), index=[f"bin{j}" for j in range(bins)], name=t))
            if pos:
                pm = pd.DataFrame(pos)
                save_figure_v2(ctx, S, f"error_position_heatmap_{model}", V.heatmap(pm, f"Phone error rate by relative position — {model}", diverging=False, annotate=True),
                               pm, "Initial → final position bins")
        ex = det[det["spans"].astype(str).str.len() > 0]
        if not ex.empty:
            k = int(ph_cfg.get("example_clips", 6))
            pick = ex.groupby(["task", "item_score"], dropna=False).head(max(1, k // 6)).head(k)
            examples = [{"label": f"{r.task} | score={r.item_score} | {r.model} | {r.ref_phones}", "spans": r.spans} for r in pick.itertuples()]
            save_figure_v2(ctx, S, "alignment_examples", V.alignment_segments(examples, "Expected-text forced alignment (colour = GOP)"),
                           pick[[FILE, "task", "item_score", "model", "spans"]], "Alignment visualisation")
    gop = [x for x in cols if x.endswith("::gop_mean")]
    if gop:
        g0 = gop[0]
        vals = {f"{t} / {'correct' if s == 1 else 'incorrect'}": d[g0].to_numpy(float)
                for (t, s), d in c[c["item_score"].notna()].groupby(["task", "item_score"])}
        save_figure_v2(ctx, S, "confidence_distributions", V.distributions_by_group(vals, f"GOP distribution ({g0})", "GOP"),
                       c[[FILE, "task", "item_score", g0]], "Confidence distributions by task and official score")
    prof_t = pd.DataFrame(Zc, columns=cols).assign(task=c["task"].to_numpy()).groupby("task").mean().T
    save_figure_v2(ctx, S, "task_phonetic_profiles", V.heatmap(prof_t.head(60), "Task-specific phonetic profiles (item-centred mean)"),
                   prof_t, "Mean item-centred phonetic features by task")
    return {"status": "ok", "models": [j["key"] for j in jobs], "n_clips": int(len(c))}
