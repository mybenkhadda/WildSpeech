"""P25 — exhaustive, label-free phonetic feature engineering from the P24 outputs.

Per model: word-level, phoneme-level, confidence (only where the model exposes probabilities/posteriors), timing
and task-specific features -> cache/v3/phonetic/<model>/<split>/ ; feature registry.
Validation against the only available training annotation (official item correctness): per-feature clip AUROC by
task; child-vs-general recogniser comparison with student-block bootstrap (V3H03). Manual-review sheet for
permitted human review of training clips (text only — no audio is copied). Figures: phoneme confusion matrices,
expected-vs-recognised alignments, phoneme error heatmap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..asr.checkpointing import ASRCheckpointStore, load_all_asr
from ..asr.model_registry import registry_for
from ..phonetics.alignment import ops_from_string
from ..phonetics.builder import PhoneticFeatureBuilder, load_phonetic_tables, model_prefix
from ..phonetics.error_analysis import confusion_matrix, confusion_table
from ..phonetics.g2p import is_vowel
from ..v3 import viz as V
from ..v3.common import begin, not_executed
from ..v3.registry import save_figure_v3

S = "P25"
DIRECTION = {"ph_per": -1, "w_wer": -1, "c_cer": -1, "ph_near_correct": 1, "w_completeness": 1, "cf_gop_mean": 1,
             "cf_tok_lp_mean": 1, "cf_expected_ll": 1, "cf_post_mean": 1, "cf_supported_frac": 1, "ph_exact": 1,
             "w_exact": 1, "cf_frame_entropy": -1, "tm_s_per_phone": -1}


def model_outputs(ctx) -> dict[str, str]:
    specs, _ = registry_for(ctx)
    return {k: s.output for k, s in specs.items()}


def run(ctx) -> dict:
    from ..discovery.residual_analysis import auc, auc_diff_block_bootstrap

    from ..asr.outputs import require_outputs, usable_pairs

    reg, v3r = begin(ctx, S)
    p = ctx.cfg.get_path("v3.phonetics", {}) or {}
    outputs = model_outputs(ctx)
    specs, _ = registry_for(ctx)
    splits = list(ctx.cfg.get_path("v3.asr.splits", ["train"]))
    # ---- gate: real, accepted ASR transcripts / phones must exist (never inferred from P24's status)
    ok, acc, why = require_outputs(ctx, S, "text", 1, ("train",))       # recognisers only (encoders feed P28)
    if not ok:
        return not_executed(ctx, S, why)
    usable = usable_pairs(ctx, tuple(splits))
    b = PhoneticFeatureBuilder(ctx)
    built = []
    for k in [m for m in ctx.cfg.get_path("v3.asr.models", []) if m in outputs and m in acc]:
        st = ASRCheckpointStore(ctx.paths.v3_cache, k)
        for split in splits:
            if (k, split) not in usable:
                continue
            asr = st.load(split)
            if asr.empty:
                continue
            fp = (st.store.fingerprint(k, split) or {}).get("fingerprint")
            F, D = b.build(k, split, asr, outputs[k], fp)
            built.append({"model_key": k, "split": split, "n_clips": len(F), "n_features": max(F.shape[1] - 1, 0),
                          "output": outputs[k], "g2p_backend": b.exp.g2p.backend, "g2p_exact": b.exp.exact})
            v3r.register_features([c for c in F.columns if c != FILE], S, k, "clip", True, f"P25 clip features ({outputs[k]})")
    ctx.save_table(pd.DataFrame(built), S, "feature_tables")
    if not built:
        return not_executed(ctx, S, f"accepted ASR outputs {acc} are not among v3.asr.models / registry outputs")
    wide, det = load_phonetic_tables(ctx, ("train",))
    clips = ctx.bundle().train_clips[[FILE, ID, "task", "item_score", "expected_text"]].drop_duplicates(FILE)
    m = clips.merge(wide, on=FILE, how="inner")
    # ---- validation vs official correctness (the available annotation)
    rows = []
    models = sorted({c.split("::")[0] for c in wide.columns if "::" in c})
    for col in [c for c in wide.columns if "::" in c]:
        feat = col.split("::", 1)[1]
        for t, g in m[m["item_score"].notna()].groupby("task"):
            a = auc(g["item_score"], g[col])
            rows.append({"feature": col, "model": col.split("::")[0], "name": feat, "task": t, "auc": a,
                         "auc_oriented": (a if DIRECTION.get(feat, 1) > 0 else 1 - a) if np.isfinite(a) else np.nan,
                         "coverage": float(g[col].notna().mean()), "n": int(g[col].notna().sum())})
    val = pd.DataFrame(rows)
    ctx.save_table(val, S, "correctness_validation_auc")
    # ---- child vs general recogniser (V3H03)
    key_feat = {}
    for k in outputs:
        pre = model_prefix(k)
        for n in ("ph_near_correct", "w_completeness", "ph_per", "w_wer"):
            c = f"{pre}::{n}"
            if c in wide and wide[c].notna().mean() > 0.5:
                key_feat[k] = (c, DIRECTION.get(n, 1))
                break
    nboot = int(ctx.cfg.get_path("v3.discovery.bootstrap", 1000))
    comp = []
    child = [k for k in key_feat if specs.get(k) and specs[k].is_child_model]
    general = [k for k in key_feat if specs.get(k) and not specs[k].is_child_model]
    for cm in child:
        for gm in general:
            d = m.dropna(subset=["item_score", key_feat[cm][0], key_feat[gm][0]]).copy()
            d["s_child"] = d[key_feat[cm][0]] * key_feat[cm][1]
            d["s_general"] = d[key_feat[gm][0]] * key_feat[gm][1]
            if len(d) < 100:
                continue
            r = auc_diff_block_bootstrap(d, "item_score", "s_child", "s_general", nboot, ctx.seed)
            comp.append({"child_model": cm, "general_model": gm, "child_feature": key_feat[cm][0],
                         "general_feature": key_feat[gm][0], "n": len(d), **r})
    comp = pd.DataFrame(comp)
    ctx.save_table(comp, S, "child_vs_general_auc")
    if not comp.empty:
        best = comp.sort_values("delta_auc", ascending=False).iloc[0]
        sup = bool(np.isfinite(best["ci_low"]) and best["ci_low"] > 0)
        v3r.update_hypothesis("V3H03", "supported" if sup else "not_supported", S,
                              f"{best['child_model']} vs {best['general_model']}: ΔAUROC={best['delta_auc']:.4f} "
                              f"[{best['ci_low']:.4f}, {best['ci_high']:.4f}]", effect=float(best["delta_auc"]),
                              ci_low=float(best["ci_low"]), ci_high=float(best["ci_high"]), n=int(best["n"]))
        save_figure_v3(ctx, S, "asr_model_comparison", V.forest(comp.assign(label=comp["child_model"] + " vs " + comp["general_model"]),
                       "delta_auc", "ci_low", "ci_high", "label", "Clip-correctness AUROC: child vs general recogniser",
                       xlabel="ΔAUROC (student-block bootstrap 95% CI)"), comp, "ASR model comparison on official correctness")
    else:
        v3r.update_hypothesis("V3H03", "not_tested", S, "needs >= 1 child and >= 1 general model with outputs")
    # ---- confusion matrices, error heatmaps, alignment examples (phone-output models + G2P-derived)
    fig_n = 0
    top = int(ctx.cfg.get_path("v3.disagreement.top_confusions", 25))
    for k, g in det.groupby("model_key") if not det.empty else []:
        long = confusion_table(g["ops"].fillna(""), k)
        ctx.save_table(long, S, f"confusion_long_{k}")
        mat = confusion_matrix(long, top)
        if not mat.empty:
            save_figure_v3(ctx, S, f"phoneme_confusion_{k}", V.heatmap(mat, f"Expected→recognised phone confusion ({k}, row-normalised)",
                           xlabel="recognised", ylabel="expected"), mat, f"Phoneme confusion matrix — {k}")
        heat = _error_heatmap(g)
        if not heat.empty:
            save_figure_v3(ctx, S, f"phoneme_error_heatmap_{k}", V.heatmap(heat, f"Phoneme error rate by position and task ({k})",
                           annotate=True), heat, f"Phoneme-level error heatmap — {k}")
        exs = _alignment_examples(g, 8)
        if exs:
            save_figure_v3(ctx, S, f"alignment_examples_{k}", V.alignment_panel(exs, f"Expected vs recognised phones ({k})"),
                           pd.DataFrame([{"label": e["label"], "ops": str(e["ops"])} for e in exs]), f"Alignment examples — {k}")
        fig_n += 1
    # ---- manual review sheet (text only)
    n_rev = int(p.get("manual_review_clips", 60))
    if not det.empty and n_rev:
        piv = det.pivot_table(index=FILE, columns="model_key", values="transcript", aggfunc="first")
        php = det.pivot_table(index=FILE, columns="model_key", values="hyp_phones", aggfunc="first").add_suffix("__phones")
        sheet = clips.set_index(FILE)[["task", "expected_text", "item_score"]].join(piv, how="inner").join(php, how="left")
        sheet = sheet.sample(min(n_rev, len(sheet)), random_state=ctx.seed).reset_index()
        sheet["reviewer_heard_expected"] = ""
        sheet["reviewer_phone_errors"] = ""
        sheet["reviewer_notes"] = ""
        ctx.save_table(sheet, S, "manual_review_sheet")
        done = ctx.stage_dir(S) / "manual_review_completed.csv"
        from ..io_utils import read_csv_safe

        rv = read_csv_safe(done) if done.exists() else pd.DataFrame()
        if not rv.empty and {"reviewer_heard_expected", FILE} <= set(rv.columns):
            rv = rv[rv["reviewer_heard_expected"].astype(str).str.strip().isin(["0", "1"])]
            agree = []
            for k, (c, sgn) in key_feat.items():
                j = rv.merge(m[[FILE, c]], on=FILE, how="inner")
                agree.append({"model_key": k, "feature": c, "n_reviewed": len(j),
                              "auc_vs_reviewer": auc(j["reviewer_heard_expected"].astype(float), j[c] * sgn)})
            ctx.save_table(pd.DataFrame(agree), S, "manual_review_agreement")
    return {"models": models, "n_clips_with_features": int(len(wide)), "figures_models": fig_n}


def _error_heatmap(g: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    rows = []
    for task, ops_s in zip(g["task"], g["ops"].fillna("")):
        ops = ops_from_string(ops_s)
        n = max([o.i for o in ops if o.op != "I"] + [0]) + 1
        for o in ops:
            if o.op == "I" or o.ref is None:
                continue
            b = ["initial", "medial", "final"][min(bins - 1, int(bins * o.i / max(n, 1)))]
            rows.append({"row": f"{task} · {'vowel' if is_vowel(o.ref) else 'consonant'}", "pos": b, "err": float(o.op != "M")})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return d.pivot_table(index="row", columns="pos", values="err", aggfunc="mean").reindex(columns=["initial", "medial", "final"])


def _alignment_examples(g: pd.DataFrame, n: int) -> list[dict]:
    out = []
    g = g[g["ops"].fillna("") != ""]
    for task, gg in g.groupby("task"):
        gg = gg.assign(nerr=gg["ops"].map(lambda s: sum(o.op != "M" for o in ops_from_string(s))))
        pick = pd.concat([gg[gg["nerr"] == 0].head(1), gg[gg["nerr"] > 0].sort_values(FILE).head(2)])
        for r in pick.to_dict("records"):
            out.append({"label": f"{task}: {r['expected_text']}"[:40],
                        "ops": [(o.op, o.ref, o.hyp) for o in ops_from_string(r["ops"])]})
    return out[:n]


def all_asr(ctx, splits=("train",)) -> pd.DataFrame:
    return load_all_asr(ctx.paths.v3_cache, splits)
