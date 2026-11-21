"""Section 28 - final automated scientific report.

All conclusions are generated programmatically from executed artifacts.
Skipped or failed modules are reported as such; no findings are fabricated.
"""
from __future__ import annotations

import html
import os
import shutil

import numpy as np
import pandas as pd

from .config import config_hash
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_parquet, atomic_write_text, read_json

QUESTIONS = [
    "What is the actual structure of the competition data?",
    "How much variation is explained by conventional screening?",
    "Which tasks carry the strongest predictive information?",
    "Which items are the most discriminative?",
    "Does adaptive testing behavior provide additional signal?",
    "Which acoustic feature families contain robust information?",
    "Which features predict outcomes after screening adjustment?",
    "Which discovered patterns generalize across schools?",
    "Which features appear sensitive to technical confounds?",
    "Does phonetic processing provide additional value?",
    "Are there reproducible student phenotypes?",
    "Which student profiles are difficult to predict?",
    "Which modalities recover baseline failures?",
    "Which experimental literacy indices are worth retaining?",
    "Which feature families are redundant or unstable?",
    "What should be prioritized in P02?",
    "Which hypotheses remain untested?",
    "What are the limitations of the current evidence?",
]


class Report:
    def __init__(self, ctx):
        self.ctx = ctx
        self.manifest = read_json(ctx.paths.outputs / "00_environment" / "stage_manifest.json", default={}) or {}

    def status(self, sid: str) -> str:
        return self.manifest.get(sid, {}).get("status", "NOT_STARTED")

    def ok(self, sid: str) -> bool:
        return self.ctx.stage_available(sid)

    def df(self, name: str, sid: str):
        return self.ctx.load_df(name, required=False) if self.ok(sid) else None

    def js(self, name: str, sid: str):
        return self.ctx.load_json(name, required=False) if self.ok(sid) else None

    def missing(self, *sids) -> str:
        return "Not answered: " + ", ".join(f"Section {s} status={self.status(s)}" for s in sids if not self.ok(s)) + "."


def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}" if x is not None and np.isfinite(float(x)) else "NA"
    except (TypeError, ValueError):
        return str(x)


def answers(R: Report) -> list:
    cfg = R.ctx.config
    out = []
    q = R.js("data_quality_report", "01")
    out.append(f"{q['n_students']} students and {q['n_clips']} clips; task families {q['task_families']}; {q['n_items']} items "
               f"(key source: {q['item_key_source']}); {q['n_schools']} schools; {q['n_instruments']} instruments; "
               f"{q['clips_missing_audio']} clips without matched audio." if q else R.missing("01"))
    b = R.df("baseline_results", "14")
    if b is not None and "pooled_r2" in b:
        r = b[(b["model_id"] == cfg["primary_baseline"]) & (b["scheme"] == cfg["primary_cv_scheme"])]
        best = b[b["scheme"] == cfg["primary_cv_scheme"]].sort_values("pooled_r2", ascending=False).head(1)
        out.append(f"Primary baseline {cfg['primary_baseline']}: pooled OOF R²={_fmt(r['pooled_r2'].iloc[0])} "
                   f"[{_fmt(r['pooled_r2_ci_lower'].iloc[0])}, {_fmt(r['pooled_r2_ci_upper'].iloc[0])}], mean fold R²={_fmt(r['mean_fold_r2'].iloc[0])}; "
                   f"best screening-only model {best['model_id'].iloc[0]} (R²={_fmt(best['pooled_r2'].iloc[0])})." if len(r) else "Primary baseline not available.")
    else:
        out.append(R.missing("14"))
    rel = R.df("task_relevance_matrix", "12")
    if rel is not None and len(rel):
        cols = [c for c in ("screening_signal", "correctness_signal", "residual_signal") if c in rel]
        out.append("; ".join(f"{r['task_family']}: " + ", ".join(f"{c}={_fmt(r[c])}" for c in cols) for _, r in rel.iterrows()))
    else:
        out.append(R.missing("12"))
    it = R.df("item_registry", "04")
    if it is not None and "point_biserial_rest" in it:
        top = it.sort_values("point_biserial_rest", ascending=False).head(5)
        out.append("Most discriminative items (rest-score point-biserial): " + ", ".join(f"{r['item_key']} ({_fmt(r['point_biserial_rest'], 2)})" for _, r in top.iterrows()))
    else:
        out.append(R.missing("04"))
    ov = R.js("order_validity", "05")
    abl = R.df("ablation_results", "20")

    def abl_row(exp, scheme="CV_A_REPEATED_STUDENT_KFOLD"):
        if abl is None:
            return None
        r = abl[(abl["experiment"] == exp) & (abl["scheme"] == scheme) & (abl["model"] == cfg["ablation_primary_model"])]
        return r.iloc[0] if len(r) else None
    if ov is not None and not ov.get("valid"):
        out.append(f"Sequential analysis disabled: {ov.get('reason')}.")
    else:
        a4 = abl_row("A04_THETA_ADAPTIVE_BEHAVIOR")
        out.append(f"A04 vs screening: delta R²={_fmt(a4['delta_r2_vs_screening'], 4)} [{_fmt(a4['delta_r2_vs_screening_ci_lower'], 4)}, "
                   f"{_fmt(a4['delta_r2_vs_screening_ci_upper'], 4)}]." if a4 is not None else R.missing("05", "20"))
    reg = R.df("feature_registry", "24")
    if reg is not None:
        fam = reg[reg["selection_status"] == "ROBUST_SIGNAL"]["feature_family"].value_counts()
        out.append("Robust-signal features per family: " + (", ".join(f"{k}={v}" for k, v in fam.items()) if len(fam) else "none met all robustness criteria") + ".")
        top = reg[reg["selection_status"].isin(["ROBUST_SIGNAL", "PROMISING_SIGNAL"])].assign(a=lambda d: d["residual_spearman"].abs()).sort_values("a", ascending=False).head(8)
        out.append("Top screening-adjusted features: " + (", ".join(f"{r['feature_name']} (rho_res={_fmt(r['residual_spearman'])}, {r['selection_status']})" for _, r in top.iterrows()) or "none") + ".")
    else:
        out += [R.missing("24"), R.missing("24")]
    if abl is not None:
        b13 = abl_row("A13_TABULAR_MULTIMODAL", "CV_B_SCHOOL_GROUP_KFOLD")
        sc = R.df("sign_consistency", "17")
        txt = f"School-grouped multimodal gain vs tabular: {_fmt(b13['delta_r2_vs_full_tabular'], 4)}" if b13 is not None else "School-grouped ablation unavailable"
        if sc is not None and len(sc):
            txt += f"; median school sign consistency of top speech features {_fmt(sc['school_sign_consistency'].median(), 2)}"
        out.append(txt + ".")
    else:
        out.append(R.missing("20"))
    nc = R.df("negative_controls", "17")
    if nc is not None and "pooled_r2" in nc:
        sub = nc[nc["control"].str.startswith("NC") & (nc.get("variant") == "alone")]
        out.append("Negative controls (alone, pooled R²): " + ", ".join(f"{r['control']}={_fmt(r['pooled_r2'])}" for _, r in sub.iterrows()) + "."
                   + (f" Confounded-signal features: {int((reg['selection_status'] == 'CONFOUNDED_SIGNAL').sum())}." if reg is not None else ""))
    else:
        out.append(R.missing("17"))
    a11 = abl_row("A11_TABULAR_PHONETIC") if abl is not None else None
    asr = R.js("asr_status", "11")
    out.append(f"A11 phonetic gain vs tabular: {_fmt(a11['delta_r2_vs_full_tabular'], 4)}." if a11 is not None else
               f"Phonetic ASR not evaluated (ASR status: {(asr or {}).get('status', R.status('11'))}).")
    info = R.js("cluster_space_info", "15")
    out.append("; ".join(f"{k}: {v.get('method')} k={v.get('k')} stable={v.get('stable')}" for k, v in info.items() if v.get("method")) if info else R.missing("15"))
    fs = R.df("failure_summary", "22")
    if fs is not None:
        top = fs.sort_values("mean_abs_baseline_error", ascending=False).head(4)
        out.append("Hardest phenotypes by mean |baseline error|: " + ", ".join(f"{r['phenotype']} ({_fmt(r['mean_abs_baseline_error'])}, n={r['n']})" for _, r in top.iterrows()) + ".")
        rec = fs.sort_values("mean_error_reduction", ascending=False).head(3)
        out.append("Largest multimodal error reductions: " + ", ".join(f"{r['phenotype']} ({_fmt(r['mean_error_reduction'])})" for _, r in rec.iterrows()) + ".")
    else:
        out += [R.missing("22"), R.missing("22")]
    li = R.df("literacy_index_report", "19")
    out.append(("Retained indices: " + (", ".join(li.loc[li.get("retention") == "RETAIN", "index"]) or "none") + ".") if li is not None and "retention" in li else R.missing("19"))
    if reg is not None:
        red = reg[reg["selection_status"].isin(["REDUNDANT_SIGNAL", "UNSTABLE_SIGNAL"])]["feature_family"].value_counts()
        out.append("Redundant/unstable features by family: " + (", ".join(f"{k}={v}" for k, v in red.items()) or "none") + ".")
    else:
        out.append(R.missing("24"))
    hyp = R.df("hypothesis_registry", "25")
    prio = []
    if hyp is not None:
        prio = hyp[hyp["support_status"].isin(["SUPPORTED", "PARTIALLY_SUPPORTED"])]["research_question"].tolist()
    if reg is not None:
        fams = reg[reg["selection_status"] == "ROBUST_SIGNAL"]["feature_family"].value_counts().head(3).index.tolist()
        prio.append(f"Feature engineering for families with robust signals: {fams or 'none yet'}")
    out.append("; ".join(prio) + "." if prio else "No supported findings yet; prioritise enabling skipped modalities and confirmation.")
    out.append(", ".join(hyp.loc[hyp["support_status"] == "NOT_TESTED", "hypothesis_id"]) or "none" if hyp is not None else R.missing("25"))
    lim = ["observational associations only (no causal claims)", "voice conversion alters physiological voice measures",
           "VAD is not speaker identification; other voices may be present", "ASR disagreement is an uncertain proxy",
           "feature-level screening has selection-induced optimism; rely on nested CV deltas"]
    if not cfg.get("official_metric_verified"):
        lim.append(f"official metric assumed to be '{cfg['official_metric']}' (unverified)")
    failed = [s for s, e in R.manifest.items() if e.get("status") in {"FAILED", "BLOCKED_DEPENDENCY", "SKIPPED_DISABLED"}]
    if failed:
        lim.append(f"stages not completed: {sorted(failed)}")
    out.append("; ".join(lim) + ".")
    return out


def _copy_or_status(R: Report, name: str, sid: str, path, kind: str = "csv") -> None:
    df = R.df(name, sid)
    if df is None:
        df = pd.DataFrame([{"status": f"NOT_AVAILABLE: Section {sid} status={R.status(sid)}"}])
    if kind == "csv":
        atomic_write_csv(df, path)
    else:
        atomic_write_parquet(df, path)


def stage_final_report(ctx) -> dict:
    R = Report(ctx)
    out_dir = ctx.out_dir()
    ans = answers(R)
    q = R.js("data_quality_report", "01")
    atomic_write_json(out_dir / "P01_DATA_QUALITY_REPORT.json", q or {"status": f"NOT_AVAILABLE: Section 01 status={R.status('01')}"})
    _copy_or_status(R, "feature_registry", "24", out_dir / "P01_FEATURE_DISCOVERY_REPORT.csv")
    _copy_or_status(R, "hypothesis_registry", "25", out_dir / "P01_HYPOTHESIS_REGISTRY.csv")
    parts = []
    b = R.df("baseline_results", "14")
    if b is not None:
        parts.append(b.assign(source="baseline_screening"))
    rl = R.df("residual_learning_results", "21")
    if rl is not None:
        parts.append(rl.rename(columns={"approach": "model_id"}).assign(source="residual_learning"))
    atomic_write_csv(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame([{"status": "NOT_AVAILABLE"}]),
                     out_dir / "P01_MODEL_COMPARISON.csv")
    _copy_or_status(R, "ablation_results", "20", out_dir / "P01_ABLATION_RESULTS.csv")
    nc = R.df("negative_controls", "17")
    cm = R.df("confound_matrix", "17")
    conf = pd.concat([nc.assign(section="negative_controls"), cm.assign(section="confound_matrix")], ignore_index=True) \
        if nc is not None and cm is not None else pd.DataFrame([{"status": f"NOT_AVAILABLE: Section 17 status={R.status('17')}"}])
    atomic_write_csv(conf, out_dir / "P01_CONFOUND_REPORT.csv")
    _copy_or_status(R, "failure_registry", "22", out_dir / "P01_FAILURE_REGISTRY.parquet", kind="parquet")
    status_rows = [{"stage": s, "name": e.get("name"), "status": e.get("status"), "reason": e.get("reason", "")}
                   for s, e in sorted(R.manifest.items())]
    md = ["# WildSpeech P01 - Executive summary", "",
          f"Configuration hash: `{config_hash(ctx.config)}` | profile: {ctx.config.get('run_mode')}", "",
          "All statements below are generated from executed artifacts. Skipped or failed sections are stated explicitly.", ""]
    for i, (qq, a) in enumerate(zip(QUESTIONS, ans), 1):
        md += [f"**{i}. {qq}**", "", a, ""]
    md += ["## Stage status", "", "| stage | name | status | reason |", "|---|---|---|---|"]
    md += [f"| {r['stage']} | {r['name']} | {r['status']} | {str(r['reason'])[:120]} |" for r in status_rows]
    atomic_write_text(out_dir / "P01_EXECUTIVE_SUMMARY.md", "\n".join(md))
    hyp = R.df("hypothesis_registry", "25")
    nxt = ["# P01 next-step recommendations", ""]
    if hyp is not None:
        for _, r in hyp.iterrows():
            nxt.append(f"- **{r['hypothesis_id']}** ({r['support_status']}): {r['research_question']} -> {r['recommended_next_step']}")
    else:
        nxt.append(f"- Hypothesis registry unavailable (Section 25 status={R.status('25')}).")
    nxt += ["", f"- Priorities: {ans[15]}"]
    atomic_write_text(out_dir / "P01_NEXT_STEP_RECOMMENDATIONS.md", "\n".join(nxt))
    outputs = []
    if ctx.config.get("save_html_report", True):
        figs = ctx.figures.index
        h = ["<html><head><meta charset='utf-8'><title>WildSpeech P01 scientific report</title><style>body{font-family:sans-serif;max-width:1100px;margin:auto}"
             "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px}img{max-width:100%}</style></head><body>",
             "<h1>WildSpeech P01 - Literacy signal discovery report</h1><p>Local, self-contained report generated from execution artifacts.</p>"]
        for i, (qq, a) in enumerate(zip(QUESTIONS, ans), 1):
            h.append(f"<h3>{i}. {html.escape(qq)}</h3><p>{html.escape(a)}</p>")
        h.append("<h2>Stage status</h2>" + pd.DataFrame(status_rows).to_html(index=False, escape=True))
        for name, sid, title in (("baseline_results", "14", "Screening baselines"), ("ablation_results", "20", "Ablation"),
                                 ("hypothesis_registry", "25", "Hypotheses"), ("negative_controls", "17", "Negative controls")):
            d = R.df(name, sid)
            if d is not None:
                h.append(f"<h2>{title}</h2>" + d.head(60).to_html(index=False, float_format=lambda v: f"{v:.4f}"))
        h.append(f"<h2>Figures ({len(figs)})</h2>")
        for _, r in figs.iterrows():
            try:
                rel = os.path.relpath(str(r["filename"]), str(out_dir))
            except ValueError:
                rel = str(r["filename"])
            h.append(f"<h4>{html.escape(str(r['title']))}</h4><p>{html.escape(str(r['description']))} (n={html.escape(str(r['sample_size']))})</p>"
                     f"<img src='{html.escape(rel)}'>")
        h.append("</body></html>")
        atomic_write_text(out_dir / "P01_SCIENTIFIC_REPORT.html", "\n".join(h))
        outputs.append("P01_SCIENTIFIC_REPORT.html")
    for f in ["P01_EXECUTIVE_SUMMARY.md", "P01_DATA_QUALITY_REPORT.json", "P01_FEATURE_DISCOVERY_REPORT.csv",
              "P01_HYPOTHESIS_REGISTRY.csv", "P01_MODEL_COMPARISON.csv", "P01_ABLATION_RESULTS.csv",
              "P01_CONFOUND_REPORT.csv", "P01_FAILURE_REGISTRY.parquet", "P01_NEXT_STEP_RECOMMENDATIONS.md"] + outputs:
        ctx.store.register(f, out_dir / f, ctx.current_stage, "report")
        shutil.copy2(out_dir / f, ctx.paths.reports / f)
    names = ["P01_EXECUTIVE_SUMMARY.md", "P01_DATA_QUALITY_REPORT.json", "P01_FEATURE_DISCOVERY_REPORT.csv",
             "P01_HYPOTHESIS_REGISTRY.csv", "P01_MODEL_COMPARISON.csv", "P01_ABLATION_RESULTS.csv", "P01_CONFOUND_REPORT.csv",
             "P01_FAILURE_REGISTRY.parquet", "P01_NEXT_STEP_RECOMMENDATIONS.md"] + outputs
    return {"outputs": names, "summary": {"n_answered": sum(not a.startswith("Not answered") for a in ans)}}
