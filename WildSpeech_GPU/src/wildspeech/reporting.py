"""P15 — automated research report assembled ONLY from files written by earlier stages (nothing fabricated).
Sections without outputs are reported as 'not run'."""
from __future__ import annotations

import html
import time

import numpy as np
import pandas as pd

from .analysis.indices import DISCLAIMER
from .io_utils import atomic_write_text, read_json
from .paths import STAGE_NAMES


def _md_table(df: pd.DataFrame | None, cols: list[str] | None = None, n: int = 15, floatfmt: str = "{:.4g}") -> str:
    if df is None or df.empty:
        return "_not available (stage not run or produced no rows)_\n"
    d = df[[c for c in (cols or df.columns) if c in df.columns]].head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    body = "".join("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |\n" for r in d.itertuples(index=False))
    return head + body


def _recommendations(ctx) -> list[str]:
    rec = []
    abl = ctx.read_stage_table("P10", "family_ablation")
    if abl is not None and "delta_ci_low" in abl:
        good = abl[(abl["kind"] == "single_add") & (abl["delta_ci_low"] > 0)]
        if not good.empty:
            rec.append("Families with a robust incremental gain over screening (delta-R2 CI > 0): "
                       + ", ".join(good.sort_values("delta_r2", ascending=False)["families"].str.split("+").str[-1]) + ".")
        else:
            rec.append("No single feature family produced a delta-R2 CI above 0; prioritise screening + item-level features.")
    art = ctx.read_stage_table("P08", "artifact_correlations")
    if art is not None and not art.empty and "artifact_flag" in art:
        rec.append(f"{int(art['artifact_flag'].sum())} acoustic student features correlate with recording-quality covariates above the flag; "
                   "use artifact-adjusted estimates when interpreting them.")
    stab = ctx.read_stage_table("P08", "perturbation_stability_summary")
    if stab is not None and not stab.empty:
        rec.append(f"{int((stab['min_stability'] < float(ctx.cfg.get_path('analysis.stability_flag', 0.7))).sum())} clip features are unstable "
                   "under at least one perturbation (pitch shift / noise / low-pass); treat them as anonymisation-sensitive.")
    fail = ctx.read_stage_table("P12", "failure_recovery")
    if fail is not None and not fail.empty:
        rec.append(f"{int(fail.groupby('failure_cluster')['recovers'].any().sum())} baseline failure clusters are recovered by at least one model.")
    if not rec:
        rec.append("Run the FAST or FULL profile to generate evidence-based recommendations.")
    return rec


def build_report(ctx) -> dict:
    t = time.strftime("%Y-%m-%d %H:%M")
    parts = [f"# {ctx.cfg.get_path('report.title', 'WildSpeech report')}\n", f"_Generated {t} · profile {ctx.profile}_\n"]
    parts.append("## Stage status\n")
    rows = []
    for sid in STAGE_NAMES:
        st = read_json(ctx.stage_dir(sid) / "_status.json", {}) or {}
        rows.append({"stage": sid, "name": STAGE_NAMES[sid], "status": st.get("status", "not run"), "seconds": st.get("seconds"),
                     "finished": st.get("finished", ""), "note": st.get("note", "")})
    status = pd.DataFrame(rows)
    parts.append(_md_table(status, n=20))
    hw = ctx.read_stage_json("P00", "hardware_report", {}) or {}
    if hw:
        parts.append(f"\n**Hardware:** {hw.get('gpu_name', 'no CUDA GPU')} · torch {hw.get('torch')} (CUDA {hw.get('torch_cuda_build')}) · "
                     f"RAM {float(hw.get('ram_total_gb') or float('nan')):.1f} GB · {hw.get('logical_cpus')} logical CPUs\n")
    parts.append("\n## Data integrity\n" + _md_table(ctx.read_stage_table("P00", "integrity_checks"), n=40))
    parts.append("\n## Screening baselines (P01)\n" + _md_table(ctx.read_stage_table("P01", "tabular_baselines"),
                                                              ["name", "r2_mean", "r2_sd", "rmse_mean", "mae_mean"]))
    parts.append("\n## Screening scores vs outcome\n" + _md_table(ctx.read_stage_table("P01", "screening_associations"),
                                                                 ["screening", "scope", "spearman", "ci_low", "ci_high", "q"]))
    parts.append("\n## Acoustic correctness contrasts (P02, item-centred)\n" + _md_table(
        ctx.read_stage_table("P02", "correctness_contrasts_item_centred"), ["task", "feature", "cliffs_delta", "std_diff", "q"]))
    parts.append("\n## Residual discovery (P07)\n" + _md_table(ctx.read_stage_table("P07", "residual_associations"),
                                                              ["feature", "rho_resid", "rho_ci_low", "rho_ci_high", "q_resid", "replication_rate"]))
    parts.append("\n### Matched over- vs under-achievers\n" + _md_table(ctx.read_stage_table("P07", "matched_comparisons"),
                                                                       ["feature", "n_pairs", "mean_diff", "cohens_d", "q"]))
    parts.append("\n## Phenotypes (P06)\n" + _md_table(ctx.read_stage_table("P06", "cluster_summary")))
    parts.append("\n## Robustness (P08)\n### Domain generalisation\n" + _md_table(ctx.read_stage_table("P08", "domain_generalisation"),
                                                                                 ["name", "scheme", "r2_mean", "r2_sd", "n_students"]))
    parts.append("\n### Artifact-adjusted associations\n" + _md_table(ctx.read_stage_table("P08", "artifact_adjusted_associations"),
                                                                     ["feature", "partial_rho_adj", "q_adj"]))
    parts.append(f"\n## Experimental literacy indices (P09)\n> {DISCLAIMER}\n\n" + _md_table(ctx.read_stage_table("P09", "index_validation"),
                 ["index", "delta_r2", "ci_low", "ci_high", "cronbach_alpha_mean", "spearman_brown", "sign_stability"]))
    parts.append("\n## Model benchmark (P10)\n" + _md_table(ctx.read_stage_table("P10", "model_benchmark"),
                                                           ["name", "r2_mean", "r2_sd", "delta_r2", "delta_ci_low", "delta_ci_high"], n=25))
    parts.append("\n### Family ablation\n" + _md_table(ctx.read_stage_table("P10", "family_ablation"),
                                                      ["families", "kind", "r2_mean", "delta_r2", "delta_ci_low", "delta_ci_high"], n=25))
    parts.append("\n## Multimodal learning (P11)\n" + _md_table(ctx.read_stage_table("P11", "multimodal_results"),
                                                               ["name", "r2_mean", "r2_sd", "vs_best_p10_delta_r2", "vs_best_p10_delta_ci_low"]))
    parts.append("\n## Failure analysis (P12)\n" + _md_table(ctx.read_stage_table("P12", "failure_cluster_profiles")))
    parts.append("\n## Champion (P13)\n" + _md_table(ctx.read_stage_table("P13", "champion_comparison")))
    parts.append("\n" + _md_table(ctx.read_stage_table("P13", "conformal_coverage")))
    sub = read_json(ctx.paths.submissions / "submission_meta.json")
    parts.append("\n## Submission (P14)\n" + (f"`{sub['file']}` · {sub['rows']} rows · champion {sub['champion']}\n" if sub else "_not generated_\n"))
    hyp = ctx.registries.hypotheses.frame()
    parts.append("\n## Hypothesis registry\n" + _md_table(hyp, ["hypothesis_id", "statement", "status", "evidence", "effect", "q_value"], n=40))
    feats = ctx.registries.features.frame()
    if not feats.empty and "q_resid" in feats:
        top = feats.dropna(subset=["q_resid"]).sort_values("q_resid").head(25)
        parts.append("\n## Feature registry (top by residual q-value)\n" + _md_table(top, ["feature", "family", "rho_resid", "q_resid",
                                                                                        "replication_rate", "univariate_rho_outcome"], n=25))
    parts.append("\n## Recommendations\n" + "".join(f"- {r}\n" for r in _recommendations(ctx)))
    parts.append("\n## Safeguards\n- All audio processed locally; no external APIs; models loaded with local_files_only.\n"
                 "- Student-disjoint folds; item statistics, IRT, PCA, selection, stacking fitted inside training folds.\n"
                 "- No test labels, test-fitted normalisation or pseudo-labelling; school_id/score_status never used as features.\n"
                 "- Indices are experimental measurements, not diagnoses.\n")
    md = "\n".join(parts)
    out_md = ctx.paths.reports / "WILDSPEECH_RESEARCH_REPORT.md"
    atomic_write_text(out_md, md)
    res = {"markdown": str(out_md)}
    if ctx.cfg.get_path("report.html", True):
        try:
            import markdown

            body = markdown.markdown(md, extensions=["tables"])
        except Exception:
            body = f"<pre>{html.escape(md)}</pre>"
        page = ("<!doctype html><html><head><meta charset='utf-8'><title>WildSpeech report</title><style>"
                "body{font-family:system-ui,'Segoe UI',sans-serif;max-width:1100px;margin:2rem auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb}"
                "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #e1e0d9;padding:3px 6px}"
                "th{background:#f0efec}code{background:#f0efec;padding:1px 4px}</style></head><body>" + body + "</body></html>")
        out_html = ctx.paths.reports / "WILDSPEECH_RESEARCH_REPORT.html"
        atomic_write_text(out_html, page)
        res["html"] = str(out_html)
    for name, reg in (("feature_registry", ctx.registries.features), ("hypothesis_registry", ctx.registries.hypotheses),
                      ("experiment_registry", ctx.registries.experiments), ("failure_registry", ctx.registries.failures)):
        df = reg.frame()
        if not df.empty:
            df.to_csv(ctx.paths.reports / f"{name}.csv", index=False)
    return res
