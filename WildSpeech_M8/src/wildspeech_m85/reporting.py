"""Private full-science atlas and immutable milestone decision reporting."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wildspeech_m84.data import atomic_text,atomic_json


def render_results(root,summary,diagnostic):
    root=Path(root);fig=root/"figures";fig.mkdir(parents=True,exist_ok=True)
    records=pd.DataFrame(summary.get("model_metrics",[]))
    if not records.empty:
        records.to_csv(root/"tables"/"paired_model_results.csv",index=False)
        x=records[records.model!="E00"].sort_values("mean_student_pooled_r2",ascending=False)
        figure,ax=plt.subplots(figsize=(11,5))
        ax.bar(x.model, x.mean_student_pooled_r2)
        ax.axhline(summary["reference"]["mean_student_pooled_r2"],
                   color="black",linestyle="--",label="Frozen reference")
        ax.set(xlabel="Preregistered experiment",ylabel="Student pooled OOF R²",
               title=f"M8.5 {summary['mode']}: frozen-fold comparison")
        ax.legend()
        figure.tight_layout();figure.savefig(fig/"04_candidate_pooled_r2.png",dpi=170)
        plt.close(figure)
        if "mean_fold_delta_r2" in records:
            y=records[records.model!="E00"]
            figure,ax=plt.subplots(figsize=(11,5))
            ax.bar(y.model,y.mean_fold_delta_r2.fillna(0))
            ax.axhline(.010,color="black",linestyle="--",label="Improvement threshold (not sufficient)")
            ax.set(xlabel="Preregistered experiment",ylabel="Mean paired fold ΔR²",
                   title="Paired model improvements versus frozen reference")
            ax.legend();figure.tight_layout()
            figure.savefig(fig/"05_paired_delta_r2.png",dpi=170)
            plt.close(figure)
    lines=[
        "# WildSpeech M8.5 — M0+ Screening Milestone Decision","",
        "**Status:** "+summary["decision"],
        "**Execution mode:** "+summary["mode"]+(" (exploratory ONLY; cannot promote)" if summary["mode"]!="FULL" else ""),
        "**Reference frozen fold hash:** "+summary["fold_hash"],
        "**Reference OOF SHA-256:** "+summary["reference_oof_sha256"],
        "**Phase 0 benchmark:** 50-fold R²=0.1932274944, 10-repeat pooled R²=0.1991133687 (executed M8.4).",
        "**New results are not publicly scored and must not be interpreted as leaderboard performance.**","",
        "## Scientific hypotheses and compliance",
        "KTEA-3 school-disjoint deterioration and deletion missingness motivated targeted S4 and S5 blocks.",
        "All fitted features are learned on outer-training students; S2 training encodings leave each student's scored attempts out.",
        "Train-only label information is used exclusively for model fits, never prompt difficulty.",
        "School ID and score_status are NEVER predictor columns; chronology-derived features are disabled.","",
        "## Frozen evaluation and selection","",
        "Reference OOF is loaded from M8.4 and never retrained. Student and school splits are unmodified.",
        "Full gate requires paired Δ mean-fold R²≥+.010, corrected t≥2, ≥60% wins, school pooled Δ≥−.005 and per-instrument Δ≥−.010.",
        "E05–E09 are diagnostic ablations/audits; not independently promoted based on repeated selection over outer folds.","",
        "## Preregistered model summaries","",
    ]
    if not records.empty:
        lines.append(records.to_markdown(index=False,floatfmt=".5f"))
    else: lines.append("No experiments completed.")
    lines.extend(["","## Observational screening diagnostics","",
                 f"Instrument count: {diagnostic.get('n_instruments','?')}. "
                 f"Schools: {diagnostic.get('n_schools','?')}. "
                 f"Observed prompt keys: {diagnostic.get('n_prompt_keys','?')}.","",
                 "## Model decision","",
                 summary["decision_explanation"],"",
                 "## Risks and remaining work","",
                 "- Outer repeated-CV corrected t does not adjust for all model-family search.",
                 "- School- and instrument-level associations are observational, not causal.",
                 "- Submission #2 is generated only for FULL and PROMOTE with raw singleton inference passing.",
                 "- The reference fallback is preserved when no candidate is promoted.",
                 "- Genuine child-speech features belong to Phase 2/3, not M0+.",""])
    report=root/"reports"/"M85_M0PLUS_DECISION.md"
    atomic_text(report,"\n".join(lines))
    atomic_text(root/"reports"/"M85_SCIENTIFIC_ATLAS.md",
                "\n".join(["# M8.5 — Screening Scientific Atlas","",
                           (root/"diagnostics"/"school_transfer_diagnostics.md").read_text(),
                           "## Experiment decision","",summary["decision_explanation"],""]))
    atomic_json(root/"reports"/"M85_RESULTS.json",summary)
    return str(report)
