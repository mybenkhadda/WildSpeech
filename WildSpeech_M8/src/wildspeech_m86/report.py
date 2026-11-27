"""Formal M8.6 engineering/science decision with reproducible private artifacts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from wildspeech_m84.data import atomic_text,atomic_json
from .diagnostics import score_plots

def write_report(root,result):
    root=Path(root);(root/"reports").mkdir(parents=True,exist_ok=True)
    (root/"tables").mkdir(parents=True,exist_ok=True)
    rows=pd.DataFrame(result["model_metrics"])
    rows.to_csv(root/"tables"/"all_model_metrics.csv",index=False)
    comparisons=result.get("comparisons",[])
    flat=[]
    for c in comparisons:
        z=c["result"]
        flat.append({
          "comparison":c["comparison"],"candidate":c["candidate"],
          "baseline":c["baseline"],"mean_fold_delta_r2":z["mean_fold_delta_r2"],
          "student_pooled_delta_r2":z["mean_student_pooled_delta_r2"],
          "school_pooled_delta_r2":z["school_pooled_delta_r2"],
          "corrected_t":z["corrected_t"],"fold_win_fraction":z["win_fraction"],
          "instrument_deltas":str(z["instrument_delta_r2"]),
          "submission_eligible":c["submission_eligible"],
          "science_gate":c["science_gate"]})
    pd.DataFrame(flat).to_csv(root/"tables"/"paired_comparisons.csv",index=False)
    score_plots(root,result["model_metrics"])
    header=[
        "# WildSpeech M8.6 — M0++ Decision and Research Results","",
        "**Mode:** "+result["mode"],
        "**Engineering decision:** "+result["decision"],
        "**Winner:** "+str(result.get("winner")),
        "**No online leaderboard submission has been made by this code.**","",
        "## Frozen evaluation anchors","",
        "- M8.4 fold SHA-256: "+result["fold_hash"],
        "- M8.4 reference OOF SHA-256: "+result["reference_oof_sha256"],
        "- M8.5 E04 OOF SHA-256: "+result["e04_oof_sha256"],
        "- Student-level repeated folds: "+str(result["n_student_folds"]),
        "- Held-out-school folds: "+str(result["n_school_folds"]),"",
        "The frozen reference and E04 OOF files were **read**, never rewritten.",
        "Outer validation outcomes are NEVER supplied to feature builders or inner tuning.","",
        "## All models","",
        rows.to_markdown(index=False,floatfmt=".5f") if len(rows) else "No models completed.",
        "","## Scientific evidence interpretation","",
        "Submission gate: pooled student ΔR² >0; at least 65% student fold wins; "
        "school ΔR² >= -0.005; every instrument ΔR² >= -0.010.",
        "Scientific-effect gate: mean-fold ΔR² >= +0.010, corrected t >=2, "
        "fold wins >=60%, and same school/instrument vetoes.",
        "The corrected t is an approximate repeated-CV statistic and is not adjusted "
        "for all exploratory model comparisons. Model selection on the same folds "
        "makes formal effect claims conditional rather than independent confirmation.","",
        "## Paired comparisons","",
        pd.DataFrame(flat).to_markdown(index=False,floatfmt=".5f") if flat else "No full paired comparisons.",
        "","## Compliance and limitation notes","",
        "- No student/school identity, train-only score_status, raw audio, or future reading outcome is a predictor.",
        "- Item difficulty, Rasch item parameters, time-normalization references and spline knots are trained inside folds.",
        "- Training Rasch representations are student-crossfitted; item summary priors are student-LOO.",
        "- 'Duration' here means total clip duration, **not** prompt-end-to-child-response latency.",
        "- Phoneme count and true lexical frequency are NOT asserted; item complexity here uses transparent orthographic proxies.",
        "- Sentence-repeat partial credit and child speech localization belong to M8.7.",
        "- Existing M8.5 HOLD_REFERENCE is not retroactively changed by an M8.6 diagnostic submission.",
        "","## Decision","",result["decision_explanation"],"",
        "## Outputs","",
        "- Private Parquet OOF and fold checkpoints under runs/",
        "- Figures 01–05 and private tables under figures/ and tables/",
        "- Candidate submission (only when eligible) under submission/",
        "- No raw competition rows or model state are committed to GitHub.",""]
    atomic_text(root/"reports"/"M86_DECISION.md","\n".join(header)+"\n")
    atomic_json(root/"reports"/"M86_RESULTS.json",result)
    return root/"reports"/"M86_DECISION.md"
