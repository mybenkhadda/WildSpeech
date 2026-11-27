"""Full experiment scorecard with negative controls and honest evidence gates."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from wildspeech_m84.data import atomic_json,atomic_text

def write_report(root,result):
    root=Path(root);reports=root/"reports";reports.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(result["model_metrics"])
    c=pd.DataFrame([{
        "arm":x["candidate"],
        "mean_fold_delta_r2":x["result"]["mean_fold_delta_r2"],
        "student_pooled_delta_r2":x["result"]["mean_student_pooled_delta_r2"],
        "school_pooled_delta_r2":x["result"]["school_pooled_delta_r2"],
        "win_fraction":x["result"]["win_fraction"],
        "corrected_t":x["result"]["corrected_t"],
        "instrument_delta_r2":str(x["result"]["instrument_delta_r2"]),
        "engineering_eligible":x["engineering_eligible"],
        "science_gate":x["science_gate"],
    } for x in result["comparisons"]])
    path=root/"tables";path.mkdir(exist_ok=True)
    c.to_csv(path/"07_paired_science_gate.csv",index=False)
    lines=[
      "# WildSpeech M8.7 — Speech Evidence Decision","",
      "**Mode:** "+result["mode"],
      "**Decision:** "+result["decision"],
      "**Engineering winner:** "+str(result["winner"]),
      "**No model or audio features were fitted using hidden-test cases.**",
      "**No online submission was made.**","",
      "## Frozen anchors","",
      f"- M8.4 fold SHA-256: {result['fold_hash']}",
      f"- M8.4 reference OOF SHA-256: {result['reference_oof_sha256']}",
      f"- M8.6 I5 OOF SHA-256: {result['i5_oof_sha256']}",
      f"- M8.6 I4 OOF SHA-256: {result['i4_oof_sha256']}",
      f"- Student outer folds: {result['n_student_folds']}",
      f"- School-disjoint outer folds: {result['n_school_folds']}","",
      "## Extraction audit","",
      pd.DataFrame([result["clip_cache_coverage"]]).to_markdown(index=False),"",
      "## Model ranking","",
      table.to_markdown(index=False,floatfmt=".5f"),"",
      "## Paired comparisons versus read-only I5","",
      c.to_markdown(index=False,floatfmt=".5f") if len(c) else
       "SCREEN has no confirmatory corrected-t metrics.","",
      "## Missing/disabled experiments","",
      json.dumps(result["skipped"],indent=2),"",
      "## Research interpretations and limitations","",
      "- A00 whole-clip phone edit is a **new reimplementation**, not the exact DrivenData reference phonetic comparator.",
      "- A02/A03 target phone support is CTC evidence, NOT a calibrated probability of correct pronunciation.",
      "- VAD and a target-based segment selector cannot prove child identity. "
        "Human training-only localization QA is required before claiming child speech.",
      "- Incorrect/aligned target pronunciations can produce forced paths; the free-phone contrast is mandatory.",
      "- Pasketti-phonetic reports 33% PER on its own test data, NOT on voice-converted WildSpeech clips.",
      "- Word/phone confidence from Pasketti Word is NOT available through the verified API.",
      "- Whole-clip, segment-selected and random-segment controls can reveal prompt contamination.",
      "- CTC-aligned phone times and start fraction are only model-dependent timing proxies.",
      "- Supervised clip correctness is grouped by student. Speech feature item normalizers are training-only.",
      "- Test clips are processed independently; no target- or label-derived test adaptation.",
      "- Outer-fold model selection across many arms inflates the best observed CV score.",
      "- Corrected repeated-CV t is approximate and not multiple-testing adjusted.",
      "- A04/A05/A15 controls are intentionally ineligible for promotion.",
      "- Improvement over I5 is not sufficient alone; inspect I4 school R² and instrument robustness.",
      "- Voice conversion can distort timing, prosody and speaker identity; report perturbation QA.",
      "- Missing word ASR marks A07 NOT_ELIGIBLE, not an invented zero or model score.",
      "","## Decisions","",
      ("M8.7 has an eligible engineering experiment; independently verify "
       "training-only localization QA and full raw student inference before "
       "exporting a challenger." if result["winner"] else
       "No candidate passed all preregistered engineering gates. "
       "Preserve M8.6 I5 as experimental benchmark and M8.4 as approved fallback."),
      "","## Generated figures","",
      "- 01 cache coverage by task/split; 02 duration; 03 localization confidence",
      "- 04 whole vs segment edit; 05 CTC target differences",
      "- 06 within-correctness graded accuracy; 07 student/school R²",
      "- 08 paired deltas; 09 school-transfer frontier; 10 baseline score scatter",
      "- Optional training-only waveform QA and perturbation stability plots.",
      "","## Next step","",
      "Only speech families demonstrating incremental, school-robust signal "
      "proceed to M8.8 task-intelligence refinement and final cross-fitted fusion.",""]
    atomic_text(reports/"M87_DECISION.md","\n".join(lines)+"\n")
    atomic_json(reports/"M87_RESULTS.json",result)
    return reports/"M87_DECISION.md"
