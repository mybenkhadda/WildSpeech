"""V4 end-to-end on the synthetic project (CPU, minutes): external ASR export (long layout, two recognizers, one
with phone strings, one failed row, one duplicate) -> PREPARE -> FEATURES (no acoustic extraction) -> EXPERIMENTS
-> FIGURES -> SUBMISSION -> REPORT. Checks artefact paths, provenance, joins, fallback labelling and submission
validity. TEST DATA ONLY: nothing here is an empirical result."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE, ID

pytestmark = pytest.mark.integration


def _export(project, path):
    from wildspeech.phonetic.g2p import G2P

    data = project / "data"
    g2p = G2P("rule")
    rows = []
    for split in ("train", "test"):
        c = pd.read_csv(data / f"{split}_clip_metadata.csv")
        for i, r in enumerate(c.itertuples(index=False)):
            ok = (r.item_score == 1) if not pd.isna(r.item_score) else (i % 2 == 0)
            hyp = str(r.expected_text) if ok else str(r.expected_text)[:-1]
            rows.append({"model": "fake_whisper", "audio_name": f"{split}_audio/{r.filename}", "split_hint": split,
                         "text": hyp, "phones": "", "status": "ok", "avg_logprob": -0.1 if ok else -1.2,
                         "inference_ms": 50.0})
            ph = " ".join(g2p.phonemes(hyp))
            rows.append({"model": "fake_ctc", "audio_name": r.filename.upper(), "split_hint": split,
                         "text": hyp.lower() if i % 3 else hyp.lower() + " um", "phones": ph, "status": "ok",
                         "avg_logprob": np.nan, "inference_ms": 20.0})
    rows.append({**rows[0], "status": "ok"})                                  # duplicate model x clip
    rows[3] = {**rows[3], "text": None, "phones": None, "status": "error"}    # failed inference
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_v4_pipeline_end_to_end(project, tmp_path):
    from wildspeech.v4.pipeline import V4Pipeline

    asr_root = tmp_path / "asr"
    asr_root.mkdir()
    _export(project, asr_root / "all_models_predictions.parquet")
    cfg = {"data_root": str(project / "data"), "asr_root": str(asr_root), "output_root": str(tmp_path / "V4"),
           "local_work_root": str(tmp_path / "work"), "run_mode": "FULL", "cv_folds": 3, "cv_repeats": 1, "seed": 7,
           "use_gpu": False, "compute_missing_acoustic_features": False, "asr_drive_file_id": None}
    over = {"asr_import": {"permission_review": {"reviewed": True, "reviewer": "test", "date": "2026-11-21",
                                                 "statement": "synthetic fixture"}},
            "phonetic": {"allow_approximate_g2p": True, "reference_g2p_backend": "rule"},
            "baseline": {"model": "RIDGE", "families": ["SCREENING", "BEHAVIOR", "ITEM"]},
            "baseline_models": ["RIDGE", "GAM"], "models_early": ["RIDGE"], "budget": 5,
            "validation": {"bootstrap": 30, "robustness_schemes": ["school_kfold"], "budgets": [5],
                           "importance_permutations": 2},
            "submission": {"min_test_coverage": 0.5}}
    pipe = V4Pipeline(cfg, project_root=project, v4_overrides=over)
    out = pipe.run()
    root = tmp_path / "V4"
    for rel in ("manifests/clip_master.parquet", "manifests/student_master.parquet", "manifests/asr_model_coverage.csv",
                "manifests/join_audit.csv", "manifests/data_integrity.json", "asr_import/asr_import_report.json",
                "asr_import/asr_schema_report.json", "asr_import/asr_coverage.csv", "asr_import/duplicates.parquet",
                "asr_import/failed_inference.parquet", "student_features/feature_catalog.parquet",
                "cv/oof_predictions.parquet", "cv/cv_results.parquet", "ablations/ablation_results.csv",
                "robustness/robustness_results.csv", "diagnostics/fallback_report.csv", "models/feature_importance.parquet",
                "model_registry/champion_spec.json", "figures/figure_index.csv",
                "reports/WILDSPEECH_V4_RESEARCH_REPORT.md", "reports/hypotheses.csv"):
        assert (root / rel).exists(), rel
    rep = json.loads((root / "asr_import/asr_import_report.json").read_text())
    assert rep["provenance_type"] == "external_colab_asr" and len(rep["source_sha256"]) == 64
    assert set(rep["usable_models"]) == {"fake_whisper", "fake_ctc"}
    assert rep["n_duplicates"] == 1 and rep["n_failed"] >= 1
    prov = json.loads(next((root / "cache/v4/asr_external").rglob("_provenance.json")).read_text())
    assert prov["producer"] == "external_colab_asr"                         # never relabelled P24
    cm = pd.read_parquet(root / "manifests/clip_master.parquet")
    assert cm.groupby("split")[FILE].apply(lambda s: s.is_unique).all()
    cat = pd.read_parquet(root / "student_features/feature_catalog.parquet")
    assert {"ASRTXT", "ASRPHN"} <= set(cat["provider"])
    assert not cat["feature"].dropna().str.contains("school|score_status", case=False).any()
    res = pd.read_parquet(root / "cv/cv_results.parquet")
    assert {"M0_RIDGE", "M1_TEXT_RIDGE", "M5_RESID_TEXT", "CTRL_SHUFFLED_ASR"} <= set(res["name"])
    assert res.loc[res["status"] == "fallback_baseline", "result_class"].isin(["BASELINE_FALLBACK"]).all()
    champ = json.loads((root / "model_registry/champion_spec.json").read_text())
    crow = res[(res["name"] == champ["name"]) & (res["scheme"] == "repeated_kfold")]
    assert champ["is_baseline"] or (crow["result_class"].iloc[0] == "GENUINE_GAIN")
    sub = pd.read_csv(root / "submission/submission.csv")
    fmt = pd.read_csv(project / "data/submission_format.csv")
    assert list(sub.columns) == list(fmt.columns) and list(sub[ID].astype(str)) == list(fmt[ID].astype(str))
    assert np.isfinite(sub["outcome_score"]).all()
    val = pd.read_csv(root / "submission/submission_validation.csv")
    assert (val["status"] == "PASS").all()
    assert out["submission"]["rows"] == len(fmt)


def test_v4_resume_reuses_text_features(project, tmp_path):
    from wildspeech.v4.pipeline import V4Pipeline

    asr_root = tmp_path / "asr"
    asr_root.mkdir()
    _export(project, asr_root / "all_models_predictions.parquet")
    cfg = {"data_root": str(project / "data"), "asr_root": str(asr_root), "output_root": str(tmp_path / "V4"),
           "local_work_root": str(tmp_path / "work"), "run_mode": "FEATURES", "cv_folds": 3, "cv_repeats": 1,
           "use_gpu": False, "compute_missing_acoustic_features": False, "asr_drive_file_id": None}
    over = {"asr_import": {"permission_review": {"reviewed": True, "reviewer": "t", "date": "2026-11-21", "statement": "s"}}}
    V4Pipeline(cfg, project_root=project, v4_overrides=over).run()
    meta = tmp_path / "V4/clip_features/clip_text_long.meta.json"
    m1 = json.loads(meta.read_text())
    V4Pipeline(cfg, project_root=project, v4_overrides=over).run("FEATURES")
    m2 = json.loads(meta.read_text())
    assert m1["fingerprint"] == m2["fingerprint"] and m1["created"] == m2["created"]   # reused, not recomputed


def test_v4_without_permission_review_runs_no_asr_family(project, tmp_path):
    from wildspeech.v4.pipeline import V4Pipeline

    asr_root = tmp_path / "asr"
    asr_root.mkdir()
    _export(project, asr_root / "all_models_predictions.parquet")
    cfg = {"data_root": str(project / "data"), "asr_root": str(asr_root), "output_root": str(tmp_path / "V4"),
           "local_work_root": str(tmp_path / "work"), "run_mode": "FEATURES", "use_gpu": False,
           "compute_missing_acoustic_features": False, "asr_drive_file_id": None}
    pipe = V4Pipeline(cfg, project_root=project)
    pipe.run()
    rep = json.loads((tmp_path / "V4/asr_import/asr_import_report.json").read_text())
    assert rep["usable_models"] == []
    assert not any(n.startswith("ASR") for n in pipe.providers()["providers"])
