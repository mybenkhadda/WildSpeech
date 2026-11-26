"""Synthetic full-pipeline M8.4 -> M8.5 SCREEN; NO official-data scores."""
import json
import numpy as np
import pandas as pd
from wildspeech_m84.runner import run_phase0
from wildspeech_m85.experiments import execute,CANDIDATES
from test_m84_runner import _dataset


def test_m85_screen_reuses_reference_without_refit_and_resumes(tmp_path):
    root=tmp_path/"WildSpeech"
    root.mkdir()
    _dataset(root,n=45)
    test_path=root/"test_clip_metadata.csv"
    clips=pd.read_csv(test_path)
    clips["item_score"]=[float(i%2) for i in range(len(clips))]
    clips.to_csv(test_path,index=False)
    phase0=run_phase0(tmp_path,require_parity=False)
    reference_id=phase0["reference_run_id"]
    ref_file=(tmp_path/"WildSpeech_M84"/"phase0"/"runs"/reference_id/
              "oof_predictions_PRIVATE.parquet")
    from wildspeech_m84.data import digest_bytes
    orig_ref_sha=digest_bytes(ref_file)
    args=dict(
        drive_root=tmp_path,mode="SCREEN",
        run_models=CANDIDATES,
        seed_count=1,iterations=12,
        expected_fold_hash=None,
        allow_synthetic_fixture=True,
        export_final=False,
    )
    first=execute(**args)
    assert first["decision"]=="SCREEN_EXPLORATORY"
    assert first["screen_results_not_confirmatory"]
    assert len(first["model_metrics"])==10
    assert all(v["n_folds"]==10 for v in first["model_metrics"])
    assert len(first["run_ids"])==9
    assert digest_bytes(ref_file)==orig_ref_sha
    result_dir=tmp_path/"WildSpeech_M85"/"m0plus"
    assert (result_dir/"reports"/"M85_M0PLUS_DECISION.md").is_file()
    assert (result_dir/"diagnostics"/"item_contract.json").is_file()
    assert (result_dir/"tables"/"paired_model_results.csv").is_file()
    assert not (result_dir/"submission"/"m0plus_submission.csv").exists()
    second=execute(**args)
    assert first["run_ids"]==second["run_ids"]
    assert digest_bytes(ref_file)==orig_ref_sha


def test_synthetic_fixture_override_cannot_be_used_for_full(tmp_path):
    import pytest
    with pytest.raises(ValueError,match="Synthetic fixture override"):
        execute(tmp_path,mode="FULL",expected_fold_hash=None,
                allow_synthetic_fixture=True,export_final=False)
