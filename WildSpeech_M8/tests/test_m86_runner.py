"""Full synthetic M8.4->M8.5 E04->M8.6 pipeline. No official-data scores."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from test_m84_runner import _dataset
from wildspeech_m84.runner import run_phase0
from wildspeech_m84.data import digest_bytes
from wildspeech_m85.experiments import execute as execute_m85
from wildspeech_m86.experiments import execute as execute_m86,_decisions

def test_engineering_and_science_are_different_predeclared_rules():
    c={"instrument_delta_r2":{"KTEA-3":.001,"WJ IV":.002},
       "school_pooled_delta_r2":.001,
       "mean_student_pooled_delta_r2":.005,
       "mean_fold_delta_r2":.012,"corrected_t":1.37,
       "win_fraction":.88}
    assert _decisions(c)==(True,False)
    assert _decisions({**c,"instrument_delta_r2":{"KTEA-3":-.02,"WJ IV":.001}})==(False,False)
    assert _decisions({**c,"corrected_t":2.1})==(True,True)

def test_phase0_m85_e04_to_m86_screen_and_cache(tmp_path):
    data_root=tmp_path/"WildSpeech";data_root.mkdir()
    _dataset(data_root,n=45)
    clips=pd.read_csv(data_root/"test_clip_metadata.csv")
    clips["item_score"]=[float(j%2) for j in range(len(clips))]
    clips.to_csv(data_root/"test_clip_metadata.csv",index=False)
    m84=run_phase0(tmp_path,require_parity=False)
    m85=execute_m85(drive_root=tmp_path,mode="SCREEN",
                    run_models=("E04",),seed_count=1,iterations=8,
                    expected_fold_hash=None,allow_synthetic_fixture=True,
                    export_final=False)
    e04file=(tmp_path/"WildSpeech_M85"/"m0plus"/"runs"/
             m85["run_ids"]["E04"]/"oof_predictions_PRIVATE.parquet")
    refsha=digest_bytes(tmp_path/"WildSpeech_M84"/"phase0"/"runs"/
                         m84["reference_run_id"]/"oof_predictions_PRIVATE.parquet")
    e04sha=digest_bytes(e04file)
    args=dict(drive_root=tmp_path,mode="SCREEN",models=("B3",),
              expected_fold_hash=None,allow_synthetic_fixture=True,
              export_candidate=False,cat_iterations=8,cat_seed_count=1)
    out=execute_m86(**args)
    assert out["decision"]=="SCREEN_EXPLORATORY" and len(out["run_ids"])==1
    assert out["n_student_folds"]==5 and out["n_school_folds"]==5
    assert out["e04_oof_sha256"]==e04sha
    assert digest_bytes(e04file)==e04sha
    assert digest_bytes(tmp_path/"WildSpeech_M84"/"phase0"/"runs"/
                        m84["reference_run_id"]/"oof_predictions_PRIVATE.parquet")==refsha
    assert (tmp_path/"WildSpeech_M86"/"m0pp"/"screen"/"reports"/"M86_DECISION.md").exists()
    assert (tmp_path/"WildSpeech_M86"/"m0pp"/"screen"/"tables"/"all_model_metrics.csv").exists()
    cached=execute_m86(**args)
    assert cached["run_ids"]==out["run_ids"]
    with pytest.raises(ValueError,match="Synthetic fixture"):
        execute_m86(drive_root=tmp_path,mode="FULL",expected_fold_hash=None,
                    allow_synthetic_fixture=True,export_candidate=False)
