"""End-to-end synthetic Phase 0 fixture, including checkpoint reuse.

This does NOT test official reference parity or claim any leaderboard score.
"""
import numpy as np
import pandas as pd
from wildspeech_m84.runner import run_phase0
from wildspeech_m84.data import digest_bytes


def _dataset(root, n=45):
    rng=np.random.default_rng(123)
    ids=[f"id_{i:04d}" for i in range(n)]
    test=[f"test_{i:04d}" for i in range(12)]
    students=pd.DataFrame({
        "subject_id":ids,
        "instrument":["wj4" if i%2 else "ktea3" for i in range(n)],
        "original_sample_rate":[11025 if i%3==0 else 16000 for i in range(n)],
        "school_id":[f"s{i%9}" for i in range(n)]})
    test_students=pd.DataFrame({
        "subject_id":test,
        "instrument":["wj4" if i%2 else "ktea3" for i in range(12)],
        "original_sample_rate":[11025 if i%3==0 else 16000 for i in range(12)]})
    for task in ("blending","deletion","sentence_repetition"):
        students[task+"_theta"]=rng.normal(size=n)
        test_students[task+"_theta"]=rng.normal(size=12)
    target=99+3*students.blending_theta+rng.normal(0,8,size=n)
    students.to_csv(root/"train_student_metadata.csv",index=False)
    test_students.to_csv(root/"test_student_metadata.csv",index=False)
    pd.DataFrame({"subject_id":ids[::-1],"outcome_score":target.to_numpy()[::-1]}).to_csv(
        root/"train_labels.csv",index=False)
    pd.DataFrame({"subject_id":test,"outcome_score":np.zeros(12)}).to_csv(
        root/"submission_format.csv",index=False)
    for split,population in (("train",ids),("test",test)):
        clips=[]
        for sid in population:
            for task in ("blending","deletion","sentence_repetition"):
                for j in range(2):
                    row={"subject_id":sid,"filename":f"{sid}_{task}_{j}.flac",
                         "task":task,"duration_sec":1+j,"expected_text":"apple"}
                    if split=="train":row["item_score"]=float(j%2)
                    clips.append(row)
        pd.DataFrame(clips).to_csv(root/f"{split}_clip_metadata.csv",index=False)


def test_phase0_end_to_end_with_resumable_locked_folds(tmp_path):
    dataset=tmp_path/"WildSpeech";dataset.mkdir()
    _dataset(dataset)
    r1=run_phase0(tmp_path,require_parity=False)
    assert r1["phase0_gate"] in ("NO_GO","CONDITIONAL_GO")
    assert (tmp_path/"WildSpeech_M84"/"phase0"/"folds"/"folds.parquet").is_file()
    assert (tmp_path/"WildSpeech_M84"/"phase0"/"parity"/"M84_BASELINE_PARITY.md").is_file()
    assert (tmp_path/"WildSpeech_M84"/"phase0"/"runs"/r1["reference_run_id"]/"oof_predictions_PRIVATE.parquet").is_file()
    folder=tmp_path/"WildSpeech_M84"/"phase0"/"runs"/r1["reference_run_id"]
    instrument_path=folder/"instrument_metrics.parquet"
    oof_path=folder/"oof_predictions_PRIVATE.parquet"
    assert instrument_path.is_file()
    original=pd.read_parquet(instrument_path)
    assert len(original)>=20 and {"scheme","instrument","r2","n","sse"}.issubset(original.columns)
    unchanged_oof_sha=digest_bytes(oof_path)
    # Simulate an already-completed run saved by the earlier broken reporter.
    instrument_path.unlink()
    assert not instrument_path.exists()
    r2=run_phase0(tmp_path,require_parity=False)
    assert instrument_path.is_file()
    repaired=pd.read_parquet(instrument_path)
    pd.testing.assert_frame_equal(original,repaired)
    assert digest_bytes(oof_path)==unchanged_oof_sha
    assert r2["reference_run_id"]==r1["reference_run_id"]
    assert r2["fold_hash"]==r1["fold_hash"]
