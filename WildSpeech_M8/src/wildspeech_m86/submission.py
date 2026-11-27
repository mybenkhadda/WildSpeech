"""Auditable M8.6 candidate CSV and separate E04 diagnostic exporter.

Both paths fit on all labeled training students only and assert full *raw*
student independence; neither submits to the leaderboard or modifies M8.5.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from wildspeech_m84.data import atomic_json,digest_bytes
from wildspeech_m85.submission import fit_and_export as m85_fit_and_export
from .nested_cv import fit_full_model
from .features import select_features,assert_raw_independence


def _write_csv(data,pred,folder,prefix):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    expected=data.submission_format.index.astype(str).tolist()
    if len(pred)!=len(expected) or not np.isfinite(pred).all():
        raise ValueError("Wrong/invalid test predictions")
    frame=pd.DataFrame({"subject_id":expected,"outcome_score":np.asarray(pred,float)})
    if frame.subject_id.duplicated().any() or frame.subject_id.tolist()!=expected:
        raise ValueError("Submission identity/order failure")
    final=folder/f"{prefix}_submission.csv"
    temp=folder/f"{prefix}_submission.tmp.csv"
    frame.to_csv(temp,index=False,float_format="%.12f",lineterminator="\n")
    temp.replace(final)
    sha=digest_bytes(final)
    (folder/f"{prefix}_submission.sha256").write_text(sha+"\n",encoding="utf-8")
    return str(final),sha


def export_m86_candidate(data,model_id,folder,iterations=600,
                         seeds=(42,43,44,45,46),
                         rasch_iters=7,rasch_crossfit=3):
    if "item_score" not in data.test_clips or not data.test_clips.item_score.notna().any():
        raise RuntimeError("M86 requires observed test item scores to deploy S2/S5")
    builder,model,settings=fit_full_model(
        data,model_id,iterations=iterations,seeds=seeds,
        rasch_iters=rasch_iters,rasch_crossfit=rasch_crossfit)
    test=data.test_students.reindex(data.submission_format.index)
    if test.index.has_duplicates or test.index.isna().any():
        raise ValueError("Test/submission identity mismatch")
    pred=assert_raw_independence(
        builder,model,lambda m,x:m.predict(x),
        test,data.test_clips,model_id,atol=1e-7)
    path,sha=_write_csv(data,pred,folder,"m0pp")
    out=Path(folder);dump({"builder":builder,"model":model,"model_id":model_id,
                           "settings":settings},out/"m0pp_model.joblib",compress=3)
    manifest={"kind":"M86_ENGINEERING_CANDIDATE","model_id":model_id,
              "csv":path,"sha256":sha,"model":str(out/"m0pp_model.joblib"),
              "n_rows":len(pred),"train_only_fit":True,
              "all_test_singleton_checked":True,"test_permutation_checked":True,
              "submitted_online":False,**settings}
    atomic_json(out/"m0pp_submission_manifest.json",manifest)
    return manifest


def export_e04_diagnostic(data,folder,phase0_hash,m85_runid,source_decision):
    """Never re-label a historical HOLD_REFERENCE as PROMOTED."""
    if source_decision not in ("HOLD_REFERENCE","PROMOTE"):
        raise ValueError("E04 diagnostic requires completed FULL M8.5 evidence")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    result=m85_fit_and_export(data,"E04",folder)
    manifest={
        "kind":"E04_DIAGNOSTIC_ONLY","m85_decision_unchanged":source_decision,
        "m85_run_id":m85_runid,"m84_fold_hash":phase0_hash,
        "submission":result,
        "not_formal_promotion":True,"submitted_online":False,
        "public_r2":None,"submission_receipt":None,
    }
    atomic_json(folder/"E04_DIAGNOSTIC_MANIFEST.json",manifest)
    return manifest
