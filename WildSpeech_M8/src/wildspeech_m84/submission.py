"""Reproducible fit-on-train-only submission, format and inference invariance tests."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .data import digest_bytes
from .reference import make_reference


def predict_independent(model, test: pd.DataFrame, atol=1e-9):
    """Prove test-student independence under singleton and permuted batches."""
    n=len(test)
    if n<2:
        raise ValueError("Require two or more test rows for independence check")
    batch=np.asarray(model.predict(test),dtype=float).reshape(-1)
    if batch.shape!=(n,) or not np.isfinite(batch).all():
        raise ValueError("Nonfinite or misaligned batch predictions")
    # Check every test student, not only a handful of sampled rows.
    for i in range(n):
        one=float(model.predict(test.iloc[[i]])[0])
        if not np.isclose(one,batch[i],atol=atol,rtol=0):
            raise AssertionError(f"Test student {i} changes with batch composition")
    permutation=np.random.default_rng(42).permutation(n)
    scrambled=np.asarray(model.predict(test.iloc[permutation]),float)
    if not np.allclose(scrambled,batch[permutation],atol=atol,rtol=0):
        raise AssertionError("Test batch ordering changes predictions")
    return batch


def build_submission(x_train, y, x_test, submission_format:pd.DataFrame,
                     output_dir:Path,model_factory=make_reference,atol=1e-9):
    """Fit model exactly once on labeled train; no test-set fitted operations."""
    from joblib import dump
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    if not x_test.index.equals(submission_format.index):
        raise ValueError("Test student ordering does not match submission format")
    if not x_train.columns.equals(x_test.columns):
        raise ValueError("Train/test feature mismatch")
    model=model_factory()
    model.fit(x_train,np.asarray(y,dtype=float))
    pred=predict_independent(model,x_test,atol=atol)
    if len(pred)!=len(submission_format):
        raise ValueError("Wrong test prediction count")
    df=pd.DataFrame({"subject_id":submission_format.index.astype(str),
                     "outcome_score":pred})
    if list(df.columns)!=["subject_id","outcome_score"] or df.subject_id.duplicated().any():
        raise ValueError("Submission schema/IDs invalid")
    if df.subject_id.tolist()!=submission_format.index.astype(str).tolist():
        raise ValueError("Submission order changed")
    if not np.isfinite(df.outcome_score.to_numpy()).all():
        raise ValueError("Submission contains NaN/inf")
    csv=output_dir/"reference_submission.csv"
    tmp=csv.with_name("reference_submission.tmp.csv")
    df.to_csv(tmp,index=False,float_format="%.12f",lineterminator="\n")
    tmp.replace(csv)
    model_path=output_dir/"reference_ridge14.joblib"
    dump(model,model_path,compress=3)
    sha=digest_bytes(csv)
    (output_dir/"reference_submission.sha256").write_text(sha+"\n",encoding="utf-8")
    return {"submission_path":str(csv),"model_path":str(model_path),
            "sha256":sha,"n_rows":len(df),
            "train_only_fit":True,"singleton_batch_test":True,
            "permutation_invariance_test":True,
            "not_submitted_online":True}
