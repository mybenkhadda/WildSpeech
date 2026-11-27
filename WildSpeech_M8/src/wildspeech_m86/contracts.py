"""Invariant source, fold, item and feature contracts for the M8.6 milestone."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from wildspeech_m84.data import sha_json, digest_bytes, atomic_json
from wildspeech_m85.contracts import TASKS, normalize_prompt, assert_predictive_columns

EXPERIMENTS=("B1","B2","B3","I1","I2","I3","I4","I5","D2","D3","T1")
ITEM_BLOCKS=("I1","I2","I3","I4","T1")
FULL_FOLD_HASH="6ee0f59c8cf5ea3d8f20f5fd500353419587660e00949ac6f1e6a16e092c9ce1"
REQUIRED_META=("instrument","original_sample_rate")+tuple(f"{task}_theta" for task in TASKS)
FORBIDDEN=("school_id","score_status","outcome_score","filename","subject_id","teacher_id","examiner_id")

def validate_inputs(students, clips, allow_missing_item_score=False):
    if students.index.name!="subject_id" or students.index.has_duplicates or students.index.isna().any():
        raise ValueError("Expected unique subject_id index")
    if not set(REQUIRED_META).issubset(students):
        raise ValueError("Student metadata lacks required reference columns")
    needed={"subject_id","task","expected_text","duration_sec"}
    if not allow_missing_item_score:
        needed.add("item_score")
    if not needed.issubset(clips):
        raise ValueError(f"Clip schema missing {sorted(needed-set(clips))}")
    if not set(clips.subject_id.astype(str)).issubset(set(students.index.astype(str))):
        raise ValueError("Clip/student join failure")
    if "item_score" in clips:
        score=pd.to_numeric(clips.item_score,errors="coerce")
        if ((score.dropna()<0)|(score.dropna()>1)).any():
            raise ValueError("Item score outside [0,1]")
    if not set(clips.task.dropna().astype(str)).issubset(set(TASKS)):
        raise ValueError("Unrecognized task")
    return True

def prompt_key(task, text):
    if str(task) not in TASKS:
        raise ValueError("Unknown screening task")
    return str(task)+"\x1f"+normalize_prompt(text)

def ensure_predictor_columns(df):
    assert_predictive_columns(df.columns)
    names=[str(s).lower() for s in df.columns]
    if any(any(pat in x for pat in FORBIDDEN) for x in names):
        raise ValueError("Prohibited predictor (target, ID, school or scoring status)")
    if df.columns.has_duplicates:
        raise ValueError("Duplicate predictive feature columns")
    return True

def compare_source_bytes(data, phase0_root):
    """Use M8.4 exact file hashes; do not infer dataset equivalence from shape."""
    root=Path(phase0_root)
    data_manifest=json.loads((root/"manifests"/"data_manifest.json").read_text())
    for file, info in data_manifest["files"].items():
        actual=digest_bytes(Path(data.root)/file)
        if actual!=info["sha256"]:
            raise RuntimeError(f"Official source changed since M8.4: {file}")
    return True

def validate_same_folds(df, ids, groups, expected_hash):
    from wildspeech_m84.folds import fold_digest,validate_folds
    validate_folds(df,ids,groups)
    actual=fold_digest(df,ids,groups)
    if actual!=expected_hash:
        raise RuntimeError(f"Frozen fold contract mismatch {actual} != {expected_hash}")
    return actual

def raw_train_fingerprint(students,clips):
    cols=[c for c in ("subject_id","task","expected_text","item_score","duration_sec") if c in clips]
    return sha_json({
        "ids":students.index.astype(str).tolist(),
        "student_hash":pd.util.hash_pandas_object(students,index=True).astype(str).tolist(),
        "clip_hash":pd.util.hash_pandas_object(clips[cols],index=False).astype(str).tolist()
    })
