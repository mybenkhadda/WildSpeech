"""Strict per-student screening feature contracts. No test-set fitted statistics."""
from __future__ import annotations
import hashlib
import json
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

TASKS=("blending","deletion","sentence_repetition")
DISALLOWED=("school_id","outcome_score","score_status","subject_id","filename",
            "speaker_id","examiner_id","teacher_id","y_true","y_pred","residual")
REFERENCE_FIELDS=("instrument","original_sample_rate")+tuple(t+"_theta" for t in TASKS)
CLIP_FIELDS=("subject_id","task","expected_text","item_score")


def normalize_prompt(text):
    """NFC, casefold, collapse Unicode whitespace; never infer prompt chronology."""
    if text is None or pd.isna(text):
        return "__MISSING_PROMPT__"
    return " ".join(unicodedata.normalize("NFC",str(text)).casefold().split()) or "__MISSING_PROMPT__"


def prompt_identity(task,text):
    if str(task) not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return (str(task),normalize_prompt(text))


def validate_student_inputs(students:pd.DataFrame,clips:pd.DataFrame,require_item_score=True):
    if students.index.name!="subject_id" or students.index.has_duplicates or students.index.isna().any():
        raise ValueError("students must be indexed by unique, non-null subject_id")
    if any(c not in students for c in REFERENCE_FIELDS):
        raise ValueError("Missing reference student fields")
    needed=set(CLIP_FIELDS if require_item_score else CLIP_FIELDS[:-1])
    if not needed.issubset(clips.columns):
        raise ValueError(f"Missing clip fields: {sorted(needed-set(clips))}")
    if clips.subject_id.isna().any() or not set(clips.subject_id.astype(str)).issubset(set(students.index.astype(str))):
        raise ValueError("Clip/student ID mismatch")
    if not set(clips.task.dropna().astype(str)).issubset(set(TASKS)):
        raise ValueError("Unknown screening task")
    if require_item_score:
        values=pd.to_numeric(clips.item_score,errors="coerce")
        if (values.dropna()<0).any() or (values.dropna()>1).any():
            raise ValueError("item_score outside [0,1]; inspect scoring protocol")
    return True


def assert_predictive_columns(columns):
    forbidden=[str(c) for c in columns if
        any(token in str(c).lower() for token in DISALLOWED)]
    if forbidden:
        raise ValueError(f"Forbidden predictive features: {forbidden}")
    if len(columns)!=len(set(columns)):
        raise ValueError("Duplicate predictive feature names")
    return True


def assert_disjoint(train_students,other_students):
    a=set(train_students.index.astype(str)); b=set(other_students.index.astype(str))
    if a&b:
        raise ValueError("Train and validation/test student IDs overlap")


def source_hash(train_students,train_clips):
    """Raw screening source fingerprint, without touching hidden-test distributions."""
    fields=[c for c in ("subject_id","task","expected_text","item_score","duration_sec") if c in train_clips]
    payload={"train_ids":train_students.index.astype(str).tolist(),
             "train_meta":pd.util.hash_pandas_object(train_students, index=True).astype(str).tolist(),
             "clip_rows":pd.util.hash_pandas_object(train_clips[fields],index=False).astype(str).tolist()}
    return hashlib.sha256(json.dumps(payload,separators=(",",":")).encode()).hexdigest()


def item_contract(train_clips,test_clips):
    out={}
    for name,frame in (("train",train_clips),("test",test_clips)):
        out[name]={
            "n_clips":len(frame),
            "item_score_present":"item_score" in frame,
            "item_score_coverage":float(frame.item_score.notna().mean()) if "item_score" in frame else None,
            "expected_text_present":"expected_text" in frame,
            "expected_text_coverage":float(frame.expected_text.notna().mean()) if "expected_text" in frame else None,
            "task_values":sorted(frame.task.dropna().astype(str).unique().tolist()) if "task" in frame else [],
            "chronology_column_present":bool(set(frame)&{"item_order","item_sequence","start_time","timestamp"}),
        }
    out["enabled"]=["task+normalized_expected_text","fold_fit_item_success",
                    "student_LOO_item_success","unscored_fraction","prompt_coverage"]
    out["disabled"]={
        "furthest_item_reached":"No verified chronological item order in official schema",
        "last_three_errors":"CSV order/opaque filenames do not establish task chronology",
        "score_status":"Not an eligible hidden-test feature",
        "school_id":"Hidden-test school IDs absent and prohibited for prediction",
    }
    out["nfc_casefold_collisions_meaning"]=(
        "Whitespace/case-normalized prompt variants intentionally map to the same task-specific key; "
        "exact raw variants are audited separately.")
    return out
