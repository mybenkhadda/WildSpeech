"""Feature provenance, modeling role, confounder and duplication atlas.

Source and modeling role are independent: for example original sample rate is
a reference metadata feature AND an acoustic nuisance proxy.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .reference import REFERENCE_COLUMNS,THETAS

PROHIBITED=("subject_id","school_id","outcome_score","target_score","student_id",
            "teacher_id","examiner_id","split","y_true","y_pred","residual")


def identify_feature(feature: str):
    name=feature.lower()
    if name in PROHIBITED or any(z in name for z in ("outcome_score","school_id","teacher_id","examiner_id")):
        return dict(family="PROHIBITED",source_file="forbidden",source="identifier_or_target",
                    uses_item_score=False,uses_audio=False,nuisance=False,
                    modeling_block="excluded",child_segment="not_applicable")
    if name in REFERENCE_COLUMNS:
        if name in THETAS or name.endswith("_pct_correct"):
            role="screening_correctness"
        else:
            role="reference_metadata"
        return dict(family=role,source_file="official_metadata",
                    source="reference_exact",uses_item_score=name.endswith("_pct_correct"),
                    uses_audio=False,nuisance=name=="is_11khz" or name.endswith("_mean_duration"),
                    modeling_block="reference14",child_segment="not_applicable")
    if any(z in name for z in ("n_correct_items","item_score","accuracy","pct_correct",
                               "n_scored","theta","irt","correct_minus_wrong")):
        return dict(family="screening_correctness",source_file="m7_profile_or_m8_derived",
                    source="correctness_or_theta",uses_item_score=True,uses_audio="audio__" in name,
                    nuisance=False,modeling_block="screening",child_segment="unknown")
    if any(z in name for z in ("instrument","sample_rate","_n_clips","clip_count","n_clips")):
        return dict(family="reference_metadata",source_file="student_or_clip_metadata",
                    source="administrative_metadata",uses_item_score=False,
                    uses_audio=False,nuisance="sample_rate" in name or "clip" in name,
                    modeling_block="screening_protocol",child_segment="not_applicable")
    if any(z in name for z in ("phonetic","phoneme","transcript","asr","wer","cer")):
        return dict(family="speech_text",source_file="m7_profile_or_m8_speech",
                    source="speech_hypothesis",uses_item_score=False,uses_audio=True,
                    nuisance=False,modeling_block="phonetic_hypotheses",child_segment="unverified")
    if any(z in name for z in ("rms","spectral","mfcc","acoustic","energy","zcr","audio",
                               "pause","vad","speech_fraction")):
        nuisance=any(z in name for z in ("rms","noise","quality","silence","sample_rate"))
        return dict(family="acoustic",source_file="m7_profile_or_m8_audio",
                    source="audio_derived",uses_item_score=False,uses_audio=True,
                    nuisance=nuisance,modeling_block="acoustic_or_nuisance",
                    child_segment="unverified")
    if any(z in name for z in ("duration","latency","timing","onset","rate")):
        return dict(family="timing_or_protocol",source_file="clip_metadata_or_audio",
                    source="timing",uses_item_score=False,uses_audio="m8__audio" in name,
                    nuisance=True,modeling_block="timing",child_segment="unverified")
    return dict(family="context_behavior",source_file="profile_or_official_metadata",
                source="context",uses_item_score=False,uses_audio=False,nuisance=False,
                modeling_block="context",child_segment="not_applicable")


def build_registry(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if list(train.columns)!=list(test.columns):
        raise ValueError("Train/test feature columns differ")
    records=[]
    # Never use test values to fit transformations or select columns.
    for name in train.columns:
        record=identify_feature(str(name))
        if record["family"]=="PROHIBITED":
            raise ValueError(f"Forbidden predictive column: {name}")
        a=train[name]
        b=test[name]
        row=dict(feature_name=str(name),**record,
                 train_coverage=float(a.notna().mean()),
                 test_coverage_audit_only=float(b.notna().mean()),
                 requires_fold_fit=False,allowed_in_reference=name in REFERENCE_COLUMNS,
                 allowed_in_speech_arm=bool(record["uses_audio"] and not record["uses_item_score"]
                                             and record["child_segment"]=="verified"),
                 is_duplicate=False,canonical_of="",notes="")
        records.append(row)
    out=pd.DataFrame(records)
    # Training-only numeric value fingerprints flag duplicated measurements;
    # permutations or transforms are NOT declared exact duplicates.
    seen={}
    for i,name in enumerate(train.columns):
        s=train[name]
        hashed=tuple(pd.util.hash_pandas_object(s.reset_index(drop=True),index=False).tolist())
        if hashed in seen:
            out.at[i,"is_duplicate"]=True
            out.at[i,"canonical_of"]=seen[hashed]
        else:
            seen[hashed]=name
    return out


def registry_summary(df):
    return {
        "n_total":len(df),
        "n_duplicates":int(df.is_duplicate.sum()),
        "n_audio":int(df.uses_audio.sum()),
        "n_audio_child_verified":int(((df.uses_audio)&(df.child_segment=="verified")).sum()),
        "n_prohibited":int((df.family=="PROHIBITED").sum()),
        "family_counts":df.family.value_counts().to_dict()
    }
