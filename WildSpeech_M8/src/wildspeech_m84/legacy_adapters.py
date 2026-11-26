"""Historic M8 diagnostic adapters; NOT a compliant new scientific audio ablation.

The M8.3 feature family naming is impure; source feature-bank generation may
inspect test values when deciding whether to create a deterministic transform.
These comparisons are therefore HISTORICAL_RESEARCH_ONLY and cannot enter
the official M8.4 promotion gate without a separate train-only feature audit.

The true M7.3 source is absent from main. Never invent an M7.3 parity score.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def locate_m7_profile(drive_root: Path, explicit: str="") -> Path | None:
    if explicit:
        path=Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root=Path(drive_root)
    for p in [root/"WildSpeech_M71_PHENOTYPE_BOOSTING"/"tables"/"student_profiles_all_covariates.csv",
              root/"WildSpeech_M7_DEEP_EDA"/"tables"/"student_profiles_all_covariates.csv"]:
        if p.is_file():
            return p
    return None


def load_m83_legacy_bank(data_root:Path,output_root:Path,profile:Path|None):
    if profile is None:
        raise FileNotFoundError("M7 student profile unavailable; cannot reproduce M8.3")
    from wildspeech_m8.m8_pipeline import Config,prepare_dataset
    from wildspeech_m8.m83_bottleneck import safe_columns
    conf=Config(data_root=str(data_root),output_root=str(output_root),
                profile_csv=str(profile),quick=False,max_features=450)
    frame,test,y,school,_,_,manifest=prepare_dataset(conf)
    columns,families=safe_columns(frame,450)
    if len(frame)!=len(y):
        raise ValueError("Legacy feature-bank alignment failed")
    return frame[columns],test[columns],np.asarray(y,float),np.asarray(school,dtype=str),families,manifest


def fit_predict_m83(frame,y,train,valid,cols,seed):
    from wildspeech_m8.m83_bottleneck import fit_predict
    p,_=fit_predict(frame,y,np.asarray(train,int),np.asarray(valid,int),cols,seed)
    return p


def documented_legacy_status():
    return {
        "M73_GLOBAL":"NOT_REPRODUCED: exact M7.3 implementation absent from repository main",
        "M83_ALL":"RESEARCH_ONLY: M8.3 legacy feature assembly/taxonomy requires compliance audit",
        "M83_WITHOUT_AUDIO_NAMING":"RESEARCH_ONLY: audio-name family also contains correctness variables",
    }
