"""Exact DrivenData metadata reference features and RidgeCV model.

Ref: https://blog.drivendata.org/blog/literacy-speech-modeling-reference
No learned preprocessing is fit before outer CV, and the test matrix is
constructed row-wise without using any test-set statistics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset, TASKS, sha_json

THETAS = [f"{task}_theta" for task in TASKS]
STATS = ("pct_correct", "n_clips", "mean_duration")
REFERENCE_COLUMNS = THETAS + [f"{task}_{stat}" for stat in STATS for task in TASKS] + ["is_wj4", "is_11khz"]


def metadata_features(clips: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
    """Reproduce DrivenData's metadata_features(), preserving student index order."""
    x = clips.copy()
    if "item_score" not in x:  # Test data has no outcomes and may lack item_score
        x["item_score"] = np.nan
    per_task = (
        x.groupby(["subject_id", "task"])
        .agg(pct_correct=("item_score", "mean"),
             n_clips=("filename", "size"),
             mean_duration=("duration_sec", "mean"))
        .unstack("task")
    )
    per_task.columns = [f"{task}_{stat}" for stat, task in per_task.columns]
    result = students[THETAS].join(per_task)
    for col in REFERENCE_COLUMNS:
        if col not in result and col not in ("is_wj4", "is_11khz"):
            result[col] = np.nan
    for task in TASKS:
        result[f"{task}_n_clips"] = result[f"{task}_n_clips"].fillna(0)
    result["is_wj4"] = (students.instrument == "wj4").astype(int)
    result["is_11khz"] = (students.original_sample_rate == 11025).astype(int)
    result = result.reindex(columns=REFERENCE_COLUMNS)
    assert result.index.equals(students.index)
    assert list(result.columns) == REFERENCE_COLUMNS and len(REFERENCE_COLUMNS) == 14
    return result


def reference_frames(data: Dataset) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    x = metadata_features(data.train_clips, data.train_students)
    xt = metadata_features(data.test_clips, data.test_students)
    y = data.labels.reindex(x.index).to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Target ID alignment failure")
    return x, xt, y


def make_reference():
    """Exactly the reference model (RidgeCV default internal cross-validation)."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 5, 50)),
    )


def reference_feature_hash(frame: pd.DataFrame) -> str:
    x = frame[REFERENCE_COLUMNS]
    return sha_json({"columns": REFERENCE_COLUMNS, "subject_order": x.index.astype(str).tolist(),
                    "values": [[None if pd.isna(v) else float(v) for v in row]
                               for row in x.to_numpy(dtype=float)]})
