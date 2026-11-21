"""Data integrity checks (P00). Each check -> one row: check, status (PASS/WARN/FAIL), value, detail."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID, TARGET
from .schema import DataBundle


def _row(check: str, ok: bool, value, detail: str, warn_only: bool = False) -> dict:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    return {"check": check, "status": status, "value": value, "detail": detail}


def integrity_checks(b: DataBundle, manifests: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    rows = []
    tr_ids, te_ids = set(b.train_students[ID]), set(b.test_students[ID])
    rows.append(_row("train/test students disjoint", not (tr_ids & te_ids), len(tr_ids & te_ids), "overlap count"))
    rows.append(_row("unique train student ids", b.train_students[ID].is_unique, int(b.train_students[ID].duplicated().sum()), "duplicates"))
    rows.append(_row("unique test student ids", b.test_students[ID].is_unique, int(b.test_students[ID].duplicated().sum()), "duplicates"))
    lab = set(b.labels[ID])
    rows.append(_row("every train student has a label", tr_ids <= lab, len(tr_ids - lab), "missing labels"))
    rows.append(_row("labels only for train students", lab <= tr_ids, len(lab - tr_ids), "extra labels", warn_only=True))
    rows.append(_row("target finite", bool(np.isfinite(b.labels[TARGET]).all()), int((~np.isfinite(b.labels[TARGET])).sum()), "non-finite"))
    for split, clips, ids in [("train", b.train_clips, tr_ids), ("test", b.test_clips, te_ids)]:
        orphan = set(clips[ID]) - ids
        rows.append(_row(f"{split} clips map to known students", not orphan, len(orphan), "orphan subject ids"))
        no_clip = ids - set(clips[ID])
        rows.append(_row(f"{split} students with >=1 clip", not no_clip, len(no_clip), "students without clips", warn_only=True))
        rows.append(_row(f"{split} clip filenames unique", clips[FILE].is_unique, int(clips[FILE].duplicated().sum()), "duplicates"))
        dur = clips["duration_sec"]
        rows.append(_row(f"{split} durations positive", bool((dur.dropna() > 0).all()), int((dur <= 0).sum()), "non-positive", warn_only=True))
        sc = clips["item_score"].dropna()
        rows.append(_row(f"{split} item_score binary", bool(sc.isin([0, 1]).all()), sorted(sc.unique().tolist())[:6], "observed values", warn_only=True))
        if manifests and split in manifests and manifests[split] is not None:
            man = set(manifests[split]["filename"])
            miss = set(clips[FILE]) - man
            rows.append(_row(f"{split} clips present in TAR", not miss, len(miss), "metadata rows without audio", warn_only=True))
            extra = man - set(clips[FILE])
            rows.append(_row(f"{split} TAR has no unlisted audio", not extra, len(extra), "audio without metadata", warn_only=True))
            unsafe = int((~manifests[split]["safe"]).sum())
            rows.append(_row(f"{split} TAR member paths safe", unsafe == 0, unsafe, "unsafe member names (ignored)"))
    sub = set(b.submission_format[ID])
    rows.append(_row("submission ids == test students", sub == te_ids, len(sub ^ te_ids), "symmetric difference"))
    rows.append(_row("school_id absent from test", "school_id" not in b.test_students.columns, None,
                     "school_id is train-only: used for grouping/robustness, never as a feature", warn_only=True))
    rows.append(_row("score_status absent from test", "score_status" not in b.test_clips.columns, None,
                     "behaviour features use item_score missingness instead (available in both splits)", warn_only=True))
    for c in b.theta_cols:
        rows.append(_row(f"{c} present in test", c in b.test_students.columns, None, "screening score", warn_only=False))
    return pd.DataFrame(rows)


def leakage_guard_ids(train_ids, valid_ids) -> None:
    inter = set(train_ids) & set(valid_ids)
    if inter:
        raise AssertionError(f"student leakage: {len(inter)} students in both train and validation")
