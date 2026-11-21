"""CSV schema discovery/normalisation and the DataBundle used by every stage.

Observed schema (validated at runtime, aliases tolerated):
  clips   : filename, subject_id, task, expected_text, [score_status: train only], item_score, duration_sec
  students: subject_id, instrument, original_sample_rate, <task>_theta..., [school_id: train only]
  labels  : subject_id, outcome_score
Columns present only in train (score_status, school_id) are never used as model features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import FILE, ID, TARGET

ALIASES = {
    "filename": ["filename", "file", "file_name", "clip", "clip_id", "audio", "path"],
    "subject_id": ["subject_id", "student_id", "subject", "student", "participant_id"],
    "task": ["task", "task_name", "subtest"],
    "expected_text": ["expected_text", "prompt", "target_text", "item_text", "text"],
    "score_status": ["score_status", "status"],
    "item_score": ["item_score", "score", "correct", "is_correct"],
    "duration_sec": ["duration_sec", "duration", "duration_s", "length_sec"],
    "instrument": ["instrument", "assessment", "test_instrument"],
    "original_sample_rate": ["original_sample_rate", "sample_rate", "orig_sr", "sr"],
    "school_id": ["school_id", "school", "site_id"],
    "outcome_score": ["outcome_score", "target", "outcome", "score_eoy"],
}
CLIP_REQUIRED = ["filename", "subject_id", "task", "item_score"]
STUDENT_REQUIRED = ["subject_id"]


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def resolve_columns(df: pd.DataFrame, wanted: list[str], table: str) -> tuple[pd.DataFrame, list[dict]]:
    lower = {c.lower().strip(): c for c in df.columns}
    rename, report = {}, []
    for canon in wanted:
        hit = next((lower[a] for a in ALIASES.get(canon, [canon]) if a in lower), None)
        report.append({"table": table, "canonical": canon, "source_column": hit, "found": hit is not None})
        if hit is not None and hit != canon:
            rename[hit] = canon
    return df.rename(columns=rename), report


@dataclass
class DataBundle:
    train_clips: pd.DataFrame
    test_clips: pd.DataFrame
    train_students: pd.DataFrame
    test_students: pd.DataFrame
    labels: pd.DataFrame
    submission_format: pd.DataFrame
    theta_cols: list[str]
    tasks: list[str]
    schema_report: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def train(self) -> pd.DataFrame:
        """Student-level training frame (metadata + target)."""
        return self.train_students.merge(self.labels, on=ID, how="inner")

    @property
    def y(self) -> pd.Series:
        return self.train.set_index(ID)[TARGET].astype(float)

    @property
    def all_clips(self) -> pd.DataFrame:
        return pd.concat([self.train_clips, self.test_clips], ignore_index=True, sort=False)

    @property
    def all_students(self) -> pd.DataFrame:
        return pd.concat([self.train_students, self.test_students], ignore_index=True, sort=False)


def _prepare_clips(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df[FILE] = df[FILE].astype(str).map(lambda s: s.replace("\\", "/").split("/")[-1])
    df[ID] = df[ID].astype(str)
    df["task"] = df["task"].astype(str).str.strip().str.lower()
    if "expected_text" not in df:
        df["expected_text"] = ""
    df["expected_text"] = df["expected_text"].fillna("").astype(str)
    df["expected_text_norm"] = df["expected_text"].map(normalize_text)
    df["item_id"] = df["task"] + "::" + df["expected_text_norm"]
    df["item_score"] = pd.to_numeric(df["item_score"], errors="coerce")
    df["is_scored"] = df["item_score"].notna().astype(np.int8)
    if "duration_sec" in df:
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    else:
        df["duration_sec"] = np.nan
    df["split"] = split
    return df


def _prepare_students(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df[ID] = df[ID].astype(str)
    if "instrument" in df:
        df["instrument"] = df["instrument"].astype(str).str.strip().str.lower()
    if "original_sample_rate" in df:
        df["original_sample_rate"] = pd.to_numeric(df["original_sample_rate"], errors="coerce")
    if "school_id" in df:
        df["school_id"] = df["school_id"].astype(str)
    df["split"] = split
    return df


def _subsample_ids(ids: pd.Series, n: int | None, seed: int) -> set[str] | None:
    if not n or n >= len(ids):
        return None
    ids = ids.reset_index(drop=True).astype(str)
    order = pd.util.hash_pandas_object(ids + f"#{seed}", index=False).sort_values()
    return set(ids.loc[order.index[:n]])


def load_data_bundle(paths, cfg, logger=None) -> DataBundle:
    files = paths.files
    need = ["train_clips", "test_clips", "train_students", "test_students", "labels", "submission_format"]
    missing = [k for k in need if k not in files]
    if missing:
        raise FileNotFoundError(f"Missing competition CSVs {missing}; searched from {paths.root}. "
                                "Set paths.data_dir in configs/default.yaml.")
    report = []
    tc, r = resolve_columns(pd.read_csv(files["train_clips"]), list(ALIASES)[:7], "train_clips"); report += r
    vc, r = resolve_columns(pd.read_csv(files["test_clips"]), list(ALIASES)[:7], "test_clips"); report += r
    ts, r = resolve_columns(pd.read_csv(files["train_students"]), ["subject_id", "instrument", "original_sample_rate", "school_id"], "train_students"); report += r
    vs, r = resolve_columns(pd.read_csv(files["test_students"]), ["subject_id", "instrument", "original_sample_rate", "school_id"], "test_students"); report += r
    lb, r = resolve_columns(pd.read_csv(files["labels"]), ["subject_id", "outcome_score"], "labels"); report += r
    sf, r = resolve_columns(pd.read_csv(files["submission_format"]), ["subject_id", "outcome_score"], "submission_format"); report += r
    for name, df, req in [("train_clips", tc, CLIP_REQUIRED), ("test_clips", vc, CLIP_REQUIRED),
                          ("train_students", ts, STUDENT_REQUIRED), ("test_students", vs, STUDENT_REQUIRED),
                          ("labels", lb, [ID, TARGET]), ("submission_format", sf, [ID])]:
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise KeyError(f"{name}: required columns {miss} not resolvable; columns={list(df.columns)}")
    theta_cols = sorted(c for c in ts.columns if c.lower().endswith("_theta"))
    ts, vs = _prepare_students(ts, "train"), _prepare_students(vs, "test")
    for c in theta_cols:
        ts[c] = pd.to_numeric(ts[c], errors="coerce")
        if c in vs:
            vs[c] = pd.to_numeric(vs[c], errors="coerce")
    tc, vc = _prepare_clips(tc, "train"), _prepare_clips(vc, "test")
    lb = lb.copy()
    lb[ID] = lb[ID].astype(str)
    lb[TARGET] = pd.to_numeric(lb[TARGET], errors="coerce")
    sf = sf.copy()
    sf[ID] = sf[ID].astype(str)

    notes = []
    seed = int(cfg.get_path("data.subsample_seed", 7))
    keep_tr = _subsample_ids(ts[ID], cfg.get_path("data.max_students"), seed)
    keep_te = _subsample_ids(vs[ID], cfg.get_path("data.max_test_students"), seed)
    if keep_tr is not None:
        ts, tc, lb = ts[ts[ID].isin(keep_tr)], tc[tc[ID].isin(keep_tr)], lb[lb[ID].isin(keep_tr)]
        notes.append(f"train subsampled to {len(keep_tr)} students (profile)")
    if keep_te is not None:
        vs, vc, sf = vs[vs[ID].isin(keep_te)], vc[vc[ID].isin(keep_te)], sf[sf[ID].isin(keep_te)]
        notes.append(f"test subsampled to {len(keep_te)} students (profile) — submission disabled for subsamples")
    tasks = sorted(set(tc["task"]).union(vc["task"]))
    bundle = DataBundle(tc.reset_index(drop=True), vc.reset_index(drop=True), ts.reset_index(drop=True),
                        vs.reset_index(drop=True), lb.reset_index(drop=True), sf.reset_index(drop=True),
                        theta_cols, tasks, pd.DataFrame(report), notes)
    if logger:
        logger.info("data: train %d students / %d clips, test %d / %d, tasks=%s, thetas=%s",
                    len(ts), len(tc), len(vs), len(vc), tasks, theta_cols)
    return bundle


def feature_safe_student_columns(bundle: DataBundle) -> list[str]:
    """Student metadata columns available in BOTH splits (school_id is train-only and excluded)."""
    common = set(bundle.train_students.columns) & set(bundle.test_students.columns)
    return [c for c in bundle.train_students.columns if c in common and c not in (ID, "split")]
