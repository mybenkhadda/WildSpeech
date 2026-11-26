"""Official competition data contracts and evidence fingerprints.

Only reads six CSVs from a caller-supplied root. No audio processing, network
access, training on test rows, or automatic use of test statistics.
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FILES = (
    "train_student_metadata.csv", "test_student_metadata.csv",
    "train_labels.csv", "submission_format.csv",
    "train_clip_metadata.csv", "test_clip_metadata.csv",
)
TASKS = ("blending", "deletion", "sentence_repetition")


def digest_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str, ensure_ascii=False).encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def locate_data_root(drive_root: str | Path, override: str = "") -> Path:
    """Fail on missing/ambiguous official datasets, avoiding costly audio scans."""
    root = Path(drive_root).expanduser()
    if override:
        p = Path(override).expanduser()
        missing = [x for x in FILES if not (p / x).is_file()]
        if missing:
            raise FileNotFoundError(f"Official data incomplete in {p}: {missing}")
        return p.resolve()
    candidates = [root / x for x in (
        "WildSpeech", "WildSpeech/data", "WildSpeech/dataset",
        "WildSpeech/data/official", "WildSpeech_Data", "WildSpeech_M8/data")]
    found = {p.resolve() for p in candidates if all((p / x).is_file() for x in FILES)}
    if not found:
        queue = [(root, 0)]
        visited = 0
        while queue and visited < 350:
            folder, depth = queue.pop(0)
            visited += 1
            if not folder.is_dir():
                continue
            if all((folder / x).is_file() for x in FILES):
                found.add(folder.resolve())
                continue
            if depth >= 3:
                continue
            try:
                dirs = [x for x in folder.iterdir() if x.is_dir()]
            except OSError:
                continue
            queue += [(p, depth + 1) for p in dirs if depth == 0 or
                      any(k in p.name.lower() for k in
                          ("wild", "speech", "data", "dataset", "input"))]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one official dataset; found {sorted(map(str, found))}. "
                           "Provide DATA_ROOT_OVERRIDE.")
    return next(iter(found))


def _students(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, dtype={"subject_id": str})
    if "subject_id" not in x or x.subject_id.isna().any() or not x.subject_id.is_unique:
        raise ValueError(f"Invalid student IDs in {path}")
    return x.set_index("subject_id", drop=True)


@dataclass
class Dataset:
    root: Path
    train_students: pd.DataFrame
    test_students: pd.DataFrame
    labels: pd.Series
    train_clips: pd.DataFrame
    test_clips: pd.DataFrame
    submission_format: pd.DataFrame
    order_manifest: dict

    @property
    def ids(self) -> list[str]:
        return self.train_students.index.astype(str).tolist()

    @property
    def groups(self) -> np.ndarray:
        return self.train_students["school_id"].fillna("UNKNOWN").astype(str).to_numpy()

    @property
    def instruments(self) -> np.ndarray:
        return self.train_students["instrument"].fillna("UNKNOWN").astype(str).to_numpy()


def load_official(root: str | Path, output_dir: str | Path | None = None) -> Dataset:
    """Reference-compatible student row order comes from train_student_metadata.csv."""
    root = Path(root)
    missing = [p for p in FILES if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Official inputs missing: {missing}")
    tr = _students(root / FILES[0])
    te = _students(root / FILES[1])
    labs = _students(root / FILES[2])
    fmt = _students(root / FILES[3])
    for c in ("instrument", "original_sample_rate", "blending_theta",
              "deletion_theta", "sentence_repetition_theta"):
        if c not in tr:
            raise ValueError(f"Missing official reference field: {c}")
    if "school_id" not in tr:
        raise ValueError("school_id required for school-disjoint CV")
    if "outcome_score" not in labs or "outcome_score" not in fmt:
        raise ValueError("Missing target or submission column")
    if set(tr.index) != set(labs.index):
        raise ValueError("Training student IDs do not match label IDs")
    if set(te.index) != set(fmt.index):
        raise ValueError("Test student IDs do not match submission format")
    if set(tr.index) & set(te.index):
        raise ValueError("Train/test student identity overlap")
    target = pd.to_numeric(labs.reindex(tr.index)["outcome_score"], errors="coerce")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Labels must be finite")
    clips = {}
    for split, students in (("train", tr), ("test", te)):
        d = pd.read_csv(root / f"{split}_clip_metadata.csv",
                        dtype={"subject_id": str, "filename": str})
        needed = {"subject_id", "filename", "task", "duration_sec"}
        if split == "train":
            needed.add("item_score")
        if missing_cols := needed - set(d.columns):
            raise ValueError(f"{split} clips missing columns: {sorted(missing_cols)}")
        if d.subject_id.isna().any() or not set(d.subject_id).issubset(set(students.index)):
            raise ValueError(f"Invalid {split} clip/student join")
        if not set(d.task.dropna().astype(str)).issubset(set(TASKS)):
            raise ValueError(f"Unrecognized {split} task values")
        clips[split] = d
    if list(fmt.columns) != ["outcome_score"]:
        raise ValueError("Submission schema must be subject_id,outcome_score")
    order = {
        "canonical_source": "train_student_metadata.csv",
        "canonical_n": len(tr),
        "labels_original_order_equal": labs.index.equals(tr.index),
        "test_submission_order_equal": fmt.index.equals(te.index),
        "canonical_student_order_sha256": sha_json(tr.index.tolist()),
        "reference_order_rule": "metadata row order, labels.loc[metadata.index]",
    }
    d = Dataset(root=root, train_students=tr, test_students=te, labels=target,
                train_clips=clips["train"], test_clips=clips["test"],
                submission_format=fmt, order_manifest=order)
    if output_dir is not None:
        out = Path(output_dir)
        manifest = {
            "source": str(root), "files": {
                p: {"sha256": digest_bytes(root / p),
                    "size_bytes": (root / p).stat().st_size,
                    "columns": pd.read_csv(root / p, nrows=0).columns.tolist()} for p in FILES},
            "n_train": len(tr), "n_test": len(te),
            "n_train_clips": len(clips["train"]), "n_test_clips": len(clips["test"]),
            "n_school_groups": int(pd.Series(d.groups).nunique()),
            "order": order,
        }
        atomic_json(out / "manifests" / "data_manifest.json", manifest)
    return d
