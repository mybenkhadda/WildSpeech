"""Frozen reference CV and genuinely randomized, balanced school-disjoint folds."""
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from .data import sha_json, atomic_json


def school_group_assignment(groups, n_folds=5, seed=42):
    """Seed-sensitive greedy balanced bin packing of whole schools.

    Initial random rotation and randomized near-size priority allow real
    group-partition diversity; simply relabeling GroupKFold is not sufficient.
    """
    a = np.asarray(groups, dtype=str)
    unique, count = np.unique(a, return_counts=True)
    if len(unique) < n_folds:
        raise ValueError("At least one school per group fold is required")
    rng = np.random.default_rng(seed)
    # Jitter the ordering by 0-30% to vary assignments across repeats.
    priority = count.astype(float) * rng.uniform(.72, 1.3, len(count))
    order = np.argsort(-priority, kind="stable")
    loads = np.zeros(n_folds, dtype=int)
    bucket = {str(k): [] for k in range(n_folds)}
    # Ensure every fold has one school; then greedily balance student counts.
    for i, idx in enumerate(order):
        if i < n_folds:
            fold = i
        else:
            smallest = np.where(loads == loads.min())[0]
            fold = int(rng.choice(smallest))
        bucket[str(fold)].append(str(unique[idx]))
        loads[fold] += int(count[idx])
    return bucket


def _group_pattern(bucket):
    return tuple(sorted(tuple(sorted(v)) for v in bucket.values()))


def build_folds(ids, groups, n_splits=5, n_repeats=10, seed=42):
    """Generate precisely 100 fold definitions, each with private row positions."""
    ids = [str(v) for v in ids]
    groups = np.asarray(groups, dtype=str)
    n = len(ids)
    if len(set(ids)) != n or len(groups) != n:
        raise ValueError("Student IDs must be unique and align with school groups")
    if len(set(groups)) < n_splits:
        raise ValueError(f"Expected at least {n_splits} schools")
    folds = []
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for k, (tr, va) in enumerate(splitter.split(np.arange(n))):
        folds.append({"scheme": "student", "repeat": int(k // n_splits),
                      "fold": int(k % n_splits), "train_pos": tr.tolist(),
                      "valid_pos": va.tolist(),
                      "n_train": len(tr), "n_valid": len(va)})
    previously = set()
    for rep in range(n_repeats):
        for attempt in range(256):
            assignment = school_group_assignment(groups, n_splits, seed + rep * 1009 + attempt)
            pattern = _group_pattern(assignment)
            if pattern not in previously:
                previously.add(pattern)
                break
        else:
            warnings.warn(f"Could not find unique school partition for repeat {rep}")
        for fold in range(n_splits):
            val_schools = set(assignment[str(fold)])
            valid = np.flatnonzero(np.isin(groups, list(val_schools)))
            train = np.flatnonzero(~np.isin(groups, list(val_schools)))
            folds.append({"scheme": "school", "repeat": rep, "fold": fold,
                          "train_pos": train.tolist(), "valid_pos": valid.tolist(),
                          "n_train": len(train), "n_valid": len(valid)})
    df = pd.DataFrame(folds)
    validate_folds(df, ids, groups, n_splits=n_splits, n_repeats=n_repeats)
    return df


def validate_folds(df, ids, groups, n_splits=5, n_repeats=10):
    ids = [str(s) for s in ids]
    groups = np.asarray(groups, dtype=str)
    n = len(ids)
    for scheme in ("student", "school"):
        rows = df[df.scheme == scheme]
        if len(rows) != n_splits * n_repeats:
            raise ValueError(f"Wrong {scheme} split count: {len(rows)}")
        if rows[["repeat", "fold"]].duplicated().any():
            raise ValueError(f"Duplicate fold IDs in {scheme}")
        for rep in range(n_repeats):
            subset = rows[rows.repeat == rep]
            if set(subset.fold) != set(range(n_splits)):
                raise ValueError(f"Missing fold {scheme} repeat={rep}")
            coverage = np.zeros(n, dtype=int)
            for row in subset.itertuples():
                tr = np.asarray(row.train_pos, dtype=int)
                va = np.asarray(row.valid_pos, dtype=int)
                if (not len(tr) or not len(va) or np.intersect1d(tr, va).size or
                    set(np.r_[tr, va]) != set(range(n)) or
                    len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va)):
                    raise ValueError(f"Partition/overlap error in {scheme}/{rep}/{row.fold}")
                coverage[va] += 1
                if scheme == "school" and set(groups[tr]) & set(groups[va]):
                    raise ValueError("School leakage across group folds")
            if not np.all(coverage == 1):
                raise ValueError(f"Incomplete OOF coverage: {scheme} repeat {rep}")
    return True


def fold_digest(folds: pd.DataFrame, ids, groups, seed=42, n_splits=5, n_repeats=10):
    """Logical serialization independent of Parquet compression/library version."""
    objects = []
    for r in folds.itertuples():
        objects.append({"scheme": r.scheme, "repeat": int(r.repeat), "fold": int(r.fold),
                        "train_pos": list(map(int, r.train_pos)),
                        "valid_pos": list(map(int, r.valid_pos))})
    return sha_json({"algorithm": "m84-v1-reference-rkf-and-balanced-school",
                    "splitter": "RepeatedKFold", "seed": seed,
                    "n_splits": n_splits, "n_repeats": n_repeats,
                    "ids": [str(v) for v in ids],
                    "groups": list(map(str, groups)),
                    "folds": objects})


def freeze_folds(folds: pd.DataFrame, ids, groups, root: str | Path,
                 seed=42, n_splits=5, n_repeats=10) -> str:
    """Refuse to overwrite any existing fold contract with a different hash."""
    root = Path(root)
    validate_folds(folds, ids, groups, n_splits, n_repeats)
    h = fold_digest(folds, ids, groups, seed, n_splits, n_repeats)
    file = root / "folds" / "folds.parquet"
    info = root / "manifests" / "fold_manifest.json"
    if file.is_file() != info.is_file():
        raise RuntimeError("Partial frozen-fold artifact; inspect before recovery")
    if info.is_file():
        previous = json.loads(info.read_text(encoding="utf-8"))
        if previous.get("fold_hash") != h:
            raise RuntimeError("FROZEN FOLD HASH CHANGED; use a new versioned output directory")
        saved = pd.read_parquet(file)
        if fold_digest(saved, ids, groups, seed, n_splits, n_repeats) != h:
            raise RuntimeError("Frozen folds.parquet does not match manifest")
    else:
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_name("folds.tmp.parquet")
        folds.to_parquet(tmp, index=False)
        if fold_digest(pd.read_parquet(tmp), ids, groups, seed, n_splits, n_repeats) != h:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Parquet round-trip changed split definitions")
        tmp.replace(file)
        atomic_json(info, {
            "fold_hash": h, "algorithm": "m84-v1-reference-rkf-and-balanced-school",
            "student_n": len(ids), "school_n": len(set(map(str,groups))),
            "seed": seed, "n_splits": n_splits, "n_repeats": n_repeats,
            "n_rows": len(folds), "student_order_hash": sha_json(list(map(str,ids))),
            "school_fold_size_summary": folds[folds.scheme=="school"].n_valid.describe().to_dict(),
        })
        (root / "folds" / "fold_assignment.sha256").write_text(h + "\n")
    return h
