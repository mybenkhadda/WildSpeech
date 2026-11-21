"""Frozen, shared validation registry for every V2 stage.

* K-fold / school / LOSO / transfer schemes are generated with V1's `validation.build_folds` and V1's
  `validation.*` settings, so the folds are identical to the folds P01-P13 used (and V1 fold feature caches hit).
* Equal-training-size transfer controls separate bandwidth/domain effects from sample-size effects:
  for each target domain T a fixed holdout H_T (per repeat) is validated twice, by a model trained on n students
  of the SOURCE domain ("transfer") and by one trained on n students of T \\ H_T ("within"), with the same n.
* The registry is written once to outputs/V2/validation_registry.parquet together with a hash of the data and
  settings; later loads verify the hash so no stage can silently evaluate on different folds.
* Nested inner folds (for clip-level supervised features) are derived deterministically from the outer fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID, TARGET
from ..io_utils import atomic_write_parquet, read_json, stable_hash, write_json
from ..validation.splits import Fold, build_folds

BASE_SCHEMES = ("repeated_kfold", "school_kfold", "loso", "instrument_transfer", "sample_rate_transfer")
EQUAL_N_SCHEMES = {"sample_rate_equal_n": "sample_rate", "instrument_equal_n": "instrument"}


class RegistryMismatch(RuntimeError):
    pass


def _domain_attribute(train: pd.DataFrame, which: str) -> pd.Series:
    t = train.set_index(ID)
    if which == "sample_rate":
        if "original_sample_rate" not in t:
            return pd.Series(dtype=str)
        return pd.Series(np.where(t["original_sample_rate"].astype(float) < 16000, "low_rate", "16k"), index=t.index.astype(str))
    if which == "instrument":
        return t["instrument"].astype(str).set_axis(t.index.astype(str)) if "instrument" in t else pd.Series(dtype=str)
    raise ValueError(which)


def equal_n_transfer_folds(attr: pd.Series, scheme: str, n_repeats: int, holdout_frac: float, seed: int) -> list[Fold]:
    """Paired transfer/within folds with identical training size and identical validation students."""
    a = attr.astype(str)
    a.index = a.index.astype(str)
    domains = sorted(a.unique())
    if len(domains) < 2:
        return []
    out: list[Fold] = []
    k = 0
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + 7919 * r)
        for tgt in domains:
            tgt_ids = np.array(sorted(a.index[a == tgt]))
            src_ids = np.array(sorted(a.index[a != tgt]))
            n_hold = int(np.ceil(len(tgt_ids) * holdout_frac))
            if n_hold < 5 or len(tgt_ids) - n_hold < 5:
                continue
            perm = rng.permutation(tgt_ids)
            hold, rest = perm[:n_hold], perm[n_hold:]
            n_tr = int(min(len(src_ids), len(rest)))
            tr_transfer = rng.choice(src_ids, size=n_tr, replace=False)
            tr_within = rng.choice(rest, size=n_tr, replace=False)
            src = "+".join(sorted(set(a[src_ids])))
            out.append(Fold(scheme, r, k, tuple(sorted(tr_transfer)), tuple(sorted(hold)), label=f"transfer:{src}->{tgt}|n={n_tr}").check())
            k += 1
            out.append(Fold(scheme, r, k, tuple(sorted(tr_within)), tuple(sorted(hold)), label=f"within:{tgt}->{tgt}|n={n_tr}").check())
            k += 1
    return out


def inner_student_folds(train_ids, n_splits: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """Deterministic student-disjoint inner K-fold (rows are students, so plain shuffled K-fold is grouped)."""
    ids = np.array(sorted(map(str, train_ids)))
    if len(ids) < n_splits * 2:
        n_splits = max(2, len(ids) // 2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    parts = np.array_split(perm, n_splits)
    return [(sorted(np.concatenate([p for j, p in enumerate(parts) if j != i]).tolist()), sorted(parts[i].tolist()))
            for i in range(n_splits)]


def fold_key(fold: Fold) -> str:
    return stable_hash({"s": fold.scheme, "r": fold.repeat, "f": fold.fold, "tr": sorted(fold.train_ids),
                        "va": sorted(fold.valid_ids)})


def discovery_split(ids, fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic exploratory (discovery) / confirmatory partition of students."""
    ids = pd.Series(sorted(map(str, ids)))
    h = pd.util.hash_pandas_object(ids + f"#disc{seed}", index=False).to_numpy()
    order = np.argsort(h, kind="mergesort")
    n = int(round(len(ids) * fraction))
    disc = set(ids.iloc[order[:n]])
    return sorted(disc), sorted(set(ids) - disc)


class ValidationRegistry:
    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.path = ctx.paths.v2_root / "validation_registry.parquet"
        self.meta_path = ctx.paths.v2_root / "validation_registry.json"
        self._folds: dict[str, list[Fold]] = {}

    # -------------------------------------------------------------- hashing
    def data_hash(self) -> str:
        b = self.ctx.bundle()
        tr = b.train
        cols = [c for c in (ID, TARGET, "school_id", "instrument", "original_sample_rate") if c in tr]
        snap = tr[cols].sort_values(ID).astype(str).to_dict("list")
        # what defines the folds: data + V1 validation settings + seed + equal-n settings. The *list* of schemes
        # a profile requests is not part of the freeze: missing schemes are generated and appended.
        return stable_hash({"data": snap, "validation": self.cfg.section("validation"), "seed": self.ctx.seed,
                            "v2": {k: self.cfg.get_path(f"v2.validation.{k}") for k in
                                   ("equal_n_repeats", "equal_n_holdout_frac")}})

    # ---------------------------------------------------------------- build
    def _generate(self, schemes: list[str]) -> dict[str, list[Fold]]:
        b = self.ctx.bundle()
        train, y = b.train, b.y.copy()
        y.index = y.index.astype(str)
        v = self.cfg.section("v2").get("validation", {}) or {}
        out: dict[str, list[Fold]] = {}
        for scheme in schemes:
            if scheme in BASE_SCHEMES:
                folds = build_folds(scheme, train, y, self.cfg, self.ctx.seed)
            elif scheme in EQUAL_N_SCHEMES:
                attr = _domain_attribute(train, EQUAL_N_SCHEMES[scheme]).reindex(y.index).dropna()
                folds = equal_n_transfer_folds(attr, scheme, int(v.get("equal_n_repeats", 5)),
                                               float(v.get("equal_n_holdout_frac", 0.3)), self.ctx.seed)
            else:
                raise ValueError(f"unknown V2 validation scheme {scheme}")
            if folds:
                # canonical form (sorted ids) so a freshly built and a re-loaded registry are identical objects
                out[scheme] = [Fold(f.scheme, f.repeat, f.fold, tuple(sorted(f.train_ids)), tuple(sorted(f.valid_ids)), f.label)
                               for f in folds]
        return out

    @staticmethod
    def _to_frame(folds: dict[str, list[Fold]]) -> pd.DataFrame:
        rows = []
        for scheme, fs in folds.items():
            for f in fs:
                rows += [{"scheme": scheme, "repeat": f.repeat, "fold": f.fold, "label": f.label, ID: i, "role": "train"} for i in f.train_ids]
                rows += [{"scheme": scheme, "repeat": f.repeat, "fold": f.fold, "label": f.label, ID: i, "role": "valid"} for i in f.valid_ids]
        return pd.DataFrame(rows)

    @staticmethod
    def _from_frame(df: pd.DataFrame) -> dict[str, list[Fold]]:
        out: dict[str, list[Fold]] = {}
        for (scheme, r, k), d in df.groupby(["scheme", "repeat", "fold"], sort=True):
            tr = tuple(sorted(d.loc[d["role"] == "train", ID].astype(str)))
            va = tuple(sorted(d.loc[d["role"] == "valid", ID].astype(str)))
            lab = d["label"].iloc[0]
            out.setdefault(scheme, []).append(Fold(scheme, int(r), int(k), tr, va, "" if pd.isna(lab) else str(lab)).check())
        return out

    def load_or_build(self) -> "ValidationRegistry":
        h = self.data_hash()
        meta = read_json(self.meta_path, {}) or {}
        wanted = list(self.cfg.get_path("v2.validation.schemes", ["repeated_kfold"]) or ["repeated_kfold"])
        stored: dict[str, list[Fold]] = {}
        if self.path.exists():
            if meta.get("hash") != h:
                if not self.cfg.get_path("v2.validation.rebuild_if_changed", False):
                    raise RegistryMismatch(f"frozen validation registry hash {meta.get('hash')} != current {h}: data or "
                                           "validation settings changed. Set v2.validation.rebuild_if_changed=true to "
                                           "rebuild deliberately (all V2 comparisons are then recomputed).")
            else:
                stored = self._from_frame(pd.read_parquet(self.path))
        missing = [s for s in wanted if s not in stored]
        if missing or not self.path.exists():
            stored.update(self._generate(missing))
            atomic_write_parquet(self._to_frame(stored), self.path)
            write_json(self.meta_path, {"hash": h, "schemes": {s: len(f) for s, f in stored.items()},
                                        "previous_hash": meta.get("hash") if meta.get("hash") != h else None,
                                        "n_students": len(self.ctx.bundle().train)})
        self._folds = {s: stored[s] for s in wanted if s in stored}
        return self

    # ---------------------------------------------------------------- query
    def schemes(self) -> list[str]:
        return list(self._folds)

    def folds(self, scheme: str) -> list[Fold]:
        fs = list(self._folds.get(scheme, []))
        max_rep = self.cfg.get_path("v2.validation.max_repeats")
        if max_rep and scheme in ("repeated_kfold",) + tuple(EQUAL_N_SCHEMES):
            fs = [f for f in fs if f.repeat < int(max_rep)]
        return fs

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self.schemes():
            for f in self.folds(s):
                rows.append({"scheme": s, "repeat": f.repeat, "fold": f.fold, "label": f.label,
                             "n_train": len(f.train_ids), "n_valid": len(f.valid_ids),
                             "overlap": len(set(f.train_ids) & set(f.valid_ids))})
        return pd.DataFrame(rows)

    def inner(self, fold: Fold, n_splits: int | None = None) -> list[tuple[list[str], list[str]]]:
        k = int(n_splits or self.cfg.get_path("v2.validation.inner_splits", 4))
        seed = int(int(fold_key(fold), 16) % (2 ** 31 - 1))
        return inner_student_folds(fold.train_ids, k, seed)


def get_registry(ctx) -> ValidationRegistry:
    if "v2_validation_registry" not in ctx._cache:
        ctx._cache["v2_validation_registry"] = ValidationRegistry(ctx).load_or_build()
    return ctx._cache["v2_validation_registry"]
