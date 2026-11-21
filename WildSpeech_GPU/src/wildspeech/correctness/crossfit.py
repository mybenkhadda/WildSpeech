"""Student-disjoint cross-fitting of clip-level supervised predictions.

clip_oof              single-level OOF (GroupKFold over students) — used for clip-level metrics only.
nested_predictions    for ONE outer fold with training students T:
                        * clips of T     -> predicted by models fitted on inner folds of T that exclude the
                                            student (inner OOF), so student-level training features never come
                                            from a model that saw that student's labels;
                        * all other clips (validation / test students) -> model fitted on all clips of T.
                      Optional calibration is fitted on the inner-OOF predictions of T only.
Every function accepts a `make` factory returning an object with fit(rows) / predict_proba(rows).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..v2.validation_registry import inner_student_folds


def student_groups_kfold(ids: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row-index folds over clips such that every student's clips fall in exactly one validation fold."""
    ids = np.asarray(ids).astype(str)
    folds = inner_student_folds(np.unique(ids), n_splits, seed)
    out = []
    for tr_ids, va_ids in folds:
        tr = np.flatnonzero(np.isin(ids, tr_ids))
        va = np.flatnonzero(np.isin(ids, va_ids))
        out.append((tr, va))
    return out


def clip_oof(c: pd.DataFrame, rows: np.ndarray, make: Callable, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """OOF probabilities for `rows` (positions into c); labels required on the fitted rows."""
    p = np.full(len(rows), np.nan)
    ids = c.iloc[rows][ID].astype(str).to_numpy()
    y = c.iloc[rows]["item_score"].to_numpy(float)
    for tr, va in student_groups_kfold(ids, n_splits, seed):
        fit_rows = rows[tr][np.isfinite(y[tr])]
        if len(np.unique(c.iloc[fit_rows]["item_score"])) < 2:
            continue
        p[va] = make().fit(fit_rows).predict_proba(rows[va])
    return p


def nested_predictions(c: pd.DataFrame, train_ids, target_rows: np.ndarray, make: Callable,
                       inner_folds: list[tuple[list[str], list[str]]], calibrator_factory: Callable | None = None) -> pd.DataFrame:
    """Returns FILE, subject_id, p (calibrated if a calibrator is given), p_raw, source for target_rows."""
    train_ids = set(map(str, train_ids))
    ids = c[ID].astype(str).to_numpy()
    scored = c["item_score"].notna().to_numpy()
    tgt = np.asarray(target_rows)
    p = np.full(len(c), np.nan)
    src = np.full(len(c), "", dtype=object)
    is_tr = np.isin(ids, list(train_ids))
    for in_tr, in_va in inner_folds:
        fit_rows = np.flatnonzero(np.isin(ids, in_tr) & scored)
        pred_rows = np.intersect1d(np.flatnonzero(np.isin(ids, in_va)), tgt)
        if len(pred_rows) == 0 or len(fit_rows) < 20:
            continue
        p[pred_rows] = make().fit(fit_rows).predict_proba(pred_rows)
        src[pred_rows] = "inner_oof"
    outer = np.intersect1d(np.flatnonzero(~is_tr), tgt)
    if len(outer):
        fit_rows = np.flatnonzero(is_tr & scored)
        p[outer] = make().fit(fit_rows).predict_proba(outer)
        src[outer] = "outer"
    p_raw = p.copy()
    if calibrator_factory is not None:
        cal_rows = np.flatnonzero(is_tr & scored & np.isfinite(p_raw))
        if len(cal_rows) > 50:
            cal = calibrator_factory().fit(p_raw[cal_rows], c["item_score"].to_numpy(float)[cal_rows])
            ok = np.isfinite(p_raw)
            p[ok] = cal.transform(p_raw[ok])
    out = pd.DataFrame({FILE: c[FILE].to_numpy()[tgt], ID: ids[tgt], "p": p[tgt], "p_raw": p_raw[tgt], "source": src[tgt]})
    return out
