"""V4 champion refit on all labelled students, independent re-load, validated submission.

1. the champion spec (model_registry/champion_spec.json, chosen from validated OOF comparisons) is rebuilt
2. V4Evaluator.fit_full_v3: tabular builder fitted on all training students; every provider fitted on training
   clips only and applied to test clips; the V4FoldModel fitted on all labelled students
3. the fitted fold model is saved with joblib, deleted, RE-LOADED from disk and only then applied to the test
   design (in-memory and re-loaded predictions must agree)
4. checks: exact columns / order of submission_format.csv, exact subject ids, no missing / duplicate students,
   no NaN / inf, non-constant, no train-only field (school_id / score_status) in the design, test design rows ==
   test students, no train student in the test design, no test label anywhere
No fallback prediction is ever fabricated: any failed check raises SubmissionError and nothing is written.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import ID, TARGET
from ..evaluation.nested_asr_cv import V3Spec
from ..submission import SubmissionError, validate_submission
from . import TRAIN_ONLY_FIELDS


def spec_from_dict(d: dict) -> V3Spec:
    d = dict(d)
    if d.get("alphas") is not None:
        d["alphas"] = tuple(d["alphas"])
    if d.get("stack_models") is not None:
        d["stack_models"] = tuple(d["stack_models"])
    return V3Spec(**d)


def extra_checks(sub: pd.DataFrame, fmt: pd.DataFrame, Xtr: pd.DataFrame, Xte: pd.DataFrame, test_ids: list[str],
                 train_ids: list[str], p_mem: np.ndarray, p_disk: np.ndarray) -> pd.DataFrame:
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    vals = pd.to_numeric(sub[TARGET], errors="coerce").to_numpy(float)
    add("no NaN predictions", not np.isnan(vals).any(), f"{int(np.isnan(vals).sum())} NaN")
    add("no infinite predictions", not np.isinf(vals).any(), f"{int(np.isinf(vals).sum())} inf")
    add("no missing test students", set(fmt[ID].astype(str)) <= set(sub[ID].astype(str)))
    add("no duplicate test students", not sub[ID].astype(str).duplicated().any())
    add("test design rows == test students (same order)", list(Xte.index.astype(str)) == list(map(str, test_ids)))
    add("no train student in the test design", not (set(Xte.index.astype(str)) & set(map(str, train_ids))))
    bad = [c for c in Xtr.columns if any(t in c.lower() for t in TRAIN_ONLY_FIELDS)]
    add("no train-only field in the design", not bad, ", ".join(bad[:10]))
    add("no target-derived column in the design", not any(TARGET in c.lower() for c in Xtr.columns))
    add("re-loaded model reproduces in-memory predictions", bool(np.allclose(p_mem, p_disk, rtol=1e-6, atol=1e-6)),
        f"max |diff| = {float(np.max(np.abs(p_mem - p_disk))) if len(p_mem) else 0.0:.3g}")
    return pd.DataFrame(rows)


def make_submission(ctx, ev, spec: V3Spec, store, cfg4: dict, champion: dict) -> dict:
    import joblib

    b = ctx.bundle()
    if any("subsampled" in n for n in b.notes):
        raise SubmissionError("data subsampled (SMOKE run): submissions require the full test set")
    fmt = b.submission_format.copy()
    test_ids = list(fmt[ID].astype(str))
    train_ids = list(ev.all_train)
    if set(test_ids) & set(train_ids):
        raise SubmissionError("student overlap between train and test")
    if set(test_ids) != set(b.test_students[ID].astype(str)):
        raise SubmissionError("submission_format ids differ from test_student_metadata ids")
    fmod, Xtr, Xte = ev.fit_full_v3(spec, test_ids)
    Xte = Xte.reindex(test_ids)
    p_mem = fmod.predict(Xte)
    path = store.path("models", "champion_fold_model.joblib")
    joblib.dump({"fold_model": fmod, "columns": list(Xtr.columns), "spec": spec.as_dict()}, path, compress=3)
    del fmod
    fresh = joblib.load(path)["fold_model"]
    p_disk = fresh.predict(Xte.reindex(columns=Xtr.columns))
    raw = np.asarray(p_disk, float)
    y = ev.y.reindex(train_ids).to_numpy(float)
    pred = raw.copy()
    clipped = np.zeros(len(pred), bool)
    if (cfg4.get("submission", {}) or {}).get("clip_to_train_range", True):
        lo, hi = float(np.min(y)), float(np.max(y))
        clipped = (pred < lo) | (pred > hi)
        pred = np.clip(pred, lo, hi)
    sub = pd.DataFrame({ID: fmt[ID].values, TARGET: pred})[list(fmt.columns)]
    checks = pd.concat([validate_submission(sub, fmt),
                        extra_checks(sub, fmt, Xtr, Xte, test_ids, train_ids, np.asarray(p_mem, float), raw)],
                       ignore_index=True)
    store.save_table(checks, "submission", "submission_validation.csv")
    if (checks["status"] == "FAIL").any():
        raise SubmissionError("submission validation failed: " + "; ".join(checks.loc[checks["status"] == "FAIL", "check"]))
    tp = pd.DataFrame({ID: test_ids, "prediction": pred, "raw_prediction": raw, "clipped_to_train_range": clipped,
                       "champion": spec.name})
    store.save_table(tp, "submission", "test_predictions.parquet", csv=True)
    fname = (cfg4.get("submission", {}) or {}).get("filename", "submission.csv")
    out = store.save_table(sub, "submission", fname)
    meta = {"file": str(out), "rows": int(len(sub)), "champion": spec.name, "champion_is_baseline": bool(champion.get("is_baseline")),
            "champion_reason": champion.get("reason"), "pred_mean": float(np.mean(pred)), "pred_sd": float(np.std(pred)),
            "n_clipped": int(clipped.sum()), "model_file": str(path), "n_design_columns": int(Xtr.shape[1]),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    store.save_json(meta, "submission", "submission_meta.json")
    return meta
