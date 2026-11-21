"""P14 — independent test inference and official submission validation.

The champion spec (models/champion_spec.json, written by P13) is re-fitted on ALL training students:
builder.fit(train ids) -> transform(train), transform(test) -> FoldModel per member -> weighted combination.
The fitted bundle is saved with joblib and re-loaded from disk for inference, so inference never touches
training-time objects in memory. No test labels, statistics or pseudo-labels are used anywhere.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from . import ID, TARGET
from .features.builder import select_families
from .io_utils import atomic_write_csv, read_json, write_json
from .models.preprocessing import FoldModel
from .models.zoo import make_model, needs_scaling


class SubmissionError(RuntimeError):
    pass


def validate_submission(sub: pd.DataFrame, fmt: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("columns match format", list(sub.columns) == list(fmt.columns), f"{list(sub.columns)} vs {list(fmt.columns)}")
    add("row count", len(sub) == len(fmt), f"{len(sub)} vs {len(fmt)}")
    add("same ids", set(sub[ID].astype(str)) == set(fmt[ID].astype(str)))
    add("same order", list(sub[ID].astype(str)) == list(fmt[ID].astype(str)))
    add("unique ids", sub[ID].is_unique)
    vals = pd.to_numeric(sub[TARGET], errors="coerce")
    add("numeric target", vals.notna().all(), f"{int(vals.isna().sum())} non-numeric")
    add("finite target", np.isfinite(vals).all())
    add("non-constant predictions", vals.nunique() > 1)
    return pd.DataFrame(checks)


class InferenceBundle:
    def __init__(self, builder, members: list[dict], weights: list[float], intercept: float, clip_range: tuple | None):
        self.builder, self.members, self.weights, self.intercept, self.clip_range = builder, members, weights, intercept, clip_range

    def predict(self, test_ids: list[str]) -> np.ndarray:
        Xte = self.builder.transform(test_ids)
        preds = []
        for m in self.members:
            X = Xte.reindex(columns=m["train_columns"])
            preds.append(m["model"].predict(X[m["columns"]]))
        p = self.intercept + np.sum([w * q for w, q in zip(self.weights, preds)], axis=0)
        if self.clip_range:
            p = np.clip(p, *self.clip_range)
        return p

    def save(self, path) -> str:
        import joblib

        joblib.dump(self, path, compress=3)
        return hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]

    @staticmethod
    def load(path) -> "InferenceBundle":
        import joblib

        return joblib.load(path)


def fit_bundle(ctx, spec: dict) -> InferenceBundle:
    from .features.assemble import make_builder

    b = ctx.bundle()
    train_ids = list(b.y.index.astype(str))
    builder = make_builder(ctx, spec.get("audio_families"), with_audio=True)
    builder.fit(train_ids)
    Xtr = builder.transform(train_ids)
    y = b.y.reindex(Xtr.index).values
    members = []
    for m in spec["members"]:
        cols = [c for c in m["columns"] if c in Xtr.columns] if m.get("columns") else select_families(Xtr.columns, m["families"])
        est = make_model(m["model"], ctx.cfg, seed=ctx.seed, params=dict(m.get("params", {})))
        sel_k = m.get("selector_k") if m.get("selector_k") is not None else ctx.cfg.get_path("models.selector_k")
        fm = FoldModel(est, sel_k, float(ctx.cfg.get_path("models.max_missing_frac", 0.6)), needs_scaling(m["model"])).fit(Xtr[cols], y)
        members.append({"name": m["name"], "model": fm, "columns": cols, "train_columns": list(Xtr.columns)})
    clip = (float(np.min(y)), float(np.max(y))) if ctx.cfg.get_path("submission.clip_predictions_to_train_range", True) else None
    return InferenceBundle(builder, members, list(spec.get("weights", [1.0])), float(spec.get("intercept", 0.0)), clip)


def make_submission(ctx, stage: str = "P14") -> dict:
    b = ctx.bundle()
    if any("subsampled" in n for n in b.notes):
        raise SubmissionError("data subsampled by profile (SMOKE); submissions require the full test set")
    spec = read_json(ctx.paths.models / "champion_spec.json")
    if not spec:
        raise SubmissionError("models/champion_spec.json missing — run P13 first")
    bundle = fit_bundle(ctx, spec)
    path = ctx.paths.models / "champion_bundle.joblib"
    digest = bundle.save(path)
    del bundle
    fresh = InferenceBundle.load(path)                     # independent inference from disk
    fmt = b.submission_format
    test_ids = list(fmt[ID].astype(str))
    pred = fresh.predict(test_ids)
    sub = pd.DataFrame({ID: fmt[ID].values, TARGET: pred})
    sub = sub[list(fmt.columns)]
    checks = validate_submission(sub, fmt)
    ctx.save_table(checks, stage, "submission_validation")
    if (checks["status"] == "FAIL").any():
        raise SubmissionError("submission validation failed: " + "; ".join(checks.loc[checks.status == "FAIL", "check"]))
    out = ctx.paths.submissions / ctx.cfg.get_path("submission.filename", "submission.csv")
    atomic_write_csv(sub, out)
    meta = {"file": str(out), "rows": len(sub), "bundle": str(path), "bundle_sha1": digest,
            "champion": spec.get("name"), "pred_mean": float(np.mean(pred)), "pred_sd": float(np.std(pred)),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(ctx.paths.submissions / "submission_meta.json", meta)
    return meta
