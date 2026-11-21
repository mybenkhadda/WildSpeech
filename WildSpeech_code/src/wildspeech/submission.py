"""Section 27 - submission generation (disabled by default).

When explicitly enabled (``enable_submission`` and ``allow_test_inference``):
all transformations are fitted on training students only, each test student's
features depend only on that student's own data plus training-fitted
parameters, the official submission order and identifiers are preserved and
the file is validated against ``submission_format.csv``. No pseudo-labelling,
no test-batch statistics, no test-driven selection, no automatic upload.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from . import data_access as da
from .ablation import EXPERIMENTS
from .baseline_models import BASELINE_MODELS, make_model
from .config import config_hash
from .data_audit import _adapt_kwargs, load_read_kwargs, FILE_ROLES


def read_submission_format(ctx) -> pd.DataFrame:
    root = ctx.data_root()
    per_file, _ = load_read_kwargs(root / "pandas_read_csv_kwargs.json", list(FILE_ROLES))
    kwargs, _ = _adapt_kwargs(root / "submission_format.csv", per_file.get("submission_format.csv", {}))
    return pd.read_csv(root / "submission_format.csv", **kwargs)


def validate_submission(sub: pd.DataFrame, fmt: pd.DataFrame, id_col: str, target_col: str) -> dict:
    checks = {
        "row_count_matches": len(sub) == len(fmt),
        "column_names_match": list(sub.columns) == list(fmt.columns),
        "no_missing_predictions": bool(sub[target_col].notna().all()),
        "all_finite": bool(np.isfinite(pd.to_numeric(sub[target_col], errors="coerce")).all()),
        "no_duplicate_ids": not sub[id_col].duplicated().any(),
        "exact_id_coverage": set(sub[id_col].astype(str)) == set(fmt[id_col].astype(str)),
        "order_preserved": list(sub[id_col].astype(str)) == list(fmt[id_col].astype(str)),
    }
    checks["valid"] = all(checks.values())
    return checks


def stage_submission(ctx) -> dict:
    from .fold_features import FoldFeatureBuilder
    from .io_utils import atomic_write_csv, atomic_write_json

    cfg = ctx.config
    if not (cfg.get("enable_submission") and cfg.get("allow_test_inference")):
        return {"outputs": [], "status": "SKIPPED_DISABLED", "reason": "submission disabled"}
    schema = ctx.load_json("schema_registry")
    sub_fields = schema["groups"]["submission"]["assigned"]
    id_col = sub_fields["student_id"]
    fmt = read_submission_format(ctx)
    target_col = sub_fields.get("target") or [c for c in fmt.columns if c != id_col][0]
    fset = cfg["submission_feature_set"]
    if fset not in EXPERIMENTS or fset in ("A23_RESIDUAL_MULTIMODAL",):
        raise ValueError(f"submission_feature_set {fset} not supported for submission")
    families = set(EXPERIMENTS[fset]) or {"SCREENING"}
    kind = BASELINE_MODELS.get(cfg["submission_model"], cfg["submission_model"])
    meta = da.student_meta(ctx, "train")
    y = meta["target"].dropna().astype(float)
    if families <= {"SCREENING"}:
        tr = da.screening(ctx, "train")
        te = da.screening(ctx, "test")
        if te is None:
            raise RuntimeError("screening_features_test missing: rerun Section 03")
        base = pd.concat([tr, te])
        builder = FoldFeatureBuilder(ctx, base=base)
    else:
        if not ctx.has("student_features_test"):
            raise RuntimeError("student_features_test missing: rerun Section 13 with allow_test_inference=True")
        builder = FoldFeatureBuilder(ctx, include_test=True)
    test_ids = [str(i) for i in fmt[id_col].astype(str)]
    missing = [i for i in test_ids if i not in builder.base.index]
    if missing:
        ctx.logger.warning("%d submission ids lack features; their predictions fall back to the training mean", len(missing))
    train_ids = [i for i in y.index if i in builder.base.index]
    present_test = [i for i in test_ids if i in builder.base.index]
    Xtr, Xte = builder.build(train_ids, present_test, families, key=("SUBMISSION",))
    model = make_model(kind, cfg, n_features=Xtr.shape[1], n_samples=len(Xtr))
    model.fit(Xtr, y.loc[train_ids].to_numpy())
    pred = pd.Series(model.predict(Xte), index=present_test)
    fallback = float(y.loc[train_ids].mean())
    out = fmt.copy()
    out[target_col] = [float(pred.get(i, fallback)) for i in test_ids]
    if pd.api.types.is_integer_dtype(fmt[target_col]):
        ctx.logger.warning("submission_format target column is integer-typed; writing floats as required for regression")
    checks = validate_submission(out, fmt, id_col, target_col)
    checks["n_fallback_predictions"] = len(missing)
    path = ctx.paths.submissions / cfg["submission_filename"]
    if not checks["valid"]:
        ctx.save_json(checks, "submission_validation")
        raise RuntimeError(f"submission failed validation: {checks}")
    atomic_write_csv(out, path)
    provenance = {"model": cfg["submission_model"], "kind": kind, "feature_set": fset, "families": sorted(families),
                  "n_train_students": len(train_ids), "n_features": int(Xtr.shape[1]), "config_hash": config_hash(cfg),
                  "created_at": _dt.datetime.now().isoformat(timespec="seconds"), "file": str(path),
                  "note": "Not uploaded automatically. Fitted on training students only."}
    atomic_write_json(path.with_suffix(".provenance.json"), provenance)
    ctx.save_json(checks, "submission_validation")
    ctx.save_json(provenance, "submission_provenance")
    ctx.store.register("submission_csv", path, ctx.current_stage, "csv")
    return {"outputs": ["submission_validation", "submission_provenance", "submission_csv"], "summary": {**checks, "file": str(path)}}
