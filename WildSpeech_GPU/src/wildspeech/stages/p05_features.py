"""P05 — student-level feature engineering. Produces CROSS-FITTED features (each student's row built by a
builder fitted without that student) for the descriptive/discovery stages P06-P09, plus the feature registry."""
from __future__ import annotations

import pandas as pd

from .. import ID, TARGET, viz
from ..features.builder import crossfit_features, family_of
from ..stats import spearman
from ._common import get_runner, save_fig

S = "P05"


def _parse(col: str) -> dict:
    parts = col.split("__")
    return {"family": parts[0], "scope": parts[1] if len(parts) > 2 else "student",
            "base_feature": parts[2] if len(parts) > 3 else (parts[1] if len(parts) == 2 else parts[-1]),
            "stat": parts[3] if len(parts) > 3 else "value"}


def run(ctx) -> dict:
    b = ctx.bundle()
    runner = get_runner(ctx, with_audio=True)
    builder = runner.builder
    y = b.y
    ids = list(y.index.astype(str))
    X = crossfit_features(builder, ids, y, int(ctx.cfg.get_path("features.crossfit_folds", 5)), ctx.seed)
    ctx.save_parquet(X.reset_index().rename(columns={"index": ID}), S, "student_features_crossfit")
    fam = pd.DataFrame({"feature": X.columns, "family": [family_of(c) for c in X.columns]})
    counts = fam.groupby("family").size().rename("n_features").reset_index()
    miss = X.isna().mean().groupby([family_of(c) for c in X.columns]).mean().rename("mean_missing").rename_axis("family").reset_index()
    fs = counts.merge(miss, on="family")
    ctx.save_table(fs, S, "feature_family_summary")
    save_fig(ctx, S, "features_per_family", viz.bar_ci(fs.sort_values("n_features"), "family", "n_features", None, None,
                                                      "Student-level features per family"), fs)
    reg = []
    for c in X.columns:
        r, p, n = spearman(X[c], y.reindex(X.index))
        reg.append({"feature": c, **_parse(c), "source_stage": S, "missing_frac": float(X[c].isna().mean()),
                    "univariate_rho_outcome": r, "univariate_p": p, "n": n})
    ctx.registries.features.upsert(reg)
    ctx.save_table(pd.DataFrame(reg), S, "feature_catalog")
    return {"n_students": len(X), "n_features": X.shape[1], "families": fs["family"].tolist()}
