"""Inventory of existing artefacts (P00-P22 statuses, freshness, tables/figures, V1 and V2 caches)."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd


def stage_inventory(ctx) -> pd.DataFrame:
    from ..orchestrator import STAGES, Pipeline

    pipe = Pipeline(ctx)
    rows = []
    for sid in STAGES:
        st = pipe.status(sid)
        d = ctx.paths.stage_dir(sid)
        rows.append({"stage": sid, "status": st.get("status", "never_run"),
                     "up_to_date": (not pipe.is_stale(sid)) if st.get("status") == "completed" else None,
                     "finished": st.get("finished"), "seconds": st.get("seconds"), "profile": st.get("profile"),
                     "n_tables": len(list((d / "tables").glob("*.csv"))), "n_figures": len(list((d / "figures").glob("*.png"))),
                     "dir": str(d.relative_to(ctx.paths.root)), "note": st.get("note", "")})
    return pd.DataFrame(rows)


def cache_inventory(ctx) -> pd.DataFrame:
    from ..acoustic.store import FeatureStore

    rows = []
    for tag, root in (("v1", ctx.paths.cache / "features"), ("v2", ctx.paths.v2_cache / "features")):
        if root.exists():
            st = FeatureStore(root).status()
            if not st.empty:
                rows.append(st.assign(store=tag))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    extra = []
    for kind, pattern in (("sequences", "sequences/*/*/shard-*.npz"), ("embeddings", "embeddings/*/*/*/shard-*.npy"),
                          ("fold_providers", "fold_providers/*/*.parquet"), ("eval", "eval/*_oof.parquet"),
                          ("perturbation", "perturbation/*.parquet"), ("reference_pairs", "reference_pairs/*/pairs-*.parquet")):
        files = glob.glob(str(ctx.paths.v2_cache / pattern))
        extra.append({"store": "v2", "family": kind, "split": "-", "parts": len(files),
                      "bytes": int(sum(Path(f).stat().st_size for f in files))})
    return pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
