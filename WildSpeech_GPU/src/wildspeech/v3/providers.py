"""One factory for every V3 student-level provider so P28 and P29 evaluate exactly the same representations
(same tables, same fingerprints, same fold-provider caches)."""
from __future__ import annotations

import pandas as pd

from .. import FILE


def load_embeddings(ctx, models: list[str], splits=("train",)) -> pd.DataFrame | None:
    """Pooled embeddings of the preferred model (registry priority, then child phonetic > child word > others) with
    saved vectors, taken only from (model, split) pairs whose ASR output is currently usable."""
    from ..asr.checkpointing import ASRCheckpointStore
    from ..asr.model_registry import registry_for
    from ..asr.outputs import usable_pairs

    specs, _ = registry_for(ctx)
    usable = usable_pairs(ctx, tuple(splits))
    order = sorted(models, key=lambda k: (getattr(specs.get(k), "priority", 99),
                                          {"child_phonetic": 0, "child_word": 1}.get(getattr(specs.get(k), "role", ""), 2), k))
    for k in order:
        st = ASRCheckpointStore(ctx.paths.v3_cache, k)
        parts = [st.load_embeddings(s) for s in splits if (k, s) in usable]
        parts = [p for p in parts if not p.empty]
        if parts:
            e = pd.concat(parts, ignore_index=True).drop_duplicates(FILE, keep="last")
            e.attrs["model_key"] = k
            return e
    return None


def load_encoder_embeddings(ctx, splits=("train",), exclude: tuple[str, ...] = ()) -> dict[str, pd.DataFrame]:
    """{encoder model_key: pooled embeddings} for accepted frozen-encoder outputs, in registry priority order."""
    from ..asr.checkpointing import ASRCheckpointStore
    from ..asr.model_registry import by_priority, registry_for
    from ..asr.outputs import usable_pairs

    specs, _ = registry_for(ctx)
    usable = usable_pairs(ctx, tuple(splits))
    out = {}
    for k in by_priority([k for k, s in specs.items() if s.output == "embeddings" and k not in set(exclude)], specs):
        st = ASRCheckpointStore(ctx.paths.v3_cache, k)
        parts = [st.load_embeddings(s) for s in splits if (k.lower(), s) in usable or (k, s) in usable]
        parts = [p for p in parts if not p.empty]
        if parts:
            e = pd.concat(parts, ignore_index=True).drop_duplicates(FILE, keep="last")
            e.attrs["model_key"] = k
            out[k] = e
    return out


def build_all(ctx, ev=None, splits=None, include: tuple[str, ...] | None = None,
              exclude_models: tuple[str, ...] = ()):
    """-> (providers dict, availability DataFrame, primary model key, models list). `splits` defaults to
    v3.asr.splits, so test-split clip features (V3_SUBMISSION) feed the provider tables for test students; rows of
    training students never depend on test clips. `exclude_models` drops every feature derived from those models
    (overlap sensitivity: e.g. P29 re-evaluates without the Pasketti models)."""
    splits = tuple(splits or ctx.cfg.get_path("v3.asr.splits", ["train"]) or ["train"])
    from ..evaluation.robustness import recording_provider
    from ..phonetics.builder import load_phonetic_tables
    from ..representations.compact_features import build_v3_providers
    from ..stages.p26_correct_only import primary_model
    from ..stages.p27_asr_disagreement import consensus_path

    excl = tuple(sorted({m.lower() for m in exclude_models}))
    key = ("v3_providers", tuple(splits), tuple(include or ()), excl)
    if key in ctx._cache:
        return ctx._cache[key]
    wide, det = load_phonetic_tables(ctx, splits)
    if excl and not det.empty:
        det = det[~det["model_key"].str.lower().isin(excl)]
        from ..phonetics.builder import model_prefix

        drop = [c for c in wide.columns if "::" in c and any(c.startswith(model_prefix(m) + "::") for m in excl)]
        wide = wide.drop(columns=drop)
    models = sorted(det["model_key"].unique()) if not det.empty else []
    pm = primary_model(ctx, wide) if models else None
    cp = consensus_path(ctx)
    cons = pd.read_parquet(cp) if (cp.exists() and not excl) else None   # consensus mixes every model
    emb = load_embeddings(ctx, models, splits) if models else None
    enc = load_encoder_embeddings(ctx, splits, excl)
    residual_fn = None
    if ev is not None:
        from ..evaluation.nested_asr_cv import InnerResidualOracle

        residual_fn = InnerResidualOracle(ev)
    provs, avail = build_v3_providers(ctx, wide, det, models, pm, cons, emb, residual_fn, include, enc)
    rec = recording_provider(ctx)
    if rec is not None:
        provs["REC"] = rec
    avail = pd.concat([avail, pd.DataFrame([{"family": "REC", "available": rec is not None, "note": "recording-conditions control"}])],
                      ignore_index=True)
    out = (provs, avail, pm, models)
    ctx._cache[key] = out
    return out
