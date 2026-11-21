"""One factory for every V2 student-level feature provider, so P17, P19, P21 and P22 evaluate exactly the same
representations (same caches, same fingerprints). Selections made by earlier stages are read from their
selection.json files (or forced through config), never re-derived silently.

  HARM_<variant>   ConditionalAcousticProvider on the P16-selected harmonized variant (core features)
  SOFT_<set>       nested soft correctness for the P17-selected audio feature set   (family SOFT)
  SOFTMETA         nested soft correctness from item metadata only                   (family SOFTMETA)
  PHON             conditional aggregation of P18 phonetic clip features              (family PHON)
  REF              nested reference-conditioned deviations (P19)                      (family REF)
  FM               compact representation of the P20-selected layer                   (family FM)
  CPX              complexity-sensitivity slopes                                      (family CPX)
"""
from __future__ import annotations

import pandas as pd

from .. import FILE
from ..features.conditional_aggregation import CachedProvider, ConditionalAcousticProvider


def read_selection(ctx, sid: str) -> dict:
    return ctx.read_stage_json(sid, "selection", {}) or {}


def selected_variant(ctx) -> str:
    forced = ctx.cfg.get_path("v2.harmonization.selected", "auto")
    if forced and forced != "auto":
        return str(forced)
    return str(read_selection(ctx, "P16").get("selected_variant", "orig"))


def harm_columns(ctx, table: pd.DataFrame) -> list[str]:
    core = ctx.cfg.get_path("v2.harmonization.core_features", {}) or {}
    cols = [f"{fam.upper()}::{f}" for fam, fs in core.items() for f in fs]
    return [c for c in cols if c in table.columns]


def harm_provider(ctx, variant: str, splits=("train",)) -> CachedProvider | None:
    from ..harmonization.bandwidth import load_variant_features

    fams = list(ctx.cfg.get_path("v2.harmonization.families", [])) + (["robust"] if ctx.cfg.get_path("v2.harmonization.robust_family", True) else [])
    tab = load_variant_features(ctx, variant, fams, splits)
    cols = harm_columns(ctx, tab)
    if not cols:
        return None
    p = ConditionalAcousticProvider(ctx, f"HARM_{variant}", tab, cols, prefix="HARM",
                                    controls=tuple(ctx.cfg.get_path("v2.harmonization.residualize_controls",
                                                                     ["item", "domain", "duration", "complexity", "correctness"])),
                                    shrinkage_k=float(ctx.cfg.get_path("v2.harmonization.shrinkage_k", 10.0)),
                                    description=f"harmonized acoustics ({variant})", version="harm-agg-1")
    return CachedProvider(p)


def build_bank(ctx, splits=("train",)):
    """ClipFeatureBank with every block that has cached clip features."""
    from ..acoustic.store import FeatureStore
    from ..correctness.classifiers import ClipFeatureBank
    from ..harmonization.bandwidth import load_variant_features
    from ..phonetic.runner_v2 import load_phonetic_features

    key = ("v2_bank", tuple(splits))
    if key in ctx._cache:
        return ctx._cache[key]
    bank = ClipFeatureBank(ctx)
    store = FeatureStore(ctx.paths.cache / "features")
    meta = ("true_duration_s", "decoded_duration_s", "cropped", "file_sr")

    def v1(fams):
        out = None
        for f in fams:
            parts = [store.load(f, s, prefix=True) for s in splits]
            parts = [p for p in parts if not p.empty]
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True).drop_duplicates(FILE, keep="last")
            df = df.drop(columns=[c for c in df.columns if c.split("::")[-1] in meta])
            out = df if out is None else out.merge(df, on=FILE, how="outer")
        return out

    bank.add_block("TEMPORAL", v1(["temporal"]), "cache/features temporal")
    bank.add_block("SPECTRAL", v1(["spectral", "mfcc", "mel"]), "cache/features spectral+mfcc+mel")
    bank.add_block("QUALITY", v1(["quality"]), "cache/features quality")
    bank.add_block("WAVELET_PITCH", v1(["wavelet", "pitch"]), "cache/features wavelet+pitch")
    v = selected_variant(ctx)
    fams = list(ctx.cfg.get_path("v2.harmonization.families", [])) + ["robust"]
    hv = load_variant_features(ctx, v, fams, splits)
    if not hv.empty and v != "orig":
        bank.add_block("HARMONIZED", hv, f"P16 variant {v}")
    elif not hv.empty:
        rob = [c for c in hv.columns if c.startswith("ROBUST::")]
        if rob:
            bank.add_block("HARMONIZED", hv[[FILE] + rob], "P16 variant orig (robust family)")
    ph = load_phonetic_features(ctx, splits)
    if not ph.empty:
        bank.add_block("PHONETIC", ph, "P18 phonetic families")
    emb = fm_table(ctx, splits)
    if emb is not None and not emb.empty:
        bank.add_block("EMBEDDING", emb, "P20 selected layer")
    ctx._cache[key] = bank
    return bank


def fm_table(ctx, splits=("train",)) -> pd.DataFrame | None:
    sel = read_selection(ctx, "P20")
    if not sel.get("model"):
        return None
    from ..foundation.encoders import EmbeddingShardStore

    st = EmbeddingShardStore(ctx.paths.v2_cache / "embeddings", sel["model"], sel["store_fp"])
    return st.load(int(sel["layer_index"]), int(sel["pooling_index"]), splits)


def soft_provider(ctx, bank, feature_set: str, prefix: str, model: str | None = None) -> CachedProvider:
    from ..correctness.soft_features import SoftCorrectnessProvider

    m = model or ctx.cfg.get_path("v2.correctness.crossfit_model", "LIGHTGBM")
    try:
        from ..correctness.classifiers import make_classifier

        make_classifier(m, ctx.cfg)
    except Exception:  # noqa: BLE001 - library unavailable -> pre-specified fallback
        m = "LOGISTIC"
    return CachedProvider(SoftCorrectnessProvider(ctx, bank, feature_set, m, prefix=prefix))


def phon_provider(ctx, splits=("train",)) -> CachedProvider | None:
    from ..phonetic.runner_v2 import load_phonetic_features

    ph = load_phonetic_features(ctx, splits)
    cols = [c for c in ph.columns if c != FILE and not c.split("::")[-1].startswith("post_")]
    if not cols:
        return None
    p = ConditionalAcousticProvider(ctx, "PHON", ph, cols, prefix="PHON", controls=("item", "domain"),
                                    stats=("mean", "sd", "correct_mean", "incorrect_mean"), description="phonetic", version="phon-agg-1")
    return CachedProvider(p)


def cpx_provider(ctx, splits=("train",)) -> CachedProvider:
    from ..features.complexity_sensitivity import ComplexitySensitivityProvider
    from ..phonetic.runner_v2 import load_phonetic_features

    ph = load_phonetic_features(ctx, splits)
    cols = [c for c in ph.columns if c.endswith(("::ph_per", "::gop_mean", "::w_er"))]
    return CachedProvider(ComplexitySensitivityProvider(ctx, "CPX", ph if cols else None, cols))


def fm_provider(ctx, splits=("train",)) -> CachedProvider | None:
    from ..foundation.compression import FoundationProvider

    emb = fm_table(ctx, splits)
    if emb is None or emb.empty:
        return None
    sel = read_selection(ctx, "P20")
    f = ctx.cfg.get_path("v2.foundation", {}) or {}
    return CachedProvider(FoundationProvider(ctx, f"FM_{sel['model']}_L{sel['layer']}_{sel['pooling']}", emb,
                                             int(f.get("student_pcs", 8)), 32, store_fp=f"{sel['store_fp']}:{sel['layer']}:{sel['pooling']}"))


def ref_provider(ctx, splits=("train",), save_clip_dir=None) -> CachedProvider | None:
    from ..harmonization.bandwidth import configured_variants
    from ..phonetic.runner_v2 import load_phonetic_features
    from ..reference.similarity import PairDistanceCache, ReferenceDeviationComputer, ReferenceProvider
    from ..reference.templates import SequenceExtractionRunner, SequenceStore, SyntheticReferences

    r = ctx.cfg.get_path("v2.reference", {}) or {}
    vname = r.get("variant", "auto")
    vname = selected_variant(ctx) if vname in (None, "auto") else vname
    variants = {v.name: v for v in configured_variants(ctx.cfg)}
    if vname not in variants:
        from ..harmonization.bandwidth import Variant

        vv = (ctx.cfg.get_path("v2.harmonization.variants", {}) or {}).get(vname, {})
        variants[vname] = Variant(vname, vv.get("lowpass_hz"), vv.get("highpass_hz"), bool(vv.get("amp_norm", False)))
    runner = SequenceExtractionRunner(ctx, variants[vname])
    seqs = runner.store.load(splits)
    if not seqs.get("mfcc"):
        return None
    syn_map, syn_seq = SyntheticReferences(ctx, runner).load(sorted(ctx.bundle().all_clips["item_id"].unique()))
    for k in seqs:
        seqs[k].update(syn_seq.get(k, {}))
    post = emb = ctc = None
    ph = load_phonetic_features(ctx, splits, key_only=False)
    if not ph.empty:
        pc = [c for c in ph.columns if "::post_" in c]
        if pc and r.get("use_posteriors", True):
            fam = pc[0].split("::")[0]
            post = ph[[FILE] + [c for c in pc if c.startswith(fam + "::")]]
        ll = [c for c in ph.columns if c.endswith("::expected_ll") and c.startswith("PH_CTC")]
        if ll:
            ctc = ph.set_index(FILE)[ll[0]].dropna()
    if r.get("use_embeddings", True):
        emb = fm_table(ctx, splits)
    cache = PairDistanceCache(ctx.paths.v2_cache / "reference_pairs", runner.fp + f"_b{r.get('band_frac')}_w{r.get('local_window')}")
    comp = ReferenceDeviationComputer(ctx, seqs, cache, post, emb, ctc)
    extra = {"seq": runner.fp, "post": post is not None, "emb": read_selection(ctx, "P20").get("store_fp") if emb is not None else None,
             "ctc": ctc is not None}
    return CachedProvider(ReferenceProvider(ctx, f"REF_{vname}", comp, extra, syn_map, save_clip_dir))


def build_providers(ctx, include=("HARM", "SOFT", "SOFTMETA", "PHON", "REF", "FM", "CPX"), splits=("train",)) -> dict:
    """name -> CachedProvider for every available representation (unavailable ones are simply absent)."""
    out = {}
    if "HARM" in include:
        p = harm_provider(ctx, selected_variant(ctx), splits)
        if p is not None:
            out["HARM"] = p
    if "SOFT" in include or "SOFTMETA" in include:
        bank = build_bank(ctx, splits)
        sel = read_selection(ctx, "P17")
        if "SOFT" in include:
            s = sel.get("selected_set", "COMBINED")
            if bank.set_available(s):
                out["SOFT"] = soft_provider(ctx, bank, s, "SOFT", sel.get("model"))
        if "SOFTMETA" in include:
            out["SOFTMETA"] = soft_provider(ctx, bank, "META", "SOFTMETA", sel.get("model"))
    if "PHON" in include:
        p = phon_provider(ctx, splits)
        if p is not None:
            out["PHON"] = p
    if "REF" in include:
        p = ref_provider(ctx, splits)
        if p is not None:
            out["REF"] = p
    if "FM" in include:
        p = fm_provider(ctx, splits)
        if p is not None:
            out["FM"] = p
    if "CPX" in include:
        out["CPX"] = cpx_provider(ctx, splits)
    return out
