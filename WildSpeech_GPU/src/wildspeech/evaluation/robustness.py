"""Robustness, stability, confounding and negative controls for V3 reading prediction (P29).

recording_provider(ctx)       'REC' negative/confound control: V1 recording-quality + temporal clip features
                              aggregated per student (no phonetic content). If phonetic families only match what
                              recording conditions give, their value is not phonetic.
transfer_table(rows)          ΔR² of every experiment across repeated K-fold, school-grouped K-fold, leave-one-
                              school-out, instrument transfer and sample-rate transfer + sign consistency
selection_stability(ev, ...)  per fold, the columns a 'select' reducer keeps -> selection frequency and mean
                              pairwise Jaccard (feature-family stability)
confounding_table(ev, ...)    student-level association of each family's (training-fold-fitted) features with
                              recording metadata (sample rate, instrument) and with the baseline residual before
                              and after partialling recording metadata out
subgroups(oof, meta)          R² / bias per instrument, sample-rate group and outcome quintile
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import CachedProvider, ConditionalAcousticProvider


def recording_provider(ctx) -> CachedProvider | None:
    from ..acoustic.store import FeatureStore

    store = FeatureStore(ctx.paths.cache / "features")
    parts = []
    for fam in ("quality", "temporal"):
        df = store.load(fam, "train", prefix=True)
        if not df.empty:
            parts.append(df.drop_duplicates(FILE))
    if not parts:
        return None
    tab = parts[0]
    for p in parts[1:]:
        tab = tab.merge(p, on=FILE, how="outer")
    want = ("snr_db", "noise_floor_db", "bandlimit_hz", "hf_ratio", "speech_ratio", "trailing_silence_s", "onset_latency_s")
    cols = [c for c in tab.columns if c != FILE and c.split("::")[-1] in want]
    if not cols:
        return None
    p = ConditionalAcousticProvider(ctx, "REC_conditions", tab, cols, prefix="REC", controls=("item",),
                                    stats=("mean", "sd"), description="recording conditions control", version="rec-1")
    return CachedProvider(p, root=ctx.paths.v3_cache / "fold_providers")


TRANSFER_SCHEMES = ("repeated_kfold", "school_kfold", "loso", "instrument_transfer", "sample_rate_transfer")


def transfer_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    r = rows[rows["scheme"].isin(TRANSFER_SCHEMES)]
    piv = r.pivot_table(index=["experiment", "name"], columns="scheme", values="delta_r2", aggfunc="first")
    piv["n_schemes"] = piv.notna().sum(1)
    piv["n_positive"] = (piv[[c for c in TRANSFER_SCHEMES if c in piv]] > 0).sum(1)
    piv["sign_consistent"] = piv["n_positive"].isin([0]) | (piv["n_positive"] == piv["n_schemes"])
    st = r.pivot_table(index=["experiment", "name"], columns="scheme", values="status", aggfunc="first")
    st.columns = [f"status_{c}" for c in st.columns]
    return piv.join(st).reset_index()


def selection_stability(ev, spec, scheme: str, top_k: int = 20) -> tuple[pd.DataFrame, dict]:
    from ..representations.compact_features import CompactReducer
    from ..v3.common import folds
    from .nested_asr_cv import _inner_oof_residual
    from ..models.preprocessing import TabularPreprocessor

    sel = []
    for f in folds(ev.ctx, scheme):
        Xtr, _, base_cols, groups = ev.fold_frames(f, spec)
        y = ev.y.reindex(Xtr.index).to_numpy(float)
        prep = TabularPreprocessor().fit(Xtr[base_cols])
        resid = _inner_oof_residual(prep.transform(Xtr[base_cols]), y, ev.ctx.seed + f.fold)
        added = [c for cols in groups.values() for c in cols]
        if not added:
            continue
        r = CompactReducer("select", top_k).fit(Xtr[added], resid)
        sel.append({r.in_cols_[i] for i in getattr(r, "idx_", [])})
    if not sel:
        return pd.DataFrame(), {"jaccard_mean": np.nan, "n_folds": 0}
    allc = sorted(set().union(*sel))
    freq = pd.DataFrame({"feature": allc, "selection_freq": [np.mean([c in s for s in sel]) for c in allc]})
    jac = [len(a & b) / max(len(a | b), 1) for i, a in enumerate(sel) for b in sel[i + 1:]]
    freq["family"] = freq["feature"].str.split("__").str[0]
    return freq.sort_values("selection_freq", ascending=False), {"jaccard_mean": float(np.mean(jac)) if jac else np.nan,
                                                                  "n_folds": len(sel)}


def confounding_table(ev, providers: dict, resid: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    from ..stats import partial_corr, spearman

    ids = sorted(resid.index)
    m = meta.reindex(ids)
    sr_low = (m["original_sample_rate"].astype(float) < 16000).astype(float) if "original_sample_rate" in m else pd.Series(np.nan, index=ids)
    ins = pd.get_dummies(m["instrument"].astype(str), drop_first=True, dtype=float) if "instrument" in m else pd.DataFrame(index=ids)
    Z = pd.concat([sr_low.rename("sr_low"), ins], axis=1).reindex(ids)
    rows = []
    for key, p in providers.items():
        T = p.table(ids, ids)                          # descriptive (all training students): association only
        for c in T.columns:
            x = T[c].reindex(ids)
            if x.notna().sum() < 30 or x.std() == 0:
                continue
            r_sr, _, _ = spearman(x, sr_low)
            r_res, p_res, n = spearman(x, resid.reindex(ids))
            pr, pp, _ = partial_corr(x.to_numpy(float), resid.reindex(ids).to_numpy(float), Z.to_numpy(float))
            rows.append({"provider": key, "feature": c, "rho_sample_rate": r_sr, "rho_residual": r_res, "p_residual": p_res,
                         "partial_rho_residual": pr, "partial_p": pp, "n": n,
                         "attenuation": (abs(r_res) - abs(pr)) / abs(r_res) if (np.isfinite(r_res) and np.isfinite(pr) and abs(r_res) > 1e-9) else np.nan})
    return pd.DataFrame(rows)


def subgroups(oof: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    from ..validation.metrics import subgroup_metrics

    m = meta.reset_index().copy()
    if "original_sample_rate" in m:
        m["sample_rate_group"] = np.where(m["original_sample_rate"].astype(float) < 16000, "low_rate", "16k")
    cols = [c for c in ("instrument", "sample_rate_group") if c in m]
    return subgroup_metrics(oof, m, cols) if cols else pd.DataFrame()
