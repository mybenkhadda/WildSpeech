"""ASR model disagreement and consensus (P27).

pairwise_disagreement(details, wide, models, g2p) -> clip x model-pair table
    dis_transcript        normalized word edit distance between the two transcripts (both word/char models)
    dis_phones            phone edit distance between the two phone sequences (recognised or G2P-derived)
    dis_word_vs_phone     G2P(word transcript) vs recognised phones (word model x phone model pairs)
    dis_expected_ll       |z(expected_ll_a) - z(expected_ll_b)|   (z per model on training clips)
    dis_confidence        |z(conf_a) - z(conf_b)|
    dis_near_correct      |near_a - near_b| (expected-text agreement of each model)
    both_child            the two models are child-specific
consensus(...) -> per clip: n_models, mean/max disagreement, consensus near-correctness, high_uncertainty
    (max disagreement above the configured quantile AND mean confidence z below 0)
DisagreementProvider (family ASRD, label-free, z-statistics fitted on training-fold clips)
variance_decomposition -> sequential R² of disagreement explained by item, task, recording conditions,
    official correctness and student identity. Disagreement is NEVER labelled a pronunciation difficulty: the
    decomposition shows how much of it is item / recording / model artefact versus stable student behaviour.
split_half_consistency -> odd/even-item student means correlation (Spearman-Brown corrected)
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import FoldFeatureProvider, aggregate_student_task
from ..io_utils import stable_hash
from ..phonetics.alignment import align, phonological_align
from ..phonetics.builder import model_prefix
from ..phonetics.error_analysis import words_of


def _ned(a: list, b: list, phon: bool = False) -> float:
    if not a and not b:
        return 0.0
    ops = phonological_align(a, b) if phon else align(a, b)
    e = sum(o.op != "M" for o in ops)
    return e / max(len(a), len(b), 1)


def _conf_col(wide: pd.DataFrame, m: str) -> str | None:
    pre = model_prefix(m) + "::"
    for n in ("cf_gop_mean", "cf_tok_lp_mean", "cf_post_mean", "cf_expected_ll"):
        if pre + n in wide.columns and wide[pre + n].notna().mean() > 0.3:
            return pre + n
    return None


def _z(s: pd.Series, train_files: set[str], files: pd.Series) -> pd.Series:
    t = s[files.isin(train_files).to_numpy()]
    mu, sd = float(np.nanmean(t)) if len(t) else 0.0, float(np.nanstd(t)) if len(t) else 1.0
    return (s - mu) / (sd if sd > 1e-9 else 1.0)


def pairwise_disagreement(details: pd.DataFrame, wide: pd.DataFrame, outputs: dict[str, str], child: set[str],
                          g2p, train_files: set[str]) -> pd.DataFrame:
    models = sorted(m for m in outputs if (details["model_key"] == m).any())
    D = {m: details[details["model_key"] == m].drop_duplicates(FILE).set_index(FILE) for m in models}
    W = wide.set_index(FILE)
    files = pd.Series(W.index, index=W.index)
    zc, zl, near = {}, {}, {}
    for m in models:
        cc = _conf_col(wide, m)
        if cc:
            zc[m] = _z(W[cc], train_files, files)
        ec = model_prefix(m) + "::cf_expected_ll"
        if ec in W:
            zl[m] = _z(W[ec], train_files, files)
        for n in ("ph_near_correct", "w_completeness"):
            if model_prefix(m) + "::" + n in W:
                near[m] = W[model_prefix(m) + "::" + n]
                break
    rows = []
    for a, b in itertools.combinations(models, 2):
        common = D[a].index.intersection(D[b].index)
        oa, ob = outputs[a], outputs[b]
        for f in common:
            ra, rb = D[a].loc[f], D[b].loc[f]
            ta, tb = words_of(ra.get("transcript", "")), words_of(rb.get("transcript", ""))
            pa, pb = str(ra.get("hyp_phones", "") or "").split(), str(rb.get("hyp_phones", "") or "").split()
            r = {FILE: f, "model_a": a, "model_b": b, "pair": f"{a}|{b}", "both_child": float(a in child and b in child)}
            r["dis_transcript"] = _ned(ta, tb) if (oa != "phones" and ob != "phones") else np.nan
            r["dis_phones"] = _ned(pa, pb, phon=True) if (pa or pb) else np.nan
            if (oa == "phones") != (ob == "phones"):
                words = ta if oa != "phones" else tb
                ph = pb if oa != "phones" else pa
                r["dis_word_vs_phone"] = _ned(g2p.phonemes(" ".join(words)), ph, phon=True) if g2p.usable else np.nan
            else:
                r["dis_word_vs_phone"] = np.nan
            r["dis_expected_ll"] = float(abs(zl[a].get(f, np.nan) - zl[b].get(f, np.nan))) if (a in zl and b in zl) else np.nan
            r["dis_confidence"] = float(abs(zc[a].get(f, np.nan) - zc[b].get(f, np.nan))) if (a in zc and b in zc) else np.nan
            r["dis_near_correct"] = float(abs(near[a].get(f, np.nan) - near[b].get(f, np.nan))) if (a in near and b in near) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


DIS_COLS = ("dis_transcript", "dis_phones", "dis_word_vs_phone", "dis_expected_ll", "dis_confidence", "dis_near_correct")


def consensus(pairs: pd.DataFrame, wide: pd.DataFrame, models: list[str], q: float, train_files: set[str]) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=[FILE])
    p = pairs.copy()
    p["dis_any"] = p[[c for c in DIS_COLS if c in p]].mean(1)
    g = p.groupby(FILE)
    out = pd.DataFrame({"n_pairs": g.size(), "dis_mean": g["dis_any"].mean(), "dis_max": g["dis_any"].max()})
    for c in DIS_COLS:
        if c in p:
            out[c] = g[c].mean()
    W = wide.set_index(FILE)
    files = pd.Series(W.index, index=W.index)
    zs, nears = [], []
    for m in models:
        cc = _conf_col(wide, m)
        if cc:
            zs.append(_z(W[cc], train_files, files).rename(m))
        for n in ("ph_near_correct", "w_completeness"):
            col = model_prefix(m) + "::" + n
            if col in W:
                nears.append(W[col].rename(m))
                break
    out["conf_z_mean"] = pd.concat(zs, axis=1).mean(1).reindex(out.index) if zs else np.nan
    out["consensus_near_correct"] = pd.concat(nears, axis=1).mean(1).reindex(out.index) if nears else np.nan
    tr = out.index.isin(list(train_files))
    thr = float(np.nanquantile(out.loc[tr, "dis_max"], q)) if tr.any() else np.nan
    out["high_uncertainty"] = ((out["dis_max"] > thr) & (out["conf_z_mean"].fillna(0) < 0)).astype(float)
    out.attrs["threshold"] = thr
    return out.reset_index()


class DisagreementProvider(FoldFeatureProvider):
    family = "ASRD"

    def __init__(self, ctx, name: str, cons: pd.DataFrame, q: float):
        super().__init__(ctx, name, "ASR disagreement")
        self.cons = cons.drop_duplicates(FILE)
        self.q = float(q)
        self.cols = [c for c in ("dis_mean", "dis_max", *DIS_COLS) if c in self.cons and self.cons[c].notna().any()]

    def available(self) -> bool:
        return bool(self.cols)

    def fingerprint_payload(self) -> dict:
        return {"cols": self.cols, "q": self.q, "files": stable_hash(sorted(self.cons[FILE].astype(str))), "v": "asrd-1"}

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        from .item_conditioning import v3_clip_frame

        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].isin(ids)].merge(self.cons, on=FILE, how="inner").reset_index(drop=True)
        tr = c[ID].isin(set(map(str, train_ids))).to_numpy()
        V = c[self.cols].to_numpy(float)
        mu = np.nanmean(V[tr], 0)
        sd = np.nanstd(V[tr], 0)
        Z = (V - mu) / np.where(sd > 1e-9, sd, 1.0)
        thr = np.nanquantile(c.loc[tr, "dis_max"], self.q) if tr.any() else np.nan       # training-fold threshold
        hu = ((c["dis_max"] > thr) & (c["conf_z_mean"].fillna(0) < 0)).astype(float).to_numpy()[:, None]
        out = aggregate_student_task(c, np.hstack([Z, hu]), self.cols + ["high_uncertainty"], "ASRD",
                                     stats=("mean", "sd"))
        return out.reindex(sorted(ids))


def variance_decomposition(cons: pd.DataFrame, clipf: pd.DataFrame, target: str = "dis_mean",
                           recording_cols: list[str] | None = None) -> pd.DataFrame:
    """Sequential (type-I) R² of `target`: item -> recording -> correctness -> student, plus each block alone."""
    from sklearn.linear_model import LinearRegression

    d = clipf.merge(cons[[FILE, target]], on=FILE, how="inner").dropna(subset=[target])
    d = d[d["split"] == "train"].reset_index(drop=True)
    if len(d) < 50:
        return pd.DataFrame()
    y = d[target].to_numpy(float)
    rec = [c for c in (recording_cols or ["orig_sr_low", "log_dur"]) if c in d]
    blocks = {
        "item": pd.get_dummies(d["item_id"].astype(str), drop_first=True, dtype=float),
        "recording": pd.concat([d[rec].astype(float).fillna(d[rec].astype(float).median()),
                                pd.get_dummies(d["instrument"].astype(str), drop_first=True, dtype=float)], axis=1),
        "correctness": pd.DataFrame({"score": d["item_score"].fillna(0.5).astype(float)}),
        "student": pd.get_dummies(d[ID].astype(str), drop_first=True, dtype=float),
    }

    def r2(X):
        if X.shape[1] == 0:
            return 0.0
        return float(LinearRegression().fit(X, y).score(X, y))

    rows, cur, prev = [], pd.DataFrame(index=d.index), 0.0
    for name, B in blocks.items():
        cur = pd.concat([cur, B], axis=1)
        tot = r2(cur.to_numpy(float))
        rows.append({"target": target, "block": name, "r2_alone": r2(B.to_numpy(float)), "r2_cumulative": tot,
                     "r2_increment": tot - prev, "n_params": B.shape[1], "n": len(d)})
        prev = tot
    return pd.DataFrame(rows)


def split_half_consistency(cons: pd.DataFrame, clipf: pd.DataFrame, col: str = "dis_mean") -> dict:
    from ..stats import spearman

    d = clipf.merge(cons[[FILE, col]], on=FILE, how="inner").dropna(subset=[col])
    d = d[d["split"] == "train"].sort_values([ID, "item_id"])
    d["half"] = d.groupby(ID).cumcount() % 2
    piv = d.groupby([ID, "half"])[col].mean().unstack()
    if piv.shape[1] < 2:
        return {"feature": col, "r_half": np.nan, "r_sb": np.nan, "n": 0}
    r, p, n = spearman(piv[0], piv[1])
    return {"feature": col, "r_half": r, "r_sb": 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan, "p": p, "n": n}
