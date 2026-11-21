"""Correct-response speech intelligence (P26) — the central V3 experiment.

Question: among children who answered the SAME item correctly, do phonetic confidence, hesitation,
pronunciation consistency and speech representations explain different reading outcomes?

CorrectResponseProvider  (FoldFeatureProvider, family ASRO) — every reference statistic fitted on training-fold
clips only (ItemReference: item means / task SDs / domain-duration-complexity coefficients / difficulty / ability
strata). Per student x task (+ 'all'):
    corr_conf_mean        item-adjusted confidence on CORRECT clips (mean z)
    corr_conf_sd          its variability across correct items
    inc_near_correct      item-adjusted phonetic near-correctness on INCORRECT clips
    conf_corr_discrepancy mean z(conf | correct) - mean z(conf | incorrect)   (does confidence track correctness?)
    adj_per               item-adjusted phoneme error rate (all scored clips)
    cluster_consistency   SD of consonant-cluster accuracy across items with clusters (lower = more consistent)
    conf_difficulty_slope slope of standardized confidence on fold-fitted item difficulty (shrunk to pooled)
    error_recovery        share of errorful clips that contain a self-correction (word models)
    stability             1 - within-student SD of correct-clip z(conf) / pooled training SD
    asym_<t1>_<t2>        cross-task phonological asymmetry of corr_conf_mean
    n_correct             number of correct clips used (students below min_correct_clips get NaN features)

ClipCorrectnessIncrement — does phonetics predict official clip correctness beyond META (item, task, duration,
complexity, domain) and TEMPORAL (V1 temporal acoustics)? Student-grouped CV, AUROC, student-block bootstrap of
ΔAUROC. Fold-fitted item difficulty enters META via inner fitting on the training clips of each split.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import FoldFeatureProvider
from ..io_utils import stable_hash
from .item_conditioning import ItemReference, v3_clip_frame

ROLE_CANDIDATES = {
    "conf": ("cf_gop_mean", "cf_tok_lp_mean", "cf_expected_ll", "cf_post_mean"),
    "near": ("ph_near_correct", "w_completeness"),
    "per": ("ph_per", "w_wer"),
    "cluster": ("ph_cluster_acc",),
    "selfcorr": ("w_self_corrections",),
    "err": ("ph_per", "w_wer"),
}


def pick_role_columns(wide: pd.DataFrame, model_key: str) -> dict[str, str]:
    """First available column per role for one model (with enough non-missing values)."""
    from ..phonetics.builder import model_prefix

    pre = model_prefix(model_key) + "::"
    out = {}
    for role, names in ROLE_CANDIDATES.items():
        for n in names:
            c = pre + n
            if c in wide.columns and wide[c].notna().mean() > 0.3:
                out[role] = c
                break
    return out


class CorrectResponseProvider(FoldFeatureProvider):
    family = "ASRO"

    def __init__(self, ctx, name: str, wide: pd.DataFrame, roles: dict[str, str], model_key: str):
        super().__init__(ctx, name, f"correct-response intelligence from {model_key}")
        d = ctx.cfg.get_path("v3.discovery", {}) or {}
        self.roles = dict(roles)
        self.model_key = model_key
        self.cols = sorted(set(self.roles.values()))
        self.wide = wide[[FILE] + self.cols].drop_duplicates(FILE)
        self.controls = tuple(d.get("controls", ["item", "domain", "duration", "complexity"]))
        self.k = float(d.get("shrinkage_k", 10.0))
        self.min_n = int(d.get("min_correct_clips_per_student", 3))
        self.bins = int(d.get("similar_ability_bins", 5))

    def available(self) -> bool:
        return "conf" in self.roles and len(self.wide) > 0

    def fingerprint_payload(self) -> dict:
        return {"roles": self.roles, "controls": self.controls, "k": self.k, "min_n": self.min_n,
                "files": stable_hash(sorted(self.wide[FILE].astype(str))), "v": "asro-1"}

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = v3_clip_frame(self.ctx)
        c = c[c[ID].isin(ids)].merge(self.wide, on=FILE, how="inner").reset_index(drop=True)
        if c.empty:
            return pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        F = c[self.cols].to_numpy(float)
        tr = c[ID].isin(set(map(str, train_ids))).to_numpy()
        ref = ItemReference(self.controls, self.k, self.bins).fit(c[tr], F[tr])
        Z, diff, _ = ref.transform(c, F)
        Zc = pd.DataFrame(Z, columns=self.cols, index=c.index)
        score = c["item_score"].astype(float)
        conf = self.roles["conf"]
        # pooled training statistics (for standardisation / stability / slope shrinkage)
        trc = tr & (score.to_numpy() == 1)
        pooled_sd = float(np.nanstd(Zc.loc[trc, conf])) or 1.0
        raw = c[conf].astype(float)
        mu, sd = float(np.nanmean(raw[tr])), float(np.nanstd(raw[tr])) or 1.0
        cz = (raw - mu) / sd
        dd = pd.DataFrame({"sid": c[ID].astype(str), "task": c["task"], "score": score, "zc": Zc[conf], "cz": cz,
                           "diff": diff})
        for role in ("near", "per", "cluster", "selfcorr", "err"):
            if role in self.roles:
                col = self.roles[role]
                dd[role] = Zc[col] if role in ("near", "per") else c[col].astype(float)
        pooled = self._slope(dd[tr & dd["cz"].notna().to_numpy()])
        rows = []
        for (sid, task), g in dd.groupby(["sid", "task"]):
            rows.append({"sid": sid, "task": task, **self._summ(g, pooled, pooled_sd)})
        for sid, g in dd.groupby("sid"):
            rows.append({"sid": sid, "task": "all", **self._summ(g, pooled, pooled_sd)})
        T = pd.DataFrame(rows)
        wide = T.pivot(index="sid", columns="task")
        wide.columns = [f"ASRO__{t}__{f}__value" for f, t in wide.columns]
        wide = wide.loc[:, wide.notna().any()]
        tasks = sorted(t for t in dd["task"].unique())
        base = T[T["task"] != "all"].pivot(index="sid", columns="task", values="corr_conf_mean")
        for a, b in itertools.combinations(tasks, 2):
            if a in base and b in base:
                wide[f"ASRO__all__asym_{a}_{b}__value"] = base[a] - base[b]
        wide.index = wide.index.astype(str)
        wide.index.name = ID
        return wide.reindex(sorted(ids)).replace([np.inf, -np.inf], np.nan)

    def _slope(self, g: pd.DataFrame) -> float:
        x, y = g["diff"].to_numpy(float), g["cz"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3 or np.std(x[m]) < 1e-9:
            return np.nan
        return float(np.polyfit(x[m], y[m], 1)[0])

    def _summ(self, g: pd.DataFrame, pooled_slope: float, pooled_sd: float) -> dict:
        cor, inc = g[g["score"] == 1], g[g["score"] == 0]
        n_c = int(cor["zc"].notna().sum())
        ok = n_c >= self.min_n
        out = {"n_correct": float(n_c),
               "corr_conf_mean": float(cor["zc"].mean()) if ok else np.nan,
               "corr_conf_sd": float(cor["zc"].std(ddof=0)) if ok else np.nan,
               "stability": float(1 - cor["zc"].std(ddof=0) / pooled_sd) if ok else np.nan,
               "inc_near_correct": float(inc["near"].mean()) if ("near" in g and inc["near"].notna().sum() >= 2) else np.nan,
               "conf_corr_discrepancy": float(cor["zc"].mean() - inc["zc"].mean()) if (ok and inc["zc"].notna().sum() >= 2) else np.nan,
               "adj_per": float(g["per"].mean()) if "per" in g else np.nan,
               "cluster_consistency": float(g["cluster"].std(ddof=0)) if ("cluster" in g and g["cluster"].notna().sum() >= 3) else np.nan}
        s = self._slope(g)
        n = int(g["cz"].notna().sum())
        if np.isfinite(s) and np.isfinite(pooled_slope):
            w = n / (n + 10.0)
            out["conf_difficulty_slope"] = float(w * s + (1 - w) * pooled_slope)
        else:
            out["conf_difficulty_slope"] = np.nan
        if "selfcorr" in g and "err" in g:
            e = g[g["err"] > 0]
            out["error_recovery"] = float((e["selfcorr"] > 0).mean()) if len(e) >= 2 else np.nan
        return out


# ============================================================================ clip correctness
class ClipCorrectnessIncrement:
    """item_score ~ META | META+TEMPORAL | META+PHON | META+TEMPORAL+PHON, student-grouped K-fold on training clips.
    PHON uses only label-free P25 features. Item difficulty is fitted inside each training split."""

    def __init__(self, ctx, n_splits: int = 5, models=("LOGISTIC",), seed: int = 0):
        self.ctx, self.n_splits, self.models, self.seed = ctx, int(n_splits), tuple(models), int(seed)

    @staticmethod
    def _estimator(name: str, seed: int):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if name == "LIGHTGBM":
            try:
                from lightgbm import LGBMClassifier

                return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                                      random_state=seed, verbose=-1)
            except ImportError:
                pass
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))

    def run(self, phon: pd.DataFrame, phon_cols: list[str], temporal: pd.DataFrame | None, temporal_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import GroupKFold

        c = v3_clip_frame(self.ctx)
        c = c[(c["split"] == "train") & (c["is_scored"] == 1)].merge(phon[[FILE] + phon_cols], on=FILE, how="inner")
        if temporal is not None and temporal_cols:
            c = c.merge(temporal[[FILE] + temporal_cols], on=FILE, how="left")
        c = c.reset_index(drop=True)
        if c.empty:
            return pd.DataFrame(), pd.DataFrame()
        meta_cols = ["log_dur", "orig_sr_low"] + [k for k in ("n_phonemes", "n_syllables", "n_words", "max_cluster") if k in c]
        tasks = sorted(c["task"].unique())
        for t in tasks:
            c[f"task_{t}"] = (c["task"] == t).astype(float)
        meta_cols += [f"task_{t}" for t in tasks]
        sets = {"META": meta_cols + ["difficulty"], "META+PHON": meta_cols + ["difficulty"] + phon_cols}
        if temporal_cols:
            sets["META+TEMPORAL"] = meta_cols + ["difficulty"] + temporal_cols
            sets["META+TEMPORAL+PHON"] = meta_cols + ["difficulty"] + temporal_cols + phon_cols
        y = c["item_score"].astype(int).to_numpy()
        groups = c[ID].astype(str).to_numpy()
        oof = {(m, s): np.full(len(c), np.nan) for m in self.models for s in sets}
        for tr, va in GroupKFold(self.n_splits).split(c, y, groups):
            ref = ItemReference(("item",)).fit(c.iloc[tr])
            c.loc[c.index[tr], "difficulty"] = ref.difficulty(c.iloc[tr])
            c.loc[c.index[va], "difficulty"] = ref.difficulty(c.iloc[va])
            for m in self.models:
                for s, cols in sets.items():
                    X = c[cols].to_numpy(float)
                    est = self._estimator(m, self.seed).fit(X[tr], y[tr])
                    oof[(m, s)][va] = est.predict_proba(X[va])[:, 1]
        P = c[[FILE, ID, "task", "item_id", "item_score"]].copy()
        for (m, s), v in oof.items():
            P[f"p__{m}__{s}"] = v
        return P, pd.DataFrame([{"model": m, "set": s, "n_features": len(sets[s])} for m in self.models for s in sets])
