"""Student-level soft-correctness features from nested cross-fitted P(correct | audio, expected text, task).

Per student x task (prefix e.g. SOFT):
  p_mean                 mean soft correctness
  p_correct_mean         confidence on officially correct responses
  p_incorrect_mean       confidence on officially incorrect responses (near-correct proxy level)
  discrepancy            mean |official - p|
  brier                  mean (official - p)^2
  uncertainty            mean p(1-p)
  near_correct_frac      share of incorrect responses with p > 0.5
  doubtful_correct_frac  share of correct responses with p < 0.5
  conf_difficulty_slope  slope of logit(p) on fold-fitted item difficulty (shrunk)
  consistency_sd         SD of (official - p) within the task
  audio_lift (+ _correct / _incorrect)
                         mean logit(p_audio) - logit(p_meta): what the audio adds beyond item metadata
These are learned summaries, NOT verified pronunciation-quality measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import FoldFeatureProvider
from .calibration import calibrator_factory
from .classifiers import ClipFeatureBank, ClipPipeline, ItemEncoder, logit
from .crossfit import nested_predictions

SOFT_VALUE_FEATURES = ("p_mean", "p_correct_mean", "p_incorrect_mean", "discrepancy", "brier", "uncertainty",
                       "near_correct_frac", "doubtful_correct_frac", "conf_difficulty_slope", "consistency_sd")


def soft_student_features(c: pd.DataFrame, p: np.ndarray, d_item: np.ndarray, prefix: str,
                          p_meta: np.ndarray | None = None, shrink_k: float = 5.0) -> pd.DataFrame:
    y = c["item_score"].to_numpy(float)
    p = np.asarray(p, float)
    df = pd.DataFrame({ID: c[ID].astype(str).to_numpy(), "task": c["task"].to_numpy(), "y": y, "p": p,
                       "d": np.asarray(d_item, float), "lp": logit(p)})
    df = df[np.isfinite(df["p"])]
    sc = df[np.isfinite(df["y"])]
    keys = [ID, "task"]
    out = pd.DataFrame({"p_mean": df.groupby(keys)["p"].mean()})
    out["p_correct_mean"] = sc[sc["y"] == 1].groupby(keys)["p"].mean()
    out["p_incorrect_mean"] = sc[sc["y"] == 0].groupby(keys)["p"].mean()
    e = sc.assign(ae=(sc["y"] - sc["p"]).abs(), se=(sc["y"] - sc["p"]) ** 2, r=sc["y"] - sc["p"])
    out["discrepancy"] = e.groupby(keys)["ae"].mean()
    out["brier"] = e.groupby(keys)["se"].mean()
    out["consistency_sd"] = e.groupby(keys)["r"].std(ddof=0)
    out["uncertainty"] = df.assign(u=df["p"] * (1 - df["p"])).groupby(keys)["u"].mean()
    inc, cor = sc[sc["y"] == 0], sc[sc["y"] == 1]
    out["near_correct_frac"] = inc.assign(f=(inc["p"] > 0.5).astype(float)).groupby(keys)["f"].mean()
    out["doubtful_correct_frac"] = cor.assign(f=(cor["p"] < 0.5).astype(float)).groupby(keys)["f"].mean()
    g = df.assign(dd=df["d"] * df["lp"], d2=df["d"] ** 2).groupby(keys)
    n = g["p"].count()
    sd_, sl_, sdl, sdd = g["d"].sum(), g["lp"].sum(), g["dd"].sum(), g["d2"].sum()
    cov = sdl - sd_ * sl_ / n
    var = sdd - sd_ ** 2 / n
    out["conf_difficulty_slope"] = ((cov / var.where(var > 1e-9)) * (n / (n + shrink_k))).where(n >= 3)
    if p_meta is not None:
        lm = logit(np.asarray(p_meta, float))
        dl = pd.DataFrame({ID: c[ID].astype(str).to_numpy(), "task": c["task"].to_numpy(), "y": y,
                           "lift": logit(p) - lm}).dropna(subset=["lift"])
        out["audio_lift"] = dl.groupby(keys)["lift"].mean()
        out["audio_lift_correct"] = dl[dl["y"] == 1].groupby(keys)["lift"].mean()
        out["audio_lift_incorrect"] = dl[dl["y"] == 0].groupby(keys)["lift"].mean()
    w = out.unstack("task")
    w.columns = [f"{prefix}__{t}__{f}__value" for f, t in w.columns]
    a = df.groupby(ID).agg(p_mean=("p", "mean"))
    a["discrepancy"] = e.groupby(ID)["ae"].mean()
    a.columns = [f"{prefix}__all__{f}__value" for f in a.columns]
    res = w.join(a, how="outer")
    res.index = res.index.astype(str)
    res.index.name = ID
    return res.replace([np.inf, -np.inf], np.nan)


class SoftCorrectnessProvider(FoldFeatureProvider):
    """Nested cross-fitted clip probabilities -> soft-correctness student features (one outer fold per call)."""

    family = "SOFT"

    def __init__(self, ctx, bank: ClipFeatureBank, feature_set: str, model: str, prefix: str = "SOFT",
                 with_lift: bool = True, inner_splits: int | None = None, save_clip_dir=None):
        super().__init__(ctx, f"{prefix}_{feature_set}_{model}", f"soft correctness ({feature_set}, {model})")
        self.family = prefix
        self.bank, self.set, self.model = bank, feature_set, model
        self.with_lift = with_lift and feature_set != "META"
        self.inner = int(inner_splits or ctx.cfg.get_path("v2.validation.inner_splits", 4))
        self.cal = ctx.cfg.get_path("v2.correctness.calibration", "isotonic")
        self.save_clip_dir = save_clip_dir

    def available(self) -> bool:
        return self.bank.set_available(self.set)

    def fingerprint_payload(self) -> dict:
        from .classifiers import FEATURE_SETS

        blocks = FEATURE_SETS[self.set]["blocks"]
        return {"set": self.set, "model": self.model, "inner": self.inner, "cal": self.cal, "lift": self.with_lift,
                "bank": self.bank.fingerprint(blocks), "cfg": {k: self.ctx.cfg.get_path(f"v2.correctness.{k}")
                                                               for k in ("lgbm", "mlp", "embedding_pca", "include_screening_theta")}}

    def _make(self, feature_set: str):
        cfg, ctx = self.ctx.cfg, self.ctx

        def make():
            return ClipPipeline(self.bank, feature_set, self.model, cfg, seed=ctx.seed, device="cpu",
                                embedding_pca=int(cfg.get_path("v2.correctness.embedding_pca", 32)),
                                include_theta=bool(cfg.get_path("v2.correctness.include_screening_theta", False)))
        return make

    def clip_predictions(self, train_ids, target_ids) -> pd.DataFrame:
        from ..v2.validation_registry import inner_student_folds
        from ..io_utils import stable_hash

        c = self.bank.c
        ids = set(train_ids) | set(target_ids)
        rows = np.flatnonzero(c[ID].astype(str).isin(ids).to_numpy())
        inner = inner_student_folds(sorted(train_ids), self.inner, int(int(stable_hash(sorted(train_ids)), 16) % (2 ** 31 - 1)))
        cal = calibrator_factory(self.cal)
        if self.set == "META":
            return self._meta_predictions(c, train_ids, rows, inner, cal)
        pa = nested_predictions(c, train_ids, rows, self._make(self.set), inner, cal)
        if self.with_lift:
            pm = self._meta_predictions(c, train_ids, rows, inner, cal)
            pa["p_meta"] = pm.set_index(FILE)["p"].reindex(pa[FILE].to_numpy()).to_numpy()
        return pa

    def _meta_predictions(self, c, train_ids, rows, inner, cal) -> pd.DataFrame:
        """Metadata-only nested predictions for the same fold, shared on disk by every audio feature set."""
        from ..io_utils import atomic_write_parquet, stable_hash

        key = stable_hash({"bank": self.bank.fingerprint([]), "model": self.model, "inner": self.inner, "cal": self.cal,
                           "tr": sorted(map(str, train_ids)), "rows": stable_hash(c[FILE].to_numpy()[rows].tolist())})
        d = self.ctx.paths.v2_cache / "soft_meta"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{key}.parquet"
        if path.exists():
            pm = pd.read_parquet(path)
        else:
            pm = nested_predictions(c, train_ids, rows, self._make("META"), inner, cal)
            atomic_write_parquet(pm, path)
        return pm

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        c_all = self.bank.c
        pa = self.clip_predictions(train_ids, target_ids)
        if self.save_clip_dir is not None:
            from ..io_utils import atomic_write_parquet, stable_hash

            atomic_write_parquet(pa, self.save_clip_dir / f"{self.name}__{stable_hash(sorted(train_ids))}.parquet")
        c = c_all.set_index(FILE).loc[pa[FILE]].reset_index()
        enc = ItemEncoder().fit(c_all[c_all[ID].astype(str).isin(set(train_ids))])
        d = enc.difficulty(c, loso=False)
        T = soft_student_features(c, pa["p"].to_numpy(), d, self.family, pa["p_meta"].to_numpy() if "p_meta" in pa else None)
        return T.reindex(sorted(set(train_ids) | set(target_ids)))
