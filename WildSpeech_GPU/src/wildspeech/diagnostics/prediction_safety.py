"""Deployability checks, quarantine, and the diagnosis of V1's catastrophic task ensemble (P11).

PredictionSafetyChecker.check(...) -> one row per check (PASS / FAIL / WARN):
  finite_oof, oof_complete, extreme_fraction (|pred - mean_y| > max_abs_z * sd_y), quantile_range
  (prediction 0.5% / 99.5% quantiles inside [min_y - m*sd, max_y + m*sd]), min_fold_r2, repeat_sd,
  and (for test predictions) finite_test, test_quantile_range, test_extreme_fraction, non_constant,
  artifact_consistency (upstream fingerprints recorded at training == current fingerprints).
A model is deployable only if no check FAILs; otherwise it is quarantined with its reasons.

TaskEnsembleDiagnosis re-runs V1's TASK_ENSEMBLE fold by fold with instrumentation: inner-OOF member ranges,
NNLS weights (unbounded, no sum constraint), group condition numbers, the most extreme validation predictions,
the features with the largest standardised extrapolation |z| for those students, and missing-task status.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID
from ..stats import r2


class PredictionSafetyChecker:
    def __init__(self, y_train: np.ndarray, cfg):
        s = cfg.get_path("v2.failure.safety", {}) or {}
        y = np.asarray(y_train, float)
        self.mu, self.sd, self.lo_y, self.hi_y = float(y.mean()), float(y.std()), float(y.min()), float(y.max())
        self.max_z = float(s.get("max_abs_z", 6.0))
        self.max_extreme = float(s.get("max_extreme_frac", 0.0))
        self.margin = float(s.get("range_margin_sd", 1.0))
        self.min_fold_r2 = float(s.get("min_fold_r2", -0.10))
        self.max_repeat_sd = float(s.get("max_repeat_sd", 0.05))

    def _range(self):
        return self.lo_y - self.margin * self.sd, self.hi_y + self.margin * self.sd

    def _pred_checks(self, p: np.ndarray, tag: str) -> list[dict]:
        p = np.asarray(p, float)
        out = []
        fin = np.isfinite(p)
        out.append({"check": f"finite_{tag}", "status": "PASS" if fin.all() else "FAIL", "value": float(1 - fin.mean())})
        if fin.any():
            z = np.abs(p[fin] - self.mu) / max(self.sd, 1e-9)
            frac = float((z > self.max_z).mean())
            out.append({"check": f"extreme_fraction_{tag}", "status": "PASS" if frac <= self.max_extreme else "FAIL", "value": frac})
            lo, hi = self._range()
            q = np.quantile(p[fin], [0.005, 0.995])
            ok = (q[0] >= lo) and (q[1] <= hi)
            out.append({"check": f"quantile_range_{tag}", "status": "PASS" if ok else "FAIL", "value": f"{q[0]:.2f}..{q[1]:.2f} vs {lo:.2f}..{hi:.2f}"})
            out.append({"check": f"non_constant_{tag}", "status": "PASS" if np.nanstd(p) > 1e-9 else "FAIL", "value": float(np.nanstd(p))})
        return out

    def check(self, oof: pd.DataFrame, fold_metrics: pd.DataFrame | None, n_students: int,
              test_pred: np.ndarray | None = None, lineage_ok: bool | None = None) -> pd.DataFrame:
        rows = []
        if oof is None or oof.empty:
            return pd.DataFrame([{"check": "oof_available", "status": "FAIL", "value": 0}])
        rows += self._pred_checks(oof["y_pred"].to_numpy(float), "oof")
        cov = oof[ID].nunique() / max(n_students, 1)
        rows.append({"check": "oof_complete", "status": "PASS" if cov >= 0.999 else "FAIL", "value": float(cov)})
        if fold_metrics is not None and not fold_metrics.empty and "r2" in fold_metrics:
            mn = float(fold_metrics["r2"].min())
            rows.append({"check": "min_fold_r2", "status": "PASS" if mn >= self.min_fold_r2 else "FAIL", "value": mn})
        if "repeat" in oof and oof["repeat"].nunique() > 1:
            per = oof.groupby("repeat").apply(lambda d: r2(d["y_true"], d["y_pred"]))
            sd = float(per.std(ddof=0))
            rows.append({"check": "repeat_sd", "status": "PASS" if sd <= self.max_repeat_sd else "WARN", "value": sd})
        if test_pred is not None:
            rows += self._pred_checks(test_pred, "test")
        if lineage_ok is not None:
            rows.append({"check": "artifact_consistency", "status": "PASS" if lineage_ok else "FAIL", "value": bool(lineage_ok)})
        return pd.DataFrame(rows)

    @staticmethod
    def deployable(checks: pd.DataFrame) -> tuple[bool, list[str]]:
        bad = checks.loc[checks["status"] == "FAIL", "check"].tolist()
        return (len(bad) == 0), bad


class TaskEnsembleDiagnosis:
    def __init__(self, ctx, runner, scheme: str = "repeated_kfold", max_folds: int | None = None, top: int = 15):
        self.ctx, self.runner, self.scheme, self.max_folds, self.top = ctx, runner, scheme, max_folds, top

    def run(self) -> dict[str, pd.DataFrame]:
        from scipy.optimize import nnls
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold

        from ..models.fusion import ALPHAS, _groups_by_task
        from ..models.preprocessing import TabularPreprocessor

        tasks = sorted(self.ctx.bundle().all_clips["task"].unique())
        fold_rows, member_rows, extreme_rows, feat_rows = [], [], [], []
        folds = self.runner.folds(self.scheme)
        for f in folds[: self.max_folds] if self.max_folds else folds:
            Xtr, Xva = self.runner.fold_data(f)
            ytr = self.runner.y.reindex(Xtr.index).to_numpy(float)
            yva = self.runner.y.reindex(Xva.index).to_numpy(float)
            prep = TabularPreprocessor().fit(Xtr)
            Ptr, Pva = prep.transform(Xtr), prep.transform(Xva)
            groups = {k: v for k, v in _groups_by_task(Ptr.columns, tasks).items() if v}
            names = list(groups)
            Z = np.zeros((len(ytr), len(names)))
            for tr, va in KFold(int(self.ctx.cfg.get_path("validation.inner_splits", 4)), shuffle=True, random_state=self.ctx.seed).split(Ptr):
                for j, g in enumerate(names):
                    Z[va, j] = RidgeCV(alphas=ALPHAS).fit(Ptr.iloc[tr][groups[g]], ytr[tr]).predict(Ptr.iloc[va][groups[g]])
            Zc = Z - Z.mean(0)
            w, _ = nnls(Zc, ytr - ytr.mean())
            if w.sum() <= 0:
                w = np.zeros(len(names))
                w[0] = 1.0
            b = float(ytr.mean() - Z.mean(0) @ w)
            preds = {}
            for j, g in enumerate(names):
                m = RidgeCV(alphas=ALPHAS).fit(Ptr[groups[g]], ytr)
                preds[g] = m.predict(Pva[groups[g]])
                s = np.linalg.svd(Ptr[groups[g]].to_numpy(float), compute_uv=False)
                member_rows.append({"repeat": f.repeat, "fold": f.fold, "group": g, "n_cols": len(groups[g]), "weight": float(w[j]),
                                    "alpha": float(m.alpha_), "cond_number": float(s[0] / max(s[-1], 1e-12)),
                                    "inner_oof_min": float(Z[:, j].min()), "inner_oof_max": float(Z[:, j].max()),
                                    "valid_pred_min": float(preds[g].min()), "valid_pred_max": float(preds[g].max()),
                                    "valid_r2_member": r2(yva, preds[g])})
            p = b + sum(w[j] * preds[g] for j, g in enumerate(names))
            fold_rows.append({"repeat": f.repeat, "fold": f.fold, "r2": r2(yva, p), "weight_sum": float(w.sum()),
                              "intercept": b, "pred_min": float(p.min()), "pred_max": float(p.max()),
                              "y_min": float(yva.min()), "y_max": float(yva.max())})
            z_va = Pva.abs()
            order = np.argsort(-np.abs(p - ytr.mean()))[: self.top]
            for i in order:
                sid = Xva.index[i]
                miss = {t: bool(not np.isfinite(Xva.iloc[i].get(f"BEHAVIOR__{t}__n_items__value", np.nan)) or
                                Xva.iloc[i].get(f"BEHAVIOR__{t}__n_items__value", 0) == 0) for t in tasks}
                extreme_rows.append({"repeat": f.repeat, "fold": f.fold, ID: sid, "y_true": float(yva[i]), "y_pred": float(p[i]),
                                     **{f"pred_{g}": float(preds[g][i]) for g in names},
                                     **{f"missing_{t}": v for t, v in miss.items()}})
                zz = z_va.iloc[i].sort_values(ascending=False).head(5)
                for col, val in zz.items():
                    feat_rows.append({"repeat": f.repeat, "fold": f.fold, ID: sid, "feature": col, "abs_z": float(val),
                                      "raw_value": float(Xva.iloc[i].get(col, np.nan)) if col in Xva.columns else np.nan,
                                      "train_median": float(prep.median_.get(col, np.nan)) if col in prep.median_.index else np.nan})
        return {"te_folds": pd.DataFrame(fold_rows), "te_members": pd.DataFrame(member_rows),
                "te_extreme_students": pd.DataFrame(extreme_rows), "te_extrapolating_features": pd.DataFrame(feat_rows)}


def summarize_task_ensemble_diagnosis(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Turns the instrumentation into evidence rows (derived from the run, nothing assumed)."""
    rows = []
    folds, mem, ext, feat = d["te_folds"], d["te_members"], d["te_extreme_students"], d["te_extrapolating_features"]
    if folds.empty:
        return pd.DataFrame()
    bad = folds[folds["r2"] < -1]
    rows.append({"finding": "folds with R2 < -1", "value": int(len(bad)), "detail": f"of {len(folds)} folds"})
    rows.append({"finding": "max NNLS weight sum", "value": float(folds["weight_sum"].max()),
                 "detail": "V1 weights are non-negative but not constrained to sum to 1"})
    if not mem.empty:
        worst = mem.sort_values("valid_pred_max", ascending=False).iloc[0]
        rows.append({"finding": "largest member validation prediction", "value": float(worst["valid_pred_max"]),
                     "detail": f"group={worst['group']} fold={worst['repeat']}/{worst['fold']} cond={worst['cond_number']:.3g}"})
        rows.append({"finding": "median member condition number", "value": float(mem["cond_number"].median()), "detail": ""})
    if not ext.empty:
        mcols = [c for c in ext.columns if c.startswith("missing_")]
        top = ext.assign(err=(ext["y_pred"] - ext["y_true"]).abs()).sort_values("err", ascending=False).head(20)
        for c in mcols:
            rows.append({"finding": f"share of 20 worst predictions with {c}", "value": float(top[c].mean()), "detail": ""})
    if not feat.empty:
        f = feat.groupby("feature")["abs_z"].max().sort_values(ascending=False).head(5)
        rows.append({"finding": "largest validation |z| (extrapolation)", "value": float(f.iloc[0]),
                     "detail": "; ".join(f"{k}={v:.1f}" for k, v in f.items())})
    return pd.DataFrame(rows)
