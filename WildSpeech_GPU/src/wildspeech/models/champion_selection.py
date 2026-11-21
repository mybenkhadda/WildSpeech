"""V2 candidate selection (deployable models only) and validated candidate submissions.

Candidate roles (v2.failure.candidates):
  conservative_screening  ridge on SCREENING + RECORDING
  best_tabular            the strongest validated tabular comparator
  harmonized_acoustic     best model whose new representation is HARM only
  phonetic_multimodal     best model containing PHON (or REF / SOFT built on phonetics); absent if P18 did not run
  residual_fusion         best fusion='residual' model
  final_ensemble          equal-weight average of the deployable non-baseline candidates whose paired delta-R2
                          CI over the comparator is > 0 (no fitted weights -> no extra selection optimism);
                          falls back to best_tabular when none qualify
Ranking score inside a role = r2_mean - r2_sd (robust), among candidates that passed every safety check.
Submissions: fit on ALL training students -> save the fitted model (joblib) -> RE-LOAD it from disk -> predict the
test students -> safety checks on test predictions -> validate the CSV against submission_format -> write.
"""
from __future__ import annotations

import hashlib
import time

import joblib
import numpy as np
import pandas as pd

from .. import ID, TARGET
from ..io_utils import atomic_write_csv, write_json
from ..submission import validate_submission


def _robust(df: pd.DataFrame) -> pd.Series:
    return df["r2_mean"].astype(float) - df["r2_sd"].fillna(0).astype(float)


def select_candidates(results: pd.DataFrame, specs: dict, deployable: dict[str, bool]) -> dict[str, str | None]:
    """results: P21 repeated_kfold rows (name, r2_mean, r2_sd, delta_ci_low, providers, fusion)."""
    r = results[results["name"].map(lambda n: deployable.get(n, False))].copy()
    r["robust"] = _robust(r) if not r.empty else []

    def best(mask) -> str | None:
        d = r[mask] if not r.empty else r
        return None if d.empty else str(d.sort_values("robust", ascending=False).iloc[0]["name"])

    prov = r["providers"].fillna("").astype(str) if not r.empty else pd.Series(dtype=str)
    out = {"conservative_screening": "V2_CONSERVATIVE" if deployable.get("V2_CONSERVATIVE") else None,
           "best_tabular": "V2_BASELINE" if deployable.get("V2_BASELINE") else None}
    if not r.empty:
        out["harmonized_acoustic"] = best(prov == "HARM")
        out["phonetic_multimodal"] = best(prov.str.contains("PHON"))
        out["residual_fusion"] = best(r["fusion"] == "residual")
    else:
        out.update({"harmonized_acoustic": None, "phonetic_multimodal": None, "residual_fusion": None})
    return out


def ensemble_members(results: pd.DataFrame, chosen: dict[str, str | None]) -> list[str]:
    roles = ("harmonized_acoustic", "phonetic_multimodal", "residual_fusion")
    names = [chosen.get(k) for k in roles if chosen.get(k)]
    r = results.set_index("name")
    keep = [n for n in dict.fromkeys(names) if n in r.index and float(r.loc[n, "delta_ci_low"]) > 0]
    return keep


def blend_oof(oofs: list[pd.DataFrame]) -> pd.DataFrame:
    base = oofs[0][[ID, "repeat", "fold", "y_true", "fold_label"]].copy()
    P = np.mean([o.set_index([ID, "repeat"]).reindex(base.set_index([ID, "repeat"]).index)["y_pred"].to_numpy() for o in oofs], axis=0)
    base["y_pred"] = P
    return base


class V2SubmissionBuilder:
    def __init__(self, ctx, evaluator, stage: str = "P22"):
        self.ctx, self.ev, self.stage = ctx, evaluator, stage
        self.dir = ctx.paths.root / ctx.cfg.get_path("v2.submission.directory", "submissions/V2")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bundle_dir = ctx.paths.models / "v2"
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

    def predict_member(self, spec, test_ids: list[str]) -> tuple[np.ndarray, dict]:
        fmod, Xtr, Xte = self.ev.fit_full(spec, test_ids)
        path = self.bundle_dir / f"{spec.name}.joblib"
        joblib.dump({"model": fmod, "columns": list(Xtr.columns), "spec": spec.as_dict()}, path, compress=3)
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        del fmod
        fresh = joblib.load(path)                                   # independent re-load from disk
        p = fresh["model"].predict(Xte.reindex(columns=fresh["columns"]))
        return np.asarray(p, float), {"bundle": str(path), "sha1": digest, "n_features": len(fresh["columns"])}

    def write(self, role: str, pred: np.ndarray, meta: dict, checker) -> dict:
        b = self.ctx.bundle()
        fmt = b.submission_format
        if self.ctx.cfg.get_path("v2.submission.clip_to_train_range", True):
            y = b.y.to_numpy(float)
            pred = np.clip(pred, float(y.min()), float(y.max()))
        sub = pd.DataFrame({ID: fmt[ID].values, TARGET: pred})[list(fmt.columns)]
        checks = validate_submission(sub, fmt)
        safety = checker._pred_checks(pred, "test")
        ok = not (checks["status"] == "FAIL").any() and not any(s["status"] == "FAIL" for s in safety)
        out = {"role": role, "status": "written" if ok else "refused", **meta,
               "pred_mean": float(np.mean(pred)), "pred_sd": float(np.std(pred)), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if ok:
            path = self.dir / f"WILDSPEECH_V2_{role}.csv"
            atomic_write_csv(sub, path)
            out["file"] = str(path)
        self.ctx.save_table(pd.concat([checks.assign(kind="format"), pd.DataFrame(safety).assign(kind="safety")], ignore_index=True),
                            self.stage, f"submission_checks_{role}")
        write_json(self.dir / f"WILDSPEECH_V2_{role}.meta.json", out)
        return out
