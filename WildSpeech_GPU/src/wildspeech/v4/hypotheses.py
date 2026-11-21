"""H1-H10: methods, sample sizes, effect estimates, uncertainty, evidence strength and limitations.

Every statement is computed from runtime results (experiment rows, paired bootstrap comparisons, robustness rows,
diagnostic tables). Associations are never labelled causal. Evidence strength (pre-specified):
  strong            GENUINE_GAIN on the primary scheme (ΔR² CI > 0, BH q < alpha, no fallback fold), the sign is
                    positive on every evaluated robustness scheme, and the shuffled-ASR control shows no gain
  moderate          GENUINE_GAIN on the primary scheme only
  weak              ΔR² point estimate > 0 but the CI spans 0 (NO_DETECTABLE_EFFECT)
  none              ΔR² <= 0 (NO_DETECTABLE_EFFECT / GENUINE_NEGATIVE)
  not_interpretable BASELINE_FALLBACK / NOT_REPORTABLE / UNDETERMINED (fallback rows say nothing about features)
  unavailable       required modality absent (e.g. no measured phones)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID

QUESTIONS = {
    "H1": "Does ASR recognized-text deviation predict end-of-year reading after adjusting for screening?",
    "H2": "Which task (blending, deletion, sentence repetition) contributes the most independent signal?",
    "H3": "Do acoustic / recognized-text characteristics of officially CORRECT responses predict reading differences?",
    "H4": "Does disagreement between recognizers carry literacy information or mainly recording artefacts?",
    "H5": "Do measured phonetic error patterns and their within-student stability predict future reading?",
    "H6": "How much of the apparent speech signal is explained by sample rate / instrument / school?",
    "H7": "Can speech features explain screening-model residuals on unseen students?",
    "H8": "Which students are systematically under/over-estimated by screening-only models, and does multimodal "
          "modelling improve them?",
    "H9": "Do nonlinear interactions (theta x difficulty x fluency x text errors) add signal beyond additive models?",
    "H10": "Which feature families remain reliable across folds, schools and recording conditions?",
}
LIMITS_COMMON = ("Observational association on ~541 students; not causal. Recognized-text error mixes child "
                 "responses with recognizer error on child speech.")


def strength(row: pd.Series | dict | None, robust_signs: list[float] | None = None, control_gain: bool = False) -> str:
    if row is None:
        return "unavailable"
    cls = str(row.get("result_class", ""))
    if cls in ("BASELINE_FALLBACK", "NOT_REPORTABLE", "UNDETERMINED", ""):
        return "not_interpretable"
    d = row.get("delta_r2", np.nan)
    if cls == "GENUINE_GAIN":
        rs = [s for s in (robust_signs or []) if np.isfinite(s)]
        if rs and all(s > 0 for s in rs) and not control_gain:
            return "strong"
        return "moderate"
    if cls == "NO_DETECTABLE_EFFECT" and np.isfinite(d) and d > 0:
        return "weak"
    return "none"


def _row(rows: pd.DataFrame, name: str, scheme: str) -> dict | None:
    r = rows[(rows["name"] == name) & (rows["scheme"] == scheme)]
    return None if r.empty else r.iloc[0].to_dict()


def _best(rows: pd.DataFrame, prefix: str, scheme: str) -> dict | None:
    r = rows[rows["name"].astype(str).str.startswith(prefix) & (rows["scheme"] == scheme)]
    r = r[r["status"].isin(["ok", "partial_fallback", "fallback_baseline"])]
    if r.empty:
        return None
    return r.sort_values("delta_r2", ascending=False, na_position="last").iloc[0].to_dict()


def _signs(rows: pd.DataFrame, name: str, schemes: list[str]) -> list[float]:
    r = rows[(rows["name"] == name) & rows["scheme"].isin(schemes)]
    return [float(x) for x in pd.to_numeric(r["delta_r2"], errors="coerce")]


def _rec(h, method, row, n, extra_lim="", **kw) -> dict:
    return {"hypothesis": h, "question": QUESTIONS[h], "method": method,
            "experiment": (row or {}).get("name"), "scheme": (row or {}).get("scheme"),
            "n_students": n, "effect": (row or {}).get("delta_r2", kw.pop("effect", np.nan)),
            "ci_low": (row or {}).get("delta_ci_low", kw.pop("ci_low", np.nan)),
            "ci_high": (row or {}).get("delta_ci_high", kw.pop("ci_high", np.nan)),
            "q_value": (row or {}).get("q"), "result_class": (row or {}).get("result_class"),
            "limitations": (LIMITS_COMMON + " " + extra_lim).strip(), **kw}


def evaluate(rows: pd.DataFrame, primary: str, robust_schemes: list[str], n_students: int, extra: dict) -> pd.DataFrame:
    """rows: classified result rows of every scheme; extra: dict of diagnostic tables / numbers (see pipeline)."""
    out = []
    ctrl = _row(rows, "CTRL_SHUFFLED_ASR", primary)
    ctrl_gain = bool(ctrl and str(ctrl.get("result_class")) == "GENUINE_GAIN")

    # H1
    r1 = _best(rows, "M1_TEXT_", primary) or _row(rows, "M5_RESID_TEXT", primary)
    out.append(_rec("H1", "ΔR² of baseline + recognized-text families (early fusion; best M1 estimator) vs the "
                          "reference baseline, paired bootstrap of repeat-averaged OOF predictions; same folds",
                    r1, n_students, evidence=strength(r1, _signs(rows, (r1 or {}).get("name", ""), robust_schemes), ctrl_gain),
                    shuffled_asr_control_class=(ctrl or {}).get("result_class")))
    # H2
    tasks = rows[rows["name"].astype(str).str.startswith("ABL_TASK_") & (rows["scheme"] == primary)]
    if tasks.empty:
        out.append(_rec("H2", "per-task ASR text families", None, n_students, evidence="unavailable"))
    else:
        best = tasks.sort_values("delta_r2", ascending=False).iloc[0].to_dict()
        detail = "; ".join(f"{r['name'].replace('ABL_TASK_', '')}: ΔR²={r['delta_r2']:.4f} [{r.get('delta_ci_low', np.nan):.4f}, "
                           f"{r.get('delta_ci_high', np.nan):.4f}] {r.get('result_class')}" for _, r in tasks.iterrows()
                           if pd.notna(r.get("delta_r2")))
        out.append(_rec("H2", "one provider per task (that task's clips only), each added alone to the baseline",
                        best, n_students, evidence=strength(best), detail=detail,
                        extra_lim="Tasks differ in clip counts and adaptive administration; contributions are not additive."))
    # H3
    r3 = _row(rows, "H3_CORRECT_ONLY", primary) or _row(rows, "H3_CORRECT_ONLY_RESID", primary)
    out.append(_rec("H3", "text + acoustic features aggregated over officially CORRECT clips only (item-conditioned "
                          "z-scores, fold-fitted), added to the baseline", r3, n_students, evidence=strength(r3),
                    extra_lim="Correct-response subsets are smaller for low-ability students (selection)."))
    # H4
    r4 = _row(rows, "ABL_FAMILY_ASRDIS", primary) or _row(rows, "H4_DISAGREEMENT_RESID", primary)
    vd = extra.get("disagreement_variance")
    vd_txt = ""
    if isinstance(vd, pd.DataFrame) and not vd.empty:
        vd_txt = "; ".join(f"{r.block}: +{r.r2_increment:.3f}" for r in vd.itertuples())
    out.append(_rec("H4", "ΔR² of the cross-recognizer disagreement family + sequential variance decomposition of "
                          "clip disagreement (item -> recording -> correctness -> student)", r4, n_students,
                    evidence=strength(r4) if r4 else "unavailable", variance_decomposition=vd_txt,
                    split_half=str(extra.get("disagreement_split_half", "")),
                    extra_lim="Needs >= 2 usable recognizers; disagreement is never interpreted as pronunciation difficulty."))
    # H5
    if not extra.get("phonetic_available"):
        out.append(_rec("H5", "measured phonetic errors (ASRPHN)", None, n_students, evidence="unavailable",
                        extra_lim="The export contains no measured phone predictions; G2P of word transcripts is not a "
                                  "phonetic measurement."))
    else:
        r5 = _row(rows, "ABL_FAMILY_ASRPHN", primary)
        out.append(_rec("H5", "ΔR² of measured phonetic errors + split-half reliability of student PER", r5, n_students,
                        evidence=strength(r5), split_half=str(extra.get("phonetic_split_half", ""))))
    # H6
    h6 = extra.get("h6_attenuation") or {}
    if h6:
        lo6, d6 = float(h6.get("delta_ci_low", np.nan)), float(h6.get("delta_r2", np.nan))
        ev6 = "moderate" if (np.isfinite(lo6) and lo6 > 0) else ("weak" if (np.isfinite(d6) and d6 > 0) else "none")
    else:
        ev6 = "unavailable"
    out.append(_rec("H6", "ΔR²(text + recording-conditions control) − ΔR²(recording only), paired; transfer across "
                          "sample rate / instrument / school schemes; partial correlations with recording metadata",
                    None, n_students, effect=h6.get("delta_r2", np.nan), ci_low=h6.get("delta_ci_low", np.nan),
                    ci_high=h6.get("delta_ci_high", np.nan), evidence=ev6,
                    sample_rate_transfer_delta=_signs(rows, (r1 or {}).get("name", ""), ["sample_rate_transfer"]),
                    confounding_median_attenuation=extra.get("confounding_attenuation"),
                    extra_lim="school_id is used only for grouping (train-only); confounding is assessed, not removed."))
    # H7
    r7 = _row(rows, "M5_RESID_ALL", primary)
    out.append(_rec("H7", "nested residual correction: speech families fitted to inner-OOF baseline residuals, "
                          "shrinkage alpha chosen by inner CV; ΔR² on unseen students", r7, n_students,
                    evidence=strength(r7, _signs(rows, "M5_RESID_ALL", robust_schemes), ctrl_gain),
                    residual_r2_explained=extra.get("residual_r2_explained"),
                    alpha_zero_folds=extra.get("m5_alpha_zero_folds")))
    # H8
    h8 = extra.get("h8") or {}
    out.append(_rec("H8", "students in the top / bottom quintile of the baseline OOF residual (under / over-estimated): "
                          "MAE reduction of the champion / best multimodal model, student bootstrap", None, n_students,
                    effect=h8.get("mae_reduction_under", np.nan), ci_low=h8.get("ci_low_under", np.nan),
                    ci_high=h8.get("ci_high_under", np.nan),
                    evidence=("moderate" if h8.get("ci_low_under", -1) > 0 else ("weak" if h8.get("mae_reduction_under", 0) > 0 else "none")) if h8 else "unavailable",
                    over_estimated=str({k: v for k, v in h8.items() if "over" in k}),
                    extra_lim="Quintiles of OOF residuals regress to the mean; reductions are paired on the same students."))
    # H9
    h9 = extra.get("h9") or {}
    out.append(_rec("H9", "paired ΔR² of boosted trees vs ridge on identical inputs (baseline + speech families)",
                    None, n_students, effect=h9.get("delta_r2", np.nan), ci_low=h9.get("delta_ci_low", np.nan),
                    ci_high=h9.get("delta_ci_high", np.nan),
                    evidence=("moderate" if h9.get("delta_ci_low", -1) > 0 else ("weak" if h9.get("delta_r2", 0) > 0 else "none")) if h9 else "unavailable",
                    compared=h9.get("compared")))
    # H10
    h10 = extra.get("h10")
    if isinstance(h10, pd.DataFrame) and not h10.empty:
        best = h10.sort_values(["n_positive", "jaccard_mean"], ascending=False).iloc[0]
        out.append(_rec("H10", "per family: ΔR² sign consistency across repeated K-fold, school-grouped, LOSO, "
                               "instrument and sample-rate transfer + selection stability (Jaccard)", None, n_students,
                        effect=float(best.get("n_positive", np.nan)), evidence="descriptive",
                        detail="; ".join(f"{r.family}: positive in {int(r.n_positive)}/{int(r.n_schemes)} schemes, "
                                         f"Jaccard={r.jaccard_mean:.2f}" for r in h10.itertuples())))
    else:
        out.append(_rec("H10", "family generalization", None, n_students, evidence="unavailable"))
    return pd.DataFrame(out)


# ================================================================================ helper analyses
def residual_explained(oof: pd.DataFrame, ref: str, name: str, scheme: str) -> float:
    """OOF share of baseline residual variance explained by the model's correction (pred - baseline pred)."""
    from ..validation.metrics import mean_oof

    d = oof[oof["scheme"] == scheme]
    a, b = mean_oof(d[d["name"] == ref]), mean_oof(d[d["name"] == name])
    j = a.join(b, lsuffix="_b", rsuffix="_n", how="inner")
    if j.empty:
        return np.nan
    res = j["y_true_b"] - j["y_pred_b"]
    corr = j["y_pred_n"] - j["y_pred_b"]
    ss = float(np.sum((res - res.mean()) ** 2))
    return float(1 - np.sum((res - corr) ** 2) / ss) if ss > 0 else np.nan


def failure_phenotypes(oof: pd.DataFrame, ref: str, model: str, scheme: str, meta: pd.DataFrame, n_boot: int = 1000,
                       seed: int = 0) -> tuple[dict, pd.DataFrame]:
    from ..validation.metrics import mean_oof

    d = oof[oof["scheme"] == scheme]
    a, b = mean_oof(d[d["name"] == ref]), mean_oof(d[d["name"] == model])
    j = a.join(b, lsuffix="_b", rsuffix="_n", how="inner")
    if j.empty or model == ref:
        return {}, pd.DataFrame()
    res = j["y_true_b"] - j["y_pred_b"]
    q = res.quantile([0.2, 0.8])
    grp = np.where(res >= q[0.8], "underestimated", np.where(res <= q[0.2], "overestimated", "typical"))
    j["phenotype"] = grp
    j["abs_err_base"] = (j["y_true_b"] - j["y_pred_b"]).abs()
    j["abs_err_new"] = (j["y_true_b"] - j["y_pred_n"]).abs()
    rng = np.random.default_rng(seed)
    out, rows = {}, []
    for g in ("underestimated", "overestimated", "typical"):
        s = j[j["phenotype"] == g]
        if len(s) < 5:
            continue
        diff = (s["abs_err_base"] - s["abs_err_new"]).to_numpy()
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(int(n_boot))]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        key = "under" if g == "underestimated" else ("over" if g == "overestimated" else "typical")
        out.update({f"mae_reduction_{key}": float(diff.mean()), f"ci_low_{key}": float(lo), f"ci_high_{key}": float(hi),
                    f"n_{key}": int(len(s))})
        rows.append({"phenotype": g, "n": len(s), "mae_base": float(s["abs_err_base"].mean()),
                     "mae_model": float(s["abs_err_new"].mean()), "mae_reduction": float(diff.mean()),
                     "ci_low": float(lo), "ci_high": float(hi)})
    prof = j.join(meta, how="left") if meta is not None else j
    tab = pd.DataFrame(rows)
    tab.attrs["profiles"] = prof
    return out, tab


def family_generalization(rows: pd.DataFrame, schemes: list[str], stability: dict[str, float]) -> pd.DataFrame:
    fam_rows = rows[rows["name"].astype(str).str.startswith("ABL_FAMILY_")]
    if fam_rows.empty:
        return pd.DataFrame()
    piv = fam_rows.pivot_table(index="name", columns="scheme", values="delta_r2", aggfunc="first")
    cols = [c for c in schemes if c in piv.columns]
    out = pd.DataFrame({"family": [n.replace("ABL_FAMILY_", "") for n in piv.index],
                        "n_schemes": piv[cols].notna().sum(1).to_numpy(),
                        "n_positive": (piv[cols] > 0).sum(1).to_numpy()})
    for c in cols:
        out[f"delta_{c}"] = piv[c].to_numpy()
    out["jaccard_mean"] = out["family"].map(stability).astype(float)
    return out


def split_half(clip_table: pd.DataFrame, clipf: pd.DataFrame, col: str) -> dict:
    from ..discovery.asr_disagreement import split_half_consistency
    from .. import FILE

    if clip_table is None or clip_table.empty or col not in clip_table:
        return {}
    return split_half_consistency(clip_table[[FILE, col]], clipf, col)


def n_students(oof: pd.DataFrame, scheme: str) -> int:
    d = oof[oof["scheme"] == scheme] if not oof.empty else oof
    return int(d[ID].nunique()) if not d.empty else 0
