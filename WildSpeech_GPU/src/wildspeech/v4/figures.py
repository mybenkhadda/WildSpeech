"""V4 scientific figures. Every figure is saved as figures/<name>.png and .svg, with the data it was drawn from in
figures/data/<name>.csv and .parquet, and indexed in figures/figure_index.csv (caption, n, units, source tables).
A figure whose inputs are absent is written as an explicit 'not available' panel (never silently skipped).
Plot style and helpers: wildspeech.v3.viz (fixed categorical slots, single-hue sequential, blue/red diverging).
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from .. import FILE, ID, TARGET
from ..phonetic.error_features import ops_from_string
from ..v3 import viz as V


class FigureMaker:
    def __init__(self, store, dpi: int = 150):
        self.store, self.dpi = store, int(dpi)
        self.index: list[dict] = []

    # ------------------------------------------------------------------ save
    def save(self, name: str, fig, data: pd.DataFrame | None, caption: str, n: int | None = None, units: str = "",
             sources: str = "") -> None:
        import matplotlib.pyplot as plt

        png = self.store.path("figures", f"{name}.png")
        svg = self.store.path("figures", f"{name}.svg")
        fig.savefig(png, dpi=self.dpi, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        dpath = None
        if data is not None:
            d = data.reset_index() if (data.index.name is not None or not isinstance(data.index, pd.RangeIndex)) else data
            d = d.copy()
            d.columns = [str(c) for c in d.columns]
            dpath = self.store.save_table(d, "figures", f"data/{name}.parquet", csv=True)
        self.index.append({"figure": name, "png": str(png), "svg": str(svg), "data": str(dpath) if dpath else None,
                           "caption": caption, "n": n, "units": units, "sources": sources,
                           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "ok"})

    def run(self, name: str, fn, *a, **k) -> None:
        try:
            fn(name, *a, **k)
        except Exception as e:  # noqa: BLE001 - a broken figure never stops the run; recorded in the index
            self.index.append({"figure": name, "status": "failed", "caption": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-2000:]})

    def na(self, name: str, title: str, why: str) -> None:
        self.save(name, V.empty(title, why), pd.DataFrame([{"status": "not_available", "reason": why}]),
                  f"NOT AVAILABLE: {why}")

    def write_index(self) -> None:
        self.store.save_table(pd.DataFrame(self.index), "figures", "figure_index.csv")

    # ================================================================== data / ASR
    def asr_coverage(self, name: str, cov: pd.DataFrame) -> None:
        if cov is None or cov.empty:
            return self.na(name, "ASR model coverage", "no ASR import")
        plt = V._plt()
        d = cov.copy()
        models = sorted(d["model_key"].unique())
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(models) + 2), 3.8))
        w = 0.38
        for k, s in enumerate(("train", "test")):
            dd = d[d["split"] == s].set_index("model_key").reindex(models)
            ax.bar(np.arange(len(models)) + (k - 0.5) * w, dd["usable_frac"].fillna(0), width=w, color=V.SERIES[k],
                   label=f"{s} (n clips={int(dd['n_metadata_clips'].max() or 0)})")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_ylabel("share of metadata clips with a usable prediction")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title("ASR model coverage by split")
        self.save(name, fig, d, "Usable (successful, matched, non-null) predictions per model and split.",
                  n=int(d["n_metadata_clips"].sum() // max(len(models), 1)), units="fraction of clips", sources="asr_coverage.csv")

    def join_alignment(self, name: str, clip_master: pd.DataFrame) -> None:
        if clip_master is None or clip_master.empty:
            return self.na(name, "Audio / text alignment", "no clip master")
        d = clip_master.assign(has_asr=(clip_master.get("asr_n_usable_models_ok", 0) > 0).astype(float),
                               has_audio=clip_master.get("in_audio_manifest", np.nan).astype(float))
        t = d.groupby(["split", "task"]).agg(n=(FILE, "size"), asr_coverage=("has_asr", "mean"),
                                              audio_coverage=("has_audio", "mean")).reset_index()
        t["label"] = t["split"] + " · " + t["task"] + " (n=" + t["n"].astype(str) + ")"
        fig = V.bars(t, "label", "asr_coverage", "Clips with >= 1 usable ASR prediction (joined on split + filename)",
                     ylabel="share of clips", ref=1.0)
        self.save(name, fig, t, "Join coverage per split and task (audio manifest and ASR).", n=int(t["n"].sum()),
                  units="fraction of clips", sources="clip_master.parquet")

    def wer_cer_by_task(self, name: str, text_long: pd.DataFrame) -> None:
        if text_long is None or text_long.empty or "w_er" not in text_long:
            return self.na(name, "WER / CER by task", "no recognized-text features")
        d = text_long[(text_long["split"] == "train") & text_long["asr_ok"]].copy()
        m = d.melt(id_vars=["task", "model_key"], value_vars=[c for c in ("w_er", "c_er") if c in d], var_name="metric")
        m["value"] = m["value"].clip(upper=2.0)
        m["panel"] = m["task"] + " · " + m["metric"].map({"w_er": "WER", "c_er": "CER"})
        fig = V.small_multiple_distributions(m, "value", "panel", "model_key",
                                             f"Recognized-text error vs expected text (train clips, n={d[FILE].nunique()})")
        s = m.groupby(["panel", "model_key"])["value"].describe().reset_index()
        self.save(name, fig, s, "Distribution of WER and CER (clipped at 2) per task and recognizer; recognized-text "
                                "error, not verified pronunciation error.", n=int(d[FILE].nunique()), units="edit rate")

    def transcript_examples(self, name: str, ex: pd.DataFrame) -> None:
        if ex is None or ex.empty:
            return self.na(name, "Transcript examples", "no examples")
        items = []
        for r in ex.head(14).itertuples(index=False):
            ops = [(o[0], o[1], o[2]) for o in ops_from_string(r.word_ops)]
            items.append({"label": f"{r.task}/{r.model_key} score={r.item_score}", "ops": ops})
        self.save(name, V.alignment_panel(items, "Expected vs recognized words (training examples)"), ex,
                  "Word-level edit alignments (M match, S substitution, D deletion, I insertion).", n=len(items))

    def phoneme_confusion(self, name: str, conf: pd.DataFrame, available: bool) -> None:
        if not available or conf is None or conf.empty:
            return self.na(name, "Phoneme confusions", "no measured phone predictions in the export")
        if "ref" in conf.columns:
            conf = conf.set_index("ref")
        conf = conf.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        norm = conf.div(conf.sum(1).replace(0, np.nan), axis=0)
        self.save(name, V.heatmap(norm, "Expected (rows) vs recognized (columns) phones, row-normalized",
                                  xlabel="recognized phone", ylabel="expected phone"), conf,
                  "Measured phone confusions on training clips (∅ = deletion / insertion).", n=int(conf.to_numpy().sum()),
                  units="row share")

    # ================================================================== student-level science
    def scatter_pair(self, name: str, df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str,
                     hue: str | None = None, caption: str = "") -> None:
        if df is None or df.empty or x not in df or y not in df:
            return self.na(name, title, f"missing {x} / {y}")
        d = df[[c for c in (x, y, hue) if c]].dropna(subset=[x, y])
        self.save(name, V.scatter(d, x, y, f"{title} (n={len(d)})", hue=hue, xlabel=xlabel, ylabel=ylabel), d,
                  caption, n=len(d))

    def residual_correlations(self, name: str, conf: pd.DataFrame) -> None:
        if conf is None or conf.empty:
            return self.na(name, "Residual correlations", "no confounding table")
        d = conf.dropna(subset=["rho_residual"]).copy()
        d["abs"] = d["rho_residual"].abs()
        d = d.sort_values("abs", ascending=False).head(25)
        d["label"] = d["feature"].str.slice(0, 60)
        fig = V.bars(d, "label", "rho_residual", "Spearman ρ with the baseline OOF residual (top 25 features)",
                     ylabel="ρ (residual)", ref=0.0)
        self.save(name, fig, d, "Descriptive association of student features (fitted on all training students) with the "
                                "repeated-CV baseline residual; partial ρ controls sample rate and instrument.",
                  n=int(d["n"].max()) if "n" in d else None, units="Spearman ρ")

    def task_conditioned(self, name: str, text_cons: pd.DataFrame, clipf: pd.DataFrame, col: str = "w_er") -> None:
        if text_cons is None or text_cons.empty or col not in text_cons:
            return self.na(name, "Task-conditioned distributions", "no text features")
        d = clipf[[FILE, "task", "item_score", "split"]].merge(text_cons[[FILE, col]], on=FILE)
        d = d[(d["split"] == "train") & d["item_score"].notna()]
        d["official"] = d["item_score"].map({1.0: "correct", 0.0: "incorrect"})
        d[col] = d[col].clip(upper=2.0)
        self.save(name, V.small_multiple_distributions(d, col, "task", "official",
                                                       f"Recognized-text {col} by official correctness (n={len(d)})"),
                  d.groupby(["task", "official"])[col].describe().reset_index(),
                  "Consensus recognized-text error by task and official item score.", n=len(d), units=col)

    def correct_response(self, name: str, text_cons: pd.DataFrame, acou: pd.DataFrame | None, clipf: pd.DataFrame,
                         y: pd.Series, cols: list[str]) -> None:
        d = clipf[[FILE, ID, "task", "item_score", "split"]]
        d = d[(d["split"] == "train") & (d["item_score"] == 1.0)]
        parts = []
        if text_cons is not None and not text_cons.empty:
            parts.append(text_cons[[FILE] + [c for c in cols if c in text_cons]])
        if acou is not None and not acou.empty:
            parts.append(acou[[FILE] + [c for c in cols if c in acou]])
        for p in parts:
            d = d.merge(p, on=FILE, how="left")
        have = [c for c in cols if c in d and d[c].notna().any()]
        if not have:
            return self.na(name, "Correct-response comparison", "no features on correct clips")
        terc = pd.qcut(y.rank(method="first"), 3, labels=["low outcome", "middle", "high outcome"])
        d["outcome_tercile"] = d[ID].map(terc)
        d = d[d["outcome_tercile"].isin(["low outcome", "high outcome"])]
        m = d.melt(id_vars=["task", "outcome_tercile"], value_vars=have, var_name="feature")
        m["panel"] = m["task"] + " · " + m["feature"].str.replace("::", " ")
        self.save(name, V.small_multiple_distributions(m.dropna(subset=["value"]), "value", "panel", "outcome_tercile",
                                                       f"Officially CORRECT responses: low vs high outcome tercile (n clips={len(d)})"),
                  m.groupby(["panel", "outcome_tercile"])["value"].describe().reset_index(),
                  "Speech / recognized-text characteristics among correct responses only, by end-of-year outcome tercile "
                  "(descriptive; associations, not causal).", n=len(d))

    def disagreement(self, name: str, dis: pd.DataFrame, clipf: pd.DataFrame, vd: pd.DataFrame | None) -> None:
        if dis is None or dis.empty:
            return self.na(name, "ASR disagreement", "fewer than two usable recognizers")
        d = clipf[[FILE, "task", "split", "orig_sr_low"]].merge(dis, on=FILE)
        d["domain"] = np.where(d["orig_sr_low"] > 0.5, "low original rate", "16 kHz original")
        self.save(name, V.small_multiple_distributions(d, "dis_word_mean", "task", "domain",
                                                       f"Mean pairwise word disagreement between recognizers (n={len(d)})"),
                  d.groupby(["task", "domain"])["dis_word_mean"].describe().reset_index(),
                  "Cross-recognizer disagreement by task and recording domain.", n=len(d), units="normalized edit distance")
        if vd is not None and not vd.empty:
            self.save(name + "_variance", V.bars(vd, "block", "r2_increment",
                                                 "Sequential R² of clip disagreement (item → recording → correctness → student)",
                                                 ylabel="ΔR²"), vd,
                      "Variance decomposition: how much disagreement is item / recording / correctness / student.",
                      n=int(vd["n"].max()))

    def domain_confounding(self, name: str, conf: pd.DataFrame) -> None:
        if conf is None or conf.empty:
            return self.na(name, "Domain confounding", "no confounding table")
        d = conf.copy()
        d["family"] = d["feature"].str.split("__").str[0]
        t = d.groupby("family").agg(rho_sample_rate=("rho_sample_rate", lambda s: float(np.nanmedian(np.abs(s)))),
                                    rho_residual=("rho_residual", lambda s: float(np.nanmedian(np.abs(s)))),
                                    partial_rho=("partial_rho_residual", lambda s: float(np.nanmedian(np.abs(s)))),
                                    attenuation=("attenuation", "median"), n_features=("feature", "size"))
        self.save(name, V.heatmap(t[["rho_sample_rate", "rho_residual", "partial_rho"]],
                                  "Median |ρ| per family: sample rate, baseline residual, residual | domain", annotate=True),
                  t, "Recording-domain confounding of each feature family.", units="median |Spearman ρ|")

    # ================================================================== models
    def forest_rows(self, name: str, rows: pd.DataFrame, title: str, caption: str) -> None:
        if rows is None or rows.empty or "delta_r2" not in rows:
            return self.na(name, title, "no evaluated rows")
        d = rows.dropna(subset=["delta_r2"]).copy()
        d["label"] = d["name"] + " [" + d.get("result_class", pd.Series("", index=d.index)).astype(str) + "]"
        self.save(name, V.forest(d, "delta_r2", "delta_ci_low", "delta_ci_high", "label", title,
                                 group="result_class" if "result_class" in d else None,
                                 xlabel="ΔR² vs reference baseline (paired bootstrap 95% CI)"),
                  d, caption, n=int(d["n_students"].max()) if "n_students" in d and d["n_students"].notna().any() else None,
                  units="ΔR²")

    def fold_stability(self, name: str, st: pd.DataFrame) -> None:
        if st is None or st.empty:
            return self.na(name, "Fold stability", "no OOF")
        plt = V._plt()
        names = list(dict.fromkeys(st["name"]))[:12]
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(names) + 2), 4))
        ax.boxplot([st.loc[st["name"] == n, "r2"].to_numpy() for n in names], showmeans=True)
        ax.set_xticks(np.arange(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("R² per validation fold")
        ax.set_title(f"Per-fold R² (n folds={st.groupby('name').size().max()})")
        self.save(name, fig, st, "Distribution of validation-fold R² across repeats and folds.", units="R²")

    def predicted_vs_actual(self, name: str, oof: pd.DataFrame, model: str, scheme: str) -> None:
        from ..validation.metrics import mean_oof

        d = oof[(oof["name"] == model) & (oof["scheme"] == scheme)]
        if d.empty:
            return self.na(name, "Predicted vs actual", f"no OOF for {model}")
        m = mean_oof(d).reset_index()
        self.save(name, V.scatter(m, "y_true", "y_pred", f"{model}: repeat-averaged OOF prediction vs outcome (n={len(m)})",
                                  xlabel="actual outcome_score", ylabel="predicted outcome_score"), m,
                  "Out-of-fold predictions (never in-sample) averaged over repeats.", n=len(m), units="outcome_score")

    def importance(self, name: str, grouped: pd.DataFrame, shap: pd.DataFrame | None) -> None:
        if grouped is None or grouped.empty:
            self.na(name, "Grouped importance", "no importance computed")
        else:
            self.save(name, V.bars(grouped, "family", "importance_mean", "Grouped permutation importance (validation folds)",
                                   lo="ci_low", hi="ci_high", ylabel="R² drop when the family is permuted", ref=0.0),
                      grouped, "Fold-safe grouped permutation importance of the champion's feature families.",
                      units="R² drop")
        if shap is not None and not shap.empty:
            top = shap.head(25).assign(label=lambda d: d["feature"].str.slice(0, 60))
            self.save(name + "_model", V.bars(top, "label", "importance", f"Champion refit: {shap['method'].iloc[0]} (top 25)",
                                              ylabel=str(shap["method"].iloc[0])), shap,
                      "Importance in the champion refitted on all labelled students (SHAP for trees when available).")

    def error_archetypes(self, name: str, tab: pd.DataFrame) -> None:
        if tab is None or tab.empty:
            return self.na(name, "Error archetypes", "no failure phenotypes")
        m = tab.melt(id_vars=["phenotype", "n"], value_vars=["mae_base", "mae_model"], var_name="model", value_name="mae")
        plt = V._plt()
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ph = list(tab["phenotype"])
        for k, mm in enumerate(("mae_base", "mae_model")):
            v = m[m["model"] == mm].set_index("phenotype").reindex(ph)["mae"]
            ax.bar(np.arange(len(ph)) + (k - 0.5) * 0.38, v, width=0.38, color=V.SERIES[k],
                   label={"mae_base": "baseline", "mae_model": "multimodal"}[mm])
        ax.set_xticks(np.arange(len(ph)))
        ax.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(ph, tab["n"])])
        ax.set_ylabel("MAE (outcome_score)")
        ax.legend(fontsize=8)
        ax.set_title("Error archetypes: baseline residual quintiles")
        self.save(name, fig, tab, "MAE of baseline vs multimodal model for under-/over-estimated students.",
                  n=int(tab["n"].sum()), units="outcome_score")

    def failure_recovery(self, name: str, prof: pd.DataFrame) -> None:
        if prof is None or prof.empty:
            return self.na(name, "Multimodal failure recovery", "no failure phenotypes")
        d = prof.reset_index()[["abs_err_base", "abs_err_new", "phenotype"]]
        self.save(name, V.scatter(d, "abs_err_base", "abs_err_new", f"Absolute error: baseline vs multimodal (n={len(d)})",
                                  hue="phenotype", xlabel="|error| baseline", ylabel="|error| multimodal", trend=False), d,
                  "Points below the diagonal are students whose error the multimodal model reduced.", n=len(d),
                  units="outcome_score")

    def robustness(self, name: str, rob: pd.DataFrame) -> None:
        if rob is None or rob.empty:
            return self.na(name, "Robustness", "no robustness rows")
        piv = rob.pivot_table(index="name", columns="scheme", values="delta_r2", aggfunc="first")
        self.save(name, V.heatmap(piv, "ΔR² vs baseline by validation scheme", diverging=True, annotate=True),
                  piv, "Transfer of incremental R² across school-grouped, leave-one-school-out, instrument and sample-rate "
                       "schemes (baseline refitted per scheme).", units="ΔR²")


def student_descriptives(providers: dict, ids: list[str], names: list[str]) -> pd.DataFrame:
    """Provider tables fitted on ALL training students — DESCRIPTIVE figures only, never used for evaluation."""
    parts = []
    for n in names:
        if n in providers and providers[n].available():
            parts.append(providers[n].table(ids, ids))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=pd.Index(ids, name=ID))


def outcome_frame(bundle) -> pd.DataFrame:
    tr = bundle.train.set_index(ID)
    out = pd.DataFrame({TARGET: tr[TARGET]})
    th = [c for c in bundle.theta_cols if c in tr]
    if th:
        out["theta_mean"] = tr[th].astype(float).mean(1)
    for c in ("instrument", "original_sample_rate"):
        if c in tr:
            out[c] = tr[c]
    out.index = out.index.astype(str)
    return out
