"""V2 research report: every sentence is computed from runtime tables; missing or failed stages are reported as
such (nothing is filled in). Output: outputs/V2/scientific_report.html (+ .md), figures embedded from the
figure index, registries summarised, exploratory vs confirmatory evidence separated.
"""
from __future__ import annotations

import base64
import html
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import V2_STAGE_IDS

TITLES = {"P16": "Acoustic domain harmonization", "P17": "Clip-level correctness intelligence",
          "P18": "Phonetic and linguistic intelligence", "P19": "Reference-conditioned speech modelling",
          "P20": "Speech foundation-model discovery", "P21": "Compact multimodal regression",
          "P22": "Failure recovery and champion selection"}


def _t(ctx, sid, name):
    try:
        return ctx.read_stage_table(sid, name)
    except Exception:  # noqa: BLE001 - unreadable table is reported as missing
        return None


def _f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    """Minimal Markdown table (no optional dependency)."""
    if df is None or df.empty:
        return "_(empty)_"
    d = df.head(max_rows)
    cells = [[str(c) for c in d.columns]] + [[_f(v, 4) if isinstance(v, (float, np.floating)) else str(v) for v in row]
                                             for row in d.itertuples(index=False)]
    cells = [[c.replace("|", "/").replace("\n", " ") for c in r] for r in cells]
    head = "| " + " | ".join(cells[0]) + " |"
    sep = "|" + "|".join(["---"] * len(cells[0])) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in cells[1:]])


def _ci(row, est="delta_r2", lo="delta_ci_low", hi="delta_ci_high"):
    return f"{_f(row.get(est))} [{_f(row.get(lo))}, {_f(row.get(hi))}]"


def status_line(ctx, sid) -> str | None:
    st = ctx.read_stage_json(sid, "_status", {}) or {}
    s = st.get("status", "never_run")
    if s != "completed":
        return f"Stage {sid} status: **{s}**" + (f" — {st.get('note')}" if st.get("note") else "") + ". No findings are reported for it."
    return None


def stage_findings(ctx, sid: str) -> list[str]:
    msg = status_line(ctx, sid)
    if msg:
        return [msg]
    out: list[str] = []
    if sid == "P16":
        vs = _t(ctx, sid, "variant_summary")
        sel = ctx.read_stage_json(sid, "selection", {}) or {}
        if vs is not None and not vs.empty:
            for r in vs.to_dict("records"):
                out.append(f"`{r['variant']}`: sample-rate AUROC (item-centred) {_f(r.get('sr_auc_item_centred'), 3)}, "
                           f"after out-of-sample domain residualization {_f(r.get('sr_auc_domain_residualized'), 3)}; "
                           f"correctness AUROC {_f(r.get('correct_auc'), 3)}; median |SMD| {_f(r.get('median_abs_smd'), 3)}; "
                           f"median perturbation stability {_f(r.get('stability_median'), 3)}; "
                           f"incremental R² over the comparator {_ci(r)}.")
        if sel:
            out.append(f"Selected variant: **{sel.get('selected_variant')}** ({sel.get('rule', '')}).")
        en = _t(ctx, sid, "equal_n_transfer")
        if en is not None and not en.empty:
            for r in en.to_dict("records"):
                out.append(f"Equal-size control `{r.get('variant')}` → target {r.get('target')}: within-domain R² "
                           f"{_f(r.get('r2_within'), 3)} vs transfer R² {_f(r.get('r2_transfer'), 3)} (gap {_f(r.get('gap_mean'), 3)}).")
    elif sid == "P17":
        cm = _t(ctx, sid, "clip_metrics")
        if cm is not None and not cm.empty:
            best = cm.sort_values("auroc", ascending=False).iloc[0]
            out.append(f"Best clip classifier: {best['feature_set']} / {best['model']} AUROC {_f(best['auroc'], 3)}, "
                       f"Brier {_f(best['brier'], 4)}, ECE {_f(best.get('ece'), 4)}.")
        ga = _t(ctx, sid, "audio_specific_gain")
        if ga is not None and not ga.empty:
            for r in ga.to_dict("records"):
                out.append(f"{r['feature_set']}: ΔAUROC over metadata-only {_f(r['delta_auroc'], 3)} "
                           f"[{_f(r['ci_low'], 3)}, {_f(r['ci_high'], 3)}] (student-block bootstrap).")
        sr = _t(ctx, sid, "student_soft_results")
        if sr is not None and not sr.empty:
            for r in sr.to_dict("records"):
                out.append(f"Student level `{r['name']}`: R² {_f(r.get('r2_mean'))}, Δ vs comparator {_ci(r)}"
                           + (f", Δ vs metadata-only soft features {_ci(r, 'delta_vs_meta', 'delta_vs_meta_lo', 'delta_vs_meta_hi')}"
                              if "delta_vs_meta" in r else "") + ".")
        out.append("Classification confidence is a learned summary, not a verified pronunciation-quality measure.")
    elif sid == "P18":
        el = _t(ctx, sid, "model_eligibility")
        if el is not None and not el.empty:
            ok = el[el["eligible"] == True]  # noqa: E712
            out.append(f"{len(ok)} of {len(el)} configured phonetic models were eligible (license + overlap verified).")
        cc = _t(ctx, sid, "correct_vs_incorrect")
        if cc is not None and not cc.empty and "q" in cc:
            out.append(f"{int((cc['q'] < 0.05).sum())} of {len(cc)} task × feature contrasts differ between correct and incorrect "
                       "responses (student-block bootstrap, FDR q < 0.05).")
        wc = _t(ctx, sid, "within_correct_association")
        if wc is not None and not wc.empty:
            conf = wc[wc.get("tier", "") == "confirmatory"] if "tier" in wc else wc
            n = int((conf["q"] < 0.05).sum()) if "q" in conf else 0
            out.append(f"Among officially correct responses, {n} phonetic quality features replicated in the confirmatory half "
                       "against the comparator's residuals (FDR q < 0.05).")
    elif sid == "P19":
        ra = _t(ctx, sid, "reference_clip_auc")
        if ra is not None and not ra.empty:
            b = ra.sort_values("auc_signal", ascending=False).iloc[0]
            out.append(f"Strongest reference deviation for correctness: {b['feature']} (|AUROC−0.5|·2 = {_f(b['auc_signal'], 3)}).")
        wc = _t(ctx, sid, "within_correct_association")
        if wc is not None and not wc.empty and "q" in wc:
            conf = wc[wc["tier"] == "confirmatory"] if "tier" in wc else wc
            out.append(f"Deviation among correct responses vs comparator residuals: {int((conf['q'] < 0.05).sum())} confirmatory "
                       "associations at FDR q < 0.05.")
        sr = _t(ctx, sid, "student_results")
        if sr is not None and not sr.empty:
            for r in sr.to_dict("records"):
                out.append(f"`{r['name']}`: R² {_f(r.get('r2_mean'))}, Δ vs comparator {_ci(r)}.")
    elif sid == "P20":
        lm = _t(ctx, sid, "layer_metrics")
        sel = ctx.read_stage_json(sid, "selection", {}) or {}
        if lm is not None and not lm.empty:
            b = lm.sort_values("correct_auc", ascending=False).iloc[0]
            out.append(f"Most correctness-informative layer: {b['model']} L{b['layer']} ({b['pooling']}) AUROC {_f(b['correct_auc'], 3)}, "
                       f"domain AUROC {_f(b['sr_auc'], 3)}, content accuracy {_f(b['content_acc'], 3)} (chance {_f(b['content_chance'], 3)}).")
        if sel:
            out.append(f"Selected representation for P21: {sel.get('model')} layer {sel.get('layer')} ({sel.get('pooling')}); rule: {sel.get('rule')}.")
        sr = _t(ctx, sid, "student_results")
        if sr is not None and not sr.empty:
            for r in sr.to_dict("records"):
                out.append(f"`{r['name']}`: Δ vs comparator {_ci(r)}.")
    elif sid == "P21":
        fv = _t(ctx, sid, "family_value")
        if fv is not None and not fv.empty:
            for r in fv.to_dict("records"):
                out.append(f"{r['family']}: Δ R² {_ci(r)} (q={_f(r.get('q'), 3)}); school-grouped Δ {_f(r.get('delta_school'))}; "
                           f"verdict: **{r.get('verdict')}**.")
        mc = _t(ctx, sid, "model_comparison_p21")
        if mc is not None and not mc.empty:
            b = mc.sort_values("r2_mean", ascending=False).iloc[0]
            out.append(f"Highest mean repeat R²: `{b['name']}` {_f(b['r2_mean'])} (Δ {_ci(b)}).")
    elif sid == "P22":
        de = _t(ctx, sid, "deployability")
        if de is not None and not de.empty:
            q = de[de["deployable"] == False]  # noqa: E712
            out.append(f"{len(de) - len(q)} deployable, {len(q)} quarantined models.")
        tg = _t(ctx, sid, "task_ensemble_diagnosis")
        if tg is not None and not tg.empty:
            for r in tg.to_dict("records"):
                out.append(f"Task-ensemble diagnosis — {r['finding']}: {_f(r['value'], 3)} {r.get('detail', '')}")
        rc = _t(ctx, sid, "recovery_matrix")
        if rc is not None and not rc.empty:
            rec = rc[rc["recovered"] == True]  # noqa: E712
            for pop, d in rec.groupby("population"):
                out.append(f"`{pop}` recovered by: " + ", ".join(sorted(d["model"].astype(str))))
        ca = _t(ctx, sid, "candidates")
        if ca is not None and not ca.empty:
            for r in ca.to_dict("records"):
                out.append(f"{r['role']}: {r.get('model') or 'not available'}" + (f" — submission {r.get('submission_status')}" if r.get("submission_status") else ""))
    if not out:
        out.append("Stage completed but produced no summary tables.")
    return out


def conclusions(ctx) -> list[str]:
    """Answer to the central question, derived from P21 family_value and P22 candidates only."""
    fv = _t(ctx, "P21", "family_value")
    if fv is None or fv.empty:
        return ["P21 has not produced family-level evidence yet; the central question cannot be answered from this run."]
    genuine = fv[fv["verdict"] == "genuine_incremental_value"]
    out = []
    if genuine.empty:
        out.append("No V2 representation produced genuine incremental predictive value over the strongest tabular comparator "
                   "(paired bootstrap CI ≤ 0 or not replicated under school-grouped validation).")
    else:
        out.append("Representations with genuine incremental value over the strongest tabular comparator: "
                   + ", ".join(f"{r['family']} (Δ {_ci(r)})" for r in genuine.to_dict("records")) + ".")
    weak = fv[fv["verdict"] == "suggestive"]
    if not weak.empty:
        out.append("Suggestive but not confirmed: " + ", ".join(weak["family"].astype(str)) + ".")
    return out


def recommendations(ctx) -> list[str]:
    """Next steps implied by THIS run's tables (rules, not opinions; nothing is assumed for missing stages)."""
    out = []
    for sid, what in (("P18", "phonetic"), ("P20", "foundation-model")):
        st = ctx.read_stage_json(sid, "_status", {}) or {}
        summ = st.get("summary") or {}
        if st.get("status") != "completed" or summ.get("status") == "unavailable":
            out.append(f"{sid} ({what}) produced no evidence in this run ({st.get('status', 'never_run')}"
                       f"{'; ' + str(summ.get('reason')) if summ.get('reason') else ''}). Download eligible checkpoints locally, "
                       "verify licence and training-data overlap in configs/foundation_models.yaml, then run V2_PHONETIC / V2_FOUNDATION.")
    vs = _t(ctx, "P16", "variant_summary")
    sel = ctx.read_stage_json("P16", "selection", {}) or {}
    if vs is not None and not vs.empty and sel.get("selected_variant") in set(vs["variant"]):
        r = vs[vs["variant"] == sel["selected_variant"]].iloc[0]
        if pd.notna(r.get("sr_auc_item_centred")) and float(r["sr_auc_item_centred"]) > 0.7:
            out.append(f"Recording-domain information remains high after harmonization (sample-rate AUROC "
                       f"{_f(r['sr_auc_item_centred'], 3)} for {sel['selected_variant']}): keep domain residualization and "
                       "report sample-rate transfer separately.")
    fv = _t(ctx, "P21", "family_value")
    if fv is not None and not fv.empty:
        g = fv[fv["verdict"] == "genuine_incremental_value"]
        s = fv[fv["verdict"] == "suggestive"]
        if not g.empty:
            out.append("Carry forward: " + ", ".join(g["family"].astype(str)) + " (confirmed incremental value).")
        if not s.empty:
            out.append("Replicate before use: " + ", ".join(s["family"].astype(str))
                       + " (positive but not confirmed under the pre-registered rule).")
        if g.empty and s.empty:
            out.append("No V2 family improved on the comparator: prefer the strongest tabular model for deployment.")
    de = _t(ctx, "P22", "deployability")
    if de is not None and not de.empty:
        q = de[de["deployable"] == False]  # noqa: E712
        if not q.empty:
            out.append("Quarantined (do not deploy): " + ", ".join(q["model"].astype(str).head(12))
                       + (" …" if len(q) > 12 else ""))
    ca = _t(ctx, "P22", "candidates")
    if ca is not None and not ca.empty and (ca["role"] == "final_ensemble").any():
        r = ca[ca["role"] == "final_ensemble"].iloc[0]
        out.append(f"Champion for submission: {r.get('model')} (role final_ensemble"
                   + (f"; {r.get('note')}" if isinstance(r.get("note"), str) else "") + ").")
    return out or ["No stage has produced results yet."]


def _fig_html(path: str, caption: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption or p.stem)}"/><figcaption>{html.escape(caption or p.stem)}</figcaption></figure>'


def build_report(ctx, max_figures_per_stage: int = 12) -> dict:
    from .registry import v2_registries

    reg = v2_registries(ctx)
    md = [f"# WildSpeech V2 — Robust Acoustic & Phonetic Intelligence", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} from runtime artefacts in `{ctx.paths.v2_root}`.", "",
          "**Central question.** Can robust, phonetic and expected-text-conditioned speech representations explain future "
          "reading outcomes beyond the strongest conventional literacy-assessment baseline?", ""]
    from .baseline import baseline_table

    md += ["## Comparator", "", md_table(baseline_table(ctx)), ""]
    for sid in V2_STAGE_IDS:
        md += [f"## {sid} · {TITLES[sid]}", ""] + [f"- {s}" for s in stage_findings(ctx, sid)] + [""]
    md += ["## Conclusions (derived from P21/P22 tables)", ""] + [f"- {s}" for s in conclusions(ctx)] + [""]
    md += ["## Recommendations (rules applied to this run)", ""] + [f"- {s}" for s in recommendations(ctx)] + [""]
    hyp = reg.hypotheses.frame()
    if not hyp.empty:
        cols = [c for c in ("hypothesis_id", "statement", "status", "evidence_tier", "effect", "ci_low", "ci_high", "q_value", "evidence") if c in hyp]
        md += ["## Hypotheses (exploratory vs confirmatory)", "", md_table(hyp[cols]), ""]
    md += ["## Safeguards", "",
           "- Frozen student-disjoint validation registry; nested student-disjoint cross-fitting for every supervised clip feature.",
           "- Residualizers, templates, item statistics, PCA, selection, calibration, stacking and residual shrinkage fitted inside training folds.",
           "- Test data never used for discovery, normalisation, pseudo-labelling or selection; submissions disabled by default.",
           "- Soft-correctness confidence is not a verified pronunciation-quality measure; indices are experimental constructs.", ""]
    text = "\n".join(md)
    md_path = ctx.paths.v2_root / "scientific_report.md"
    md_path.write_text(text, encoding="utf-8")
    try:
        import markdown

        body = markdown.markdown(text, extensions=["tables"])
    except ImportError:
        body = "<pre>" + html.escape(text) + "</pre>"
    figs = reg.figures.frame()
    fig_html = []
    if not figs.empty:
        for sid in V2_STAGE_IDS:
            d = figs[figs["stage"] == sid].head(max_figures_per_stage)
            if d.empty:
                continue
            fig_html.append(f"<h2>{sid} figures</h2>")
            fig_html += [_fig_html(r["path"], r.get("caption") or r["name"]) for r in d.to_dict("records")]
    style = ("body{font-family:Segoe UI,system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#0b0b0b;"
             "background:#fcfcfb}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #e1e0d9;padding:3px 6px}"
             "figure{margin:12px 0}img{max-width:100%}figcaption{font-size:12px;color:#52514e}")
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>WildSpeech V2 report</title><style>{style}</style></head><body>{body}{''.join(fig_html)}</body></html>"
    html_path = ctx.paths.v2_root / "scientific_report.html"
    html_path.write_text(page, encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path), "n_figures": int(len(figs))}
