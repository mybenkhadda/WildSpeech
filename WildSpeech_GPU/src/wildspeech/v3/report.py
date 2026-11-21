"""outputs/V3/scientific_report.html — built EXCLUSIVELY from runtime tables and figures.

Every number in the report is read from outputs/V3/**/tables/*.csv or the V3 registries at build time; sections
whose tables do not exist say so ("not executed") instead of showing placeholders. Figures are embedded (base64)
so the file is self-contained. Light/dark themes follow the viewer's setting.
"""
from __future__ import annotations

import base64
import html
import time
from pathlib import Path

import pandas as pd

SECTIONS = [
    ("P23", "P23 — Multimodal evaluation audit", ["baseline_reproduction", "fusion_verdicts", "v2_budget_cap", "task_ensemble_summary",
                                                   "lineage_audit", "v3_repairs"]),
    ("P24", "P24 — Local ASR models and inference", ["model_eligibility", "throughput", "asr_outputs_inventory", "transcription_examples"]),
    ("P25", "P25 — Phonetic features and validation", ["feature_tables", "child_vs_general_auc", "correctness_validation_auc"]),
    ("P26", "P26 — Correct-response speech intelligence", ["primary_model_roles", "clip_correctness_increment",
                                                           "residual_associations", "student_incremental_r2"]),
    ("P27", "P27 — ASR disagreement and consensus", ["variance_decomposition", "student_disagreement_consistency", "disagreement_by_task"]),
    ("P28", "P28 — Compact representations", ["family_availability", "exploratory_best_config", "fallback_share_by_family",
                                             "representation_results"]),
    ("P29", "P29 — Reading prediction, ablation and robustness", ["baseline_by_scheme", "validated_families", "champion_comparison",
                                                                  "ablation_results", "transfer_robustness", "family_stability",
                                                                  "confounding", "champion_subgroups"]),
]
KEY_COLS = {"ablation_results": ["experiment", "name", "scheme", "status", "model", "fusion", "r2_mean", "delta_r2",
                                 "delta_ci_low", "delta_ci_high", "q", "reportable", "fallback_folds"],
            "representation_results": ["experiment", "name", "method", "budget", "status", "delta_r2", "delta_ci_low",
                                       "delta_ci_high", "fallback_folds"],
            "residual_associations": ["feature", "tier", "rho", "p", "q", "partial_rho_ability", "same_sign", "n"],
            "correctness_validation_auc": ["feature", "task", "auc_oriented", "coverage", "n"]}

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--accent:#2a78d6;--bad:#d03b3b;--good:#006300}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--accent:#3987e5;--bad:#e66767;--good:#0ca30c}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--accent:#3987e5;--bad:#e66767;--good:#0ca30c}
body{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:16px;line-height:1.45}
main{max-width:1180px;margin:0 auto}
h1{font-size:1.6rem}h2{font-size:1.2rem;border-bottom:1px solid var(--grid);padding-bottom:4px;margin-top:2rem}
h3{font-size:1rem;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:12px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;font-size:.8rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:3px 8px;text-align:left;white-space:nowrap}
th{color:var(--ink2)}
img{max-width:100%;height:auto;border-radius:6px;background:#fcfcfb}
.muted{color:var(--muted)}.na{color:var(--bad)}
figure{margin:10px 0}figcaption{color:var(--ink2);font-size:.85rem}
"""


def _table_html(df: pd.DataFrame, name: str, max_rows: int = 40) -> str:
    cols = [c for c in KEY_COLS.get(name, list(df.columns)) if c in df.columns]
    d = df[cols].head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    note = f'<p class="muted">{len(df)} rows{" (first %d shown)" % max_rows if len(df) > max_rows else ""} — full table: tables/{name}.csv</p>'
    return d.to_html(index=False, escape=True, border=0) + note


def _img(path: Path) -> str:
    try:
        b = base64.b64encode(Path(path).read_bytes()).decode()
        return f'<img alt="{html.escape(Path(path).stem)}" src="data:image/png;base64,{b}">'
    except OSError:
        return '<p class="na">figure file missing</p>'


def build_report(ctx) -> str:
    from .registry import v3_registries

    reg = v3_registries(ctx)
    parts = [f"<h1>WildSpeech V3 — Child ASR, Phonetic Intelligence &amp; Reading Prediction</h1>",
             f'<p class="muted">Generated {time.strftime("%Y-%m-%d %H:%M")} from runtime artefacts in outputs/V3 '
             f'(profile {html.escape(ctx.profile)}). Nothing in this report is typed by hand.</p>']
    hyp = reg.hypotheses.frame()
    parts.append("<h2>Pre-registered hypotheses</h2>")
    if hyp.empty:
        parts.append('<p class="na">no hypotheses registered</p>')
    else:
        cols = [c for c in ("hypothesis_id", "statement", "status", "evidence_tier", "evidence", "effect", "ci_low", "ci_high", "q_value") if c in hyp]
        parts.append(f'<div class="card">{_table_html(hyp[cols], "hypotheses", 50)}</div>')
    figs = reg.figures.frame()
    for sid, title, tables in SECTIONS:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        st = ctx.read_stage_table(sid, "not_executed")
        if st is not None and not st.empty:
            parts.append(f'<p class="na">NOT_EXECUTED — {html.escape(str(st.iloc[0].get("reason")))}</p>')
        found = False
        for t in tables:
            df = ctx.read_stage_table(sid, t)
            if df is None:
                continue
            found = True
            parts.append(f'<h3>{html.escape(t.replace("_", " "))}</h3><div class="card">{_table_html(df, t)}</div>')
        if not found and (st is None or st.empty):
            parts.append('<p class="na">stage not executed yet (no tables)</p>')
        if not figs.empty and "stage" in figs:
            for r in figs[figs["stage"] == sid].sort_values("name").to_dict("records"):
                parts.append(f'<figure class="card">{_img(r["path"])}<figcaption>{html.escape(str(r.get("caption") or r["name"]))}'
                             f' — data: {html.escape(str(r.get("data_csv") or "n/a"))}</figcaption></figure>')
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>WildSpeech V3 Report</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")
    out = ctx.paths.v3_root / "scientific_report.html"
    out.write_text(doc, encoding="utf-8")
    return str(out)


def export_root_tables(ctx) -> dict:
    """Mirror the key registries to the outputs/V3 root names the project layout expects."""
    from .registry import v3_registries

    reg = v3_registries(ctx)
    root = ctx.paths.v3_root
    out = {}
    for name, r in (("model_registry", reg.models), ("hypothesis_registry", reg.hypotheses),
                    ("experiment_registry", reg.experiments), ("ablation_results", reg.ablations)):
        r.save()
        out[name] = str(root / f"{name}.csv")
    reg.features.save()
    out["phonetic_feature_registry"] = str(root / "phonetic_feature_registry.parquet")
    return out
