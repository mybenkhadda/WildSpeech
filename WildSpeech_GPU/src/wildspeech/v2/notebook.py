"""Thin notebook helpers for notebooks/WILDSPEECH_V2_ROBUST_PHONETIC_EXPERIMENT.ipynb (no computation here)."""
from __future__ import annotations

import json
import time

import pandas as pd


def setup_v2(profile: str = "V2_CORE", overrides: dict | None = None, root=None):
    from ..context import build_context
    from ..orchestrator import Pipeline

    ctx = build_context(profile, overrides=overrides, root=root)
    return ctx, Pipeline(ctx)


def run_stage(pipe, sid: str, enabled: bool, force: list[str] | tuple = (), log: list | None = None) -> pd.DataFrame | None:
    """Runs `sid` (dependencies resolved, cached when unchanged) only when `enabled` is True."""
    if not enabled:
        return None
    t0 = time.perf_counter()
    res = pipe.run([sid], force=[s for s in force if s == sid])
    if log is not None:
        log.append(res.assign(wall_s=round(time.perf_counter() - t0, 1)))
    return res


def show_stage_v2(ctx, sid: str, key_tables=(), rows: int = 12, all_tables: bool = False, figures: bool = True,
                  max_figures: int = 40) -> None:
    from IPython.display import Image, Markdown, display

    from .research_report_v2 import TITLES, stage_findings

    st = ctx.read_stage_json(sid, "_status", {}) or {}
    status = st.get("status", "never_run")
    display(Markdown(f"### {sid} · {TITLES.get(sid, '')} · `{status}`"
                     + (f" · {st.get('seconds', 0):.0f}s" if st.get("seconds") else "") + (f" — {st.get('note')}" if st.get("note") else "")))
    if status == "failed":
        print(st.get("traceback", ""))
        return
    if st.get("summary"):
        display(pd.DataFrame([{"key": k, "value": json.dumps(v, default=str)[:300] if isinstance(v, (dict, list)) else v}
                              for k, v in st["summary"].items()]))
    display(Markdown("**Findings (computed from this run's tables)**\n\n" + "\n".join(f"- {s}" for s in stage_findings(ctx, sid))))
    d = ctx.stage_dir(sid)
    names = sorted(p.stem for p in (d / "tables").glob("*.csv") if not p.stem.startswith("fig_"))
    show = [n for n in key_tables if n in names] + ([n for n in names if n not in key_tables] if all_tables else [])
    for n in show:
        df = ctx.read_stage_table(sid, n)
        if df is not None:
            display(Markdown(f"**{n}** — {len(df)} rows × {df.shape[1]} cols"))
            display(df.head(rows))
    if figures:
        for p in sorted((d / "figures").glob("*.png"))[:max_figures]:
            display(Markdown(f"`{p.stem}`"))
            display(Image(filename=str(p)))


def figure_atlas(ctx, stages=None, max_per_stage: int = 60) -> None:
    from IPython.display import Image, Markdown, display

    from .registry import v2_registries

    idx = v2_registries(ctx).figures.frame()
    if idx.empty:
        print("figure index is empty — no V2 figure has been produced yet")
        return
    for sid, d in idx.groupby("stage"):
        if stages and sid not in stages:
            continue
        display(Markdown(f"## {sid} — {len(d)} figures"))
        for r in d.head(max_per_stage).to_dict("records"):
            display(Markdown(f"**{r['name']}** — {r.get('caption') or ''}"))
            display(Image(filename=r["path"]))
