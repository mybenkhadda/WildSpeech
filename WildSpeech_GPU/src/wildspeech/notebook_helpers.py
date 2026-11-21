"""Thin helpers so notebooks stay declarative: setup, run a stage, display its tables and figures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def setup(profile: str = "FAST", overrides: dict | None = None, root: Path | None = None):
    from .context import build_context
    from .orchestrator import Pipeline

    ctx = build_context(profile, overrides=overrides, root=root)
    return ctx, Pipeline(ctx)


def run_stage(pipe, sid: str, force: bool = False) -> pd.DataFrame:
    return pipe.run([sid], force=[sid] if force else [])


def show_stage(ctx, sid: str, tables: list[str] | None = None, max_rows: int = 15, figures: bool = True, max_figures: int = 12) -> None:
    from IPython.display import Image, Markdown, display

    d = ctx.stage_dir(sid)
    st = ctx.read_stage_json(sid, "_status", {}) or {}
    display(Markdown(f"### {sid} — status: `{st.get('status', 'not run')}` {st.get('note', '')}"))
    names = tables or sorted(p.stem for p in (d / "tables").glob("*.csv") if not p.stem.startswith("fig_"))
    for n in names:
        df = ctx.read_stage_table(sid, n)
        if df is None:
            continue
        display(Markdown(f"**{n}** — {len(df)} rows"))
        display(df.head(max_rows))
    if figures:
        for p in sorted((d / "figures").glob("*.png"))[:max_figures]:
            display(Markdown(f"`{p.name}`"))
            display(Image(filename=str(p)))
