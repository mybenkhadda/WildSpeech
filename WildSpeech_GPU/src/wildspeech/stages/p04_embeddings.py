"""P04 — speech foundation-model embeddings (opt-in; licence/overlap gated; fully local)."""
from __future__ import annotations

from ..embeddings import EmbeddingRunner
from ..model_registry import eligibility_table, load_registry

S = "P04"


def run(ctx) -> dict:
    ctx.save_table(eligibility_table(load_registry(ctx.paths.configs)), S, "model_eligibility")
    r = EmbeddingRunner(ctx)
    out = {"train": r.run("train")}
    if ctx.cfg.get_path("audio.extract_test", False):
        out["test"] = r.run("test")
    ctx.save_json(out, S, "embedding_run_summary")
    return {k: {m: v.get("status") for m, v in s["models"].items()} for k, s in out.items()}
