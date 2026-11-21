"""Centralised V2 registries in outputs/V2 and the figure helper that keeps figure_index.csv complete.

outputs/V2/feature_registry.parquet      every student-level V2 feature (family, provider, stage, description)
outputs/V2/hypothesis_registry.csv       pre-registered V2 hypotheses + evidence tier (exploratory/confirmatory)
outputs/V2/experiment_registry.csv       every evaluated (spec, scheme) with metrics
outputs/V2/model_comparison.csv          model-level comparison vs the strongest tabular reference
outputs/V2/ablation_results.csv          family ablations / budgets / pairwise / leave-one-out
outputs/V2/failure_registry.parquet      failure populations and recovery
outputs/V2/figure_index.csv              every figure with its data table and caption
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import load_yaml
from ..registry import Registry

EVIDENCE_TIERS = ("exploratory", "confirmatory", "leaderboard")
STATUSES = ("supported", "not_supported", "inconclusive", "proposed", "not_tested", "unavailable")


class V2Registries:
    def __init__(self, root: Path):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.features = Registry(root / "feature_registry.parquet", ["feature"])
        self.hypotheses = Registry(root / "hypothesis_registry.parquet", ["hypothesis_id"])
        self.experiments = Registry(root / "experiment_registry.parquet", ["experiment_id"])
        self.models = Registry(root / "model_comparison.parquet", ["model_id"])
        self.ablations = Registry(root / "ablation_results.parquet", ["ablation_id"])
        self.failures = Registry(root / "failure_registry.parquet", ["failure_id"])
        self.figures = Registry(root / "figure_index.parquet", ["figure_id"])

    # -------------------------------------------------------------- hypotheses
    def seed_hypotheses(self, items: list[dict]) -> None:
        existing = set(self.hypotheses.frame().get("hypothesis_id", pd.Series(dtype=str)).astype(str))
        rows = [{"hypothesis_id": h["id"], "statement": h["statement"], "family": h.get("family"),
                 "tested_in": h.get("tested_in"), "rule": h.get("rule"), "tier": h.get("tier", "confirmatory"),
                 "status": "proposed"} for h in items if h["id"] not in existing]
        if rows:
            self.hypotheses.upsert(rows)

    def update_hypothesis(self, hid: str, status: str, stage: str, evidence: str, tier: str = "confirmatory",
                          effect: float | None = None, ci_low: float | None = None, ci_high: float | None = None,
                          q_value: float | None = None, n: int | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown hypothesis status {status}")
        if tier not in EVIDENCE_TIERS:
            raise ValueError(f"unknown evidence tier {tier}")
        self.hypotheses.upsert([{"hypothesis_id": hid, "status": status, "evidence_stage": stage, "evidence": evidence,
                                 "evidence_tier": tier, "effect": effect, "ci_low": ci_low, "ci_high": ci_high,
                                 "q_value": q_value, "n": n}])

    # ----------------------------------------------------------- experiments
    def log_experiment(self, experiment_id: str, **fields) -> None:
        self.experiments.upsert([{"experiment_id": experiment_id, **fields}])

    def log_model(self, model_id: str, **fields) -> None:
        self.models.upsert([{"model_id": model_id, **fields}])

    def log_ablation(self, ablation_id: str, **fields) -> None:
        self.ablations.upsert([{"ablation_id": ablation_id, **fields}])

    def register_features(self, columns, stage: str, provider: str, description: str = "") -> None:
        from ..features.builder import family_of

        rows = [{"feature": c, "family": family_of(c), "stage": stage, "provider": provider,
                 "scope": c.split("__")[1] if c.count("__") >= 2 else "all", "description": description}
                for c in columns]
        if rows:
            self.features.upsert(rows)


def v2_registries(ctx) -> V2Registries:
    key = "v2_registries"
    if key not in ctx._cache:
        reg = V2Registries(ctx.paths.v2_root)
        hyp = ctx.paths.configs / "hypotheses_v2.yaml"
        if hyp.exists():
            reg.seed_hypotheses(load_yaml(hyp) or [])
        ctx._cache[key] = reg
    return ctx._cache[key]


def save_figure_v2(ctx, stage: str, name: str, fig, table: pd.DataFrame | None = None, caption: str = "",
                   section: str | None = None) -> Path:
    """Save the figure in the stage folder AND outputs/V2/figures, its data table next to it, and index it."""
    import matplotlib.pyplot as plt

    dpi = int(ctx.cfg.get_path("v2.project.figure_dpi", 150))
    p_stage = ctx.stage_dir(stage) / "figures" / f"{name}.png"
    p_central = ctx.paths.v2_root / "figures" / f"{stage}_{name}.png"
    fig.savefig(p_stage, dpi=dpi, bbox_inches="tight")
    fig.savefig(p_central, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    data_path = None
    if table is not None:
        keep_index = table.index.name is not None or not isinstance(table.index, pd.RangeIndex)
        data_path = ctx.save_table(table.reset_index() if keep_index else table, stage, f"fig_{name}")
    v2_registries(ctx).figures.upsert([{"figure_id": f"{stage}:{name}", "stage": stage, "name": name,
                                        "section": section or stage, "path": str(p_central),
                                        "stage_path": str(p_stage), "data_csv": str(data_path) if data_path else None,
                                        "caption": caption, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}])
    return p_central
