"""V3 registries in outputs/V3 (Parquet + CSV mirror via wildspeech.registry.Registry).

outputs/V3/model_registry.csv                  every configured ASR model + eligibility verdict + provenance
outputs/V3/phonetic_feature_registry.parquet   every clip/student-level V3 feature (family, source model, stage,
                                               level, label_free flag, description)
outputs/V3/hypothesis_registry.csv             pre-registered V3 hypotheses (configs/hypotheses_v3.yaml) + evidence
outputs/V3/experiment_registry.csv             every evaluated (experiment, scheme) with metrics and fallback flags
outputs/V3/ablation_results.csv                ASR00-ASR12 family ablations, budgets, controls
outputs/V3/figure_index.csv                    every figure with its data table and caption
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import load_yaml
from ..registry import Registry

EVIDENCE_TIERS = ("exploratory", "confirmatory")
STATUSES = ("supported", "not_supported", "inconclusive", "proposed", "not_tested", "unavailable")


class V3Registries:
    def __init__(self, root: Path):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.models = Registry(root / "model_registry.parquet", ["model_key"])
        self.features = Registry(root / "phonetic_feature_registry.parquet", ["feature"])
        self.hypotheses = Registry(root / "hypothesis_registry.parquet", ["hypothesis_id"])
        self.experiments = Registry(root / "experiment_registry.parquet", ["experiment_id"])
        self.ablations = Registry(root / "ablation_results.parquet", ["ablation_id"])
        self.figures = Registry(root / "figure_index.parquet", ["figure_id"])

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

    def log_model(self, model_key: str, **fields) -> None:
        self.models.upsert([{"model_key": model_key, **fields}])

    def log_experiment(self, experiment_id: str, **fields) -> None:
        self.experiments.upsert([{"experiment_id": experiment_id, **fields}])

    def log_ablation(self, ablation_id: str, **fields) -> None:
        self.ablations.upsert([{"ablation_id": ablation_id, **fields}])

    def register_features(self, columns, stage: str, source: str, level: str, label_free: bool,
                          description: str = "") -> None:
        rows = [{"feature": str(c), "family": str(c).split("__", 1)[0] if "__" in str(c) else str(c).split("::", 1)[0],
                 "stage": stage, "source": source, "level": level, "label_free": bool(label_free),
                 "description": description} for c in columns]
        if rows:
            self.features.upsert(rows)


def v3_registries(ctx) -> V3Registries:
    key = "v3_registries"
    if key not in ctx._cache:
        reg = V3Registries(ctx.paths.v3_root)
        hyp = ctx.paths.configs / "hypotheses_v3.yaml"
        if hyp.exists():
            reg.seed_hypotheses(load_yaml(hyp) or [])
        ctx._cache[key] = reg
    return ctx._cache[key]


def save_figure_v3(ctx, stage: str, name: str, fig, table: pd.DataFrame | None = None, caption: str = "") -> Path:
    """Save the figure in the stage folder AND outputs/V3/figures, its data table next to it, and index it."""
    import matplotlib.pyplot as plt

    dpi = int(ctx.cfg.get_path("v3.project.figure_dpi", 150))
    p_stage = ctx.stage_dir(stage) / "figures" / f"{name}.png"
    p_central = ctx.paths.v3_root / "figures" / f"{stage}_{name}.png"
    fig.savefig(p_stage, dpi=dpi, bbox_inches="tight")
    fig.savefig(p_central, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    data_path = None
    if table is not None:
        keep_index = table.index.name is not None or not isinstance(table.index, pd.RangeIndex)
        data_path = ctx.save_table(table.reset_index() if keep_index else table, stage, f"fig_{name}")
    v3_registries(ctx).figures.upsert([{"figure_id": f"{stage}:{name}", "stage": stage, "name": name,
                                        "path": str(p_central), "stage_path": str(p_stage),
                                        "data_csv": str(data_path) if data_path else None, "caption": caption,
                                        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}])
    return p_central
