"""Exhaustive-but-budgeted feature-family ablations over the CV runner.

single_add      base + each family
leave_one_out   ALL minus each family
exhaustive      every subset of non-base families up to max_subset (ordered by size), until the evaluation or
                time budget is exhausted
forward         greedy forward selection over families
All deltas are paired against the base model's OOF predictions (same folds), with bootstrap CIs.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from ..validation.metrics import incremental_r2
from ..validation.runner import CVRunner, ModelSpec


class FamilyAblation:
    def __init__(self, runner: CVRunner, model: str, base: list[str], families: list[str], scheme: str = "repeated_kfold",
                 max_evals: int = 80, time_budget_s: float = 3600, n_boot: int = 500, stage: str = "P10"):
        self.runner, self.model, self.base, self.scheme = runner, model, list(base), scheme
        self.families = [f for f in families if f not in self.base]
        self.max_evals, self.time_budget_s, self.n_boot, self.stage = max_evals, time_budget_s, n_boot, stage
        self.results: dict[tuple, dict] = {}
        self.oofs: dict[tuple, pd.DataFrame] = {}
        self._t0 = time.perf_counter()
        self._evals = 0

    def _budget_ok(self) -> bool:
        return self._evals < self.max_evals and (time.perf_counter() - self._t0) < self.time_budget_s

    def evaluate(self, fams: tuple, kind: str) -> dict | None:
        key = tuple(sorted(fams))
        if key in self.results:
            return self.results[key]
        if not self._budget_ok():
            return None
        name = f"ABL_{self.model}_{'+'.join(key) or 'BASE'}"
        res = self.runner.evaluate(ModelSpec(name, self.model, list(key)), self.scheme, stage=self.stage)
        self._evals += 1
        if res.status != "ok":
            return None
        self.oofs[key] = res.oof
        row = {"families": "+".join(key), "n_families": len(key), "kind": kind, **res.summary}
        self.results[key] = row
        return row

    def run(self, max_subset: int = 3, forward_steps: int = 6) -> pd.DataFrame:
        base_key = tuple(sorted(self.base))
        self.evaluate(base_key, "base")
        for f in self.families:
            self.evaluate(tuple(self.base + [f]), "single_add")
        all_f = self.base + self.families
        self.evaluate(tuple(all_f), "all")
        for f in self.families:
            self.evaluate(tuple(x for x in all_f if x != f), "leave_one_out")
        # greedy forward
        cur = list(self.base)
        for _ in range(forward_steps):
            cands = [(f, self.evaluate(tuple(cur + [f]), "forward")) for f in self.families if f not in cur]
            cands = [(f, r) for f, r in cands if r]
            if not cands:
                break
            f_best, r_best = max(cands, key=lambda t: t[1]["r2_mean"])
            cur_r2 = self.results.get(tuple(sorted(cur)), {}).get("r2_mean", -np.inf)
            if r_best["r2_mean"] <= cur_r2:
                break
            cur.append(f_best)
        # exhaustive subsets by size
        for size in range(2, max_subset + 1):
            for combo in itertools.combinations(self.families, size):
                if not self._budget_ok():
                    break
                self.evaluate(tuple(self.base + list(combo)), f"subset_{size}")
        table = pd.DataFrame(list(self.results.values()))
        base_oof = self.oofs.get(base_key)
        if base_oof is not None:
            deltas = []
            for key, oof in self.oofs.items():
                d = incremental_r2(base_oof, oof, self.n_boot, 0) if key != base_key else {"delta_r2": 0.0}
                deltas.append({"families": "+".join(key), **d})
            table = table.merge(pd.DataFrame(deltas), on="families", how="left")
        table["forward_path"] = "+".join(cur)
        table["evals_used"], table["budget_exhausted"] = self._evals, not self._budget_ok()
        return table.sort_values("r2_mean", ascending=False).reset_index(drop=True)
