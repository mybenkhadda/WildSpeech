"""Budget-controlled experiment scheduling: tasks run in priority order until the evaluation-count or
wall-clock budget is exhausted; every skipped task is recorded with its reason (never silently dropped)."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass(order=True)
class Task:
    priority: int
    name: str = field(compare=False)
    fn: Callable[[], Any] = field(compare=False, repr=False)
    group: str = field(default="", compare=False)
    cost: int = field(default=1, compare=False)


class ExperimentScheduler:
    def __init__(self, max_evals: int | None = None, time_budget_s: float | None = None, logger=None):
        self.max_evals = max_evals
        self.time_budget_s = time_budget_s
        self.logger = logger
        self.tasks: list[Task] = []
        self.log: list[dict] = []
        self.results: dict[str, Any] = {}

    def add(self, name: str, fn: Callable[[], Any], priority: int = 100, group: str = "", cost: int = 1) -> None:
        self.tasks.append(Task(priority, name, fn, group, cost))

    def run(self) -> pd.DataFrame:
        t0 = time.perf_counter()
        used = 0
        for task in sorted(self.tasks):
            elapsed = time.perf_counter() - t0
            if self.max_evals is not None and used + task.cost > self.max_evals:
                self.log.append({"task": task.name, "group": task.group, "status": "skipped_budget_evals", "seconds": 0.0})
                continue
            if self.time_budget_s is not None and elapsed > self.time_budget_s:
                self.log.append({"task": task.name, "group": task.group, "status": "skipped_budget_time", "seconds": 0.0})
                continue
            ts = time.perf_counter()
            try:
                self.results[task.name] = task.fn()
                status, err = "ok", ""
            except Exception as e:  # noqa: BLE001 - recorded, scheduling continues
                status, err = "failed", f"{type(e).__name__}: {e}"
                if self.logger:
                    self.logger.warning("task %s failed: %s\n%s", task.name, err, traceback.format_exc())
            used += task.cost
            self.log.append({"task": task.name, "group": task.group, "status": status, "error": err,
                             "seconds": time.perf_counter() - ts})
        return pd.DataFrame(self.log)
