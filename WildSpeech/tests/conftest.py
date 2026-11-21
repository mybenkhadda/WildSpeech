"""Shared fixtures. Tests use tiny synthetic tables only - never competition data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def cfg():
    from wildspeech.config import BASE_CONFIG, EXTENDED_DEFAULTS

    c = {**BASE_CONFIG, **EXTENDED_DEFAULTS}
    c.update({"environment": "LOCAL_LINUX", "cv_folds": 3, "cv_repeats": 2, "bootstrap_iterations": 20, "inner_cv_folds": 3,
              "run_mode": "FULL", "save_figures": False})
    return c


@pytest.fixture
def ctx(tmp_path, cfg):
    from wildspeech.io_utils import ArtifactStore
    from wildspeech.logging_utils import get_logger
    from wildspeech.paths import build_project_tree
    from wildspeech.pipeline import RunContext
    from wildspeech.visualization import FigureRegistry

    cfg = dict(cfg, output_root=str(tmp_path / "WildSpeech"), local_root=str(tmp_path / "local"))
    paths = build_project_tree(cfg, "LOCAL_LINUX")
    store = ArtifactStore(paths.cache_dir("manifests") / "artifact_index.json")
    figs = FigureRegistry(paths.outputs / "figure_index.csv", enabled=False)
    return RunContext(cfg, paths, get_logger("wildspeech.test"), store, figs)


@pytest.fixture
def synthetic_students():
    rng = np.random.default_rng(0)
    n = 60
    ids = [f"S{i:03d}" for i in range(n)]
    theta = rng.normal(size=(n, 3))
    y = theta @ np.array([0.5, 0.3, 0.2]) + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"student_id": ids, "school_id": [f"SCH{i % 4}" for i in range(n)],
                         "instrument": ["A" if i % 2 else "B" for i in range(n)], "target": y,
                         "theta_blending": theta[:, 0], "theta_deletion": theta[:, 1], "theta_repetition": theta[:, 2]}).set_index("student_id")
