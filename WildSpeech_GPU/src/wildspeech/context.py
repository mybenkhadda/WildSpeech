"""Run context shared by all stages: config, paths, logger, registries, device and lazily-loaded data."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, load_config, load_yaml
from .io_utils import atomic_write_csv, atomic_write_parquet, read_json, write_json
from .logging_utils import get_logger
from .paths import ProjectPaths
from .registry import Registries


def resolve_device(pref: str = "auto") -> str:
    try:
        import torch

        if pref in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class RunContext:
    cfg: Config
    paths: ProjectPaths
    profile: str
    default_stages: list[str] | None
    logger: Any
    registries: Registries
    device: str
    _bundle: Any = None
    _cache: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ data
    def bundle(self):
        if self._bundle is None:
            from .data.schema import load_data_bundle

            self._bundle = load_data_bundle(self.paths, self.cfg, self.logger)
        return self._bundle

    @property
    def seed(self) -> int:
        return int(self.cfg.get_path("project.seed", 42))

    # ----------------------------------------------------------------- outputs
    def stage_dir(self, stage_id: str) -> Path:
        return self.paths.stage_dir(stage_id)

    def save_table(self, df: pd.DataFrame, stage_id: str, name: str, index: bool = False, parquet: bool = False) -> Path:
        d = self.stage_dir(stage_id) / "tables"
        path = atomic_write_csv(df, d / f"{name}.csv", index=index)
        if parquet:
            atomic_write_parquet(df, d / f"{name}.parquet", index=index)
        return path

    def save_parquet(self, df: pd.DataFrame, stage_id: str, name: str, index: bool = False) -> Path:
        return atomic_write_parquet(df, self.stage_dir(stage_id) / "tables" / f"{name}.parquet", index=index)

    def read_stage_table(self, stage_id: str, name: str) -> pd.DataFrame | None:
        """None when the table does not exist. An EMPTY result file (zero bytes, no columns, marker only) returns an
        empty DataFrame with attrs['status'] == 'NO_RESULTS' instead of raising pandas.errors.EmptyDataError."""
        from .io_utils import read_csv_safe, read_parquet_safe

        d = self.stage_dir(stage_id) / "tables"
        if (d / f"{name}.parquet").exists():
            return read_parquet_safe(d / f"{name}.parquet")
        if (d / f"{name}.csv").exists():
            return read_csv_safe(d / f"{name}.csv")
        return None

    def table_path(self, stage_id: str, name: str) -> Path | None:
        d = self.stage_dir(stage_id) / "tables"
        for ext in (".parquet", ".csv"):
            if (d / f"{name}{ext}").exists():
                return d / f"{name}{ext}"
        return None

    def save_json(self, obj: Any, stage_id: str, name: str) -> Path:
        return write_json(self.stage_dir(stage_id) / f"{name}.json", obj)

    def read_stage_json(self, stage_id: str, name: str, default: Any = None) -> Any:
        return read_json(self.stage_dir(stage_id) / f"{name}.json", default)

    def save_figure(self, fig, stage_id: str, name: str) -> Path:
        import matplotlib.pyplot as plt

        path = self.stage_dir(stage_id) / "figures" / f"{name}.png"
        fig.savefig(path, dpi=int(self.cfg.get_path("analysis.figure_dpi", 150)), bbox_inches="tight")
        plt.close(fig)
        return path


def build_context(profile: str = "FAST", overrides: dict | None = None, root: Path | None = None,
                  config_dir: Path | None = None) -> RunContext:
    from .config import find_config_dir

    cdir = Path(config_dir) if config_dir else (Path(root) / "configs" if root else find_config_dir())
    # RESUME re-loads the last run's config; SUBMISSION re-loads the config that produced the champion (P13),
    # so upstream stage fingerprints match and nothing upstream is recomputed.
    if profile.upper() == "SUBMISSION":
        last_run = cdir.parent / "models" / "champion_config.json"
    elif profile.upper() == "V2_SUBMISSION":
        last_run = cdir.parent / "models" / "v2_champion_config.json"
    else:
        last_run = cdir.parent / "experiments" / "last_run_config.json"
    # V2 profiles inherit the configuration of the completed V1 run (the champion's frozen config) so that
    # P00-P15 are reused, never recomputed. A different snapshot can be chosen with WILDSPEECH_V1_CONFIG.
    import os

    v1_base = Path(os.environ.get("WILDSPEECH_V1_CONFIG", cdir.parent / "models" / "champion_config.json"))
    cfg, stages = load_config(profile, cdir, overrides, last_run_path=last_run, v1_base_path=v1_base)
    paths = ProjectPaths.build(cfg, root=cdir.parent if root is None else Path(root))
    logger = get_logger("wildspeech", paths.experiments / "logs")
    registries = Registries(paths.experiments)
    hyp_file = paths.configs / "hypotheses.yaml"
    if hyp_file.exists():
        registries.seed_hypotheses(load_yaml(hyp_file) or [])
    device = resolve_device(cfg.get_path("runtime.device", "auto"))
    try:
        import torch

        torch.set_num_threads(int(cfg.get_path("runtime.torch_threads", 6)))
    except Exception:
        pass
    ctx = RunContext(cfg=cfg, paths=paths, profile=profile.upper(), default_stages=stages, logger=logger,
                     registries=registries, device=device)
    logger.info("context: profile=%s device=%s root=%s data=%s", ctx.profile, device, paths.root, paths.data_dir)
    return ctx


def save_last_run(ctx: RunContext, stages: list[str]) -> None:
    write_json(ctx.paths.experiments / "last_run_config.json",
               {"profile": ctx.profile, "stages": stages, "config": ctx.cfg.to_dict(),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
