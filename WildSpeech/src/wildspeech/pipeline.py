"""Stage orchestrator with explicit dependency resolution, caching and manifests.

Stage statuses: NOT_STARTED, RUNNING, COMPLETE, SKIPPED_DISABLED,
BLOCKED_DEPENDENCY, FAILED, CACHED.

A stage is CACHED when ``resume`` is on, ``overwrite_cache`` is off, the
previous run completed, every registered output still exists and the input
fingerprint (stage version + semantic config hash + fingerprints of every
available dependency) is unchanged. Re-running any dependency changes its
fingerprint and therefore invalidates all dependants (dependency-aware cache
invalidation).
"""
from __future__ import annotations

import datetime as _dt
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import (ablation, adaptive_behavior, audio_processing, baseline_models, clustering,
               confound_analysis, data_audit, environment, failure_atlas, feature_interactions,
               feature_registry, foundation_models, hypothesis_registry, item_intelligence,
               literacy_indices, phonetic_features, prosody_features, reporting, residual_discovery,
               residual_models, screening_eda, spectral_features, student_aggregation, submission,
               target_eda, task_intelligence, temporal_features, visualization, wavelet_features)
from .config import PROFILE_DEFAULT_STAGES, build_config, config_hash, save_config, stable_hash
from .io_utils import ArtifactMissing, ArtifactStore, atomic_write_csv, atomic_write_json, file_fingerprint, read_json
from .logging_utils import StageTimer, get_logger, release_memory
from .paths import ProjectPaths, build_project_tree

NOT_STARTED = "NOT_STARTED"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
SKIPPED_DISABLED = "SKIPPED_DISABLED"
BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
FAILED = "FAILED"
CACHED = "CACHED"
AVAILABLE_STATES = {COMPLETE, CACHED}

STAGE_REGISTRY: dict = {
    "00": environment.stage_environment,
    "01": data_audit.stage_data_audit,
    "02": target_eda.stage_target_intelligence,
    "03": screening_eda.stage_screening_intelligence,
    "04": item_intelligence.stage_item_intelligence,
    "05": adaptive_behavior.stage_adaptive_testing,
    "06": audio_processing.stage_acoustic_atlas,
    "07": temporal_features.stage_temporal_features,
    "08": spectral_features.stage_spectral_features,
    "09": wavelet_features.stage_wavelet_features,
    "10": prosody_features.stage_prosody,
    "11": phonetic_features.stage_phonetic_intelligence,
    "12": task_intelligence.stage_task_intelligence,
    "13": student_aggregation.stage_student_aggregation,
    "14": baseline_models.stage_validation_and_baseline,
    "15": clustering.stage_student_phenotypes,
    "16": residual_discovery.stage_residual_discovery,
    "17": confound_analysis.stage_confound_analysis,
    "18": feature_interactions.stage_feature_interactions,
    "19": literacy_indices.stage_literacy_indices,
    "20": ablation.stage_predictive_ablation,
    "21": residual_models.stage_residual_learning,
    "22": failure_atlas.stage_failure_atlas,
    "23": foundation_models.stage_foundation_models,
    "24": feature_registry.stage_feature_registry,
    "25": hypothesis_registry.stage_hypothesis_registry,
    "26": visualization.stage_visualization_atlas,
    "27": submission.stage_submission,
    "28": reporting.stage_final_report,
}


def _flag(*keys: str, any_of: bool = False) -> Callable[[dict], tuple[bool, str]]:
    def check(cfg: dict) -> tuple[bool, str]:
        values = [bool(cfg.get(k)) for k in keys]
        ok = any(values) if any_of else all(values)
        if ok:
            return True, ""
        off = [k for k, v in zip(keys, values) if not v]
        return False, f"disabled by configuration: {', '.join(off)}"
    return check


def _always(cfg: dict) -> tuple[bool, str]:
    return True, ""


@dataclass
class StageSpec:
    stage_id: str
    name: str
    func: Callable
    requires: list = field(default_factory=list)
    optional: list = field(default_factory=list)
    enabled: Callable = _always
    version: str = "1.0.0"
    always_run: bool = False


ALL_ANALYSIS = [f"{i:02d}" for i in range(1, 26)]

STAGE_SPECS: dict = {s.stage_id: s for s in [
    StageSpec("00", "environment", STAGE_REGISTRY["00"], always_run=True),
    StageSpec("01", "data_audit", STAGE_REGISTRY["01"], ["00"]),
    StageSpec("02", "target_intelligence", STAGE_REGISTRY["02"], ["01"], [], _flag("enable_tabular_eda")),
    StageSpec("03", "screening_intelligence", STAGE_REGISTRY["03"], ["01"], [], _flag("enable_tabular_eda")),
    StageSpec("04", "item_intelligence", STAGE_REGISTRY["04"], ["01"], ["03"], _flag("enable_item_analysis")),
    StageSpec("05", "adaptive_testing", STAGE_REGISTRY["05"], ["01"], ["04", "06"], _flag("enable_item_analysis")),
    StageSpec("06", "acoustic_atlas", STAGE_REGISTRY["06"], ["01"], ["03", "04", "14"], _flag("enable_acoustic_features")),
    StageSpec("07", "temporal_features", STAGE_REGISTRY["07"], ["06"], ["03", "04", "11", "13", "14"], _flag("enable_acoustic_features")),
    StageSpec("08", "spectral_features", STAGE_REGISTRY["08"], ["06"], ["03", "04", "11", "13", "14"], _flag("enable_acoustic_features")),
    StageSpec("09", "wavelet_features", STAGE_REGISTRY["09"], ["06"], ["03", "04", "11", "13", "14"], _flag("enable_acoustic_features", "enable_wavelets")),
    StageSpec("10", "prosody", STAGE_REGISTRY["10"], ["06"], ["03", "04", "11", "13", "14"], _flag("enable_acoustic_features")),
    StageSpec("11", "phonetic_intelligence", STAGE_REGISTRY["11"], ["01"], ["06"], _flag("enable_linguistic_features", "enable_phonetic_analysis", any_of=True)),
    StageSpec("12", "task_intelligence", STAGE_REGISTRY["12"], ["13", "14"], ["11"]),
    StageSpec("13", "student_aggregation", STAGE_REGISTRY["13"], ["01"], ["03", "04", "05", "06", "11", "23"]),
    StageSpec("14", "validation_and_baseline", STAGE_REGISTRY["14"], ["01", "03"], [], _flag("enable_baseline_models")),
    StageSpec("15", "student_phenotypes", STAGE_REGISTRY["15"], ["13"], ["14"], _flag("enable_student_clustering")),
    StageSpec("16", "residual_discovery", STAGE_REGISTRY["16"], ["13", "14"], [], _flag("enable_residual_analysis")),
    StageSpec("17", "confound_analysis", STAGE_REGISTRY["17"], ["13", "14"], ["10"], _flag("enable_confound_analysis")),
    StageSpec("18", "feature_interactions", STAGE_REGISTRY["18"], ["13", "14"], ["04"], _flag("enable_feature_interactions")),
    StageSpec("19", "literacy_indices", STAGE_REGISTRY["19"], ["13", "14"], ["04"], _flag("enable_residual_analysis")),
    StageSpec("20", "predictive_ablation", STAGE_REGISTRY["20"], ["13", "14"], ["04", "15", "19"], _flag("enable_baseline_models")),
    StageSpec("21", "residual_learning", STAGE_REGISTRY["21"], ["13", "14"], [], _flag("enable_residual_analysis", "enable_baseline_models")),
    StageSpec("22", "failure_atlas", STAGE_REGISTRY["22"], ["14"], ["13", "20", "21"], _flag("enable_failure_analysis")),
    StageSpec("23", "foundation_models", STAGE_REGISTRY["23"], ["06"], [], _flag("enable_embeddings")),
    StageSpec("24", "feature_registry", STAGE_REGISTRY["24"], ["13", "14"], ["10", "16", "17", "20"]),
    StageSpec("25", "hypothesis_registry", STAGE_REGISTRY["25"], ["24"], ["12", "15", "16", "17", "18", "19", "20", "21", "22", "23"]),
    StageSpec("26", "visualization_atlas", STAGE_REGISTRY["26"], [], [s for s in ALL_ANALYSIS], _flag("save_figures")),
    StageSpec("27", "submission", STAGE_REGISTRY["27"], ["14"], ["13", "20"], _flag("enable_submission", "allow_test_inference")),
    StageSpec("28", "final_report", STAGE_REGISTRY["28"], [], [f"{i:02d}" for i in range(0, 28)]),
]}


class RunContext:
    """Everything a stage needs: config, paths, logger, artifact store, figures."""

    def __init__(self, config: dict, paths: ProjectPaths, logger, store: ArtifactStore, figures):
        self.config = config
        self.paths = paths
        self.logger = logger
        self.store = store
        self.figures = figures
        self.state: dict = {}
        self.current_stage: str = "00"
        self.available_stages: set = set()

    # ---- stage helpers ------------------------------------------------------
    def out_dir(self, sub: str | None = None) -> Path:
        path = self.paths.stage_dir(self.current_stage)
        if sub:
            path = path / sub
            path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_available(self, stage_id: str) -> bool:
        return stage_id in self.available_stages

    def save_df(self, df: pd.DataFrame, name: str, fmt: str = "parquet", also_csv: bool = False) -> Path:
        if not self.config.get("save_tables", True) and fmt == "csv" and not name.startswith("P01_"):
            # Tables are always written as parquet for downstream stages.
            fmt = "parquet"
        path = self.store.save_df(df, name, self.out_dir(), self.current_stage, fmt=fmt)
        if also_csv and fmt != "csv" and self.config.get("save_tables", True):
            atomic_write_csv(df, path.with_suffix(".csv"))
        return path

    def save_json(self, obj, name: str) -> Path:
        return self.store.save_json(obj, name, self.out_dir(), self.current_stage)

    def load_df(self, name: str, required: bool = True) -> pd.DataFrame | None:
        try:
            return self.store.load_df(name)
        except ArtifactMissing:
            if required:
                raise
            return None

    def load_json(self, name: str, required: bool = True):
        try:
            return self.store.load_json(name)
        except ArtifactMissing:
            if required:
                raise
            return None

    def has(self, name: str) -> bool:
        return self.store.has(name)

    def set_data_root(self, path: Path) -> None:
        self.state["data_root"] = Path(path)
        atomic_write_json(self.paths.cache_dir("manifests") / "data_root.json", {"data_root": str(path)})

    def data_root(self) -> Path:
        if "data_root" in self.state:
            return self.state["data_root"]
        stored = read_json(self.paths.cache_dir("manifests") / "data_root.json")
        if stored and Path(stored["data_root"]).exists():
            self.state["data_root"] = Path(stored["data_root"])
            return self.state["data_root"]
        if self.config.get("data_root"):
            return Path(self.config["data_root"])
        raise FileNotFoundError("Data root unknown: run stage 00 first")

    def figure(self, fig, fig_id: str, title: str, question: str, source_table: str, n, description: str,
               sub: str | None = None) -> Path | None:
        return self.figures.save(fig, section=self.current_stage, fig_id=fig_id, title=title,
                                 question=question, source_table=source_table, sample_size=n,
                                 description=description, directory=self.out_dir(sub or "figures"))


class Pipeline:
    def __init__(self, ctx: RunContext, specs: dict | None = None):
        self.ctx = ctx
        self.specs = specs or STAGE_SPECS
        self.manifest_path = ctx.paths.outputs / "00_environment" / "stage_manifest.json"
        self.report_path = ctx.paths.outputs / "00_environment" / "stage_execution_report.csv"
        self.manifest: dict = read_json(self.manifest_path, default={}) or {}
        self.session_rows: list = []
        self.cfg_hash = config_hash(ctx.config)
        for sid, entry in self.manifest.items():
            if entry.get("status") in AVAILABLE_STATES and ctx.store.outputs_exist(entry.get("outputs", [])):
                ctx.available_stages.add(sid)

    # ---- dependency resolution ---------------------------------------------
    def resolve(self, targets: list, include_dependencies: bool = True) -> list:
        unknown = [t for t in targets if t not in self.specs]
        if unknown:
            raise KeyError(f"unknown stages {unknown}")
        selected: set = set(targets)
        if include_dependencies:
            stack = list(targets)
            while stack:
                sid = stack.pop()
                for dep in self.specs[sid].requires:
                    if dep not in selected:
                        selected.add(dep)
                        stack.append(dep)
        edges = {sid: [d for d in self.specs[sid].requires + self.specs[sid].optional if d in selected]
                 for sid in selected}
        order: list = []
        remaining = dict(edges)
        while remaining:
            ready = sorted(s for s, deps in remaining.items() if all(d in order for d in deps))
            if not ready:
                raise RuntimeError(f"dependency cycle among {sorted(remaining)}")
            order.append(ready[0])
            remaining.pop(ready[0])
        return order

    def default_targets(self) -> list:
        cfg = self.ctx.config
        if cfg.get("stage_selection"):
            return [str(s).zfill(2) for s in cfg["stage_selection"]]
        mode = cfg.get("resumed_profile") or cfg.get("run_mode")
        if mode in PROFILE_DEFAULT_STAGES:
            return list(PROFILE_DEFAULT_STAGES[mode])
        return sorted(self.specs)

    # ---- execution ------------------------------------------------------------
    def _dep_ok(self, sid: str) -> bool:
        entry = self.manifest.get(sid)
        return bool(entry and entry.get("status") in AVAILABLE_STATES
                    and self.ctx.store.outputs_exist(entry.get("outputs", [])))

    def _record(self, sid: str, **fields) -> None:
        entry = self.manifest.get(sid, {})
        entry.update(fields)
        entry["stage_id"] = sid
        entry["name"] = self.specs[sid].name
        entry["updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.manifest[sid] = entry
        atomic_write_json(self.manifest_path, self.manifest)

    def run_stage(self, sid: str, force: bool = False) -> str:
        spec = self.specs[sid]
        cfg = self.ctx.config
        logger = self.ctx.logger
        enabled, reason = spec.enabled(cfg)
        if not enabled:
            self._finish(sid, SKIPPED_DISABLED, reason=reason)
            return SKIPPED_DISABLED
        missing = [d for d in spec.requires if not self._dep_ok(d)]
        if missing:
            states = {d: self.manifest.get(d, {}).get("status", NOT_STARTED) for d in missing}
            self._finish(sid, BLOCKED_DEPENDENCY, reason=f"required dependencies unavailable: {states}")
            return BLOCKED_DEPENDENCY
        optional_ok = [d for d in spec.optional if self._dep_ok(d)]
        dep_fps = {d: self.manifest[d].get("fingerprint") for d in spec.requires + optional_ok}
        input_fp = stable_hash({"version": spec.version, "config": self.cfg_hash, "deps": dep_fps})
        prev = self.manifest.get(sid)
        if (not force and not spec.always_run and cfg.get("resume") and not cfg.get("overwrite_cache")
                and prev and prev.get("status") in AVAILABLE_STATES and prev.get("input_fingerprint") == input_fp
                and self.ctx.store.outputs_exist(prev.get("outputs", []))):
            self.ctx.available_stages.add(sid)
            self._finish(sid, CACHED, reason="validated cached artifacts", outputs=prev.get("outputs", []),
                         fingerprint=prev.get("fingerprint"), input_fingerprint=input_fp)
            return CACHED
        self._record(sid, status=RUNNING, started_at=_dt.datetime.now().isoformat(timespec="seconds"))
        self.ctx.current_stage = sid
        self.ctx.available_stages.discard(sid)
        timer = StageTimer(f"stage {sid} {spec.name}", logger)
        try:
            with timer:
                result = spec.func(self.ctx) or {}
        except Exception as exc:
            tb = traceback.format_exc()
            tb_path = self.ctx.paths.logs / f"stage_{sid}_traceback.txt"
            tb_path.write_text(tb, encoding="utf-8")
            logger.error("Stage %s failed: %s (traceback: %s)", sid, exc, tb_path)
            self._finish(sid, FAILED, reason=f"{type(exc).__name__}: {exc}", traceback_path=str(tb_path),
                         timing=timer.as_dict())
            release_memory()
            return FAILED
        status = result.get("status", COMPLETE)
        outputs = list(result.get("outputs", []))
        if "fingerprint_extra" in result:
            fingerprint = stable_hash({"input": input_fp, "extra": result["fingerprint_extra"]})
        else:
            # Re-executing a stage rewrites its outputs, so dependants are invalidated.
            out_fps = [file_fingerprint(self.ctx.store.path(o)) for o in outputs if self.ctx.store.has(o)]
            fingerprint = stable_hash({"input": input_fp, "outputs": out_fps})
        if status == COMPLETE:
            self.ctx.available_stages.add(sid)
        self._finish(sid, status, reason=result.get("reason", ""), outputs=outputs, summary=result.get("summary", {}),
                     fingerprint=fingerprint, input_fingerprint=input_fp, timing=timer.as_dict())
        release_memory()
        return status

    def _finish(self, sid: str, status: str, **fields) -> None:
        self._record(sid, status=status, **fields)
        row = {"stage_id": sid, "name": self.specs[sid].name, "status": status,
               "reason": fields.get("reason", ""), "elapsed_s": (fields.get("timing") or {}).get("elapsed_s"),
               "finished_at": _dt.datetime.now().isoformat(timespec="seconds")}
        self.session_rows.append(row)
        self.ctx.logger.info("Stage %s -> %s %s", sid, status, fields.get("reason", ""))

    def run(self, stages: list | None = None, include_dependencies: bool = True, force: tuple = ()) -> pd.DataFrame:
        targets = [str(s).zfill(2) for s in (stages or self.default_targets())]
        order = self.resolve(targets, include_dependencies)
        self.ctx.logger.info("Execution order: %s", order)
        for sid in order:
            self.run_stage(sid, force=sid in force)
        report = pd.DataFrame(self.session_rows)
        previous = pd.read_csv(self.report_path) if self.report_path.exists() else pd.DataFrame()
        atomic_write_csv(pd.concat([previous, report], ignore_index=True), self.report_path)
        return report

    def status_table(self) -> pd.DataFrame:
        rows = []
        for sid, spec in sorted(self.specs.items()):
            entry = self.manifest.get(sid, {})
            rows.append({"stage_id": sid, "name": spec.name, "status": entry.get("status", NOT_STARTED),
                         "requires": ",".join(spec.requires), "optional": ",".join(spec.optional),
                         "reason": entry.get("reason", ""), "updated_at": entry.get("updated_at", "")})
        return pd.DataFrame(rows)


def build_context(user_config: dict | None = None, profile: str | None = None,
                  yaml_path: str | Path | None = None) -> RunContext:
    """Create a run context. Mounts Drive only in an authorised Colab runtime."""
    requested = (profile or (user_config or {}).get("run_mode") or "FULL").upper()
    prelim = build_config(user_config, profile="FULL" if requested == "RESUME" else requested, yaml_path=yaml_path)
    detected = environment.detect_environment()
    environment.check_environment_authorisation(prelim, detected)
    boot_logger = get_logger("wildspeech.bootstrap")
    if detected == "COLAB":
        environment.mount_drive(prelim, boot_logger)
    paths = build_project_tree(prelim, prelim["environment"])
    if requested == "RESUME":
        cfg = build_config(user_config, profile="RESUME", yaml_path=yaml_path,
                           last_run_path=paths.configs / prelim["last_run_config_name"])
    else:
        cfg = prelim
    logger = get_logger("wildspeech", paths.logs)
    store = ArtifactStore(paths.cache_dir("manifests") / "artifact_index.json")
    figures = visualization.FigureRegistry(paths.outputs / "figure_index.csv", dpi=int(cfg["figure_dpi"]),
                                           enabled=bool(cfg["save_figures"]))
    save_config(cfg, paths.configs)
    environment.set_seeds(int(cfg["seed"]))
    return RunContext(cfg, paths, logger, store, figures)


def run_pipeline(ctx: RunContext, stages: list | None = None, include_dependencies: bool = True,
                 force: tuple = ()) -> pd.DataFrame:
    return Pipeline(ctx).run(stages, include_dependencies=include_dependencies, force=force)
