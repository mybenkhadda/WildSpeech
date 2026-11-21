"""Project directory layout, stage output mapping and data-folder discovery."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

OUTPUT_SECTION_DIRS = [
    "00_environment", "01_data_audit", "02_target", "03_screening",
    "04_item_intelligence", "05_adaptive_testing", "06_audio_quality",
    "07_temporal", "08_spectral", "09_wavelets", "10_phonetic",
    "11_task_intelligence", "12_aggregation", "13_student_phenotypes",
    "14_residual_discovery", "15_confound_analysis", "16_feature_interactions",
    "17_literacy_indices", "18_baselines", "19_failure_phenotypes",
    "20_feature_registry", "21_final_report",
]

# Stages without a dedicated directory in the requested tree are given
# sub-directories of the closest section so the requested tree stays intact.
STAGE_OUTPUT_DIR = {
    "00": "00_environment",
    "01": "01_data_audit",
    "02": "02_target",
    "03": "03_screening",
    "04": "04_item_intelligence",
    "05": "05_adaptive_testing",
    "06": "06_audio_quality",
    "07": "07_temporal",
    "08": "08_spectral",
    "09": "09_wavelets",
    "10": "06_audio_quality/prosody_anonymization",
    "11": "10_phonetic",
    "12": "11_task_intelligence",
    "13": "12_aggregation",
    "14": "18_baselines",
    "15": "13_student_phenotypes",
    "16": "14_residual_discovery",
    "17": "15_confound_analysis",
    "18": "16_feature_interactions",
    "19": "17_literacy_indices",
    "20": "18_baselines/ablation",
    "21": "18_baselines/residual_learning",
    "22": "19_failure_phenotypes",
    "23": "12_aggregation/foundation_models",
    "24": "20_feature_registry",
    "25": "20_feature_registry/hypotheses",
    "26": "21_final_report/visual_atlas",
    "27": "21_final_report/submission_validation",
    "28": "21_final_report",
}

CACHE_SUBDIRS = [
    "manifests", "audio", "vad", "acoustic", "spectral", "wavelets",
    "phonetic", "embeddings", "item_features", "student_features",
]


@dataclass
class ProjectPaths:
    root: Path
    local_root: Path
    experiment: str
    data_source: Path | None = None
    cache_dirs: dict = field(default_factory=dict)

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def extracted_root(self) -> Path:
        return self.local_root / "data" / "extracted"

    @property
    def extracted_train(self) -> Path:
        return self.extracted_root / "train"

    @property
    def extracted_test(self) -> Path:
        return self.extracted_root / "test"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def notebooks(self) -> Path:
        return self.root / "notebooks"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def local_cache(self) -> Path:
        return self.local_root / "cache"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs" / self.experiment

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.outputs / "00_environment" / "logs"

    def stage_dir(self, stage_id: str) -> Path:
        path = self.outputs / STAGE_OUTPUT_DIR.get(stage_id, f"stage_{stage_id}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_dir(self, name: str) -> Path:
        path = self.cache_dirs.get(name, self.cache / name)
        path.mkdir(parents=True, exist_ok=True)
        return path


def default_roots(cfg: dict, environment: str) -> tuple[Path, Path]:
    """Return (persistent project root, local fast root)."""
    project = cfg.get("project_name", "WildSpeech")
    if cfg.get("output_root"):
        root = Path(cfg["output_root"]).expanduser()
    elif environment == "COLAB":
        persist = cfg.get("persist_root") or str(Path(cfg.get("drive_mount_point", "/content/drive")) / "MyDrive" / project)
        root = Path(persist)
    else:
        root = Path.cwd() / project
    if cfg.get("local_root"):
        local = Path(cfg["local_root"]).expanduser()
    elif environment == "COLAB":
        local = Path("/content") / f"{project}_local"
    else:
        local = root
    return root, local


def build_project_tree(cfg: dict, environment: str) -> ProjectPaths:
    root, local = default_roots(cfg, environment)
    paths = ProjectPaths(root=root, local_root=local, experiment=cfg.get("experiment_name", "P01"))
    for directory in [
        paths.data_raw, paths.extracted_train, paths.extracted_test, paths.configs,
        paths.notebooks, paths.cache, paths.outputs, paths.models, paths.submissions,
        paths.reports, paths.logs, paths.local_cache,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    for sub in OUTPUT_SECTION_DIRS:
        (paths.outputs / sub).mkdir(parents=True, exist_ok=True)
    for sub in CACHE_SUBDIRS:
        # Heavy intermediate audio caches live on fast local storage; the
        # persistent cache keeps manifests and compact feature tables.
        base = paths.local_cache if sub in {"audio", "vad"} else paths.cache
        paths.cache_dirs[sub] = base / sub
        paths.cache_dirs[sub].mkdir(parents=True, exist_ok=True)
    return paths


def _walk_limited(root: Path, max_depth: int):
    root = Path(root)
    if not root.exists():
        return
    base_depth = len(root.parts)
    for current, dirs, files in os.walk(root, followlinks=False):
        depth = len(Path(current).parts) - base_depth
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") or d == ".shortcut-targets-by-id"]
        yield Path(current), files


def candidate_search_roots(cfg: dict, environment: str) -> list[Path]:
    roots: list[Path] = []
    if cfg.get("data_search_roots"):
        roots.extend(Path(p).expanduser() for p in cfg["data_search_roots"])
    if environment == "COLAB":
        mount = Path(cfg.get("drive_mount_point", "/content/drive"))
        folder_id = cfg.get("drive_folder_id")
        if folder_id:
            # Shared-folder shortcuts are exposed under this path by Colab.
            roots.append(mount / ".shortcut-targets-by-id" / folder_id)
        roots.extend([mount / "MyDrive", mount / "Shareddrives", mount / ".shortcut-targets-by-id"])
        roots.append(Path("/content"))
    else:
        roots.extend([Path.cwd(), Path.cwd() / "data", Path.home()])
    if cfg.get("drive_root"):
        roots.insert(0, Path(cfg["drive_root"]).expanduser())
    unique: list[Path] = []
    for r in roots:
        if r not in unique:
            unique.append(r)
    return unique


def discover_data_root(cfg: dict, environment: str) -> dict:
    """Recursively search for the directory containing all required files.

    Returns a report with the chosen root and all candidates. Never picks one
    of several non-identical candidates silently.
    """
    required = list(cfg["required_files"])
    report: dict = {"required_files": required, "candidates": [], "partial_candidates": [],
                    "chosen": None, "status": "NOT_FOUND", "message": ""}
    if cfg.get("data_root"):
        root = Path(cfg["data_root"]).expanduser()
        missing = [f for f in required if not (root / f).is_file()]
        report["candidates"] = [str(root)]
        if missing:
            report["status"] = "MISSING_FILES"
            report["missing"] = missing
            report["message"] = f"Configured data_root {root} lacks: {missing}"
        else:
            report["status"] = "CONFIGURED"
            report["chosen"] = str(root)
        return report
    seen: set = set()
    full: list[Path] = []
    for search_root in candidate_search_roots(cfg, environment):
        for directory, files in _walk_limited(search_root, int(cfg.get("data_search_max_depth", 7))):
            key = str(directory.resolve()) if directory.exists() else str(directory)
            if key in seen:
                continue
            seen.add(key)
            present = set(files) & set(required)
            if len(present) == len(required):
                full.append(directory)
            elif len(present) >= max(2, len(required) // 2):
                report["partial_candidates"].append({"path": str(directory), "present": sorted(present)})
    report["candidates"] = [str(p) for p in full]
    if not full:
        report["message"] = ("No directory containing every required file was found. Set CONFIG['data_root'] "
                             "explicitly or add the shared folder to your Drive.")
        return report
    if len(full) == 1:
        report["status"] = "FOUND"
        report["chosen"] = str(full[0])
        return report
    signatures = []
    for p in full:
        signatures.append(tuple((f, (p / f).stat().st_size) for f in required))
    report["duplicates"] = [{"path": str(p), "signature": dict(sig)} for p, sig in zip(full, signatures)]
    if len(set(signatures)) == 1:
        chosen = sorted(full, key=lambda p: (len(p.parts), str(p)))[0]
        report["status"] = "FOUND_IDENTICAL_DUPLICATES"
        report["chosen"] = str(chosen)
        report["message"] = f"{len(full)} identical copies found; using {chosen}."
    else:
        report["status"] = "AMBIGUOUS_DUPLICATES"
        report["message"] = ("Several non-identical candidate data folders were found. Set CONFIG['data_root'] "
                             "to the intended folder.")
    return report
