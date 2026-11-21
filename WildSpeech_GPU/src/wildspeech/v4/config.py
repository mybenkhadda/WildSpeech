"""Notebook CONFIG -> WildSpeech configuration overrides and Google Drive path resolution.

The notebook keeps one flat CONFIG dict (see DEFAULT_CONFIG). build_overrides() maps it onto the existing
configuration tree (paths.*, validation.*, v2.validation.*, v2.evaluation.baseline, v4.*) so the V1-V3 machinery
(DataBundle discovery, validation registry, fold caches, evaluators) runs unchanged on the Drive data.

Path policy
  * competition CSVs/TARs are discovered under data_root (existing paths.discover_data_files)
  * every artefact lives under output_root (Drive) so a new Colab session resumes; cache/ inside it
  * local_work_root (/content) is used only for staged archive copies and downloads (fast local disk)
  * the ASR export is read from the mounted Drive path; ONLY if that file is missing is it downloaded by its
    Drive file ID (gdown) into local_work_root
"""
from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any

ASR_DRIVE_FILE_ID = "12SzHSmkNbeoS_mO1L80uEd6A2k4jH-8V"

RUN_MODES = ("FULL", "PREPARE", "FEATURES", "EXPERIMENTS", "SUBMISSION", "SMOKE")

DEFAULT_CONFIG: dict[str, Any] = {
    "repo": "mybenkhadda/WildSpeech",
    "ref": "main",
    "pinned_commit": None,          # exact commit for a reproducible run (overrides `ref` in the notebook clone)
    "project_subdir": "WildSpeech_GPU",

    "data_root": "/content/drive/MyDrive/WildSpeech",
    "asr_root": "/content/drive/MyDrive/WildSpeech/WildSpeech_ASR_Exports",
    "asr_filename": "all_models_predictions.parquet",
    "asr_drive_file_id": ASR_DRIVE_FILE_ID,

    "output_root": "/content/drive/MyDrive/WildSpeech/WildSpeech_V4",
    "local_work_root": "/content/wildspeech_v4",

    "run_mode": "FULL",
    "max_clips": None,
    "splits": ["train", "test"],

    "reuse_existing_features": True,
    "compute_missing_acoustic_features": True,
    "rerun_asr": False,
    "train_models": True,
    "generate_submission": True,

    "cv_folds": 5,
    "cv_repeats": 3,
    "seed": 2026,

    "use_gpu": True,
    "resume": True,
    "save_checkpoints": True,
}

OUTPUT_DIRS = ("manifests", "asr_import", "clip_features", "student_features", "model_registry", "cv", "experiments",
               "ablations", "robustness", "diagnostics", "figures", "models", "submission", "reports")


def merged_config(user: dict | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(copy.deepcopy(user or {}))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    mode = str(cfg.get("run_mode", "FULL")).upper()
    if mode not in RUN_MODES:
        raise ValueError(f"run_mode {mode!r} not in {RUN_MODES}")
    if int(cfg.get("cv_folds", 5)) < 2:
        raise ValueError("cv_folds must be >= 2")
    if int(cfg.get("cv_repeats", 1)) < 1:
        raise ValueError("cv_repeats must be >= 1")
    bad = set(map(str, cfg.get("splits", []))) - {"train", "test"}
    if bad:
        raise ValueError(f"unknown splits {bad}")
    if cfg.get("rerun_asr"):
        raise ValueError("rerun_asr=True is not supported by V4: the completed external ASR export is imported; "
                         "local inference belongs to P24 (V3) and its eligibility gate")


def is_smoke(cfg: dict) -> bool:
    return str(cfg.get("run_mode", "FULL")).upper() == "SMOKE"


def build_overrides(cfg: dict, v4_overrides: dict | None = None) -> dict:
    """Configuration overrides for wildspeech.context.build_context(profile='V4', overrides=...)."""
    out_root = Path(cfg["output_root"])
    smoke = is_smoke(cfg)
    repeats = 1 if smoke else int(cfg["cv_repeats"])
    o: dict[str, Any] = {
        "project": {"seed": int(cfg["seed"])},
        "paths": {"data_dir": "auto", "data_search_roots": [str(cfg["data_root"])], "data_search_max_depth": 3,
                  "cache_dir": str(out_root / "cache"), "outputs_dir": str(out_root / "pipeline"),
                  "models_dir": str(out_root / "models"), "experiments_dir": str(out_root / "model_registry"),
                  "submissions_dir": str(out_root / "submission"), "reports_dir": str(out_root / "reports")},
        "runtime": {"device": "auto" if cfg.get("use_gpu", True) else "cpu", "resume": bool(cfg.get("resume", True)),
                    "torch_threads": max(1, (os.cpu_count() or 2))},
        "validation": {"n_splits": int(cfg["cv_folds"]), "n_repeats": repeats},
        "models": {"gbm_threads": max(1, (os.cpu_count() or 2))},
        "audio": {"extract_test": "test" in cfg.get("splits", []), "max_clips": cfg.get("max_clips")},
        "v2": {"validation": {"schemes": ["repeated_kfold", "school_kfold", "loso", "instrument_transfer",
                                          "sample_rate_transfer", "sample_rate_equal_n"],
                              "rebuild_if_changed": True},
               "evaluation": {}},
        "v3": {"runtime": {"max_repeats": None}},
        "v4": {"run": {k: cfg.get(k) for k in DEFAULT_CONFIG}},
    }
    if smoke:
        o["data"] = {"max_students": 120, "max_test_students": 40}
        o["validation"]["bootstrap"] = 100
        o["v4"]["validation"] = {"bootstrap": 100, "robustness_schemes": ["school_kfold", "sample_rate_transfer"],
                                 "budgets": [5, 20]}
    if v4_overrides:
        from ..config import deep_merge

        o["v4"] = deep_merge(o["v4"], v4_overrides)
    return o


def finalize_baseline(ctx) -> None:
    """The V4 comparator is fixed in v4.baseline (default GAM on SCREENING+BEHAVIOR+ITEM, the historical reference
    model). It is written into v2.evaluation.baseline so StudentLevelEvaluator uses it instead of P10 artefacts
    that do not exist in a fresh Colab output tree."""
    b = ctx.cfg.get_path("v4.baseline") or {"model": "GAM", "families": ["SCREENING", "BEHAVIOR", "ITEM"]}
    ctx.cfg.set_path("v2.evaluation.baseline", {"model": str(b["model"]), "families": list(b["families"])})


def build_v4_context(cfg: dict, project_root: Path | str | None = None, v4_overrides: dict | None = None):
    from ..context import build_context

    cfg = merged_config(cfg)
    ensure_output_tree(cfg)
    ctx = build_context("V4", overrides=build_overrides(cfg, v4_overrides),
                        root=Path(project_root) if project_root else None)
    finalize_baseline(ctx)
    ctx._cache["v4_config"] = cfg
    return ctx


def ensure_output_tree(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["output_root"])
    out = {}
    for d in OUTPUT_DIRS:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        out[d] = p
    Path(cfg["local_work_root"]).mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------------------------------------------------------------------- ASR export
def resolve_asr_file(cfg: dict, logger=None, allow_download: bool = True) -> tuple[Path | None, dict]:
    """Mounted Drive path first; Drive file-ID download into local_work_root only if the mounted file is missing."""
    mounted = Path(cfg["asr_root"]) / cfg["asr_filename"]
    info = {"mounted_path": str(mounted), "mounted_exists": mounted.exists(), "source": None,
            "file_id": cfg.get("asr_drive_file_id")}
    if mounted.exists():
        info["source"] = "mounted_drive"
        return mounted, info
    local = Path(cfg["local_work_root"]) / "asr_download" / cfg["asr_filename"]
    if local.exists() and local.stat().st_size > 0:
        info["source"] = "previous_file_id_download"
        return local, info
    if not allow_download or not cfg.get("asr_drive_file_id"):
        info["source"] = "missing"
        return None, info
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown

        if logger:
            logger.warning("ASR export not found at %s; downloading Drive file id %s", mounted, cfg["asr_drive_file_id"])
        gdown.download(id=str(cfg["asr_drive_file_id"]), output=str(local), quiet=False)
    except Exception as e:  # noqa: BLE001 - recorded, the import reports NOT_EXECUTED
        info["source"] = "download_failed"
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info
    if local.exists() and local.stat().st_size > 0:
        info["source"] = "file_id_download"
        return local, info
    info["source"] = "download_failed"
    return None, info


def stage_archive(src: Path, local_root: Path, logger=None) -> Path:
    """Copy a TAR from Drive to local disk once (copy2 keeps mtime, so the cached manifest fingerprint still
    matches). Random seeks on the Drive FUSE mount are slow; local reads are not. Returns the staged path."""
    src = Path(src)
    dst = Path(local_root) / "archives" / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst.parent).free
    if free < src.stat().st_size * 1.1:
        if logger:
            logger.warning("not enough local disk to stage %s (%d free); reading it from Drive", src.name, free)
        return src
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return dst
