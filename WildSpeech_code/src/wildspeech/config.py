"""Central configuration, execution profiles and configuration hashing.

Configuration precedence (lowest to highest):

1. ``BASE_CONFIG`` (the notebook configuration cell defaults)
2. ``EXTENDED_DEFAULTS`` (advanced parameters)
3. optional YAML file (``configs/p01_config.yaml``)
4. user overrides (the notebook ``CONFIG`` dict)
5. execution-profile overrides, except keys listed in ``profile_exempt_keys``

The ``RESUME`` profile reloads the semantic configuration of the last run and
only switches on resumption, so cached artifacts remain valid.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

BASE_CONFIG: dict = {
    "project_name": "WildSpeech",
    "experiment_name": "P01_LITERACY_SIGNAL_DISCOVERY",
    "drive_folder_id": "10hOoRyDO2VwZ9rgU1grTe0tb97_MQjnf",
    "drive_folder_url": (
        "https://drive.google.com/drive/folders/"
        "10hOoRyDO2VwZ9rgU1grTe0tb97_MQjnf"
    ),
    "environment": "COLAB",
    "cloud_storage_authorized": False,
    "drive_root": None,
    "data_root": None,
    "local_root": None,
    "output_root": None,
    "run_mode": "FULL",
    "seed": 42,
    "use_gpu": True,
    "n_jobs": -1,
    "sample_rate": 16000,
    "audio_batch_size": 64,
    "debug_max_students": 30,
    "debug_max_clips": 300,
    "extract_train_audio": True,
    "extract_test_audio": False,
    "train_only_eda": True,
    "allow_test_inference": False,
    "allow_external_apis": False,
    "enable_tabular_eda": True,
    "enable_item_analysis": True,
    "enable_irt": True,
    "enable_acoustic_features": True,
    "enable_vad": True,
    "enable_wavelets": True,
    "enable_prosody": True,
    "enable_phonetic_analysis": False,
    "enable_asr": False,
    "enable_embeddings": False,
    "enable_student_clustering": True,
    "enable_residual_analysis": True,
    "enable_confound_analysis": True,
    "enable_feature_interactions": True,
    "enable_baseline_models": True,
    "enable_failure_analysis": True,
    "enable_submission": False,
    "cv_folds": 5,
    "cv_repeats": 3,
    "bootstrap_iterations": 1000,
    "fdr_alpha": 0.05,
    "resume": True,
    "overwrite_cache": False,
    "save_figures": True,
    "save_tables": True,
    "save_html_report": True,
    "figure_dpi": 160,
    "max_features_for_heatmap": 60,
}

EXTENDED_DEFAULTS: dict = {
    # --- environment / storage -------------------------------------------
    "mount_drive": True,
    "drive_mount_point": "/content/drive",
    "persist_root": None,            # Colab: defaults to <MyDrive>/WildSpeech
    "data_search_roots": None,       # explicit list of directories to search
    "data_search_max_depth": 7,
    "required_files": [
        "pandas_read_csv_kwargs.json",
        "submission_format.csv",
        "train_audio.tar",
        "test_audio.tar",
        "train_clip_metadata.csv",
        "test_clip_metadata.csv",
        "train_student_metadata.csv",
        "test_student_metadata.csv",
        "train_labels.csv",
    ],
    "min_free_disk_gb": 2.0,
    "disk_safety_factor": 1.15,
    "install_optional_dependencies": False,
    # --- evaluation --------------------------------------------------------
    "official_metric": "r2",            # must be verified on the competition page
    "official_metric_verified": False,
    "primary_baseline": "M01_RIDGE",
    "primary_cv_scheme": "CV_A_REPEATED_STUDENT_KFOLD",
    "target_bins_for_stratification": 5,
    "max_schools_for_loso": 60,
    "min_group_size": 10,
    "inner_cv_folds": 4,
    # --- schema -------------------------------------------------------------
    # {"student": {"student_id": "colname"}, "clip": {...}, "labels": {...}}
    "schema_overrides": {},
    # {"raw task value": "blending" | "deletion" | "repetition" | "other"}
    "task_family_overrides": {},
    # --- audio --------------------------------------------------------------
    "audio_inventory_decode": True,
    "audio_inventory_hash": True,
    "min_audio_duration_s": 0.05,
    "max_audio_duration_s": 180.0,
    "n_fft": 512,
    "hop_length": 160,
    "win_length": 400,
    "n_mels": 64,
    "n_mfcc": 13,
    "clip_level_threshold": 0.99,
    "vad_backend": "energy",          # energy | webrtc | silero_local
    "vad_model_path": None,           # local TorchScript Silero file (license verified by user)
    "vad_model_license_verified": False,
    "vad_webrtc_aggressiveness": 2,
    "vad_energy_offset_db": 12.0,
    "vad_max_below_peak_db": 40.0,
    "vad_min_dynamic_range_db": 6.0,
    "vad_min_speech_s": 0.08,
    "vad_min_silence_s": 0.15,
    "short_pause_s": 0.15,
    "long_pause_s": 0.5,
    "spectral_robust_stats": [
        "mean", "std", "median", "mad", "min", "max", "p05", "p10", "p25",
        "p75", "p90", "p95", "iqr", "skewness", "kurtosis",
    ],
    "spectral_min_frames": 5,
    "enable_hpss": False,
    "enable_chroma": False,
    "wavelet_families_dwt": ["haar", "db4", "sym8"],
    "wavelet_families_all": [
        "haar", "db2", "db4", "db6", "db8", "sym4", "sym8", "coif3",
        "morl", "mexh", "cmor1.5-1.0",
    ],
    "wavelet_primary_family": "db4",
    "wavelet_max_level": 8,
    "wavelet_packet_family": "db4",
    "wavelet_packet_level": 4,
    "enable_cwt": True,
    "cwt_wavelets": ["morl"],
    "cwt_sr": 4000,
    "cwt_n_scales": 24,
    "cwt_max_seconds": 4.0,
    "prosody_f0_method": "pyin",      # pyin | yin
    "f0_min_hz": 75.0,
    "f0_max_hz": 600.0,
    "formant_max_frames": 60,
    "extraction_version": "1.0.0",
    "extraction_chunk_size": 400,
    "atlas_enabled": True,
    "atlas_examples_per_stratum": 1,
    "atlas_max_examples": 24,
    # --- aggregation --------------------------------------------------------
    "aggregation_core_features": None,  # None -> curated default list
    "aggregation_global_stats": ["mean", "std", "median"],
    "aggregation_core_stats": [
        "mean", "median", "std", "mad", "min", "max", "p10", "p25", "p75",
        "p90", "iqr", "cv", "count",
    ],
    "aggregation_min_obs": 3,
    "slope_shrinkage_k": 5.0,
    "item_shrinkage_k": 10.0,
    "quality_good_snr_db": 15.0,
    "length_bins_s": [0.0, 1.5, 3.0, 1e9],
    "activity_bins": [0.0, 0.3, 0.6, 1.0001],
    "phonetic_confidence_bins": [0.0, 0.5, 0.8, 1.0001],
    # --- item / IRT ---------------------------------------------------------
    "irt_models": ["rasch", "1pl", "2pl"],
    "irt_min_obs_per_item": 5,
    "irt_2pl_min_obs_per_item": 50,
    "irt_prior_theta_sd": 1.0,
    "irt_prior_b_sd": 2.0,
    "irt_prior_loga_sd": 0.5,
    "irt_max_iter": 500,
    "item_redundancy_min_coexposure": 30,
    "item_rare_threshold": 10,
    # --- discovery ----------------------------------------------------------
    "residual_group_quantiles": [0.2, 0.8],
    "matching_calipers": [0.1, 0.2, 0.5],
    "permutation_iterations": 1000,
    "permutation_top_k": 50,
    "discovery_split_repeats": 20,
    "discovery_top_k": 50,
    "min_n_correlation": 20,
    "cluster_k_range": [2, 8],
    "cluster_bootstrap": 50,
    "cluster_max_pca_components": 20,
    "umap_enabled": True,
    "tsne_enabled": True,
    "hdbscan_min_cluster_size": 10,
    "spectral_clustering_max_n": 3000,
    "interaction_candidates": [
        ["SCREENING", "TEMPORAL"], ["SCREENING", "SPECTRAL"], ["SCREENING", "PHONETIC"],
        ["SCREENING", "FLUENCY"], ["DIFFICULTY", "DURATION"], ["DIFFICULTY", "PHONETIC_CONFIDENCE"],
        ["DIFFICULTY", "SPECTRAL_ENTROPY"], ["CORRECTNESS", "ACOUSTIC_CONSISTENCY"],
        ["TASK", "ACOUSTIC"], ["INSTRUMENT", "SCREENING"], ["TASK", "COMPLEXITY"],
    ],
    "interaction_h_statistic_max_n": 600,
    "index_min_components": 2,
    "ablation_models": ["RIDGE", "ELASTIC_NET", "CATBOOST", "XGBOOST", "LIGHTGBM", "PLS"],
    "ablation_primary_model": "RIDGE",
    "ablation_max_features": 300,
    "ablation_pairwise_families": True,
    "ablation_forward_max_steps": 6,
    "residual_fusion_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
    "feature_registry_cv_top_k": 100,
    "feature_registry_bootstrap_top_k": 200,
    "redundancy_threshold": 0.9,
    "sensitivity_artifact_corr": 0.5,
    "sensitivity_item_icc": 0.1,
    "sensitivity_correctness_r": 0.1,
    # --- linguistic / phonetic / foundation models --------------------------
    "enable_linguistic_features": True,
    "g2p_backend": "auto",            # auto | g2p_en | pronouncing | rules
    "asr_models": [],                 # list of dicts, see phonetic_features.ASR_MODEL_SPEC
    "embedding_models": [],           # list of dicts, see foundation_models.EMBEDDING_MODEL_SPEC
    "hf_local_files_only": True,
    "embedding_max_seconds": 30.0,
    "embedding_pca_components": 32,
    # --- submission ---------------------------------------------------------
    "submission_model": "M01_RIDGE",
    "submission_feature_set": "A01_THETA_ONLY",
    "submission_filename": "WILDSPEECH_P01_SCREENING_BASELINE_submission.csv",
    # --- orchestration ------------------------------------------------------
    "stage_selection": None,
    "profile_exempt_keys": [],
    "last_run_config_name": "last_run_config.json",
}

PROFILES: dict = {
    "SMOKE": {
        "debug_max_students": 10,
        "debug_max_clips": 60,
        "cv_repeats": 1,
        "bootstrap_iterations": 50,
        "permutation_iterations": 50,
        "cluster_bootstrap": 5,
        "discovery_split_repeats": 3,
        "enable_prosody": False,
        "enable_wavelets": False,
        "enable_cwt": False,
        "atlas_max_examples": 2,
        "ablation_models": ["RIDGE"],
    },
    "DEBUG": {
        "cv_repeats": 1,
        "bootstrap_iterations": 100,
        "permutation_iterations": 100,
        "cluster_bootstrap": 10,
        "discovery_split_repeats": 5,
        "wavelet_families_dwt": ["db4"],
        "atlas_max_examples": 6,
        "ablation_models": ["RIDGE", "CATBOOST"],
    },
    "FAST": {
        "enable_prosody": False,
        "enable_cwt": False,
        "enable_hpss": False,
        "enable_chroma": False,
        "wavelet_families_dwt": ["db4"],
        "bootstrap_iterations": 300,
        "permutation_iterations": 300,
        "cluster_bootstrap": 20,
        "ablation_models": ["RIDGE", "CATBOOST"],
    },
    "FULL": {},
    "RESEARCH": {
        "enable_phonetic_analysis": True,
        "enable_hpss": True,
        "enable_chroma": True,
        "enable_cwt": True,
        "cwt_wavelets": ["morl", "mexh", "cmor1.5-1.0"],
        "wavelet_families_dwt": ["haar", "db2", "db4", "db6", "db8", "sym4", "sym8", "coif3"],
    },
    "RESUME": {"resume": True, "overwrite_cache": False},
}

PROFILE_DEFAULT_STAGES: dict = {
    "SMOKE": ["00", "01", "02", "03", "06"],
}

VALID_ENVIRONMENTS = {"COLAB", "LOCAL_WINDOWS", "LOCAL_LINUX", "PRIVATE_CLOUD"}
VALID_RUN_MODES = set(PROFILES)

# Keys that never change scientific results and are excluded from config hashes.
NON_SEMANTIC_KEYS = {
    "resume", "overwrite_cache", "n_jobs", "save_figures", "save_tables",
    "save_html_report", "figure_dpi", "use_gpu", "audio_batch_size",
    "stage_selection", "max_features_for_heatmap", "install_optional_dependencies",
    "profile_exempt_keys", "last_run_config_name", "mount_drive",
    "drive_mount_point", "data_search_roots", "data_search_max_depth",
    "drive_root", "local_root", "output_root", "persist_root",
    "cloud_storage_authorized", "environment", "drive_folder_url",
}


class ConfigError(ValueError):
    """Raised for invalid or unsafe configuration values."""


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ConfigError("PyYAML is required to read YAML configuration files") from exc
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"YAML config {path} must contain a mapping")
    return data


def build_config(
    user_config: Mapping[str, Any] | None = None,
    profile: str | None = None,
    yaml_path: str | Path | None = None,
    last_run_path: str | Path | None = None,
) -> dict:
    """Assemble the effective configuration for a run."""
    cfg: dict = copy.deepcopy(BASE_CONFIG)
    cfg.update(copy.deepcopy(EXTENDED_DEFAULTS))
    if yaml_path is not None and Path(yaml_path).exists():
        cfg.update(_load_yaml(Path(yaml_path)))
    if user_config:
        cfg.update(copy.deepcopy(dict(user_config)))
    profile = (profile or cfg.get("run_mode") or "FULL").upper()
    if profile not in PROFILES:
        raise ConfigError(f"Unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    if profile == "RESUME":
        if last_run_path is None or not Path(last_run_path).exists():
            raise ConfigError(
                "RESUME profile requires the configuration of a previous run "
                f"(looked for {last_run_path}). Run another profile first."
            )
        with open(last_run_path, "r", encoding="utf-8") as fh:
            previous = json.load(fh)
        keep = {k: cfg[k] for k in NON_SEMANTIC_KEYS if k in cfg}
        cfg.update(previous)
        cfg.update(keep)
        cfg["resume"] = True
        cfg["overwrite_cache"] = False
        cfg["resumed_profile"] = previous.get("run_mode")
    else:
        exempt = set(cfg.get("profile_exempt_keys") or [])
        for key, value in PROFILES[profile].items():
            if key not in exempt:
                cfg[key] = copy.deepcopy(value)
        cfg["run_mode"] = profile
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate configuration types and safety constraints."""
    env = str(cfg.get("environment", "")).upper()
    if env not in VALID_ENVIRONMENTS:
        raise ConfigError(f"environment must be one of {sorted(VALID_ENVIRONMENTS)}, got {env!r}")
    if cfg.get("run_mode") not in VALID_RUN_MODES:
        raise ConfigError(f"run_mode must be one of {sorted(VALID_RUN_MODES)}")
    if bool(cfg.get("allow_external_apis")):
        raise ConfigError(
            "allow_external_apis=True is not supported: P01 never transmits competition "
            "data to external APIs, LLM providers or remote speech services."
        )
    if int(cfg.get("cv_folds", 0)) < 2:
        raise ConfigError("cv_folds must be >= 2")
    if int(cfg.get("cv_repeats", 0)) < 1:
        raise ConfigError("cv_repeats must be >= 1")
    if not 0.0 < float(cfg.get("fdr_alpha", 0.05)) < 1.0:
        raise ConfigError("fdr_alpha must be in (0, 1)")
    if int(cfg.get("bootstrap_iterations", 0)) < 10:
        raise ConfigError("bootstrap_iterations must be >= 10")
    if int(cfg.get("sample_rate", 0)) not in {8000, 16000, 22050, 24000, 32000, 44100, 48000}:
        raise ConfigError("sample_rate must be a standard audio rate")
    if cfg.get("hop_length", 1) <= 0 or cfg.get("n_fft", 1) <= 0:
        raise ConfigError("n_fft and hop_length must be positive")
    if cfg.get("win_length", 0) > cfg.get("n_fft", 0):
        raise ConfigError("win_length must be <= n_fft")
    if cfg.get("vad_backend") not in {"energy", "webrtc", "silero_local"}:
        raise ConfigError("vad_backend must be energy, webrtc or silero_local")
    if cfg.get("enable_submission") and not cfg.get("allow_test_inference"):
        raise ConfigError("enable_submission=True requires allow_test_inference=True")
    for spec in cfg.get("asr_models") or []:
        if not isinstance(spec, dict) or "checkpoint" not in spec:
            raise ConfigError("each asr_models entry must be a dict with a 'checkpoint'")
    for spec in cfg.get("embedding_models") or []:
        if not isinstance(spec, dict) or "checkpoint" not in spec:
            raise ConfigError("each embedding_models entry must be a dict with a 'checkpoint'")


def semantic_config(cfg: Mapping[str, Any]) -> dict:
    return {k: v for k, v in cfg.items() if k not in NON_SEMANTIC_KEYS}


def stable_hash(obj: Any, length: int = 16) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Hash of all scientifically relevant configuration values."""
    return stable_hash(semantic_config(cfg))


def subset_hash(cfg: Mapping[str, Any], keys: list[str] | tuple[str, ...]) -> str:
    return stable_hash({k: cfg.get(k) for k in keys})


EXTRACTION_KEYS = (
    "sample_rate", "n_fft", "hop_length", "win_length", "n_mels", "n_mfcc",
    "clip_level_threshold", "enable_vad", "vad_backend", "vad_model_path",
    "vad_webrtc_aggressiveness", "vad_energy_offset_db", "vad_max_below_peak_db",
    "vad_min_dynamic_range_db", "vad_min_speech_s", "vad_min_silence_s",
    "short_pause_s", "long_pause_s", "spectral_robust_stats", "spectral_min_frames",
    "enable_hpss", "enable_chroma", "enable_wavelets", "wavelet_families_dwt",
    "wavelet_primary_family", "wavelet_max_level", "wavelet_packet_family",
    "wavelet_packet_level", "enable_cwt", "cwt_wavelets", "cwt_sr", "cwt_n_scales",
    "cwt_max_seconds", "enable_prosody", "prosody_f0_method", "f0_min_hz",
    "f0_max_hz", "formant_max_frames", "extraction_version", "min_audio_duration_s",
    "max_audio_duration_s",
)


def extraction_config(cfg: Mapping[str, Any]) -> dict:
    """Small, picklable configuration subset for worker processes."""
    return {k: copy.deepcopy(cfg.get(k)) for k in EXTRACTION_KEYS}


def save_config(cfg: Mapping[str, Any], directory: Path, name: str = "p01_effective_config.json") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    from .io_utils import atomic_write_json

    path = directory / name
    atomic_write_json(path, dict(cfg))
    last = directory / str(cfg.get("last_run_config_name", "last_run_config.json"))
    atomic_write_json(last, semantic_config(cfg))
    return path


def update_in_place(cfg: MutableMapping[str, Any], **kwargs: Any) -> None:
    cfg.update(kwargs)
    validate_config(cfg)
