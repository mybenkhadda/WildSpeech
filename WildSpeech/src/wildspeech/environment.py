"""Section 00 - environment detection, diagnostics, Drive mounting and seeds."""
from __future__ import annotations

import importlib
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import discover_data_root

MIN_PYTHON = (3, 9)

CORE_PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn", "pyarrow", "matplotlib", "joblib",
    "tqdm", "psutil", "yaml", "soundfile", "librosa", "pywt",
]
OPTIONAL_PACKAGES = [
    "catboost", "xgboost", "lightgbm", "umap", "hdbscan", "shap", "torch",
    "torchaudio", "transformers", "webrtcvad", "g2p_en", "pronouncing", "nltk",
]
PACKAGE_DIST_NAMES = {"sklearn": "scikit-learn", "yaml": "PyYAML", "pywt": "PyWavelets", "umap": "umap-learn"}


def detect_environment() -> str:
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return "COLAB"
    if platform.system() == "Windows":
        return "LOCAL_WINDOWS"
    return "LOCAL_LINUX"


def check_python() -> dict:
    ok = sys.version_info[:2] >= MIN_PYTHON
    return {"python_version": platform.python_version(), "minimum": ".".join(map(str, MIN_PYTHON)), "ok": ok}


def package_versions(names: list[str]) -> dict:
    from importlib import metadata

    out = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None)
            if version is None:
                version = metadata.version(PACKAGE_DIST_NAMES.get(name, name))
            out[name] = {"available": True, "version": str(version)}
        except Exception as exc:  # pragma: no cover - environment dependent
            out[name] = {"available": False, "error": type(exc).__name__}
    return out


def is_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def gpu_report(use_gpu: bool) -> dict:
    report: dict = {"requested": bool(use_gpu), "torch_cuda": False, "devices": []}
    try:
        import torch  # type: ignore

        report["torch_cuda"] = bool(torch.cuda.is_available())
        if report["torch_cuda"]:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                report["devices"].append({"index": i, "name": props.name, "total_memory_gb": props.total_memory / 1e9})
    except Exception:
        pass
    if not report["devices"] and shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
            report["nvidia_smi"] = res.stdout.strip()
        except Exception:
            pass
    report["will_use_gpu"] = bool(use_gpu and report["torch_cuda"])
    return report


def hardware_report(paths_to_check: list[Path]) -> dict:
    info: dict = {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total / 1e9
        info["ram_available_gb"] = vm.available / 1e9
    except Exception:
        info["ram_total_gb"] = float("nan")
    disks = {}
    for p in paths_to_check:
        try:
            p.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(p)
            disks[str(p)] = {"total_gb": usage.total / 1e9, "free_gb": usage.free / 1e9}
        except Exception as exc:
            disks[str(p)] = {"error": str(exc)}
    info["disks"] = disks
    return info


def disk_free_bytes(path: Path) -> int:
    path = Path(path)
    probe = path if path.exists() else next((p for p in path.parents if p.exists()), Path("/"))
    return int(shutil.disk_usage(probe).free)


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_n_jobs(n_jobs: int) -> int:
    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, cpus + 1 + n_jobs)
    return min(int(n_jobs), cpus)


def mount_drive(cfg: dict, logger) -> dict:
    """Mount Google Drive in Colab when configured and authorised."""
    mount_point = cfg.get("drive_mount_point", "/content/drive")
    if Path(mount_point, "MyDrive").exists():
        return {"mounted": True, "mount_point": mount_point, "note": "already mounted"}
    if not cfg.get("mount_drive", True):
        return {"mounted": False, "note": "mount_drive disabled"}
    try:
        from google.colab import drive  # type: ignore
    except Exception as exc:
        return {"mounted": False, "error": f"google.colab unavailable: {exc}"}
    logger.info("Mounting Google Drive at %s (interactive authorisation)", mount_point)
    drive.mount(mount_point)
    return {"mounted": Path(mount_point, "MyDrive").exists(), "mount_point": mount_point}


def check_environment_authorisation(cfg: dict, detected: str) -> None:
    configured = str(cfg.get("environment", "")).upper()
    cloud = configured in {"COLAB", "PRIVATE_CLOUD"} or detected == "COLAB"
    if cloud and not cfg.get("cloud_storage_authorized", False):
        raise PermissionError(
            "Cloud storage/processing is disabled. Confirm that your environment complies with the "
            "competition data-use agreement, then set CONFIG['cloud_storage_authorized'] = True."
        )
    if configured == "COLAB" and detected != "COLAB":
        raise RuntimeError("CONFIG['environment'] is COLAB but the runtime is not Google Colab")


def stage_environment(ctx) -> dict:
    cfg = ctx.config
    logger = ctx.logger
    detected = detect_environment()
    check_environment_authorisation(cfg, detected)
    py = check_python()
    if not py["ok"]:
        raise RuntimeError(f"Python >= {py['minimum']} required, found {py['python_version']}")
    mount = {"mounted": False, "note": "not Colab"}
    if detected == "COLAB":
        mount = mount_drive(cfg, logger)
    set_seeds(int(cfg["seed"]))
    discovery = discover_data_root(cfg, cfg["environment"] if detected != "COLAB" else "COLAB")
    if discovery["chosen"] is None:
        ctx.save_json(discovery, "data_discovery_report")
        raise FileNotFoundError(discovery["message"])
    data_root = Path(discovery["chosen"])
    ctx.set_data_root(data_root)
    file_info = []
    for name in cfg["required_files"]:
        p = data_root / name
        st = p.stat()
        file_info.append({"file": name, "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    hw = hardware_report([ctx.paths.root, ctx.paths.local_root])
    gpu = gpu_report(bool(cfg.get("use_gpu")))
    versions = {"core": package_versions(CORE_PACKAGES), "optional": package_versions(OPTIONAL_PACKAGES)}
    missing_core = [k for k, v in versions["core"].items() if not v["available"]]
    report = {
        "detected_environment": detected,
        "configured_environment": cfg["environment"],
        "python": py,
        "drive": mount,
        "data_discovery": discovery,
        "data_root": str(data_root),
        "required_files": file_info,
        "hardware": hw,
        "gpu": gpu,
        "packages": versions,
        "missing_core_packages": missing_core,
        "n_jobs_effective": resolve_n_jobs(int(cfg.get("n_jobs", -1))),
        "seed": int(cfg["seed"]),
    }
    ctx.save_json(report, "environment_report")
    ctx.save_df(pd.DataFrame(file_info), "required_file_inventory", fmt="csv")
    source = {"data_root": str(data_root), "note": "Original files are read in place and never modified."}
    from .io_utils import atomic_write_json

    atomic_write_json(ctx.paths.data_raw / "SOURCE.json", source)
    if missing_core:
        logger.warning("Missing core packages: %s - dependent stages will fail clearly.", missing_core)
    return {
        "outputs": ["environment_report", "required_file_inventory"],
        "summary": {"environment": detected, "data_root": str(data_root), "missing_core": missing_core,
                    "gpu": gpu["will_use_gpu"]},
        "fingerprint_extra": file_info,
    }
