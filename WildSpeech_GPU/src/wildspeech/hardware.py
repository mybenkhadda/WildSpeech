"""Environment & hardware discovery: OS, Python, packages, RAM, CPU, CUDA/driver compatibility."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "torch", "torchaudio", "soundfile", "soxr", "pywt", "pyarrow",
            "matplotlib", "tqdm", "psutil", "yaml", "statsmodels", "lightgbm", "xgboost", "catboost", "umap",
            "pygam", "transformers", "g2p_en", "hdbscan"]


def package_versions() -> list[dict]:
    rows = []
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            rows.append({"package": name, "available": True, "version": getattr(m, "__version__", "?")})
        except Exception as e:  # noqa: BLE001
            rows.append({"package": name, "available": False, "version": type(e).__name__})
    return rows


def nvidia_smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"nvidia_smi": "not found"}
    try:
        q = "name,driver_version,memory.total,memory.used,temperature.gpu,power.limit"
        out = subprocess.run([exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10).stdout.strip().splitlines()
        if not out:
            return {"nvidia_smi": "no output"}
        vals = [v.strip() for v in out[0].split(",")]
        return dict(zip(["smi_name", "smi_driver", "smi_mem_total_mb", "smi_mem_used_mb", "smi_temp_c", "smi_power_w"], vals))
    except Exception as e:  # noqa: BLE001
        return {"nvidia_smi": f"error {type(e).__name__}"}


def hardware_report() -> dict:
    r = {"os": platform.platform(), "python": sys.version.split()[0], "executable": sys.executable,
         "conda_env": os.environ.get("CONDA_DEFAULT_ENV"), "cpu": platform.processor(),
         "logical_cpus": os.cpu_count()}
    try:
        import psutil

        vm = psutil.virtual_memory()
        r.update({"physical_cpus": psutil.cpu_count(logical=False), "ram_total_gb": vm.total / 1e9,
                  "ram_available_gb": vm.available / 1e9})
        freq = psutil.cpu_freq()
        r["cpu_max_mhz"] = getattr(freq, "max", None)
    except Exception:
        pass
    try:
        import numpy as np

        cfg = np.show_config(mode="dicts") if "mode" in np.show_config.__code__.co_varnames else {}
        blas = (cfg or {}).get("Build Dependencies", {}).get("blas", {})
        r["numpy_blas"] = blas.get("name")
    except Exception:
        pass
    try:
        import torch

        r.update({"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version()})
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            r.update({"gpu_name": pr.name, "gpu_vram_gb": pr.total_memory / 1e9,
                      "compute_capability": f"{pr.major}.{pr.minor}", "gpu_sm_count": pr.multi_processor_count,
                      "bf16_supported": torch.cuda.is_bf16_supported()})
    except Exception as e:  # noqa: BLE001
        r["torch"] = f"unavailable ({type(e).__name__})"
    r.update(nvidia_smi())
    r["compatibility_notes"] = compatibility_notes(r)
    return r


def compatibility_notes(r: dict) -> list[str]:
    notes = []
    if not r.get("cuda_available"):
        notes.append("CUDA not available to torch: install a CUDA build (e.g. cu124/cu126 wheels); "
                     "driver 581.80 (CUDA 13.0) runs any CUDA 12.x torch build.")
    else:
        build = str(r.get("torch_cuda_build") or "")
        if build.startswith("11."):
            notes.append(f"torch built for CUDA {build}; upgrade to a 12.x build for Ampere kernels and recent fixes.")
        if r.get("compute_capability") == "8.6":
            notes.append("RTX 30xx (sm_86): fp16 autocast is efficient; TF32 matmul enabled for embeddings.")
        if r.get("gpu_vram_gb", 0) and r["gpu_vram_gb"] < 9:
            notes.append("8 GB VRAM: keep base-size foundation models, fp16, batch <= 8 for 20 s crops.")
    if (r.get("ram_total_gb") or 0) < 16:
        notes.append("Less than 16 GB RAM: lower runtime.prefetch_batches to 1 and decode_threads to 4.")
    if platform.system() == "Windows":
        notes.append("Windows: no multiprocessing workers are used (threads only); DataLoader num_workers=0.")
    return notes
