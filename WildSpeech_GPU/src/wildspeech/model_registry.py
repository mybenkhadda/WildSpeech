"""Local foundation-model registry with license / training-overlap gating. Never downloads unless allowed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_yaml


class ModelNotEligible(RuntimeError):
    pass


def load_registry(config_dir: Path) -> dict:
    p = Path(config_dir) / "foundation_models.yaml"
    if not p.exists():
        return {}
    return load_yaml(p) or {}


def eligibility_table(reg: dict) -> pd.DataFrame:
    rows = []
    for key, m in reg.items():
        ok = bool(m.get("license_verified")) and bool(m.get("overlap_checked")) and bool(m.get("hf_id") or m.get("local_path"))
        rows.append({"model": key, "hf_id": m.get("hf_id"), "kind": m.get("kind"), "license": m.get("license"),
                     "child_speech": m.get("child_speech"), "license_verified": m.get("license_verified"),
                     "overlap_checked": m.get("overlap_checked"), "eligible": ok})
    return pd.DataFrame(rows)


def resolve_model_source(key: str, reg: dict, cfg) -> tuple[str, dict]:
    """Return (path_or_id, from_pretrained kwargs) or raise ModelNotEligible."""
    if key not in reg:
        raise ModelNotEligible(f"{key} not in configs/foundation_models.yaml")
    m = reg[key]
    if not (m.get("license_verified") and m.get("overlap_checked")):
        raise ModelNotEligible(f"{key}: set license_verified and overlap_checked to true after reviewing the model card")
    src = m.get("local_path") or m.get("hf_id")
    if not src:
        raise ModelNotEligible(f"{key}: no hf_id/local_path configured")
    allow_dl = bool(cfg.get_path("runtime.allow_model_download", False))
    kwargs = {"local_files_only": (not allow_dl) or bool(cfg.get_path("runtime.hf_local_files_only", True))}
    cache = cfg.get_path("paths.hf_cache_dir")
    if cache:
        kwargs["cache_dir"] = cache
    return src, kwargs
