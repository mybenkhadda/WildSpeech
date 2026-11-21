"""Centralised configuration: YAML defaults + execution profile + user overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from .io_utils import read_json, stable_hash

PROFILES = ("SMOKE", "FAST", "FULL", "RESEARCH", "RESUME", "SUBMISSION",
            "V2_SMOKE", "V2_CORE", "V2_PHONETIC", "V2_FOUNDATION", "V2_FULL", "V2_SUBMISSION",
            "V3_SMOKE", "V3_AUDIT", "V3_ASR", "V3_SCIENCE", "V3_FULL", "V3_SUBMISSION",
            "V4")
RELOAD_PROFILES = ("RESUME", "SUBMISSION", "V2_SUBMISSION")


class Config(dict):
    """A nested dict with dotted-path access: cfg.get_path("audio.n_fft")."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict:
        return dict(self.get(name, {}) or {})

    def fingerprint(self, keys: Iterable[str]) -> str:
        return stable_hash({k: self.get_path(k) for k in sorted(set(keys))})

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def parse_set_args(items: Iterable[str] | None) -> dict:
    """CLI helper: ["audio.n_fft=1024", "gpu.batch_size=32"] -> nested dict."""
    cfg = Config()
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        cfg.set_path(key.strip(), _parse_scalar(value.strip()))
    return dict(cfg)


def find_config_dir(start: Path | None = None) -> Path:
    here = Path(start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        cand = parent / "configs" / "default.yaml"
        if cand.exists():
            return cand.parent
    raise FileNotFoundError("configs/default.yaml not found above " + str(here))


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def v1_snapshot(path: Path | None) -> dict:
    """V1 sections of a saved run configuration (the `v2` section and the profile name are dropped)."""
    snap = read_json(path) if path is not None else None
    if not snap or not isinstance(snap.get("config"), dict):
        return {}
    cfg = copy.deepcopy(snap["config"])
    cfg.pop("v2", None)
    cfg.get("runtime", {}).pop("profile", None)
    return cfg


def load_config(profile: str = "FAST", config_dir: Path | None = None, overrides: dict | None = None,
                last_run_path: Path | None = None, v1_base_path: Path | None = None) -> tuple[Config, list[str] | None]:
    """Return (config, default stage list of the profile).

    V2 research profiles (V2_*) are layered on the configuration of the completed V1 run (v1_base_path, by default
    models/champion_config.json written by P13): default.yaml < V1 run snapshot < v2.yaml < profile < overrides.
    Every P00-P15 fingerprint therefore matches the completed run and those stages stay cached.
    """
    profile = profile.upper()
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile}; choose from {PROFILES}")
    cdir = Path(config_dir) if config_dir else find_config_dir()
    base = load_yaml(cdir / "default.yaml")
    if profile.startswith(("V2_", "V3_")) and profile not in RELOAD_PROFILES:
        base = deep_merge(base, v1_snapshot(v1_base_path))
    # V2 settings live exclusively under the top-level `v2` key, so merging them never changes the
    # fingerprint of a P00-P15 stage (those only hash the sections they declare). V3 lives under `v3`.
    if (cdir / "v2.yaml").exists():
        base = deep_merge(base, load_yaml(cdir / "v2.yaml") or {})
    if (cdir / "v3.yaml").exists():
        base = deep_merge(base, load_yaml(cdir / "v3.yaml") or {})
    # V4 (Colab multimodal research) lives under the top-level `v4` key only -> no P00-P29 fingerprint changes.
    if (cdir / "v4.yaml").exists():
        base = deep_merge(base, load_yaml(cdir / "v4.yaml") or {})
    profiles = load_yaml(cdir / "profiles.yaml")
    spec = profiles.get(profile, {}) or {}
    stages = spec.get("stages")
    # V3 profiles reproduce the configuration the V2 stages were run with (base_profile overrides) so that the
    # completed V2 artefacts keep their fingerprints instead of being re-planned as stale.
    bp = spec.get("base_profile")
    if bp:
        base = deep_merge(base, (profiles.get(str(bp).upper(), {}) or {}).get("overrides"))
    if profile in RELOAD_PROFILES and last_run_path is not None:
        last = read_json(last_run_path)
        if last:
            base = deep_merge(base, last.get("config", {}))
            stages = stages or last.get("stages")
    merged = deep_merge(base, spec.get("overrides"))
    merged = deep_merge(merged, overrides)
    merged.setdefault("runtime", {})["profile"] = profile
    return Config(merged), (list(stages) if stages else None)
