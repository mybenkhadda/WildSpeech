"""Hugging Face model manager: plan, download, verify, convert and record local checkpoints. Network ONLY on execute.

Gate (both required, checked per model on every call):
  1. eligibility: registry_validity, license_documentation, license_approval, training_provenance,
     overlap_assessment and competition_permission all PASS (EligibilityRecord.download_allowed). A model whose
     reviews are pending is NEVER downloaded, whatever the flag says.
  2. ALLOW_MODEL_DOWNLOADS: v3.asr.downloads.allow_model_downloads: true, or the environment variable
     ALLOW_MODEL_DOWNLOADS / WILDSPEECH_ALLOW_MODEL_DOWNLOADS = 1|true|yes, or execute(..., allow=True) from the
     CLI flag --execute. Default False.

Storage per model (v3.asr.downloads.storage = auto | local_dir | hf_cache):
  * the Hugging Face cache already holds a COMPLETE snapshot at the pinned commit -> used in place, nothing is
    downloaded or copied (e.g. facebook/wav2vec2-base-960h on this machine)
  * the cache holds a PARTIAL snapshot of the repo (auto) -> only the missing manifest files are fetched INTO the
    cache with hf_hub_download(revision=<sha>), so existing blobs are reused, never duplicated
  * otherwise -> models/<local_dir>/ with hf_hub_download(local_dir=...): real files (no symlinks, works on Windows
    without Developer Mode), resumable (.incomplete files are continued), metadata under .cache/huggingface
Only files listed in the registry manifest are fetched (no TF / Flax / ONNX duplicates). Revisions are always the
pinned 40-character commit. After download every file is checked against the published size and, where the Hub
publishes one, the LFS sha256 (hash results cached in models/_sha256_cache.json).

Verified .bin -> safetensors conversion (v3.asr.downloads.convert_verified_bin, default true): transformers >= 4.50
refuses torch.load on torch < 2.6 (CVE-2025-32434). Instead of upgrading the working CUDA torch, the manager checks
that pytorch_model.bin has exactly the sha256 published by the official repo, loads it with
torch.load(weights_only=True, map_location="cpu"), writes model.safetensors (+ config / processor / tokenizer files)
to models/<local_dir>/ with manifest kind "converted", and re-opens the result to compare every tensor name and
shape. The original file is never modified or deleted.

hf_xet: huggingface_hub uses the optional `hf_xet` package automatically for Xet-backed repos when installed
(pip install hf_xet); v3.asr.downloads.use_hf_xet: false sets HF_HUB_DISABLE_XET=1.
The pipeline never loads with trust_remote_code=True (see asr.environment.framework_check).
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..io_utils import read_json, write_json
from .snapshots import (MANIFEST, file_present, file_size, hf_cache_has_repo, hf_cache_snapshot, hf_hub_dir,
                        manifest_problems, manifest_status, sha256_file, write_manifest)

TRUE = {"1", "true", "yes", "on"}
INSTALL_MANIFEST = "model_installation_manifest.json"


def downloads_allowed(cfg, explicit: bool | None = None, model_key: str | None = None) -> tuple[bool, str]:
    """Global switch (explicit argument > environment > config) AND, when v3.asr.downloads.authorized_models is a
    non-empty list, the model must be named in it (per-model download authorization)."""
    ok, why = False, "ALLOW_MODEL_DOWNLOADS is not enabled"
    if explicit is not None:
        ok, why = bool(explicit), "explicit argument"
    else:
        for var in ("ALLOW_MODEL_DOWNLOADS", "WILDSPEECH_ALLOW_MODEL_DOWNLOADS"):
            if str(os.environ.get(var, "")).strip().lower() in TRUE:
                ok, why = True, f"environment {var}"
                break
        else:
            if bool(cfg.get_path("v3.asr.downloads.allow_model_downloads", False)):
                ok, why = True, "v3.asr.downloads.allow_model_downloads"
    auth = cfg.get_path("v3.asr.downloads.authorized_models") or []
    if ok and model_key is not None and auth and model_key not in set(auth):
        return False, f"{model_key} is not in v3.asr.downloads.authorized_models {list(auth)}"
    return ok, why + (f" (authorized_models={list(auth)})" if (ok and auth) else "")


def repair_from_blobs(spec, hub: Path, hc: dict, verify: bool = True, log=None) -> list[dict]:
    """Complete the PINNED cache snapshot from blobs already on disk, without any network access.

    Only LFS files whose registry sha256 names an existing blob (blobs/<sha256>, cached by ANY revision of the same
    repo) are linked; with verify=True the blob is re-hashed first. Small git files have no content hash in the
    registry, so they are never borrowed from another revision (they must be fetched at the pinned commit).
    Link order: relative symlink -> hard link (same volume, no Developer Mode needed on Windows) -> copy."""
    from .snapshots import blob_for, pinned_snapshot_dir

    snap = pinned_snapshot_dir(spec, hub)
    out = []
    if snap is None or not spec.files:
        return out
    for f in spec.files:
        dst = snap / f.name
        if file_present(dst) and (f.size is None or file_size(dst) == f.size):
            continue
        b = blob_for(spec, f.name, hub)
        row = {"model_key": spec.key, "file": f.name, "blob": str(b) if b else None, "action": "none", "ok": False}
        if b is None:
            row["action"] = "needs_download" if f.sha256 is None else "no_cached_blob"
            out.append(row)
            continue
        if verify:
            key = f"{b.resolve()}:{file_size(b)}:{int(b.stat().st_mtime)}"
            got = hc.get(key) or sha256_file(b)
            hc[key] = got
            if got != f.sha256:
                row.update(action="blob_hash_mismatch", error=f"{got} != {f.sha256}")
                out.append(row)
                continue
        snap.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() and not dst.exists():
            dst.unlink()                                   # broken link of the PINNED snapshot: nothing is lost
        elif dst.exists():                                 # wrong-size file: kept aside, never deleted
            os.replace(dst, dst.with_name(f"{dst.name}.size_mismatch_{time.strftime('%Y%m%dT%H%M%S')}"))
        how = None
        try:
            os.symlink(os.path.relpath(b, dst.parent), dst)
            how = "symlink"
        except (OSError, NotImplementedError):
            try:
                os.link(b, dst)
                how = "hardlink"
            except OSError:
                shutil.copyfile(b, dst)
                how = "copy"
        row.update(action=f"linked_{how}", ok=file_present(dst) and file_size(dst) == (f.size or file_size(dst)))
        if log:
            log.info("repaired %s/%s from cached blob %s (%s)", spec.key, f.name, b.name[:12], how)
        out.append(row)
    return out


def repair_plan(ctx, keys: list[str] | None = None) -> pd.DataFrame:
    """What repair_from_blobs could complete offline (no linking, no hashing): filesystem only."""
    from .model_registry import document_for
    from .snapshots import blob_for, pinned_snapshot_dir

    doc = document_for(ctx)
    hub = _hub(ctx)
    rows = []
    for k in list(keys or doc.specs):
        s = doc.specs.get(k)
        if s is None or not s.hf_id:
            continue
        snap = pinned_snapshot_dir(s, hub)
        for f in s.files:
            dst = snap / f.name if snap is not None else None
            have = bool(dst is not None and file_present(dst) and (f.size is None or file_size(dst) == f.size))
            b = None if have else blob_for(s, f.name, hub)
            rows.append({"model_key": k, "file": f.name, "size": f.size, "pinned_present": have,
                         "lfs_sha256": bool(f.sha256), "offline_repairable": bool(b is not None),
                         "needs_download": (not have) and b is None,
                         "note": "" if have else ("cached blob at the pinned sha256" if b is not None else
                                                  ("not cached: fetch at the pinned commit" if f.sha256 else
                                                   "small git file: fetch at the pinned commit (never borrowed "
                                                   "from another revision)"))})
    return pd.DataFrame(rows)


def _dl_cfg(cfg) -> dict:
    return dict(cfg.get_path("v3.asr.downloads", {}) or {})


def _hub(ctx) -> Path:
    return hf_hub_dir(ctx.cfg.get_path("paths.hf_cache_dir"))


def _install_manifest_path(ctx) -> Path:
    from .run_state import models_root

    return models_root(ctx) / INSTALL_MANIFEST


def _local_dir(spec) -> Path:
    return Path(spec.default_local_dir)


# =========================================================================================== planning
def _cache_rows(spec, hub: Path, hc: dict) -> tuple[Path | None, list[dict]]:
    p, _ = hf_cache_snapshot(spec.hf_id, spec.revision, hub)
    return p, (manifest_status(p, spec, False, hc) if p is not None else [])


def plan_model(ctx, spec, rec=None, hc: dict | None = None) -> dict:
    """What install() would do for one model (filesystem only)."""
    from .environment import bin_load_blocked

    d = _dl_cfg(ctx.cfg)
    hub = _hub(ctx)
    hc = hc if hc is not None else {}
    storage_pref = str(d.get("storage", "auto"))
    local = _local_dir(spec)
    loc_rows = manifest_status(local, spec, False, hc) if local.exists() else []
    loc_ok = bool(loc_rows) and not manifest_problems(loc_rows)
    cpath, c_rows = _cache_rows(spec, hub, hc)
    c_ok = bool(c_rows) and not manifest_problems(c_rows)
    if loc_ok:
        storage, base, rows = "local_dir", local, loc_rows
    elif c_ok:
        storage, base, rows = "hf_cache_in_place", cpath, c_rows
    elif storage_pref == "hf_cache" or (storage_pref == "auto" and hf_cache_has_repo(spec.hf_id, hub)):
        storage, base, rows = "hf_cache", cpath, c_rows
    else:
        storage, base, rows = "local_dir", local, loc_rows
    have = {r["file"] for r in rows if r["present"] and r["size_ok"] is not False}
    missing = [f for f in spec.files if f.name not in have]
    need = int(sum(f.size or 0 for f in missing))
    blocked, why = bin_load_blocked()
    conv_needed = bool(spec.bin_only() and blocked and bool(d.get("convert_verified_bin", True)))
    conv_done = conv_needed and (local / "model.safetensors").exists() and \
        (read_json(local / MANIFEST, {}) or {}).get("kind") == "converted"
    conv_bytes = int(sum(f.size or 0 for f in spec.weight_files())) if (conv_needed and not conv_done) else 0
    target_for_disk = (hub if storage.startswith("hf_cache") else local)
    probe = target_for_disk
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free if probe.exists() else None
    return {"model_key": spec.key, "hf_id": spec.hf_id, "revision": spec.revision, "storage": storage,
            "path": str(base) if base else None, "local_dir": str(local),
            "lifecycle": getattr(rec, "lifecycle", None), "download_allowed": getattr(rec, "download_allowed", None),
            "manifest_files": len(spec.files), "present_files": len(have), "missing_files": ",".join(f.name for f in missing),
            "download_bytes": need, "download_gb": round(need / 1e9, 3), "conversion_needed": conv_needed and not conv_done,
            "conversion_note": why if conv_needed else "", "conversion_gb": round(conv_bytes / 1e9, 3),
            "disk_free_gb": round(free / 1e9, 2) if free is not None else None,
            "disk_ok": (free is None) or (free > 1.10 * (need + conv_bytes) + 2e8)}


def install_plan(ctx, keys: list[str] | None = None, recs: dict | None = None) -> pd.DataFrame:
    from .eligibility import checker_for
    from .model_registry import document_for
    from .run_state import load_hash_cache

    doc = document_for(ctx)
    keys = list(keys or ctx.cfg.get_path("v3.asr.models", list(doc.specs)) or list(doc.specs))
    recs = recs or checker_for(ctx).check_all(doc.specs, keys, probe_files=True)
    hc = load_hash_cache(ctx)
    return pd.DataFrame([plan_model(ctx, doc.specs[k], recs.get(k), hc) for k in keys if k in doc.specs])


# =========================================================================================== execution
class _online:
    """Lift HF offline mode for the duration of an approved download only. huggingface_hub reads HF_HUB_OFFLINE
    into `constants` at import time, so both the environment and the constant are switched and restored."""

    def __enter__(self):
        self.env = {v: os.environ.get(v) for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        for v in self.env:
            os.environ.pop(v, None)
        try:
            from huggingface_hub import constants

            self.const = getattr(constants, "HF_HUB_OFFLINE", None)
            constants.HF_HUB_OFFLINE = False
        except Exception:  # noqa: BLE001
            self.const = None
        return self

    def __exit__(self, *exc):
        for v, val in self.env.items():
            if val is not None:
                os.environ[v] = val
        if self.const is not None:
            from huggingface_hub import constants

            constants.HF_HUB_OFFLINE = self.const
        return False


def _download_file(spec, fname: str, storage: str, local: Path, hub: Path, log=None) -> Path:
    from huggingface_hub import hf_hub_download

    kw = {"repo_id": spec.hf_id, "filename": fname, "revision": spec.revision, "etag_timeout": 30}
    if storage.startswith("hf_cache"):
        kw["cache_dir"] = str(hub)
    else:
        local.mkdir(parents=True, exist_ok=True)
        kw["local_dir"] = str(local)
    t0 = time.perf_counter()
    with _online():
        p = Path(hf_hub_download(**kw))
    if log:
        log.info("downloaded %s/%s (%.1f MB, %.1fs)", spec.key, fname, (file_size(p) or 0) / 1e6, time.perf_counter() - t0)
    return p


def convert_verified_bin(spec, src_dir: Path, dst_dir: Path, hc: dict, log=None) -> dict:
    """sha256-verified pytorch_model.bin -> model.safetensors (+ small files) in dst_dir. Raises on any mismatch."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    wf = next((f for f in spec.weight_files() if f.name.endswith(".bin")), None)
    if wf is None:
        raise ValueError(f"{spec.key}: registry lists no .bin weights to convert")
    if not wf.sha256:
        raise ValueError(f"{spec.key}: refusing to torch.load {wf.name} without a published sha256")
    src = Path(src_dir) / wf.name
    if not file_present(src):
        raise FileNotFoundError(src)
    if (file_size(src) or -1) != (wf.size or -2):
        raise ValueError(f"{spec.key}: {wf.name} size {file_size(src)} != published {wf.size}")
    key = f"{src.resolve()}:{file_size(src)}:{int(src.stat().st_mtime)}"
    got = hc.get(key) or sha256_file(src)
    hc[key] = got
    if got != wf.sha256:
        raise ValueError(f"{spec.key}: {wf.name} sha256 {got} != published {wf.sha256}; not loading it")
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in spec.files:
        if f.role != "weights" and file_present(Path(src_dir) / f.name) and Path(src_dir).resolve() != dst_dir.resolve():
            shutil.copyfile(Path(src_dir) / f.name, dst_dir / f.name)
    t0 = time.perf_counter()
    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    tensors = {k: v.detach().clone().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    out = dst_dir / "model.safetensors.tmp"
    save_file(tensors, str(out), metadata={"format": "pt", "source": wf.name, "source_sha256": wf.sha256})
    with safe_open(str(out), framework="pt") as fh:
        keys = set(fh.keys())
        bad = [k for k in tensors if k not in keys or tuple(fh.get_slice(k).get_shape()) != tuple(tensors[k].shape)]
    if bad or len(keys) != len(tensors):
        out.unlink(missing_ok=True)
        raise ValueError(f"{spec.key}: converted file does not match the state dict ({bad[:5]})")
    os.replace(out, dst_dir / "model.safetensors")
    res = {"pytorch_model.bin": {"to": "model.safetensors", "source": str(src), "source_sha256": wf.sha256,
                                 "source_sha256_verified": True, "tensors": len(tensors),
                                 "output_sha256": sha256_file(dst_dir / "model.safetensors"),
                                 "seconds": round(time.perf_counter() - t0, 1)}}
    if log:
        log.info("converted %s: %d tensors -> %s", spec.key, len(tensors), dst_dir / "model.safetensors")
    return res


def install_model(ctx, spec, rec, allow: bool, verify: bool = True, log=None, hc: dict | None = None) -> dict:
    """Install ONE model if (and only if) its eligibility allows downloads and `allow` is set."""
    from .environment import bin_load_blocked
    from .run_state import HeartbeatLock, download_lock_path

    log = log or ctx.logger
    d = _dl_cfg(ctx.cfg)
    hc = hc if hc is not None else {}
    plan = plan_model(ctx, spec, rec, hc)
    res = {**plan, "action": "none", "ok": False, "error": ""}
    if not getattr(rec, "download_allowed", False):
        res.update(action="blocked_by_eligibility",
                   error=f"lifecycle {getattr(rec, 'lifecycle', '?')}: reviews must pass before any download")
        return res
    if not spec.hf_id or not spec.revision:
        res.update(action="blocked", error="no hf_id / pinned revision")
        return res
    if d.get("use_hf_xet") is False:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    hub, local = _hub(ctx), _local_dir(spec)
    storage = plan["storage"]
    missing = [m for m in plan["missing_files"].split(",") if m]
    res["offline_repaired"] = ""
    if missing and storage.startswith("hf_cache") and bool(d.get("offline_repair", True)):
        rep = repair_from_blobs(spec, hub, hc, verify=bool(verify), log=log)
        fixed = [r["file"] for r in rep if r.get("ok")]
        if fixed:
            res["offline_repaired"] = ",".join(fixed)
            plan = plan_model(ctx, spec, rec, hc)
            res.update({k: v for k, v in plan.items() if k in res})
            storage = plan["storage"]
            missing = [m for m in plan["missing_files"].split(",") if m]
    if missing and not allow:
        res.update(action="dry_run", error="ALLOW_MODEL_DOWNLOADS not enabled: nothing downloaded")
        return res
    if missing and not plan["disk_ok"]:
        res.update(action="blocked_disk", error=f"not enough free disk: {plan['disk_free_gb']} GB")
        return res
    t0 = time.perf_counter()
    with HeartbeatLock(download_lock_path(ctx, spec.key), {"model_key": spec.key, "files": missing, "storage": storage}):
        try:
            for fname in missing:
                _download_file(spec, fname, storage, local, hub, log)
            base = local if storage == "local_dir" else hf_cache_snapshot(spec.hf_id, spec.revision, hub)[0]
            if base is None:
                raise FileNotFoundError(f"{spec.key}: snapshot {spec.revision} not found after download")
            rows = manifest_status(base, spec, bool(verify), hc)
            probs = manifest_problems(rows)
            if probs:
                raise ValueError(f"{spec.key}: integrity check failed: {'; '.join(probs)}")
            extra = {"storage": storage, "source_path": str(base), "installed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "kind": "download" if storage == "local_dir" else "hf_cache",
                     "sha256_verified": {f"{r['file']}": r["expected_sha256"] for r in rows if r["sha_ok"]}}
            if plan["conversion_needed"]:
                conv = convert_verified_bin(spec, base, local, hc, log)
                extra.update(kind="converted", converted=conv, storage="converted", source_path=str(base))
                write_manifest(local, spec, spec.revision, extra)
            elif (read_json(local / MANIFEST, {}) or {}).get("kind") == "converted":
                pass                                                    # verified conversion already in place
            elif storage == "local_dir":
                write_manifest(local, spec, spec.revision, extra)
            else:
                local.mkdir(parents=True, exist_ok=True)
                write_manifest(local, spec, spec.revision, extra)       # pointer manifest: storage hf_cache
            res.update(action="installed" if missing else ("converted" if extra.get("kind") == "converted" else "verified"),
                       ok=True, path=str(local if extra.get("kind") in ("converted", "download") else base),
                       storage=extra["storage"], seconds=round(time.perf_counter() - t0, 1),
                       sha256_verified=",".join(extra["sha256_verified"]))
        except Exception as e:  # noqa: BLE001 - recorded; the next model still installs
            res.update(action="failed", ok=False, error=f"{type(e).__name__}: {e}")
            log.exception("install %s failed", spec.key)
    return res


def install(ctx, keys: list[str] | None = None, allow: bool | None = None, verify: bool = True) -> pd.DataFrame:
    """Install every requested model whose eligibility allows it. Writes models/model_installation_manifest.json."""
    from .eligibility import checker_for
    from .model_registry import document_for
    from .run_state import load_hash_cache, save_hash_cache

    doc = document_for(ctx)
    keys = list(keys or ctx.cfg.get_path("v3.asr.models", list(doc.specs)) or list(doc.specs))
    ok_flag, why = downloads_allowed(ctx.cfg, allow)
    recs = checker_for(ctx).check_all(doc.specs, keys, probe_files=True)
    hc = load_hash_cache(ctx)
    rows = []
    for k in keys:
        if k not in doc.specs:
            continue
        ok_k, why_k = downloads_allowed(ctx.cfg, allow, k)
        r = install_model(ctx, doc.specs[k], recs[k], ok_k, verify, ctx.logger, hc)
        r["allow_source"] = why_k
        rows.append(r)
    save_hash_cache(ctx, hc)
    t = pd.DataFrame(rows)
    write_installation_manifest(ctx)
    return t


def write_installation_manifest(ctx) -> Path:
    """models/model_installation_manifest.json: per model the resolved snapshot, storage, revision and file checks."""
    from .eligibility import checker_for
    from .model_registry import document_for
    from .run_state import load_hash_cache
    from .snapshots import resolve_snapshot

    doc = document_for(ctx)
    hc = load_hash_cache(ctx)
    recs = checker_for(ctx).check_all(doc.specs, list(doc.specs), probe_files=True)
    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "hf_hub_cache": str(_hub(ctx)), "models": {}}
    for k, s in doc.specs.items():
        sn = resolve_snapshot(s, ctx.cfg.get_path("paths.hf_cache_dir"))
        rows = manifest_status(sn.path, s, False, hc) if sn.path else []
        out["models"][k] = {"hf_id": s.hf_id, "pinned_revision": s.revision, "lifecycle": recs[k].lifecycle,
                            "installed": sn.path is not None and not manifest_problems(rows),
                            "path": sn.path, "origin": sn.origin, "kind": sn.kind, "revision": sn.revision,
                            "manifest_gb": round(s.manifest_bytes() / 1e9, 3),
                            "files": [{k2: v for k2, v in r.items() if k2 in ("file", "present", "size", "size_ok",
                                                                               "sha_ok", "substitute")} for r in rows],
                            "local_manifest": read_json(Path(s.default_local_dir) / MANIFEST, {}) or {}}
    p = _install_manifest_path(ctx)
    write_json(p, out)
    return p


def verify_installed(ctx, keys: list[str] | None = None) -> pd.DataFrame:
    """Hash every installed manifest file with a published sha256 (cached) -> one row per file."""
    from .model_registry import document_for
    from .run_state import load_hash_cache, save_hash_cache
    from .snapshots import resolve_snapshot

    doc = document_for(ctx)
    hc = load_hash_cache(ctx)
    rows = []
    for k in list(keys or doc.specs):
        s = doc.specs.get(k)
        if s is None:
            continue
        sn = resolve_snapshot(s, ctx.cfg.get_path("paths.hf_cache_dir"))
        for r in (manifest_status(sn.path, s, True, hc) if sn.path else []):
            rows.append({"model_key": k, "path": sn.path, **r})
    save_hash_cache(ctx, hc)
    return pd.DataFrame(rows)


def plan_as_records(df: pd.DataFrame) -> list[dict]:
    return [] if df is None or df.empty else df.to_dict("records")


__all__ = ["downloads_allowed", "install_plan", "install", "install_model", "convert_verified_bin",
           "verify_installed", "write_installation_manifest", "asdict"]
