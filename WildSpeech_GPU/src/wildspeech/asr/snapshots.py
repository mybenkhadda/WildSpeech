"""Local checkpoint availability: resolution, file manifest, revision, integrity. Filesystem only — no network.

Resolution order for a model (the first COMPLETE candidate wins; otherwise the first one that exists):
  1. registry `local_path`
  2. <repo>/models/<local_dir>          installed by asr.model_manager (real files, no symlinks), or a pointer
                                         manifest (`storage: hf_cache`) to a complete Hugging Face cache snapshot
  3. Hugging Face cache  <hub>/models--<org>--<name>/snapshots/<pinned sha>   (never refs/main when pinned)
     hub = paths.hf_cache_dir | $HF_HUB_CACHE | $HF_HOME/hub | ~/.cache/huggingface/hub
Cache snapshot entries may be symlinks into ../../blobs (or copies on Windows without symlink support). A broken
symlink counts as a MISSING file. Revision of a local directory: _wildspeech_checkpoint.json (written by the model
manager) > .cache/huggingface/download/*.metadata (huggingface_hub --local-dir) > snapshots/<sha> path > unknown.

Completeness is judged against the registry FILE MANIFEST (name, size, LFS sha256) when the registry has one,
else against per-architecture required file groups. sha256 is checked only when asked (verify_hashes=True) and the
result is cached in the install manifest keyed by (size, mtime) so multi-GB weights are hashed once.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..io_utils import stable_hash

MANIFEST = "_wildspeech_checkpoint.json"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
REQUIRED = {
    "hf_whisper": {"config": ["config.json"], "processor": ["preprocessor_config.json"],
                   "tokenizer": [["tokenizer.json"], ["vocab.json", "merges.txt"]]},
    "hf_ctc": {"config": ["config.json"], "processor": ["preprocessor_config.json"], "tokenizer": [["vocab.json"]]},
    "hf_encoder": {"config": ["config.json"], "processor": ["preprocessor_config.json"]},
    "qwen3_asr": {"config": ["config.json"], "processor": ["preprocessor_config.json"],
                  "tokenizer": [["tokenizer.json"], ["vocab.json", "merges.txt"]]},
}
RECOMMENDED = {"hf_whisper": ["generation_config.json", "tokenizer_config.json"], "hf_ctc": ["tokenizer_config.json"],
               "hf_encoder": [], "qwen3_asr": ["generation_config.json", "tokenizer_config.json"]}
DOWNLOAD_IGNORE = ["*.msgpack", "*.h5", "*.ot", "flax_model*", "tf_model*", "*.onnx", "*.tflite", "*.mlmodel*", "coreml/*"]
SSL_MODEL_TYPES = ("wav2vec2", "hubert", "wavlm", "data2vec-audio", "wav2vec2-conformer", "sew", "sew-d",
                   "unispeech-sat", "unispeech", "wav2vec2-bert")


@dataclass
class SnapshotInfo:
    path: str | None
    origin: str | None                   # local_path | default_dir | hf_cache | hf_cache_pointer | None
    revision: str | None
    revision_source: str | None
    reason: str = ""
    searched: list[str] = field(default_factory=list)
    complete: bool | None = None         # against the registry manifest (None = no manifest to judge)
    missing: list[str] = field(default_factory=list)
    kind: str | None = None              # download | hf_cache | converted | manual


# ------------------------------------------------------------------------------------------ HF cache
def hf_hub_dir(cfg_dir: str | None = None) -> Path:
    if cfg_dir:
        return Path(cfg_dir)
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_repo_dir(hf_id: str | None, hub: Path) -> Path | None:
    if not hf_id or "/" not in hf_id:
        return None
    return Path(hub) / f"models--{hf_id.replace('/', '--')}"


def hf_cache_snapshot(hf_id: str | None, revision: str | None, hub: Path) -> tuple[Path | None, str | None]:
    base = hf_repo_dir(hf_id, hub)
    if base is None:
        return None, None
    snaps = base / "snapshots"
    if not snaps.exists():
        return None, None
    if revision:
        p = snaps / revision
        return (p, revision) if p.exists() else (None, None)
    ref = base / "refs" / "main"
    if ref.exists():
        sha = ref.read_text(encoding="utf-8").strip()
        if (snaps / sha).exists():
            return snaps / sha, sha
    dirs = sorted((d for d in snaps.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    return (dirs[-1], dirs[-1].name) if dirs else (None, None)


def hf_cache_has_repo(hf_id: str | None, hub: Path) -> bool:
    b = hf_repo_dir(hf_id, hub)
    return bool(b is not None and (b / "snapshots").exists())


def file_present(p: Path) -> bool:
    """True for a regular file or a symlink whose target exists (broken symlinks are missing)."""
    try:
        return p.exists() and p.is_file()
    except OSError:
        return False


def file_size(p: Path) -> int | None:
    try:
        return int(p.stat().st_size)          # follows symlinks -> blob size
    except OSError:
        return None


def sha256_file(p: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ revision
def read_manifest(path: str | Path) -> dict:
    p = Path(path)
    if p.is_file():
        p = p.parent
    m = p / MANIFEST
    if m.exists():
        try:
            return json.loads(m.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def read_local_revision(path: str | Path) -> tuple[str | None, str | None]:
    p = Path(path)
    if p.is_file():
        p = p.parent
    rev = read_manifest(p).get("revision")
    if rev:
        return str(rev), "manifest"
    md = p / ".cache" / "huggingface" / "download"
    if md.exists():
        for f in sorted(md.rglob("*.metadata")):
            try:
                first = f.read_text(encoding="utf-8").splitlines()[0].strip()
            except Exception:  # noqa: BLE001
                continue
            if SHA_RE.match(first):
                return first, "hf_download_metadata"
    if p.parent.name == "snapshots" and SHA_RE.match(p.name):
        return p.name, "hf_cache_path"
    return None, None


def looks_like_snapshot(p: Path) -> bool:
    return p.exists() and ((p.is_file() and p.suffix == ".nemo") or file_present(p / "config.json") or
                           (p.is_dir() and any(p.glob("*.nemo"))))


# --------------------------------------------------------------------------------------- manifest
def manifest_status(path: str | Path | None, spec, verify_hashes: bool = False, hash_cache: dict | None = None,
                    accept_converted: bool = True) -> list[dict]:
    """One row per registry FileSpec: present / size_ok / sha_ok (None when not verified or no hash published).
    A converted directory (manifest kind 'converted') may replace pytorch_model.bin by a verified model.safetensors."""
    rows = []
    if not path or not getattr(spec, "files", None):
        return rows
    p = Path(path)
    man = read_manifest(p)
    conv = man.get("converted") or {}
    hc = hash_cache if hash_cache is not None else {}
    for f in spec.files:
        q = p / f.name
        r = {"file": f.name, "role": f.role, "required": f.required, "expected_size": f.size,
             "expected_sha256": f.sha256, "present": file_present(q), "size": None, "size_ok": None,
             "sha_ok": None, "substitute": None}
        if not r["present"] and accept_converted and f.name in conv:
            sub = p / conv[f.name].get("to", "model.safetensors")
            if file_present(sub):
                r.update({"present": True, "substitute": sub.name, "size": file_size(sub), "size_ok": True,
                          "sha_ok": bool(conv[f.name].get("source_sha256_verified"))})
                rows.append(r)
                continue
        if r["present"]:
            r["size"] = file_size(q)
            r["size_ok"] = (r["size"] == f.size) if f.size is not None else None
            if f.sha256 and verify_hashes and r["size_ok"] is not False:
                key = f"{q.resolve()}:{r['size']}:{int(q.stat().st_mtime)}"
                got = hc.get(key)
                if got is None:
                    got = sha256_file(q)
                    hc[key] = got
                r["sha_ok"] = (got == f.sha256)
        rows.append(r)
    return rows


def manifest_problems(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        if not r["required"]:
            continue
        if not r["present"]:
            out.append(f"missing {r['file']}")
        elif r["size_ok"] is False:
            out.append(f"size mismatch {r['file']} ({r['size']} != {r['expected_size']})")
        elif r["sha_ok"] is False:
            out.append(f"sha256 mismatch {r['file']}")
    return out


def _judge(p: Path, spec) -> tuple[bool | None, list[str]]:
    if getattr(spec, "files", None):
        probs = manifest_problems(manifest_status(p, spec))
        return (not probs), probs
    arch = detect_architecture(str(p))
    if arch in REQUIRED or arch == "nemo":
        miss, _ = missing_files(p, arch)
        return (not miss), miss
    return None, []


def resolve_snapshot(spec, hf_cache_dir: str | None = None) -> SnapshotInfo:
    searched, cands = [], []
    for origin, cand in (("local_path", spec.local_path), ("default_dir", spec.default_local_dir)):
        if not cand:
            continue
        p = Path(cand)
        searched.append(str(p))
        man = read_manifest(p) if p.exists() else {}
        if man.get("storage") == "hf_cache" and man.get("source_path"):
            q = Path(man["source_path"])
            if looks_like_snapshot(q):
                cands.append(("hf_cache_pointer", q, man.get("revision"), "manifest", "hf_cache"))
            continue
        if looks_like_snapshot(p):
            rev, src = read_local_revision(p)
            cands.append((origin, p, rev, src, man.get("kind") or ("download" if man else "manual")))
        elif origin == "local_path":
            return SnapshotInfo(None, None, None, None, f"local_path {p} has no config.json / .nemo", searched)
    hub = hf_hub_dir(hf_cache_dir)
    searched.append(str(hub))
    p, rev = hf_cache_snapshot(spec.hf_id, spec.revision, hub)
    if p is not None and looks_like_snapshot(p):
        cands.append(("hf_cache", p, rev, "hf_cache_path", "hf_cache"))
    if not cands:
        why = "checkpoint not on disk"
        if spec.revision:
            why += f" at pinned revision {spec.revision}"
        partial = hf_cache_snapshot(spec.hf_id, spec.revision, hub)[0]
        if partial is not None:
            why += f" (HF cache has a partial snapshot without config.json: {partial})"
        return SnapshotInfo(None, None, None, None, why + f" (searched: {'; '.join(searched)})", searched)
    judged = [(c, *_judge(c[1], spec)) for c in cands]
    pinned = [j for j in judged if not spec.revision or j[0][2] in (None, spec.revision)]
    pool = pinned or judged
    best = next((j for j in pool if j[1] is True), pool[0])
    (origin, path, rev, src, kind), complete, missing = best
    return SnapshotInfo(str(path), origin, rev, src, "", searched, complete, list(missing), kind)


# ------------------------------------------------------------------------------ cross-revision view
def hf_refs(hf_id: str | None, hub: Path) -> dict[str, list[str]]:
    """{commit sha: [ref names]} from <repo>/refs (main, pr/11, ...). Filesystem only."""
    base = hf_repo_dir(hf_id, hub)
    out: dict[str, list[str]] = {}
    if base is None or not (base / "refs").is_dir():
        return out
    for f in (base / "refs").rglob("*"):
        if f.is_file():
            try:
                sha = f.read_text(encoding="utf-8").strip()
            except Exception:  # noqa: BLE001
                continue
            out.setdefault(sha, []).append(str(f.relative_to(base / "refs")).replace("\\", "/"))
    return out


def revision_inventory(spec, hub: Path) -> list[dict]:
    """One row per cached snapshot of the model's repo x manifest file: which revision holds which file.
    Makes "weights in one revision, config in another" visible. Nothing is moved or linked here."""
    base = hf_repo_dir(spec.hf_id, hub)
    if base is None or not (base / "snapshots").is_dir():
        return []
    refs = hf_refs(spec.hf_id, hub)
    rows = []
    names = [f.name for f in (getattr(spec, "files", None) or [])] or ["config.json", "preprocessor_config.json",
                                                                      "model.safetensors", "pytorch_model.bin"]
    for d in sorted(p for p in (base / "snapshots").iterdir() if p.is_dir()):
        for n in names:
            q = d / n
            f = spec.file(n) if hasattr(spec, "file") else None
            sz = file_size(q) if file_present(q) else None
            rows.append({"model_key": spec.key, "snapshot": d.name, "refs": ",".join(refs.get(d.name, [])),
                         "is_pinned": d.name == spec.revision, "file": n, "present": sz is not None, "size": sz,
                         "size_ok": (sz == f.size) if (sz is not None and f is not None and f.size is not None) else None,
                         "broken_link": (q.is_symlink() and not q.exists()) if hasattr(q, "is_symlink") else False})
    return rows


def blob_for(spec, fname: str, hub: Path) -> Path | None:
    """The content-addressed blob of an LFS file at the PINNED manifest hash, if any revision cached it.
    HF stores LFS blobs as blobs/<sha256>; a match by name + size is the same content as the pinned file, so
    re-using it never mixes revisions. Small git files (no sha256 in the registry) have no such guarantee."""
    f = spec.file(fname) if hasattr(spec, "file") else None
    base = hf_repo_dir(spec.hf_id, hub)
    if f is None or not f.sha256 or base is None:
        return None
    b = base / "blobs" / f.sha256
    if file_present(b) and (f.size is None or file_size(b) == f.size):
        return b
    return None


def pinned_snapshot_dir(spec, hub: Path) -> Path | None:
    base = hf_repo_dir(spec.hf_id, hub)
    if base is None or not spec.revision:
        return None
    return base / "snapshots" / spec.revision


# ---------------------------------------------------------------------------------------- files
def missing_files(path: str | Path, architecture: str) -> tuple[list[str], list[str]]:
    """(missing required groups, missing recommended files) for a snapshot of `architecture`."""
    p = Path(path)
    if architecture == "nemo":
        ok = (p.is_file() and p.suffix == ".nemo") or (p.is_dir() and any(p.glob("*.nemo")))
        return ([] if ok else ["*.nemo"]), []
    req = REQUIRED.get(architecture)
    if req is None:
        return [f"no file rules for architecture {architecture}"], []
    miss = []
    if not any(file_present(p / w) for w in WEIGHT_FILES) and not list(p.glob("model-*.safetensors")):
        miss.append("weights (" + " | ".join(WEIGHT_FILES) + ")")
    for group, alts in req.items():
        alts = alts if isinstance(alts[0], list) else [alts]
        if not any(all(file_present(p / f) for f in alt) for alt in alts):
            miss.append(f"{group} (" + " | ".join("+".join(a) for a in alts) + ")")
    rec = [f for f in RECOMMENDED.get(architecture, []) if not file_present(p / f)]
    return miss, rec


def read_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg = Path(path) / "config.json"
    if not file_present(cfg):
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def detect_architecture(path: str | None) -> str:
    """hf_whisper | hf_ctc | hf_encoder | qwen3_asr | hf_seq2seq_other | nemo | unknown (config only, no weights)."""
    if not path:
        return "unknown"
    p = Path(path)
    if p.is_file() and p.suffix == ".nemo":
        return "nemo"
    if p.is_dir() and list(p.glob("*.nemo")):
        return "nemo"
    c = read_config(str(p))
    if not c:
        return "unknown"
    mt = str(c.get("model_type", "")).lower()
    arch = " ".join(c.get("architectures", []) or []).lower()
    if mt == "whisper" or "whisper" in arch:
        return "hf_whisper"
    if mt == "qwen3_asr" or "qwen3asr" in arch:
        return "qwen3_asr"
    if "forctc" in arch:
        return "hf_ctc"
    if mt in SSL_MODEL_TYPES:
        return "hf_encoder"
    if "seq2seq" in arch or "conditionalgeneration" in arch:
        return "hf_seq2seq_other"
    return "unknown"


def remote_code_files(path: str | None) -> dict:
    """auto_map entries of config.json (custom code that would need trust_remote_code)."""
    return dict(read_config(path).get("auto_map") or {})


# ------------------------------------------------------------------------------- download commands
def target_dir(spec) -> str:
    return spec.local_path or spec.default_local_dir or f"models/{getattr(spec, 'local_dir', None) or spec.key}"


def download_commands(spec) -> dict:
    """Ready-to-run commands to fetch the pinned snapshot. This function only builds strings; nothing is executed.
    The project commands download ONLY when the model's legal/provenance/competition dimensions pass and
    ALLOW_MODEL_DOWNLOADS is enabled (see asr.model_manager)."""
    if not spec.hf_id:
        return {"model_key": spec.key, "note": "no hf_id; copy the checkpoint to the target directory manually",
                "target_dir": target_dir(spec)}
    rev = spec.revision or "main"
    tgt = target_dir(spec)
    names = [f.name for f in getattr(spec, "files", []) or []]
    inc = " ".join(f'"{n}"' for n in names) if names else ""
    ign = " ".join(f'"{x}"' for x in DOWNLOAD_IGNORE)
    cli = (f'huggingface-cli download {spec.hf_id} {inc} --revision {rev} --local-dir "{tgt}"' if names else
           f'huggingface-cli download {spec.hf_id} --revision {rev} --local-dir "{tgt}" --exclude {ign}')
    return {"model_key": spec.key, "hf_id": spec.hf_id, "hf_id_status": spec.hf_id_status, "revision": rev,
            "target_dir": tgt, "download_gb": round(sum((f.size or 0) for f in getattr(spec, "files", []) or []) / 1e9, 3),
            "project_cli": f"python -m wildspeech v3-install --model {spec.key} --execute",
            "project_script": f"python scripts\\v3_download_asr_models.py --model {spec.key} --execute",
            "powershell_script": f".\\scripts\\v3_download_asr_models.ps1 -Model {spec.key} -Execute",
            "huggingface_cli": cli,
            "note": ("hf_id NOT verified: confirm the official repo first" if spec.hf_id_status != "verified_public"
                     else "downloads only after the model's reviews pass; review the card at the pinned revision")}


def write_manifest(path: str | Path, spec, revision: str | None, extra: dict | None = None) -> Path:
    """Install manifest written by the model manager (records repo, resolved commit, storage, files, hashes)."""
    from ..io_utils import write_json

    p = Path(path) / MANIFEST
    old = read_manifest(path)
    files = sorted(str(f.relative_to(path)).replace("\\", "/") for f in Path(path).rglob("*")
                   if f.is_file() and ".cache" not in f.parts and f.name != MANIFEST)
    write_json(p, {**old, "model_key": spec.key, "hf_id": spec.hf_id, "revision": revision, "files": files,
                   **(extra or {})})
    return p


# ---------------------------------------------------------------------------- state token (P24)
def _marker(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    out = {"path": str(p)}
    for f in (MANIFEST, "config.json", "model.safetensors", "pytorch_model.bin"):
        q = p / f
        if file_present(q):
            st = q.stat()
            out[f] = [st.st_size, int(st.st_mtime)]
    return out


def p24_state_token(ctx) -> str:
    """Changes when the registry, the reviews or the local snapshots change, so P24 is re-planned after you record
    a review or install a checkpoint (P24 is otherwise fingerprinted on v3.asr config only). Filesystem only."""
    from .model_registry import document_for, reviews_path

    doc = document_for(ctx)
    texts = {}
    for name, p in (("registry", doc.path), ("reviews", reviews_path(ctx))):
        texts[name] = Path(p).read_text(encoding="utf-8") if p and Path(p).exists() else ""
    hub = ctx.cfg.get_path("paths.hf_cache_dir")
    snaps = {k: _marker(resolve_snapshot(s, hub).path) for k, s in sorted(doc.specs.items())}
    return stable_hash({"texts": texts, "snapshots": snaps})
