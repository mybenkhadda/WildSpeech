"""TAR manifests with byte offsets: one header pass, then O(1) random access without extraction.

The original archive is opened read-only and never modified or extracted to disk.
Compressed archives cannot be seeked; they are flagged and served by sequential streaming.
"""
from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

from ..io_utils import atomic_write_parquet, file_fingerprint, read_json, write_json

AUDIO_EXT = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a")


def detect_compression(path: Path) -> str | None:
    with open(path, "rb") as fh:
        head = fh.read(6)
    if head[:2] == b"\x1f\x8b":
        return "gz"
    if head[:3] == b"BZh":
        return "bz2"
    if head[:6] == b"\xfd7zXZ\x00":
        return "xz"
    return None


def safe_member_name(name: str) -> bool:
    """Reject absolute paths, parent traversal and drive letters (we never extract, but we audit)."""
    norm = name.replace("\\", "/")
    p = PurePosixPath(norm)
    if not p.parts or norm.startswith("/") or p.is_absolute():
        return False
    if ".." in p.parts:
        return False
    if ":" in p.parts[0]:
        return False
    return True


def build_tar_manifest(tar_path: Path, cache_dir: Path, force: bool = False, logger=None) -> pd.DataFrame:
    """Columns: filename, member, offset, size, compressed, safe. Cached by archive fingerprint."""
    tar_path = Path(tar_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{tar_path.stem}_manifest.parquet"
    meta_path = cache_dir / f"{tar_path.stem}_manifest.json"
    fp = file_fingerprint(tar_path)
    meta = read_json(meta_path, {})
    if out.exists() and not force and meta.get("fingerprint") == fp:
        return pd.read_parquet(out)
    comp = detect_compression(tar_path)
    rows = []
    mode = f"r:{comp}" if comp else "r:"
    from tqdm.auto import tqdm

    with tarfile.open(tar_path, mode=mode) as tf:
        bar = tqdm(desc=f"manifest {tar_path.name}", unit="member", dynamic_ncols=True)
        for m in tf:
            bar.update(1)
            if not m.isfile():
                continue
            base = m.name.replace("\\", "/").split("/")[-1]
            if not base.lower().endswith(AUDIO_EXT) or base.startswith("._"):
                continue
            rows.append({"filename": base, "member": m.name, "offset": int(m.offset_data) if not comp else -1,
                         "size": int(m.size), "compressed": bool(comp), "safe": safe_member_name(m.name)})
        bar.close()
    df = pd.DataFrame(rows, columns=["filename", "member", "offset", "size", "compressed", "safe"])
    dup = df["filename"].duplicated(keep="first")
    if dup.any() and logger:
        logger.warning("%s: %d duplicate basenames; keeping first occurrence", tar_path.name, int(dup.sum()))
    df = df[~dup].reset_index(drop=True)
    atomic_write_parquet(df, out)
    write_json(meta_path, {"fingerprint": fp, "n_members": len(df), "compression": comp})
    return df
