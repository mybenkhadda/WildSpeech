"""Safe TAR inspection, audio inventory and safe extraction.

Archives are never extracted before their members have been inspected and the
available disk space has been checked. Only regular files with safe relative
paths are ever written; symlinks, hard links, devices, absolute paths and
path traversal are rejected.
"""
from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .io_utils import atomic_write_json, read_json

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm", ".aiff", ".aif"}


class UnsafeArchiveError(RuntimeError):
    pass


def member_safety(member: tarfile.TarInfo) -> tuple[bool, str]:
    name = member.name
    if member.issym():
        return False, "symlink"
    if member.islnk():
        return False, "hardlink"
    if member.isdir():
        return False, "directory"
    if not member.isreg():
        return False, "not_regular_file"
    if not name or "\x00" in name:
        return False, "invalid_name"
    normalized = name.replace("\\", "/")
    p = PurePosixPath(normalized)
    if p.is_absolute() or normalized.startswith("/"):
        return False, "absolute_path"
    if len(normalized) >= 2 and normalized[1] == ":":
        return False, "drive_letter"
    if any(part == ".." for part in p.parts):
        return False, "path_traversal"
    if member.size < 0:
        return False, "negative_size"
    return True, "ok"


def inspect_archive(tar_path: Path) -> pd.DataFrame:
    """List archive members without extracting anything."""
    rows = []
    with tarfile.open(tar_path, mode="r:*") as tf:
        for m in tf:
            safe, reason = member_safety(m)
            if m.isdir():
                continue
            name = m.name.replace("\\", "/")
            rows.append({
                "archive": tar_path.name, "member": name, "basename": PurePosixPath(name).name,
                "stem": PurePosixPath(name).stem, "extension": PurePosixPath(name).suffix.lower(),
                "size_bytes": int(m.size), "offset_data": int(getattr(m, "offset_data", -1)),
                "safe": safe, "safety_reason": reason,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["archive", "member", "basename", "stem", "extension", "size_bytes",
                                     "offset_data", "safe", "safety_reason", "is_audio_ext"])
    df["is_audio_ext"] = df["extension"].isin(AUDIO_EXTENSIONS)
    return df


def _audio_info_from_bytes(data: bytes) -> dict:
    info = {"decode_status": "OK", "sample_rate": np.nan, "channels": np.nan, "frames": np.nan,
            "duration_s": np.nan, "format": None, "subtype": None, "decode_backend": None}
    try:
        import soundfile as sf  # type: ignore

        with sf.SoundFile(io.BytesIO(data)) as f:
            info.update(sample_rate=float(f.samplerate), channels=float(f.channels), frames=float(f.frames),
                        duration_s=float(f.frames) / float(f.samplerate) if f.samplerate else np.nan,
                        format=f.format, subtype=f.subtype, decode_backend="soundfile")
            probe = f.read(frames=min(int(f.frames), 2048), dtype="float32", always_2d=True)
            if probe.size and not np.all(np.isfinite(probe)):
                info["decode_status"] = "INVALID_VALUES"
        if info["frames"] == 0:
            info["decode_status"] = "EMPTY"
        return info
    except Exception as exc:
        first_error = f"{type(exc).__name__}: {exc}"
    try:
        import tempfile

        import librosa  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=False)
        finally:
            os.unlink(tmp_path)
        channels = 1 if y.ndim == 1 else y.shape[0]
        frames = y.shape[-1]
        info.update(sample_rate=float(sr), channels=float(channels), frames=float(frames),
                    duration_s=frames / sr if sr else np.nan, decode_backend="librosa")
        if frames == 0:
            info["decode_status"] = "EMPTY"
        return info
    except Exception as exc:
        info["decode_status"] = "CORRUPT"
        info["decode_error"] = f"{first_error} | {type(exc).__name__}: {exc}"
        return info


def audio_inventory(tar_path: Path, members: pd.DataFrame, decode: bool = True, hash_bytes: bool = True,
                    allowed_members: set | None = None, logger=None) -> pd.DataFrame:
    """Read each safe audio member once (sequentially) to collect metadata and hashes."""
    from .logging_utils import progress

    wanted = members[members["safe"] & members["is_audio_ext"]]
    if allowed_members is not None:
        wanted = wanted[wanted["member"].isin(allowed_members)]
    wanted_set = set(wanted["member"])
    records = []
    with tarfile.open(tar_path, mode="r:*") as tf:
        iterator = progress(tf, total=len(members), desc=f"inventory {tar_path.name}")
        for m in iterator:
            name = m.name.replace("\\", "/")
            if name not in wanted_set:
                continue
            fh = tf.extractfile(m)
            data = fh.read() if fh is not None else b""
            rec = {"member": name}
            if hash_bytes:
                rec["sha1"] = hashlib.sha1(data).hexdigest()
            if decode:
                rec.update(_audio_info_from_bytes(data))
            records.append(rec)
    inv = pd.DataFrame(records)
    if inv.empty:
        return members.assign(sha1=np.nan, decode_status="NOT_INSPECTED")
    out = members.merge(inv, on="member", how="left")
    if "decode_status" not in out.columns:
        out["decode_status"] = np.where(out["member"].isin(wanted_set), "HEADER_NOT_DECODED", None)
    out.loc[out["decode_status"].isna(), "decode_status"] = np.where(
        out.loc[out["decode_status"].isna(), "safe"], "NOT_INSPECTED", "UNSAFE_MEMBER")
    return out


def match_clips_to_members(clips: pd.DataFrame, members: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict]:
    """Map clip rows to archive members via the resolved audio file or clip id.

    Matching keys tried (in order): exact member path, basename, stem. A key that
    maps to more than one member is reported as ambiguous and left unmatched.
    """
    report: dict = {"split": split, "n_clips": int(len(clips))}
    m = members[members["safe"] & members["is_audio_ext"]].copy()
    clips = clips.copy()
    clips["audio_member"] = np.nan
    clips["audio_match_key"] = np.nan
    lookups = {
        "member": m.groupby("member")["member"].agg(list).to_dict(),
        "basename": m.groupby("basename")["member"].agg(list).to_dict(),
        "stem": m.groupby("stem")["member"].agg(list).to_dict(),
    }
    candidates = []
    if "audio_file" in clips.columns:
        af = clips["audio_file"].astype(str).str.replace("\\", "/", regex=False)
        candidates += [("audio_file", "member", af), ("audio_file", "basename", af.map(lambda s: PurePosixPath(s).name)),
                       ("audio_file", "stem", af.map(lambda s: PurePosixPath(s).stem))]
    candidates += [("clip_id", "stem", clips["clip_id"].astype(str)), ("clip_id", "basename", clips["clip_id"].astype(str))]
    ambiguous = 0
    for source, kind, keys in candidates:
        todo = clips["audio_member"].isna()
        if not todo.any():
            break
        found = keys[todo].map(lambda k: lookups[kind].get(k))
        single = found.map(lambda v: isinstance(v, list) and len(v) == 1)
        multi = found.map(lambda v: isinstance(v, list) and len(v) > 1)
        ambiguous += int(multi.sum())
        idx = found[single].index
        clips.loc[idx, "audio_member"] = found[single].map(lambda v: v[0])
        clips.loc[idx, "audio_match_key"] = f"{source}->{kind}"
    matched = clips["audio_member"].notna()
    dup_members = clips.loc[matched, "audio_member"].duplicated(keep=False)
    report.update({
        "n_matched": int(matched.sum()),
        "n_missing_audio": int((~matched).sum()),
        "n_ambiguous_keys": ambiguous,
        "n_members_shared_by_several_clips": int(dup_members.sum()),
        "match_methods": clips["audio_match_key"].value_counts(dropna=False).to_dict(),
    })
    used = set(clips.loc[matched, "audio_member"])
    orphans = sorted(set(m["member"]) - used)
    report["n_orphaned_recordings"] = len(orphans)
    report["orphan_examples"] = orphans[:20]
    return clips, report


def required_extraction_bytes(members: pd.DataFrame, factor: float) -> int:
    return int(members["size_bytes"].sum() * factor)


def safe_extract(tar_path: Path, dest: Path, members: list, free_bytes: int, factor: float = 1.15,
                 min_free_gb: float = 2.0, logger=None) -> dict:
    """Extract only the listed members after safety and disk-space checks.

    Resumable: a member already present with the expected size is skipped. A
    completion marker records the extracted set.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extraction_complete.json"
    wanted = set(members)
    done = read_json(marker, default={}) or {}
    if done.get("archive") == tar_path.name and set(done.get("members", [])) >= wanted:
        return {"status": "ALREADY_EXTRACTED", "n": len(wanted), "dest": str(dest)}
    with tarfile.open(tar_path, mode="r:*") as tf:
        infos = [m for m in tf.getmembers() if m.name.replace("\\", "/") in wanted]
        unsafe = [(m.name, member_safety(m)[1]) for m in infos if not member_safety(m)[0]]
        if unsafe:
            raise UnsafeArchiveError(f"unsafe members requested for extraction: {unsafe[:5]}")
        need = int(sum(m.size for m in infos) * factor) + int(min_free_gb * 1e9)
        if need > free_bytes:
            raise OSError(f"insufficient disk space: need {need / 1e9:.2f} GB, have {free_bytes / 1e9:.2f} GB")
        from .logging_utils import progress

        written = skipped = 0
        for m in progress(sorted(infos, key=lambda i: getattr(i, "offset_data", 0)), total=len(infos),
                          desc=f"extract {tar_path.name}"):
            rel = PurePosixPath(m.name.replace("\\", "/"))
            target = (dest / Path(*rel.parts)).resolve()
            if dest not in target.parents:
                raise UnsafeArchiveError(f"member escapes destination: {m.name}")
            if target.exists() and target.stat().st_size == m.size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                raise UnsafeArchiveError(f"cannot read member {m.name}")
            tmp = target.with_name(target.name + ".part")
            with open(tmp, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(tmp, target)
            written += 1
    atomic_write_json(marker, {"archive": tar_path.name, "members": sorted(wanted | set(done.get("members", [])))})
    return {"status": "EXTRACTED", "written": written, "skipped_existing": skipped, "dest": str(dest)}


def read_member_bytes(tar_path: Path, members: list) -> dict:
    """Read selected members (ordered by archive offset) into memory."""
    wanted = set(members)
    out = {}
    with tarfile.open(tar_path, mode="r:*") as tf:
        infos = [m for m in tf.getmembers() if m.name.replace("\\", "/") in wanted]
        for m in sorted(infos, key=lambda i: getattr(i, "offset_data", 0)):
            if not member_safety(m)[0]:
                continue
            fh = tf.extractfile(m)
            if fh is not None:
                out[m.name.replace("\\", "/")] = fh.read()
    return out
