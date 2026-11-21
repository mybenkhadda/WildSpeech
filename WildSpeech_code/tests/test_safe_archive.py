import io
import tarfile

import pytest

from wildspeech.audio_manifest import UnsafeArchiveError, inspect_archive, safe_extract


def _add(tf, name, data=b"RIFF0000WAVE"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "a.tar"
    with tarfile.open(path, "w") as tf:
        _add(tf, "audio/good.wav")
        _add(tf, "../evil.wav")
        _add(tf, "/abs/evil.wav")
        link = tarfile.TarInfo("audio/link.wav")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
        hard = tarfile.TarInfo("audio/hard.wav")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "audio/good.wav"
        tf.addfile(hard)
        _add(tf, "notes.txt", b"x")
    return path


def test_inspection_flags_unsafe_members(archive):
    m = inspect_archive(archive).set_index("member")
    assert m.loc["audio/good.wav", "safe"]
    assert m.loc["../evil.wav", "safety_reason"] == "path_traversal"
    assert m.loc["/abs/evil.wav", "safety_reason"] == "absolute_path"
    assert m.loc["audio/link.wav", "safety_reason"] == "symlink"
    assert m.loc["audio/hard.wav", "safety_reason"] == "hardlink"
    assert not m.loc["notes.txt", "is_audio_ext"]


def test_safe_extract_only_writes_safe_members(archive, tmp_path):
    dest = tmp_path / "out"
    rep = safe_extract(archive, dest, ["audio/good.wav"], free_bytes=10 ** 12, min_free_gb=0)
    assert rep["status"] == "EXTRACTED" and (dest / "audio" / "good.wav").exists()
    assert not (tmp_path / "evil.wav").exists()
    again = safe_extract(archive, dest, ["audio/good.wav"], free_bytes=10 ** 12, min_free_gb=0)
    assert again["status"] == "ALREADY_EXTRACTED"


def test_safe_extract_rejects_unsafe_requests(archive, tmp_path):
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "o2", ["../evil.wav"], free_bytes=10 ** 12, min_free_gb=0)
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "o3", ["audio/link.wav"], free_bytes=10 ** 12, min_free_gb=0)


def test_disk_space_is_checked_before_extraction(archive, tmp_path):
    with pytest.raises(OSError):
        safe_extract(archive, tmp_path / "o4", ["audio/good.wav"], free_bytes=1, min_free_gb=0)
    assert not (tmp_path / "o4" / "audio" / "good.wav").exists()
