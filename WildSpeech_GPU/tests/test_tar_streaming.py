import tarfile

import numpy as np

from wildspeech.data.audio_io import PrefetchingBatchLoader, TarRandomAccessReader, decode_audio_bytes
from wildspeech.data.tar_manifest import build_tar_manifest, detect_compression, safe_member_name


def test_manifest_offsets_match_tarfile(ctx):
    tar = ctx.paths.files["train_audio"]
    man = build_tar_manifest(tar, ctx.paths.cache / "manifests")
    assert detect_compression(tar) is None
    reader = TarRandomAccessReader(tar, man)
    with tarfile.open(tar) as tf:
        for m in list(tf)[:5]:
            base = m.name.split("/")[-1]
            assert reader.read(base) == tf.extractfile(m).read()
    reader.close()


def test_manifest_cached(ctx):
    tar = ctx.paths.files["train_audio"]
    a = build_tar_manifest(tar, ctx.paths.cache / "manifests")
    b = build_tar_manifest(tar, ctx.paths.cache / "manifests")
    assert a.equals(b)


def test_safe_member_names():
    assert safe_member_name("train_audio/a.flac")
    assert not safe_member_name("../evil.flac")
    assert not safe_member_name("/abs/a.flac")
    assert not safe_member_name("C:/x.flac")


def test_decode_and_crop(ctx):
    tar = ctx.paths.files["train_audio"]
    man = build_tar_manifest(tar, ctx.paths.cache / "manifests")
    r = TarRandomAccessReader(tar, man)
    y, meta = decode_audio_bytes(r.read(man.filename.iloc[0]), 16000, max_seconds=0.5)
    assert y.dtype == np.float32 and len(y) <= 8000 and meta["file_sr"] == 16000
    y8, _ = decode_audio_bytes(r.read(man.filename.iloc[0]), 8000)
    assert abs(len(y8) - meta["true_duration_s"] * 8000) < 50
    r.close()


def test_prefetching_loader_bounded_and_complete(ctx):
    tar = ctx.paths.files["train_audio"]
    man = build_tar_manifest(tar, ctx.paths.cache / "manifests")
    names = list(man.filename[:23])
    batches = [names[i:i + 5] for i in range(0, len(names), 5)] + [["missing.flac"]]
    r = TarRandomAccessReader(tar, man)
    seen, fails = [], []
    for b in PrefetchingBatchLoader(r, batches, 16000, None, n_threads=3, prefetch_batches=1):
        seen += b.filenames
        fails += b.failures
        assert len(b.filenames) + len(b.failures) <= 5
    r.close()
    assert seen == names
    assert len(fails) == 1 and fails[0][0] == "missing.flac"
