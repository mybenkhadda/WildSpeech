import numpy as np
import pandas as pd

from wildspeech.config import Config, deep_merge, load_config, parse_set_args
from wildspeech.data.integrity import integrity_checks
from wildspeech.data.schema import feature_safe_student_columns, normalize_text


def test_profiles_merge(project):
    cfg, stages = load_config("SMOKE", project / "configs")
    assert cfg.get_path("data.max_students") == 40
    assert "P00" in stages
    cfg2, _ = load_config("RESEARCH", project / "configs")
    assert cfg2.get_path("phonetic.enabled") is True
    base, _ = load_config("FAST", project / "configs")
    assert base.get_path("phonetic.enabled") is False and base.get_path("submission.enabled") is False


def test_overrides_and_fingerprint(project):
    o = parse_set_args(["gpu.batch_size=32", "audio.enable_pitch=true"])
    assert o == {"gpu": {"batch_size": 32}, "audio": {"enable_pitch": True}}
    a, _ = load_config("FAST", project / "configs")
    b, _ = load_config("FAST", project / "configs", overrides={"audio": {"n_fft": 1024}})
    assert a.fingerprint(["audio"]) != b.fingerprint(["audio"])
    assert a.fingerprint(["models"]) == b.fingerprint(["models"])
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert Config({"x": {"y": 1}}).get_path("x.z", 5) == 5


def test_bundle_schema(ctx):
    b = ctx.bundle()
    assert len(b.train_students) == 36 and len(b.test_students) == 12
    assert "school_id" not in feature_safe_student_columns(b)
    assert b.theta_cols == ["blending_theta", "deletion_theta", "sentence_repetition_theta"]
    assert b.train_clips["item_id"].str.contains("::").all()
    assert set(b.train_clips["is_scored"].unique()) <= {0, 1}
    assert normalize_text("Put the chicken, in!") == "put the chicken in"


def test_integrity(ctx):
    from wildspeech.data.tar_manifest import build_tar_manifest

    b = ctx.bundle()
    man = {s: build_tar_manifest(ctx.paths.files[f"{s}_audio"], ctx.paths.cache / "manifests") for s in ("train", "test")}
    chk = integrity_checks(b, man)
    assert (chk["status"] != "FAIL").all(), chk[chk.status == "FAIL"]
    assert chk.set_index("check").loc["train/test students disjoint", "status"] == "PASS"


def test_subsample_is_deterministic(project):
    from wildspeech.context import build_context

    o = {"runtime": {"device": "cpu"}, "data": {"max_students": 10, "max_test_students": 5}}
    a = build_context("FAST", overrides=o, root=project).bundle()
    b = build_context("FAST", overrides=o, root=project).bundle()
    assert list(a.train_students.subject_id) == list(b.train_students.subject_id)
    assert set(a.train_clips.subject_id) <= set(a.train_students.subject_id)
    assert any("subsampled" in n for n in a.notes)
