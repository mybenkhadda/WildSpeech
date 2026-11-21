"""Synthetic project fixture: configs + CSVs mimicking the competition schema + a small FLAC TAR."""
from __future__ import annotations

import io
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

TASKS = ["blending", "deletion", "sentence_repetition"]
WORDS = {"blending": ["Pot", "Boy", "Sweet", "Zoo", "Rabbit", "Baby"], "deletion": ["Sand", "Sun", "Tie", "Cat"],
         "sentence_repetition": ["Put the chicken in the freezer before you leave.", "She does not like pancakes.",
                                 "The dog by the tree in the park has a new toy."]}


def tone(sr: int, seconds: float, f0: float, rng) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    env = np.clip(np.sin(np.pi * t / max(seconds, 1e-3)), 0, None)
    y = 0.2 * env * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 4))
    pad = np.zeros(int(0.2 * sr))
    return np.concatenate([pad, y, pad]).astype(np.float32) + 0.002 * rng.standard_normal(len(y) + 2 * len(pad)).astype(np.float32)


def make_project(root: Path, n_train: int = 36, n_test: int = 12, clips_per_task: int = 3, seed: int = 0) -> Path:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    shutil.copytree(REPO / "configs", root / "configs")
    data = root / "data"
    data.mkdir(parents=True)
    rows = {"train": [], "test": []}
    studs = {"train": [], "test": []}
    labels = []
    audio = {"train": {}, "test": {}}
    for split, n in (("train", n_train), ("test", n_test)):
        for s in range(n):
            sid = f"{split[:2]}{s:010x}"
            ability = rng.normal()
            inst = "wj4" if s % 2 else "ktea3"
            st = {"subject_id": sid, "instrument": inst, "original_sample_rate": 16000 if s % 3 else 11025,
                  "blending_theta": round(ability + rng.normal(0, .5), 2),
                  "deletion_theta": np.nan if inst == "ktea3" and s % 4 == 0 else round(ability + rng.normal(0, .5), 2),
                  "sentence_repetition_theta": round(ability + rng.normal(0, .5), 2)}
            if split == "train":
                st["school_id"] = s % 4
                labels.append({"subject_id": sid, "outcome_score": int(100 + 10 * ability + rng.normal(0, 5))})
            studs[split].append(st)
            for t in TASKS:
                for k in range(clips_per_task):
                    fname = f"{split}{s:04d}{t[:3]}{k}.flac"
                    text = WORDS[t][(s + k) % len(WORDS[t])]
                    correct = float(rng.random() < 1 / (1 + np.exp(-(ability - 0.2 * k))))
                    dur = float(rng.uniform(0.6, 2.5))
                    y = tone(16000, dur, 180 + 40 * correct + 20 * rng.random(), rng)
                    buf = io.BytesIO()
                    sf.write(buf, y, 16000, format="FLAC")
                    audio[split][fname] = buf.getvalue()
                    r = {"filename": fname, "subject_id": sid, "task": t, "expected_text": text,
                         "item_score": correct if k < clips_per_task - 1 or s % 5 else np.nan, "duration_sec": round(len(y) / 16000, 2)}
                    if split == "train":
                        r["score_status"] = "scored" if not np.isnan(r["item_score"]) else "no_score_value"
                    rows[split].append(r)
    cols_tr = ["filename", "subject_id", "task", "expected_text", "score_status", "item_score", "duration_sec"]
    pd.DataFrame(rows["train"])[cols_tr].to_csv(data / "train_clip_metadata.csv", index=False)
    pd.DataFrame(rows["test"]).to_csv(data / "test_clip_metadata.csv", index=False)
    pd.DataFrame(studs["train"]).to_csv(data / "train_student_metadata.csv", index=False)
    pd.DataFrame(studs["test"]).to_csv(data / "test_student_metadata.csv", index=False)
    pd.DataFrame(labels).to_csv(data / "train_labels.csv", index=False)
    pd.DataFrame({"subject_id": [s["subject_id"] for s in studs["test"]], "outcome_score": 0.0}).to_csv(data / "submission_format.csv", index=False)
    for split in ("train", "test"):
        with tarfile.open(data / f"{split}_audio.tar", "w") as tf:
            for fname, payload in audio[split].items():
                info = tarfile.TarInfo(f"{split}_audio/{fname}")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
    return root


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> Path:
    return make_project(tmp_path_factory.mktemp("wsproj"))


@pytest.fixture()
def ctx(project):
    from wildspeech.context import build_context

    overrides = {"runtime": {"device": "cpu"}, "data": {"max_students": None, "max_test_students": None},
                 "validation": {"n_splits": 3, "n_repeats": 1, "bootstrap": 50}, "analysis": {"bootstrap": 50}}
    c = build_context("FAST", overrides=overrides, root=project)
    return c
