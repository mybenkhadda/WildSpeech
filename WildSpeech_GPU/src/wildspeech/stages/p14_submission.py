"""P14 — independent test inference: extract test features needed by the champion, refit on all training
students, reload the bundle from disk, predict, validate against submission_format.csv."""
from __future__ import annotations

from ..acoustic.extractor import FAMILY_PARAM_KEYS, AcousticExtractionRunner
from ..io_utils import read_json
from ..submission import make_submission

S = "P14"


def run(ctx) -> dict:
    spec = read_json(ctx.paths.models / "champion_spec.json")
    if not spec:
        raise FileNotFoundError("champion spec missing — run P13")
    fams = [f for f in spec.get("audio_families", []) if f in FAMILY_PARAM_KEYS or f == "wavelet"]
    if fams:
        AcousticExtractionRunner(ctx).run("test", fams)
    if any(f in ("asr", "align") for f in spec.get("audio_families", [])):
        from ..phonetic.runner import PhoneticRunner

        PhoneticRunner(ctx).run("test")
    if any(f.startswith("emb_") for f in spec.get("audio_families", [])):
        from ..embeddings import EmbeddingRunner

        EmbeddingRunner(ctx).run("test", [f[4:] for f in spec["audio_families"] if f.startswith("emb_")])
    ctx._cache.clear()                     # reload clip features including the freshly extracted test split
    meta = make_submission(ctx, S)
    ctx.save_json(meta, S, "submission_meta")
    return meta
