"""Checkpoint resolution / revision / download commands (never executed) and real-ASR-output detection (P25-P29 gate)."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE
from wildspeech.asr.model_registry import parse_model
from wildspeech.asr.snapshots import (MANIFEST, download_commands, hf_cache_snapshot, missing_files, read_local_revision,
                                      resolve_snapshot)
from wildspeech.config import Config

REPO = Path(__file__).resolve().parents[1]
SHA = "1" * 40
SHA2 = "2" * 40


def _spec(tmp_path, **over):
    m = {"family": "whisper", "role": "general_word", "adapter": "whisper", "output": "words",
         "hf_id": "openai/whisper-small.en", "hf_id_status": "verified_public", "expected_architectures": ["hf_whisper"]}
    m.update(over)
    return parse_model("whisper_small_en", m, tmp_path, "models/asr")


def _whisper_dir(d: Path, rev=SHA):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    for f in ("preprocessor_config.json", "vocab.json", "merges.txt"):
        (d / f).write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"x")
    if rev:
        (d / MANIFEST).write_text(json.dumps({"revision": rev}), encoding="utf-8")
    return d


# ------------------------------------------------------------------------------------------- snapshots
def test_resolution_order_local_path_then_default_dir_then_hf_cache(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    snap = hub / "models--openai--whisper-small.en" / "snapshots" / SHA2
    _whisper_dir(snap, None)
    (hub / "models--openai--whisper-small.en" / "refs").mkdir(parents=True)
    (hub / "models--openai--whisper-small.en" / "refs" / "main").write_text(SHA2, encoding="utf-8")
    s = _spec(tmp_path)
    r = resolve_snapshot(s, str(hub))
    assert r.origin == "hf_cache" and r.revision == SHA2
    _whisper_dir(Path(s.default_local_dir))
    r = resolve_snapshot(s, str(hub))
    assert r.origin == "default_dir" and r.revision == SHA and r.revision_source == "manifest"
    lp = _whisper_dir(tmp_path / "custom", SHA2)
    r = resolve_snapshot(_spec(tmp_path, local_path=str(lp)), str(hub))
    assert r.origin == "local_path" and r.revision == SHA2


def test_missing_local_path_does_not_fall_back(tmp_path):
    _whisper_dir(Path(_spec(tmp_path).default_local_dir))
    r = resolve_snapshot(_spec(tmp_path, local_path=str(tmp_path / "nope")), str(tmp_path / "hub"))
    assert r.path is None and "local_path" in r.reason


def test_hf_cache_pinned_revision(tmp_path):
    hub = tmp_path / "hub"
    _whisper_dir(hub / "models--openai--whisper-small.en" / "snapshots" / SHA, None)
    assert hf_cache_snapshot("openai/whisper-small.en", SHA, hub)[1] == SHA
    assert hf_cache_snapshot("openai/whisper-small.en", SHA2, hub) == (None, None)


def test_revision_from_hf_download_metadata(tmp_path):
    d = _whisper_dir(tmp_path / "w", None)
    md = d / ".cache" / "huggingface" / "download"
    md.mkdir(parents=True)
    (md / "config.json.metadata").write_text(f"{SHA2}\netag\n1700000000\n", encoding="utf-8")
    assert read_local_revision(d) == (SHA2, "hf_download_metadata")


def test_missing_files_alternatives(tmp_path):
    d = _whisper_dir(tmp_path / "w")
    assert missing_files(d, "hf_whisper") == ([], ["generation_config.json", "tokenizer_config.json"])
    (d / "merges.txt").unlink()
    miss, _ = missing_files(d, "hf_whisper")
    assert any(m.startswith("tokenizer") for m in miss)
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert missing_files(d, "hf_whisper")[0] == []
    (d / "model.safetensors").unlink()
    assert any("weights" in m for m in missing_files(d, "hf_whisper")[0])


def test_download_commands_are_strings_only(tmp_path):
    c = download_commands(_spec(tmp_path, revision=SHA))
    assert c["revision"] == SHA and "--execute" in c["project_script"]
    assert c["huggingface_cli"].startswith("huggingface-cli download openai/whisper-small.en --revision " + SHA)
    u = download_commands(_spec(tmp_path, hf_id_status="unverified"))
    assert "NOT verified" in u["note"]


# --------------------------------------------------------------------------------------------- fake ctx
class FakeCtx:
    """Minimal RunContext stand-in: paths, cfg, _cache, save_table, bundle."""

    def __init__(self, root: Path, overrides: dict | None = None, n_clips: int = 6):
        shutil.copytree(REPO / "configs", root / "configs", dirs_exist_ok=True)
        self.paths = SimpleNamespace(root=root, configs=root / "configs", v3_cache=root / "cache" / "v3")
        self.paths.v3_cache.mkdir(parents=True, exist_ok=True)
        cfg = {"runtime": {"allow_model_download": False}, "paths": {},
               "v3": {"asr": {"splits": ["train"], "models": ["whisper_small_en"]}}}
        for k, v in (overrides or {}).items():
            cfg.setdefault(k, {}).update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)
        self.cfg = Config(cfg)
        self._cache = {}
        self.saved = {}
        clips = pd.DataFrame({FILE: [f"c{i}.flac" for i in range(n_clips)], "task": "blending",
                              "expected_text": "cat", "item_score": 1.0, "duration_sec": 1.0})
        self._bundle = SimpleNamespace(train_clips=clips, test_clips=clips.iloc[:0])

    def bundle(self):
        return self._bundle

    def save_table(self, df, stage, name, parquet=False):
        self.saved[(stage, name)] = df


def _write_asr(ctx, key, prov, n=3, transcript="cat", phones=""):
    from wildspeech.asr.base import ASRResult
    from wildspeech.asr.checkpointing import ASRCheckpointStore, asr_fingerprint

    st = ASRCheckpointStore(ctx.paths.v3_cache, key)
    st.prepare("train", asr_fingerprint(key, "src", None, {}), prov)
    rows = []
    for i in range(n):
        row = ASRResult(transcript=transcript, phones=phones, token_logprobs=[-0.1]).row()
        row[FILE] = f"c{i}.flac"
        rows.append(row)
    st.append("train", rows, f"{key}@test")
    return st


P24_OK = {"producer": "P24", "eligibility": {"eligible": True, "eligibility": "ELIGIBLE"}}


def test_fixture_outputs_rejected_unless_test_flag(tmp_path):
    from wildspeech.asr.outputs import accepted_models, scan_outputs

    ctx = FakeCtx(tmp_path)
    _write_asr(ctx, "fake", {"fixture": True})
    t = scan_outputs(ctx)
    assert t.loc[0, "origin"] == "fixture" and not t.loc[0, "accepted"]
    assert accepted_models(ctx) == []
    ctx2 = FakeCtx(tmp_path, {"v3": {"asr": {"splits": ["train"], "accept_fixture_outputs": True}}})
    assert accepted_models(ctx2) == ["fake"]


def test_unverified_provenance_is_rejected(tmp_path):
    from wildspeech.asr.outputs import scan_outputs

    ctx = FakeCtx(tmp_path)
    _write_asr(ctx, "whisper_small_en", {"producer": "P24", "eligibility": {"eligible": False}})
    t = scan_outputs(ctx)
    assert t.loc[0, "origin"] == "unverified" and not t.loc[0, "accepted"]


def test_empty_transcripts_are_not_real_outputs(tmp_path):
    from wildspeech.asr.outputs import scan_outputs

    ctx = FakeCtx(tmp_path)
    _write_asr(ctx, "whisper_small_en", P24_OK, transcript="", phones="")
    t = scan_outputs(ctx)
    assert t.loc[0, "rows"] == 3 and not t.loc[0, "accepted"] and "non-empty" in t.loc[0, "reason"]


def test_gate_revalidates_current_eligibility(tmp_path):
    """Real P24 outputs of a model whose legal/provenance reviews are missing NOW are not usable downstream."""
    from wildspeech.asr.outputs import require_outputs, usable_pairs
    from wildspeech.asr.reviews import ReviewEntry, record_review
    from wildspeech.asr.model_registry import load_registry_document

    ctx = FakeCtx(tmp_path)
    _write_asr(ctx, "whisper_small_en", P24_OK)
    ok, acc, why = require_outputs(ctx, "P25")
    assert not ok and acc == [] and "no longer eligible" in why
    gate = ctx.saved[("P25", "asr_output_gate")]
    assert gate.loc[0, "accepted"] and not gate.loc[0, "used"]
    doc = load_registry_document(ctx.paths.configs)
    e = ReviewEntry("whisper_small_en", ["license", "provenance", "overlap"], "approved", "A. Reviewer", "2026-11-01",
                    "f" * 40, "apache-2.0",
                    [{"url": "https://huggingface.co/openai/whisper-small.en", "summary": "card read, apache-2.0"}],
                    ["WSP-OVERLAP"])
    assert record_review(ctx.paths.configs / "v3_asr_reviews.yaml", e, doc.specs["whisper_small_en"], doc.allowed_licenses) == []
    ok2, acc2, _ = require_outputs(ctx, "P25")
    assert ok2 and acc2 == ["whisper_small_en"]
    assert ("whisper_small_en", "train") in usable_pairs(ctx, ("train",))


def test_require_outputs_min_models_and_need(tmp_path):
    from wildspeech.asr.outputs import require_outputs

    ctx = FakeCtx(tmp_path, {"v3": {"asr": {"splits": ["train"], "accept_fixture_outputs": True}}})
    ok, _, why = require_outputs(ctx, "P25")
    assert not ok and "no ASR cache found" in why
    _write_asr(ctx, "fx_word", {"fixture": True})
    assert require_outputs(ctx, "P27", "any", 2)[0] is False
    assert require_outputs(ctx, "P25", "phones", 1)[0] is False
    _write_asr(ctx, "fx_phon", {"fixture": True}, transcript="", phones="K AE T")
    assert require_outputs(ctx, "P27", "any", 2)[0] is True
    assert require_outputs(ctx, "P25", "phones", 1)[1] == ["fx_phon"]


def test_execution_state(tmp_path):
    from wildspeech.asr.outputs import execution_state

    ctx = FakeCtx(tmp_path, {"v3": {"asr": {"splits": ["train"], "accept_fixture_outputs": True}}}, n_clips=6)
    _write_asr(ctx, "fx_a", {"fixture": True}, n=3)
    _write_asr(ctx, "fx_b", {"fixture": True}, n=6)
    s = execution_state(ctx, ["fx_a", "fx_b", "never"]).set_index("model_key")
    assert s.loc["fx_a", "state"] == "partial" and s.loc["fx_a", "coverage"] == pytest.approx(0.5)
    assert s.loc["fx_b", "state"] == "complete"
    assert s.loc["never", "state"] == "never_run"


def test_output_state_token_tracks_accepted_outputs_only(tmp_path):
    from wildspeech.asr.outputs import output_state_token

    ctx = FakeCtx(tmp_path, {"v3": {"asr": {"splits": ["train"], "accept_fixture_outputs": True}}})
    t0 = output_state_token(ctx)
    _write_asr(ctx, "unverified", {"producer": "P24", "eligibility": {"eligible": False}})
    assert output_state_token(ctx) == t0                       # rejected caches never change the token
    st = _write_asr(ctx, "fx", {"fixture": True}, n=2)
    t1 = output_state_token(ctx)
    assert t1 != t0
    from wildspeech.asr.base import ASRResult

    row = ASRResult(transcript="dog").row()
    row[FILE] = "c5.flac"
    st.append("train", [row], "fx@test")
    assert output_state_token(ctx) != t1


def test_p24_state_token_changes_with_reviews_and_snapshots(tmp_path):
    from wildspeech.asr.snapshots import p24_state_token

    ctx = FakeCtx(tmp_path, {"paths": {"hf_cache_dir": str(tmp_path / "hub")}})   # isolate from the user's HF cache
    t0 = p24_state_token(ctx)
    p = ctx.paths.configs / "v3_asr_reviews.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    t1 = p24_state_token(ctx)
    assert t1 != t0
    _whisper_dir(tmp_path / "models" / "asr" / "whisper_small_en")
    assert p24_state_token(ctx) != t1


def test_load_phonetic_tables_skips_unusable_models(tmp_path):
    from wildspeech.io_utils import atomic_write_parquet
    from wildspeech.phonetics.builder import load_phonetic_tables

    ctx = FakeCtx(tmp_path, {"v3": {"asr": {"splits": ["train"], "accept_fixture_outputs": True}}})
    _write_asr(ctx, "fx", {"fixture": True})
    for k in ("fx", "stale_model"):
        d = ctx.paths.v3_cache / "phonetic" / k / "train"
        d.mkdir(parents=True)
        atomic_write_parquet(pd.DataFrame({FILE: ["c0.flac"], f"M_{k.upper()}::ph_per": np.float32([0.1])}), d / "features.parquet")
        atomic_write_parquet(pd.DataFrame({FILE: ["c0.flac"], "model_key": [k]}), d / "details.parquet")
    wide, det = load_phonetic_tables(ctx, ("train",))
    assert [c for c in wide.columns if "::" in c] == ["M_FX::ph_per"]
    wide_all, _ = load_phonetic_tables(ctx, ("train",), only_real=False)
    assert len([c for c in wide_all.columns if "::" in c]) == 2
