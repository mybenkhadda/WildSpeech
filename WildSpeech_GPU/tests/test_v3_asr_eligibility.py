"""V3 speech-model registry (schema 3) + eligibility in 13 independent dimensions + lifecycle. No override switch."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from wildspeech.asr import environment as E
from wildspeech.asr.eligibility import (CHECKS, DIMENSIONS, GATING, GROUPS, LIFECYCLE, EligibilityChecker,
                                        EligibilityRecord, detect_architecture)
from wildspeech.asr.model_registry import CompetitionPolicy, LicensePolicy, load_registry_document, parse_model
from wildspeech.asr.reviews import ReviewEntry
from wildspeech.config import Config

REPO = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OTHER = "b" * 40
TODAY = dt.date(2026, 11, 21)
ENV_GPU = {"cuda_available": True, "gpu_total_gb": 8.0, "gpu_name": "RTX 3070 (test)"}
LP = LicensePolicy(permissive=["apache-2.0", "mit"],
                   conditional={"bigcode-openrail-m": {"conditions": [{"id": "C1", "text": "use restrictions"},
                                                                      {"id": "C2", "text": "not OSI"}]}},
                   prohibited=["cc-by-nc-4.0"])


def _cfg(**asr):
    return Config({"runtime": {}, "paths": {}, "v3": {"asr": asr}})


def _policy(named=("wav2vec2", "whisper", "pasketti"), general=("hubert", "wavlm"), url="https://example.org/rules"):
    return CompetitionPolicy(statement="eligible pretrained models are permitted", recorded_by="test", source_url=url,
                             permitted_families=list(named) + list(general), named_families=list(named),
                             general_rule_families=list(general))


def _snapshot(d: Path, revision: str | None = SHA, weights: bytes = b"weights-bytes") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]}), encoding="utf-8")
    (d / "vocab.json").write_text(json.dumps({"<pad>": 0, "|": 1, "A": 2}), encoding="utf-8")
    (d / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(weights)
    if revision:
        (d / "_wildspeech_checkpoint.json").write_text(json.dumps({"revision": revision}), encoding="utf-8")
    return d


def _files(d: Path, sha_override: str | None = None):
    out = []
    for n, role in (("config.json", "config"), ("preprocessor_config.json", "processor"), ("vocab.json", "tokenizer"),
                    ("tokenizer_config.json", "tokenizer"), ("model.safetensors", "weights")):
        f = {"name": n, "size": (d / n).stat().st_size, "role": role}
        if role == "weights":
            f["sha256"] = sha_override or hashlib.sha256((d / n).read_bytes()).hexdigest()
        out.append(f)
    return out


def _spec(tmp_path: Path, snapshot: bool = True, revision: str | None = SHA, license_id="apache-2.0",
          family="wav2vec2", restrictions=None, sha_override=None, **over):
    snap = _snapshot(tmp_path / "snap", revision)
    m = {"family": family, "role": "general_char", "adapter": "ctc", "output": "chars", "hf_id": "org/model",
         "hf_id_status": "verified_public", "revision": revision, "expected_architectures": ["hf_ctc"],
         "expected_model_type": "wav2vec2", "expected_class": "Wav2Vec2ForCTC", "expected_params_millions": 95,
         "files": _files(snap, sha_override),
         "evidence": {"model_card_url": "https://example.org/card", "license_id": license_id,
                      "license_url": "https://example.org/license", "training_sources": ["LibriSpeech 960 h"],
                      "training_evidence_url": "https://example.org/paper", "enumerated_corpora": True,
                      "release_date": "2021-01-01", "evidence_status": "assembled"},
         "restrictions": restrictions or []}
    if snapshot:
        m["local_path"] = str(snap)
    else:
        import shutil

        shutil.rmtree(snap)
    m.update(over)
    return parse_model("m", m, tmp_path, "models")


def _review(checks=("license", "provenance", "overlap"), decision="approved", revision=SHA, date="2026-11-01", **kw):
    base = dict(model_key="m", checks=list(checks), decision=decision, reviewer="Test Reviewer", date=date,
                revision_reviewed=revision, license_id="apache-2.0" if "license" in checks else None,
                evidence=[{"url": "https://example.org/card", "summary": "license and training data verified"}],
                overlap_basis="enumerated_public_corpora_disjoint" if "overlap" in checks else None)
    base.update(kw)
    return ReviewEntry(**base)


def _checker(reviews=(), policy=None, runtime=None, env=None, lp=LP, **asr):
    return EligibilityChecker(_cfg(**asr), lp.allowed(), policy or _policy(), list(reviews), today=TODAY,
                              license_policy=lp, runtime=runtime or {}, env=env if env is not None else ENV_GPU)


@pytest.fixture()
def comp_ok(monkeypatch):
    """The processor/tokenizer deep check needs real checkpoints; its own tests live elsewhere."""
    monkeypatch.setattr(E, "component_check", lambda spec, path, arch: {"ok": True, "problems": "", "notes": "test"})


@pytest.fixture()
def pkgs_ok(monkeypatch):
    monkeypatch.setattr(E, "package_version", lambda n: (True, "9.9", ""))


# --------------------------------------------------------------------------------------- repo registry
def test_repo_registry_schema3_is_evidence_backed():
    doc = load_registry_document(REPO / "configs")
    assert doc.schema_version == 3 and doc.checkpoint_root == "models"
    want = {"pasketti_phonetic", "pasketti_word", "whisper_small_en", "wav2vec2_phoneme_espeak", "wav2vec2_base_960h",
            "wav2vec2_base", "hubert_base", "wavlm_base_plus"}
    assert want == set(doc.specs)
    dirs = {"pasketti_phonetic": "pasketti_phonetic", "pasketti_word": "pasketti_word", "whisper_small_en": "whisper_small_en",
            "wav2vec2_phoneme_espeak": "wav2vec2_phoneme", "wav2vec2_base_960h": "wav2vec2_base_960h",
            "wav2vec2_base": "wav2vec2_base", "hubert_base": "hubert_base", "wavlm_base_plus": "wavlm_base_plus"}
    for k, s in doc.specs.items():
        assert not s.validate(), (k, s.validate())
        assert s.hf_id_status == "verified_public" and s.revision and len(s.revision) == 40, k
        assert s.evidence.model_card_url.startswith("https://") and s.evidence.training_sources, k
        assert s.weight_files() and all(w.sha256 and len(w.sha256) == 64 for w in s.weight_files()), k
        assert all(f.size for f in s.files), k
        assert not any(s.legacy_flags.values()), k
        assert s.default_local_dir.replace("\\", "/").endswith(f"models/{dirs[k]}"), k
    assert [doc.specs[k].priority for k in ("pasketti_phonetic", "whisper_small_en", "pasketti_word")] == [1, 2, 3]
    assert doc.specs["pasketti_phonetic"].expected_class == "WavLMForCTC" and doc.specs["pasketti_phonetic"].precision == "fp32"
    assert doc.specs["pasketti_word"].expected_architectures == ["qwen3_asr"]       # NOT a Whisper checkpoint
    assert doc.specs["pasketti_word"].runtime_backend == "qwen_asr_vllm"
    assert {doc.specs[k].output for k in ("wav2vec2_base", "hubert_base", "wavlm_base_plus")} == {"embeddings"}
    for k in ("wav2vec2_phoneme_espeak", "wav2vec2_base", "hubert_base", "wavlm_base_plus"):
        assert doc.specs[k].bin_only(), k                                            # needs verified conversion
    lp = doc.license_policy
    assert lp.classify("apache-2.0") == "permissive" and lp.classify("bigcode-openrail-m") == "conditional"
    assert lp.classify("cc-by-sa-3.0") == "conditional" and lp.classify("cc-by-nc-4.0") == "prohibited"
    assert {r.id for r in doc.specs["pasketti_phonetic"].blocking_restrictions("overlap")} == {"PSK-PH-READNET-OVERLAP"}
    assert {r.id for r in doc.specs["pasketti_word"].blocking_restrictions("overlap")} == {"PSK-WD-READNET-OVERLAP"}
    assert doc.policy.basis("pasketti") == "named" and doc.policy.basis("hubert") == "general_rule"
    assert doc.policy.source_url.startswith("https://platform.k12-ai-infrastructure.org/")


def test_repo_models_are_never_eligible_without_reviews():
    doc = load_registry_document(REPO / "configs")
    ch = EligibilityChecker(_cfg(), doc.allowed_licenses, doc.policy, [], today=TODAY,
                            license_policy=doc.license_policy, env={})
    recs = ch.check_all(doc.specs, probe_files=False)
    for k, r in recs.items():
        assert not r.eligible and not r.download_allowed, k
        assert r.lifecycle == "LEGAL_REVIEW_PENDING", (k, r.lifecycle)
        assert r.dimensions["license_approval"] == "PENDING" and r.dimensions["overlap_assessment"] == "PENDING", k
        assert r.dimensions["checkpoint_availability"] == "NOT_PROBED", k
        assert any("accept-proposal" in a for a in r.actions) or r.actions, k


def test_proposals_are_suggested_but_never_approve(tmp_path):
    from wildspeech.asr.reviews import load_proposals

    doc = load_registry_document(REPO / "configs")
    props = load_proposals(REPO / "configs" / "v3_asr_review_proposals.yaml")
    assert props and all(not p.entry.reviewer for p in props)
    ch = EligibilityChecker(_cfg(), doc.allowed_licenses, doc.policy, [], today=TODAY,
                            license_policy=doc.license_policy, env={}, proposals=props)
    r = ch.check(doc.specs["whisper_small_en"], probe_files=False)
    assert r.checks["license_review"] == "PENDING" and not r.download_allowed
    assert any("accept-proposal --id P-WSP-LP" in a for a in r.actions)
    pk = {p.id: p for p in props}
    assert pk["P-PSKPH-OV"].ready is False and pk["P-PSKPH-OV"].open_questions        # human decision required


# -------------------------------------------------------------------------------------------- dimensions
def test_thirteen_dimensions_and_rows(tmp_path):
    assert len(DIMENSIONS) == 13 and "inference_readiness" not in GATING
    r = _checker([_review()]).check(_spec(tmp_path), probe_files=True)
    assert set(r.dimensions) == set(DIMENSIONS) and set(r.groups) == set(GROUPS)
    assert set(CHECKS) <= set(r.checks)
    row = r.as_row()
    for d in DIMENSIONS:
        assert f"dim_{d}" in row
    assert {x["dimension"] for x in r.check_rows()} == set(DIMENSIONS)
    json.dumps(r.evidence_json())                                                # serialisable for model_evidence.json
    assert r.lifecycle in LIFECYCLE


def test_ready_for_inference_when_every_gating_dimension_passes(tmp_path, comp_ok, pkgs_ok):
    r = _checker([_review()]).check(_spec(tmp_path), probe_files=True, deep=True, verify_hashes=True)
    assert r.eligible and r.lifecycle == "READY_FOR_INFERENCE", r.reasons
    assert all(r.dimensions[d] == "PASS" for d in GATING)
    assert r.checks["integrity"] == "PASS" and r.checks["smoke_test"] == "NOT_PROBED"
    assert r.download_allowed and r.architecture == "hf_ctc" and r.snapshot_revision == SHA


def test_deep_check_is_required_for_inference(tmp_path, pkgs_ok):
    r = _checker([_review()]).check(_spec(tmp_path), probe_files=True, deep=False)
    assert r.dimensions["processor_compatibility"] == "NOT_PROBED" and not r.eligible


# -------------------------------------------------------------------------------------------- lifecycle
def test_legal_review_pending(tmp_path):
    r = _checker([]).check(_spec(tmp_path))
    assert r.lifecycle == "LEGAL_REVIEW_PENDING" and not r.download_allowed


def test_provenance_review_pending_after_license(tmp_path):
    r = _checker([_review(checks=("license",))]).check(_spec(tmp_path))
    assert r.dimensions["license_approval"] == "PASS" and r.lifecycle == "PROVENANCE_REVIEW_PENDING"


def test_overlap_without_positive_basis_is_ignored(tmp_path):
    rv = [_review(checks=("license", "provenance")), _review(checks=("overlap",), overlap_basis=None, date="2026-11-02")]
    r = _checker(rv).check(_spec(tmp_path))
    assert r.checks["overlap_review"] == "PENDING" and r.invalid_reviews
    assert "positive basis" in r.results["overlap_review"].detail or r.invalid_reviews
    rv2 = [_review(checks=("license", "provenance")),
           _review(checks=("overlap",), overlap_basis="no_known_overlap", date="2026-11-02")]
    r2 = _checker(rv2).check(_spec(tmp_path))
    assert r2.lifecycle == "PROVENANCE_REVIEW_PENDING"
    assert any("refused" in iv["problems"] for iv in r2.invalid_reviews)


def test_ready_for_download_when_reviews_pass_but_no_checkpoint(tmp_path):
    r = _checker([_review()]).check(_spec(tmp_path, snapshot=False), probe_files=True)
    assert r.lifecycle == "READY_FOR_DOWNLOAD" and r.download_allowed and not r.eligible
    assert r.checks["checkpoint_present"] == "FAIL" and "v3-install" in r.results["checkpoint_present"].action


def test_downloading_lock(tmp_path):
    r = _checker([_review()], runtime={"m": {"downloading": True}}).check(_spec(tmp_path, snapshot=False))
    assert r.lifecycle == "DOWNLOADING"


def test_checkpoint_incomplete_on_size_hash_or_revision(tmp_path, pkgs_ok):
    s = _spec(tmp_path)
    (Path(s.local_path) / "model.safetensors").write_bytes(b"truncated")
    r = _checker([_review()]).check(s)
    assert r.checks["manifest_files"] == "FAIL" and r.lifecycle == "CHECKPOINT_INCOMPLETE"
    s2 = _spec(tmp_path / "h", sha_override="0" * 64)
    r2 = _checker([_review()]).check(s2, verify_hashes=True)
    assert r2.checks["integrity"] == "FAIL" and r2.lifecycle == "CHECKPOINT_INCOMPLETE"
    s3 = _spec(tmp_path / "r")
    (Path(s3.local_path) / "_wildspeech_checkpoint.json").write_text(json.dumps({"revision": OTHER}), encoding="utf-8")
    r3 = _checker([_review()]).check(s3)
    assert r3.checks["revision_match"] == "FAIL" and r3.lifecycle == "CHECKPOINT_INCOMPLETE"


def test_framework_incompatible(tmp_path, pkgs_ok, comp_ok):
    r = _checker([_review()]).check(_spec(tmp_path, expected_architectures=["hf_whisper"]), deep=True)
    assert r.checks["architecture_expected"] == "FAIL" and r.lifecycle == "FRAMEWORK_INCOMPATIBLE"
    r2 = _checker([_review()], env={"cuda_available": False}).check(_spec(tmp_path / "x", expected_params_millions=2000), deep=True)
    assert r2.checks["gpu_fit"] == "FAIL" and r2.lifecycle == "FRAMEWORK_INCOMPATIBLE"


def test_remote_code_is_never_trusted(tmp_path, pkgs_ok, comp_ok):
    s = _spec(tmp_path)
    cfgp = Path(s.local_path) / "config.json"
    c = json.loads(cfgp.read_text(encoding="utf-8"))
    c["auto_map"] = {"AutoModel": "modeling_custom.CustomModel"}
    cfgp.write_text(json.dumps(c), encoding="utf-8")
    s = parse_model("m", {**{k: getattr(s, k) for k in ("family", "role", "adapter", "output", "hf_id", "hf_id_status",
                                                        "revision", "expected_architectures", "local_path")},
                          "files": [vars(f) for f in s.files if f.name != "config.json"] + [
                              {"name": "config.json", "size": cfgp.stat().st_size, "role": "config"}],
                          "evidence": vars(s.evidence) | {"links": {}}}, tmp_path, "models")
    r = _checker([_review()]).check(s, deep=True)
    assert r.checks["remote_code"] == "FAIL" and r.lifecycle == "FRAMEWORK_INCOMPATIBLE"


def test_runtime_states(tmp_path, comp_ok, pkgs_ok):
    s = _spec(tmp_path)
    assert _checker([_review()], runtime={"m": {"running": True}}).check(s, deep=True).lifecycle == "INFERENCE_RUNNING"
    assert _checker([_review()], runtime={"m": {"complete": True}}).check(s, deep=True).lifecycle == "INFERENCE_COMPLETE"
    assert _checker([_review()], runtime={"m": {"last_failed": True}}).check(s, deep=True).lifecycle == "FAILED"
    sm = {"ok": False, "revision": SHA, "error": "boom"}
    r = _checker([_review()], runtime={"m": {"smoke": sm}}).check(s, deep=True)
    assert r.checks["smoke_test"] == "FAIL" and r.lifecycle == "FAILED" and r.eligible   # eligible, but not ready
    ok = {"ok": True, "revision": SHA, "n_clips": 8, "clips_per_s": 3.0, "dtype": "fp16", "finished": "t"}
    assert _checker([_review()], runtime={"m": {"smoke": ok}}).check(s, deep=True).checks["smoke_test"] == "PASS"


def test_ineligible_on_rejection_prohibited_license_or_family(tmp_path):
    r = _checker([_review(), _review(checks=("provenance",), decision="rejected", date="2026-11-05")]).check(_spec(tmp_path))
    assert r.checks["provenance_review"] == "FAIL" and r.lifecycle == "INELIGIBLE"
    r2 = _checker([]).check(_spec(tmp_path / "nc", license_id="cc-by-nc-4.0"))
    assert r2.checks["license_documented"] == "FAIL" and r2.lifecycle == "INELIGIBLE"
    r3 = _checker([_review()], policy=_policy(named=("whisper",), general=())).check(_spec(tmp_path / "f"))
    assert r3.checks["competition_policy"] == "FAIL" and r3.lifecycle == "INELIGIBLE"


def test_conditional_license_needs_every_condition(tmp_path):
    s = _spec(tmp_path, license_id="bigcode-openrail-m", family="pasketti")
    part = _review(license_id="bigcode-openrail-m", acknowledged_conditions=["C1"])
    r = _checker([part]).check(s)
    assert r.checks["license_review"] == "PENDING" and r.invalid_reviews
    full = _review(license_id="bigcode-openrail-m", acknowledged_conditions=["C1", "C2"])
    r2 = _checker([full]).check(s)
    assert r2.checks["license_review"] == "PASS" and r2.checks["license_documented"] == "WARN"
    assert r2.dimensions["license_documentation"] == "PASS"


def test_general_rule_family_with_conditional_license_waits_for_license_review(tmp_path):
    s = _spec(tmp_path, license_id="bigcode-openrail-m", family="wavlm")
    r = _checker([]).check(s)
    assert r.checks["competition_policy"] == "PENDING"
    full = _review(license_id="bigcode-openrail-m", acknowledged_conditions=["C1", "C2"])
    assert _checker([full]).check(s).checks["competition_policy"] == "WARN"          # general rule: passes with note


def test_blocking_restrictions_need_resolving_review(tmp_path):
    s = _spec(tmp_path, restrictions=[{"id": "R-OV", "check": "overlap", "text": "corpus question"}])
    r = _checker([_review()]).check(s)
    assert r.checks["overlap_restrictions"] == "PENDING" and r.lifecycle == "PROVENANCE_REVIEW_PENDING"
    r2 = _checker([_review(resolves=["R-OV"])]).check(s)
    assert r2.checks["overlap_restrictions"] == "PASS"


def test_unverified_repo_or_unpinned_revision_is_pending(tmp_path):
    r = _checker([_review(revision=SHA)]).check(_spec(tmp_path, hf_id_status="unverified"))
    assert r.checks["repo_identity"] == "PENDING" and r.lifecycle == "LEGAL_REVIEW_PENDING"


def test_legacy_flags_ignored(tmp_path):
    s = _spec(tmp_path, license_verified=True, provenance_verified=True, overlap_checked=True)
    r = _checker([]).check(s)
    assert not r.eligible and any("legacy registry flags ignored" in w for w in r.warnings)


def test_detect_architecture_variants(tmp_path):
    def mk(name, cfg):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return str(d)

    assert detect_architecture(mk("w", {"model_type": "whisper"})) == "hf_whisper"
    assert detect_architecture(mk("c", {"model_type": "wavlm", "architectures": ["WavLMForCTC"]})) == "hf_ctc"
    assert detect_architecture(mk("e", {"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForPreTraining"]})) == "hf_encoder"
    assert detect_architecture(mk("h", {"model_type": "hubert", "architectures": ["HubertModel"]})) == "hf_encoder"
    assert detect_architecture(mk("q", {"model_type": "qwen3_asr", "architectures": ["Qwen3ASRForConditionalGeneration"]})) == "qwen3_asr"
    assert detect_architecture(None) == "unknown"


def test_record_default_is_not_eligible():
    rec = EligibilityRecord("m", "NOT_EXECUTED", ["license review pending"], {})
    assert not rec.eligible and rec.lifecycle == "LEGAL_REVIEW_PENDING"
