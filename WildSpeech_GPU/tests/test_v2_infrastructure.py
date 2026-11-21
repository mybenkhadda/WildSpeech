"""Validation registry (freeze, equal-size transfer controls), stage dependency invalidation, lineage, provider /
sequence / embedding / perturbation caches and resume, fold-fitted reference templates, missing modalities."""
import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE, ID


# ------------------------------------------------------------------------------- registry
def test_equal_n_transfer_folds_paired_and_equal_size():
    from wildspeech.v2.validation_registry import equal_n_transfer_folds

    attr = pd.Series(["16k"] * 70 + ["low_rate"] * 30, index=[f"s{i}" for i in range(100)])
    folds = equal_n_transfer_folds(attr, "sample_rate_equal_n", 2, 0.3, 0)
    assert folds
    by = {}
    for f in folds:
        assert not set(f.train_ids) & set(f.valid_ids)
        mode, rest = f.label.split(":", 1)
        tgt = rest.split("->")[1].split("|")[0]
        by.setdefault((f.repeat, tgt), {})[mode] = f
        if mode == "transfer":
            assert set(attr[list(f.train_ids)]) != {tgt}
        else:
            assert set(attr[list(f.train_ids)]) == {tgt}
    for pair in by.values():
        assert len(pair["transfer"].train_ids) == len(pair["within"].train_ids)
        assert pair["transfer"].valid_ids == pair["within"].valid_ids


def test_registry_frozen_reproducible_and_guarded(ctx):
    from wildspeech.v2.validation_registry import RegistryMismatch, ValidationRegistry

    ctx.cfg.set_path("v2.validation.schemes", ["repeated_kfold", "sample_rate_transfer"])
    r1 = ValidationRegistry(ctx).load_or_build()
    r2 = ValidationRegistry(ctx).load_or_build()
    a = [(f.train_ids, f.valid_ids) for f in r1.folds("repeated_kfold")]
    b = [(f.train_ids, f.valid_ids) for f in r2.folds("repeated_kfold")]
    assert a == b and len(a) > 0
    from wildspeech.stages._common import get_runner

    v1 = get_runner(ctx, with_audio=False).folds("repeated_kfold")      # identical to V1 folds -> V1 caches reused
    assert [(set(f.train_ids), set(f.valid_ids)) for f in v1] == [(set(x), set(y)) for x, y in a]
    for f in r1.folds("sample_rate_transfer"):
        assert not set(f.train_ids) & set(f.valid_ids)
    ctx.cfg.set_path("validation.n_splits", 4)                          # data-defining setting changed
    with pytest.raises(RegistryMismatch):
        ValidationRegistry(ctx).load_or_build()
    ctx.cfg.set_path("v2.validation.rebuild_if_changed", True)
    ValidationRegistry(ctx).load_or_build()
    ctx.cfg.set_path("validation.n_splits", 3)
    ValidationRegistry(ctx).load_or_build()
    ctx.cfg.set_path("v2.validation.rebuild_if_changed", False)


def test_inner_folds_partition_training_students():
    from wildspeech.v2.validation_registry import discovery_split, inner_student_folds

    ids = [f"s{i}" for i in range(37)]
    folds = inner_student_folds(ids, 4, 0)
    va = [x for _, v in folds for x in v]
    assert sorted(va) == sorted(ids) and len(set(va)) == len(ids)
    d, c = discovery_split(ids, 0.5, 0)
    assert not set(d) & set(c) and len(d) + len(c) == len(ids) and discovery_split(ids, 0.5, 0)[0] == d


# ------------------------------------------------------------------------ stage invalidation
def test_v2_stage_dependency_invalidation(ctx):
    from wildspeech.orchestrator import STAGES, V2_STAGES, Pipeline
    from wildspeech.v2.lineage import StaleUpstreamError, require_fresh

    pipe = Pipeline(ctx)
    assert list(V2_STAGES) == ["P16", "P18", "P20", "P17", "P19", "P21", "P22"] and len(STAGES) == 23
    for sid in ("P00", "P01", "P16"):
        pipe._write(sid, "completed", pipe.fingerprint(sid), 0.0)
    assert not pipe.is_stale("P16")
    pipe._write("P17", "completed", pipe.fingerprint("P17"), 0.0)
    require_fresh(ctx, "P17")                                           # fresh upstream -> no error
    ctx.cfg.set_path("v2.harmonization.student_budget", 7)              # P16 config changed
    assert pipe.is_stale("P16")
    with pytest.raises(StaleUpstreamError):
        require_fresh(ctx, "P17")
    plan = pipe.resolve(["P17"])
    assert plan.index("P16") < plan.index("P17")
    res = pipe.run(["P17"], dry_run=True).set_index("stage")["status"]
    assert res["P16"] == "would_run"                                    # stale upstream re-planned before P17
    # soft V2 dependency that became stale is pulled into the plan as well
    pipe._write("P18", "completed", "outdated-fingerprint", 0.0)
    ctx.cfg.set_path("v2.phonetic.enabled", True)
    assert "P18" in pipe.resolve(["P17"])
    ctx.cfg.set_path("v2.phonetic.enabled", False)
    ctx.cfg.set_path("v2.harmonization.student_budget", 20)
    for sid in ("P00", "P01", "P16", "P17", "P18"):
        pipe.invalidate([sid])


def test_v2_outputs_live_under_outputs_v2(ctx):
    d = ctx.stage_dir("P16")
    assert d.parent.name == "V2" and d.name == "P16_HARMONIZATION"
    assert ctx.stage_dir("P07").parent == ctx.paths.outputs


# ------------------------------------------------------------------------------ caches
def test_cached_provider_hits_and_recomputes_for_new_targets(ctx):
    from wildspeech.features.conditional_aggregation import CachedProvider, FoldFeatureProvider

    calls = []

    class P(FoldFeatureProvider):
        family = "TST"

        def fit_table(self, train_ids, target_ids):
            calls.append((tuple(train_ids), tuple(target_ids)))
            return pd.DataFrame({"TST__all__x__value": np.arange(len(target_ids), dtype=float)},
                                index=pd.Index(list(target_ids), name=ID))

    cp = CachedProvider(P(ctx, "TEST_PROVIDER"))
    t1 = cp.table(["a", "b"], ["a", "b", "c"])
    t2 = CachedProvider(P(ctx, "TEST_PROVIDER")).table(["b", "a"], ["c"])         # disk hit, other order
    assert len(calls) == 1 and t2.loc["c"].item() == t1.loc["c"].item()
    cp.table(["a", "b"], ["d"])                                                     # unseen target -> recompute
    assert len(calls) == 2


def test_sequence_and_embedding_stores_resume(tmp_path):
    from wildspeech.foundation.encoders import EmbeddingShardStore
    from wildspeech.reference.templates import SequenceStore

    ss = SequenceStore(tmp_path, "fp1")
    rows = [{FILE: f"f{i}", "mfcc": np.ones((5 + i, 13), np.float16) * i, "mel": np.zeros((5 + i, 24), np.float16), "n_speech": 3}
            for i in range(3)]
    ss.append("train", rows[:2])
    ss.append("train", rows[2:])
    assert ss.done("train") == {"f0", "f1", "f2"}
    seq = ss.load(("train",))
    assert seq["mfcc"]["f2"].shape == (7, 13) and float(seq["mfcc"]["f1"][0, 0]) == 1.0
    ss.add_failures("train", [{FILE: "bad"}])
    assert ss.failed("train") == {"bad"}
    es = EmbeddingShardStore(tmp_path, "m", "fp")
    es.append("train", [{FILE: f"e{i}", "emb": np.full((3, 2, 4), i, np.float16)} for i in range(4)])
    es.append("train", [{FILE: "e9", "emb": np.full((3, 2, 4), 9, np.float16)}])
    assert es.done("train") == {"e0", "e1", "e2", "e3", "e9"}
    df = es.load(2, 1, ("train",))
    assert list(df.columns) == [FILE, "e0", "e1", "e2", "e3"] and df.set_index(FILE).loc["e9"].tolist() == [9.0] * 4
    assert not list((tmp_path / "m" / "fp" / "train").glob("*.partial"))


def test_perturbation_cache_reuses_features(ctx, monkeypatch):
    from wildspeech.harmonization.perturbations import PerturbationFeatureCache

    calls = []

    def fn(ws):
        calls.append(len(ws))
        return [{"F::rms": float(np.sqrt(np.mean(w ** 2)))} for w in ws]

    cache = PerturbationFeatureCache(ctx, fn, "fp-test", "TEST")
    waves = {f"c{i}": np.random.default_rng(i).normal(0, 0.1, 8000).astype(np.float32) for i in range(6)}
    monkeypatch.setattr(cache, "_decode", lambda clips, seconds: waves)
    b1, _ = cache.get(sorted(waves), None)
    n1 = len(calls)
    b2, _ = cache.get(sorted(waves), None)
    assert len(calls) == n1 and b1.equals(b2)
    g, method = cache.get(sorted(waves), "gain_minus_12db")
    assert method == "gain" and np.allclose(g["F::rms"].to_numpy(), b1["F::rms"].to_numpy() * 10 ** (-12 / 20), rtol=1e-4)


# --------------------------------------------------------------- reference templates
def test_exemplar_bank_uses_only_training_students_and_excludes_self():
    from wildspeech.reference.templates import ExemplarBank

    c = pd.DataFrame({FILE: [f"f{i}" for i in range(20)], ID: [f"s{i % 5}" for i in range(20)], "item_id": "blending::pot",
                      "item_score": [1.0] * 16 + [0.0] * 4})
    train = c[c[ID].isin(["s0", "s1", "s2"])]
    bank = ExemplarBank(k=3, min_correct=2, seed=0).fit(train, set(c[FILE]))
    ex = bank.exemplars("blending::pot")
    assert len(ex) == 3 and set(c.set_index(FILE).loc[ex, ID]) <= {"s0", "s1", "s2"}
    assert all(bank.owner_[e] != "s0" for e in bank.exemplars("blending::pot", exclude_student="s0"))
    assert ExemplarBank(k=3, min_correct=50).fit(train, set(c[FILE])).exemplars("blending::pot") == []


# ---------------------------------------------------------------- missing modalities
def test_conditional_provider_handles_missing_clip_features(ctx):
    from wildspeech.features.conditional_aggregation import ConditionalAcousticProvider, clip_frame

    c = clip_frame(ctx)
    tr = c[c["split"] == "train"]
    feats = pd.DataFrame({FILE: tr[FILE].iloc[::2].to_numpy(), "TEMPORAL::speech_ratio": np.random.default_rng(0).uniform(0, 1, len(tr.iloc[::2]))})
    p = ConditionalAcousticProvider(ctx, "MISS", feats, ["TEMPORAL::speech_ratio"], prefix="HARM")
    ids = sorted(tr[ID].unique())
    T = p.fit_table(ids[:20], ids)
    assert len(T) == len(ids) and T.index.name == ID
    assert np.isfinite(T.filter(like="__mean").to_numpy(dtype=float)).any()


def test_strongest_baseline_falls_back_without_artifacts(ctx):
    from wildspeech.v2.baseline import strongest_tabular

    b = strongest_tabular(ctx)
    assert b["model"] and all(f in ("SCREENING", "RECORDING", "BEHAVIOR", "ITEM", "LINGUISTIC") for f in b["families"])


def test_report_renders_without_runs(ctx):
    from wildspeech.v2.research_report_v2 import build_report, stage_findings

    assert "not" in stage_findings(ctx, "P21")[0] or "status" in stage_findings(ctx, "P21")[0]
    out = build_report(ctx)
    html = open(out["html"], encoding="utf-8").read()
    assert "WildSpeech V2" in html and "cannot be answered" in open(out["markdown"], encoding="utf-8").read()
