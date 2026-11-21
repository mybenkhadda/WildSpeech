import pandas as pd

from wildspeech.pipeline import CACHED, COMPLETE, BLOCKED_DEPENDENCY, FAILED, Pipeline, StageSpec


def _specs(calls):
    def a(ctx):
        calls.append("A")
        ctx.save_df(pd.DataFrame({"x": [1]}), "art_a")
        return {"outputs": ["art_a"]}

    def b(ctx):
        calls.append("B")
        ctx.save_df(ctx.load_df("art_a").assign(y=2), "art_b")
        return {"outputs": ["art_b"]}

    def boom(ctx):
        raise ValueError("boom")

    def c(ctx):
        calls.append("C")
        return {"outputs": []}
    return {"A": StageSpec("A", "a", a), "B": StageSpec("B", "b", b, ["A"]),
            "X": StageSpec("X", "x", boom), "C": StageSpec("C", "c", c, ["X"])}


def test_second_run_is_cached(ctx):
    calls = []
    Pipeline(ctx, _specs(calls)).run(["B"])
    assert calls == ["A", "B"]
    rep = Pipeline(ctx, _specs(calls)).run(["B"])
    assert set(rep["status"]) == {CACHED} and calls == ["A", "B"]


def test_config_change_and_dependency_rerun_invalidate(ctx):
    calls = []
    Pipeline(ctx, _specs(calls)).run(["B"])
    ctx.config["seed"] = 7
    Pipeline(ctx, _specs(calls)).run(["B"])
    assert calls == ["A", "B", "A", "B"]
    Pipeline(ctx, _specs(calls)).run(["B"], force=("A",))
    assert calls[-2:] == ["A", "B"]


def test_missing_output_forces_recompute(ctx):
    calls = []
    Pipeline(ctx, _specs(calls)).run(["B"])
    ctx.store.path("art_b").unlink()
    Pipeline(ctx, _specs(calls)).run(["B"])
    assert calls[-1] == "B"


def test_failures_block_dependants(ctx):
    calls = []
    rep = Pipeline(ctx, _specs(calls)).run(["C"]).set_index("stage_id")["status"]
    assert rep["X"] == FAILED and rep["C"] == BLOCKED_DEPENDENCY and "C" not in calls
