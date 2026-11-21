"""Dependency-aware stage orchestration with fingerprint-based cache validation and resume.

A stage is skipped when (resume=True) its last status is 'completed' and its fingerprint is unchanged.
fingerprint = hash(config sections the stage declares + code version + fingerprints of upstream stages).
Hard dependencies must be completed; soft dependencies are used when available.
"""
from __future__ import annotations

import importlib
import time
import traceback
from dataclasses import dataclass, field

import pandas as pd

from .context import RunContext, save_last_run
from .io_utils import read_json, stable_hash, write_json
from .paths import STAGE_NAMES


@dataclass(frozen=True)
class StageSpec:
    id: str
    module: str
    deps: tuple = ()
    soft: tuple = ()
    keys: tuple = ()
    flag: str | None = None
    expensive: bool = False
    description: str = ""
    # "module:function(ctx) -> str": cheap token of external state (local checkpoints, reviews, real outputs) that
    # is folded into the fingerprint, so the stage is re-planned when that state changes. None for V1/V2 stages.
    state: str | None = None


# Status written when a V3 stage returns {"status": "NOT_EXECUTED"}: it is NOT 'completed', so hard dependants are
# blocked and the stage is re-run (cheaply re-checked) next time instead of being cached as a success.
NOT_EXECUTED_STATUS = "not_executed"


COMMON = ("data", "project.seed", "project.code_version")
STAGES: dict[str, StageSpec] = {s.id: s for s in [
    StageSpec("P00", "p00_infrastructure", (), (), COMMON + ("gpu", "audio.sample_rate", "paths"), None, False,
              "environment, hardware benchmark, data integrity, manifests, cache report"),
    StageSpec("P01", "p01_tabular_eda", ("P00",), (), COMMON + ("validation", "analysis", "features", "models.selector_k"), None, False,
              "targets, screening, items, IRT, adaptive behaviour, tabular baselines"),
    StageSpec("P02", "p02_acoustic", ("P00",), (), COMMON + ("audio", "gpu.parity_rtol", "analysis.bootstrap"), "audio.extract_train", True,
              "GPU acoustic extraction + acoustic EDA"),
    StageSpec("P03", "p03_phonetic", ("P00",), (), COMMON + ("phonetic",), "phonetic.enabled", True,
              "local ASR, CTC alignment confidence, expected-text deviation"),
    StageSpec("P04", "p04_embeddings", ("P00",), (), COMMON + ("embeddings",), "embeddings.enabled", True,
              "foundation-model embeddings"),
    StageSpec("P05", "p05_features", ("P01",), ("P02", "P03", "P04"), COMMON + ("features", "embeddings.pca_components"), None, False,
              "student-level features (cross-fitted), feature registry"),
    StageSpec("P06", "p06_phenotypes", ("P05",), (), COMMON + ("analysis",), None, False, "phenotypes"),
    StageSpec("P07", "p07_residuals", ("P05",), (), COMMON + ("analysis", "validation"), None, False, "residual discovery"),
    StageSpec("P08", "p08_robustness", ("P05",), ("P07", "P10"), COMMON + ("analysis", "validation", "audio"), None, False, "robustness"),
    StageSpec("P09", "p09_indices", ("P05",), (), COMMON + ("indices", "validation"), None, False, "literacy indices"),
    StageSpec("P10", "p10_benchmark", ("P05",), ("P09",), COMMON + ("models", "validation", "features"), None, False,
              "model benchmark + family ablations"),
    StageSpec("P11", "p11_multimodal", ("P10",), (), COMMON + ("multimodal", "models", "validation"), "multimodal.enabled", False,
              "fusion, residual learning, neural MIL / transformer / MoE"),
    StageSpec("P12", "p12_failures", ("P10",), ("P11",), COMMON + ("analysis",), None, False, "failure analysis"),
    StageSpec("P13", "p13_champion", ("P10",), ("P11",), COMMON + ("validation", "models"), None, False, "champion selection"),
    StageSpec("P14", "p14_submission", ("P13",), (), COMMON + ("submission", "audio"), "submission.enabled", True,
              "independent test inference + validation"),
    StageSpec("P15", "p15_report", (), tuple(f"P{i:02d}" for i in range(15)), COMMON + ("report",), None, False, "scientific report"),
]}

# ---------------------------------------------------------------------------------------------- V2 (P16-P22)
# Insertion order == execution order: the phonetic (P18) and foundation-model (P20) extractions run before
# the correctness classifiers (P17) and reference modelling (P19) that can consume them.
V2C = COMMON + ("v2.project", "v2.validation", "v2.evaluation")
V2_STAGES: dict[str, StageSpec] = {s.id: s for s in [
    StageSpec("P16", "p16_harmonization", ("P01",), ("P02", "P10"), V2C + ("audio", "v2.harmonization"),
              "v2.harmonization.enabled", True, "acoustic domain harmonization + item-conditioned residualization"),
    StageSpec("P18", "p18_phonetics", ("P01",), (), V2C + ("v2.phonetic",), "v2.phonetic.enabled", True,
              "content-conditioned phonetic analysis (ASR / CTC / forced alignment / G2P)"),
    StageSpec("P20", "p20_embeddings", ("P01",), (), V2C + ("v2.foundation",), "v2.foundation.enabled", True,
              "frozen speech foundation-model representations, layer discovery"),
    StageSpec("P17", "p17_correctness", ("P16",), ("P18", "P20", "P10"), V2C + ("v2.correctness", "v2.harmonization.selected"),
              "v2.correctness.enabled", True, "clip-level correctness intelligence (nested cross-fitting)"),
    StageSpec("P19", "p19_reference", ("P16",), ("P18", "P20"), V2C + ("v2.reference", "v2.harmonization.selected"),
              "v2.reference.enabled", True, "reference-conditioned speech modelling (templates, DTW, similarity)"),
    StageSpec("P21", "p21_multimodal", ("P17",), ("P16", "P18", "P19", "P20", "P10"), V2C + ("v2.multimodal",),
              "v2.multimodal.enabled", False, "compact multimodal student-level regression"),
    StageSpec("P22", "p22_failure_recovery", ("P21",), ("P11", "P12"), V2C + ("v2.failure", "v2.submission"),
              None, False, "failure recovery, deployability checks, champion selection, submissions"),
]}
STAGES.update(V2_STAGES)

# ---------------------------------------------------------------------------------------------- V3 (P23-P29)
# V3 stages declare V1/V2 stages only as SOFT dependencies (artefacts they read). They never re-plan them:
# a stale V1/V2 artefact is recorded by v3.common.begin (and audited in P23) instead of being recomputed.
V3C = COMMON + ("v3.project", "v3.runtime")
V3_STAGES: dict[str, StageSpec] = {s.id: s for s in [
    StageSpec("P23", "p23_fusion_audit", (), ("P10", "P11", "P21"), V3C + ("v3.audit",),
              "v3.audit.enabled", False, "audit of the V2 fusion system: zero-ΔR² diagnosis, fallback tracing, task-ensemble repair"),
    StageSpec("P24", "p24_asr", (), ("P01",), V3C + ("v3.asr",), "v3.asr.enabled", True,
              "eligible local child/general ASR inference (resumable, provenance-tracked)",
              state="wildspeech.asr.snapshots:p24_state_token"),
    # P25 gates on REAL accepted ASR outputs (asr.outputs), not on P24's status: an eligibility-only P24 run is
    # 'not_executed', and outputs from an earlier eligible run stay usable.
    StageSpec("P25", "p25_phonetic_features", (), ("P24",), V3C + ("v3.asr.models", "v3.asr.splits", "v3.phonetics"),
              "v3.phonetics.enabled", False, "expected-text-conditioned word/phoneme/confidence/timing/task features",
              state="wildspeech.asr.outputs:output_state_token"),
    StageSpec("P26", "p26_correct_only", ("P25",), ("P17", "P21"), V3C + ("v3.discovery",), "v3.discovery.enabled", False,
              "correct-response speech intelligence (item-conditioned, fold-fitted)"),
    StageSpec("P27", "p27_asr_disagreement", ("P25",), (), V3C + ("v3.disagreement",), "v3.disagreement.enabled", False,
              "ASR model disagreement and consensus"),
    StageSpec("P28", "p28_representations", ("P25",), ("P26", "P27"), V3C + ("v3.representations",),
              "v3.representations.enabled", False, "compact phonetic representations (nested, budget-controlled)"),
    StageSpec("P29", "p29_reading_ablation", ("P28",), ("P23", "P26", "P27"), V3C + ("v3.ablation", "v3.representations"),
              "v3.ablation.enabled", False, "reading prediction ASR00-ASR12, robustness, controls, champion comparison"),
]}
STAGES.update(V3_STAGES)


def protected_stages(cfg) -> set[str]:
    """Completed stages never re-run implicitly (v3.orchestration.protected_stages; not fingerprinted)."""
    return {str(s) for s in (cfg.get_path("v3.orchestration.protected_stages", []) or [])}


def final_status(sid: str, summary: dict) -> str:
    """'completed' unless a V3 stage reports NOT_EXECUTED (eligibility-only P24, P25-P29 without real outputs)."""
    if sid in V3_STAGES and str((summary or {}).get("status", "")).upper() == "NOT_EXECUTED":
        return NOT_EXECUTED_STATUS
    return "completed"


class Pipeline:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    # ------------------------------------------------------------ status IO
    def status_path(self, sid: str):
        return self.ctx.stage_dir(sid) / "_status.json"

    def status(self, sid: str) -> dict:
        return read_json(self.status_path(sid), {}) or {}

    def fingerprint(self, sid: str) -> str:
        spec = STAGES[sid]
        up = {d: self.status(d).get("fingerprint") for d in spec.deps + spec.soft
              if self.status(d).get("status") in ("completed",)}
        payload = {"stage": sid, "cfg": self.ctx.cfg.fingerprint(spec.keys), "up": up}
        if spec.state:
            payload["state"] = self.state_token(spec)
        return stable_hash(payload)

    def state_token(self, spec: StageSpec) -> str:
        mod, fn = spec.state.split(":")
        try:
            return str(getattr(importlib.import_module(mod), fn)(self.ctx))
        except Exception as e:  # noqa: BLE001 - an unreadable state never blocks planning; it forces a re-run
            return f"state-error:{type(e).__name__}:{time.time()}"

    # ------------------------------------------------------------- planning
    def is_stale(self, sid: str) -> bool:
        """Completed, but its recorded fingerprint no longer matches config + upstream fingerprints."""
        st = self.status(sid)
        return st.get("status") == "completed" and st.get("fingerprint") != self.fingerprint(sid)

    def stale_upstream(self, sid: str) -> list[str]:
        spec = STAGES[sid]
        return [d for d in spec.deps + spec.soft if self.is_stale(d)]

    def resolve(self, requested: list[str]) -> list[str]:
        out: list[str] = []
        refresh = bool(self.ctx.cfg.get_path("v2.runtime.refresh_stale_soft_deps", True))

        refresh_v3 = bool(self.ctx.cfg.get_path("v3.runtime.refresh_stale_v3_soft_deps", True))
        prot = protected_stages(self.ctx.cfg)

        def visit(s):
            if s in out:
                return
            for d in STAGES[s].deps:
                visit(d)
            # V3 stages re-plan only stale V3 soft dependencies; V1/V2 artefacts are consumed read-only.
            if refresh_v3 and s in V3_STAGES:
                for d in STAGES[s].soft:
                    if d in V3_STAGES and d not in prot and self.enabled(d) and self.is_stale(d):
                        visit(d)
            # V2 stages also re-run soft dependencies whose artefacts became stale, so a downstream stage never
            # combines outputs produced under different upstream states.
            if refresh and s in V2_STAGES:
                for d in STAGES[s].soft:
                    if d in V2_STAGES and self.enabled(d) and self.is_stale(d):
                        visit(d)
            out.append(s)

        for s in requested:
            if s not in STAGES:
                raise KeyError(f"unknown stage {s}")
            visit(s)
        return sorted(out, key=lambda s: list(STAGES).index(s))

    def enabled(self, sid: str) -> bool:
        flag = STAGES[sid].flag
        return True if flag is None else bool(self.ctx.cfg.get_path(flag, False))

    # ------------------------------------------------------------------ run
    def run(self, stages: list[str] | None = None, force: list[str] | tuple = (), dry_run: bool = False) -> pd.DataFrame:
        requested = list(stages or self.ctx.default_stages or list(STAGES))
        plan = self.resolve(requested)
        resume = bool(self.ctx.cfg.get_path("runtime.resume", True))
        stop = bool(self.ctx.cfg.get_path("runtime.stop_on_error", False))
        if not dry_run:
            save_last_run(self.ctx, requested)
        rows = []
        would: set[str] = set()
        prot = protected_stages(self.ctx.cfg)
        force = tuple(force or ())
        for sid in plan:
            spec = STAGES[sid]
            if not self.enabled(sid):
                if not dry_run:
                    self._write(sid, "disabled", None, 0.0, note=f"{spec.flag}=false")
                rows.append({"stage": sid, "status": "disabled", "note": spec.flag})
                continue
            blocked = [d for d in spec.deps if self.status(d).get("status") != "completed" and d not in would]
            if blocked:
                if not dry_run:
                    self._write(sid, "blocked", None, 0.0, note=f"waiting for {blocked}")
                rows.append({"stage": sid, "status": "blocked", "note": ",".join(blocked)})
                continue
            fp = self.fingerprint(sid)
            prev = self.status(sid)
            if sid in prot and sid not in force and prev.get("status") == "completed":
                stale = prev.get("fingerprint") != fp
                rows.append({"stage": sid, "status": "protected", "note": f"completed {prev.get('finished')}"
                             + (" (fingerprint changed; force it explicitly to recompute)" if stale else "")})
                continue
            if resume and sid not in force and prev.get("status") == "completed" and prev.get("fingerprint") == fp:
                rows.append({"stage": sid, "status": "cached", "note": prev.get("finished")})
                continue
            if dry_run:
                would.add(sid)
                rows.append({"stage": sid, "status": "would_run", "note": spec.description})
                continue
            self.ctx.logger.info("=== %s %s — %s", sid, STAGE_NAMES[sid], spec.description)
            t0 = time.perf_counter()
            self._write(sid, "running", fp, 0.0)
            try:
                mod = importlib.import_module(f"wildspeech.stages.{spec.module}")
                summary = mod.run(self.ctx) or {}
                dt = time.perf_counter() - t0
                upstream = {d: self.status(d).get("fingerprint") for d in spec.deps + spec.soft
                            if self.status(d).get("status") == "completed"}
                outcome = final_status(sid, summary)
                if spec.state:                          # the stage may have changed its own state (new outputs)
                    fp = self.fingerprint(sid)
                note = f"{dt:.1f}s" if outcome == "completed" else str(summary.get("reason", ""))[:300]
                self._write(sid, outcome, fp, dt, summary=summary, upstream=upstream, note=note)
                rows.append({"stage": sid, "status": outcome, "note": note})
            except Exception as e:  # noqa: BLE001 - recorded in the stage status file
                dt = time.perf_counter() - t0
                tb = traceback.format_exc()
                self._write(sid, "failed", fp, dt, note=f"{type(e).__name__}: {e}", traceback=tb)
                self.ctx.logger.error("%s failed: %s\n%s", sid, e, tb)
                rows.append({"stage": sid, "status": "failed", "note": f"{type(e).__name__}: {e}"})
                if stop:
                    break
        return pd.DataFrame(rows)

    def _write(self, sid, status, fp, seconds, **extra) -> None:
        payload = {"stage": sid, "name": STAGE_NAMES[sid], "status": status, "fingerprint": fp, "seconds": seconds,
                   "profile": self.ctx.profile, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        write_json(self.status_path(sid), payload)

    def status_table(self) -> pd.DataFrame:
        rows = []
        for sid, spec in STAGES.items():
            st = self.status(sid)
            cur = self.fingerprint(sid) if st.get("status") == "completed" else None
            rows.append({"stage": sid, "name": STAGE_NAMES[sid], "status": st.get("status", "never_run"),
                         "up_to_date": (cur == st.get("fingerprint")) if cur else None, "seconds": st.get("seconds"),
                         "finished": st.get("finished"), "note": st.get("note", ""), "enabled": self.enabled(sid),
                         "expensive": spec.expensive})
        return pd.DataFrame(rows)

    def invalidate(self, stages: list[str]) -> None:
        for sid in stages:
            p = self.status_path(sid)
            if p.exists():
                p.unlink()
