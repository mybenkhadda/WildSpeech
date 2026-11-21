"""P00 — environment, hardware benchmark, CPU/GPU parity, data integrity, TAR manifests, cache report."""
from __future__ import annotations

import pandas as pd

from .. import viz
from ..acoustic.benchmark import parity_checks, run_benchmark, synthetic_waves, load_sample_waves
from ..acoustic.extractor import AcousticExtractionRunner, active_families
from ..acoustic.store import FeatureStore
from ..data.integrity import integrity_checks
from ..data.tar_manifest import build_tar_manifest
from ..hardware import hardware_report, package_versions
from ..model_registry import eligibility_table, load_registry
from ._common import save_fig

S = "P00"


def run(ctx) -> dict:
    hw = hardware_report()
    ctx.save_json(hw, S, "hardware_report")
    ctx.save_table(pd.DataFrame(package_versions()), S, "package_versions")
    files = {k: str(v) for k, v in ctx.paths.files.items()}
    missing = [k for k in ctx.cfg.get_path("data.required_files", {}) if k not in ctx.paths.files]
    ctx.save_json({"data_dir": str(ctx.paths.data_dir), "files": files, "missing": missing}, S, "data_files")
    if any(k in missing for k in ("train_clips", "train_students", "labels")):
        raise FileNotFoundError(f"required competition files missing: {missing}")

    b = ctx.bundle()
    ctx.save_table(b.schema_report, S, "schema_report")
    manifests = {}
    for split in ("train", "test"):
        key = f"{split}_audio"
        if key in ctx.paths.files:
            manifests[split] = build_tar_manifest(ctx.paths.files[key], ctx.paths.cache / "manifests", logger=ctx.logger)
            ctx.save_table(manifests[split].describe(include="all").T.reset_index(), S, f"manifest_{split}_summary")
    checks = integrity_checks(b, manifests)
    ctx.save_table(checks, S, "integrity_checks")
    ctx.save_table(eligibility_table(load_registry(ctx.paths.configs)), S, "foundation_model_eligibility")

    # hardware benchmark + parity (synthetic fallback if audio is unavailable)
    runner = AcousticExtractionRunner(ctx)
    gpu_fams = [f for f in active_families(ctx.cfg) if f != "wavelet"] or ["quality", "spectral"]
    g = ctx.cfg.section("gpu")
    bench, rec = run_benchmark(runner, ctx.paths.cache / "runtime" / "batch_recommendation.json",
                               int(g.get("benchmark_clips", 192)), float(g.get("benchmark_seconds", 6.0)),
                               list(g.get("benchmark_batch_sizes", [8, 16, 24, 32])), gpu_fams)
    ctx.save_table(bench, S, "gpu_benchmark")
    ctx.save_json(rec, S, "batch_recommendation")
    cuda = bench[bench["device"] == "cuda"]
    if not cuda.empty:
        save_fig(ctx, S, "gpu_throughput_by_batch", viz.lines(cuda.assign(series="cuda"), "batch_size", "clips_per_s", "series",
                                                              "GPU throughput by batch size"), cuda)
    n_par = int(g.get("parity_clips", 24))
    waves = load_sample_waves(runner, n_par) or synthetic_waves(n_par, 4.0, runner.p.sr, seed=1)
    par = parity_checks(runner, waves, gpu_fams)
    ctx.save_table(par, S, "numerical_parity")

    store = FeatureStore(ctx.paths.cache / "features")
    cache = store.status()
    ctx.save_table(cache, S, "feature_cache_status")
    n_fail = int((checks["status"] == "FAIL").sum())
    return {"integrity_fail": n_fail, "integrity_warn": int((checks["status"] == "WARN").sum()),
            "cuda": bool(hw.get("cuda_available")), "gpu": hw.get("gpu_name"), "recommended_batch": rec.get("recommended_batch_size"),
            "parity_pass_frac": float(par["pass"].mean()) if len(par) else None, "cache_rows": int(cache["rows"].sum()) if len(cache) else 0}
