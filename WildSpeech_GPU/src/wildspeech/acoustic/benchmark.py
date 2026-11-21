"""CPU/GPU throughput benchmark, batch-size recommendation and numerical parity checks (P00)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import FILE
from ..io_utils import write_json
from .cpu_reference import compare_parity, numpy_parity_stats
from .dsp import AcousticParams


def synthetic_waves(n: int, seconds: float, sr: int, seed: int = 0, vary: bool = True) -> list[np.ndarray]:
    """Harmonic 'voiced' bursts + noise; lengths vary to exercise masking."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        dur = seconds * (rng.uniform(0.3, 1.0) if vary else 1.0)
        t = np.arange(int(dur * sr)) / sr
        f0 = rng.uniform(180, 320)
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        env = (np.sin(2 * np.pi * rng.uniform(1.5, 4.0) * t) > 0).astype(float)
        y = 0.1 * sig * env + 0.003 * rng.standard_normal(len(t))
        out.append(y.astype(np.float32))
    return out


def load_sample_waves(runner, n: int, split: str = "train") -> list[np.ndarray]:
    """Decode n real clips from the TAR (shortest-first is avoided: random by filename hash)."""
    from ..data.audio_io import decode_audio_bytes, make_reader

    man = runner.manifest(split)
    if man is None or man.empty:
        return []
    reader = make_reader(runner.ctx.paths.files[f"{split}_audio"], man)
    if reader is None:
        return []
    names = man.loc[man["safe"], FILE]
    order = pd.util.hash_pandas_object(names.reset_index(drop=True), index=False).sort_values().index[:n]
    waves = []
    try:
        for f in names.reset_index(drop=True).iloc[order]:
            try:
                y, _ = decode_audio_bytes(reader.read(f), runner.p.sr, runner.cfg.get_path("audio.max_clip_seconds", 30.0))
                waves.append(y)
            except Exception:
                continue
    finally:
        reader.close()
    return waves


def benchmark_extraction(runner, waves: list[np.ndarray], batch_sizes: list[int], families: list[str]) -> pd.DataFrame:
    import torch

    from .gpu_features import GPUAcousticExtractor

    rows = []
    devices = ["cpu"] + (["cuda"] if runner.device == "cuda" else [])
    for dev in devices:
        ext = GPUAcousticExtractor(runner.p, dev)
        sizes = batch_sizes if dev == "cuda" else [min(batch_sizes)]
        sample = waves if dev == "cuda" else waves[: max(8, len(waves) // 4)]
        for bs in sizes:
            ordered = sorted(sample, key=len)
            if dev == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            ext.extract_batch(ordered[:min(bs, len(ordered))], families)   # warm-up
            t0 = time.perf_counter()
            status = "ok"
            try:
                for i in range(0, len(ordered), bs):
                    ext.extract_batch(ordered[i:i + bs], families)
                if dev == "cuda":
                    torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                status = f"{type(e).__name__}"
                if dev == "cuda":
                    torch.cuda.empty_cache()
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_reserved() / 1e9 if dev == "cuda" else np.nan
            rows.append({"device": dev, "batch_size": bs, "clips": len(ordered), "seconds": dt,
                         "clips_per_s": len(ordered) / dt if status == "ok" else np.nan,
                         "audio_s_per_s": sum(len(w) for w in ordered) / runner.p.sr / dt if status == "ok" else np.nan,
                         "peak_vram_gb": peak, "status": status})
    return pd.DataFrame(rows)


def numpy_throughput(waves: list[np.ndarray], p: AcousticParams) -> dict:
    t0 = time.perf_counter()
    for y in waves:
        numpy_parity_stats(y, p)
    dt = time.perf_counter() - t0
    return {"device": "numpy_reference", "batch_size": 1, "clips": len(waves), "seconds": dt,
            "clips_per_s": len(waves) / dt, "status": "ok (subset of features)"}


def recommend_batch_size(bench: pd.DataFrame, total_vram_gb: float, safety: float) -> dict:
    g = bench[(bench["device"] == "cuda") & (bench["status"] == "ok")]
    if g.empty:
        return {"recommended_batch_size": 16, "reason": "no successful CUDA benchmark"}
    g = g[g["peak_vram_gb"] <= safety * total_vram_gb] if total_vram_gb else g
    if g.empty:
        return {"recommended_batch_size": int(bench["batch_size"].min()), "reason": "all sizes exceed VRAM safety"}
    best = g.sort_values("clips_per_s", ascending=False).iloc[0]
    # prefer the smallest batch within 5% of the best throughput (less padding/OOM risk)
    near = g[g["clips_per_s"] >= 0.95 * best["clips_per_s"]].sort_values("batch_size").iloc[0]
    return {"recommended_batch_size": int(near["batch_size"]), "best_clips_per_s": float(best["clips_per_s"]),
            "peak_vram_gb": float(near["peak_vram_gb"]), "reason": "smallest size within 5% of max throughput under VRAM safety"}


def parity_checks(runner, waves: list[np.ndarray], families: list[str]) -> pd.DataFrame:
    """(1) torch-vs-numpy core DSP; (2) torch CUDA vs torch CPU on every GPU family feature."""
    from .gpu_features import GPUAcousticExtractor

    g = runner.cfg.section("gpu")
    rtol, atol = float(g.get("parity_rtol", 2e-3)), float(g.get("parity_atol", 1e-3))
    cpu_ext = GPUAcousticExtractor(runner.p, "cpu")
    ref = [numpy_parity_stats(y, runner.p) for y in waves]
    tables = [compare_parity(cpu_ext.parity_stats(waves), ref, rtol, atol, "torch_cpu", "numpy")]
    if runner.device == "cuda":
        dev_ext = GPUAcousticExtractor(runner.p, "cuda")
        tables.append(compare_parity(dev_ext.parity_stats(waves), ref, rtol, atol, "torch_cuda", "numpy"))
        a, b = dev_ext.extract_batch(waves, families), cpu_ext.extract_batch(waves, families)
        for f in families:
            t = compare_parity(a[f], b[f], max(rtol, 1e-2), max(atol, 1e-2), f"cuda:{f}", f"cpu:{f}")
            tables.append(t)
    return pd.concat(tables, ignore_index=True)


def run_benchmark(runner, out_json, n_clips: int, seconds: float, batch_sizes: list[int], families: list[str]) -> tuple[pd.DataFrame, dict]:
    waves = load_sample_waves(runner, n_clips) or synthetic_waves(n_clips, seconds, runner.p.sr)
    bench = benchmark_extraction(runner, waves, batch_sizes, families)
    bench = pd.concat([bench, pd.DataFrame([numpy_throughput(waves[:16], runner.p)])], ignore_index=True)
    total = 0.0
    if runner.device == "cuda":
        import torch

        total = torch.cuda.get_device_properties(0).total_memory / 1e9
    rec = recommend_batch_size(bench, total, float(runner.cfg.get_path("gpu.vram_safety_fraction", 0.8)))
    rec["source"] = "real clips" if len(waves) and runner.manifest("train") is not None else "synthetic"
    write_json(out_json, rec)
    return bench, rec
