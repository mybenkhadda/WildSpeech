"""Verify the local ASR setup WITHOUT running the pipeline (run by you; no network, no downloads).

    python scripts\\v3_verify_asr.py                       # environment + CUDA + eligibility + components
    python scripts\\v3_verify_asr.py --load-weights        # + load each ELIGIBLE model and decode 1 s of silence
    python scripts\\v3_verify_asr.py --model whisper_small_en --load-weights --device cuda

Sections:
  1. PyTorch / CUDA / package versions (RTX 3070 expected: CUDA available, ~8 GB, compute capability 8.6, bf16)
  2. per-model eligibility by dimension + required actions (same checker as P24)
  3. deep component checks for every installed snapshot: config, processor / feature extractor, tokenizer,
     adapter contract (no weights)
  4. optional smoke forward for ELIGIBLE models (weights from the local snapshot, local_files_only)
Writes outputs/V3/P24_ASR/verification/*.csv. Exit code 0 when the environment is usable (CUDA available or
--allow-cpu), otherwise 1. Ineligible models are reported, not treated as errors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    import pandas as pd

    from wildspeech.asr.environment import component_check, framework_environment, smoke_forward
    from wildspeech.asr.snapshots import detect_architecture, resolve_snapshot
    from wildspeech.context import build_context
    from wildspeech.stages.p24_asr import eligibility, eligibility_tables

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="V3_ASR")
    ap.add_argument("--model", nargs="*", default=None)
    ap.add_argument("--load-weights", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--verify-hashes", action="store_true", help="sha256 every installed manifest file (cached)")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 90)

    print("=== 1. environment")
    env = framework_environment()
    for k, v in env.items():
        print(f"  {k:<22} {v}")
    cuda = bool(env.get("cuda_available"))
    if not cuda:
        print("  !! CUDA not available: install CUDA torch with scripts\\v3_install_asr_env.ps1 -InstallTorch")

    ctx = build_context(a.profile, overrides={"v3": {"asr": {"models": a.model}}} if a.model else None)
    out = ctx.stage_dir("P24") / "verification"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([env]).to_csv(out / "environment.csv", index=False)

    print("\n=== 2. eligibility (local files only)")
    specs, recs, table = eligibility(ctx, deep=True, verify_hashes=a.verify_hashes, env=env)
    cols = ["model_key", "priority", "lifecycle", "eligible", "download_allowed", "legal_status", "provenance_status",
            "checkpoint_status", "framework_status", "architecture", "snapshot_kind", "snapshot_revision"]
    print(table[[c for c in cols if c in table]].sort_values("priority").to_string(index=False))
    tabs = eligibility_tables(ctx, specs, recs)
    if not tabs["required_actions"].empty:
        print("\n  required actions:")
        for r in tabs["required_actions"].sort_values(["priority", "model_key", "step"]).itertuples():
            print(f"   [{r.model_key}] {r.action}")
    table.to_csv(out / "eligibility.csv", index=False)
    tabs["eligibility_checks"].to_csv(out / "eligibility_checks.csv", index=False)

    print("\n=== 3. components (config / processor / tokenizer from local snapshots, no weights)")
    comp = []
    for k in recs:
        if k not in specs:
            continue
        sn = resolve_snapshot(specs[k], ctx.cfg.get_path("paths.hf_cache_dir"))
        if not sn.path:
            print(f"  {k:<26} not installed")
            continue
        c = component_check(specs[k], sn.path, detect_architecture(sn.path))
        comp.append(c)
        print(f"  {k:<26} {'OK ' if c['ok'] else 'BAD'} {c.get('tokenizer_class', '')} {c.get('problems') or c.get('notes') or ''}")
    pd.DataFrame(comp).to_csv(out / "components.csv", index=False)

    if a.load_weights:
        print("\n=== 4. smoke forward (eligible models only)")
        dev = ("cuda" if cuda else "cpu") if a.device == "auto" else a.device
        smoke = []
        for k, r in recs.items():
            if not r.eligible:
                print(f"  {k:<26} skipped (not eligible)")
                continue
            try:
                s = smoke_forward(specs[k], r, ctx.cfg, dev)
            except Exception as e:  # noqa: BLE001
                s = {"model_key": k, "ok": False, "error": f"{type(e).__name__}: {e}"}
            smoke.append(s)
            print(f"  {k:<26} {s}")
        pd.DataFrame(smoke).to_csv(out / "smoke_forward.csv", index=False)
    print(f"\nverification tables: {out}")
    return 0 if (cuda or a.allow_cpu) else 1


if __name__ == "__main__":
    sys.exit(main())
