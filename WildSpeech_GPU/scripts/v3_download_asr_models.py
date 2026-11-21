"""Install the V3 speech-model checkpoints (run by you). Dry run unless --execute.

    python scripts\\v3_download_asr_models.py                          # plan: storage, missing files, GB, disk
    python scripts\\v3_download_asr_models.py --model whisper_small_en --execute
    python scripts\\v3_download_asr_models.py --execute                # every model whose reviews pass
    python scripts\\v3_download_asr_models.py --verify                 # sha256 of installed files (no network)

This is a thin wrapper over wildspeech.asr.model_manager (the same code P24 uses), so there is exactly one
download path:
  * a model is downloaded ONLY if its eligibility record says download_allowed, i.e. registry_validity,
    license_documentation, license_approval, training_provenance, overlap_assessment and
    competition_permission all PASS (approved reviews in configs/v3_asr_reviews.yaml). Pending models are listed
    with the exact review / proposal commands instead;
  * --execute is the explicit ALLOW_MODEL_DOWNLOADS for this call;
  * only the files in the registry manifest are fetched, at the pinned commit, resumably, with hf_hub_download
    (complete HF-cache snapshots are reused in place; partial ones are completed in the cache);
  * every file is checked against the published size and LFS sha256; .bin-only checkpoints are converted to a
    sha256-verified local safetensors copy because transformers 4.57 refuses torch.load on torch 2.5.1;
  * models/model_installation_manifest.json is rewritten afterwards.
trust_remote_code is never used. No audio is involved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    import pandas as pd

    from wildspeech.asr.eligibility import checker_for
    from wildspeech.asr.model_manager import install, install_plan, verify_installed
    from wildspeech.asr.model_registry import document_for
    from wildspeech.context import build_context

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="V3_ASR")
    ap.add_argument("--model", nargs="*", default=None)
    ap.add_argument("--execute", action="store_true", help="download (ALLOW_MODEL_DOWNLOADS for this call)")
    ap.add_argument("--verify", action="store_true", help="only hash installed files")
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 after download (size is still checked)")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)
    ctx = build_context(a.profile)
    if a.verify:
        t = verify_installed(ctx, a.model)
        print(t.to_string(index=False) if not t.empty else "no installed snapshot")
        return 1 if (not t.empty and ((t["sha_ok"] == False) | (t["size_ok"] == False)).any()) else 0  # noqa: E712
    doc = document_for(ctx)
    keys = list(a.model or ctx.cfg.get_path("v3.asr.models", list(doc.specs)))
    recs = checker_for(ctx).check_all(doc.specs, keys, probe_files=True)
    print("=== lifecycle")
    for k in keys:
        r = recs.get(k)
        if r is None:
            continue
        print(f"  {k:<26} {r.lifecycle:<26} download_allowed={r.download_allowed}")
        if not r.download_allowed:
            for act in r.actions[:4]:
                print(f"      -> {act}")
    print("\n=== plan")
    print(install_plan(ctx, keys, recs).to_string(index=False))
    if not a.execute:
        print("\nDRY RUN: nothing downloaded. Re-run with --execute (approved models only).")
        return 0
    print("\n=== install")
    t = install(ctx, keys, allow=True, verify=not a.no_verify)
    print(t[[c for c in ("model_key", "action", "storage", "path", "download_gb", "ok", "error") if c in t]].to_string(index=False)
          if not t.empty else "nothing to install")
    return 1 if (not t.empty and (t["action"] == "failed").any()) else 0


if __name__ == "__main__":
    sys.exit(main())
