"""Windows-compatible command line interface.

  python -m wildspeech run --profile FAST
  python -m wildspeech run --profile FULL --stages P02 P07 --force P07
  python -m wildspeech run --profile FAST --set gpu.batch_size=32 --set audio.enable_pitch=true
  python -m wildspeech plan --profile RESEARCH          (dry run: what would execute)
  python -m wildspeech status
  python -m wildspeech cache                             (feature-family cache report)
  python -m wildspeech clear-cache --family pitch --split train
  python -m wildspeech invalidate P07 P08
  V2 (P16-P22):
  python -m wildspeech plan --profile V2_CORE            (V2_SMOKE | V2_CORE | V2_PHONETIC | V2_FOUNDATION | V2_FULL)
  python -m wildspeech run  --profile V2_FULL
  python -m wildspeech run  --profile V2_SUBMISSION      (P22 with test inference, after reviewing P22)
  python -m wildspeech v2-inventory | v2-registry | v2-audit | v2-report
  V3 (P23-P29):
  python -m wildspeech plan --profile V3_FULL            (V3_SMOKE | V3_AUDIT | V3_ASR | V3_SCIENCE | V3_FULL | V3_SUBMISSION)
  python -m wildspeech run  --profile V3_AUDIT
  python -m wildspeech v3-eligibility [--deep] [--verify-hashes]   (13 dimensions + lifecycle + required actions)
  python -m wildspeech v3-checkpoints                    (where each snapshot is / is searched, revision, files)
  python -m wildspeech v3-install [--model K ...] [--execute]      (plan; --execute downloads APPROVED models only)
  python -m wildspeech v3-verify [--model K ...]         (sha256 of every installed file with a published hash)
  python -m wildspeech v3-smoke [--model K ...] [--clips 8]        (few-clip decode per READY model, GPU)
  python -m wildspeech v3-smoke --load-only [--model K ...]       (weights + synthetic-audio forward; no review needed)
  python -m wildspeech v3-readiness                      (reviews -> download auth -> installed -> load -> smoke -> inference)
  python -m wildspeech v3-revisions [--model K ...]      (which cached revision holds which file; offline repair plan)
  python -m wildspeech.asr.reviews record|defer|show|board|commands|proposals|accept-proposal   (the only approvals)
  python -m wildspeech v3-report                         (outputs/V3/scientific_report.html)
"""
from __future__ import annotations

import argparse
import sys


def _ctx(args):
    from .config import parse_set_args
    from .context import build_context

    return build_context(args.profile, overrides=parse_set_args(getattr(args, "set", None)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wildspeech")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="FAST")
        s.add_argument("--stages", nargs="*")
        s.add_argument("--force", nargs="*", default=[])
        s.add_argument("--set", action="append", default=[], help="override config, e.g. gpu.batch_size=32")
    for name in ("status", "cache"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="FAST")
        s.add_argument("--set", action="append", default=[])
    s = sub.add_parser("clear-cache")
    s.add_argument("--family", required=True)
    s.add_argument("--split", default="train", choices=["train", "test"])
    s.add_argument("--profile", default="FAST")
    s = sub.add_parser("invalidate")
    s.add_argument("stages", nargs="+")
    s.add_argument("--profile", default="FAST")
    for name in ("v2-inventory", "v2-report", "v2-registry", "v2-audit"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="V2_CORE")
        s.add_argument("--set", action="append", default=[])
    for name in ("v3-eligibility", "v3-checkpoints", "v3-report", "v3-install", "v3-verify", "v3-smoke",
                 "v3-readiness", "v3-revisions"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="V3_FULL")
        s.add_argument("--set", action="append", default=[])
        if name == "v3-eligibility":
            s.add_argument("--deep", action="store_true", help="also load configs/processors/tokenizers (no weights)")
            s.add_argument("--verify-hashes", action="store_true", help="sha256 every manifest file with a published hash")
        if name in ("v3-install", "v3-verify", "v3-smoke", "v3-revisions"):
            s.add_argument("--model", nargs="*", default=None)
        if name == "v3-install":
            s.add_argument("--execute", action="store_true",
                           help="download / convert (sets ALLOW_MODEL_DOWNLOADS for this call; reviews still required)")
            s.add_argument("--no-verify", action="store_true")
        if name == "v3-smoke":
            s.add_argument("--clips", type=int, default=8)
            s.add_argument("--load-only", action="store_true",
                           help="technical load check on SYNTHETIC audio only (no review needed, no competition data)")
    args = p.parse_args(argv)

    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    ctx = _ctx(args)
    from .orchestrator import Pipeline

    pipe = Pipeline(ctx)
    if args.cmd in ("run", "plan"):
        res = pipe.run(args.stages or None, force=args.force, dry_run=args.cmd == "plan")
        print(res.to_string(index=False))
        return 1 if (res["status"] == "failed").any() else 0
    if args.cmd == "status":
        print(pipe.status_table().to_string(index=False))
        return 0
    if args.cmd == "cache":
        from .acoustic.store import FeatureStore

        print(FeatureStore(ctx.paths.cache / "features").status().to_string(index=False))
        return 0
    if args.cmd == "clear-cache":
        from .acoustic.store import FeatureStore

        FeatureStore(ctx.paths.cache / "features").invalidate(args.family, args.split)
        print(f"cleared {args.family}/{args.split}")
        return 0
    if args.cmd == "invalidate":
        pipe.invalidate(args.stages)
        print("invalidated", args.stages)
        return 0
    if args.cmd == "v2-inventory":
        from .v2.inventory import cache_inventory, stage_inventory

        print(stage_inventory(ctx).to_string(index=False))
        print(cache_inventory(ctx).to_string(index=False))
        return 0
    if args.cmd == "v2-report":
        from .v2.research_report_v2 import build_report

        print(build_report(ctx))
        return 0
    if args.cmd == "v2-registry":
        from .v2.validation_registry import get_registry

        print(get_registry(ctx).summary().groupby("scheme").agg(folds=("fold", "count"), n_train=("n_train", "mean"),
                                                               n_valid=("n_valid", "mean"), overlap=("overlap", "max")).to_string())
        return 0
    if args.cmd == "v2-audit":
        from .diagnostics.scientific_validation import PreviousFindingsAudit

        print(PreviousFindingsAudit(ctx).run().to_string(index=False))
        return 0
    if args.cmd == "v3-install":
        from .asr.model_manager import install, install_plan

        if not args.execute:
            print("DRY RUN (nothing downloaded). Add --execute to install the models whose reviews pass.\n")
            print(install_plan(ctx, args.model).to_string(index=False))
            return 0
        t = install(ctx, args.model, allow=True, verify=not args.no_verify)
        print(t.to_string(index=False) if not t.empty else "no model to install")
        return 1 if (not t.empty and (t["action"] == "failed").any()) else 0
    if args.cmd == "v3-verify":
        from .asr.model_manager import verify_installed

        t = verify_installed(ctx, args.model)
        print(t.to_string(index=False) if not t.empty else "no installed snapshot found")
        return 1 if (not t.empty and ((t["sha_ok"] == False) | (t["size_ok"] == False)).any()) else 0  # noqa: E712
    if args.cmd == "v3-readiness":
        from .v3.readiness import model_readiness, stage_readiness

        pd.set_option("display.max_colwidth", 90)
        print(model_readiness(ctx).drop(columns=["checkpoint_detail", "technical_error"]).to_string(index=False))
        print()
        print(stage_readiness(ctx).to_string(index=False))
        return 0
    if args.cmd == "v3-revisions":
        from .v3.notebook import repair_plan, revision_inventory

        print(revision_inventory(ctx, args.model).to_string(index=False))
        print()
        print(repair_plan(ctx, args.model).to_string(index=False))
        return 0
    if args.cmd == "v3-smoke" and args.load_only:
        from .v3.notebook import load_checks

        t = load_checks(ctx, args.model)
        print(t.to_string(index=False) if not t.empty else "no technically ready model")
        return 1 if (not t.empty and (~t["ok"].astype(bool)).any()) else 0
    if args.cmd == "v3-smoke":
        from .v3.notebook import smoke

        t = smoke(ctx, args.model, args.clips)
        print(t.to_string(index=False) if not t.empty else "no READY_FOR_INFERENCE model")
        return 1 if (not t.empty and (~t["ok"].astype(bool)).any()) else 0
    if args.cmd == "v3-eligibility":
        from .v3.notebook import eligibility_report

        rep = eligibility_report(ctx, deep=bool(getattr(args, "deep", False)),
                                 verify_hashes=bool(getattr(args, "verify_hashes", False)))
        print(rep["summary"].to_string(index=False))
        print("\nDIMENSIONS")
        print(rep["matrix"].to_string())
        if not rep["actions"].empty:
            print("\nREQUIRED ACTIONS")
            print(rep["actions"].sort_values(["priority", "model_key", "step"]).to_string(index=False))
        if not rep["invalid_reviews"].empty:
            print("\nINVALID REVIEW ENTRIES (ignored)")
            print(rep["invalid_reviews"].to_string(index=False))
        return 0
    if args.cmd == "v3-checkpoints":
        from .v3.notebook import installed_checkpoints

        print(installed_checkpoints(ctx).to_string(index=False))
        return 0
    if args.cmd == "v3-report":
        from .v3.notebook import report

        print(report(ctx))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
