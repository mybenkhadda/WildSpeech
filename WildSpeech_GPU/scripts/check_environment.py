"""Print hardware/CUDA/package report and discovered data files without running any stage.

    python scripts/check_environment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from wildspeech.config import load_config
    from wildspeech.hardware import hardware_report, package_versions
    from wildspeech.paths import ProjectPaths

    rep = hardware_report()
    print(json.dumps(rep, indent=2, default=str))
    missing = [p["package"] for p in package_versions() if not p["available"]]
    print("\nmissing optional/required packages:", missing or "none")
    cfg, _ = load_config("FAST", ROOT / "configs")
    paths = ProjectPaths.build(cfg, root=ROOT)
    print("\ndata directory:", paths.data_dir)
    for k, v in sorted(paths.files.items()):
        print(f"  {k:18s} {v}  ({v.stat().st_size / 1e6:.1f} MB)")
    miss = paths.missing_files(cfg.get_path("data.required_files", {}))
    print("missing data files:", miss or "none")
    return 0 if not miss else 1


if __name__ == "__main__":
    sys.exit(main())
