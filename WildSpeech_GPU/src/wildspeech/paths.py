"""Repository layout and automatic discovery of the competition files."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

STAGE_NAMES = {
    "P00": "infrastructure", "P01": "tabular_eda", "P02": "acoustic_eda", "P03": "phonetic",
    "P04": "embeddings", "P05": "features", "P06": "phenotypes", "P07": "residual_discovery",
    "P08": "robustness", "P09": "literacy_indices", "P10": "model_benchmark", "P11": "multimodal",
    "P12": "failure_analysis", "P13": "champion", "P14": "submission", "P15": "report",
    # V2 research line (outputs live under outputs/V2/<id>_<NAME>)
    "P16": "harmonization", "P17": "correctness", "P18": "phonetics", "P19": "reference",
    "P20": "embeddings_v2", "P21": "multimodal_v2", "P22": "failure_recovery",
    # V3 research line (outputs live under outputs/V3/<id>_<NAME>)
    "P23": "fusion_audit", "P24": "asr", "P25": "phonetic_features", "P26": "correct_only",
    "P27": "asr_disagreement", "P28": "representations", "P29": "reading_ablation",
}
V2_STAGE_DIRS = {
    "P16": "P16_HARMONIZATION", "P17": "P17_CORRECTNESS", "P18": "P18_PHONETICS", "P19": "P19_REFERENCE",
    "P20": "P20_EMBEDDINGS", "P21": "P21_MULTIMODAL", "P22": "P22_FAILURE_RECOVERY",
}
V3_STAGE_DIRS = {
    "P23": "P23_FUSION_AUDIT", "P24": "P24_ASR", "P25": "P25_PHONETIC_FEATURES", "P26": "P26_CORRECT_ONLY",
    "P27": "P27_ASR_DISAGREEMENT", "P28": "P28_REPRESENTATIONS", "P29": "P29_READING_ABLATION",
}


def _walk(root: Path, max_depth: int):
    root = root.resolve()
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        yield node
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in node.iterdir() if p.is_dir() and not p.name.startswith("."))
        except (PermissionError, OSError):
            continue
        skip = {"cache", "outputs", "models", "experiments", "submissions", "reports", "__pycache__",
                "node_modules", ".git", "src", "notebooks", "tests"}
        stack.extend((c, depth + 1) for c in children if c.name not in skip)


def discover_data_files(roots: list[Path], required: dict[str, str], max_depth: int = 3) -> tuple[dict[str, Path], Path | None]:
    """Find the directory that contains the most required files (ties -> first root order)."""
    best_dir, best_hits = None, -1
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for d in _walk(root, max_depth):
            if d in seen:
                continue
            seen.add(d)
            hits = sum((d / fname).exists() for fname in required.values())
            if hits > best_hits:
                best_dir, best_hits = d, hits
            if hits == len(required):
                return {k: d / v for k, v in required.items()}, d
    if best_dir is None or best_hits <= 0:
        return {}, None
    return {k: best_dir / v for k, v in required.items() if (best_dir / v).exists()}, best_dir


@dataclass
class ProjectPaths:
    root: Path
    data_dir: Path | None
    cache: Path
    outputs: Path
    models: Path
    experiments: Path
    submissions: Path
    reports: Path
    configs: Path
    files: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: Config, root: Path | None = None) -> "ProjectPaths":
        from .config import find_config_dir

        cfg_root = cfg.get_path("paths.root", "auto")
        if root is None:
            root = find_config_dir().parent if cfg_root in (None, "auto") else Path(cfg_root)
        root = Path(root).resolve()
        required = cfg.get_path("data.required_files", {})
        data_dir_cfg = cfg.get_path("paths.data_dir", "auto")
        if data_dir_cfg not in (None, "auto"):
            data_dir = Path(data_dir_cfg)
            if not data_dir.is_absolute():
                data_dir = (root / data_dir).resolve()
            files = {k: data_dir / v for k, v in required.items() if (data_dir / v).exists()}
        else:
            roots = [(root / r).resolve() for r in cfg.get_path("paths.data_search_roots", ["data"])]
            files, data_dir = discover_data_files(roots, required, int(cfg.get_path("paths.data_search_max_depth", 3)))

        def sub(key: str, default: str) -> Path:
            p = Path(cfg.get_path(f"paths.{key}", default))
            return p if p.is_absolute() else root / p

        paths = cls(root=root, data_dir=data_dir, cache=sub("cache_dir", "cache"), outputs=sub("outputs_dir", "outputs"),
                    models=sub("models_dir", "models"), experiments=sub("experiments_dir", "experiments"),
                    submissions=sub("submissions_dir", "submissions"), reports=sub("reports_dir", "reports"),
                    configs=root / "configs", files=files)
        for p in (paths.cache, paths.outputs, paths.models, paths.experiments, paths.submissions, paths.reports):
            p.mkdir(parents=True, exist_ok=True)
        return paths

    @property
    def v2_root(self) -> Path:
        d = self.outputs / "V2"
        (d / "figures").mkdir(parents=True, exist_ok=True)
        return d

    @property
    def v2_cache(self) -> Path:
        """V2 caches are kept apart from cache/features so V1 builders never pick them up."""
        d = self.cache / "v2"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def v3_root(self) -> Path:
        d = self.outputs / "V3"
        (d / "figures").mkdir(parents=True, exist_ok=True)
        return d

    @property
    def v3_cache(self) -> Path:
        """V3 caches (ASR outputs, phonetic clip tables, fold providers, evaluation) live in cache/v3."""
        d = self.cache / "v3"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_dir(self, stage_id: str) -> Path:
        if stage_id in V3_STAGE_DIRS:
            d = self.outputs / "V3" / V3_STAGE_DIRS[stage_id]
        elif stage_id in V2_STAGE_DIRS:
            d = self.outputs / "V2" / V2_STAGE_DIRS[stage_id]
        else:
            d = self.outputs / f"{stage_id}_{STAGE_NAMES.get(stage_id, 'stage')}"
        (d / "figures").mkdir(parents=True, exist_ok=True)
        (d / "tables").mkdir(parents=True, exist_ok=True)
        return d

    def feature_dir(self, family: str, split: str) -> Path:
        d = self.cache / "features" / family.lower() / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def missing_files(self, required: dict[str, str]) -> list[str]:
        return [k for k in required if k not in self.files]
