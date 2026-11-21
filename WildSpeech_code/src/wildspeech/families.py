"""Feature naming conventions and feature-family assignment.

Clip-level prefixes: q_ quality, t_ temporal, s_full_/s_sp_ spectral,
s_bg_ background spectral, m_ MFCC/cepstral, w_ wavelet, p_ prosody,
l_ linguistic item complexity, ph_ phonetic/ASR, e_ embeddings.

Student-level prefixes: theta_ screening, acc_ correctness, irt_ item
difficulty (fold-fitted for prediction), traj_ adaptive behaviour,
meta_ recording metadata/missingness, ia_ item-adjusted audio (fold-fitted),
cx_ complexity sensitivity, cc_ correctness contrasts, xt_ cross-task,
idx_ literacy indices, clu_ cluster features. Aggregated clip features keep
their clip prefix: ``<clip_feature>__<condition>__<stat>``.
"""
from __future__ import annotations

import re

FAMILY_RULES = [
    ("SCREENING", r"^theta_"),
    ("CORRECTNESS", r"^acc_"),
    ("ITEM_DIFFICULTY", r"^irt_"),
    ("ADAPTIVE", r"^traj_"),
    ("RECORDING_METADATA", r"^meta_"),
    ("ITEM_ADJUSTED_AUDIO", r"^ia_"),
    ("COMPLEXITY_SENSITIVITY", r"^cx_"),
    ("CORRECTNESS_CONTRAST", r"^cc_"),
    ("CROSS_TASK", r"^xt_"),
    ("LITERACY_INDICES", r"^idx_"),
    ("CLUSTER", r"^clu_"),
    ("QUALITY", r"^q_"),
    ("BACKGROUND", r"^s_bg_"),
    ("TEMPORAL", r"^t_"),
    ("MFCC", r"^m_"),
    ("SPECTRAL", r"^s_"),
    ("WAVELETS", r"^w_"),
    ("PROSODY", r"^p_"),
    ("LINGUISTIC", r"^l_"),
    ("PHONETIC", r"^ph_"),
    ("EMBEDDINGS", r"^e_"),
]
_COMPILED = [(fam, re.compile(rx)) for fam, rx in FAMILY_RULES]

ID_COLUMNS = {"student_id", "clip_id", "school_id", "instrument", "split", "target", "item_key", "task",
              "task_family", "item_id", "expected_text", "correct", "order", "audio_member", "audio_archive"}

TABULAR_FAMILIES = ["SCREENING", "CORRECTNESS", "ITEM_DIFFICULTY", "ADAPTIVE", "LINGUISTIC"]
AUDIO_FAMILIES = ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC", "ITEM_ADJUSTED_AUDIO",
                  "COMPLEXITY_SENSITIVITY", "CORRECTNESS_CONTRAST", "CROSS_TASK", "EMBEDDINGS"]
TECHNICAL_FAMILIES = ["QUALITY", "BACKGROUND", "RECORDING_METADATA"]

# Families that must be computed inside training folds before predictive use.
FOLD_FITTED_FAMILIES = {"ITEM_DIFFICULTY", "ITEM_ADJUSTED_AUDIO", "LITERACY_INDICES", "CLUSTER"}
FOLD_FITTED_COLUMNS = {"theta_common_factor"}
# Conditional aggregates that rely on all-training item statistics (EDA only).
EDA_ONLY_PATTERNS = [re.compile(r"__diff=")]

VOICE_PRIOR_PATTERNS = [re.compile(p) for p in (
    r"^p_f0", r"^p_f[123]_", r"^p_f2_f1", r"^p_jitter", r"^p_shimmer", r"^p_hnr", r"^p_voiced",
)]

COMPUTATION_COST = {
    "SCREENING": "LOW", "CORRECTNESS": "LOW", "ITEM_DIFFICULTY": "MEDIUM", "ADAPTIVE": "LOW",
    "RECORDING_METADATA": "LOW", "QUALITY": "LOW", "BACKGROUND": "MEDIUM", "TEMPORAL": "LOW",
    "SPECTRAL": "MEDIUM", "MFCC": "MEDIUM", "WAVELETS": "HIGH", "PROSODY": "HIGH",
    "LINGUISTIC": "LOW", "PHONETIC": "VERY_HIGH", "EMBEDDINGS": "VERY_HIGH",
    "ITEM_ADJUSTED_AUDIO": "MEDIUM", "COMPLEXITY_SENSITIVITY": "MEDIUM", "CORRECTNESS_CONTRAST": "LOW",
    "CROSS_TASK": "LOW", "LITERACY_INDICES": "MEDIUM", "CLUSTER": "MEDIUM", "OTHER": "UNKNOWN",
}
INTERPRETABILITY = {
    "SCREENING": "HIGH", "CORRECTNESS": "HIGH", "ITEM_DIFFICULTY": "HIGH", "ADAPTIVE": "HIGH",
    "RECORDING_METADATA": "HIGH", "QUALITY": "HIGH", "BACKGROUND": "MEDIUM", "TEMPORAL": "HIGH",
    "SPECTRAL": "MEDIUM", "MFCC": "LOW", "WAVELETS": "LOW", "PROSODY": "MEDIUM", "LINGUISTIC": "HIGH",
    "PHONETIC": "HIGH", "EMBEDDINGS": "LOW", "ITEM_ADJUSTED_AUDIO": "MEDIUM",
    "COMPLEXITY_SENSITIVITY": "HIGH", "CORRECTNESS_CONTRAST": "HIGH", "CROSS_TASK": "MEDIUM",
    "LITERACY_INDICES": "HIGH", "CLUSTER": "MEDIUM", "OTHER": "UNKNOWN",
}
FAMILY_HYPOTHESIS = {
    "SCREENING": "Initial literacy ability predicts end-of-year reading.",
    "CORRECTNESS": "Item-level accuracy adds detail beyond summary ability estimates.",
    "ITEM_DIFFICULTY": "Difficulty-aware performance captures information beyond raw accuracy (H03, H18).",
    "ADAPTIVE": "Response trajectories carry complementary information (H18).",
    "TEMPORAL": "Response organisation and fluency predict later reading (H01, H14).",
    "SPECTRAL": "Frequency-domain articulation properties carry literacy signal (H01).",
    "MFCC": "Cepstral dynamics reflect articulation precision (H01, H04).",
    "WAVELETS": "Multiscale transients add signal beyond spectral features (H08).",
    "PROSODY": "Prosodic organisation is informative after anonymisation (H01).",
    "LINGUISTIC": "Complexity of attempted items reflects adaptive progression (H03).",
    "PHONETIC": "Expected-versus-recognised speech deviations are informative (H02, H06).",
    "EMBEDDINGS": "Pretrained representations capture complementary information (H09).",
    "ITEM_ADJUSTED_AUDIO": "Item-conditioned acoustics beat unadjusted acoustics (H16).",
    "COMPLEXITY_SENSITIVITY": "Sensitivity to item complexity predicts outcomes (H03).",
    "CORRECTNESS_CONTRAST": "Correct responses carry heterogeneous information (H02).",
    "CROSS_TASK": "Tasks provide complementary information (H05).",
    "QUALITY": "Negative control: technical quality should not predict literacy (H12).",
    "BACKGROUND": "Negative control: background regions should not predict literacy (H12).",
    "RECORDING_METADATA": "Negative control: recording metadata should not predict literacy (H12).",
    "LITERACY_INDICES": "Composite indices improve interpretability or prediction (H17).",
    "CLUSTER": "Distinct profiles exist among similar screening scores (H07).",
    "OTHER": "Unclassified feature.",
}


def feature_family(column: str) -> str:
    for fam, rx in _COMPILED:
        if rx.search(column):
            return fam
    return "OTHER"


def feature_subfamily(column: str) -> str:
    base = column.split("__", 1)[0]
    parts = base.split("_")
    if base.startswith(("s_", "m_")) and len(parts) > 2:
        return "_".join(parts[1:3])
    return parts[1] if len(parts) > 1 else base


def is_feature_column(column: str) -> bool:
    return column not in ID_COLUMNS and feature_family(column) != "OTHER"


def columns_by_family(columns) -> dict:
    out: dict = {}
    for c in columns:
        if c in ID_COLUMNS:
            continue
        out.setdefault(feature_family(c), []).append(c)
    return out


def select_family_columns(columns, families, exclude_eda_only: bool = True) -> list:
    families = set(families)
    out = []
    for c in columns:
        if c in ID_COLUMNS:
            continue
        if feature_family(c) not in families:
            continue
        if exclude_eda_only and any(p.search(c) for p in EDA_ONLY_PATTERNS):
            continue
        out.append(c)
    return out


def modeling_columns(columns, families) -> list:
    """Columns usable directly for prediction (fold-fitted ones are rebuilt in folds)."""
    return [c for c in select_family_columns(columns, families)
            if feature_family(c) not in FOLD_FITTED_FAMILIES and c not in FOLD_FITTED_COLUMNS]


def is_voice_prior(column: str) -> bool:
    base = column.split("__", 1)[0]
    return any(p.search(base) for p in VOICE_PRIOR_PATTERNS)


def sanitize_token(value) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_").lower() or "na"
