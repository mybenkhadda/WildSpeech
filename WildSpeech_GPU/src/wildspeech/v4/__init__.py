"""WildSpeech V4 — speech-text-acoustic multimodal intelligence (Google Colab research line).

V4 reuses the V1-V3 package: the competition DataBundle, TAR manifests, acoustic feature caches and extractor, the
item-conditioned residualizer, word/phone alignment, the frozen V2 validation registry, the traced and
fallback-aware V3Evaluator / V3FoldModel, V2 fusion estimators, robustness helpers, figure helpers and submission
validation. What V4 adds:

    config            notebook CONFIG -> configuration overrides, Drive path resolution (file-ID fallback)
    artifacts         output tree, atomic writes, schema versions, fingerprints, resume compatibility
    asr_import        schema inspection + canonical adapter for EXTERNAL Colab ASR exports (provenance
                      `external_colab_asr`, source hash, original row fields, duplicates, failures) and an
                      independent import gate (permission review, coverage, joins) — never relabelled as P24 output
    joins             split + normalized-filename joins of metadata, audio manifests and ASR predictions with
                      one-to-one assertions -> clip_master / student_master / join_audit / data_integrity
    text_features     recognized-text deviation from the expected response (WER/CER/S-I-D, coverage, repetition,
                      task-specific blending / deletion / sentence-repetition deviations, uncertainty) and
                      cross-recognizer disagreement. Recognized-text error is NOT a verified pronunciation error.
    phonetic_features measured phone strings only (no G2P of word transcripts) -> PER, S/D/I, vowel/consonant,
                      position profile, confusions; explicitly UNAVAILABLE when the export has no phones
    acoustic          reuse of cached acoustic families (fingerprint-compatible), optional extraction of what is
                      missing with the existing AcousticExtractionRunner, optional V2 harmonized variants
    aggregation       fold-fitted student-level provider (robust stats, task / difficulty / correct-only,
                      reliability weighting, coverage, cross-task contrasts)
    models            V4FoldModel (+ reliability-gated mixture of experts, compact hierarchical network) and the
                      V4Evaluator (missing-modality / shuffled-ASR controls) on top of V3Evaluator
    experiments       M0-M8, ablations, robustness, champion selection (validated OOF comparisons only)
    hypotheses        H1-H10 evidence table
    importance        grouped permutation importance (fold-safe) + optional SHAP on the fitted champion
    figures           every figure saved as PNG + SVG with its data (CSV + Parquet)
    submission        champion refit on all labelled students, re-load from disk, validated submission
    report            WILDSPEECH_V4_RESEARCH_REPORT.md built from runtime tables only
    pipeline          V4Pipeline: PREPARE / FEATURES / EXPERIMENTS / SUBMISSION / FULL modes with resume
Nothing executes on import.
"""
STAGE = "V4"
SCHEMA_VERSION = "4.0.0"
EXTERNAL_PROVENANCE = "external_colab_asr"

# student-level family prefixes (FAMILY__scope__feature__stat; FAMILY = text before the first "__")
FAM_TEXT = "ASRTXT"       # recognized-text deviation, consensus over usable recognizers
FAM_TASK = "ASRTASK"      # task-specific recognized-text deviations
FAM_UNC = "ASRUNC"        # recognizer uncertainty (only when the export carries confidences)
FAM_DIS = "ASRDIS"        # cross-recognizer disagreement
FAM_PHON = "ASRPHN"       # MEASURED phonetic errors (phone-recognizer output only)
FAM_REL = "ASRREL"        # ASR reliability / coverage (gate features for M7)
FAM_COR = "ASRCOR"        # correct-response-only text + acoustic characteristics (H3)
FAM_ACOU = "ACOU"         # item-conditioned, domain-controlled acoustic features
FAM_REC = "REC"           # recording-conditions control (existing evaluation.robustness.recording_provider)
V4_FAMILIES = (FAM_TEXT, FAM_TASK, FAM_UNC, FAM_DIS, FAM_PHON, FAM_REL, FAM_COR, FAM_ACOU)
ASR_FAMILIES = (FAM_TEXT, FAM_TASK, FAM_UNC, FAM_DIS, FAM_PHON, FAM_REL)

UNAVAILABLE = "UNAVAILABLE"
NOT_EXECUTED = "NOT_EXECUTED"

# columns that exist only in the training split and must never reach a prediction feature
TRAIN_ONLY_FIELDS = ("school_id", "score_status")
