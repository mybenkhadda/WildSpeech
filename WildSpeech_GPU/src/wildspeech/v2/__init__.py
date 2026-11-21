"""WildSpeech V2 — robust, phonetic and expected-text-conditioned speech intelligence (stages P16-P22).

Core infrastructure shared by the V2 stages:
    validation_registry  frozen student-disjoint fold registry (+ equal-size transfer controls, nested inner folds)
    lineage              stale-upstream detection that blocks invalid comparisons
    audio_pass           bounded-memory TAR -> duration-aware GPU batch engine with OOM recovery and resume
    registry             V2 registries (features, hypotheses, experiments, models, ablations, failures, figures)
    scheduler            budget-controlled experiment scheduling
    baseline             strongest validated tabular comparator
    inventory            artefact inventory across P00-P22
    visualization_v2     figure helpers (one consistent style, figure + data table)
    research_report_v2   findings derived exclusively from runtime tables -> outputs/V2/scientific_report.html
    notebook             thin notebook helpers
Nothing executes on import.
"""
V2_STAGE_IDS = ("P16", "P17", "P18", "P19", "P20", "P21", "P22")
V2_RUN_ORDER = ("P16", "P18", "P20", "P17", "P19", "P21", "P22")

# Student-level feature family prefixes produced by V2 providers (FAMILY__scope__feature__stat).
FAM_HARM = "HARM"
FAM_SOFT = "SOFT"
FAM_PHON = "PHON"
FAM_REF = "REF"
FAM_FM = "FM"
FAM_IDX = "IDX"
FAM_CPX = "CPX"
V2_FAMILIES = (FAM_HARM, FAM_SOFT, FAM_PHON, FAM_REF, FAM_FM, FAM_IDX, FAM_CPX)
TABULAR_FAMILIES = ("SCREENING", "RECORDING", "BEHAVIOR", "ITEM", "LINGUISTIC")
