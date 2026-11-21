"""WildSpeech V3 — child ASR, phonetic intelligence and reading prediction (stages P23-P29).

Infrastructure shared by the V3 stages:
    common     stage start (lineage: stale V3 upstream = error, stale V1/V2 = recorded warning), repeat limits
    registry   V3 registries in outputs/V3 (model registry, phonetic feature registry, hypotheses, experiments,
               ablations, figures)
    viz        figure helpers; every figure is saved with the data table it was drawn from
    report     scientific_report.html built exclusively from runtime tables
    notebook   thin helpers for the master notebook
Nothing executes on import.
"""
V3_STAGE_IDS = ("P23", "P24", "P25", "P26", "P27", "P28", "P29")

# Student-level V3 feature family prefixes (FAMILY__scope__feature__stat)
FAM_WORD = "ASRW"        # word transcripts (ASR01)
FAM_PHON = "ASRP"        # phoneme errors (ASR02)
FAM_CONF = "ASRC"        # expected-phoneme posterior confidence (ASR03)
FAM_TIME = "ASRT"        # phonetic duration / fluency (ASR04)
FAM_TASK = "ASRK"        # task-specific pronunciation errors (ASR05)
FAM_CORR = "ASRO"        # correct-response-only (ASR06)
FAM_DIS = "ASRD"         # ASR model disagreement (ASR07)
FAM_EMB = "ASRE"         # compact pretrained embeddings (ASR08)
FAM_CONS = "ASRS"        # phonetic consistency / stability (ASR09)
FAM_SOFT = "ASRN"        # nested supervised soft correctness from ASR features
FAM_CONFU = "ASRX"       # phoneme-confusion embeddings
FAM_ENC = "ASRF"         # frozen self-supervised encoder embeddings (wav2vec2-base / HuBERT / WavLM) (ASR13)
V3_FAMILIES = (FAM_WORD, FAM_PHON, FAM_CONF, FAM_TIME, FAM_TASK, FAM_CORR, FAM_DIS, FAM_EMB, FAM_CONS, FAM_SOFT, FAM_CONFU,
               FAM_ENC)

NOT_EXECUTED = "NOT_EXECUTED"
ELIGIBLE = "ELIGIBLE"
