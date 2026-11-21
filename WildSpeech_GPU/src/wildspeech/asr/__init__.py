"""Local ASR for V3 (P24): evidence registry, manual reviews, eligibility by dimension, adapters, inference.

    model_registry     configs/v3_asr_models.yaml -> ASRModelSpec (evidence, restrictions, priority) + competition policy
    reviews            configs/v3_asr_reviews.yaml: validated manual review records (the ONLY approvals)
    eligibility        LEGAL / PROVENANCE / CHECKPOINT / FRAMEWORK dimensions -> ELIGIBLE | NOT_EXECUTED + actions
    snapshots          local checkpoint resolution, required files, revision, download commands (never executed)
    environment        package versions, PyTorch CUDA facts, architecture / tokenizer / processor checks
    outputs            real-output detection (P24 provenance + non-empty transcripts/phones), execution state,
                       gates for P25-P29
    base               ASR output schema, chunking + stitching, precision, single-GPU-model slot
    whisper_adapter    Whisper (transformers): transcripts, token log-probabilities, expected-text likelihood
    ctc_adapter        CTC (char / phoneme): posteriors, greedy decoding, forced alignment, embeddings
    pasketti_phonetic  child phoneme recogniser (architecture detected from the local snapshot)
    pasketti_word      child word recogniser (transformers or NeMo; detected from the local snapshot)
    inference          adaptive micro-batching, precision guard, OOM-safe resumable GPU inference
    checkpointing      incremental Parquet checkpoints + provenance manifests
No model is downloaded and no audio leaves the machine. Nothing executes on import.
"""
