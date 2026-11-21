"""P20 — frozen speech foundation-model representations (wav2vec2 / HuBERT / WavLM / Whisper encoder / eligible
child-speech encoders): per-layer masked pooling, compact float16 shard storage, fold-fitted compression and
layer-wise discovery. (Named `foundation` because `wildspeech/embeddings.py` — the V1 P04 module — already
occupies the `embeddings` name; this package reuses its loading conventions and model registry.)"""
