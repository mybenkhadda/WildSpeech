# WildSpeech V3 — execution runbook (after the 2026-11-21 blocker repair)

Everything below is run **by you**, from `WildSpeech_GPU/` in the `MITSUI` env. Nothing was executed when this was written.

## 0. Notebook
The repaired notebook is `notebooks/WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER_REPAIRED.ipynb` (clean: no outputs).
The executed original was left untouched. When you're happy, replace it:
```powershell
Move-Item notebooks\WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER.ipynb notebooks\archive_WILDSPEECH_V3_MASTER_executed_2026-11-21.ipynb
Move-Item notebooks\WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER_REPAIRED.ipynb notebooks\WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER.ipynb
```
Controls (cell 0): `RUN_PIPELINE`, `ALLOW_MODEL_DOWNLOADS`, `RUN_SMOKE`, `FORCE` (all `False`), plus `FORCE_STAGES` and `DOWNLOAD_MODELS`.

## 1. Tests
```powershell
python -m pytest tests/test_v3_blockers.py -q
python -m pytest tests -q -k v3          # older V3 tests may be stale against the schema-3 APIs
```

## 2. Reviews (the only approvals)
```powershell
python -m wildspeech.asr.reviews board        # per model / check: decision, open restrictions, drafts
python -m wildspeech.asr.reviews commands     # exact accept / reject / defer lines for every open check
```
* Drafts that are `ready: true` (read the cited sources at the pinned revision first):
  `accept-proposal --id P-B960-LP P-B960-OV`, `P-WSP-LP P-WSP-OV`, `P-ESP-LP P-ESP-OV`, `P-W2VB-LP P-W2VB-OV`, `P-HUB-LP P-HUB-OV`, `P-WLM-OV`.
* **Pasketti (both models):** ReadNet = Reach Every Reader data is a named training corpus, so overlap is likely. You choose:
  accept (`--resolved-open-questions --mitigation "<yours>"`), `record --decision rejected`, or `defer --outstanding "<question>"`.
  Pasketti-word also needs the vLLM / qwen_asr license question settled and runs only under WSL2.
* **wavlm_base_plus:** the license that governs the weights is unresolved (`P-WLM-LP`, `ready: false`). Accept, reject or defer it.

## 3. Install (approved models only)
```powershell
python -m wildspeech v3-revisions                     # which cached revision holds which file; offline repair plan
python -m wildspeech v3-install                       # dry run
python -m wildspeech v3-install --model wav2vec2_base_960h whisper_small_en --execute
```
* `wav2vec2_base_960h` is complete in the HF cache at the pinned commit. Nothing is downloaded; it's used in place.
* `wav2vec2_base`: the pinned snapshot has `config.json` and `pytorch_model.bin` but is missing `preprocessor_config.json` (159 B).
  The `refs/pr/11` snapshot is never used. The install fetches that one file at the pinned commit, verifies the `.bin` sha256 and
  converts it to safetensors in `models/wav2vec2_base/`.
* LFS files cached under any revision are linked into the pinned snapshot only when the blob's sha256 equals the registry hash.
* Manual fallback per model: the `huggingface_cli` column of `python -m wildspeech v3-eligibility` (or `download_commands.csv` in P24).

## 4. Technical load check, then smoke decode
```powershell
python -m wildspeech v3-smoke --load-only            # weights + synthetic audio; works before reviews pass
python -m wildspeech v3-smoke --clips 8              # training clips; eligible models only
python -m wildspeech v3-readiness                    # reviews -> download auth -> installed -> load -> smoke -> inference
```

## 5. Pipeline
```powershell
python -m wildspeech run --profile V3_FULL --stages P24
python -m wildspeech run --profile V3_FULL --stages P25 P26 P27 P28 P29
```
P23 is protected (`v3.orchestration.protected_stages`). It's recomputed only with `--force P23`.

## Expected artefacts
| stage | key outputs |
|---|---|
| P24 | `outputs/V3/P24_*/tables/model_readiness.csv`, `revision_inventory.csv`, `repair_plan.csv`, `model_eligibility.csv`, `install_results.csv`, `smoke_tests.csv`, `inference_runs.csv`, `throughput.csv`, `transcription_examples.csv`, `p24_outcome.json`; `cache/v3/asr/<model>/train/part-*.parquet`, `cache/v3/asr_emb/...`; `cache/v3/asr_load_checks/<model>.json`; `models/model_installation_manifest.json` |
| P25 | `feature_tables`, `correctness_validation_auc`, `child_vs_general_auc`, `confusion_long_<model>`, `manual_review_sheet`; confusion / error-heatmap / alignment figures |
| P26–P28 | as before. A stage without real inputs writes `not_executed.csv` + `_outcome.json` (never `completed`) |
| P29 | `ablation_results` (+ `result_class`), `result_classes`, `baseline_reference_check` (vs P10 GAM 0.2153), `overlap_sensitivity`, `champion_comparison`, `outputs/V3/v3_champion.json` |
