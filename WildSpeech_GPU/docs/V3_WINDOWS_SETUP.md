# WildSpeech V3 — Windows execution (P23–P29)

Run everything yourself, in the `MITSUI` conda env, from `C:\Users\myben\Downloads\WildSpeech_P01\WildSpeech_GPU`.

## 1. Environment (torch 2.5.1+cu124 is kept)
```powershell
.\scripts\v3_install_asr_env.ps1 -CheckOnly        # versions, CUDA, bin_weights_loadable
.\scripts\v3_install_asr_env.ps1 [-WithXet]        # ASR packages; the installed torch/torchaudio are pinned (-c)
```
transformers 4.57 refuses `torch.load` of `.bin` weights on torch < 2.6. The four `.bin`-only models
(espeak phoneme, wav2vec2-base, HuBERT, WavLM-base-plus) are therefore converted at install time to a local
safetensors copy, after their sha256 has been checked against the official LFS hash. torch is not upgraded.

## 2. Eligibility: 13 dimensions → lifecycle
```powershell
python -m wildspeech v3-eligibility --deep          # lifecycle, dimension matrix, required actions
```
Dimensions: registry_validity, license_documentation, license_approval, training_provenance, overlap_assessment,
competition_permission, checkpoint_availability, checkpoint_files, architecture_compatibility,
processor_compatibility, framework_dependencies, gpu_readiness, inference_readiness.

Evidence was assembled on 2026-11-21 from the HF API, the model cards and the competition pages. See
`configs/v3_asr_models.yaml` (facts) and `outputs/V3/model_evidence.json`.

| model | license | competition | open question |
|---|---|---|---|
| whisper_small_en, wav2vec2_* , hubert_base | apache-2.0 | named / general rule | none: drafts ready |
| pasketti_phonetic / pasketti_word | bigcode-openrail-m (conditional, not OSI) | named | **ReadNet (Reach Every Reader data) is a training corpus → likely overlap**; license interpretation |
| wavlm_base_plus | card has none; README → UniSpeech CC BY-SA 3.0 | general rule | which license governs the weights |
| pasketti_word | — | — | Qwen3-ASR needs qwen_asr + vLLM → Linux / WSL2 only |

## 3. Approve (named human reviewer only)
```powershell
python -m wildspeech.asr.reviews proposals
python -m wildspeech.asr.reviews accept-proposal --id P-WSP-LP P-WSP-OV P-ESP-LP P-ESP-OV P-B960-LP P-B960-OV P-W2VB-LP P-W2VB-OV P-HUB-LP P-HUB-OV --reviewer "<your full name>"
# Pasketti / WavLM drafts are ready:false -> decide, then:
python -m wildspeech.asr.reviews accept-proposal --id P-PSKPH-L P-PSKPH-P P-PSKPH-OV --reviewer "<name>" --resolved-open-questions --mitigation "<your words>"
python -m wildspeech.asr.reviews show
```
Overlap approvals need a positive basis. `no_known_overlap` is refused.

## 4. Install (approved models only)
```powershell
python -m wildspeech v3-install                     # plan: storage, missing files, GB, disk, conversions
python -m wildspeech v3-install --execute           # = ALLOW_MODEL_DOWNLOADS for this call
python -m wildspeech v3-verify                      # sha256 of every installed weight file
```
- Complete HF-cache snapshots at the pinned commit are reused in place (wav2vec2-base-960h is already cached).
- Partial cache snapshots are completed inside the cache.
- Every other model goes to `models/<dir>/` as real files (no symlinks).
- Downloads are resumable.
- The result is recorded in `models/model_installation_manifest.json`.

## 5. Smoke test, then the pipeline
```powershell
python -m wildspeech v3-smoke --clips 8             # few-clip GPU decode per READY_FOR_INFERENCE model
python -m wildspeech run --profile V3_SMOKE         # small end-to-end pass
python -m wildspeech run --profile V3_FULL          # P23 -> P29 (P24 resumes; Ctrl+C is safe)
python -m wildspeech v3-report
```
Notebook: `notebooks/WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER.ipynb`. Set `RUN_PIPELINE`,
`ALLOW_MODEL_DOWNLOADS` and `RUN_SMOKE` in cell 0.

## 6. pasketti_word under WSL2
```bash
bash scripts/v3_pasketti_word_wsl.sh setup && bash scripts/v3_pasketti_word_wsl.sh install
bash scripts/v3_pasketti_word_wsl.sh smoke && bash scripts/v3_pasketti_word_wsl.sh run
```
This writes to the same `cache/v3/asr/pasketti_word`, so P25–P29 on Windows use it afterwards.
