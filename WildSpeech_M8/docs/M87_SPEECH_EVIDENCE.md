# WildSpeech M8.7 — Complete Child Speech Evidence and Outcome Experiments

**Research objective:** determine whether isolating the child's answer and
measuring nuanced phonetic behavior improves future-reading-score prediction
beyond the strongest screening and item-response baseline. This is **new
source code, not a claimed official-data R² improvement** until you execute
the private audio and frozen-fold study.

[Open the 50-cell M8.7 Colab master notebook](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M87_SPEECH_EVIDENCE_MASTER.ipynb)

## Exact inputs and original evidence

- Six official metadata and label CSVs in the original WildSpeech Drive root.
- Training audio 14,704 clips and hidden-test audio 9,814 clips in private
  uncompressed TAR archives (or local directories); filenames must match the
  CSV clip metadata.
- M8.4 frozen original fold SHA-256:
  6ee0f59c8cf5ea3d8f20f5fd500353419587660e00949ac6f1e6a16e092c9ce1.
- M8.4 reference OOF (Ridge mean-fold R² 0.193227; pooled 0.199113).
- M8.6 completed FULL I5 OOF (student pooled R² 0.237018) and I4 OOF
  (school pooled R² 0.170575). The originals remain READ ONLY.
- Raw test audio must NEVER be manually listened to or transcribed.

## Source modules

| Module | Function |
|---|---|
| contracts.py | Unique clip/students, asset eligibility and revision manifest, frozen evaluation hashes |
| audio_io.py | Seekable TAR byte-offset indexing, no private audio extraction, soundfile mono/resample |
| vad_quality.py | Local Silero speech VAD, explicit energy-test fallback, quality/nuisance stats |
| phonetic_asr.py | FP32 Pasketti WavLMForCTC, padding-safe CTC, pooled hidden layers 6–10 |
| word_asr.py | Optional Pasketti Word Qwen3-ASR local vLLM for sentence repetition |
| targets.py | Target expected_text→IPA, token mapping, invalid/unknown target audit |
| ctc_alignment.py | Pure NumPy forward/Viterbi CTC alignment, per-phone confidence, phonetic edit |
| child_localizer.py | Candidate windows, target-vs-free likelihood, timing prior, uncertainty flags |
| task_scorers.py | Graded phone evidence, omission/substitution position, sentence partial credit |
| gpu_cache.py | Resume-safe private shard cache, compressed posteriors and WavLM states |
| clip_models.py | GroupKFold clip correctness and unscored-clip predictions |
| student_aggregation.py | Train-only item normalization, correctness strata, task-aware reliability weighting |
| fold_models.py | A00–A16 nested training models, residual comparison and proper inner-OOF stacking |
| evaluation.py | Original 100 folds, A00–A16 experiment ledger, comparisons and gate decisions |
| localization_qa.py | Stratified human QA template and evaluation TRAIN only |
| stability.py | Small training-only speed/noise perturbation study |
| diagnostics.py | 10+ real-data figures, task/scoring and instrument tables |
| report.py | Exact evidence report, null results and scientific/engineering decision |
| submission.py | Candidate fit, full test singleton and row-order invariance, private SHA CSV |
| cli.py | Preflight, extract, word, QA, stability, evaluate and guarded submission stages |

## What the ASR code actually does

Pasketti-phonetic uses the verified standard WavLMForCTC/Processor interface,
input audio resampled to 16kHz, **FP32 model weights and inference**.
Never use fp16/bfloat16 for the model itself: the official card warns about
relative-position-bias NaNs. Its frame-level log probabilities are converted to
float16 **only after** inference, for compressed private storage. Hidden
WavLM layers 6–10 are mean-pooled on padded-length-corrected hidden frames.
Padded frames do not contribute to the transcript or features.

For clips longer than 30 seconds, deterministic overlapping chunks are
processed and posterior streams approximatively spliced; metadata labels
these recordings as out-of-distribution. No audio is sent to a cloud API.

Pasketti Word is an optional **second process**, because its vLLM/CUDA
requirements (qwen_asr[vllm]==0.0.6, vllm==0.14.0, transformers==4.57.6)
may conflict with a generic Colab runtime. Word scores, token confidence,
and word boundaries are not inferred unless a valid output is actually
provided. The verified transcribe API does not automatically provide
token log-probabilities; unavailable is not zero.

The token registry uses eng_to_ipa plus strict vocabulary mapping.
Out-of-vocabulary phones and invalid G2P outcomes are audited and not
force-aligned. The standalone CTC aligner is independent of removed
torchaudio forced_align APIs.

## Experiment registry (ALL arms present)

| Arm | Research test | Input | Treatment |
|---|---|---|---|
| A00 | Whole-clip phonetic edit reference reimplementation | Whole-clip phone edit | Normal |
| A01 | Whole-clip CTC target and weighted edit | Whole-clip posteriors | Normal |
| A02 | Localized candidate response CTC+edits | Child-selected window | Primary |
| A03 | Candidate response posterior uncertainty | Entropy, margin, phone confidence | Primary |
| A04 | Recording/nuisance-only predictivity | Duration, RMS, VAD, sample rate | **Control only** |
| A05 | Wrong/controlled candidate response | Deterministic random candidate segment | **Control only** |
| A06 | Correctness-conditioned speech features | Within correct/incorrect groups | Primary diagnostic |
| A07 | Sentence repetition partial credit | Pasketti Word word coverage/order | Primary; NEED WORD CACHE |
| A08 | Blending/deletion phone error topology | Omissions, substitutions, order, position | Primary |
| A09 | Graded task phone + word partial | Item-specific partial correctness | Primary |
| A10 | Student-grouped correctness head | Train clip item scores, grouped OOF | Mechanistic |
| A11 | Recovery of unscored items | Clip correctness probabilities | Mechanistic |
| A12 | Reliability weighting | Localization confidence × quality | Aggregation |
| A13 | WavLM hidden layer comparison | Deterministic sketch + train-fold PCA | Representation |
| A14 | Residual correction arm | Cross-fitted I5 residual, fixed shrinkage | Comparison |
| A15 | Random-segment evidence | Random candidate phone statistics | **Control only** |
| A16 | Speech+screening two-head fusion | Inner-OOF stacking | Primary |

The A00 feature is an independent whole-clip reimplementation, NOT the
exact published reference ASR score. A07 will be marked SKIPPED if the
optional Word pipeline has not completed. Skipped experiments cannot be
claimed as tested. A09/A16 may run with phonetic features only; the report
must state if word evidence was absent.

## Leakage protections

- Source hashes and original folds must match M8.4/M8.6 before training.
- Student-level grouping is inherited from the official student ID, never
  random clip splits.
- Clip score models use GroupKFold by student.
- All supervised target fitting is inside the OUTER training students only.
- Three inner folds refit both item/difficulty normalizers and outcome
  transformations; no outer-validation outcomes enter their parameters.
- Level-2 A16 stacking is trained only on inner-OOF predictions, never
  on training-in-sample predictions.
- Test rows are not pooled to learn PCA, item statistics, scalers, prototypes,
  speech class priors or pseudo-labels.
- Training Rasch features remain student-crossfitted as in M8.6.
- All sampling and shuffling controls are restricted to deterministic
  same-clip perturbations or training-only examples.
- Official item_score is an input; future outcome_score is not a clip feature.
- No prompt-conditioned ASR decoding. Only decoded posteriors are compared
  with a target AFTER unconstrained recognition.
- No inference from filenames or CSV row order to prompt chronology.

## Execution protocol

Install the package:

~~~bash
python -m pip install -e './WildSpeech_M8[m87,test]'
~~~

Review the exact Pasketti checkpoint and code license(s) and competition
eligibility first. The code records them as REVIEW_REQUIRED, not
automatically approved. Then set the explicit license-review acknowledgement.

**1. Preflight:**

~~~bash
wildspeech-m87 --stage preflight --drive-root /content/drive/MyDrive
~~~

**2. GPU pilot:**

~~~bash
wildspeech-m87 --stage extract --drive-root /content/drive/MyDrive \
  --train-tar /path/private/train_audio.tar \
  --test-tar /path/private/test_audio.tar \
  --acknowledge-license-review --max-clips 64 --batch-size 2
~~~

**3. GPU full extraction (restartable):**

~~~bash
wildspeech-m87 --stage extract --drive-root /content/drive/MyDrive \
  --train-tar /path/private/train_audio.tar \
  --test-tar /path/private/test_audio.tar \
  --acknowledge-license-review --batch-size 2
~~~

**4. Pasketti Word (separate supported CUDA/vLLM environment):**

~~~bash
python -m pip install 'qwen_asr[vllm]==0.0.6' 'vllm==0.14.0'
wildspeech-m87 --stage word --drive-root /content/drive/MyDrive \
  --train-tar /path/private/train_audio.tar \
  --test-tar /path/private/test_audio.tar \
  --acknowledge-license-review
~~~

**5. Generate training-only QA; fill in human labels on the permitted
training clips:**

~~~bash
wildspeech-m87 --stage qa-template --drive-root /content/drive/MyDrive
wildspeech-m87 --stage qa-review --drive-root /content/drive/MyDrive
~~~

This localizer is NOT verified by simply finding target-like segments;
adult prompts can have the same target words. The QA gate requires
overall accuracy ≥80%, each n≥5 stratum ≥65%, false adult selection ≤10%.
No gating success is asserted until reviewer labels exist.

**6. Frozen-fold experiments:**

~~~bash
wildspeech-m87 --stage evaluate --mode SCREEN \
  --drive-root /content/drive/MyDrive
wildspeech-m87 --stage evaluate --mode FULL \
  --drive-root /content/drive/MyDrive
~~~

FULL uses original 50 student and 50 held-out-school folds, all A00–A16
arms (with explicit SKIPPED status for unavailable Word features), and
three independent inner-training folds per outer fold. The most expensive
part is repeated fold-fitted M8.6 item intelligence; all 17 arms share it
inside each fold. Completed Parquet checkpoints are reused on resume.

**7. Guarded candidate export:**

~~~bash
wildspeech-m87 --stage submit --drive-root /content/drive/MyDrive
~~~

Requires FULL, an eligible non-control model, and human-reviewed
TRAIN localization QA gate. Each test student's prediction must match its
singleton result and be invariant to test-row permutation. Candidate CSV,
model artifact and SHA are PRIVATE. There is no online upload.

## Model promotion

Engineering:
- Student pooled OOF R² versus I5 increases.
- ≥65% of original student fold comparisons improve.
- School-disjoint pooled ΔR² versus I5 ≥−.005.
- No assessment-instrument ΔR² below −.010.

Scientific:
- Mean paired fold ΔR²≥+.010, corrected t≥2, wins≥60%,
  and same school/instrument generalization vetoes.
- Correlated repeated-fold t-statistics are approximate and do not
  correct for selecting among 17 families.

Negative controls A04, A05 and A15 are never promoted. Any speech
improvement must be interpreted alongside within-correctness and
nuisance-only controls; child localization needs independent training QA.

## Visual outputs

The notebook generates **real figures only when the relevant private
experiment has actually executed**:

01. Cache valid/failed/incomplete by task and split.
02. Decoded audio duration distribution.
03. Candidate localization confidence by task.
04. Whole-clip versus selected-window phonetic edits.
05. Whole-to-segment target CTC change.
06. Graded phonetic evidence within correct/incorrect groups.
07. Student and school held-out R² across A00–A16.
08. Paired Δ fold R² versus frozen I5.
09. School-transfer Pareto frontier.
10. I5 score prediction scatter / instrument differences.
11. Optional localization human QA group accuracy figure.
12. Optional training-only example waveform regions (12 plots).
13. Optional perturbation-stability boxplots.

No screenshots of fabricated results or hand-created illustrative score charts
are included. All row-level child data and school tables stay in private Drive.

## Private artifact layout

~~~text
MyDrive/WildSpeech_M87/speech/
  manifests/
    extraction.json
    coverage.json
    model_provenance_PRIVATE.json
    preflight_PRIVATE.json
    target_registry_PRIVATE.parquet
  indexes/train_tar.parquet
  indexes/test_tar.parquet
  shards/00000_phonetic.parquet
  shards/00000_phonetic.npz
  ...
  features/phonetic_clip_features_PRIVATE.parquet
  features/word_asr_PRIVATE.parquet         [optional]
  figures/01_cache_coverage.png ... 06_grading_within_correctness.png
  figures/qa_training/train_clip_*.png    [TRAIN ONLY]
  qa/training_qa_template_PRIVATE.csv
  qa/localization_qa_results.json
  evaluations/screen/
    ledger/, runs/, figures/, tables/, reports/
  evaluations/full/
    ledger/, runs/, figures/, tables/
    reports/M87_DECISION.md
    reports/M87_RESULTS.json
    submission/m87_submission.csv         [only gated]
    submission/m87_predictor.joblib
  stability/stability_train_PRIVATE.csv
  stability/stability_feature_deltas.png
~~~

## Scientific limitations and expected decisions

- Pasketti's reported PER/WER applies to its own evaluation set, not
  automatically to voice-converted WildSpeech recordings.
- Target-based CTC segmentation can select the adult prompt. Child
  localization is an experimental proxy until independently reviewed.
- Clip duration is not response latency.
- IPA forced alignment can find a valid path for a mispronounced or
  irrelevant sequence; compare to free decoding.
- The word model's expected-text hallucination must be controlled by
  unprompted inference and transcript confidence audits.
- Layer 6–10 embeddings use deterministic sketches and train-fold PCA,
  NOT trained student-level networks.
- The 541 target labels are the effective outcome sample size.
- Child audio may be voice-converted; pitch, timbre and identity markers
  are less reliable than speech sequence and content, requiring QA.
- No valid model gain is promised. A well-controlled null result
  is still a real scientific finding, preserving I5/I4 screening.

## Verified external technical references

- Competition rules: https://platform.k12-ai-infrastructure.org/competitions/4/literacy-speech-modeling/rules/
- Reference: https://blog.drivendata.org/blog/literacy-speech-modeling-reference
- Pasketti Phonetic card: https://huggingface.co/drivendata/pasketti-phonetic
- Pasketti Word card: https://huggingface.co/drivendata/pasketti-word
- TorchAudio CTC forced alignment deprecation:
  https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html

Do not report official M8.7 numbers until the full private Drive run ends.

## Fast Colab resume on a second Google account (November 2026)

The master notebook selects `/content/drive/MyDrive/WildSpeech_M87_ACCESS`
**when all six official metadata CSVs are available under**
`WildSpeech_M87_ACCESS/WildSpeech`. Otherwise it selects the original
`/content/drive/MyDrive` layout. The file-shortcut workspace must contain the
frozen M8.4 phase0 and M8.6 FULL checkpoints, as well as both private TARs.
The notebook no longer calls `locate_data_root(DRIVE)` without an explicit
dataset override, preventing false selection of an unrelated same-named folder.

In notebook section 2, review license and competition rules, then set
`LICENSE_ACKNOWLEDGED=True`, `RUN_GPU_EXTRACT=True`, `MODE="FULL"` and
`MAX_CLIPS=0`. Keep `RUN_WORD_ASR=False` until the separate Word runtime is
ready, and `RUN_EVALUATION=False` until all 24,518 clips are accounted for.
`BATCH_SIZE=2` is the FP32 T4-safe baseline; GPU batch-size tuning should
be done by timing real examples and watching VRAM usage rather than assuming
larger batches are faster.

The extraction cell:

- Pins the original model checkpoint SHA from
  `OUTPUT/manifests/extraction.json` if it exists.
- Optionally copies the two private TARs to `/content/wildspeech_audio` when
  2 GiB of additional local free disk space remains after the copy. Do not
  change the source archives or cache fingerprint.
- Optionally stages each lossless compressed `.npz` and Parquet locally in
  `/content/wildspeech_shard_stage`, then uploads both files to the *original*
  private output, before it moves to the next shard. Compression is DEFLATE-1
  for faster serialization, and arrays remain identical when read via NumPy.
- Keeps individual completed shards atomic. Interrupting between shards
  preserves the existing cached UIDs; partially uploaded files are not treated
  as completed.
- Records per-shard decode/VAD/quality/GPU/feature/serialization/upload
  timings at `manifests/timing_profile_PRIVATE.parquet`.
- Bisections FP32 GPU batches when a CUDA OOM occurs, and does not mark
  unrecovered CUDA OOM as a permanently FAILED audio clip.
- Allows partial **descriptive** feature EDA, but the FULL R² evaluation and
  submission gates still demand complete frozen-split inputs.

A 189-shard resume is not a 0-shard restart. The number of *pending* shards
depends on whether the previous shard was fully committed before interruption.
For example, 189 `48`-clip shards account for 9,072 clips, leaving
15,446 clips and 322 additional full/partial shards.

**Throughput caveat:** The workload is sorted by clip duration, and later
shards often contain longer and more expensive recordings. The T4 may remain
GPU bound even after NumPy CTC vectorization and local compression; do not
claim an end-to-end speedup without checking the logged profile. Local
staging uses temporary Colab disk and must be recreated after a runtime
reset; the private Google Drive output is persistent.

The optional Word ASR now checkpoints every 8 inference batches instead of
rewriting the cumulative output file after each batch. Lost work on interruption
is bounded by the checkpoint interval and automatically recomputed on resume.
No private audio is uploaded to external inference services.
