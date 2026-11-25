# WildSpeech M8.1 — GitHub-first score-aware speech discovery

A self-contained, installable package inside [`mybenkhadda/WildSpeech`](https://github.com/mybenkhadda/WildSpeech). Reusable implementation is under `src/wildspeech_m8/`; the Colab notebook only orchestrates stages and displays their analyses.

## Open the notebook

[**Open M8.1 in Colab**](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M81_GITHUB_MASTER.ipynb)

The first cell clones `mybenkhadda/WildSpeech` and installs `WildSpeech_M8` in editable mode. No ZIP extraction or manually copying `.py` files is necessary. In Colab, point `DATA_ROOT` at the official WildSpeech data directory in Google Drive, and set `PROFILE_CSV` to the prior M7 profile export if present. You may choose `MODE='SMOKE'` before `MODE='FULL'`.

## Layout

```text
WildSpeech_M8/
  src/wildspeech_m8/
    m8_audio.py         # tar streaming, quality proxies, timing and spectral extraction
    m8_speech.py        # existing local ASR reference comparisons
    m8_phonetic.py      # optional local GPU phonetic hypothesis inference
    m8_pipeline.py      # safe joins, feature transforms, nested CV, residual discovery, prediction
    m8_research.py      # ablations, drift, subgroup/cluster diagnostics, E01-E13 atlas
  notebooks/WILDSPEECH_M81_GITHUB_MASTER.ipynb
  tests/                 # unit, smoke and synthetic full-data integration
  docs/README.md         # detailed methodology and output interpretation
  docs/EXPERIMENT_MATRIX.md
  pyproject.toml
  requirements.txt
  optional-requirements.txt
```

## Run locally

From the repository root:

```bash
python -m pip install -e './WildSpeech_M8[test]'
python -m pytest -q WildSpeech_M8/tests/test_m8.py WildSpeech_M8/tests/test_m81.py
python -m wildspeech_m8.m8_audio --data-root /path/WildSpeech --output-root ./outputs/m8
python -m wildspeech_m8.m8_pipeline --data-root /path/WildSpeech --profile-csv /path/profiles.csv --output-root ./outputs/m8
python -m wildspeech_m8.m8_research --data-root /path/WildSpeech --profile-csv /path/profiles.csv --output-root ./outputs/m8
```

Optional local-only phonetics (needs model license validation and GPU environment):

```bash
python -m pip install -e './WildSpeech_M8[phonetic]'
python -m wildspeech_m8.m8_phonetic --data-root /path/WildSpeech --output-root ./outputs/m8
```

## Outputs

`outputs/m8/tables/` contains validation scores, fold-aligned ablations, conditional associations, group stability, drift, and matched student comparisons. `outputs/m8/figures/` includes the main modeling figures and E01–E13 research atlas. `outputs/m8/RESEARCH_SUMMARY.md` interprets evidence limitations. `outputs/m8/predictions/m8_submission.csv` is written by the full training pipeline.

**Scientific limits:** Audio VAD/quality proxies are not validated speaker diarization; phonetic hypotheses are not ground-truth child phoneme labels. Supervised groups trained on labeled data are descriptive until they generalize to unseen students. A synthetic smoke score is never a competition score. Commit only code and small docs; never commit raw child speech/audio, personal student identifiers, local checkpoint weights or competition outputs.