# WildSpeech M8.1 — GitHub-first score-aware speech discovery

A self-contained, installable package inside [`mybenkhadda/WildSpeech`](https://github.com/mybenkhadda/WildSpeech). Reusable implementation is under `src/wildspeech_m8/`; the Colab notebook only orchestrates stages and displays their analyses.

## Open the notebook

[**Open M8.1 in Colab**](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M81_GITHUB_MASTER.ipynb)

The setup cells clone `mybenkhadda/WildSpeech`, install `WildSpeech_M8` in editable mode, and add its `src/` directory to the current Colab kernel's import path (**no runtime restart needed**). The first import cell also repairs a missing package automatically. No ZIP extraction or manually copying `.py` files is necessary. In Colab, point `DATA_ROOT` at the official WildSpeech data directory in Google Drive, and set `PROFILE_CSV` to the prior M7 profile export if present. You may choose `MODE='SMOKE'` before `MODE='FULL'`.

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
## Troubleshooting: `ModuleNotFoundError: No module named 'wildspeech_m8'`

This was fixed in the GitHub notebook in commit `44bd650a`. If using a personal Colab Drive copy, that copy **does not update automatically** when GitHub changes; reopen the [GitHub notebook](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M81_GITHUB_MASTER.ipynb), or insert this before the first `wildspeech_m8` import:

```python
import sys, subprocess
from pathlib import Path
p = Path('/content/WildSpeech_git/WildSpeech_M8')
if not (p / 'pyproject.toml').is_file():
    subprocess.check_call(['git','clone','--depth','1',
        'https://github.com/mybenkhadda/WildSpeech.git','/content/WildSpeech_git'])
subprocess.check_call([sys.executable,'-m','pip','install','-q','-e',str(p)])
src = str(p.resolve() / 'src')
if src not in sys.path:
    sys.path.insert(0, src)
from wildspeech_m8.m8_pipeline import Config
print('M8 import successful')
```

If you see missing `DATA_ROOT`, `OUTPUT_ROOT`, or `PROFILE_CSV` after that, run the initial environment-and-paths cell. Never install a non-existent PyPI package named `wildspeech_m8` as a workaround; install the checked-out local project.

## M8.2 — Pairwise consensus and behavioral twins (new)

[**Open M8.2 in Google Colab**](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M82_PAIRWISE_MASTER.ipynb)

M8.2 reuses the M8.1 **student-level screening, acoustic and phonetic feature caches**. It never downloads or re-extracts raw audio. Modules are in `src/wildspeech_m8/m82_*.py`; the notebook orchestrates them via an installed editable package.

- Outcome-blind K-Means, Gaussian-mixture and Ward ensemble clustering across independently defined views and sample/feature perturbations.
- Coverage-aware co-association over all training-student pairs, balanced over screening, task behavior, acoustic and phonetic domains. Derived conditional speech is not given an extra independent vote.
- Actual cluster-size chance correction; robust together/apart, conflicting-view and under-covered pair categories.
- Reading-score gaps and **cross-fitted** screening residual gaps analyzed **after** unsupervised consensus construction.
- School, recording-quality and task-content confounder audits; school identifiers never enter clustering inputs.
- Conservative inductive prototype-based residual-correction model evaluated against the same global CatBoost baseline folds.
- Sixteen (or more, depending on available views) exploratory figures in an HTML research atlas; full pairs exported in Parquet or compressed CSV.

Run locally from `WildSpeech_M8/` after `pip install -e '.[test]'`:

```bash
python -m wildspeech_m8.m82_pipeline --config configs/m82_experiments.yaml --mode SMOKE
python -m wildspeech_m8.m82_pipeline --config configs/m82_experiments.yaml --mode FULL
python -m pytest -q tests/test_m8.py tests/test_m81.py tests/test_m82.py
```

The M8.2 results appear in `<M8_OUTPUT_ROOT>/m82/`, independently of M8.1 outputs. Detailed methodology is in [M82_PAIRWISE_EXPERIMENTS.md](docs/M82_PAIRWISE_EXPERIMENTS.md).

**Validation status:** 18 local unit tests and both smoke/full-mode synthetic integration runs passed. The official competition-data M8.2 run and any leaderboard gain remain unverified. Synthetic fixture plots are not official results.


## M8.3 — Exhaustive bottleneck research, Drive-first

[**Open the 60-cell M8.3 research notebook directly in Google Colab**](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M83_BOTTLENECK_MASTER.ipynb)

The notebook mounts Google Drive **before** loading data; discovers and validates all six official CSVs; selects an existing M7 profile and the M8 audio/speech/phonetic feature caches; reports missing/partial caches rather than silently interpreting metadata-only runs; executes matching-fold OOF ablations, random-vs-school CV, nested residual correction, negative controls and learning curves; and generates **18+ research figures, an HTML atlas, a figure index, diagnostics, and a ranked bottleneck scorecard**. It does not download or re-extract raw audio. Outputs stay private under `MyDrive/WildSpeech_M8/m83_bottleneck` or the discovered M8 output root.

Run Colab in **SMOKE** first. Change `MODE="FULL"` in the notebook config to execute the full experiment. Full mode requires a complete audio cache by default; the M7 profile is strongly recommended. No official-data bottleneck conclusions can be claimed until the notebook executes on real data.

Local command: `python -m wildspeech_m8.m83_bottleneck --data-root /data/WildSpeech --output-root /data/WildSpeech_M8 --folds 5 --repeats 2`.

Validation: `python -m pytest -q WildSpeech_M8/tests/test_m83.py WildSpeech_M8/tests/test_m83_notebook.py`.

More: [M83_BOTTLENECK.md](docs/M83_BOTTLENECK.md).


## M8.4 — Frozen evaluation harness and DrivenData reference parity

[**Open the M8.4 Phase 0 notebook in Colab**](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M84_PHASE0_MASTER.ipynb)

A new `wildspeech_m84` package implements a strict official-data contract, 14-feature RidgeCV reference, 5×10 student CV plus genuinely randomized 5×10 school CV, immutable fold SHA and run ledger, held-out R² comparisons, provenance registry, synthetic CI checks, and fit-train-only reference submission export with singleton/batch invariance. The private executed report appears at `MyDrive/WildSpeech_M84/phase0/parity/M84_BASELINE_PARITY.md`. See [full documentation](docs/M84_PHASE0.md).

```bash
pip install -e './WildSpeech_M8[test]'
python -m wildspeech_m84.runner --drive-root /content/drive/MyDrive
pytest -q WildSpeech_M8/tests/test_m84_*.py
```

**Status:** Committed code is not itself official-data parity evidence. The reference's published mean-fold R²≈0.193 and public R²≈0.2888 are external benchmarks, not new M8.4 results. Exact M7.3 source is missing from main; historic M8.3 feature bank comparisons are research-only until a train-only compliance audit. Do not upload child-level data, labels, school IDs, OOF files, private models or audio to GitHub.
