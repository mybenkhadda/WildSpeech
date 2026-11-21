#!/usr/bin/env bash
# WildSpeech V3 — run drivendata/pasketti-word (Qwen3-ASR-1.7B, qwen_asr + vLLM) from WSL2 / Linux.
#
# Why: the card states that transformers alone cannot load this checkpoint; the supported loader is
# qwen_asr[vllm]==0.0.6 with vllm==0.14.0, and vLLM has no native Windows build. The Windows MITSUI env therefore
# reports pasketti_word as FRAMEWORK_INCOMPATIBLE. This script runs ONLY that model in a separate Linux venv, on the
# SAME repository and SAME cache/v3/asr directory (via /mnt/c), so P25-P29 on Windows pick its outputs up.
#
# Prerequisites (you): WSL2 with an NVIDIA driver that exposes CUDA to WSL (nvidia-smi works inside WSL), and the
# pasketti_word reviews approved in configs/v3_asr_reviews.yaml (otherwise eligibility refuses to run it).
#
#   bash scripts/v3_pasketti_word_wsl.sh setup        # create ~/.venvs/wildspeech-qwen3asr (one time)
#   bash scripts/v3_pasketti_word_wsl.sh install      # download the APPROVED checkpoint (pinned commit)
#   bash scripts/v3_pasketti_word_wsl.sh smoke        # few-clip decode
#   bash scripts/v3_pasketti_word_wsl.sh run          # full resumable pass (V3_ASR profile, P24 for this model)
# Nothing here sends audio anywhere: inference is local, HF is used only by `install` for the approved weights.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${WILDSPEECH_QWEN_VENV:-$HOME/.venvs/wildspeech-qwen3asr}"
PY="$VENV/bin/python"
export PYTHONPATH="$REPO/src"
export PYTHONUTF8=1
cmd="${1:-help}"

case "$cmd" in
  setup)
    python3 -m venv "$VENV"
    "$PY" -m pip install --upgrade pip
    # versions verified on the model card (qwen_asr 0.0.6 / vllm 0.14.0 / transformers 4.57.6)
    "$PY" -m pip install "qwen_asr[vllm]==0.0.6" "vllm==0.14.0" "transformers==4.57.6"
    "$PY" -m pip install -e "$REPO" || "$PY" -m pip install pandas pyarrow pyyaml soundfile scipy scikit-learn tqdm psutil
    "$PY" -c "import torch, vllm, qwen_asr; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"
    ;;
  install)
    "$PY" -m wildspeech v3-install --profile V3_ASR --model pasketti_word --execute
    ;;
  smoke)
    HF_HUB_OFFLINE=1 "$PY" -m wildspeech v3-smoke --profile V3_ASR --model pasketti_word --clips 8
    ;;
  run)
    HF_HUB_OFFLINE=1 "$PY" -m wildspeech run --profile V3_ASR --stages P24 --force P24 \
      --set "v3.asr.models=[pasketti_word]"
    ;;
  *)
    sed -n 2,17p "$0"
    ;;
esac
