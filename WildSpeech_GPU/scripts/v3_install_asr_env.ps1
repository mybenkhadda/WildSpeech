# WildSpeech V3 — install / check the local-ASR packages in the Conda env (Windows PowerShell 5.1+).
# Nothing runs automatically: YOU run this script. It never downloads model checkpoints.
#
#   .\scripts\v3_install_asr_env.ps1 -CheckOnly                 # report versions + CUDA, change nothing
#   .\scripts\v3_install_asr_env.ps1                            # pip-install the ASR packages into MITSUI
#   .\scripts\v3_install_asr_env.ps1 -InstallTorch -CudaTag cu121   # ALSO (re)install CUDA torch + torchaudio
#   .\scripts\v3_install_asr_env.ps1 -WithNemo                  # only if a Pasketti snapshot is a .nemo model
#   .\scripts\v3_install_asr_env.ps1 -WithXet                   # optional hf_xet download acceleration
# The working CUDA torch (2.5.1+cu124 on MITSUI) is PINNED with a pip constraints file, so installing the ASR
# packages can never replace it. Do not use -InstallTorch unless CUDA torch is actually missing.
# pasketti_word (Qwen3-ASR via qwen_asr + vLLM) is Linux-only: scripts/v3_pasketti_word_wsl.sh (WSL2).
param(
    [string]$CondaEnv = "MITSUI",
    [switch]$CheckOnly,
    [switch]$InstallTorch,
    [string]$CudaTag = "cu121",        # RTX 3070 (Ampere, sm_86): cu118 / cu121 / cu124 wheels all work
    [switch]$WithNemo,
    [switch]$WithXet
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Invoke-EnvPython([string[]]$PyArgs) {
    Write-Host ">> conda run -n $CondaEnv python $($PyArgs -join ' ')"
    conda run --no-capture-output -n $CondaEnv python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "python exited with $LASTEXITCODE" }
}

$probe = @'
import importlib, json
out = {}
for n in ("torch", "torchaudio", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_xet",
          "soundfile", "g2p_en", "nemo", "qwen_asr", "vllm"):
    try:
        m = importlib.import_module(n); out[n] = getattr(m, "__version__", "?")
    except Exception as e:
        out[n] = "MISSING (" + type(e).__name__ + ")"
try:
    import torch
    out["cuda_available"] = torch.cuda.is_available()
    out["torch_cuda_build"] = torch.version.cuda
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = p.name; out["vram_gb"] = round(p.total_memory / 1e9, 2)
        out["compute_capability"] = f"{p.major}.{p.minor}"; out["bf16"] = torch.cuda.is_bf16_supported()
except Exception as e:
    out["cuda_available"] = False; out["torch_error"] = str(e)
try:
    tv = tuple(int(x) for x in out["torch"].split("+")[0].split(".")[:2])
    xv = tuple(int(x) for x in out["transformers"].split(".")[:2])
    out["bin_weights_loadable"] = not (xv >= (4, 50) and tv < (2, 6))
    if not out["bin_weights_loadable"]:
        out["bin_weights_note"] = ("transformers refuses torch.load(.bin) on torch < 2.6: the model manager converts "
                                   "sha256-verified .bin checkpoints to safetensors (no torch upgrade needed)")
except Exception:
    pass
print(json.dumps(out, indent=2))
'@
$probeFile = Join-Path $env:TEMP "wildspeech_v3_env_probe.py"
Set-Content -Path $probeFile -Value $probe -Encoding UTF8

Write-Host "=== current environment ($CondaEnv)"
Invoke-EnvPython @($probeFile)
if ($CheckOnly) { exit 0 }

if ($InstallTorch) {
    Write-Host "=== installing CUDA PyTorch ($CudaTag) — torch and torchaudio from the same index so versions match"
    conda run --no-capture-output -n $CondaEnv python -m pip install --upgrade torch torchaudio --index-url "https://download.pytorch.org/whl/$CudaTag"
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
}

Write-Host "=== pinning the installed torch / torchaudio (never replaced by the ASR packages)"
$pinFile = Join-Path $env:TEMP "wildspeech_v3_torch_pin.txt"
conda run --no-capture-output -n $CondaEnv python -c "import torch, torchaudio; print(f'torch=={torch.__version__}'); print(f'torchaudio=={torchaudio.__version__}')" | Out-File -FilePath $pinFile -Encoding ascii
Get-Content $pinFile
Write-Host "=== installing ASR packages"
# transformers >= 4.38: Whisper attention masks / compute_transition_scores; huggingface_hub for snapshots
conda run --no-capture-output -n $CondaEnv python -m pip install -c $pinFile "transformers>=4.38,<5" "accelerate>=0.30" "huggingface_hub>=0.20,<1.0" "safetensors>=0.4" "soundfile>=0.12" "tqdm>=4.66" "psutil>=5.9" "g2p_en>=2.1"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
if ($WithXet) {
    conda run --no-capture-output -n $CondaEnv python -m pip install -c $pinFile "hf_xet>=1.1"
    if ($LASTEXITCODE -ne 0) { throw "hf_xet install failed" }
}
# g2p_en needs the NLTK tagger data (local download of NLTK data, not a model checkpoint)
conda run --no-capture-output -n $CondaEnv python -c "import nltk; [nltk.download(p, quiet=True) for p in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict')]"

if ($WithNemo) {
    Write-Host "=== installing NeMo ASR (large; only for .nemo checkpoints)"
    conda run --no-capture-output -n $CondaEnv python -m pip install -c $pinFile "nemo_toolkit[asr]"
    if ($LASTEXITCODE -ne 0) { throw "nemo install failed" }
}

Write-Host "=== environment after install"
Invoke-EnvPython @($probeFile)
Write-Host "next: python -m wildspeech v3-eligibility --deep   (lifecycle per model + required review actions)"
Write-Host "      .\scripts\v3_download_asr_models.ps1   (install plan; add -Execute to install APPROVED models)"
Write-Host "      .\scripts\v3_verify_asr.ps1            (CUDA + eligibility + tokenizer/processor checks)"
