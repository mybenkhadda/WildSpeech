# WildSpeech V3 — verify CUDA, ASR eligibility and checkpoint components (Windows PowerShell 5.1+).
#   .\scripts\v3_verify_asr.ps1
#   .\scripts\v3_verify_asr.ps1 -LoadWeights                       # + 1 s smoke decode per ELIGIBLE model
#   .\scripts\v3_verify_asr.ps1 -Model whisper_small_en -LoadWeights
param(
    [string[]]$Model = @(),
    [switch]$LoadWeights,
    [string]$Device = "auto",
    [switch]$AllowCpu,
    [string]$CondaEnv = "MITSUI"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:HF_HUB_OFFLINE = "1"            # verification never touches the network
$env:TRANSFORMERS_OFFLINE = "1"
$argsList = @("scripts\v3_verify_asr.py", "--device", $Device)
if ($Model.Count -gt 0) { $argsList += "--model"; $argsList += $Model }
if ($LoadWeights) { $argsList += "--load-weights" }
if ($AllowCpu) { $argsList += "--allow-cpu" }
Write-Host "conda run -n $CondaEnv python $($argsList -join ' ')"
conda run --no-capture-output -n $CondaEnv python @argsList
exit $LASTEXITCODE
