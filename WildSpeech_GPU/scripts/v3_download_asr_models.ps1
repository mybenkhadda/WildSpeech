# WildSpeech V3 — speech-model installer (Windows PowerShell 5.1+). Dry run unless -Execute.
#   .\scripts\v3_download_asr_models.ps1                                  # lifecycle + install plan for every model
#   .\scripts\v3_download_asr_models.ps1 -Model whisper_small_en -Execute # approved models only
#   .\scripts\v3_download_asr_models.ps1 -Execute                         # every model whose reviews pass
#   .\scripts\v3_download_asr_models.ps1 -Verify                          # sha256 of installed files (offline)
# Models with pending reviews are NEVER downloaded; the script prints the review / proposal commands instead.
param(
    [string[]]$Model = @(),
    [switch]$Execute,
    [switch]$Verify,
    [switch]$NoVerify,
    [switch]$WithXet,
    [string]$CondaEnv = "MITSUI"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if ($WithXet) {
    conda run --no-capture-output -n $CondaEnv python -m pip install "hf_xet>=1.1"
}
$argsList = @("scripts\v3_download_asr_models.py")
if ($Model.Count -gt 0) { $argsList += "--model"; $argsList += $Model }
if ($Execute) { $argsList += "--execute" }
if ($Verify) { $argsList += "--verify" }
if ($NoVerify) { $argsList += "--no-verify" }
Write-Host "conda run -n $CondaEnv python $($argsList -join ' ')"
conda run --no-capture-output -n $CondaEnv python @argsList
exit $LASTEXITCODE
