# WildSpeech pipeline launcher (Windows PowerShell 5.1+)
#   .\scripts\run_pipeline.ps1 -RunProfile SMOKE
#   .\scripts\run_pipeline.ps1 -RunProfile FAST -Stages P02,P07 -Force P07
#   .\scripts\run_pipeline.ps1 -RunProfile FAST -Set "gpu.batch_size=32","audio.enable_pitch=true"
#   .\scripts\run_pipeline.ps1 -RunProfile RESEARCH -Plan
param(
    [string]$RunProfile = "FAST",
    [string[]]$Stages = @(),
    [string[]]$Force = @(),
    [string[]]$Set = @(),
    [string]$CondaEnv = "MITSUI",
    [switch]$Plan
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$cmd = if ($Plan) { "plan" } else { "run" }
$argsList = @("-m", "wildspeech", $cmd, "--profile", $RunProfile)
if ($Stages.Count -gt 0) { $argsList += "--stages"; $argsList += $Stages }
if ($Force.Count -gt 0) { $argsList += "--force"; $argsList += $Force }
foreach ($s in $Set) { $argsList += "--set"; $argsList += $s }

Write-Host "conda run -n $CondaEnv python $($argsList -join ' ')"
conda run --no-capture-output -n $CondaEnv python @argsList
exit $LASTEXITCODE
