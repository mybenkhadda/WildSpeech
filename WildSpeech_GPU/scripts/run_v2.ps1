# WildSpeech V2 launcher (P16-P22), Windows PowerShell 5.1+
#   .\scripts\run_v2.ps1 -RunProfile V2_CORE -Plan
#   .\scripts\run_v2.ps1 -RunProfile V2_FULL
#   .\scripts\run_v2.ps1 -RunProfile V2_CORE -Stages P21 -Force P21
#   .\scripts\run_v2.ps1 -Command v2-inventory
#   .\scripts\run_v2.ps1 -RunProfile V2_SUBMISSION          # only after reviewing P22 (test inference)
param(
    [string]$RunProfile = "V2_CORE",
    [string]$Command = "run",
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

$cmd = if ($Plan) { "plan" } else { $Command }
$argsList = @("-m", "wildspeech", $cmd, "--profile", $RunProfile)
if (($cmd -eq "run" -or $cmd -eq "plan") -and $Stages.Count -gt 0) { $argsList += "--stages"; $argsList += $Stages }
if (($cmd -eq "run" -or $cmd -eq "plan") -and $Force.Count -gt 0) { $argsList += "--force"; $argsList += $Force }
foreach ($s in $Set) { $argsList += "--set"; $argsList += $s }

Write-Host "conda run -n $CondaEnv python $($argsList -join ' ')"
conda run --no-capture-output -n $CondaEnv python @argsList
exit $LASTEXITCODE
