# Unit tests on synthetic data (no competition data, no GPU needed; GPU-only tests auto-skip).
#   .\scripts\run_tests.ps1            all tests
#   .\scripts\run_tests.ps1 -K gpu     subset by keyword
param([string]$K = "", [string]$CondaEnv = "MITSUI")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$a = @("-m", "pytest", "tests")
if ($K) { $a += "-k"; $a += $K }
conda run --no-capture-output -n $CondaEnv python @a
exit $LASTEXITCODE
