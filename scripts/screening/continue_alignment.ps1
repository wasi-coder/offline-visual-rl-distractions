$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
$python = 'E:\offline_rl_smoke_test\environment\python\python.exe'
$lock = [IO.File]::Open((Join-Path (Get-Location) 'experiments/screening_20261106/followup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    $active = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match '[\\/]run_screen\.py' }
    if ($active) { Wait-Process -Id $active.ProcessId }
    & $python -u scripts/screening/run_screen.py
    if ($LASTEXITCODE -ne 0) { throw "Screening failed with exit $LASTEXITCODE" }
    & $python -u scripts/screening/analyze.py
    if ($LASTEXITCODE -ne 0) { throw "Screen analysis failed with exit $LASTEXITCODE" }
    & $python -u scripts/screening/long_alignment.py
    if ($LASTEXITCODE -ne 0) { throw "Long follow-up failed with exit $LASTEXITCODE" }
} finally {
    $lock.Dispose()
}
