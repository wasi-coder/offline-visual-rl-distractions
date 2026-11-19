$ErrorActionPreference = 'Stop'
$repo = 'E:\offline_rl'
$script = Join-Path $repo 'scripts\gating_svea\run.py'
$out = Join-Path $repo 'experiments\gating_svea_20261112'
$mutex = [Threading.Mutex]::new($false, 'Local\OfflineRLGatingSVEA')
$locked = $false
try {
    try { $locked = $mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $locked = $true }
    if (!$locked -or (Test-Path -LiteralPath (Join-Path $out 'REPORT.md'))) { return }
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'scripts[/\\]gating_svea[/\\]run\.py' }
    if ($running) { return }
    Set-Location -LiteralPath $repo
    $stamp = Get-Date -Format yyyyMMdd_HHmmss
    $worker = Start-Process -FilePath 'E:\offline_rl_smoke_test\environment\python\python.exe' `
        -ArgumentList '-u', $script -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $out "resume_$stamp.log") `
        -RedirectStandardError (Join-Path $out "resume_${stamp}_error.log")
    $worker.WaitForExit()
    exit $worker.ExitCode
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
