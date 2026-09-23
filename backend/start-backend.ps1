$ErrorActionPreference = "Stop"
$BackendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $BackendRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "backend\.venv is missing or broken. Run powershell -ExecutionPolicy Bypass -File backend\setup-backend.ps1 -SkipModelExport first."
}
$venvConfig = Join-Path $BackendRoot ".venv\pyvenv.cfg"
if (Test-Path -LiteralPath $venvConfig) {
    $basePython = ((Get-Content -LiteralPath $venvConfig | Where-Object { $_ -match '^executable\s*=' } | Select-Object -First 1) -replace '^executable\s*=\s*', '')
    if ($basePython -and -not (Test-Path -LiteralPath $basePython)) {
        throw "backend\.venv points to a missing Python: $basePython. Rebuild it with powershell -ExecutionPolicy Bypass -File backend\setup-backend.ps1 -SkipModelExport."
    }
}

Set-Location $BackendRoot
Write-Host "Starting local Grounding DINO + SAM2 vision service: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Model status: http://127.0.0.1:8000/api/health" -ForegroundColor Cyan
& $Python main.py
