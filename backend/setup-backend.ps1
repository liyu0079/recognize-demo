param(
    [switch]$SkipModelExport,
    [switch]$SkipModelDownload,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
# Only change execution policy for this PowerShell process.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$SkipModelExport = $SkipModelExport -or ($RemainingArgs -contains "--SkipModelExport")
$SkipModelDownload = $SkipModelDownload -or ($RemainingArgs -contains "--SkipModelDownload")
$BackendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $BackendRoot ".venv"
$Python = Join-Path $VenvRoot "Scripts\python.exe"

function Refresh-ProcessPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $nodeHome = Join-Path $env:ProgramFiles "nodejs"
    if (Test-Path -LiteralPath $nodeHome) { $env:Path = "$nodeHome;$env:Path" }
}

function Ensure-FrontendToolchain {
    Refresh-ProcessPath
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $winget) { throw "Node.js and winget were not found. Install Node.js 20 LTS and retry." }
        Write-Host "[0/4] Installing Node.js LTS" -ForegroundColor Cyan
        & $winget.Source install --id OpenJS.NodeJS.LTS --exact --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "Node.js LTS installation failed." }
        Refresh-ProcessPath
    }
    $corepack = Get-Command corepack -ErrorAction SilentlyContinue
    if (-not $corepack) { throw "Node.js is installed but corepack is missing. Install the full Node.js 20 LTS package." }
    & $corepack.Source enable
    if ($LASTEXITCODE -ne 0) { throw "corepack enable failed." }
    & $corepack.Source prepare pnpm@9.15.4 --activate
    if ($LASTEXITCODE -ne 0) { throw "pnpm installation failed." }
    Refresh-ProcessPath
    & $corepack.Source pnpm --version
    if ($LASTEXITCODE -ne 0) { throw "pnpm verification failed." }
}

Ensure-FrontendToolchain

function Get-Python311 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        try { & $launcher.Source -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,11) else 1)" 2>$null; if ($LASTEXITCODE -eq 0) { return @($launcher.Source, '-3.11') } } catch {}
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        try { & $command.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,11) else 1)" 2>$null; if ($LASTEXITCODE -eq 0) { return @($command.Source) } } catch {}
    }
    return $null
}

function Ensure-Python311 {
    $found = Get-Python311
    if ($found) { return $found }
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { throw "Python 3.11 and winget were not found. Install Python 3.11 from python.org and retry." }
    Write-Host "[0/4] Installing Python 3.11" -ForegroundColor Cyan
    & $winget.Source install --id Python.Python.3.11 --exact --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 installation failed." }
    Refresh-ProcessPath
    $found = Get-Python311
    if (-not $found) { throw "Python 3.11 was installed but is not on PATH. Restart PowerShell and retry." }
    return $found
}

function Test-WorkingVenv {
    if (-not (Test-Path -LiteralPath $Python)) { return $false }
    $VenvConfig = Join-Path $VenvRoot "pyvenv.cfg"
    if (Test-Path -LiteralPath $VenvConfig) {
        $BasePython = (Get-Content -LiteralPath $VenvConfig | Where-Object { $_ -match '^executable\s*=' } | Select-Object -First 1) -replace '^executable\s*=\s*', ''
        if ($BasePython -and -not (Test-Path -LiteralPath $BasePython)) { return $false }
    }
    try {
        # This is Python code, so use Python's == operator rather than PowerShell -eq.
        & $Python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

if (-not (Test-WorkingVenv)) {
    $Python311 = Ensure-Python311
    if (Test-Path -LiteralPath $VenvRoot) {
        Write-Host "Removing broken or incompatible virtual environment: $VenvRoot" -ForegroundColor Yellow
        Remove-Item -LiteralPath $VenvRoot -Recurse -Force
    }
    Write-Host "[1/4] Creating Python 3.11 virtual environment: $VenvRoot" -ForegroundColor Cyan
    if ($Python311.Count -gt 1) {
        & $Python311[0] $Python311[1] -m venv $VenvRoot
    }
    else {
        & $Python311[0] -m venv $VenvRoot
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Python virtual environment."
    }
}

Write-Host "[2/4] Installing compatible pip, setuptools, and wheel" -ForegroundColor Cyan
# setuptools 81+ removed pkg_resources used by older build scripts.
& $Python -m pip install --upgrade pip "setuptools<81" wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade Python packaging tools." }

$TorchIndex = "https://download.pytorch.org/whl/cpu"
$PypiMirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
Write-Host "[3/4] Installing CPU PyTorch, OpenVINO, and backend dependencies" -ForegroundColor Cyan
& $Python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU PyTorch." }
$RequirementsFile = Join-Path $BackendRoot "requirements.txt"
$RuntimeRequirements = Join-Path $env:TEMP "recognize-demo-runtime-$PID.txt"
try {
    # RTMPose is converted from its official ONNX artifact. The OpenMMLab Python
    # stack remains documented in requirements but is not needed at runtime.
    Get-Content -LiteralPath $RequirementsFile | Where-Object { $_ -notmatch '^(mmpose|mmdet|mmcv-lite|mmengine)==' } | Set-Content -LiteralPath $RuntimeRequirements -Encoding Ascii
    # 使用国内镜像安装 PaddlePaddle/PaddleOCR doc-parser 等 CPU 依赖；
    # PyTorch 已在上方通过其官方 CPU 索引单独安装。
    & $Python -m pip install -r $RuntimeRequirements -i $PypiMirror
    if ($LASTEXITCODE -ne 0) { throw "Failed to install backend runtime dependencies." }
}
finally {
    Remove-Item -LiteralPath $RuntimeRequirements -Force -ErrorAction SilentlyContinue
}

# Avoid importing Paddle/PaddleOCR during setup. Paddle's import probes optional
# C++ tools on Windows and prints misleading "could not find files"/ccache
# messages even when the package is installed correctly. Metadata validation is
# enough here; FastAPI performs the actual local OCR-model check lazily.
Write-Host "[verify] Checking installed package versions (PaddleOCR loads when the service starts)" -ForegroundColor Cyan
& $Python -u -c "import sys, importlib.metadata as md, torch, openvino; print('Python:', sys.version.split()[0], flush=True); print('PyTorch:', torch.__version__, flush=True); print('OpenVINO:', openvino.__version__, flush=True); print('PaddlePaddle:', md.version('paddlepaddle'), flush=True); print('PaddleOCR:', md.version('paddleocr'), flush=True)"
if ($LASTEXITCODE -ne 0) { throw "Python environment validation failed." }

Write-Host "[4/4] Downloading domestic model assets and auditing checksums" -ForegroundColor Cyan
Push-Location $BackendRoot
try {
    # --SkipModelExport skips only ONNX/IR export. --SkipModelDownload is for
    # repairing the Python environment while offline; it still performs a
    # visible local audit so no model is falsely reported ready.
    if ($SkipModelDownload) {
        Write-Host "[models] Download skipped by --SkipModelDownload; auditing local assets only." -ForegroundColor Yellow
        & $Python -u model_downloader.py --no-download
    }
    else {
        Write-Host "[models] ModelScope download/audit started. Large files show MD5 progress below." -ForegroundColor Cyan
        & $Python -u model_downloader.py
    }
    if ($LASTEXITCODE -ne 0) { throw "Model asset download or audit failed." }
    if (-not $SkipModelExport) {
        Write-Host "[export] Exporting local OpenVINO IR" -ForegroundColor Cyan
        & $Python export_all_models.py
        if ($LASTEXITCODE -ne 0) { throw "Model export failed." }
    }
    else {
        Write-Host "[export] Model export skipped by --SkipModelExport." -ForegroundColor Yellow
    }
}
finally {
    Pop-Location
}

Write-Host "Backend setup completed. Run backend\start-backend.ps1 to start the service." -ForegroundColor Green
