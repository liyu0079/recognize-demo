$ErrorActionPreference = "Stop"
# Only change execution policy for this PowerShell process.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Refresh-ProcessPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $nodeHome = Join-Path $env:ProgramFiles "nodejs"
    if (Test-Path -LiteralPath $nodeHome) { $env:Path = "$nodeHome;$env:Path" }
}

function Invoke-Pnpm {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $script:Corepack.Source pnpm @Arguments
    if ($LASTEXITCODE -ne 0) { throw "pnpm $($Arguments -join ' ') failed." }
}

Refresh-ProcessPath
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { throw "Node.js and winget were not found. Run backend\\setup-backend.ps1 or install Node.js 20 LTS." }
    & $winget.Source install --id OpenJS.NodeJS.LTS --exact --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Node.js LTS installation failed." }
    Refresh-ProcessPath
}
$script:Corepack = Get-Command corepack -ErrorAction SilentlyContinue
if (-not $script:Corepack) { throw "Node.js does not provide corepack; pnpm cannot be initialized." }
& $script:Corepack.Source enable
& $script:Corepack.Source prepare pnpm@9.15.4 --activate
if ($LASTEXITCODE -ne 0) { throw "pnpm installation failed." }
Invoke-Pnpm --version

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "node_modules"))) {
    Invoke-Pnpm install
}

$FrontendPort = 5173
while (Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue) {
    $FrontendPort++
}

Write-Host "Starting Vue frontend: http://127.0.0.1:$FrontendPort" -ForegroundColor Green
Invoke-Pnpm run dev -- --host 0.0.0.0 --port $FrontendPort
