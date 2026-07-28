# packaging/build_gui.ps1
# Local build script for Windows — produces a PyInstaller one-file .exe.
#
# Usage (from repo root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File packaging/build_gui.ps1

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot..'

Write-Host "=== pySimpleMask: Local GUI build (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# Check PyInstaller
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Error "PyInstaller not found. Install it first:`n  pip install pyinstaller"
    exit 1
}

# Clean previous build
Remove-Item -Recurse -Force build, dist\pySimpleMask* -ErrorAction SilentlyContinue

Write-Host "Running PyInstaller..."
pyinstaller pysimplemask.spec

$exe = Get-ChildItem dist -Filter "pySimpleMask.exe" -ErrorAction SilentlyContinue
if ($exe) {
    Write-Host ""
    Write-Host "Build succeeded: $($exe.FullName)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Launch with:"
    Write-Host "  $($exe.FullName)"
} else {
    Write-Error "Build FAILED — .exe not found in dist/."
    exit 1
}