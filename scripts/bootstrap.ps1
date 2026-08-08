# Create the local development environment (Windows).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
#
# Run this once per machine.  The .venv it creates is deliberately gitignored:
# a venv records the absolute path of the interpreter that built it, so copying
# one between laptops produces an environment that half-works in confusing ways.
# Ship the repo; recreate the environment.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# 3.11 is the pinned target (architecture v1.1 §0.4) and is what Kaggle runs.
# PyTorch has no 3.13+ wheels, so building on a newer interpreter would install
# cleanly today and fail at Phase 3.
$python = $null
try { $python = (py -3.11 -c "import sys; print(sys.executable)") } catch { }
if (-not $python) {
    Write-Error "Python 3.11 not found. Install it from python.org, then re-run. (`py -0p` lists what is available.)"
}
Write-Host "Using $python"

if (Test-Path ".venv") {
    Write-Host ".venv already exists; reusing it. Delete it to rebuild from scratch."
} else {
    & $python -m venv .venv
}

$venvPython = Join-Path $repo ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install -e .

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Then check the install with:"
Write-Host "    pytest -q"
