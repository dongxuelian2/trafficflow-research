param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path "$PSScriptRoot\..").Path
& $Python -m venv "$taskRoot\.venv"
if ($LASTEXITCODE -ne 0) { throw 'Creating Python environment failed' }
& "$taskRoot\.venv\Scripts\python.exe" -m pip install -r "$taskRoot\requirements.lock.txt"
if ($LASTEXITCODE -ne 0) { throw 'Installing dependencies failed' }
& "$taskRoot\.venv\Scripts\python.exe" -m pip install --no-deps --no-build-isolation -e $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'Installing project failed' }
