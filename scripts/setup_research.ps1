param([string]$Python = 'python', [string]$Uv = 'uv')
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path "$PSScriptRoot\..").Path
$taskTemp = Join-Path $taskRoot '.cache\build_tmp'
New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
$env:TEMP=$taskTemp
$env:TMP=$taskTemp
& $Uv venv --python $Python "$taskRoot\.venv-research"
if ($LASTEXITCODE -ne 0) { throw 'Research environment creation failed' }
& $Uv --cache-dir "$taskRoot\.cache\uv" pip install --python "$taskRoot\.venv-research\Scripts\python.exe" -r "$taskRoot\requirements.research.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Research dependencies failed' }
& $Uv --cache-dir "$taskRoot\.cache\uv" pip install --python "$taskRoot\.venv-research\Scripts\python.exe" --no-deps --no-build-isolation -e $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed' }
& $Uv --cache-dir "$taskRoot\.cache\uv" pip check --python "$taskRoot\.venv-research\Scripts\python.exe"
if ($LASTEXITCODE -ne 0) { throw 'Research dependency consistency check failed' }
