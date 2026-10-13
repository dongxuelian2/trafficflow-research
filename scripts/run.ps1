param(
    [string]$ReleaseRoot = "$PSScriptRoot\..\data\kaggle_public",
    [string]$Output = "$PSScriptRoot\..\artifacts\final",
    [string]$Models = "$PSScriptRoot\..\artifacts\experiments"
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = "$taskRoot\src"
& "$taskRoot\.venv\Scripts\python.exe" -m trafficflow run --release-root $ReleaseRoot --output $Output --models-dir $Models --full-hash
if ($LASTEXITCODE -ne 0) { throw 'Pipeline failed. Inspect the reported error.' }
