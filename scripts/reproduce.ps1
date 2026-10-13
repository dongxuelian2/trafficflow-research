param(
    [string]$ReleaseRoot = 'E:\ieee\data\kaggle_public',
    [string]$RunRoot = 'artifacts\reproduction',
    [string]$Python = '.\.venv-repro\Scripts\python.exe',
    [switch]$PhysicsCandidate
)
$ErrorActionPreference = 'Stop'
& $Python scripts\verify_vendor.py
if ($LASTEXITCODE -ne 0) { throw 'Upstream checksum failed' }
& $Python -m trafficflow validate --release-root $ReleaseRoot --output "$RunRoot\audit" --full-hash
if ($LASTEXITCODE -ne 0) { throw 'Release audit failed' }
& $Python scripts\campaign.py --release-root $ReleaseRoot --output "$RunRoot\campaign" --workers 2
if ($LASTEXITCODE -ne 0) { throw 'Campaign failed' }
& $Python scripts\refine_queues.py --release-root $ReleaseRoot --output "$RunRoot\queues"
if ($LASTEXITCODE -ne 0) { throw 'Queue calibration failed' }
& $Python scripts\promote_models.py --release-root $ReleaseRoot --campaign "$RunRoot\campaign" --queue-refinement "$RunRoot\queues" --output "$RunRoot\models"
if ($LASTEXITCODE -ne 0) { throw 'Model freeze failed' }
& $Python scripts\assemble_campaign.py --release-root $ReleaseRoot --campaign "$RunRoot\campaign" --queue-refinement "$RunRoot\queues" --output "$RunRoot\submission"
if ($LASTEXITCODE -ne 0) { throw 'Submission completeness failed' }
if ($PhysicsCandidate) {
    & $Python scripts\physics_tradeoff.py --release-root $ReleaseRoot --campaign "$RunRoot\campaign" --output "$RunRoot\physics" --workers 2
    if ($LASTEXITCODE -ne 0) { throw 'Physics candidate research failed' }
    & $Python scripts\assemble_campaign.py --release-root $ReleaseRoot --campaign "$RunRoot\campaign" --queue-refinement "$RunRoot\queues" --state-override "$RunRoot\physics" --output "$RunRoot\submission_physics"
    if ($LASTEXITCODE -ne 0) { throw 'Physics candidate completeness failed' }
    Write-Output "$RunRoot\submission_physics\submission.csv"
}
Write-Output "$RunRoot\submission\submission.csv"
