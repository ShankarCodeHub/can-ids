# Reproduce the full ROAD pipeline on Windows (run from the repository root, venv active).
# Prerequisite: ROAD extracted into data\raw\road (see scripts\download_data.md).
# Approximate time on a 4-core laptop: 45-60 minutes.
$ErrorActionPreference = "Stop"
$steps = @(
    "scripts\build_dataset.py --checksums",
    "scripts\verify_road_labels.py",
    "scripts\build_features.py --dataset road",
    "scripts\verify_features.py --dataset road",
    "scripts\train.py anomaly --dataset road",
    "scripts\train.py hybrid --dataset road",
    "scripts\evaluate.py anomaly --dataset road",
    "scripts\evaluate.py hybrid --dataset road",
    "scripts\explain.py --dataset road",
    "scripts\ablation.py --dataset road",
    "scripts\error_analysis.py --dataset road",
    "scripts\benchmark.py --dataset road --frames 1000000",
    "scripts\make_results.py"
)
foreach ($s in $steps) {
    Write-Host "`n=== python $s ===" -ForegroundColor Cyan
    python $s.Split(" ")
    if ($LASTEXITCODE -ne 0) { throw "step failed: $s" }
}
Write-Host "`nDone. See reports\RESULTS.md" -ForegroundColor Green
