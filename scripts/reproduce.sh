#!/usr/bin/env bash
# Reproduce the full ROAD pipeline on Linux (run from the repository root, venv active).
# Prerequisite: ROAD extracted into data/raw/road (see scripts/download_data.md).
# Approximate time on a 4-core laptop: 45-60 minutes.
set -euo pipefail
steps=(
  "scripts/build_dataset.py --checksums"
  "scripts/verify_road_labels.py"
  "scripts/build_features.py --dataset road"
  "scripts/verify_features.py --dataset road"
  "scripts/train.py anomaly --dataset road"
  "scripts/train.py hybrid --dataset road"
  "scripts/evaluate.py anomaly --dataset road"
  "scripts/evaluate.py hybrid --dataset road"
  "scripts/explain.py --dataset road"
  "scripts/ablation.py --dataset road"
  "scripts/error_analysis.py --dataset road"
  "scripts/benchmark.py --dataset road --frames 1000000"
  "scripts/make_results.py"
)
for s in "${steps[@]}"; do
  echo -e "\n=== python $s ==="
  # shellcheck disable=SC2086
  python $s
done
echo -e "\nDone. See reports/RESULTS.md"
