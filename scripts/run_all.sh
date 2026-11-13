#!/usr/bin/env bash
set -euo pipefail

for dataset in activity physionet ushcn mimic; do
  python run_models.py --dataset "${dataset}"
done
