#!/usr/bin/env bash
# Regression: run synthesis for existing checkpoint modes to ensure 3-tuple return and orchestrator changes do not break behavior.
# Requires: chakra, PyYAML. Run from CheckWork: ./run_regression_modes.sh
# If you see ModuleNotFoundError: chakra, install the project dependencies first.

set -e
cd "$(dirname "$0")"

echo "Running sync checkpoint..."
python3 synthesise_workload.py -c input_checkpoint.yaml

echo "Running async checkpoint..."
python3 synthesise_workload.py -c input_async_checkpoint.yaml

echo "Running checkfreq_like checkpoint..."
python3 synthesise_workload.py -c input_checkfreq_like.yaml

echo "Running checkfreq_like validation..."
python3 test_checkfreq_like.py

echo "Regression modes OK."
