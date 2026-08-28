#!/bin/bash
# Model Comparison Script
# Usage: ./compare_models.sh [OPTIONS]
# Example: ./compare_models.sh --stock AAPL --episodes 100

cd "$(dirname "$0")" || exit 1

echo "======================================================================"
echo "Trading Agent Model Comparison Framework"
echo "======================================================================"

PYTHONPATH=src python compare_models.py "$@"
