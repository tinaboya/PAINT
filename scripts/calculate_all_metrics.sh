#!/bin/bash
# Calculate metrics for all evaluation JSONs in outputs/runs/
# Run from project root: ./scripts/calculate_all_metrics.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Calculating metrics for evaluation results..."
python -m paint.extraction_evaluation outputs/runs/*.json

echo ""
echo "✅ All done!"
