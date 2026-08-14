#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for Packet 2 Intervention Smoke Dataset

echo "======================================================================"
echo "Starting Packet 2 Intervention Smoke Pipeline"
echo "======================================================================"

# 1. Verify environment
export PATH=/home/projects/long-horizon/miniconda3/envs/infeasibility-latent/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export MUJOCO_GL=egl

CONFIG_PATH="configs/intervention/smoke.yaml"
OUTPUT_DIR="data/intervention_smoke"
REPORT_PATH="artifacts/intervention_smoke/smoke_validation_report.json"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found at $CONFIG_PATH" >&2
    exit 1
fi

# 2. Clean only the designated smoke output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning existing smoke directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# 3. Generate dataset
echo "Running dataset generator..."
python scripts/generate_intervention_dataset.py --config "$CONFIG_PATH"

# 4. Run independent validation
echo "Running independent validation..."
python scripts/validate_intervention_dataset.py \
    --manifest "$OUTPUT_DIR/manifest.jsonl" \
    --report "$REPORT_PATH"

echo "======================================================================"
echo "Smoke Pipeline Completed Successfully!"
echo "Report: $REPORT_PATH"
echo "======================================================================"
