#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=.

echo "======================================================="
echo " Starting MuJoCo Relational Precondition Smoke Test "
echo "======================================================="

# 1. Run Unit Tests
echo "[Phase 1/5] Running PyTest Unit Test Suite..."
$PYTHON_BIN -m pytest tests/ -v

# 2. Run Demonstration Video Clips Generation
echo "[Phase 2/5] Generating Demonstration Videos..."
$PYTHON_BIN -c "
from src.generation.demonstration_generator import DemonstrationGenerator
demo_gen = DemonstrationGenerator(output_dir='data/demos')
p1 = demo_gen.generate_task_1_demo('demo_task1_smoke')
p2 = demo_gen.generate_task_2_demo('demo_task2_smoke')
print(f'Generated demos: {p1}, {p2}')
"

# 3. Run Query & Counterfactual Pair Generation
echo "[Phase 3/5] Generating Matched Query Pairs..."
$PYTHON_BIN -c "
from src.generation.query_generator import QueryGenerator
gen = QueryGenerator(config_path='configs/smoke.yaml')
records = gen.run_generation()
print(f'Generated {len(records)} counterfactual query pair records.')
"

# 4. Run Dataset Validator
echo "[Phase 4/5] Running Dataset Validator..."
$PYTHON_BIN src/validation/dataset_validator.py data/manifests/smoke_manifest.jsonl

# 5. Render HTML & Contact Sheet Previews & Tracked Smoke Artifacts
echo "[Phase 5/5] Generating Previews & Tracked Artifacts..."
$PYTHON_BIN -c "
from src.preview.html_preview import HTMLPreviewGenerator
from src.preview.contact_sheet import ContactSheetGenerator
from src.preview.smoke_artifacts import TrackedSmokeArtifactsGenerator

html_gen = HTMLPreviewGenerator(output_dir='data/previews')
html_path = html_gen.generate_html_report('data/manifests/smoke_manifest.jsonl')

cs_gen = ContactSheetGenerator(output_dir='data/previews')
cs_path = cs_gen.generate_contact_sheet('data/manifests/smoke_manifest.jsonl')

art_gen = TrackedSmokeArtifactsGenerator(artifacts_dir='artifacts/smoke')
art_gen.generate_all_smoke_artifacts()

print(f'Previews and tracked smoke artifacts generated successfully:')
print(f'  HTML Report: {html_path}')
print(f'  Contact Sheet: {cs_path}')
print(f'  Tracked Smoke Artifacts: artifacts/smoke/')
"

echo "======================================================="
echo " Smoke Test Completed Successfully! All Checks Passed. "
echo "======================================================="
