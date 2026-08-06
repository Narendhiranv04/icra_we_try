#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=.

mkdir -p data/reports

echo "======================================================="
echo " Starting MuJoCo Relational Precondition Smoke Test "
echo "======================================================="

# 1. Run Unit Tests with JUnit XML output
echo "[Phase 1/6] Running PyTest Unit Test Suite..."
$PYTHON_BIN -m pytest tests/ -v --junitxml=data/reports/pytest_results.xml

# 2. Run Demonstration Video Clips Generation & Distinctness Validation
echo "[Phase 2/6] Generating Demonstration Directories & Validating Distinctness..."
$PYTHON_BIN -c "
from src.generation.demonstration_generator import DemonstrationGenerator
from src.validation.demonstration_validator import DemonstrationValidator
from src.validation.demonstration_distinctness import DemonstrationDistinctnessValidator

demo_gen = DemonstrationGenerator(output_dir='data/demos')
p1 = demo_gen.generate_task_1_demo('demo_task1_smoke', background_id='bg_neutral_wood', seed=42)
p2 = demo_gen.generate_task_2_demo('demo_task2_smoke', obj_name='coffee_can', background_id='bg_neutral_wood', seed=43)

val1, issues1 = DemonstrationValidator.validate_demo_dir('data/demos/open_box/demo_task1_smoke')
val2, issues2 = DemonstrationValidator.validate_demo_dir('data/demos/place_object/demo_task2_smoke')

print(f'Task 1 Demo Validation: {val1}, issues={issues1}')
print(f'Task 2 Demo Validation: {val2}, issues={issues2}')

if not (val1 and val2):
    raise RuntimeError(f'Demonstration validation failed: {issues1} {issues2}')

dist_val = DemonstrationDistinctnessValidator('data/demos')
dist_valid, dist_rep = dist_val.validate_all_demos()
print(f'Demonstration Distinctness Validation: {dist_valid}')
"

# 3. Run Query & Counterfactual Pair Generation
echo "[Phase 3/6] Generating Matched Query Pairs & Standalone Controls..."
$PYTHON_BIN -c "
from src.generation.query_generator import QueryGenerator
gen = QueryGenerator(config_path='configs/smoke.yaml')
records = gen.run_generation()
print(f'Generated {len(records)} query records (pairs & positive controls).')
"

# 4. Run Dataset Validator (including Deep Diff, Split Holdouts, Actual Reproducibility)
echo "[Phase 4/6] Running Comprehensive Dataset Validator..."
$PYTHON_BIN src/validation/dataset_validator.py data/manifests/smoke_manifest.jsonl

# 5. Render HTML & Contact Sheet Previews
echo "[Phase 5/6] Generating Previews & HTML Reports..."
$PYTHON_BIN -c "
from src.preview.html_preview import HTMLPreviewGenerator
from src.preview.contact_sheet import ContactSheetGenerator

html_gen = HTMLPreviewGenerator(output_dir='data/previews')
html_path = html_gen.generate_html_report('data/manifests/smoke_manifest.jsonl', output_filename='benchmark_preview.html')

cs_gen = ContactSheetGenerator(output_dir='data/previews')
cs_path = cs_gen.generate_contact_sheet('data/manifests/smoke_manifest.jsonl', output_filename='contact_sheet.png')

print(f'Previews generated successfully:')
print(f'  HTML Report: {html_path}')
print(f'  Contact Sheet: {cs_path}')
"

# 6. Compile Tracked Smoke Artifacts & Final Report
echo "[Phase 6/6] Compiling Tracked Smoke Artifacts..."
$PYTHON_BIN -c "
from src.preview.smoke_artifacts import TrackedSmokeArtifactsGenerator
smoke_gen = TrackedSmokeArtifactsGenerator(artifacts_dir='artifacts/smoke')
status = smoke_gen.generate_all_smoke_artifacts()
print(f'Tracked Smoke Artifacts status: {status}')
if status != 'PASSED':
    raise RuntimeError('Smoke test artifacts report status is FAILED!')
"

echo "======================================================="
echo " Smoke Test Completed Successfully! All Checks Passed. "
echo "======================================================="
