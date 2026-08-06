#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=.

# Clean previous pilot data
rm -rf data/demos/open_box/demo_task1* data/demos/place_object/demo_task2* data/queries/pair_* data/queries/control_* data/manifests/pilot_* data/reports/pilot_*
mkdir -p data/reports

echo "======================================================="
echo " Starting MuJoCo Relational Precondition Pilot Generation "
echo "======================================================="

# 1. Run Unit Tests with JUnit XML output
echo "[Phase 1/5] Running PyTest Unit Test Suite..."
$PYTHON_BIN -m pytest tests/ -v --junitxml=data/reports/pytest_results.xml

# 2. Run Demonstration Video Clips Generation & Distinctness Validation
echo "[Phase 2/5] Generating 6 Genuinely Distinct Pilot Demonstration Directories & Validating..."
$PYTHON_BIN -c "
import json, shutil
from pathlib import Path
from src.generation.demonstration_generator import DemonstrationGenerator
from src.validation.demonstration_validator import DemonstrationValidator
from src.validation.demonstration_distinctness import DemonstrationDistinctnessValidator

demo_gen = DemonstrationGenerator(output_dir='data/demos')

bgs = ['bg_neutral_wood', 'bg_blue_counter', 'bg_granite_dark']
objs = ['coffee_can', 'sugar_box', 'mug']
start_bins = ['pick_left', 'pick_right', 'pick_far_left']
target_bins = ['centre', 'left', 'right']

demos_t1 = []
for i in range(1, 4):
    demo_id = f'demo_task1_{i:03d}'
    path = demo_gen.generate_task_1_demo(demo_id, background_id=bgs[i-1], seed=100+i)
    demos_t1.append(path)

demos_t2 = []
for i in range(1, 4):
    demo_id = f'demo_task2_{i:03d}'
    path = demo_gen.generate_task_2_demo(demo_id, obj_name=objs[i-1], start_bin=start_bins[i-1], target_bin=target_bins[i-1], background_id=bgs[i-1], seed=200+i)
    demos_t2.append(path)

val, rep = DemonstrationValidator.generate_demonstration_validation_report('data/demos', 'data/reports/pilot_demonstration_validation.json')
print(f'Pilot Demonstration Validation: {val}')
if not val:
    raise RuntimeError('Pilot demonstration validation failed!')

dist_val = DemonstrationDistinctnessValidator('data/demos')
dist_valid, dist_rep = dist_val.validate_all_demos()
print(f'Pilot Demonstration Distinctness Validation: {dist_valid}')
if not dist_valid:
    raise RuntimeError('Pilot demonstration distinctness validation failed!')

with open('data/reports/pilot_demo_distinctness.json', 'w', encoding='utf-8') as f:
    json.dump(dist_rep, f, indent=2)

print('All 6 pilot demonstration directories generated and validated cleanly!')
"

# 3. Run Query & Counterfactual Pair Generation
echo "[Phase 3/5] Generating Full Pilot Matched Pairs & Standalone Controls..."
$PYTHON_BIN -c "
from src.generation.query_generator import QueryGenerator
gen = QueryGenerator(config_path='configs/pilot.yaml')
records = gen.run_generation()
print(f'Generated {len(records)} total pilot query records.')
"

# 4. Run Dataset Validator (including Deep Diff, Split Holdouts, Actual Reproducibility)
echo "[Phase 4/5] Running Comprehensive Dataset Validator on Pilot Manifest..."
$PYTHON_BIN -c "
import json, shutil
from pathlib import Path
from src.validation.dataset_validator import DatasetValidator

validator = DatasetValidator('data/manifests/pilot_manifest.jsonl')
valid, logs = validator.validate_dataset()
for log in logs:
    print(log)

if not valid:
    raise RuntimeError('Pilot dataset validation failed!')

shutil.copy('data/reports/dataset_validation.json', 'data/reports/pilot_validation.json')
shutil.copy('data/reports/split_validation.json', 'data/reports/pilot_split_validation.json')
shutil.copy('data/reports/reproducibility_report.json', 'data/reports/pilot_reproducibility.json')
shutil.copy('data/reports/distribution_report.json', 'data/reports/pilot_distribution.json')

# Build pilot_control_distribution.json with control subtype breakdown
records = []
with open('data/manifests/pilot_manifest.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

controls = [r for r in records if r.get('sample_type') == 'positive_control']
subtype_dist = {'task_1': {}, 'task_2': {}}
for ctrl in controls:
    task_id = ctrl.get('task_id', 'task_1')
    subtype = ctrl.get('control_subtype', 'unknown')
    if task_id not in subtype_dist:
        subtype_dist[task_id] = {}
    subtype_dist[task_id][subtype] = subtype_dist[task_id].get(subtype, 0) + 1

required_t1 = {'empty_lid', 'one_object_beside', 'two_objects_beside', 'near_lid_outside_footprint'}
required_t2 = {'empty_target', 'one_object_beside_target', 'one_object_near_target_outside', 'multiple_distractors_outside'}
missing_t1 = required_t1 - set(subtype_dist.get('task_1', {}).keys())
missing_t2 = required_t2 - set(subtype_dist.get('task_2', {}).keys())
ctrl_dist_status = 'PASSED' if (not missing_t1 and not missing_t2) else 'FAILED'

ctrl_dist = {
    'status': ctrl_dist_status,
    'total_controls': len(controls),
    'subtype_distribution': subtype_dist,
    'missing_task1_subtypes': list(missing_t1),
    'missing_task2_subtypes': list(missing_t2),
}
with open('data/reports/pilot_control_distribution.json', 'w') as f:
    json.dump(ctrl_dist, f, indent=2)

print(f'Pilot control distribution status: {ctrl_dist_status}')
if ctrl_dist_status != 'PASSED':
    raise RuntimeError('Missing required pilot control subtypes!')
"

# 5. Render HTML & Contact Sheet Previews
echo "[Phase 5/5] Generating Pilot Previews..."
$PYTHON_BIN -c "
from src.preview.html_preview import HTMLPreviewGenerator
from src.preview.contact_sheet import ContactSheetGenerator

html_gen = HTMLPreviewGenerator(output_dir='data/previews')
html_path = html_gen.generate_html_report('data/manifests/pilot_manifest.jsonl', output_filename='pilot_benchmark_preview.html')

cs_gen = ContactSheetGenerator(output_dir='data/previews')
cs_path = cs_gen.generate_contact_sheet('data/manifests/pilot_manifest.jsonl', output_filename='pilot_contact_sheet.png')

print(f'Pilot previews generated successfully:')
print(f'  HTML Report: {html_path}')
print(f'  Contact Sheet: {cs_path}')
"

echo "======================================================="
echo " Pilot Generation Completed Successfully! All Checks Passed. "
echo "======================================================="
