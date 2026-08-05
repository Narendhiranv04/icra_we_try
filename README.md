# Relational Precondition Benchmark v0.1 for Demonstration-Conditioned Reasoning

This repository provides a complete, physically grounded MuJoCo simulation benchmark for evaluation of demonstration-conditioned relational preconditions across robotic manipulation tasks.

## Tasks & Semantics

The benchmark evaluates two relational precondition task families:

1. **Task 1: Open Box**
   - **Goal Instruction**: `"Open the box."`
   - **PROCEED Condition**: Box lid interaction surface (`B1_lid_panel`) is clear.
   - **STOP Condition**: One or more designated objects occupy or rest on the lid (`ON_TOP_OF(blocker, box_lid)`).

2. **Task 2: Place Object1 in Target Region**
   - **Goal Instruction**: `"Place object1 in the target region."`
   - **PROCEED Condition**: Single-capacity target region (`target_region_body` / `target_region_geom`) is empty and available.
   - **STOP Condition**: Non-target object occupies the target region (`OCCUPIES(object2, target_region)`).

---

## Quickstart Commands

### 1. Environment Setup
```bash
# Clone and enter workspace
cd /home/naren/RA_iiith_new

# Set Python environment & MuJoCo GL backend
export PYTHON_BIN="${PYTHON_BIN:-python}"
export MUJOCO_GL="egl"
export PYTHONPATH=.
```

### 2. Running Unit Tests
```bash
$PYTHON_BIN -m pytest tests/ -v
```

### 3. Generating Smoke Robot Demonstrations
```bash
$PYTHON_BIN -c "
from src.generation.demonstration_generator import DemonstrationGenerator
gen = DemonstrationGenerator(output_dir='data/demos')
p1 = gen.generate_task_1_demo('demo_task1_smoke')
p2 = gen.generate_task_2_demo('demo_task2_smoke')
print('Smoke Demos:', p1, p2)
"
```

### 4. Generating Smoke Query Pairs
```bash
$PYTHON_BIN -c "
from src.generation.query_generator import QueryGenerator
gen = QueryGenerator(config_path='configs/smoke.yaml')
records = gen.run_generation()
print(f'Generated {len(records)} query pairs.')
"
```

### 5. Running the Complete Smoke Pipeline
```bash
bash scripts/run_smoke_test.sh
```

### 6. Validating Smoke Dataset
```bash
$PYTHON_BIN src/validation/dataset_validator.py data/manifests/smoke_manifest.jsonl
```

### 7. Generating HTML & Contact Sheet Previews
```bash
$PYTHON_BIN -c "
from src.preview.html_preview import HTMLPreviewGenerator
from src.preview.contact_sheet import ContactSheetGenerator

HTMLPreviewGenerator(output_dir='data/previews').generate_html_report('data/manifests/smoke_manifest.jsonl', 'benchmark_preview.html')
ContactSheetGenerator(output_dir='data/previews').generate_contact_sheet('data/manifests/smoke_manifest.jsonl', 'contact_sheet.png')
"
```

### 8. Generating Full Pilot Benchmark
```bash
$PYTHON_BIN scripts/run_pilot_generation.py
```

### 9. Validating Pilot Dataset
```bash
$PYTHON_BIN src/validation/dataset_validator.py data/manifests/pilot_manifest.jsonl
```

---

## Benchmark Output Structure

```
data/
├── demos/                      # Genuine robot demonstration videos (MP4)
├── queries/                    # Matched query scene pairs (pair_task_X_YYY)
│   └── pair_task_1_001/
│       ├── stop_rgb.png
│       ├── stop_instance_segmentation.png
│       ├── stop_candidate_object_mask.png
│       ├── stop_relation_target_mask.png
│       ├── stop_causal_violation_mask.png
│       ├── stop_combined_relation_visualization.png
│       ├── proceed_rgb.png
│       ├── proceed_instance_segmentation.png
│       ├── proceed_candidate_object_mask.png
│       ├── proceed_relation_target_mask.png
│       ├── proceed_causal_violation_mask.png
│       ├── proceed_combined_relation_visualization.png
│       └── metadata.json
├── manifests/                  # JSONL manifest files (smoke_manifest.jsonl, pilot_manifest.jsonl)
└── previews/                   # HTML reports and grid contact sheets

artifacts/smoke/                # Tracked smoke run artifacts
├── contact_sheet.png
├── demonstration_montage.png
├── smoke_report.json
├── smoke_report.md
├── representative_metadata.json
└── README.md
```

---

## Documentation Index
- [Dataset Specification](docs/DATASET_SPEC.md)
- [Label Definitions](docs/LABEL_DEFINITIONS.md)
- [Demonstration Generation Spec](docs/DEMONSTRATION_GENERATION.md)
- [Counterfactual Pair Specification](docs/COUNTERFACTUAL_PAIR_SPEC.md)
- [Validation Specification](docs/VALIDATION_SPEC.md)
- [Reference Project Notes](docs/REFERENCE_PROJECT_NOTES.md)
