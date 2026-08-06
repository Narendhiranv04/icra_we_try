# Relational Precondition Benchmark v0.1 for Demonstration-Conditioned Reasoning

This repository provides a complete, physically grounded MuJoCo simulation benchmark for evaluation of demonstration-conditioned relational preconditions across robotic manipulation tasks.

## Tasks & Semantics

The benchmark evaluates two relational precondition task families:

1. **Task 1: Open Box**
   - **Goal Instruction**: `"Open the box."`
   - **PROCEED Condition**: Box lid interaction surface (`B1_lid_panel`) is clear.
   - **STOP Condition**: One or more designated objects occupy or rest on the lid (`ON_TOP_OF(blocker, box_lid)`).
   - **Demonstration Semantics**: Fetch robot arm reaches the lid handle, closes gripper fingers, activates weld constraint after validated 3cm proximity, pulls the lid open to $\ge 45^\circ$, and holds the lid open in the final static phase.

2. **Task 2: Place Object1 in Target Region**
   - **Goal Instruction**: `"Place object1 in the target region."`
   - **PROCEED Condition**: Single-capacity target region (`target_region_body` / `target_region_geom`) is empty and available.
   - **STOP Condition**: Non-target object occupies the target region (`OCCUPIES(object2, target_region)`).
   - **Demonstration Semantics**: Fetch robot arm reaches `object1`, commands gripper fingers closed, steps physics for finger closure settling while weld remains inactive, validates closure and 3cm proximity, activates equality weld constraint, transports `object1` along a parabolic trajectory, releases object in target region, and retreats.

---

## Benchmark Setup & Execution Commands

### 1. Environment Setup
```bash
cd /home/naren/RA_iiith_new
source /home/naren/RA_iiith/.venv/bin/activate
export MUJOCO_GL="egl"
export PYTHONPATH=.
```

### 2. Unit & Regression Tests
```bash
pytest tests/ -v --junitxml=data/reports/pytest_results.xml
```

### 3. Run Clean Smoke Test Pipeline
```bash
bash scripts/run_smoke_test.sh
```

### 4. Run Clean Pilot Benchmark Pipeline
```bash
bash scripts/run_pilot_generation.sh
```

### 5. Validate Dataset Manifests
```bash
# Validate Smoke Manifest
python src/validation/dataset_validator.py data/manifests/smoke_manifest.jsonl

# Validate Pilot Manifest
python src/validation/dataset_validator.py data/manifests/pilot_manifest.jsonl
```

### 6. Run Final Release Verification Script
```bash
python scripts/verify_release_state.py
```

---

## Dataset Splits & Positive Controls

### Dataset Splits
- **ID (`id`)**: Familiar objects (`coffee_can`, `sugar_box`, `mug`), familiar backgrounds (`bg_neutral_wood`), familiar position bins (`centre`, `front_left`, `front_right` for Task 1; `centre`, `left`, `right` for Task 2).
- **Unseen Object (`unseen_object`)**: Holdout object types (`cup`, `bowl`).
- **Unseen Background (`unseen_background`)**: Holdout backgrounds (`bg_blue_counter`, `bg_granite_dark`).
- **Pure Compositional (`compositional`)**: 100% familiar components (objects, backgrounds, position bins, blocker counts), but novel factor combinations whose full factor tuple `(task_id, object_type, background_id, position_bin, blocker_count, start_bin, lighting_family)` is absent from development ID data.

### Positive Control Subtypes
- **Task 1 Controls**:
  1. `empty_lid`: Lid clear, no candidate blockers, empty candidate mask.
  2. `one_object_beside`: Object beside box (`beside_margin = 0.35m`), relation false.
  3. `two_objects_beside`: Two objects beside box, relation false for both.
  4. `near_lid_outside_footprint`: Object adjacent to lid boundary (`near_boundary_margin = 0.04m`), footprint overlap = 0.0, relation false.

- **Task 2 Controls**:
  1. `empty_target`: Target empty, no occupant, empty candidate mask.
  2. `one_object_beside_target`: Object outside target (`beside_margin = 0.30m`), occupancy false.
  3. `one_object_near_target_outside`: Object adjacent to target boundary (`near_boundary_margin = 0.04m`), footprint overlap = 0.0, occupancy false.
  4. `multiple_distractors_outside`: Multiple objects outside target region, occupancy false.

---

## Benchmark Record & Image Counts

- **Smoke Run**: 16 total records (8 matched pairs = 16 paired images + 8 positive controls; 2 robot demonstration directories).
- **Pilot Run**: 152 total records (120 matched pairs = 240 paired images + 32 positive controls; 6 robot demonstration directories).
- **Matched Pairs vs Images**: Each matched pair consists of 1 STOP image and 1 PROCEED image sharing exact background, lighting, objects, camera, and random seed, differing ONLY in the minimal declared intervention path (e.g. blocker position on lid vs beside box).
- **Grasp Weld Assistance Disclosure**: All robot manipulation demonstrations use physics-based Fetch joint actuation and end-effector IK. Deterministic equality weld assistance is activated strictly AFTER validated finger closure settling and surface-aware 3cm grasp proximity checks.

---

## Benchmark Output Directory Structure

```
data/                           # Ignored in git (.gitignore)
├── demos/                      # Genuine robot demonstration directories (MP4, state_log.jsonl, metadata.json)
├── queries/                    # Matched query scene pairs & positive controls
├── manifests/                  # JSONL manifest files (smoke_manifest.jsonl, pilot_manifest.jsonl)
├── reports/                    # Machine-readable validation reports (reproducibility, splits, controls)
└── previews/                   # Interactive HTML previews and high-res contact sheets

artifacts/smoke/                # Version-tracked smoke artifacts for release verification
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
