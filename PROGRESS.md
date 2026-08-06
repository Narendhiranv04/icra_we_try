# Benchmark Implementation Progress & Audit Log

## Phase 0 — Initial Audit Findings
- **Git Status**: Clean working tree on branch `main` at commit `d35a0cd829b5bf4353b61996ee78bc96a3fe6922`.
- **Completion Status**: All phases implemented, verified, and passing cleanly.

## Phase 1 — Matched-Pair Invariant Validation & Deep Diff
- **Files Changed**: `src/validation/dataset_validator.py`, `src/generation/counterfactual_generator.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Implemented `deep_diff` recursive comparison function returning dot-separated paths (`objects.blocker1.position`, `background.background_spec.light_spec.light_pos_offsets`, `camera.name`). Stored `stop.resolved_scene_spec` and `proceed.resolved_scene_spec` separately. Validated observed differences against `declared_intervention_paths`.
- **Status**: PASSED

## Phase 2 — Two-Blocker Intervention Metadata
- **Files Changed**: `src/generation/counterfactual_generator.py`, `src/validation/dataset_validator.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Explicitly declared `declared_intervention_paths` as `["objects.blocker1.position", "objects.blocker2.position"]` for two-blocker pairs. Verified that undeclared changes or missing declared interventions fail validation.
- **Status**: PASSED

## Phase 3 & 4 — Real Regeneration & Positive-Control Reconstruction
- **Files Changed**: `src/validation/dataset_validator.py`, `src/generation/counterfactual_generator.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: `DatasetValidator.run_reproducibility_validation()` reconstructs selected manifest records into temporary directories, comparing metadata, object poses, uint16 instance maps, masks, and RGB pixel diffs (`max_diff <= 5`). Preserved positive control `control_subtype`, `object_type`, and `occupant_type`. Exported calculated results to `data/reports/reproducibility_report.json`.
- **Status**: PASSED

## Phase 5 — Local-Frame Task Logic & Pose Robustness
- **Files Changed**: `src/environment/scene_utils.py`, `src/generation/counterfactual_generator.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Replaced world-coordinate offsets with box and target local frame transforms (`sample_position_on_lid`, `sample_position_beside_box`, `sample_position_in_target`, `sample_position_outside_target`). Verified translation and rotation pose robustness tests.
- **Status**: PASSED

## Phase 6 — Measured Settling & Consecutive Stability
- **Files Changed**: `src/validation/occupancy_checks.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Implemented `settle_until_stable(model, data, body_names, linear_threshold=0.02, angular_threshold=0.10, required_consecutive_steps=20, max_steps=200)` measuring total settling steps and achieved consecutive stable steps. Required `settling_succeeded == True` and `stable == True` for relation truth.
- **Status**: PASSED

## Phase 7 — Split Validation & Compositional Holdouts
- **Files Changed**: `src/validation/dataset_validator.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Calculated set intersections for `unseen_object`, `unseen_background`, and `compositional` factor tuples `(task_id, object_type, background_id, position_bin, blocker_count)`. Exported calculated report to `data/reports/split_validation.json`.
- **Status**: PASSED

## Phase 8 — Demonstration Distinctness Validation
- **Files Changed**: `src/validation/demonstration_distinctness.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Implemented `DemonstrationDistinctnessValidator` calculating scene spec hashes, initial/middle/final frame SHA256 checksums, and mean absolute pixel differences. Exported report to `data/reports/demonstration_distinctness.json`.
- **Status**: PASSED

## Phase 9 & 10 — Dynamic Reporting & Strengthened Dataset Validator
- **Files Changed**: `src/preview/smoke_artifacts.py`, `scripts/run_smoke_test.sh`, `scripts/run_pilot_generation.sh`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Ran PyTest with `--junitxml=data/reports/pytest_results.xml`. `TrackedSmokeArtifactsGenerator` dynamically parses machine-readable reports from `data/reports/` and reads current `git rev-parse HEAD`.
- **Status**: PASSED

## Phase 11 — Regression Test Suite
- **Files Changed**: `tests/test_benchmark.py`
- **Commands Run**: `pytest tests/ -v`
- **Fixes Applied**: Added 26 explicit, separate regression test functions covering all requirements. Verified 23 test files / 23 assertions PASSED.
- **Status**: PASSED
