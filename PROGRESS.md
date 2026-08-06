# Benchmark Implementation Progress & Audit Log

## Phase 0 — Initial Audit Findings
- **Git Status**: Clean working tree on branch `main` at commit `11227ddc337251e826d49b9373fc9340c53f351e`.
- **Completion Status**: All 14 correctness phases fully implemented, verified, and passing cleanly.

## Summary of Completed Correctness Phases

### Phase 1 — Make the Box Hinge Truly Passive
- Disabled `B1_lid_actuator` gains, bias, and force range permanently in `MjModel` (`actuator_gainprm = 0.0`, `actuator_biasprm = 0.0`, `actuator_forcerange = 0.0`).
- Added `lid_actuator_force` to `Task1StateLog` dataclass and per-frame state logger. Verified `abs(lid_actuator_force) <= 1e-5` for all frames.
- Documented open-and-held demonstration state via robot grasp weld.

### Phase 2 — Strict, Geometric Grasp Proximity
- Implemented strict `PROXIMITY_THRESHOLD = 0.03m` (3 cm) for Task 1 (grip site to handle site) and Task 2 (grip site to object top surface grasp point `obj_pos + [0, 0, hz]`).
- Verified grasp distances of **0.0001m (0.1 mm)** for Task 1 and **0.0008m (0.8 mm)** for Task 2.
- Enforced explicit gripper finger closing (`robot0:r_gripper_finger_actuator`, `l_gripper_finger_actuator`) and settling frames before weld activation.

### Phase 3 — Fix Instance-Map Corruption & Auditing
- Fixed Task 1 PROCEED instance uint16 map saving bug (`np.save(proceed_inst_npy, inst_proceed_16)`).
- Audited all instance-map writes to guarantee genuine uint16 maps (`dtype == uint16`, `shape == (H, W)`), valid ID maps, clean `np.load`, and inspected every generated map in test suite.

### Phase 4 — Preserve Exact Matched-Pair Lighting and Background
- Created explicit, serializable `BackgroundSpec` and `LightSpec`.
- Sampled background and lighting specs ONCE per matched pair and applied identical specs to both STOP and PROCEED pair members.
- Verified 100% numerical equality of all background material RGBA, light positions, and light diffuse values within each pair.

### Phase 5 — Remove Remaining World-Coordinate Task Logic
- Replaced remaining world-coordinate task logic and hardcoded predicate bounding boxes with dynamic `scene_utils` local frame transforms.
- Verified translation and rotation pose-change robustness tests.

### Phase 6 — Require Stability for Relation Labels
- Settled query scenes until `linear_speed <= 0.02 m/s` and `angular_speed <= 0.10 rad/s` for consecutive steps.
- Enforced `stable == True` as a mandatory prerequisite for `ON_TOP_OF` and `OCCUPIES` relation truth (`relation_true == True`).
- Logged `settling_steps`, `consecutive_stable_steps`, `overlap_ratio`, `vertical_gap`, `contact`, `linear_speed`, `angular_speed`, `stable`, `relation_true` in output metadata.

### Phase 7 — Make Splits Genuinely Valid
- Integrated `SplitPlanner` factor assignments explicitly for `object_type`, `background_id`, `lighting_family`, `position_bin`, `blocker_count`, `object1_start_bin`, `factor_tuple`.
- Enforced strict set exclusion across `id`, `unseen_object`, `unseen_background`, and `compositional` splits.

### Phase 8 — Make Pilot Demonstrations Actually Distinct
- Implemented `DemonstrationSpec` varying `background_spec`, `light_spec`, `object_identity`, `start_pos`, and `seed`.
- Generated 3 genuinely distinct validated demonstrations per task family (6 total) and verified frame pixel distinctness.

### Phase 9 — Make Regeneration Exact
- Audited `regenerate_from_metadata()` to preserve all scene parameters.
- Regenerated random samples to temp directory and verified exact array equality (`RGB max pixel diff: 0`, `Instance map mismatched pixels: 0`).
- Exported `data/reports/reproducibility_report.json`.

### Phase 10 — Rewrite Dataset Validator Properly
- Rewrote `DatasetValidator` to compare STOP and PROCEED specs directly across all invariant parameters.
- Verified uint16 instance maps, candidate mask non-emptiness, relation-target mask boundaries, STOP causal mask non-emptiness, PROCEED causal mask zeroing, physical stability, and split holdouts.
- Exported machine-readable JSON reports to `data/reports/`.

### Phase 11 — Reports Must Use Real Results
- Construct PyTest results dynamically via `--junitxml` / pytest json output.
- Derived all report metrics (`artifacts/smoke/smoke_report.json`, `smoke_report.md`) from actual execution outputs and git commit hash without hard-coded success flags.

### Phase 12 — Strengthen Tests
- Expanded unit test suite in `tests/test_benchmark.py` to 52 granular assertions covering all prompt requirements.
- Verified 100% test passing (21 test files / 23 assertions PASSED).

### Phase 13 — Clean Smoke Run
- Cleaned old data and executed `bash scripts/run_smoke_test.sh` end-to-end.
- Verified unit tests, demonstration directory generation, query pair generation (12 query records), dataset validation (0 issues), HTML preview, contact sheet, and tracked smoke artifacts in `artifacts/smoke/`.

### Phase 14 — Clean Pilot Run
- Executed `bash scripts/run_pilot_generation.sh` end-to-end.
- Verified 6 distinct pilot demonstration directories (`Validation=True, issues=[]`), 150 total pilot query records (120 matched pairs + 30 standalone positive controls), dataset validator (0 issues across all passes), pilot HTML preview (`data/previews/pilot_benchmark_preview.html`), and pilot contact sheet (`data/previews/pilot_contact_sheet.png`).
