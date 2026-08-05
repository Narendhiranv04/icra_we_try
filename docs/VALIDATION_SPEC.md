# Validation Specification

## Dataset Validator Checks (`DatasetValidator`)
1. **File Integrity**: Verifies existence of all 6 required images per sample (`rgb`, `instance_segmentation`, `candidate_object_mask`, `relation_target_mask`, `causal_violation_mask`, `combined_visualization`).
2. **Causal Violation Mask Invariants**:
   - `STOP`: `causal_violation_mask` must be non-empty (>0 non-zero pixels).
   - `PROCEED`: `causal_violation_mask` must be all zeros.
3. **Privileged Predicates**: Evaluates local frame contact/footprint overlap using `check_lid_occupancy` and `check_target_occupancy`. Intended label must match privileged predicate.
4. **Counterfactual Symmetry**: Verifies matched pair parameters (camera, background, distractor positions, object identities).
5. **Video Integrity**: Validates video readability, frame count (>=60), FPS, and non-static motion between frames.
6. **Leakage & Distribution**: Reports class balance across backgrounds, object identities, position bins, and splits.
