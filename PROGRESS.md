# Implementation Progress: Relational Precondition MuJoCo Benchmark v0.1

## Phase 0: Audit Report

### Environment & Project State
- **Workspace**: `/home/naren/RA_iiith_new`
- **Reference Project**: `/home/naren/RA_iiith` (UNMODIFIED)
- **Git Remote**: `https://github.com/Narendhiranv04/icra_we_try.git` (branch `main`, clean tree)
- **Python Environment**: `/home/naren/miniconda3/bin/python` (Python 3.13.11)
- **Installed Packages**: `mujoco` 3.10.0, `gymnasium-robotics` (with Fetch assets), `numpy`, `scipy`, `PIL`, `cv2`, `imageio`, `pytest`, `pyyaml`.

### Audit Findings & Identified Known Issues
1. **Known Issue 1 (Demonstration Videos)**:
   - `BoxOpenExecutor` in `src/tasks/open_box.py` directly steps `data.qpos` of `B1_lid_hinge` without any robot.
   - `PlaceObjectExecutor` in `src/tasks/place_object.py` directly steps `data.qpos` of `object1` along a synthetic parabolic arc without any robot.
   - Demonstration videos in `data/demos/` show synthetic joint teleportation rather than robot manipulation.

2. **Known Issue 2 (Task 2 Target Region Inconsistency)**:
   - Discrepancy in target coordinates across the codebase:
     - `PlaceObjectExecutor`: `(0.25, 0.45, 0.82)`
     - `check_target_occupancy`: `(0.0, 0.20, 0.57)`
     - `CounterfactualPairGenerator` STOP: `[0.0, 0.20, 0.64]`
     - `CounterfactualPairGenerator` PROCEED: `[-0.35, 0.20, 0.64]`
   - Missing explicit single source of truth in MJCF (`target_region_body`, `target_region_geom`, `target_region_site`).

3. **Known Issue 3 (Scene Variations)**:
   - Generator currently uses hardcoded object positions rather than consuming `EpisodeSpec` or full variation configurations (blocker counts, position bins, background variation, distractor placement).

4. **Known Issue 4 (Annotation & Mask Semantics)**:
   - Current implementation generates generic `culprit_mask` and `lid_mask`/`target_mask`.
   - Missing separate `instance_segmentation.png`, `candidate_object_mask.png`, `relation_target_mask.png`, and `causal_violation_mask.png` (empty for PROCEED, non-empty for STOP).

5. **Known Issue 5 (Privileged Predicates)**:
   - Occupancy checks use broad hardcoded bounding boxes (`dx < 0.22`, `dy < 0.22`) and radial approximations rather than local frame overlap, contact, and surface footprint validation.

6. **Known Issue 6 & 7 (Configuration & Environment Portability)**:
   - Shell scripts hardcode `/home/naren/miniconda3/bin/python`.
   - `pilot.yaml` contains excessive default counts (`num_demos_per_task: 50`, `num_pairs_per_task: 100`) rather than benchmark v0.1 spec (3 demos, ~60 pairs per task family).

---

## Progress Overview

- **Current Phase**: Phase 3: Episode Specification and Query Generation
- **Completed Work**: Phase 0 Audit, Phase 1 Target-Region Correction, and Phase 2 Genuine Robot Demonstrations completed.
  - Integrated Fetch mobile manipulator (`robot0:...`) with 7-DOF arm, position actuators, and `VerticalIK` inverse kinematics solver into `src/environment/robot_integration.py` and `SceneBuilder`.
  - Replaced frame-by-frame joint teleportation in `src/tasks/open_box.py` (`BoxOpenExecutor`) and `src/tasks/place_object.py` (`PlaceObjectExecutor`) with genuine robot arm movement, trajectory stepping, end-effector alignment, and contact/weld grasp assistance.
  - Generated successful demonstration videos `data/demos/demo_task1_smoke.mp4` and `data/demos/demo_task2_smoke.mp4` (120 frames each @ 30 FPS).
- **Files Modified**: `src/environment/robot_integration.py`, `src/environment/model_loading.py`, `src/environment/scene_builder.py`, `src/tasks/open_box.py`, `src/tasks/place_object.py`, `src/generation/demonstration_generator.py`, `PROGRESS.md`.
- **Commands Run**:
  - `pytest` (PASSED 4/4 tests)
  - `DemonstrationGenerator` execution (PASSED: 2 genuine robot MP4 clips rendered)
- **Exact Next Step**: Phase 3: Implement EpisodeSpec dataclass and deterministic query scene generation with scene variations (blocker count, blocker/occupant identities, positions, backgrounds, positive relational controls).
- **Known Limitations**: Scene variations and EpisodeSpec need full integration into query generator (Phase 3).
