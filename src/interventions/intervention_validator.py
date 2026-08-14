"""
Authoritative physical simulator validator for causal interventions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import mujoco

from src.interventions.intervention_types import (
    Intervention,
    InterventionOperator,
    InterventionOutcome,
    ObjectPose,
    PrivilegedCategory,
)
from src.validation.occupancy_checks import (
    evaluate_relational_feasibility,
    RelationalFeasibilityResult,
)


EXPECTED_CATEGORY_DELTAS: Dict[PrivilegedCategory, int] = {
    PrivilegedCategory.REPAIR: 1,
    PrivilegedCategory.HARD_NEGATIVE: 0,
    PrivilegedCategory.IRRELEVANT: 0,
    PrivilegedCategory.IDENTITY: 0,
    PrivilegedCategory.HARMFUL: -1,
}


def snapshot_simulator_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Capture exact, bitwise physical state buffer using complete integration state."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    size = mujoco.mj_stateSize(model, spec)
    buf = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(model, data, buf, spec)
    return buf


def restore_simulator_state(model: mujoco.MjModel, data: mujoco.MjData, snapshot: np.ndarray) -> None:
    """Restore exact physical state and recompute forward kinematics/kinetics."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, snapshot, spec)
    mujoco.mj_forward(model, data)


def get_body_freejoint_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> ObjectPose:
    """Read current translation and orientation of a body directly from its freejoint qpos."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise KeyError(f"Body '{body_name}' not found in MuJoCo model.")

    jnt_id = model.body_jntadr[body_id]
    if jnt_id == -1:
        raise ValueError(f"Body '{body_name}' has no joint defined.")

    if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body '{body_name}' joint is type {model.jnt_type[jnt_id]}, expected freejoint.")

    qpos_adr = model.jnt_qposadr[jnt_id]
    pos = tuple(float(x) for x in data.qpos[qpos_adr : qpos_adr + 3])
    quat = tuple(float(x) for x in data.qpos[qpos_adr + 3 : qpos_adr + 7])
    return ObjectPose(position=pos, quaternion_wxyz=quat)


def apply_intervention(model: mujoco.MjModel, data: mujoco.MjData, intervention: Intervention) -> None:
    """Apply physical rigid body intervention through freejoint generalized coordinates.
    
    Mutates data.qpos and zeroes data.qvel for the targeted body only.
    Never writes to derived data.xpos/data.xmat directly.
    """
    if intervention.operator == InterventionOperator.NONE:
        return

    if intervention.operator != InterventionOperator.RELOCATE:
        raise ValueError(f"Unsupported intervention operator: {intervention.operator}")

    body_name = intervention.object_name
    if not body_name:
        raise ValueError("RELOCATE intervention missing object_name")
    if intervention.destination_pose is None:
        raise ValueError("RELOCATE intervention missing destination_pose")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise KeyError(f"Target body '{body_name}' not found in MuJoCo model.")

    jnt_id = model.body_jntadr[body_id]
    if jnt_id == -1:
        raise ValueError(f"Body '{body_name}' has no joint defined.")

    if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body '{body_name}' joint is type {model.jnt_type[jnt_id]}, expected freejoint.")

    qpos_adr = model.jnt_qposadr[jnt_id]
    dof_adr = model.jnt_dofadr[jnt_id]

    pos = intervention.destination_pose.position
    quat = intervention.destination_pose.quaternion_wxyz

    # Apply 7-D translation and orientation to freejoint qpos
    data.qpos[qpos_adr : qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

    # Zero 6-D velocity degrees of freedom for the relocated body
    data.qvel[dof_adr : dof_adr + 6] = 0.0

    # Recompute forward kinematics, site positions, and geom bounding boxes
    mujoco.mj_forward(model, data)


class InterventionValidator:
    """Validator that orchestrates canonical baseline snapshotting, controlled candidate
    intervention execution, live post-state callback invocation, and authoritative observed Delta derivation.
    """

    def __init__(
        self,
        settle_steps: int = 300,
        max_collateral_translation_m: float = 0.05,
        hold_observation_robot: bool = True,
    ):
        self.settle_steps = settle_steps
        self.max_collateral_translation_m = max_collateral_translation_m
        self.hold_observation_robot = hold_observation_robot

    def evaluate_candidate_from_canonical_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        canonical_snapshot: np.ndarray,
        pre_result: RelationalFeasibilityResult,
        intervention: Intervention,
        task_id: str,
        candidate_objects: Optional[List[str]] = None,
        initial_poses: Optional[Dict[str, ObjectPose]] = None,
        on_post_settled_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData, InterventionOutcome], None]] = None,
    ) -> InterventionOutcome:
        """Evaluate a candidate intervention starting strictly from the canonical state snapshot.
        
        The entire evaluation and optional callback are wrapped in a try-finally block that
        guarantees canonical state restoration even if evaluation, rendering, or disk writing fails.
        """
        # Restore canonical state before applying intervention
        restore_simulator_state(model, data, canonical_snapshot)

        try:
            # 1. Apply intervention
            apply_intervention(model, data, intervention)

            # 2. Evaluate post-intervention relational feasibility
            post_result = evaluate_relational_feasibility(
                model, data, task_id, candidate_objects=candidate_objects,
                settle_steps=self.settle_steps, hold_observation_robot=self.hold_observation_robot,
            )
            post_feasible = post_result.feasible
            pre_feasible = pre_result.feasible
            active_culprits_before = tuple(pre_result.active_culprits)
            active_culprits_after = tuple(post_result.active_culprits)

            # 3. Authoritative observed Delta in {-1, 0, +1}
            observed_delta = int(post_feasible) - int(pre_feasible)

            # 4. Audit category consistency
            if intervention.intended_category is not None:
                expected_delta = EXPECTED_CATEGORY_DELTAS.get(intervention.intended_category, observed_delta)
                intended_matches = bool(observed_delta == expected_delta)
            else:
                intended_matches = True

            # 5. Read realized post-settling object pose
            realized_pose = None
            if intervention.operator == InterventionOperator.RELOCATE and intervention.object_name:
                try:
                    realized_pose = get_body_freejoint_pose(model, data, intervention.object_name)
                except (KeyError, ValueError):
                    realized_pose = None

            # 6. Measure collateral displacement of non-intervened candidate objects
            collateral_displacements: Dict[str, float] = {}
            if candidate_objects and initial_poses:
                for name in candidate_objects:
                    if intervention.operator == InterventionOperator.RELOCATE and name == intervention.object_name:
                        continue
                    try:
                        curr_p = np.array(get_body_freejoint_pose(model, data, name).position)
                        init_p = np.array(initial_poses[name].position)
                        disp = float(np.linalg.norm(curr_p - init_p))
                        collateral_displacements[name] = disp
                    except (KeyError, ValueError):
                        pass

            max_collateral = max(collateral_displacements.values()) if collateral_displacements else 0.0
            collateral_acceptable = bool(max_collateral <= self.max_collateral_translation_m)

            # 7. Construct outcome
            diagnostics = {
                "pre_measurements": pre_result.measurements,
                "post_measurements": post_result.measurements,
                "pre_settling_succeeded": pre_result.settling_succeeded,
                "post_settling_succeeded": post_result.settling_succeeded,
                "collateral_displacements": collateral_displacements,
                "max_collateral_displacement": max_collateral,
                "collateral_acceptable": collateral_acceptable,
            }

            outcome = InterventionOutcome(
                intervention=intervention,
                pre_feasible=pre_feasible,
                post_feasible=post_feasible,
                causal_effect=observed_delta,
                intended_effect_matches=intended_matches,
                active_culprits_before=active_culprits_before,
                active_culprits_after=active_culprits_after,
                realized_object_pose=realized_pose,
                validation_diagnostics=diagnostics,
            )

            # 8. Invoke live post-settled callback while simulation state is exactly as evaluated
            if on_post_settled_callback is not None:
                on_post_settled_callback(model, data, outcome)

            return outcome

        finally:
            # Guarantee canonical baseline restoration
            restore_simulator_state(model, data, canonical_snapshot)

    def validate_scene_interventions(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        interventions: List[Intervention],
        task_id: str,
        candidate_objects: Optional[List[str]] = None,
        on_post_settled_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData, InterventionOutcome], None]] = None,
    ) -> List[InterventionOutcome]:
        """Evaluate a set of candidate interventions against a base scene.

        Guarantees candidate order invariance by capturing a canonical integration snapshot
        after initial pre-feasibility settling, and restoring that exact snapshot before
        and after every candidate evaluation.
        """
        # Step 1: Evaluate initial pre-feasibility (runs settle_until_stable)
        pre_result = evaluate_relational_feasibility(
            model, data, task_id, candidate_objects=candidate_objects,
            settle_steps=self.settle_steps, hold_observation_robot=self.hold_observation_robot,
        )

        # Step 2: Record canonical initial poses
        initial_poses: Dict[str, ObjectPose] = {}
        if candidate_objects:
            for name in candidate_objects:
                try:
                    initial_poses[name] = get_body_freejoint_pose(model, data, name)
                except (KeyError, ValueError):
                    pass

        # Step 3: Establish canonical baseline state snapshot
        canonical_snapshot = snapshot_simulator_state(model, data)

        outcomes: List[InterventionOutcome] = []

        # Step 4: Evaluate each candidate from identical canonical state
        for interv in interventions:
            outcome = self.evaluate_candidate_from_canonical_state(
                model=model,
                data=data,
                canonical_snapshot=canonical_snapshot,
                pre_result=pre_result,
                intervention=interv,
                task_id=task_id,
                candidate_objects=candidate_objects,
                initial_poses=initial_poses,
                on_post_settled_callback=on_post_settled_callback,
            )
            outcomes.append(outcome)

        return outcomes

    def validate_single_intervention(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        intervention: Intervention,
        task_id: str,
        candidate_objects: Optional[List[str]] = None,
        on_post_settled_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData, InterventionOutcome], None]] = None,
    ) -> InterventionOutcome:
        """Convenience method for evaluating a single intervention."""
        outcomes = self.validate_scene_interventions(
            model, data, [intervention], task_id,
            candidate_objects=candidate_objects,
            on_post_settled_callback=on_post_settled_callback,
        )
        return outcomes[0]
