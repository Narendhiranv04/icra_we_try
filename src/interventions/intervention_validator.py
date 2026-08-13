"""
Authoritative physical simulator validator for causal interventions.
"""

from typing import Any, Dict, List, Optional, Tuple
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
    intervention execution, and authoritative observed Delta derivation.
    """

    def __init__(self, settle_steps: int = 300):
        self.settle_steps = settle_steps

    def validate_scene_interventions(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        interventions: List[Intervention],
        task_id: str,
        candidate_objects: Optional[List[str]] = None,
    ) -> List[InterventionOutcome]:
        """Evaluate a set of candidate interventions against a base scene.
        
        Guarantees candidate order invariance by capturing a canonical integration snapshot
        after initial pre-feasibility settling, and restoring that exact snapshot before
        every candidate evaluation.
        """
        # Step 1: Evaluate initial pre-feasibility (runs settle_until_stable)
        pre_result = evaluate_relational_feasibility(
            model, data, task_id, candidate_objects=candidate_objects, settle_steps=self.settle_steps
        )
        pre_feasible = pre_result.feasible
        active_culprits_before = tuple(pre_result.active_culprits)

        # Step 2: Establish canonical baseline state snapshot
        canonical_snapshot = snapshot_simulator_state(model, data)

        outcomes: List[InterventionOutcome] = []

        # Step 3: Evaluate each candidate from identical canonical state
        for interv in interventions:
            # A. Restore canonical baseline
            restore_simulator_state(model, data, canonical_snapshot)

            # B. Apply candidate intervention
            apply_intervention(model, data, interv)

            # C. Evaluate post-intervention relational feasibility
            post_result = evaluate_relational_feasibility(
                model, data, task_id, candidate_objects=candidate_objects, settle_steps=self.settle_steps
            )
            post_feasible = post_result.feasible
            active_culprits_after = tuple(post_result.active_culprits)

            # D. Authoritative observed Delta in {-1, 0, +1}
            observed_delta = int(post_feasible) - int(pre_feasible)

            # E. Audit category consistency
            if interv.intended_category is not None:
                expected_delta = EXPECTED_CATEGORY_DELTAS.get(interv.intended_category, observed_delta)
                intended_matches = bool(observed_delta == expected_delta)
            else:
                intended_matches = True

            # F. Read realized post-settling object pose
            realized_pose = None
            if interv.operator == InterventionOperator.RELOCATE and interv.object_name:
                try:
                    realized_pose = get_body_freejoint_pose(model, data, interv.object_name)
                except (KeyError, ValueError):
                    realized_pose = None

            # G. Diagnostics
            diagnostics = {
                "pre_measurements": pre_result.measurements,
                "post_measurements": post_result.measurements,
                "pre_settling_succeeded": pre_result.settling_succeeded,
                "post_settling_succeeded": post_result.settling_succeeded,
            }

            outcome = InterventionOutcome(
                intervention=interv,
                pre_feasible=pre_feasible,
                post_feasible=post_feasible,
                causal_effect=observed_delta,
                intended_effect_matches=intended_matches,
                active_culprits_before=active_culprits_before,
                active_culprits_after=active_culprits_after,
                realized_object_pose=realized_pose,
                validation_diagnostics=diagnostics,
            )
            outcomes.append(outcome)

        # Step 4: Restore canonical snapshot before returning
        restore_simulator_state(model, data, canonical_snapshot)

        return outcomes

    def validate_single_intervention(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        intervention: Intervention,
        task_id: str,
        candidate_objects: Optional[List[str]] = None,
    ) -> InterventionOutcome:
        """Convenience method for evaluating a single intervention."""
        outcomes = self.validate_scene_interventions(
            model, data, [intervention], task_id, candidate_objects=candidate_objects
        )
        return outcomes[0]
