"""
Deterministic candidate intervention generator for robot manipulation tasks.
"""

from dataclasses import replace
import hashlib
from typing import List, Optional, Tuple, Union
import numpy as np
import mujoco

from src.interventions.intervention_types import (
    Intervention,
    InterventionOperator,
    ObjectPose,
    PrivilegedCategory,
)
from src.interventions.intervention_validator import get_body_freejoint_pose
from src.environment.scene_utils import (
    sample_position_on_lid,
    sample_position_beside_box,
    sample_position_in_target,
    sample_position_outside_target,
)


def generate_deterministic_intervention_id(
    scene_id: str,
    operator: InterventionOperator,
    object_name: Optional[str] = None,
    destination_pos: Optional[Tuple[float, float, float]] = None,
    destination_quat: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """Generate deterministic, category-opaque intervention ID from physical payload."""
    pos_str = f"{destination_pos[0]:.4f},{destination_pos[1]:.4f},{destination_pos[2]:.4f}" if destination_pos else "none"
    quat_str = f"{destination_quat[0]:.4f},{destination_quat[1]:.4f},{destination_quat[2]:.4f},{destination_quat[3]:.4f}" if destination_quat else "none"
    obj_str = str(object_name) if object_name else "none"
    payload = f"{scene_id}:{operator.value}:{obj_str}:{pos_str}:{quat_str}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"int_{digest}"


class InterventionGenerator:
    """Generates physically grounded candidate interventions using local reference frames."""

    def __init__(self):
        pass

    def generate_candidates(
        self,
        scene_id: str,
        task_id: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        culprit_names: List[str],
        distractor_names: List[str],
        rng: np.random.Generator,
        is_stop_scene: bool = True,
    ) -> List[Intervention]:
        """Generate candidate interventions (repair, hard-negative, irrelevant, identity, harmful)
        for a given base scene.
        
        Args:
            scene_id: Unique base scene identifier
            task_id: Canonical task identifier ('task_1' or 'task_2')
            model: MuJoCo MjModel
            data: MuJoCo MjData (at settled canonical state)
            culprit_names: List of active culprit object names (empty for PROCEED scene)
            distractor_names: List of distractor object names
            rng: NumPy random generator
            is_stop_scene: True if base scene is infeasible (STOP), False if feasible (PROCEED)
            
        Returns:
            Shuffled list of Intervention objects with assigned intervention_idx.
        """
        if task_id not in ("task_1", "task_2"):
            raise ValueError(f"Unknown task_id: '{task_id}'. Expected 'task_1' or 'task_2'.")

        # Cache canonical settled orientations for all movable objects
        settled_quats = {}
        for name in list(culprit_names) + list(distractor_names):
            try:
                settled_quats[name] = get_body_freejoint_pose(model, data, name).quaternion_wxyz
            except (KeyError, ValueError):
                settled_quats[name] = (1.0, 0.0, 0.0, 0.0)

        raw_candidates: List[Intervention] = []

        if task_id == "task_1":
            # --- TASK 1: OPEN(box_B1) ---
            if is_stop_scene:
                if not culprit_names:
                    raise ValueError("Task 1 STOP scene requires at least 1 culprit name.")
                culprit = culprit_names[0]
                culprit_quat = settled_quats.get(culprit, (1.0, 0.0, 0.0, 0.0))

                # 1. Intended REPAIR: Move culprit beside the box
                rep_pos = sample_position_beside_box(
                    model, data, rng,
                    offset_x=float(rng.uniform(-0.35, -0.25)),
                    offset_y=float(rng.uniform(-0.20, 0.15)),
                    height_above_table=0.04,
                )
                rep_pose = ObjectPose(position=tuple(float(x) for x in rep_pos), quaternion_wxyz=culprit_quat)
                rep_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, culprit, rep_pose.position, rep_pose.quaternion_wxyz)
                raw_candidates.append(Intervention(
                    intervention_id=rep_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=culprit,
                    destination_pose=rep_pose,
                    intended_category=PrivilegedCategory.REPAIR,
                    destination_semantic_tag="safe_region_beside_box",
                ))

                # 2. Intended HARD_NEGATIVE: Move culprit to another position still on the lid
                hn_pos = sample_position_on_lid(
                    model, data, rng,
                    x_frac=float(rng.uniform(-0.25, 0.25)),
                    y_frac=float(rng.uniform(-0.20, 0.20)),
                    height_above=0.02,
                )
                hn_pose = ObjectPose(position=tuple(float(x) for x in hn_pos), quaternion_wxyz=culprit_quat)
                hn_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, culprit, hn_pose.position, hn_pose.quaternion_wxyz)
                raw_candidates.append(Intervention(
                    intervention_id=hn_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=culprit,
                    destination_pose=hn_pose,
                    intended_category=PrivilegedCategory.HARD_NEGATIVE,
                    destination_semantic_tag="still_on_lid",
                ))

                # 3. Intended IRRELEVANT: Move each distractor to a clear location
                for dist in distractor_names:
                    dist_quat = settled_quats.get(dist, (1.0, 0.0, 0.0, 0.0))
                    irr_pos = sample_position_beside_box(
                        model, data, rng,
                        offset_x=float(rng.uniform(-0.35, -0.25)),
                        offset_y=float(rng.uniform(-0.20, 0.15)),
                        height_above_table=0.04,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz)
                    raw_candidates.append(Intervention(
                        intervention_id=irr_id,
                        operator=InterventionOperator.RELOCATE,
                        object_name=dist,
                        destination_pose=irr_pose,
                        intended_category=PrivilegedCategory.IRRELEVANT,
                        destination_semantic_tag="safe_region_beside_box",
                    ))

                # 4. Intended IDENTITY: NONE control
                none_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.NONE)
                raw_candidates.append(Intervention(
                    intervention_id=none_id,
                    operator=InterventionOperator.NONE,
                    intended_category=PrivilegedCategory.IDENTITY,
                    destination_semantic_tag="identity",
                ))

            else:
                # PROCEED scene
                if not distractor_names:
                    raise ValueError("Task 1 PROCEED scene requires at least 1 distractor name.")
                
                # Pick one distractor for harmful relocation
                chosen_harmful = str(rng.choice(distractor_names))
                chosen_quat = settled_quats.get(chosen_harmful, (1.0, 0.0, 0.0, 0.0))

                # 1. Intended HARMFUL: Move distractor onto lid
                harm_pos = sample_position_on_lid(
                    model, data, rng,
                    x_frac=float(rng.uniform(-0.20, 0.20)),
                    y_frac=float(rng.uniform(-0.20, 0.20)),
                    height_above=0.02,
                )
                harm_pose = ObjectPose(position=tuple(float(x) for x in harm_pos), quaternion_wxyz=chosen_quat)
                harm_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, chosen_harmful, harm_pose.position, harm_pose.quaternion_wxyz)
                raw_candidates.append(Intervention(
                    intervention_id=harm_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=chosen_harmful,
                    destination_pose=harm_pose,
                    intended_category=PrivilegedCategory.HARMFUL,
                    destination_semantic_tag="onto_lid",
                ))

                # 2. Intended IRRELEVANT: Move each distractor to another clear location
                for dist in distractor_names:
                    dist_quat = settled_quats.get(dist, (1.0, 0.0, 0.0, 0.0))
                    irr_pos = sample_position_beside_box(
                        model, data, rng,
                        offset_x=float(rng.uniform(-0.35, -0.25)),
                        offset_y=float(rng.uniform(-0.20, 0.15)),
                        height_above_table=0.04,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz)
                    raw_candidates.append(Intervention(
                        intervention_id=irr_id,
                        operator=InterventionOperator.RELOCATE,
                        object_name=dist,
                        destination_pose=irr_pose,
                        intended_category=PrivilegedCategory.IRRELEVANT,
                        destination_semantic_tag="safe_region_beside_box",
                    ))

                # 3. Intended IDENTITY: NONE control
                none_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.NONE)
                raw_candidates.append(Intervention(
                    intervention_id=none_id,
                    operator=InterventionOperator.NONE,
                    intended_category=PrivilegedCategory.IDENTITY,
                    destination_semantic_tag="identity",
                ))

        elif task_id == "task_2":
            # --- TASK 2: PLACE(object1, target_region) ---
            if is_stop_scene:
                if not culprit_names:
                    raise ValueError("Task 2 STOP scene requires at least 1 culprit name.")
                culprit = culprit_names[0]
                culprit_quat = settled_quats.get(culprit, (1.0, 0.0, 0.0, 0.0))

                # 1. Intended REPAIR: Move occupant outside target region
                rep_pos = sample_position_outside_target(
                    model, data, rng,
                    offset_x=float(rng.uniform(0.25, 0.35)),
                    offset_y=float(rng.uniform(-0.15, 0.15)),
                    height_above=0.07,
                )
                rep_pose = ObjectPose(position=tuple(float(x) for x in rep_pos), quaternion_wxyz=culprit_quat)
                rep_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, culprit, rep_pose.position, rep_pose.quaternion_wxyz)
                raw_candidates.append(Intervention(
                    intervention_id=rep_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=culprit,
                    destination_pose=rep_pose,
                    intended_category=PrivilegedCategory.REPAIR,
                    destination_semantic_tag="outside_target",
                ))

                # 2. Intended HARD_NEGATIVE: Move occupant to another position still inside target
                hn_pos = sample_position_in_target(
                    model, data, rng,
                    x_frac=float(rng.uniform(-0.30, 0.30)),
                    y_frac=float(rng.uniform(-0.30, 0.30)),
                    height_above=0.07,
                )
                hn_pose = ObjectPose(position=tuple(float(x) for x in hn_pos), quaternion_wxyz=culprit_quat)
                hn_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, culprit, hn_pose.position, hn_pose.quaternion_wxyz)
                raw_candidates.append(Intervention(
                    intervention_id=hn_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=culprit,
                    destination_pose=hn_pose,
                    intended_category=PrivilegedCategory.HARD_NEGATIVE,
                    destination_semantic_tag="still_in_target",
                ))

                # 3. Intended IRRELEVANT: Move each distractor to another location outside target
                for dist in distractor_names:
                    dist_quat = settled_quats.get(dist, (1.0, 0.0, 0.0, 0.0))
                    irr_pos = sample_position_outside_target(
                        model, data, rng,
                        offset_x=float(rng.uniform(0.25, 0.35)),
                        offset_y=float(rng.uniform(-0.15, 0.15)),
                        height_above=0.07,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz)
                    raw_candidates.append(Intervention(
                        intervention_id=irr_id,
                        operator=InterventionOperator.RELOCATE,
                        object_name=dist,
                        destination_pose=irr_pose,
                        intended_category=PrivilegedCategory.IRRELEVANT,
                        destination_semantic_tag="outside_target",
                    ))

                # 4. Intended IDENTITY: NONE control
                none_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.NONE)
                raw_candidates.append(Intervention(
                    intervention_id=none_id,
                    operator=InterventionOperator.NONE,
                    intended_category=PrivilegedCategory.IDENTITY,
                    destination_semantic_tag="identity",
                ))

            else:
                # PROCEED scene
                if not distractor_names:
                    raise ValueError("Task 2 PROCEED scene requires at least 1 distractor name.")
                
                # Pick one distractor for harmful relocation
                chosen_harmful = str(rng.choice(distractor_names))
                chosen_quat = settled_quats.get(chosen_harmful, (1.0, 0.0, 0.0, 0.0))

                # 1. Intended HARMFUL: Move distractor into target region
                harm_pos = sample_position_in_target(
                    model, data, rng,
                    x_frac=float(rng.uniform(-0.30, 0.30)),
                    y_frac=float(rng.uniform(-0.30, 0.30)),
                    height_above=0.07,
                )
                harm_pose = ObjectPose(position=tuple(float(x) for x in harm_pos), quaternion_wxyz=chosen_quat)
                harm_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, chosen_harmful, harm_pose.position, harm_pose.quaternion_wxyz)
                raw_candidates.append(Intervention(
                    intervention_id=harm_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=chosen_harmful,
                    destination_pose=harm_pose,
                    intended_category=PrivilegedCategory.HARMFUL,
                    destination_semantic_tag="into_target",
                ))

                # 2. Intended IRRELEVANT: Move each distractor to another location outside target
                for dist in distractor_names:
                    dist_quat = settled_quats.get(dist, (1.0, 0.0, 0.0, 0.0))
                    irr_pos = sample_position_outside_target(
                        model, data, rng,
                        offset_x=float(rng.uniform(0.25, 0.35)),
                        offset_y=float(rng.uniform(-0.15, 0.15)),
                        height_above=0.07,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz)
                    raw_candidates.append(Intervention(
                        intervention_id=irr_id,
                        operator=InterventionOperator.RELOCATE,
                        object_name=dist,
                        destination_pose=irr_pose,
                        intended_category=PrivilegedCategory.IRRELEVANT,
                        destination_semantic_tag="outside_target",
                    ))

                # 3. Intended IDENTITY: NONE control
                none_id = generate_deterministic_intervention_id(scene_id, InterventionOperator.NONE)
                raw_candidates.append(Intervention(
                    intervention_id=none_id,
                    operator=InterventionOperator.NONE,
                    intended_category=PrivilegedCategory.IDENTITY,
                    destination_semantic_tag="identity",
                ))

        # Deterministic shuffle using permutation indices
        indices = rng.permutation(len(raw_candidates))
        shuffled_candidates = [
            replace(raw_candidates[int(idx)], intervention_idx=i)
            for i, idx in enumerate(indices)
        ]

        return shuffled_candidates
