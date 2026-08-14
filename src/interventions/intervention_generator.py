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

    def __init__(self, min_relocation_m: float = 0.03):
        self.min_relocation_m = min_relocation_m

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
        min_relocation_m: Optional[float] = None,
    ) -> List[Intervention]:
        """Generate candidate interventions (repair, hard-negative, irrelevant, identity, harmful)
        for a given base scene.
        
        Args:
            scene_id: Unique base scene identifier
            task_id: Canonical task identifier ('task_1' or 'task_2')
            model: MuJoCo MjModel
            data: MuJoCo MjData (at settled canonical state)
            culprit_names: List of active culprit object names (must be exactly 1 for STOP, empty for PROCEED)
            distractor_names: List of distractor object names
            rng: NumPy random generator
            is_stop_scene: True if base scene is infeasible (STOP), False if feasible (PROCEED)
            min_relocation_m: Minimum physical displacement required for RELOCATE interventions
            
        Returns:
            Shuffled list of Intervention objects with assigned intervention_idx.
        """
        if task_id not in ("task_1", "task_2"):
            raise ValueError(f"Unknown task_id: '{task_id}'. Expected 'task_1' or 'task_2'.")

        min_reloc = min_relocation_m if min_relocation_m is not None else self.min_relocation_m

        # Strict validation of role sets
        if len(culprit_names) != len(set(culprit_names)):
            raise ValueError(f"Duplicate culprit names provided: {culprit_names}")
        if len(distractor_names) != len(set(distractor_names)):
            raise ValueError(f"Duplicate distractor names provided: {distractor_names}")
        overlap = set(culprit_names).intersection(set(distractor_names))
        if overlap:
            raise ValueError(f"Culprit and distractor sets must be disjoint, got overlap: {overlap}")

        if is_stop_scene:
            if len(culprit_names) != 1:
                raise ValueError(f"STOP scene requires exactly 1 culprit name, got {len(culprit_names)}: {culprit_names}")
        else:
            if len(culprit_names) != 0:
                raise ValueError(f"PROCEED scene requires exactly 0 culprits, got {len(culprit_names)}: {culprit_names}")
            if not distractor_names:
                raise ValueError("PROCEED scene requires at least 1 distractor name.")

        # Read canonical settled poses for all candidate objects — fails loudly on missing body/freejoint
        current_poses = {}
        for name in list(culprit_names) + list(distractor_names):
            current_poses[name] = get_body_freejoint_pose(model, data, name)

        def _sample_with_min_displacement(sample_fn, obj_name: str, max_resample_attempts: int = 50) -> np.ndarray:
            curr_p = np.array(current_poses[obj_name].position)
            for _ in range(max_resample_attempts):
                cand_pos = sample_fn()
                if np.linalg.norm(cand_pos - curr_p) >= min_reloc:
                    return cand_pos
            raise RuntimeError(
                f"Failed to sample relocation with displacement >= {min_reloc}m for '{obj_name}' "
                f"after {max_resample_attempts} attempts."
            )

        raw_candidates: List[Intervention] = []

        if task_id == "task_1":
            # --- TASK 1: OPEN(box_B1) ---
            if is_stop_scene:
                culprit = culprit_names[0]
                culprit_quat = current_poses[culprit].quaternion_wxyz

                # 1. Intended REPAIR: Move culprit beside the box
                rep_pos = _sample_with_min_displacement(
                    lambda: sample_position_beside_box(
                        model, data, rng,
                        offset_x=float(rng.uniform(-0.35, -0.25)),
                        offset_y=float(rng.uniform(-0.20, 0.15)),
                        height_above_table=0.04,
                    ),
                    culprit,
                )
                rep_pose = ObjectPose(position=tuple(float(x) for x in rep_pos), quaternion_wxyz=culprit_quat)
                rep_id = generate_deterministic_intervention_id(
                    scene_id, InterventionOperator.RELOCATE, culprit, rep_pose.position, rep_pose.quaternion_wxyz
                )
                raw_candidates.append(Intervention(
                    intervention_id=rep_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=culprit,
                    destination_pose=rep_pose,
                    intended_category=PrivilegedCategory.REPAIR,
                    destination_semantic_tag="safe_region_beside_box",
                ))

                # 2. Intended HARD_NEGATIVE: Move culprit to another position still on the lid
                hn_pos = _sample_with_min_displacement(
                    lambda: sample_position_on_lid(
                        model, data, rng,
                        x_frac=float(rng.uniform(-0.25, 0.25)),
                        y_frac=float(rng.uniform(-0.20, 0.20)),
                        height_above=0.02,
                    ),
                    culprit,
                )
                hn_pose = ObjectPose(position=tuple(float(x) for x in hn_pos), quaternion_wxyz=culprit_quat)
                hn_id = generate_deterministic_intervention_id(
                    scene_id, InterventionOperator.RELOCATE, culprit, hn_pose.position, hn_pose.quaternion_wxyz
                )
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
                    dist_quat = current_poses[dist].quaternion_wxyz
                    irr_pos = _sample_with_min_displacement(
                        lambda: sample_position_beside_box(
                            model, data, rng,
                            offset_x=float(rng.uniform(-0.35, -0.25)),
                            offset_y=float(rng.uniform(-0.20, 0.15)),
                            height_above_table=0.04,
                        ),
                        dist,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(
                        scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz
                    )
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
                chosen_harmful = str(rng.choice(distractor_names))
                chosen_quat = current_poses[chosen_harmful].quaternion_wxyz

                # 1. Intended HARMFUL: Move distractor onto lid
                harm_pos = _sample_with_min_displacement(
                    lambda: sample_position_on_lid(
                        model, data, rng,
                        x_frac=float(rng.uniform(-0.20, 0.20)),
                        y_frac=float(rng.uniform(-0.20, 0.20)),
                        height_above=0.02,
                    ),
                    chosen_harmful,
                )
                harm_pose = ObjectPose(position=tuple(float(x) for x in harm_pos), quaternion_wxyz=chosen_quat)
                harm_id = generate_deterministic_intervention_id(
                    scene_id, InterventionOperator.RELOCATE, chosen_harmful, harm_pose.position, harm_pose.quaternion_wxyz
                )
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
                    dist_quat = current_poses[dist].quaternion_wxyz
                    irr_pos = _sample_with_min_displacement(
                        lambda: sample_position_beside_box(
                            model, data, rng,
                            offset_x=float(rng.uniform(-0.35, -0.25)),
                            offset_y=float(rng.uniform(-0.20, 0.15)),
                            height_above_table=0.04,
                        ),
                        dist,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(
                        scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz
                    )
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
                culprit = culprit_names[0]
                culprit_quat = current_poses[culprit].quaternion_wxyz

                # 1. Intended REPAIR: Move occupant outside target region
                rep_pos = _sample_with_min_displacement(
                    lambda: sample_position_outside_target(
                        model, data, rng,
                        offset_x=float(rng.uniform(0.25, 0.35)),
                        offset_y=float(rng.uniform(-0.15, 0.15)),
                        height_above=0.07,
                    ),
                    culprit,
                )
                rep_pose = ObjectPose(position=tuple(float(x) for x in rep_pos), quaternion_wxyz=culprit_quat)
                rep_id = generate_deterministic_intervention_id(
                    scene_id, InterventionOperator.RELOCATE, culprit, rep_pose.position, rep_pose.quaternion_wxyz
                )
                raw_candidates.append(Intervention(
                    intervention_id=rep_id,
                    operator=InterventionOperator.RELOCATE,
                    object_name=culprit,
                    destination_pose=rep_pose,
                    intended_category=PrivilegedCategory.REPAIR,
                    destination_semantic_tag="outside_target",
                ))

                # 2. Intended HARD_NEGATIVE: Move occupant to another position still inside target
                hn_pos = _sample_with_min_displacement(
                    lambda: sample_position_in_target(
                        model, data, rng,
                        x_frac=float(rng.choice([-1.0, 1.0]) * rng.uniform(0.50, 0.85)),
                        y_frac=float(rng.choice([-1.0, 1.0]) * rng.uniform(0.50, 0.85)),
                        height_above=0.07,
                    ),
                    culprit,
                )
                hn_pose = ObjectPose(position=tuple(float(x) for x in hn_pos), quaternion_wxyz=culprit_quat)
                hn_id = generate_deterministic_intervention_id(
                    scene_id, InterventionOperator.RELOCATE, culprit, hn_pose.position, hn_pose.quaternion_wxyz
                )
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
                    dist_quat = current_poses[dist].quaternion_wxyz
                    irr_pos = _sample_with_min_displacement(
                        lambda: sample_position_outside_target(
                            model, data, rng,
                            offset_x=float(rng.uniform(0.25, 0.35)),
                            offset_y=float(rng.uniform(-0.15, 0.15)),
                            height_above=0.07,
                        ),
                        dist,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(
                        scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz
                    )
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
                chosen_harmful = str(rng.choice(distractor_names))
                chosen_quat = current_poses[chosen_harmful].quaternion_wxyz

                # 1. Intended HARMFUL: Move distractor into target region
                harm_pos = _sample_with_min_displacement(
                    lambda: sample_position_in_target(
                        model, data, rng,
                        x_frac=float(rng.uniform(-0.30, 0.30)),
                        y_frac=float(rng.uniform(-0.30, 0.30)),
                        height_above=0.07,
                    ),
                    chosen_harmful,
                )
                harm_pose = ObjectPose(position=tuple(float(x) for x in harm_pos), quaternion_wxyz=chosen_quat)
                harm_id = generate_deterministic_intervention_id(
                    scene_id, InterventionOperator.RELOCATE, chosen_harmful, harm_pose.position, harm_pose.quaternion_wxyz
                )
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
                    dist_quat = current_poses[dist].quaternion_wxyz
                    irr_pos = _sample_with_min_displacement(
                        lambda: sample_position_outside_target(
                            model, data, rng,
                            offset_x=float(rng.uniform(0.25, 0.35)),
                            offset_y=float(rng.uniform(-0.15, 0.15)),
                            height_above=0.07,
                        ),
                        dist,
                    )
                    irr_pose = ObjectPose(position=tuple(float(x) for x in irr_pos), quaternion_wxyz=dist_quat)
                    irr_id = generate_deterministic_intervention_id(
                        scene_id, InterventionOperator.RELOCATE, dist, irr_pose.position, irr_pose.quaternion_wxyz
                    )
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
