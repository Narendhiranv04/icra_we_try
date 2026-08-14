"""
Record specifications, schema definitions, and validation helpers for intervention datasets.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.interventions.intervention_types import (
    Action,
    InterventionOperator,
    ObjectPose,
    PrivilegedCategory,
)
from src.generation.background_randomization import BackgroundSpec


SCHEMA_VERSION = "2.0.0"

# Strict allowlist of permissible keys under model_inputs
MODEL_INPUTS_ALLOWED_KEYS: Set[str] = {
    "task_id",
    "instruction",
    "action",
    "pre_rgb_path",
    "candidate_object_crop_path",
    "intervention_operator",
    "current_geometry",
    "destination_geometry",
    "demonstration_reference",
}

# Forbidden semantic terms that must NEVER leak into model_inputs (keys or string values)
FORBIDDEN_LEAKAGE_TERMS: Set[str] = {
    "culprit",
    "is_culprit",
    "repair",
    "harmful",
    "hard_negative",
    "irrelevant",
    "intended",
    "post_feasible",
    "causal_effect",
    "post_rgb",
    "active_culprits",
    "destination_semantic_tag",
}


@dataclass(frozen=True)
class GeometryRelativeSpec:
    """Explicit Cartesian and relative geometry without 16-D neural tensorization."""
    world_position: Tuple[float, float, float]
    world_quaternion_wxyz: Tuple[float, float, float, float]
    reference_frame: str
    relative_position: Tuple[float, float, float]
    relative_quaternion_wxyz: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "world_position": list(self.world_position),
            "world_quaternion_wxyz": list(self.world_quaternion_wxyz),
            "reference_frame": self.reference_frame,
            "relative_position": list(self.relative_position),
        }
        if self.relative_quaternion_wxyz is not None:
            d["relative_quaternion_wxyz"] = list(self.relative_quaternion_wxyz)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryRelativeSpec":
        return cls(
            world_position=tuple(float(x) for x in d["world_position"]),
            world_quaternion_wxyz=tuple(float(x) for x in d["world_quaternion_wxyz"]),
            reference_frame=str(d["reference_frame"]),
            relative_position=tuple(float(x) for x in d["relative_position"]),
            relative_quaternion_wxyz=tuple(float(x) for x in d["relative_quaternion_wxyz"]) if "relative_quaternion_wxyz" in d else None,
        )


@dataclass(frozen=True)
class SpawnedObjectSpec:
    """Exact physical resolved specification of an object in a scene."""
    name: str
    object_type: str
    position: Tuple[float, float, float]
    quaternion_wxyz: Tuple[float, float, float, float]
    semantic_role: str  # "action_subject", "culprit", "distractor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "object_type": self.object_type,
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "semantic_role": self.semantic_role,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpawnedObjectSpec":
        return cls(
            name=str(d["name"]),
            object_type=str(d["object_type"]),
            position=tuple(float(x) for x in d["position"]),
            quaternion_wxyz=tuple(float(x) for x in d.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0))),
            semantic_role=str(d["semantic_role"]),
        )


@dataclass(frozen=True)
class InterventionSceneSpec:
    """Deterministic, user/config-level requested base-scene specification."""
    scene_id: str
    task_id: str                          # "task_1" or "task_2"
    intended_base_state: str              # "STOP" or "PROCEED"
    instruction: str                      # e.g. "Open the box."
    action: Action
    action_subject_name: Optional[str]    # For Task 2: e.g. "pick_can" (excluded from obstruction candidates)
    intended_culprit_type: Optional[str]  # e.g. "coffee_can" (for STOP)
    intended_distractor_types: Tuple[str, ...]
    seed: int
    split: str = "id"
    background_profile: str = "bg_neutral_wood"
    box_pose: Optional[Tuple[float, float, float]] = None
    box_quat: Optional[Tuple[float, float, float, float]] = None
    target_region_pos: Optional[Tuple[float, float, float]] = None
    target_region_quat: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "intended_base_state": self.intended_base_state,
            "instruction": self.instruction,
            "action": self.action.to_dict(),
            "action_subject_name": self.action_subject_name,
            "intended_culprit_type": self.intended_culprit_type,
            "intended_distractor_types": list(self.intended_distractor_types),
            "seed": self.seed,
            "split": self.split,
            "background_profile": self.background_profile,
            "box_pose": list(self.box_pose) if self.box_pose else None,
            "box_quat": list(self.box_quat) if self.box_quat else None,
            "target_region_pos": list(self.target_region_pos) if self.target_region_pos else None,
            "target_region_quat": list(self.target_region_quat) if self.target_region_quat else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterventionSceneSpec":
        return cls(
            scene_id=d["scene_id"],
            task_id=d["task_id"],
            intended_base_state=d["intended_base_state"],
            instruction=d["instruction"],
            action=Action.from_dict(d["action"]),
            action_subject_name=d.get("action_subject_name"),
            intended_culprit_type=d.get("intended_culprit_type"),
            intended_distractor_types=tuple(d.get("intended_distractor_types", ())),
            seed=d["seed"],
            split=d.get("split", "id"),
            background_profile=d.get("background_profile", "bg_neutral_wood"),
            box_pose=tuple(d["box_pose"]) if d.get("box_pose") else None,
            box_quat=tuple(d["box_quat"]) if d.get("box_quat") else None,
            target_region_pos=tuple(d["target_region_pos"]) if d.get("target_region_pos") else None,
            target_region_quat=tuple(d["target_region_quat"]) if d.get("target_region_quat") else None,
        )


@dataclass(frozen=True)
class ResolvedSceneSpec:
    """Complete physical specification of an accepted base scene sufficient for exact reconstruction."""
    scene_id: str
    task_id: str
    intended_base_state: str
    instruction: str
    action: Action
    action_subject_name: Optional[str]
    obstruction_candidate_names: Tuple[str, ...]
    all_visible_object_names: Tuple[str, ...]
    spawned_objects: Tuple[SpawnedObjectSpec, ...]
    background_spec: BackgroundSpec
    observation_rig_id: str
    robot_base_pose: str
    requested_seed: int
    realized_base_attempt: int
    actual_base_seed: int
    box_pose: Optional[Tuple[float, float, float]] = None
    box_quat: Optional[Tuple[float, float, float, float]] = None
    target_region_pos: Optional[Tuple[float, float, float]] = None
    target_region_quat: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "intended_base_state": self.intended_base_state,
            "instruction": self.instruction,
            "action": self.action.to_dict(),
            "action_subject_name": self.action_subject_name,
            "obstruction_candidate_names": list(self.obstruction_candidate_names),
            "all_visible_object_names": list(self.all_visible_object_names),
            "spawned_objects": [obj.to_dict() for obj in self.spawned_objects],
            "background_spec": self.background_spec.to_dict(),
            "observation_rig_id": self.observation_rig_id,
            "robot_base_pose": self.robot_base_pose,
            "requested_seed": self.requested_seed,
            "realized_base_attempt": self.realized_base_attempt,
            "actual_base_seed": self.actual_base_seed,
            "box_pose": list(self.box_pose) if self.box_pose else None,
            "box_quat": list(self.box_quat) if self.box_quat else None,
            "target_region_pos": list(self.target_region_pos) if self.target_region_pos else None,
            "target_region_quat": list(self.target_region_quat) if self.target_region_quat else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResolvedSceneSpec":
        return cls(
            scene_id=d["scene_id"],
            task_id=d["task_id"],
            intended_base_state=d["intended_base_state"],
            instruction=d["instruction"],
            action=Action.from_dict(d["action"]),
            action_subject_name=d.get("action_subject_name"),
            obstruction_candidate_names=tuple(d.get("obstruction_candidate_names", ())),
            all_visible_object_names=tuple(d.get("all_visible_object_names", ())),
            spawned_objects=tuple(SpawnedObjectSpec.from_dict(o) for o in d.get("spawned_objects", ())),
            background_spec=BackgroundSpec.from_dict(d["background_spec"]),
            observation_rig_id=d["observation_rig_id"],
            robot_base_pose=d["robot_base_pose"],
            requested_seed=int(d["requested_seed"]),
            realized_base_attempt=int(d["realized_base_attempt"]),
            actual_base_seed=int(d["actual_base_seed"]),
            box_pose=tuple(d["box_pose"]) if d.get("box_pose") else None,
            box_quat=tuple(d["box_quat"]) if d.get("box_quat") else None,
            target_region_pos=tuple(d["target_region_pos"]) if d.get("target_region_pos") else None,
            target_region_quat=tuple(d["target_region_quat"]) if d.get("target_region_quat") else None,
        )


@dataclass(frozen=True)
class InterventionDatasetRecord:
    """Single line record in manifest.jsonl enforcing strict three-way information partitioning."""
    schema_version: str
    record_id: str
    scene_id: str
    intervention_id: str
    model_inputs: Dict[str, Any]
    supervision_targets: Dict[str, Any]
    privileged_metadata: Dict[str, Any]

    def __post_init__(self):
        # Enforce exact allowlist on model_inputs
        actual_keys = set(self.model_inputs.keys())
        unknown_keys = actual_keys - MODEL_INPUTS_ALLOWED_KEYS
        if unknown_keys:
            raise ValueError(f"Disallowed keys found in model_inputs: {unknown_keys}")

        # Recursively assert no forbidden leakage terms exist inside model_inputs
        self._check_forbidden_leakage(self.model_inputs)

    @staticmethod
    def _check_forbidden_leakage(data: Any, path: str = "model_inputs") -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                k_lower = str(k).lower()
                for term in FORBIDDEN_LEAKAGE_TERMS:
                    if term in k_lower:
                        raise ValueError(f"Forbidden term '{term}' leaked into model_inputs key '{path}.{k}'")
                InterventionDatasetRecord._check_forbidden_leakage(v, f"{path}.{k}")
        elif isinstance(data, (list, tuple)):
            for idx, item in enumerate(data):
                InterventionDatasetRecord._check_forbidden_leakage(item, f"{path}[{idx}]")
        elif isinstance(data, str):
            # Check string value for category leakage (exclude instance names like 'blocker1' or 'mug')
            val_lower = data.lower()
            for term in ["repair", "harmful", "hard_negative", "irrelevant", "culprit"]:
                # If term appears as a standalone substring in a non-instance path
                if f"_{term}" in val_lower or f"{term}_" in val_lower or val_lower == term:
                    raise ValueError(f"Forbidden term '{term}' leaked into model_inputs string value at '{path}': {data}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "scene_id": self.scene_id,
            "intervention_id": self.intervention_id,
            "model_inputs": self.model_inputs,
            "supervision_targets": self.supervision_targets,
            "privileged_metadata": self.privileged_metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterventionDatasetRecord":
        return cls(
            schema_version=str(d["schema_version"]),
            record_id=str(d["record_id"]),
            scene_id=str(d["scene_id"]),
            intervention_id=str(d["intervention_id"]),
            model_inputs=dict(d["model_inputs"]),
            supervision_targets=dict(d["supervision_targets"]),
            privileged_metadata=dict(d["privileged_metadata"]),
        )
