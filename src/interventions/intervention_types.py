"""
Core dataclasses and enums for causal intervention specification and validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class InterventionOperator(str, Enum):
    """Physical operator executable in MuJoCo simulation."""
    NONE = "NONE"          # Identity control (true physical no-op)
    RELOCATE = "RELOCATE"  # Rigid body relocation via freejoint qpos/qvel


class PrivilegedCategory(str, Enum):
    """Privileged intent metadata. Used ONLY for generator auditing, NEVER for observed Delta."""
    REPAIR = "repair"                # Intended Delta = +1 on STOP scene
    HARD_NEGATIVE = "hard_negative"  # Intended Delta = 0 on STOP scene
    IRRELEVANT = "irrelevant"        # Intended Delta = 0 on STOP or PROCEED scene
    IDENTITY = "identity"            # Intended Delta = 0 on STOP or PROCEED scene
    HARMFUL = "harmful"              # Intended Delta = -1 on PROCEED scene


@dataclass(frozen=True)
class ObjectPose:
    """Explicit 3D Cartesian position and canonicalized quaternion orientation."""
    position: Tuple[float, float, float]
    quaternion_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        # Validate position
        if not (isinstance(self.position, (tuple, list)) and len(self.position) == 3):
            raise ValueError(f"Position must be a 3-element sequence, got {self.position}")
        pos = tuple(float(x) for x in self.position)
        if not all(np.isfinite(x) for x in pos):
            raise ValueError(f"Position elements must be finite, got {pos}")
        object.__setattr__(self, "position", pos)

        # Validate and canonicalize quaternion
        if not (isinstance(self.quaternion_wxyz, (tuple, list)) and len(self.quaternion_wxyz) == 4):
            raise ValueError(f"Quaternion must be a 4-element sequence, got {self.quaternion_wxyz}")
        quat = tuple(float(x) for x in self.quaternion_wxyz)
        if not all(np.isfinite(x) for x in quat):
            raise ValueError(f"Quaternion elements must be finite, got {quat}")
        q_norm = float(np.linalg.norm(quat))
        if q_norm < 1e-6:
            raise ValueError(f"Degenerate near-zero quaternion norm: {q_norm}")
        
        # Canonicalize to unit norm
        canonical_quat = tuple(x / q_norm for x in quat)
        object.__setattr__(self, "quaternion_wxyz", canonical_quat)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectPose":
        return cls(
            position=tuple(d["position"]),
            quaternion_wxyz=tuple(d.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0))),
        )


@dataclass(frozen=True)
class Action:
    """Structured task action representation."""
    action_type: str                  # "OPEN", "PLACE"
    target: str                       # "box_B1", "target_region"
    arguments: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "arguments": dict(self.arguments),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        return cls(
            action_type=d["action_type"],
            target=d["target"],
            arguments=dict(d.get("arguments", {})),
        )


@dataclass(frozen=True)
class CandidateObject:
    """Scene entity available for candidate intervention reasoning."""
    name: str
    object_type: str
    role: str                         # "culprit", "distractor", "occupant", "target"
    initial_pose: ObjectPose
    is_culprit: bool                  # PRIVILEGED_GT_ONLY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "object_type": self.object_type,
            "role": self.role,
            "initial_pose": self.initial_pose.to_dict(),
            "is_culprit": self.is_culprit,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateObject":
        return cls(
            name=d["name"],
            object_type=d["object_type"],
            role=d["role"],
            initial_pose=ObjectPose.from_dict(d["initial_pose"]),
            is_culprit=bool(d["is_culprit"]),
        )


@dataclass(frozen=True)
class Intervention:
    """Intervention specification."""
    intervention_id: str                              # Opaque, deterministic hash/id
    operator: InterventionOperator
    object_name: Optional[str] = None                 # None for operator == NONE
    destination_pose: Optional[ObjectPose] = None     # None for operator == NONE
    intended_category: Optional[PrivilegedCategory] = None  # PRIVILEGED_GT_ONLY
    destination_semantic_tag: Optional[str] = None    # PRIVILEGED_GT_ONLY ("safe_region", "still_on_lid", etc.)
    intervention_idx: Optional[int] = None            # Index assigned after candidate set shuffle

    def __post_init__(self):
        if self.operator == InterventionOperator.NONE:
            if self.object_name is not None or self.destination_pose is not None:
                raise ValueError("NONE intervention must have object_name=None and destination_pose=None")
        elif self.operator == InterventionOperator.RELOCATE:
            if self.object_name is None or self.destination_pose is None:
                raise ValueError("RELOCATE intervention requires object_name and destination_pose")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "operator": self.operator.value,
            "object_name": self.object_name,
            "destination_pose": self.destination_pose.to_dict() if self.destination_pose else None,
            "intended_category": self.intended_category.value if self.intended_category else None,
            "destination_semantic_tag": self.destination_semantic_tag,
            "intervention_idx": self.intervention_idx,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Intervention":
        return cls(
            intervention_id=d["intervention_id"],
            operator=InterventionOperator(d["operator"]),
            object_name=d.get("object_name"),
            destination_pose=ObjectPose.from_dict(d["destination_pose"]) if d.get("destination_pose") else None,
            intended_category=PrivilegedCategory(d["intended_category"]) if d.get("intended_category") else None,
            destination_semantic_tag=d.get("destination_semantic_tag"),
            intervention_idx=d.get("intervention_idx"),
        )


@dataclass(frozen=True)
class InterventionOutcome:
    """Complete physical outcome of applying an intervention to a base scene."""
    intervention: Intervention
    pre_feasible: bool
    post_feasible: bool
    causal_effect: int                                # Authoritative: int(post_feasible) - int(pre_feasible) in {-1, 0, +1}
    intended_effect_matches: bool                     # Audit flag: whether observed Delta matches intended_category
    active_culprits_before: Tuple[str, ...] = field(default_factory=tuple)
    active_culprits_after: Tuple[str, ...] = field(default_factory=tuple)
    realized_object_pose: Optional[ObjectPose] = None
    validation_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention": self.intervention.to_dict(),
            "pre_feasible": self.pre_feasible,
            "post_feasible": self.post_feasible,
            "causal_effect": self.causal_effect,
            "intended_effect_matches": self.intended_effect_matches,
            "active_culprits_before": list(self.active_culprits_before),
            "active_culprits_after": list(self.active_culprits_after),
            "realized_object_pose": self.realized_object_pose.to_dict() if self.realized_object_pose else None,
            "validation_diagnostics": self.validation_diagnostics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterventionOutcome":
        return cls(
            intervention=Intervention.from_dict(d["intervention"]),
            pre_feasible=d["pre_feasible"],
            post_feasible=d["post_feasible"],
            causal_effect=d["causal_effect"],
            intended_effect_matches=d["intended_effect_matches"],
            active_culprits_before=tuple(d.get("active_culprits_before", ())),
            active_culprits_after=tuple(d.get("active_culprits_after", ())),
            realized_object_pose=ObjectPose.from_dict(d["realized_object_pose"]) if d.get("realized_object_pose") else None,
            validation_diagnostics=d.get("validation_diagnostics", {}),
        )
