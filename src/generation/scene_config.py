"""
Episode specification and deterministic scene configuration utilities.
"""

from dataclasses import dataclass, field, asdict
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds across random and numpy modules for reproducibility.
    
    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class EpisodeSpec:
    """Validated, deterministic specification for generating a benchmark scene."""
    task_family: str                      # "task_1" or "task_2"
    sample_id: str                        # unique sample id, e.g. "pair_task_1_001_stop"
    pair_id: str                          # counterfactual pair id, e.g. "pair_task_1_001"
    seed: int                             # episode seed
    label: str                            # "STOP" or "PROCEED"
    goal_instruction: str                 # "Open the box." or "Place object1 in the target region."
    background_id: str = "bg_default"
    lighting_id: str = "light_standard"
    object1_type: Optional[str] = None
    object1_start_pose: List[float] = field(default_factory=lambda: [-0.35, -0.20, 0.65])
    blocker_or_occupant_types: List[str] = field(default_factory=list)
    blocker_count: int = 0
    blocker_position_bins: List[str] = field(default_factory=list)
    occupant_position_bin: Optional[str] = None
    box_pose: List[float] = field(default_factory=lambda: [0.52, 0.18, 0.58])
    target_pose: List[float] = field(default_factory=lambda: [-0.10, -0.20, 0.581])
    distractor_identities_and_poses: List[Dict[str, Any]] = field(default_factory=list)
    camera_configuration: str = "robot0:ego_camera"
    split: str = "id"
    relation_before: str = ""
    relation_after: str = ""
    base_scene_id: str = "kitchen_workstation"
    objects_to_spawn: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert EpisodeSpec to dictionary format."""
        return asdict(self)

    def clone_for_counterfactual(self, new_label: str) -> "EpisodeSpec":
        """Clone EpisodeSpec and return a counterpart with only relation-defining objects modified."""
        new_spec = copy_spec(self)
        new_spec.label = new_label
        new_spec.sample_id = f"{self.pair_id}_{new_label.lower()}"
        return new_spec


def copy_spec(spec: EpisodeSpec) -> EpisodeSpec:
    """Return deep copy of EpisodeSpec."""
    return EpisodeSpec(
        task_family=spec.task_family,
        sample_id=spec.sample_id,
        pair_id=spec.pair_id,
        seed=spec.seed,
        label=spec.label,
        goal_instruction=spec.goal_instruction,
        background_id=spec.background_id,
        lighting_id=spec.lighting_id,
        object1_type=spec.object1_type,
        object1_start_pose=list(spec.object1_start_pose),
        blocker_or_occupant_types=list(spec.blocker_or_occupant_types),
        blocker_count=spec.blocker_count,
        blocker_position_bins=list(spec.blocker_position_bins),
        occupant_position_bin=spec.occupant_position_bin,
        box_pose=list(spec.box_pose),
        target_pose=list(spec.target_pose),
        distractor_identities_and_poses=[dict(d) for d in spec.distractor_identities_and_poses],
        camera_configuration=spec.camera_configuration,
        split=spec.split,
        relation_before=spec.relation_before,
        relation_after=spec.relation_after,
        base_scene_id=spec.base_scene_id,
        objects_to_spawn=[dict(o) for o in spec.objects_to_spawn],
    )


class SceneConfigManager:
    """Manager for generating deterministic scene randomizations and tracking counterfactual metadata."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_surface_offset(
        self, center: tuple[float, float, float], radius: float = 0.08
    ) -> list[float]:
        r = self.rng.uniform(0.0, radius)
        theta = self.rng.uniform(0.0, 2 * np.pi)
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        return [center[0] + dx, center[1] + dy, center[2]]
