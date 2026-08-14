"""Intervention Feature Specification and Dimensionality Constants.

This module locks the canonical tensor dimensions, shapes, and component ordering
for intervention-conditioned learning models (V2/V3).
"""

from dataclasses import dataclass
from typing import Tuple


TEXT_DIM = 384
SCENE_GLOBAL_DIM = 768
SCENE_PATCH_COUNT = 256
SCENE_PATCH_DIM = 768
SCENE_PATCH_SHAPE: Tuple[int, int] = (256, 768)

CANDIDATE_VISUAL_DIM = 768
CURRENT_GEOM_DIM = 3
DEST_GEOM_DIM = 3
OPERATOR_COUNT = 2  # NONE=0, RELOCATE=1

OPERATOR_NONE_IDX = 0
OPERATOR_RELOCATE_IDX = 1

FEATURE_CACHE_SCHEMA_VERSION = "1.1.0"


@dataclass(frozen=True)
class InterventionFeatureSpec:
    """Canonical feature specification for intervention neural models."""

    text_dim: int = TEXT_DIM
    scene_global_dim: int = SCENE_GLOBAL_DIM
    scene_patch_count: int = SCENE_PATCH_COUNT
    scene_patch_dim: int = SCENE_PATCH_DIM
    candidate_visual_dim: int = CANDIDATE_VISUAL_DIM
    current_geom_dim: int = CURRENT_GEOM_DIM
    dest_geom_dim: int = DEST_GEOM_DIM
    operator_count: int = OPERATOR_COUNT

    @property
    def scene_patch_shape(self) -> Tuple[int, int]:
        """Shape tuple for scene patch tokens (256, 768)."""
        return (self.scene_patch_count, self.scene_patch_dim)

    @property
    def raw_intervention_dim(self) -> int:
        """Total dimension of raw intervention descriptor before projection."""
        return self.candidate_visual_dim + self.current_geom_dim + self.dest_geom_dim
