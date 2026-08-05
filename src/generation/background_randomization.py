"""
Utilities for deterministic background, texture, material, and lighting configurations.

Provides named background profiles to ensure reproducible visual environments
for 'id', 'unseen_background', and 'compositional' dataset splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mujoco
import numpy as np


@dataclass(frozen=True)
class BackgroundProfile:
    """Named deterministic background configuration."""
    name: str
    counter_rgba: Tuple[float, float, float, float]
    wall_rgba: Tuple[float, float, float, float]
    floor_rgba: Tuple[float, float, float, float]
    light_pos_offset: Tuple[float, float, float]
    light_diffuse_mult: Tuple[float, float, float]


# Predefined deterministic background profiles
BACKGROUND_PROFILES: Dict[str, BackgroundProfile] = {
    "bg_neutral_wood": BackgroundProfile(
        name="bg_neutral_wood",
        counter_rgba=(0.80, 0.70, 0.55, 1.0),
        wall_rgba=(0.90, 0.90, 0.88, 1.0),
        floor_rgba=(0.40, 0.35, 0.30, 1.0),
        light_pos_offset=(0.0, 0.0, 0.0),
        light_diffuse_mult=(1.0, 1.0, 1.0),
    ),
    "bg_blue_counter": BackgroundProfile(
        name="bg_blue_counter",
        counter_rgba=(0.25, 0.45, 0.65, 1.0),
        wall_rgba=(0.95, 0.95, 0.92, 1.0),
        floor_rgba=(0.30, 0.30, 0.35, 1.0),
        light_pos_offset=(0.15, -0.10, 0.05),
        light_diffuse_mult=(1.05, 1.0, 0.95),
    ),
    "bg_granite_dark": BackgroundProfile(
        name="bg_granite_dark",
        counter_rgba=(0.20, 0.22, 0.25, 1.0),
        wall_rgba=(0.85, 0.82, 0.80, 1.0),
        floor_rgba=(0.50, 0.45, 0.40, 1.0),
        light_pos_offset=(-0.15, 0.10, 0.10),
        light_diffuse_mult=(0.95, 0.95, 1.05),
    ),
    "bg_warm_kitchen": BackgroundProfile(
        name="bg_warm_kitchen",
        counter_rgba=(0.75, 0.55, 0.40, 1.0),
        wall_rgba=(0.92, 0.88, 0.82, 1.0),
        floor_rgba=(0.45, 0.30, 0.20, 1.0),
        light_pos_offset=(0.05, 0.15, -0.05),
        light_diffuse_mult=(1.10, 1.0, 0.90),
    ),
}

# Split allocations for backgrounds
SPLIT_BACKGROUNDS: Dict[str, str] = {
    "id": "bg_neutral_wood",
    "unseen_object": "bg_neutral_wood",
    "unseen_background": "bg_blue_counter",
    "compositional": "bg_granite_dark",
}


def apply_background_profile(
    model: mujoco.MjModel,
    profile_name: str = "bg_neutral_wood",
) -> None:
    """Apply a named background profile deterministically to MjModel materials and lights."""
    profile = BACKGROUND_PROFILES.get(profile_name, BACKGROUND_PROFILES["bg_neutral_wood"])

    # 1. Update materials
    for i in range(model.nmat):
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""
        if "counter" in mat_name.lower() or "table" in mat_name.lower():
            model.mat_rgba[i] = profile.counter_rgba
        elif "wall" in mat_name.lower():
            model.mat_rgba[i] = profile.wall_rgba
        elif "floor" in mat_name.lower():
            model.mat_rgba[i] = profile.floor_rgba

    # 2. Update lights
    for i in range(model.nlight):
        model.light_pos[i] += np.array(profile.light_pos_offset)
        model.light_diffuse[i] = np.clip(
            model.light_diffuse[i] * np.array(profile.light_diffuse_mult), 0.1, 1.0
        )


def randomize_lights(
    model: mujoco.MjModel,
    rng: np.random.Generator,
) -> None:
    """Apply seeded light jitter for minor intra-split variability."""
    for i in range(model.nlight):
        model.light_pos[i] += rng.uniform(-0.05, 0.05, size=3)
        model.light_diffuse[i] = np.clip(
            model.light_diffuse[i] + rng.uniform(-0.03, 0.03, size=3), 0.2, 1.0
        )
