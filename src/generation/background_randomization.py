"""
Utilities for deterministic background, texture, material, and lighting configurations.

Provides named background profiles, explicit serializable LightSpec and BackgroundSpec,
and deterministic nuisance application to guarantee 100% identical background & lighting across matched pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import mujoco
import numpy as np


@dataclass
class LightSpec:
    """Explicit serializable configuration for all lights in a scene."""
    light_pos_offsets: List[List[float]]
    light_diffuse_mults: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LightSpec:
        return cls(**data)


@dataclass
class BackgroundSpec:
    """Explicit serializable configuration for background materials and lighting."""
    profile_name: str
    counter_rgba: List[float]
    wall_rgba: List[float]
    floor_rgba: List[float]
    light_spec: LightSpec

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["light_spec"] = self.light_spec.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackgroundSpec:
        d = data.copy()
        if isinstance(d.get("light_spec"), dict):
            d["light_spec"] = LightSpec.from_dict(d["light_spec"])
        return cls(**d)


# Predefined deterministic background profiles
BACKGROUND_PROFILES: Dict[str, Dict[str, Any]] = {
    "bg_neutral_wood": {
        "counter_rgba": [0.80, 0.70, 0.55, 1.0],
        "wall_rgba": [0.90, 0.90, 0.88, 1.0],
        "floor_rgba": [0.40, 0.35, 0.30, 1.0],
        "light_pos_offset": [0.0, 0.0, 0.0],
        "light_diffuse_mult": [1.0, 1.0, 1.0],
    },
    "bg_blue_counter": {
        "counter_rgba": [0.25, 0.45, 0.65, 1.0],
        "wall_rgba": [0.95, 0.95, 0.92, 1.0],
        "floor_rgba": [0.30, 0.30, 0.35, 1.0],
        "light_pos_offset": [0.15, -0.10, 0.05],
        "light_diffuse_mult": [1.05, 1.0, 0.95],
    },
    "bg_granite_dark": {
        "counter_rgba": [0.20, 0.22, 0.25, 1.0],
        "wall_rgba": [0.85, 0.82, 0.80, 1.0],
        "floor_rgba": [0.50, 0.45, 0.40, 1.0],
        "light_pos_offset": [-0.15, 0.10, 0.10],
        "light_diffuse_mult": [0.95, 0.95, 1.05],
    },
    "bg_warm_kitchen": {
        "counter_rgba": [0.75, 0.55, 0.40, 1.0],
        "wall_rgba": [0.92, 0.88, 0.82, 1.0],
        "floor_rgba": [0.45, 0.30, 0.20, 1.0],
        "light_pos_offset": [0.05, 0.15, -0.05],
        "light_diffuse_mult": [1.10, 1.0, 0.90],
    },
}

# Split allocations for backgrounds
SPLIT_BACKGROUNDS: Dict[str, str] = {
    "id": "bg_neutral_wood",
    "unseen_object": "bg_neutral_wood",
    "unseen_background": "bg_blue_counter",
    "compositional": "bg_granite_dark",
}


def sample_background_spec(
    profile_name: str,
    rng: np.random.Generator,
    n_lights: int = 2,
) -> BackgroundSpec:
    """Sample a single BackgroundSpec once per matched pair to ensure 100% identical nuisance factors."""
    prof = BACKGROUND_PROFILES.get(profile_name, BACKGROUND_PROFILES["bg_neutral_wood"])

    light_pos_offsets = []
    light_diffuse_mults = []

    base_pos_off = prof["light_pos_offset"]
    base_diff_mult = prof["light_diffuse_mult"]

    for _ in range(n_lights):
        pos_jitter = (np.array(base_pos_off) + rng.uniform(-0.03, 0.03, size=3)).tolist()
        diff_jitter = (np.array(base_diff_mult) + rng.uniform(-0.02, 0.02, size=3)).tolist()
        light_pos_offsets.append(pos_jitter)
        light_diffuse_mults.append(diff_jitter)

    return BackgroundSpec(
        profile_name=profile_name,
        counter_rgba=list(prof["counter_rgba"]),
        wall_rgba=list(prof["wall_rgba"]),
        floor_rgba=list(prof["floor_rgba"]),
        light_spec=LightSpec(
            light_pos_offsets=light_pos_offsets,
            light_diffuse_mults=light_diffuse_mults,
        ),
    )


def apply_background_spec(
    model: mujoco.MjModel,
    spec: BackgroundSpec,
) -> None:
    """Apply the exact same BackgroundSpec to MjModel materials and lights."""
    # 1. Apply materials
    for i in range(model.nmat):
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""
        if "counter" in mat_name.lower() or "table" in mat_name.lower():
            model.mat_rgba[i] = spec.counter_rgba
        elif "wall" in mat_name.lower():
            model.mat_rgba[i] = spec.wall_rgba
        elif "floor" in mat_name.lower():
            model.mat_rgba[i] = spec.floor_rgba

    # 2. Apply lights
    l_spec = spec.light_spec
    for i in range(min(model.nlight, len(l_spec.light_pos_offsets))):
        model.light_pos[i] += np.array(l_spec.light_pos_offsets[i])
        model.light_diffuse[i] = np.clip(
            model.light_diffuse[i] * np.array(l_spec.light_diffuse_mults[i]), 0.1, 1.0
        )
