"""
Utilities for scene background, texture, and light randomization.
"""

from typing import Dict, List, Optional
import numpy as np
import mujoco


def randomize_lights(
    model: mujoco.MjModel,
    rng: np.random.Generator,
) -> None:
    """Randomize position and intensity of lights in the scene.
    
    Args:
        model: MuJoCo MjModel instance.
        rng: Numpy random generator instance.
    """
    for i in range(model.nlight):
        # Apply slight position jitter (+/- 0.2m)
        model.light_pos[i] += rng.uniform(-0.1, 0.1, size=3)
        # Apply slight diffuse color jitter
        model.light_diffuse[i] = np.clip(
            model.light_diffuse[i] + rng.uniform(-0.1, 0.1, size=3), 0.2, 1.0
        )


def randomize_material_colors(
    model: mujoco.MjModel,
    rng: np.random.Generator,
) -> None:
    """Randomize material diffuse colors for background elements.
    
    Args:
        model: MuJoCo MjModel instance.
        rng: Numpy random generator instance.
    """
    for i in range(model.nmat):
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        if mat_name and ("wall" in mat_name or "floor" in mat_name or "counter" in mat_name):
            # Apply slight hue shift
            jitter = rng.uniform(-0.05, 0.05, size=3)
            model.mat_rgba[i, :3] = np.clip(model.mat_rgba[i, :3] + jitter, 0.1, 0.9)
