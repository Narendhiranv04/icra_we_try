"""
Deterministic seed management and scene configuration utilities.
"""

import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds across random and numpy modules for reproducibility.
    
    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


class SceneConfigManager:
    """Manager for generating deterministic scene randomizations and tracking counterfactual metadata."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_surface_offset(
        self, center: tuple[float, float, float], radius: float = 0.08
    ) -> list[float]:
        """Sample a randomized (x, y, z) position offset around a surface center coordinate.
        
        Args:
            center: (x, y, z) center coordinate.
            radius: Randomization radius in XY plane.
            
        Returns:
            [x, y, z] sampled coordinate.
        """
        r = self.rng.uniform(0.0, radius)
        theta = self.rng.uniform(0.0, 2 * np.pi)
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        return [center[0] + dx, center[1] + dy, center[2]]
