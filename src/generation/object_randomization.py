"""
Object selection, pose sampling, and distractor placement randomizations.
"""

from typing import List, Tuple
import numpy as np


def sample_distractor_objects(
    catalog: List[str],
    num_distractors: int,
    rng: np.random.Generator,
    exclude: List[str] = None,
) -> List[str]:
    """Sample a list of distractor object types from catalog.
    
    Args:
        catalog: List of available catalog object types.
        num_distractors: Number of distractors to sample.
        rng: Numpy random generator instance.
        exclude: List of object types to exclude from sampling.
        
    Returns:
        List of object type string names.
    """
    if exclude is None:
        exclude = []

    valid_choices = [item for item in catalog if item not in exclude]
    if not valid_choices:
        valid_choices = catalog

    sampled = list(rng.choice(valid_choices, size=num_distractors, replace=True))
    return sampled


def sample_valid_counter_position(
    rng: np.random.Generator,
    existing_positions: List[Tuple[float, float]],
    min_dist: float = 0.12,
    x_bounds: Tuple[float, float] = (-0.45, 0.45),
    y_bounds: Tuple[float, float] = (0.30, 0.60),
    max_tries: int = 100,
) -> Tuple[float, float]:
    """Sample a valid (x, y) placement position on countertop avoiding overlap with existing objects.
    
    Args:
        rng: Numpy random generator instance.
        existing_positions: List of existing (x, y) coordinates.
        min_dist: Minimum distance threshold between object centers.
        x_bounds: (x_min, x_max) countertop bounds.
        y_bounds: (y_min, y_max) countertop bounds.
        max_tries: Maximum sampling attempts before fallback.
        
    Returns:
        Tuple of (x, y) coordinates.
    """
    for _ in range(max_tries):
        x = rng.uniform(x_bounds[0], x_bounds[1])
        y = rng.uniform(y_bounds[0], y_bounds[1])
        
        overlap = False
        for ex_x, ex_y in existing_positions:
            if np.hypot(x - ex_x, y - ex_y) < min_dist:
                overlap = True
                break
                
        if not overlap:
            return (x, y)

    # Fallback to random uniform position if space crowded
    return (rng.uniform(x_bounds[0], x_bounds[1]), rng.uniform(y_bounds[0], y_bounds[1]))
