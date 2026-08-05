"""
Unit test for testing target region occupancy privileged predicate.
"""

import pytest
from src.environment.scene_builder import SceneBuilder
from src.validation.occupancy_checks import check_target_occupancy


def test_target_occupancy_predicates():
    builder = SceneBuilder()

    # 1. Target clear scene
    model_clear, data_clear = builder.create_environment(objects_to_spawn=None, settle_steps=50)
    is_occ_clear, culprits_clear = check_target_occupancy(model_clear, data_clear)
    assert not is_occ_clear, "Target empty scene was falsely marked as occupied!"

    # 2. Target occupied scene
    occupant_objs = [{"name": "occupant", "type": "sugar_box", "pos": [-0.10, -0.20, 0.65]}]
    model_occ, data_occ = builder.create_environment(objects_to_spawn=occupant_objs, settle_steps=50)
    is_occ, culprits = check_target_occupancy(model_occ, data_occ, candidate_objects=["occupant"])
    assert is_occ, "Target occupied scene was not detected as occupied!"
    assert "occupant" in culprits
