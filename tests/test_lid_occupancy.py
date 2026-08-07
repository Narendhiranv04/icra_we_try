"""
Unit test for testing lid occupancy privileged predicate.
"""

import pytest
from src.environment.scene_builder import SceneBuilder
from src.validation.occupancy_checks import check_lid_occupancy


def test_lid_occupancy_predicates():
    builder = SceneBuilder()

    # 1. Lid clear scene
    model_clear, data_clear = builder.create_environment(objects_to_spawn=None, settle_steps=100)
    is_occ_clear, culprits_clear, meas_clear = check_lid_occupancy(model_clear, data_clear)
    assert not is_occ_clear, "Lid clear scene was falsely marked as occupied!"
    assert len(culprits_clear) == 0

    # 2. Lid occupied scene
    blocker_objs = [{"name": "blocker1", "type": "sugar_box", "pos": [0.52, 0.18, 0.77]}]
    model_occ, data_occ = builder.create_environment(objects_to_spawn=blocker_objs, settle_steps=100)
    is_occ, culprits, meas_occ = check_lid_occupancy(model_occ, data_occ, blocker_names=["blocker1"])
    assert is_occ, "Lid occupied scene was not detected as occupied!"
    assert "blocker1" in culprits
