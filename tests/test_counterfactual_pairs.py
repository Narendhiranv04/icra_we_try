"""
Unit test for testing counterfactual pair generator.
"""

import pytest
import tempfile
from pathlib import Path
from src.generation.counterfactual_generator import CounterfactualPairGenerator


def test_counterfactual_pair_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        
        # Test Task 1 Pair
        meta1 = gen.generate_task1_pair("test_pair_001", blocker_type="coffee_can")
        assert meta1["stop"]["is_occupied"] is True
        assert meta1["proceed"]["is_occupied"] is False
        assert Path(meta1["stop"]["rgb_path"]).exists()
        assert Path(meta1["proceed"]["rgb_path"]).exists()

        # Test Task 2 Pair
        meta2 = gen.generate_task2_pair("test_pair_002", target_occupant_type="sugar_box")
        assert meta2["stop"]["is_occupied"] is True
        assert meta2["proceed"]["is_occupied"] is False
        assert Path(meta2["stop"]["rgb_path"]).exists()
        assert Path(meta2["proceed"]["rgb_path"]).exists()
