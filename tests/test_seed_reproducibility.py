"""
Unit test to verify seed reproducibility of generated scenes.
"""

import pytest
import numpy as np
from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer


def test_seed_reproducibility():
    """Verify that building environment with identical objects reproduces identical RGB rendering."""
    objects = [{"name": "test_can", "type": "coffee_can", "pos": [0.0, 0.45, 0.96]}]
    builder = SceneBuilder()

    model1, data1 = builder.create_environment(objects, settle_steps=50)
    renderer1 = OffscreenRenderer(model1, width=320, height=240)
    rgb1 = renderer1.render_rgb(data1)
    renderer1.close()

    model2, data2 = builder.create_environment(objects, settle_steps=50)
    renderer2 = OffscreenRenderer(model2, width=320, height=240)
    rgb2 = renderer2.render_rgb(data2)
    renderer2.close()

    np.testing.assert_allclose(rgb1, rgb2, atol=2, err_msg="RGB renders from identical setups do not match within tolerance!")
