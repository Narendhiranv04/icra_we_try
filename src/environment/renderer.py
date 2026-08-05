"""
Offscreen rendering wrapper for RGB images, instance masks, and target/culprit segmentation masks.
"""

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import mujoco

# Ensure EGL backend is set for offscreen rendering if not already set
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


class OffscreenRenderer:
    """Renderer class using mujoco.Renderer for generating RGB photos, instance segmentation maps,
    and binary culprit/target region masks."""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 640,
        height: int = 480,
        camera_name: str = "front_camera",
    ):
        self.model = model
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id == -1:
            # Fall back to free camera / camera 0 if specified camera name not in model
            self.camera_id = 0
            
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render_rgb(self, data: mujoco.MjData) -> np.ndarray:
        """Render RGB image from current camera perspective.
        
        Args:
            data: MjData instance.
            
        Returns:
            uint8 RGB numpy array of shape (height, width, 3).
        """
        self.renderer.update_scene(data, camera=self.camera_id)
        return self.renderer.render()

    def render_segmentation(self, data: mujoco.MjData) -> np.ndarray:
        """Render segmentation mask where each pixel value contains object/geom ID.
        
        Args:
            data: MjData instance.
            
        Returns:
            int32 segmentation numpy array of shape (height, width, 2) [geom_id, object_id].
        """
        self.renderer.update_scene(data, camera=self.camera_id)
        self.renderer.enable_segmentation_rendering()
        seg_mask = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return seg_mask

    def render_culprit_mask(
        self, data: mujoco.MjData, culprit_geom_names: list[str]
    ) -> np.ndarray:
        """Generate binary mask highlighting specified culprit object geoms.
        
        Args:
            data: MjData instance.
            culprit_geom_names: List of geom names representing culprit/blocker objects.
            
        Returns:
            uint8 binary mask of shape (height, width) with 255 for culprit pixels, 0 elsewhere.
        """
        seg_mask = self.render_segmentation(data)
        geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in culprit_geom_names
        ]
        geom_ids = [g_id for g_id in geom_ids if g_id != -1]
        
        culprit_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if not geom_ids:
            return culprit_mask
            
        for g_id in geom_ids:
            culprit_mask[seg_mask[:, :, 0] == g_id] = 255
            
        return culprit_mask

    def render_region_mask(
        self, data: mujoco.MjData, region_geom_names: list[str]
    ) -> np.ndarray:
        """Generate binary mask highlighting specified surface/region geoms (e.g. lid, target region).
        
        Args:
            data: MjData instance.
            region_geom_names: List of geom names for region surfaces.
            
        Returns:
            uint8 binary mask of shape (height, width) with 255 for region pixels, 0 elsewhere.
        """
        return self.render_culprit_mask(data, region_geom_names)

    def close(self):
        """Clean up renderer resources."""
        if hasattr(self, "renderer"):
            self.renderer.close()
