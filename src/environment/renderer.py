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
        camera_name: str = "robot0:ego_camera",
    ):
        self.model = model
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id == -1:
            raise KeyError(f"Missing required camera '{camera_name}' in MuJoCo model")
            
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def get_camera_metadata(self, data: mujoco.MjData) -> Dict[str, Union[str, int, float, list]]:
        """Extract complete camera calibration and extrinsic metadata.
        
        Returns:
            Dictionary with camera specs, extrinsics, fovy, and resolution.
        """
        cam_pos = data.cam_xpos[self.camera_id].copy().tolist()
        cam_mat = data.cam_xmat[self.camera_id].copy().reshape(3, 3)
        
        # Convert rotation matrix to quaternion [w, x, y, z]
        tr = np.trace(cam_mat)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (cam_mat[2, 1] - cam_mat[1, 2]) / S
            qy = (cam_mat[0, 2] - cam_mat[2, 0]) / S
            qz = (cam_mat[1, 0] - cam_mat[0, 1]) / S
        elif (cam_mat[0, 0] > cam_mat[1, 1]) and (cam_mat[0, 0] > cam_mat[2, 2]):
            S = np.sqrt(1.0 + cam_mat[0, 0] - cam_mat[1, 1] - cam_mat[2, 2]) * 2
            qw = (cam_mat[2, 1] - cam_mat[1, 2]) / S
            qx = 0.25 * S
            qy = (cam_mat[0, 1] + cam_mat[1, 0]) / S
            qz = (cam_mat[0, 2] + cam_mat[2, 0]) / S
        elif cam_mat[1, 1] > cam_mat[2, 2]:
            S = np.sqrt(1.0 + cam_mat[1, 1] - cam_mat[0, 0] - cam_mat[2, 2]) * 2
            qw = (cam_mat[0, 2] - cam_mat[2, 0]) / S
            qx = (cam_mat[0, 1] + cam_mat[1, 0]) / S
            qy = 0.25 * S
            qz = (cam_mat[1, 2] + cam_mat[2, 1]) / S
        else:
            S = np.sqrt(1.0 + cam_mat[2, 2] - cam_mat[0, 0] - cam_mat[1, 1]) * 2
            qw = (cam_mat[1, 0] - cam_mat[0, 1]) / S
            qx = (cam_mat[0, 2] + cam_mat[2, 0]) / S
            qy = (cam_mat[1, 2] + cam_mat[2, 1]) / S
            qz = 0.25 * S

        cam_body_id = self.model.cam_bodyid[self.camera_id]
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, cam_body_id) or "world"
        local_pos = self.model.cam_pos0[self.camera_id].copy().tolist() if hasattr(self.model, "cam_pos0") else [0.0, 0.0, 0.0]

        return {
            "camera_name": self.camera_name,
            "camera_id": int(self.camera_id),
            "camera_parent_body": body_name,
            "camera_local_position": local_pos,
            "camera_world_extrinsic": {
                "position": [float(x) for x in cam_pos],
                "quaternion_wxyz": [float(qw), float(qx), float(qy), float(qz)],
            },
            "field_of_view": float(self.model.cam_fovy[self.camera_id]),
            "resolution": [int(self.width), int(self.height)],
        }

    def validate_view_quality(
        self,
        data: mujoco.MjData,
        target_geom_names: list[str],
        candidate_geom_names: list[str],
        min_target_pixels: int = 50,
        min_candidate_pixels: int = 50,
        max_torso_fraction: float = 0.25,
    ) -> Dict[str, Union[bool, int, float, list]]:
        """Validate that the canonical egocentric camera frame meets quality standards.
        
        Returns:
            Dictionary with visibility metrics and validation status.
        """
        seg_mask = self.render_segmentation(data)
        geom_ids_in_view = seg_mask[:, :, 0]

        # Target / region pixels
        target_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in target_geom_names
        ]
        target_ids = [g for g in target_ids if g != -1]
        target_px = int(np.isin(geom_ids_in_view, target_ids).sum()) if target_ids else 0

        # Candidate object pixels
        cand_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in candidate_geom_names
        ]
        cand_ids = [g for g in cand_ids if g != -1]
        cand_px = int(np.isin(geom_ids_in_view, cand_ids).sum()) if cand_ids else 0

        # Torso body pixels
        torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:torso_lift_link")
        torso_geom_ids = []
        if torso_body_id != -1:
            torso_geom_ids = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == torso_body_id]
        torso_px = int(np.isin(geom_ids_in_view, torso_geom_ids).sum()) if torso_geom_ids else 0
        total_px = self.width * self.height
        torso_frac = float(torso_px / total_px)

        is_valid = bool(
            target_px >= min_target_pixels
            and (cand_px >= min_candidate_pixels if candidate_geom_names else True)
            and torso_frac <= max_torso_fraction
        )

        return {
            "is_valid": is_valid,
            "target_pixels": target_px,
            "candidate_pixels": cand_px,
            "torso_fraction": torso_frac,
            "target_visible": target_px >= min_target_pixels,
            "candidate_visible": cand_px >= min_candidate_pixels if candidate_geom_names else True,
            "torso_acceptable": torso_frac <= max_torso_fraction,
        }

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
