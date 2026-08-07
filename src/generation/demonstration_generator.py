"""
Demonstration video generator for rendering MP4 demonstration clips and full structured demonstration artifacts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
import hashlib
import mujoco

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.tasks.open_box import BoxOpenExecutor
from src.tasks.place_object import PlaceObjectExecutor
from src.validation.demonstration_validator import DemonstrationValidator
from src.generation.demonstration_writer import DemonstrationWriter
from src.generation.background_randomization import sample_background_spec, apply_background_spec
from src.environment.scene_utils import get_target_frame


class DemonstrationGenerator:
    """Generator for producing structured demonstration data directories of successful task executions."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/demos",
        fps: int = 15,
        resolution: tuple[int, int] = (640, 480),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width, self.height = resolution
        self.scene_builder = SceneBuilder()
        self.writer = DemonstrationWriter(base_dir=self.output_dir)

    def _get_body_geom_names(self, model: mujoco.MjModel, body_name: str) -> List[str]:
        """Retrieve visual geom names attached to a given body name in the MuJoCo model."""
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return [f"{body_name}_visual", f"{body_name}_geom"]
        geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body_id and model.geom_group[g] != 3:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                if name:
                    geoms.append(name)
        return geoms if geoms else [f"{body_name}_visual"]

    def generate_task_1_demo(
        self,
        demo_id: str = "demo_task1_001",
        robot_base_pose: str = "home",
        background_id: str = "bg_neutral_wood",
        seed: int = 42,
    ) -> str:
        """Generate full structured demonstration for Task 1 (Open Box)."""
        from src.environment.observation_rig import TASK_1_RIG, apply_observation_rig
        if robot_base_pose != TASK_1_RIG.robot_base_pose:
            raise ValueError(f"Uncalibrated base pose override '{robot_base_pose}' rejected for Task 1; expected '{TASK_1_RIG.robot_base_pose}'")

        rng = np.random.default_rng(seed)
        model, data = self.scene_builder.create_environment(
            objects_to_spawn=None,
            include_robot=True,
            robot_base_pose=robot_base_pose,
        )

        cam_metadata = apply_observation_rig(model, data, TASK_1_RIG)

        bg_spec = sample_background_spec(background_id, rng, n_lights=model.nlight)
        apply_background_spec(model, bg_spec)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height, camera_name=TASK_1_RIG.camera_name)

        # Initial visibility check before motion
        init_view = renderer.validate_instance_visibility(
            data,
            target_geom_names=["B1_lid_panel", "B1_lid_geom"],
            required_instances={},
        )

        executor = BoxOpenExecutor(model, data)
        frames = executor.run_demonstration(renderer)

        # Final visibility check after open
        final_view = renderer.validate_instance_visibility(
            data,
            target_geom_names=["B1_lid_panel", "B1_lid_geom"],
            required_instances={},
        )
        renderer.close()

        temporal_vq = {
            "is_valid": bool(init_view["is_valid"] and final_view["is_valid"]),
            "phases": {
                "initial": init_view,
                "final_open": final_view,
            },
        }

        is_valid, metrics, issues = DemonstrationValidator.validate_open_box(executor.state_log)
        val_result = {
            "is_valid": is_valid and temporal_vq["is_valid"],
            "metrics": metrics,
            "issues": issues,
            "demo_id": demo_id,
            "task_family": "open_box",
            "temporal_view_quality": temporal_vq,
        }

        scene_spec = {
            "demo_id": demo_id,
            "task_family": "open_box",
            "seed": seed,
            "robot_base_pose": robot_base_pose,
            "background_id": background_id,
            "background_spec": bg_spec.to_dict(),
            "instruction": "Open the box.",
            "camera_name": TASK_1_RIG.camera_name,
            "camera_configuration": TASK_1_RIG.camera_name,
            "camera_world_extrinsic": cam_metadata["camera_world_extrinsic"],
            "measured_camera_metadata": cam_metadata,
            "temporal_view_quality": temporal_vq,
        }

        demo_dir = self.writer.save_demonstration(
            task_family="open_box",
            demo_id=demo_id,
            frames=frames,
            state_log=executor.state_log,
            scene_spec=scene_spec,
            validation_result=val_result,
            fps=self.fps,
        )

        return str(demo_dir / "rgb.mp4")

    def generate_task_2_demo(
        self,
        demo_id: str = "demo_task2_001",
        obj_name: str = "coffee_can",
        start_bin: str = "pick_left",
        target_bin: str = "centre",
        robot_base_pose: str = "home",
        background_id: str = "bg_neutral_wood",
        seed: int = 42,
        target_region_pos: list = None,
        target_region_quat: list = None,
    ) -> str:
        """Generate full structured demonstration for Task 2 (Place Object) using scene-local frame geometry."""
        from src.environment.observation_rig import TASK_2_RIG, apply_observation_rig
        if robot_base_pose != TASK_2_RIG.robot_base_pose:
            raise ValueError(f"Uncalibrated base pose override '{robot_base_pose}' rejected for Task 2; expected '{TASK_2_RIG.robot_base_pose}'")

        rng = np.random.default_rng(seed)

        # 1. Resolve start_pos and target_pos from scene geometry local frame
        ref_model, ref_data = self.scene_builder.create_environment(
            settle_steps=0,
            target_region_pos=target_region_pos,
            target_region_quat=target_region_quat,
        )
        t_center, t_rot, t_ext = get_target_frame(ref_model, ref_data)

        pick_offsets = {
            "pick_left": np.array([0.0, -0.18, 0.07]),
            "pick_right": np.array([0.0, -0.25, 0.07]),
            "pick_far_left": np.array([0.10, -0.18, 0.07]),
            "pick_front": np.array([0.0, -0.18, 0.07]),
            "pick_rear": np.array([0.0, -0.25, 0.07]),
        }
        start_local = pick_offsets.get(start_bin, np.array([0.0, -0.18, 0.07]))
        start_pos = (t_center + t_rot @ start_local).tolist()

        target_offsets = {
            "centre": np.array([0.0, 0.0, 0.07]),
            "left": np.array([-0.03, 0.0, 0.07]),
            "right": np.array([0.03, 0.0, 0.07]),
        }
        target_local = target_offsets.get(target_bin, np.array([0.0, 0.0, 0.07]))
        target_pos = (t_center + t_rot @ target_local).tolist()

        objects = [{"name": obj_name, "type": obj_name, "pos": start_pos}]
        model, data = self.scene_builder.create_environment(
            objects_to_spawn=objects,
            include_robot=True,
            robot_base_pose=robot_base_pose,
            weld_target_body=obj_name,
            target_region_pos=target_region_pos,
            target_region_quat=target_region_quat,
        )

        cam_metadata = apply_observation_rig(model, data, TASK_2_RIG)

        bg_spec = sample_background_spec(background_id, rng, n_lights=model.nlight)
        apply_background_spec(model, bg_spec)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height, camera_name=TASK_2_RIG.camera_name)

        # Initial visibility check before motion
        obj_geoms = self._get_body_geom_names(model, obj_name)
        init_view = renderer.validate_instance_visibility(
            data,
            target_geom_names=["target_region_geom"],
            required_instances={"object1": obj_geoms},
        )

        executor = PlaceObjectExecutor(model, data, object_name=obj_name, target_pos=tuple(target_pos))
        frames = executor.run_demonstration(renderer, start_pos=tuple(start_pos))

        # Final visibility check after placement
        final_view = renderer.validate_instance_visibility(
            data,
            target_geom_names=["target_region_geom"],
            required_instances={"object1": obj_geoms},
        )
        renderer.close()

        temporal_vq = {
            "is_valid": bool(init_view["is_valid"] and final_view["is_valid"]),
            "phases": {
                "initial": init_view,
                "final_placed": final_view,
            },
        }

        is_valid, metrics, issues = DemonstrationValidator.validate_place_object(executor.state_log)
        val_result = {
            "is_valid": is_valid and temporal_vq["is_valid"],
            "metrics": metrics,
            "issues": issues,
            "demo_id": demo_id,
            "task_family": "place_object",
            "temporal_view_quality": temporal_vq,
        }

        scene_spec = {
            "demo_id": demo_id,
            "task_family": "place_object",
            "seed": seed,
            "object_name": obj_name,
            "start_bin": start_bin,
            "target_bin": target_bin,
            "start_pos": start_pos,
            "target_pos": target_pos,
            "robot_base_pose": robot_base_pose,
            "background_id": background_id,
            "background_spec": bg_spec.to_dict(),
            "instruction": "Place object1 in the target region.",
            "scene_transforms": {
                "target_region_pos": target_region_pos,
                "target_region_quat": target_region_quat,
            },
            "camera_name": TASK_2_RIG.camera_name,
            "camera_configuration": TASK_2_RIG.camera_name,
            "camera_world_extrinsic": cam_metadata["camera_world_extrinsic"],
            "measured_camera_metadata": cam_metadata,
            "temporal_view_quality": temporal_vq,
        }

        demo_dir = self.writer.save_demonstration(
            task_family="place_object",
            demo_id=demo_id,
            frames=frames,
            state_log=executor.state_log,
            scene_spec=scene_spec,
            validation_result=val_result,
            fps=self.fps,
        )

        return str(demo_dir / "rgb.mp4")
