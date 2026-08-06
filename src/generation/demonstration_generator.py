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

    def generate_task_1_demo(
        self,
        demo_id: str = "demo_task1_001",
        robot_base_pose: str = "right_side",
        background_id: str = "bg_neutral_wood",
        seed: int = 42,
    ) -> str:
        """Generate full structured demonstration for Task 1 (Open Box)."""
        rng = np.random.default_rng(seed)
        model, data = self.scene_builder.create_environment(
            objects_to_spawn=None,
            include_robot=True,
            robot_base_pose=robot_base_pose,
        )

        bg_spec = sample_background_spec(background_id, rng, n_lights=model.nlight)
        apply_background_spec(model, bg_spec)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height)

        executor = BoxOpenExecutor(model, data)
        frames = executor.run_demonstration(renderer)
        renderer.close()

        is_valid, metrics, issues = DemonstrationValidator.validate_open_box(executor.state_log)
        val_result = {
            "is_valid": is_valid,
            "metrics": metrics,
            "issues": issues,
            "demo_id": demo_id,
            "task_family": "open_box",
        }

        scene_spec = {
            "demo_id": demo_id,
            "task_family": "open_box",
            "seed": seed,
            "robot_base_pose": robot_base_pose,
            "background_id": background_id,
            "background_spec": bg_spec.to_dict(),
            "instruction": "Open the box.",
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
    ) -> str:
        """Generate full structured demonstration for Task 2 (Place Object) using scene-local frame geometry."""
        rng = np.random.default_rng(seed)

        # 1. Resolve start_pos and target_pos from scene geometry local frame
        ref_model, ref_data = self.scene_builder.create_environment(settle_steps=0)
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
        )

        bg_spec = sample_background_spec(background_id, rng, n_lights=model.nlight)
        apply_background_spec(model, bg_spec)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height)

        executor = PlaceObjectExecutor(model, data, object_name=obj_name, target_pos=tuple(target_pos))
        frames = executor.run_demonstration(renderer, start_pos=tuple(start_pos))
        renderer.close()

        is_valid, metrics, issues = DemonstrationValidator.validate_place_object(executor.state_log)
        val_result = {
            "is_valid": is_valid,
            "metrics": metrics,
            "issues": issues,
            "demo_id": demo_id,
            "task_family": "place_object",
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
