"""
Demonstration video generator for rendering MP4 demonstration clips and full structured demonstration artifacts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.tasks.open_box import BoxOpenExecutor
from src.tasks.place_object import PlaceObjectExecutor
from src.validation.demonstration_validator import DemonstrationValidator
from src.generation.demonstration_writer import DemonstrationWriter


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
        model, data = self.scene_builder.create_environment(
            objects_to_spawn=None,
            include_robot=True,
            robot_base_pose=robot_base_pose,
        )
        renderer = OffscreenRenderer(model, width=self.width, height=self.height)

        executor = BoxOpenExecutor(model, data)
        frames = executor.run_demonstration(renderer)
        renderer.close()

        # Validate demonstration state log
        is_valid, issues = DemonstrationValidator.validate_open_box(executor.state_log)
        val_result = {
            "is_valid": is_valid,
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
        start_pos: tuple[float, float, float] = (-0.30, -0.20, 0.65),
        target_pos: tuple[float, float, float] = (-0.10, -0.20, 0.65),
        robot_base_pose: str = "home",
        background_id: str = "bg_neutral_wood",
        seed: int = 42,
    ) -> str:
        """Generate full structured demonstration for Task 2 (Place Object)."""
        objects = [{"name": obj_name, "type": obj_name, "pos": list(start_pos)}]
        model, data = self.scene_builder.create_environment(
            objects_to_spawn=objects,
            include_robot=True,
            robot_base_pose=robot_base_pose,
            weld_target_body=obj_name,
        )
        renderer = OffscreenRenderer(model, width=self.width, height=self.height)

        executor = PlaceObjectExecutor(model, data, object_name=obj_name, target_pos=target_pos)
        frames = executor.run_demonstration(renderer, start_pos=start_pos)
        renderer.close()

        is_valid, issues = DemonstrationValidator.validate_place_object(executor.state_log)
        val_result = {
            "is_valid": is_valid,
            "issues": issues,
            "demo_id": demo_id,
            "task_family": "place_object",
        }

        scene_spec = {
            "demo_id": demo_id,
            "task_family": "place_object",
            "seed": seed,
            "object_name": obj_name,
            "start_pos": list(start_pos),
            "target_pos": list(target_pos),
            "robot_base_pose": robot_base_pose,
            "background_id": background_id,
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
