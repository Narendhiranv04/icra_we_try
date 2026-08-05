"""
Demonstration video generator for rendering MP4 demonstration clips of successful task executions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.tasks.open_box import BoxOpenExecutor
from src.tasks.place_object import PlaceObjectExecutor


class DemonstrationGenerator:
    """Generator for producing MP4 demonstration videos of successful task executions."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/demos",
        fps: int = 30,
        resolution: tuple[int, int] = (640, 480),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width, self.height = resolution
        self.scene_builder = SceneBuilder()

    def save_mp4(self, frames: List[np.ndarray], output_path: Union[str, Path]) -> str:
        """Save a list of RGB numpy frames as an MP4 video file.
        
        Args:
            frames: List of (height, width, 3) uint8 RGB arrays.
            output_path: Path to output MP4 file.
            
        Returns:
            Path string to saved MP4 file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # OpenCV uses BGR ordering
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (self.width, self.height)
        )

        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()
        return str(output_path)

    def generate_task_1_demo(self, demo_id: str = "demo_task1_001") -> str:
        """Generate demonstration clip for Task 1 (Open Box).
        
        Returns:
            Saved MP4 file path.
        """
        # Task 1 PROCEED condition: lid is clear
        model, data = self.scene_builder.create_environment(
            objects_to_spawn=None,
            include_robot=True,
            robot_base_pose="right_side",
        )
        renderer = OffscreenRenderer(model, width=self.width, height=self.height)
        
        executor = BoxOpenExecutor(model, data)
        frames = executor.run_demonstration(renderer)
        renderer.close()

        save_path = self.output_dir / f"{demo_id}.mp4"
        return self.save_mp4(frames, save_path)

    def generate_task_2_demo(
        self,
        demo_id: str = "demo_task2_001",
        obj_name: str = "coffee_can",
    ) -> str:
        """Generate demonstration clip for Task 2 (Place Object).
        
        Returns:
            Saved MP4 file path.
        """
        start_pos = (-0.25, -0.30, 0.65)
        target_pos = (-0.10, -0.20, 0.65)
        objects = [{"name": obj_name, "type": obj_name, "pos": list(start_pos)}]
        model, data = self.scene_builder.create_environment(
            objects_to_spawn=objects,
            include_robot=True,
            robot_base_pose="home",
            weld_target_body=obj_name,
        )
        renderer = OffscreenRenderer(model, width=self.width, height=self.height)

        executor = PlaceObjectExecutor(model, data, object_name=obj_name, target_pos=target_pos)
        frames = executor.run_demonstration(renderer, start_pos=start_pos)
        renderer.close()

        save_path = self.output_dir / f"{demo_id}.mp4"
        return self.save_mp4(frames, save_path)
