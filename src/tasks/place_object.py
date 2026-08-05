"""
Task executor for Task 2: "Place object1 in the target region."
"""

from typing import List, Tuple
import numpy as np
import mujoco

from src.environment.renderer import OffscreenRenderer


class PlaceObjectExecutor:
    """Executor that simulates pick and place movement of an object into the target region."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_name: str = "coffee_can",
        target_pos: Tuple[float, float, float] = (0.25, 0.45, 0.82),
        num_steps: int = 60,
    ):
        self.model = model
        self.data = data
        self.object_name = object_name
        self.target_pos = np.array(target_pos)
        self.num_steps = num_steps
        
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if self.body_id != -1:
            self.joint_id = model.body_jntadr[self.body_id]
            if self.joint_id != -1 and model.jnt_type[self.joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self.qpos_idx = model.jnt_qposadr[self.joint_id]
            else:
                self.qpos_idx = None
        else:
            self.qpos_idx = None

    def run_demonstration(
        self,
        renderer: OffscreenRenderer,
        start_pos: Tuple[float, float, float] = (-0.2, 0.45, 0.85),
    ) -> List[np.ndarray]:
        """Execute smooth pick-and-place trajectory and capture RGB video frames.
        
        Args:
            renderer: OffscreenRenderer instance.
            start_pos: Initial 3D position of object before move.
            
        Returns:
            List of uint8 RGB numpy arrays.
        """
        frames = []
        start_p = np.array(start_pos)
        end_p = self.target_pos

        for step in range(self.num_steps):
            t = (step + 1) / self.num_steps
            # Parabolic arc height
            arc_h = 0.2 * np.sin(np.pi * t)
            curr_pos = (1 - t) * start_p + t * end_p + np.array([0.0, 0.0, arc_h])

            if self.qpos_idx is not None:
                self.data.qpos[self.qpos_idx : self.qpos_idx + 3] = curr_pos
                self.data.qpos[self.qpos_idx + 3 : self.qpos_idx + 7] = [1.0, 0.0, 0.0, 0.0]

            mujoco.mj_forward(self.model, self.data)
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)

            rgb = renderer.render_rgb(self.data)
            frames.append(rgb)

        return frames
