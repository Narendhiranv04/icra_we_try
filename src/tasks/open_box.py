"""
Task executor for Task 1: "Open the box."
"""

from typing import List, Tuple
import numpy as np
import mujoco

from src.environment.renderer import OffscreenRenderer


class BoxOpenExecutor:
    """Executor that steps joint trajectories to open box B1 lid from 0 to target open angle."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_name: str = "B1_lid_hinge",
        target_angle: float = 1.57,
        num_steps: int = 60,
    ):
        self.model = model
        self.data = data
        self.joint_name = joint_name
        self.target_angle = target_angle
        self.num_steps = num_steps
        
        self.joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if self.joint_id != -1:
            self.qpos_idx = model.jnt_qposadr[self.joint_id]
            self.dof_idx = model.jnt_dofadr[self.joint_id]
        else:
            self.qpos_idx = None
            self.dof_idx = None

    def execute_step(self, step_idx: int) -> None:
        """Step joint angle towards target open angle.
        
        Args:
            step_idx: Current animation frame / step index.
        """
        if self.qpos_idx is not None:
            fraction = min(1.0, (step_idx + 1) / self.num_steps)
            current_angle = fraction * self.target_angle
            self.data.qpos[self.qpos_idx] = current_angle
            self.data.qvel[self.dof_idx] = (self.target_angle / self.num_steps) / self.model.opt.timestep
        mujoco.mj_forward(self.model, self.data)

    def run_demonstration(
        self,
        renderer: OffscreenRenderer,
    ) -> List[np.ndarray]:
        """Execute complete box open sequence and return list of RGB video frames.
        
        Args:
            renderer: OffscreenRenderer instance.
            
        Returns:
            List of uint8 RGB numpy arrays.
        """
        frames = []
        for step in range(self.num_steps):
            self.execute_step(step)
            # Step internal simulation steps for smooth movement
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            rgb = renderer.render_rgb(self.data)
            frames.append(rgb)
        return frames
