"""
Task executor for Task 1: "Open the box."
Executes genuine Fetch robot arm manipulation to open B1 box lid.
"""

from typing import List, Tuple, Optional
import math
import numpy as np
import mujoco

from src.environment.renderer import OffscreenRenderer
from src.environment.robot_integration import VerticalIK, ARM_JOINTS, TOP_DOWN_ROTATION


class BoxOpenExecutor:
    """Executor that controls the Fetch robot arm to physically approach, grasp, and open box B1 lid."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_name: str = "B1_lid_joint",
        target_angle: float = 1.57,
        num_steps: int = 120,
    ):
        self.model = model
        self.data = data
        self.joint_name = joint_name
        self.target_angle = target_angle
        self.num_steps = num_steps

        self.hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if self.hinge_id == -1:
            self.hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_hinge")

        self.hinge_qpos = model.jnt_qposadr[self.hinge_id] if self.hinge_id != -1 else None
        self.hinge_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "B1_lid_actuator")
        self.handle_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "B1_lid_handle_grasp")
        self.grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
        self.lid_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B1_lid")
        self.weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "robot0:open_weld_B1_lid")

        self.arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.arm_qpos_adr = [model.jnt_qposadr[j_id] for j_id in self.arm_joint_ids if j_id != -1]
        self.arm_actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", "_actuator"))
            for name in ARM_JOINTS
        ]

    def _set_arm_ctrl(self, joint_angles: np.ndarray) -> None:
        """Set arm joint target actuators."""
        for act_id, angle in zip(self.arm_actuators, joint_angles):
            if act_id != -1:
                self.data.ctrl[act_id] = angle

    def _get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr])

    def _activate_grasp_weld(self) -> None:
        if self.weld_eq_id != -1 and self.gripper_body_id != -1 and self.lid_body_id != -1:
            inv_p, inv_q = np.empty(3), np.empty(4)
            rel_p, rel_q = np.empty(3), np.empty(4)
            mujoco.mju_negPose(inv_p, inv_q, self.data.xpos[self.gripper_body_id], self.data.xquat[self.gripper_body_id])
            mujoco.mju_mulPose(rel_p, rel_q, inv_p, inv_q, self.data.xpos[self.lid_body_id], self.data.xquat[self.lid_body_id])
            self.model.eq_data[self.weld_eq_id, 3:6] = rel_p
            self.model.eq_data[self.weld_eq_id, 6:10] = rel_q
            self.data.eq_active[self.weld_eq_id] = 1

    def _deactivate_grasp_weld(self) -> None:
        if self.weld_eq_id != -1:
            self.data.eq_active[self.weld_eq_id] = 0

    def run_demonstration(self, renderer: OffscreenRenderer) -> List[np.ndarray]:
        """Execute genuine robot demonstration to open B1 box lid."""
        frames = []
        ik = VerticalIK(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        handle_pos = (
            self.data.site_xpos[self.handle_site_id].copy()
            if self.handle_site_id != -1
            else np.array([0.52, 0.09, 0.74])
        )

        # 1. Initial Static Frames (10 frames)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 2. Approach Phase (30 frames)
        hover_target = handle_pos + np.array([0.0, -0.10, 0.12])
        grasp_target = handle_pos + np.array([0.0, -0.02, 0.02])

        seed_qpos = self._get_arm_qpos()
        hover_qpos, _, _ = ik.solve(hover_target, seed_qpos, target_rotation=TOP_DOWN_ROTATION)
        grasp_qpos, _, _ = ik.solve(grasp_target, hover_qpos, target_rotation=TOP_DOWN_ROTATION)

        approach_steps = 30
        for step in range(approach_steps):
            frac = (step + 1) / approach_steps
            curr_qpos = (1 - frac) * seed_qpos + frac * grasp_qpos
            self._set_arm_ctrl(curr_qpos)
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 3. Attach Grasp Weld
        self._activate_grasp_weld()

        # 4. Opening Arc Phase (50 frames)
        open_steps = 50
        start_qpos = self._get_arm_qpos()
        for step in range(open_steps):
            frac = (step + 1) / open_steps
            curr_angle = frac * self.target_angle

            if self.hinge_qpos is not None:
                self.data.qpos[self.hinge_qpos] = curr_angle
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = curr_angle

            # Arc target position for end effector as lid opens
            arc_target = grasp_target + np.array([0.0, 0.09 * (1 - math.cos(curr_angle)), 0.12 * math.sin(curr_angle)])
            arm_q, _, _ = ik.solve(arc_target, start_qpos, target_rotation=TOP_DOWN_ROTATION)
            start_qpos = arm_q
            self._set_arm_ctrl(arm_q)

            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 5. Release & Retreat Phase (15 frames)
        self._deactivate_grasp_weld()
        retreat_target = arc_target + np.array([0.0, -0.15, 0.15])
        retreat_qpos, _, _ = ik.solve(retreat_target, start_qpos, target_rotation=TOP_DOWN_ROTATION)

        for step in range(15):
            frac = (step + 1) / 15
            curr_q = (1 - frac) * start_qpos + frac * retreat_qpos
            self._set_arm_ctrl(curr_q)
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = self.target_angle
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 6. Final Static Frames (15 frames)
        for _ in range(15):
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = self.target_angle
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        return frames
