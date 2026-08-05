"""
Task executor for Task 2: "Place object1 in the target region."
Executes genuine Fetch robot arm pick-and-place manipulation.
"""

from typing import List, Tuple, Optional
import numpy as np
import mujoco

from src.environment.renderer import OffscreenRenderer
from src.environment.robot_integration import VerticalIK, ARM_JOINTS, TOP_DOWN_ROTATION


class PlaceObjectExecutor:
    """Executor that controls the Fetch robot arm to physically pick object1 and place it in the target region."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_name: str = "coffee_can",
        target_pos: Tuple[float, float, float] = (-0.10, -0.20, 0.65),
        num_steps: int = 120,
    ):
        self.model = model
        self.data = data
        self.object_name = object_name
        self.target_pos = np.array(target_pos)
        self.num_steps = num_steps

        self.obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
        self.weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "robot0:grasp_weld_target")

        self.arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.arm_qpos_adr = [model.jnt_qposadr[j_id] for j_id in self.arm_joint_ids if j_id != -1]
        self.arm_actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", "_actuator"))
            for name in ARM_JOINTS
        ]

    def _set_arm_ctrl(self, joint_angles: np.ndarray) -> None:
        for act_id, angle in zip(self.arm_actuators, joint_angles):
            if act_id != -1:
                self.data.ctrl[act_id] = angle

    def _get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr])

    def _activate_grasp_weld(self) -> None:
        if self.weld_eq_id != -1 and self.gripper_body_id != -1 and self.obj_body_id != -1:
            inv_p, inv_q = np.empty(3), np.empty(4)
            rel_p, rel_q = np.empty(3), np.empty(4)
            mujoco.mju_negPose(inv_p, inv_q, self.data.xpos[self.gripper_body_id], self.data.xquat[self.gripper_body_id])
            mujoco.mju_mulPose(rel_p, rel_q, inv_p, inv_q, self.data.xpos[self.obj_body_id], self.data.xquat[self.obj_body_id])
            self.model.eq_data[self.weld_eq_id, 3:6] = rel_p
            self.model.eq_data[self.weld_eq_id, 6:10] = rel_q
            self.data.eq_active[self.weld_eq_id] = 1

    def _deactivate_grasp_weld(self) -> None:
        if self.weld_eq_id != -1:
            self.data.eq_active[self.weld_eq_id] = 0

    def run_demonstration(
        self,
        renderer: OffscreenRenderer,
        start_pos: Tuple[float, float, float] = (-0.35, -0.20, 0.65),
    ) -> List[np.ndarray]:
        """Execute genuine robot pick and place demonstration and return RGB video frames."""
        frames = []
        ik = VerticalIK(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        obj_start = np.array(start_pos)
        target_place = self.target_pos

        # 1. Initial Static Frames (10 frames)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 2. Approach Phase (30 frames)
        hover_start = obj_start + np.array([0.0, 0.0, 0.15])
        pick_pos = obj_start + np.array([0.0, 0.0, 0.02])

        seed_qpos = self._get_arm_qpos()
        hover_qpos, _, _ = ik.solve(hover_start, seed_qpos, target_rotation=TOP_DOWN_ROTATION)
        pick_qpos, _, _ = ik.solve(pick_pos, hover_qpos, target_rotation=TOP_DOWN_ROTATION)

        approach_steps = 30
        for step in range(approach_steps):
            frac = (step + 1) / approach_steps
            curr_qpos = (1 - frac) * seed_qpos + frac * pick_qpos
            self._set_arm_ctrl(curr_qpos)
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 3. Attach Grasp Weld
        self._activate_grasp_weld()

        # 4. Transport Phase (50 frames): Lift -> Move -> Place
        transport_steps = 50
        start_q = self._get_arm_qpos()
        for step in range(transport_steps):
            t = (step + 1) / transport_steps
            # Parabolic transport trajectory for end effector
            arc_h = 0.15 * np.sin(np.pi * t)
            curr_ee_pos = (1 - t) * pick_pos + t * target_place + np.array([0.0, 0.0, arc_h])
            arm_q, _, _ = ik.solve(curr_ee_pos, start_q, target_rotation=TOP_DOWN_ROTATION)
            start_q = arm_q
            self._set_arm_ctrl(arm_q)

            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 5. Release & Retreat Phase (15 frames)
        self._deactivate_grasp_weld()
        retreat_pos = target_place + np.array([0.0, 0.0, 0.20])
        retreat_qpos, _, _ = ik.solve(retreat_pos, start_q, target_rotation=TOP_DOWN_ROTATION)

        for step in range(15):
            frac = (step + 1) / 15
            curr_q = (1 - frac) * start_q + frac * retreat_qpos
            self._set_arm_ctrl(curr_q)
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        # 6. Final Static Frames (15 frames)
        for _ in range(15):
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))

        return frames
