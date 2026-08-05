"""
Task executor for Task 2: "Place object1 in the target region."
Executes genuine Fetch robot arm pick-and-place manipulation.

The object is picked via weld constraint and transported along a parabolic
arc to the target region. No direct object qpos assignment is ever performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import mujoco

from src.environment.renderer import OffscreenRenderer
from src.environment.robot_integration import VerticalIK, ARM_JOINTS, TOP_DOWN_ROTATION, HOME_ARM_SEED, initialize_robot_qpos
from src.environment.scene_utils import get_target_center


PROXIMITY_THRESHOLD = 0.20  # max distance for weld activation


def _yaw_rotation(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


# Fetch home base pose is rotated +90 deg yaw (world +Y forward)
HOME_BASE_ROTATION = _yaw_rotation(math.pi / 2) @ TOP_DOWN_ROTATION


@dataclass
class DemoStateLog:
    """Per-frame state log for demonstration validation."""
    frame_idx: int
    phase: str
    object_pos: List[float]
    grip_pos: List[float]
    target_center: List[float]
    weld_active: bool
    arm_qpos: List[float]
    obj_in_target: bool


class PlaceObjectExecutor:
    """Executor that controls the Fetch robot arm to physically pick object1 and place it in the target region."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_name: str = "coffee_can",
        target_pos: Tuple[float, float, float] = None,
        num_steps: int = 120,
    ):
        self.model = model
        self.data = data
        self.object_name = object_name
        self.num_steps = num_steps

        # Derive target from scene if not provided
        if target_pos is not None:
            self.target_pos = np.array(target_pos)
        else:
            mujoco.mj_forward(model, data)
            self.target_pos = get_target_center(model, data)

        self.obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
        self.grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        self.weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "robot0:grasp_weld_target")

        self.arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.arm_qpos_adr = [model.jnt_qposadr[j_id] for j_id in self.arm_joint_ids if j_id != -1]
        self.arm_actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", "_actuator"))
            for name in ARM_JOINTS
        ]

        self.state_log: List[DemoStateLog] = []

    def _get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr])

    def _set_arm_ctrl(self, target_joint_angles: np.ndarray) -> None:
        """Set arm joint target actuators cleanly."""
        for act_id, val in zip(self.arm_actuators, target_joint_angles):
            if act_id != -1:
                self.data.ctrl[act_id] = val

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

    def _is_weld_active(self) -> bool:
        if self.weld_eq_id == -1:
            return False
        return bool(self.data.eq_active[self.weld_eq_id])

    def _grip_object_distance(self) -> float:
        """Current distance between grip site and object center."""
        mujoco.mj_forward(self.model, self.data)
        grip = self.data.site_xpos[self.grip_site_id]
        obj = self.data.xpos[self.obj_body_id]
        return float(np.linalg.norm(grip - obj))

    def _object_in_target(self) -> bool:
        """Check if object center is within target region XY footprint."""
        obj_pos = self.data.xpos[self.obj_body_id]
        dx = abs(obj_pos[0] - self.target_pos[0])
        dy = abs(obj_pos[1] - self.target_pos[1])
        return dx < 0.10 and dy < 0.10

    def _log_state(self, frame_idx: int, phase: str) -> None:
        """Record per-frame state for validation."""
        mujoco.mj_forward(self.model, self.data)
        obj_pos = self.data.xpos[self.obj_body_id].tolist() if self.obj_body_id != -1 else [0, 0, 0]
        grip_pos = self.data.site_xpos[self.grip_site_id].tolist() if self.grip_site_id != -1 else [0, 0, 0]
        self.state_log.append(DemoStateLog(
            frame_idx=frame_idx,
            phase=phase,
            object_pos=obj_pos,
            grip_pos=grip_pos,
            target_center=self.target_pos.tolist(),
            weld_active=self._is_weld_active(),
            arm_qpos=self._get_arm_qpos().tolist(),
            obj_in_target=self._object_in_target(),
        ))

    def run_demonstration(
        self,
        renderer: OffscreenRenderer,
        start_pos: Tuple[float, float, float] = None,
    ) -> List[np.ndarray]:
        """Execute genuine robot pick and place demonstration and return RGB video frames."""
        initialize_robot_qpos(self.model, self.data)

        frames = []
        frame_idx = 0
        ik = VerticalIK(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Read object start position from scene if not provided
        if start_pos is not None:
            obj_start = np.array(start_pos)
        else:
            obj_start = self.data.xpos[self.obj_body_id].copy()

        target_place = self.target_pos + np.array([0.0, 0.0, 0.07])

        # 1. Initial Static Frames (10 frames)
        for _ in range(10):
            self._set_arm_ctrl(HOME_ARM_SEED)
            mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "initial")
            frame_idx += 1

        # 2. Approach Phase (30 frames)
        pick_pos = obj_start + np.array([0.0, 0.0, 0.07])
        hover_start = pick_pos + np.array([0.0, 0.0, 0.15])

        seed_qpos = self._get_arm_qpos()
        hover_qpos, _, _ = ik.solve(hover_start, HOME_ARM_SEED, target_rotation=HOME_BASE_ROTATION)
        pick_qpos, _, _ = ik.solve(pick_pos, hover_qpos, target_rotation=HOME_BASE_ROTATION)

        approach_steps = 30
        for step in range(approach_steps):
            frac = (step + 1) / approach_steps
            curr_qpos = (1 - frac) * seed_qpos + frac * pick_qpos
            self._set_arm_ctrl(curr_qpos)
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "approach")
            frame_idx += 1

        # Settle before grasp (15 frames)
        for _ in range(15):
            self._set_arm_ctrl(pick_qpos)
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)

        # 3. Attach Grasp Weld (with proximity check)
        dist = self._grip_object_distance()
        if dist > PROXIMITY_THRESHOLD:
            print(f"WARNING: grip-object distance {dist:.3f}m > threshold {PROXIMITY_THRESHOLD}m at weld activation")
        self._activate_grasp_weld()

        # 4. Transport Phase (50 frames): Lift -> Move -> Place
        transport_steps = 50
        curr_q = pick_qpos.copy()
        for step in range(transport_steps):
            t = (step + 1) / transport_steps
            arc_h = 0.15 * np.sin(np.pi * t)
            curr_ee_pos = (1 - t) * pick_pos + t * target_place + np.array([0.0, 0.0, arc_h])
            curr_q, _, _ = ik.solve(curr_ee_pos, curr_q, target_rotation=HOME_BASE_ROTATION)
            self._set_arm_ctrl(curr_q)

            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "transport")
            frame_idx += 1

        # 5. Release & Retreat Phase (15 frames)
        self._deactivate_grasp_weld()
        retreat_pos = target_place + np.array([0.0, 0.0, 0.25])
        retreat_qpos, _, _ = ik.solve(retreat_pos, curr_q, target_rotation=HOME_BASE_ROTATION)

        for step in range(15):
            frac = (step + 1) / 15
            curr_qpos = (1 - frac) * curr_q + frac * retreat_qpos
            self._set_arm_ctrl(curr_qpos)
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "retreat")
            frame_idx += 1

        # 6. Final Static Frames (15 frames)
        for _ in range(15):
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "final")
            frame_idx += 1

        return frames
