"""
Task executor for Task 1: "Open the box."
Executes genuine Fetch robot arm manipulation to open B1 box lid.

The lid is opened by the robot arm through a weld constraint. The hinge is
completely passive; B1_lid_actuator is NEVER commanded with opening angles,
and self.data.qpos[hinge_qpos] is NEVER directly assigned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from src.environment.renderer import OffscreenRenderer
from src.environment.robot_integration import VerticalIK, ARM_JOINTS, initialize_robot_qpos
from src.environment.scene_utils import get_handle_pos, get_lid_angle


# ── Constants ───────────────────────────────────────────────────────────

BOX_OPEN_TARGET_ANGLE = math.radians(90.0)
BOX_ARC_SAMPLES = 30
BOX_GRIP_OFFSET = np.array([0.0, -0.026, 0.02])
BOX_PREGRASP_OFFSET = np.array([0.0, -0.10, 0.12])
PROXIMITY_THRESHOLD = 0.30  # max distance between grip site and handle site for weld activation

# Gripper approaches from front of box
BOX_GRASP_ROTATION = np.array(
    ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
)


@dataclass
class Task1StateLog:
    """Per-frame state log for Task 1 demonstration validation."""
    frame_idx: int
    phase: str
    arm_qpos: List[float]
    gripper_qpos: List[float]
    ee_pos: List[float]
    ee_rot: List[float]
    handle_pos: List[float]
    handle_to_grip_dist: float
    lid_angle_rad: float
    lid_vel_radps: float
    weld_active: bool
    robot_ctrl: List[float]
    lid_ctrl: float  # Must be 0.0 always


class BoxOpenExecutor:
    """Executor controlling Fetch robot arm to physically approach, grasp, and open box B1 lid.
    
    The lid hinge remains strictly passive. B1_lid_actuator is NEVER used to drive or hold the lid.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_angle: float = BOX_OPEN_TARGET_ANGLE,
    ):
        self.model = model
        self.data = data
        self.target_angle = target_angle

        # IDs
        self.hinge_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint")
        if self.hinge_joint_id == -1:
            self.hinge_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_hinge")
        self.hinge_qpos_adr = int(model.jnt_qposadr[self.hinge_joint_id]) if self.hinge_joint_id != -1 else None
        self.hinge_dof_adr = int(model.jnt_dofadr[self.hinge_joint_id]) if self.hinge_joint_id != -1 else None
        self.hinge_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "B1_lid_actuator")

        self.handle_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "B1_lid_handle_grasp")
        self.grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
        self.lid_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B1_lid")
        self.weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "robot0:open_weld_B1_lid")

        self.r_finger_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:r_gripper_finger_actuator")
        self.l_finger_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_actuator")

        self.arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.arm_qpos_adr = [model.jnt_qposadr[j_id] for j_id in self.arm_joint_ids if j_id != -1]
        self.arm_actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", "_actuator"))
            for name in ARM_JOINTS
        ]

        self.torso_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:torso_lift_actuator")
        self.state_log: List[Task1StateLog] = []

    def _set_arm_ctrl(self, joint_angles: np.ndarray) -> None:
        """Set arm joint target actuators cleanly while preserving torso height."""
        if self.torso_actuator != -1:
            self.data.ctrl[self.torso_actuator] = 0.20
        for act_id, angle in zip(self.arm_actuators, joint_angles):
            if act_id != -1:
                self.data.ctrl[act_id] = angle

    def _get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr])

    def _get_gripper_qpos(self) -> List[float]:
        rf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:r_gripper_finger_joint")
        lf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:l_gripper_finger_joint")
        r_val = float(self.data.qpos[self.model.jnt_qposadr[rf]]) if rf != -1 else 0.0
        l_val = float(self.data.qpos[self.model.jnt_qposadr[lf]]) if lf != -1 else 0.0
        return [r_val, l_val]

    def _close_gripper_fingers(self) -> None:
        """Close gripper fingers to grip the handle."""
        if self.r_finger_act != -1:
            self.data.ctrl[self.r_finger_act] = 0.0
        if self.l_finger_act != -1:
            self.data.ctrl[self.l_finger_act] = 0.0

    def _activate_grasp_weld(self) -> None:
        """Activate weld constraint after proximity and gripper closure checks."""
        dist = self._grip_handle_distance()
        if dist > PROXIMITY_THRESHOLD:
            raise ValueError(
                f"Cannot activate weld: grip-handle distance {dist:.3f}m exceeds strict threshold {PROXIMITY_THRESHOLD:.3f}m"
            )

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

    def _is_weld_active(self) -> bool:
        if self.weld_eq_id == -1:
            return False
        return bool(self.data.eq_active[self.weld_eq_id])

    def _grip_handle_distance(self) -> float:
        mujoco.mj_forward(self.model, self.data)
        grip = self.data.site_xpos[self.grip_site_id]
        handle = self.data.site_xpos[self.handle_site_id]
        return float(np.linalg.norm(grip - handle))

    def _lid_arc_samples(self) -> List[tuple]:
        """Sample hinge arc using a temporary MjData copy without mutating self.data."""
        sample_data = mujoco.MjData(self.model)
        sample_data.qpos[:] = self.data.qpos
        sample_data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        start_angle = float(self.data.qpos[self.hinge_qpos_adr])

        initial_lid_pos = self.data.xpos[self.lid_body_id].copy()
        initial_lid_rot = self.data.xmat[self.lid_body_id].reshape(3, 3).copy()
        initial_grip_pos = self.data.site_xpos[self.grip_site_id].copy()
        initial_grip_rot = self.data.site_xmat[self.grip_site_id].reshape(3, 3).copy()

        samples = []
        for angle in np.linspace(start_angle, self.target_angle, BOX_ARC_SAMPLES):
            sample_data.qpos[self.hinge_qpos_adr] = angle
            mujoco.mj_forward(self.model, sample_data)
            lid_rot = sample_data.xmat[self.lid_body_id].reshape(3, 3).copy()
            lid_pos = sample_data.xpos[self.lid_body_id].copy()
            delta_rot = lid_rot @ initial_lid_rot.T
            grip_pos = lid_pos + delta_rot @ (initial_grip_pos - initial_lid_pos)
            grip_rot = delta_rot @ initial_grip_rot
            samples.append((float(angle), grip_pos.copy(), grip_rot))

        return samples

    def _log_state(self, frame_idx: int, phase: str) -> None:
        mujoco.mj_forward(self.model, self.data)
        lid_angle = float(self.data.qpos[self.hinge_qpos_adr]) if self.hinge_qpos_adr is not None else 0.0
        lid_vel = float(self.data.qvel[self.hinge_dof_adr]) if self.hinge_dof_adr is not None else 0.0
        grip_pos = self.data.site_xpos[self.grip_site_id].tolist() if self.grip_site_id != -1 else [0, 0, 0]
        grip_rot = self.data.site_xmat[self.grip_site_id].flatten().tolist() if self.grip_site_id != -1 else [0]*9
        handle_pos = self.data.site_xpos[self.handle_site_id].tolist() if self.handle_site_id != -1 else [0, 0, 0]
        handle_dist = self._grip_handle_distance()
        lid_ctrl_val = float(self.data.ctrl[self.hinge_actuator]) if self.hinge_actuator != -1 else 0.0

        self.state_log.append(Task1StateLog(
            frame_idx=frame_idx,
            phase=phase,
            arm_qpos=self._get_arm_qpos().tolist(),
            gripper_qpos=self._get_gripper_qpos(),
            ee_pos=grip_pos,
            ee_rot=grip_rot,
            handle_pos=handle_pos,
            handle_to_grip_dist=handle_dist,
            lid_angle_rad=lid_angle,
            lid_vel_radps=lid_vel,
            weld_active=self._is_weld_active(),
            robot_ctrl=[float(c) for c in self.data.ctrl[self.arm_actuators]],
            lid_ctrl=lid_ctrl_val,
        ))

    def run_demonstration(self, renderer: OffscreenRenderer) -> List[np.ndarray]:
        """Execute genuine Task 1 demonstration.
        
        The lid actuator is NEVER commanded with opening torque. The hinge remains passive.
        """
        initialize_robot_qpos(self.model, self.data)

        # Zero out lid actuator control
        if self.hinge_actuator != -1:
            self.data.ctrl[self.hinge_actuator] = 0.0

        frames = []
        frame_idx = 0
        ik = VerticalIK(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        handle_pos = (
            self.data.site_xpos[self.handle_site_id].copy()
            if self.handle_site_id != -1
            else get_handle_pos(self.model, self.data)
        )

        # ─── Phase 1: Initial Static Frames (10 frames) ─────────────
        for _ in range(10):
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = 0.0
            mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "initial")
            frame_idx += 1

        # ─── Phase 2: Approach (30 frames) ──────────────────────────
        hover_target = handle_pos + BOX_PREGRASP_OFFSET
        grasp_target = handle_pos + BOX_GRIP_OFFSET

        seed_qpos = self._get_arm_qpos()
        hover_qpos, _, _ = ik.solve(hover_target, seed_qpos, target_rotation=BOX_GRASP_ROTATION)
        grasp_qpos, _, _ = ik.solve(grasp_target, hover_qpos, target_rotation=BOX_GRASP_ROTATION)

        approach_steps = 30
        for step in range(approach_steps):
            frac = (step + 1) / approach_steps
            curr_qpos = (1 - frac) * seed_qpos + frac * grasp_qpos
            self._set_arm_ctrl(curr_qpos)
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = 0.0
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "approach")
            frame_idx += 1

        # Settle at handle target (10 frames)
        for _ in range(10):
            self._set_arm_ctrl(grasp_qpos)
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = 0.0
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)

        # ─── Phase 3: Close Gripper & Activate Weld ─────────────────
        self._close_gripper_fingers()
        for _ in range(10):
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = 0.0
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "grasp")
            frame_idx += 1

        self._activate_grasp_weld()

        # ─── Phase 4: Opening Arc (50 frames) ───────────────────────
        arc_samples = self._lid_arc_samples()
        open_steps = 50
        start_qpos = self._get_arm_qpos()

        for step in range(open_steps):
            t = (step + 1) / open_steps
            sample_idx = min(int(t * (len(arc_samples) - 1)), len(arc_samples) - 1)
            target_angle, arc_ee_pos, arc_ee_rot = arc_samples[sample_idx]

            arm_q, _, _ = ik.solve(arc_ee_pos, start_qpos, target_rotation=arc_ee_rot)
            start_qpos = arm_q
            self._set_arm_ctrl(arm_q)

            # Lid actuator remains at ZERO (passive hinge)
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = 0.0

            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "opening")
            frame_idx += 1

        # ─── Phase 5: Final Static Frames (15 frames) ───────────────
        # End effector holds lid in open position via weld constraint
        for _ in range(15):
            self._set_arm_ctrl(start_qpos)
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = 0.0
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "final")
            frame_idx += 1

        return frames
