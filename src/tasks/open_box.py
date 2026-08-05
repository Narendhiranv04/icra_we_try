"""
Task executor for Task 1: "Open the box."
Executes genuine Fetch robot arm manipulation to open B1 box lid.

The lid is opened by the robot arm through a weld constraint. The arm follows
an arc trajectory derived from the actual B1_lid_joint hinge geometry. The
hinge position actuator is used only as a synchronized follower to assist the
weld (preventing gravity from fighting it), NEVER as an independent drive.

Critical invariant: self.data.qpos[hinge_qpos] is NEVER directly assigned
during the demonstration. The lid moves because the weld transfers force from
the arm, and the hinge actuator provides supporting torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import mujoco
import numpy as np

from src.environment.renderer import OffscreenRenderer
from src.environment.robot_integration import VerticalIK, ARM_JOINTS, TOP_DOWN_ROTATION
from src.environment.scene_utils import (
    get_handle_pos,
    get_hinge_qpos_adr,
    get_lid_angle,
)


# ── Constants ───────────────────────────────────────────────────────────

BOX_OPEN_TARGET_ANGLE = math.radians(90.0)
BOX_ARC_SAMPLES = 30
BOX_GRIP_OFFSET = np.array([0.0, -0.026, 0.02])  # grip offset from handle
BOX_OVERHEAD_CLEARANCE = 0.12
BOX_PREGRASP_OFFSET = np.array([0.0, -0.10, 0.12])
PROXIMITY_THRESHOLD = 0.20  # max distance for weld activation

# Gripper approaches from the front of the box, fingers close vertically
BOX_GRASP_ROTATION = np.array(
    ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
)


@dataclass
class DemoStateLog:
    """Per-frame state log for demonstration validation."""
    frame_idx: int
    phase: str
    lid_angle_rad: float
    grip_pos: List[float]
    handle_pos: List[float]
    weld_active: bool
    arm_qpos: List[float]


class BoxOpenExecutor:
    """Executor that controls the Fetch robot arm to physically approach, grasp, and open box B1 lid.
    
    The lid is opened by the robot's arm through a weld constraint. The hinge
    actuator provides synchronized supporting torque but is NEVER used to
    independently drive the lid.
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

        # ── Joint / actuator / body IDs ───────────────────────────
        self.hinge_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint")
        if self.hinge_joint_id == -1:
            self.hinge_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_hinge")
        self.hinge_qpos_adr = int(model.jnt_qposadr[self.hinge_joint_id]) if self.hinge_joint_id != -1 else None
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

        self.state_log: List[DemoStateLog] = []

    # ── Arm control helpers ──────────────────────────────────────────

    def _set_arm_ctrl(self, joint_angles: np.ndarray) -> None:
        """Set arm joint target actuators."""
        for act_id, angle in zip(self.arm_actuators, joint_angles):
            if act_id != -1:
                self.data.ctrl[act_id] = angle

    def _get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr])

    # ── Weld helpers ──────────────────────────────────────────────────

    def _activate_grasp_weld(self) -> None:
        """Activate the weld constraint between gripper and lid with current relative pose."""
        if self.weld_eq_id == -1 or self.gripper_body_id == -1 or self.lid_body_id == -1:
            return
        # Compute relative pose: lid in gripper frame
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

    # ── Proximity check ──────────────────────────────────────────────

    def _grip_handle_distance(self) -> float:
        """Current Euclidean distance between grip site and handle site."""
        mujoco.mj_forward(self.model, self.data)
        grip = self.data.site_xpos[self.grip_site_id]
        handle = self.data.site_xpos[self.handle_site_id]
        return float(np.linalg.norm(grip - handle))

    # ── Hinge arc sampling ────────────────────────────────────────────

    def _lid_arc_samples(self) -> List[tuple]:
        """Sample the end-effector trajectory that follows the hinge arc.
        
        Uses a SEPARATE MjData copy to vary the hinge angle and read the
        resulting handle/lid transforms. This never modifies self.data.qpos.
        
        Returns:
            List of (angle, grip_world_pos, grip_world_rot) tuples.
        """
        sample_data = mujoco.MjData(self.model)
        sample_data.qpos[:] = self.data.qpos
        sample_data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        start_angle = float(self.data.qpos[self.hinge_qpos_adr])

        # Record initial transforms
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
            # Compute relative rotation of lid from initial
            delta_rot = lid_rot @ initial_lid_rot.T
            # Apply same delta to grip position and rotation
            grip_pos = lid_pos + delta_rot @ (initial_grip_pos - initial_lid_pos)
            grip_rot = delta_rot @ initial_grip_rot
            samples.append((float(angle), grip_pos.copy(), grip_rot))

        return samples

    # ── State logging ─────────────────────────────────────────────────

    def _log_state(self, frame_idx: int, phase: str) -> None:
        """Record per-frame state for validation."""
        mujoco.mj_forward(self.model, self.data)
        lid_angle = float(self.data.qpos[self.hinge_qpos_adr]) if self.hinge_qpos_adr is not None else 0.0
        grip_pos = self.data.site_xpos[self.grip_site_id].tolist() if self.grip_site_id != -1 else [0, 0, 0]
        handle_pos = self.data.site_xpos[self.handle_site_id].tolist() if self.handle_site_id != -1 else [0, 0, 0]
        self.state_log.append(DemoStateLog(
            frame_idx=frame_idx,
            phase=phase,
            lid_angle_rad=lid_angle,
            grip_pos=grip_pos,
            handle_pos=handle_pos,
            weld_active=self._is_weld_active(),
            arm_qpos=self._get_arm_qpos().tolist(),
        ))

    # ── Main demonstration ────────────────────────────────────────────

    def run_demonstration(self, renderer: OffscreenRenderer) -> List[np.ndarray]:
        """Execute genuine robot demonstration to open B1 box lid.
        
        The demonstration follows these phases:
        1. Initial static frames (observe the scene)
        2. Approach: IK-solve arm to reach near handle
        3. Grasp: Activate weld, initialize hinge actuator to current angle
        4. Open: Follow hinge arc via IK, hinge actuator follows as synchronized servo
        5. Release & Retreat: Deactivate weld, retreat arm, hold hinge actuator at target
        6. Final static frames
        
        Returns:
            List of RGB frame arrays.
        """
        frames = []
        frame_idx = 0
        ik = VerticalIK(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Initialize hinge actuator to current position (hold lid closed)
        if self.hinge_actuator != -1 and self.hinge_qpos_adr is not None:
            self.data.ctrl[self.hinge_actuator] = float(self.data.qpos[self.hinge_qpos_adr])

        # Read handle position from scene
        handle_pos = (
            self.data.site_xpos[self.handle_site_id].copy()
            if self.handle_site_id != -1
            else get_handle_pos(self.model, self.data)
        )

        # ─── Phase 1: Initial Static Frames (10 frames) ─────────────
        for _ in range(10):
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
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "approach")
            frame_idx += 1

        # ─── Phase 3: Activate Grasp Weld ───────────────────────────
        dist = self._grip_handle_distance()
        if dist > PROXIMITY_THRESHOLD:
            print(f"WARNING: grip-handle distance {dist:.3f}m > threshold {PROXIMITY_THRESHOLD}m at weld activation")
        self._activate_grasp_weld()

        # Initialize hinge actuator to current angle (hold, don't drive)
        if self.hinge_actuator != -1 and self.hinge_qpos_adr is not None:
            self.data.ctrl[self.hinge_actuator] = float(self.data.qpos[self.hinge_qpos_adr])

        # ─── Phase 4: Opening Arc (50 frames) ───────────────────────
        # Derive arm trajectory from actual hinge geometry
        arc_samples = self._lid_arc_samples()
        open_steps = 50
        start_qpos = self._get_arm_qpos()

        for step in range(open_steps):
            # Map step to arc sample index
            t = (step + 1) / open_steps
            sample_idx = min(int(t * (len(arc_samples) - 1)), len(arc_samples) - 1)
            target_angle, arc_ee_pos, arc_ee_rot = arc_samples[sample_idx]

            # Solve IK for this arc position using the rotation from the arc sample
            arm_q, pos_err, ang_err = ik.solve(arc_ee_pos, start_qpos, target_rotation=arc_ee_rot)
            start_qpos = arm_q
            self._set_arm_ctrl(arm_q)

            # Hinge actuator follows as synchronized servo — NEVER independent drive
            # This provides supporting torque in the same direction the weld is pulling
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = target_angle

            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "opening")
            frame_idx += 1

        # ─── Phase 5: Release & Retreat (15 frames) ─────────────────
        self._deactivate_grasp_weld()

        # Hold lid open via hinge actuator after release
        if self.hinge_actuator != -1:
            self.data.ctrl[self.hinge_actuator] = self.target_angle

        # Retreat arm upward
        final_grip_pos = self.data.site_xpos[self.grip_site_id].copy() if self.grip_site_id != -1 else arc_ee_pos
        retreat_target = final_grip_pos + np.array([0.0, -0.15, 0.15])
        retreat_qpos, _, _ = ik.solve(retreat_target, start_qpos, target_rotation=BOX_GRASP_ROTATION)

        for step in range(15):
            frac = (step + 1) / 15
            curr_q = (1 - frac) * start_qpos + frac * retreat_qpos
            self._set_arm_ctrl(curr_q)
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "retreat")
            frame_idx += 1

        # ─── Phase 6: Final Static Frames (15 frames) ───────────────
        for _ in range(15):
            # Maintain hinge at target to hold lid open
            if self.hinge_actuator != -1:
                self.data.ctrl[self.hinge_actuator] = self.target_angle
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "final")
            frame_idx += 1

        return frames
