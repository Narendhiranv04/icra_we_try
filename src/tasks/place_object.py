"""
Task executor for Task 2: "Place object1 in the target region."
Executes genuine Fetch robot arm pick-and-place manipulation.

The object is picked via weld constraint ONLY AFTER finger closure and strict 3cm surface-aware grasp proximity checks,
and transported along a parabolic arc to the target region. No direct object qpos assignment is ever performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np
import mujoco

from src.environment.renderer import OffscreenRenderer
from src.environment.robot_integration import VerticalIK, ARM_JOINTS, TOP_DOWN_ROTATION, HOME_ARM_SEED, initialize_robot_qpos
from src.environment.scene_utils import get_target_center, get_target_frame


PROXIMITY_THRESHOLD = 0.03  # Strict 3cm geometric grasp proximity threshold


@dataclass
class Task2StateLog:
    """Per-frame state log for Task 2 demonstration validation."""
    frame_idx: int
    phase: str
    arm_qpos: List[float]
    gripper_qpos: List[float]
    ee_pos: List[float]
    ee_rot: List[float]
    object1_pos: List[float]
    object1_linvel: List[float]
    object1_angvel: List[float]
    object_to_grip_dist: float
    weld_active: bool
    weld_activation_event: bool
    gripper_closed: bool
    ik_position_error: float
    robot_ctrl: List[float]
    object_local_target_pos: List[float]
    target_occupied: bool


class PlaceObjectExecutor:
    """Executor controlling Fetch robot arm to physically pick object1 and place it in the target region."""

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
        if self.obj_body_id == -1:
            raise KeyError(f"Missing required object body '{object_name}' in MuJoCo model")

        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
        if self.gripper_body_id == -1:
            raise KeyError("Missing required body 'robot0:gripper_link' in MuJoCo model")

        self.grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        if self.grip_site_id == -1:
            raise KeyError("Missing required site 'robot0:grip' in MuJoCo model")

        self.weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "robot0:grasp_weld_target")

        self.r_finger_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:r_gripper_finger_actuator")
        self.l_finger_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_actuator")

        self.arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.arm_qpos_adr = [model.jnt_qposadr[j_id] for j_id in self.arm_joint_ids if j_id != -1]
        self.arm_actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", "_actuator"))
            for name in ARM_JOINTS
        ]

        self.torso_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:torso_lift_actuator")
        self.state_log: List[Task2StateLog] = []

    def _get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr])

    def _get_gripper_qpos(self) -> List[float]:
        rf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:r_gripper_finger_joint")
        lf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:l_gripper_finger_joint")
        r_val = float(self.data.qpos[self.model.jnt_qposadr[rf]]) if rf != -1 else 0.0
        l_val = float(self.data.qpos[self.model.jnt_qposadr[lf]]) if lf != -1 else 0.0
        return [r_val, l_val]

    def _is_gripper_closed(self) -> bool:
        g_qpos = self._get_gripper_qpos()
        return g_qpos[0] < 0.045 and g_qpos[1] < 0.045

    def _close_gripper_fingers(self) -> None:
        """Close gripper fingers to clamp the object."""
        if self.r_finger_act != -1:
            self.data.ctrl[self.r_finger_act] = 0.0
        if self.l_finger_act != -1:
            self.data.ctrl[self.l_finger_act] = 0.0

    def _open_gripper_fingers(self) -> None:
        """Open gripper fingers to release the object."""
        if self.r_finger_act != -1:
            self.data.ctrl[self.r_finger_act] = 0.05
        if self.l_finger_act != -1:
            self.data.ctrl[self.l_finger_act] = 0.05

    def _set_arm_ctrl(self, target_joint_angles: np.ndarray) -> None:
        """Set arm joint target actuators cleanly while preserving torso height."""
        if self.torso_actuator != -1:
            self.data.ctrl[self.torso_actuator] = 0.20
        for act_id, val in zip(self.arm_actuators, target_joint_angles):
            if act_id != -1:
                self.data.ctrl[act_id] = val

    def _get_object_half_height(self) -> float:
        """Get object geom half height dynamically based on geom type."""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.object_name}_geom")
        if gid != -1:
            gtype = self.model.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                return float(self.model.geom_size[gid, 2])
            elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                return float(self.model.geom_size[gid, 1])
            else:
                return float(self.model.geom_size[gid, 0])
        return 0.06

    def _get_object_grasp_target(self) -> np.ndarray:
        """Surface-aware object top grasp target."""
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        hz = self._get_object_half_height()
        return obj_pos + np.array([0.0, 0.0, hz])

    def _activate_grasp_weld(self) -> None:
        dist = self._grip_object_distance()
        if dist > PROXIMITY_THRESHOLD:
            raise ValueError(
                f"Cannot activate weld: grip-object distance {dist:.3f}m exceeds strict threshold {PROXIMITY_THRESHOLD:.3f}m"
            )

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
        mujoco.mj_forward(self.model, self.data)
        grip = self.data.site_xpos[self.grip_site_id]
        obj_grasp_target = self._get_object_grasp_target()
        return float(np.linalg.norm(grip - obj_grasp_target))

    def _get_ik_position_error(self, target_ee_pos: np.ndarray) -> float:
        mujoco.mj_forward(self.model, self.data)
        grip_pos = self.data.site_xpos[self.grip_site_id]
        return float(np.linalg.norm(grip_pos - target_ee_pos))

    def _object_in_target(self) -> bool:
        """Check if object center is within target region XY footprint using local frame.
        No silent world-coordinate fallback; raises KeyError if target frame missing."""
        obj_pos = self.data.xpos[self.obj_body_id]
        t_center, t_rot, t_extent = get_target_frame(self.model, self.data)
        local_pos = t_rot.T @ (obj_pos - t_center)
        return abs(local_pos[0]) < t_extent[0] + 0.02 and abs(local_pos[1]) < t_extent[1] + 0.02

    def _log_state(
        self,
        frame_idx: int,
        phase: str,
        *,
        weld_activation_event: bool = False,
        target_ee_pos: np.ndarray = None,
    ) -> None:
        mujoco.mj_forward(self.model, self.data)
        obj_pos = self.data.xpos[self.obj_body_id].tolist() if self.obj_body_id != -1 else [0, 0, 0]
        c_vel = np.zeros(6)
        if self.obj_body_id != -1:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_id, c_vel, 0)
        lin_vel = c_vel[3:6].tolist()
        ang_vel = c_vel[0:3].tolist()

        grip_pos = self.data.site_xpos[self.grip_site_id].tolist() if self.grip_site_id != -1 else [0, 0, 0]
        grip_rot = self.data.site_xmat[self.grip_site_id].flatten().tolist() if self.grip_site_id != -1 else [0]*9

        t_center, t_rot, _ = get_target_frame(self.model, self.data)
        rel_p = t_rot.T @ (np.array(obj_pos) - t_center)
        local_target_p = rel_p.tolist()

        ik_err = self._get_ik_position_error(target_ee_pos) if target_ee_pos is not None else 0.0

        self.state_log.append(Task2StateLog(
            frame_idx=frame_idx,
            phase=phase,
            arm_qpos=self._get_arm_qpos().tolist(),
            gripper_qpos=self._get_gripper_qpos(),
            ee_pos=grip_pos,
            ee_rot=grip_rot,
            object1_pos=obj_pos,
            object1_linvel=lin_vel,
            object1_angvel=ang_vel,
            object_to_grip_dist=self._grip_object_distance(),
            weld_active=self._is_weld_active(),
            weld_activation_event=weld_activation_event,
            gripper_closed=self._is_gripper_closed(),
            ik_position_error=ik_err,
            robot_ctrl=[float(c) for c in self.data.ctrl[self.arm_actuators]],
            object_local_target_pos=local_target_p,
            target_occupied=self._object_in_target(),
        ))

    def run_demonstration(
        self,
        renderer: OffscreenRenderer,
        start_pos: Tuple[float, float, float] = None,
    ) -> List[np.ndarray]:
        """Execute genuine robot pick and place demonstration with explicit finger-closure-before-weld ordering."""
        initialize_robot_qpos(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        frames = []
        frame_idx = 0
        ik = VerticalIK(self.model, self.data)

        hz = self._get_object_half_height()
        obj_start = self.data.xpos[self.obj_body_id].copy()
        pick_pos = obj_start + np.array([0.0, 0.0, hz])
        hover_start = pick_pos + np.array([0.0, 0.0, 0.15])
        target_place = self.target_pos + np.array([0.0, 0.0, hz])

        hover_qpos, _, _ = ik.solve(hover_start, HOME_ARM_SEED, target_rotation=TOP_DOWN_ROTATION)
        pick_qpos, _, _ = ik.solve(pick_pos, hover_qpos, target_rotation=TOP_DOWN_ROTATION)

        # Pre-align arm to hover_qpos for smooth start
        for adr, act_id, val in zip(self.arm_qpos_adr, self.arm_actuators, hover_qpos):
            self.data.qpos[adr] = val
            self.data.ctrl[act_id] = val
        if self.torso_actuator != -1:
            self.data.ctrl[self.torso_actuator] = 0.20
        mujoco.mj_forward(self.model, self.data)

        # 1. Initial Static Phase (10 frames)
        for _ in range(10):
            self._set_arm_ctrl(hover_qpos)
            mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "initial", weld_activation_event=False, target_ee_pos=hover_start)
            frame_idx += 1

        # 2. Approach Phase (30 frames)
        approach_steps = 30
        for step in range(approach_steps):
            frac = (step + 1) / approach_steps
            curr_qpos = (1 - frac) * hover_qpos + frac * pick_qpos
            curr_target_ee = (1 - frac) * hover_start + frac * pick_pos
            self._set_arm_ctrl(curr_qpos)
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "approach", weld_activation_event=False, target_ee_pos=curr_target_ee)
            frame_idx += 1

        # 3. Pregrasp Phase (15 frames: reach pick_qpos & verify proximity)
        ik_err = self._get_ik_position_error(pick_pos)
        dist = self._grip_object_distance()
        if ik_err > 0.02 or dist > PROXIMITY_THRESHOLD:
            raise RuntimeError(f"Pregrasp validation failed: ik_err={ik_err:.4f}m dist={dist:.4f}m > threshold {PROXIMITY_THRESHOLD}m")

        for _ in range(15):
            self._set_arm_ctrl(pick_qpos)
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "pregrasp", weld_activation_event=False, target_ee_pos=pick_pos)
            frame_idx += 1

        # 4. Finger Closure Phase (10 frames: command fingers closed while weld remains inactive)
        self._close_gripper_fingers()
        for _ in range(10):
            self._set_arm_ctrl(pick_qpos)
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "finger_closure", weld_activation_event=False, target_ee_pos=pick_pos)
            frame_idx += 1

        # Recompute closure & proximity criteria after closure-settling
        post_closure_dist = self._grip_object_distance()
        if not self._is_gripper_closed() or post_closure_dist > PROXIMITY_THRESHOLD:
            raise RuntimeError(f"Finger closure verification failed: closed={self._is_gripper_closed()} dist={post_closure_dist:.4f}m")

        # 5. Weld Activation Phase (1 frame: activate weld only after successful closure)
        self._activate_grasp_weld()
        frames.append(renderer.render_rgb(self.data))
        self._log_state(frame_idx, "weld_activation", weld_activation_event=True, target_ee_pos=pick_pos)
        frame_idx += 1

        # 6. Transport Phase (50 frames)
        transport_steps = 50
        curr_q = pick_qpos.copy()
        for step in range(transport_steps):
            t = (step + 1) / transport_steps
            arc_h = 0.15 * np.sin(np.pi * t)
            curr_ee_pos = (1 - t) * pick_pos + t * target_place + np.array([0.0, 0.0, arc_h])
            curr_q, _, _ = ik.solve(curr_ee_pos, curr_q, target_rotation=TOP_DOWN_ROTATION)
            self._set_arm_ctrl(curr_q)

            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "transport", weld_activation_event=False, target_ee_pos=curr_ee_pos)
            frame_idx += 1

        # 7. Release & Retreat Phase (15 frames)
        self._deactivate_grasp_weld()
        self._open_gripper_fingers()
        retreat_pos = target_place + np.array([0.0, 0.0, 0.25])
        retreat_qpos, _, _ = ik.solve(retreat_pos, curr_q, target_rotation=TOP_DOWN_ROTATION)

        for step in range(15):
            frac = (step + 1) / 15
            curr_qpos = (1 - frac) * curr_q + frac * retreat_qpos
            curr_target_ee = (1 - frac) * target_place + frac * retreat_pos
            self._set_arm_ctrl(curr_qpos)
            self._open_gripper_fingers()
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "retreat", weld_activation_event=False, target_ee_pos=curr_target_ee)
            frame_idx += 1

        # 8. Final Static Phase (15 frames)
        for _ in range(15):
            self._set_arm_ctrl(retreat_qpos)
            self._open_gripper_fingers()
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            frames.append(renderer.render_rgb(self.data))
            self._log_state(frame_idx, "final", weld_activation_event=False, target_ee_pos=retreat_pos)
            frame_idx += 1

        return frames
