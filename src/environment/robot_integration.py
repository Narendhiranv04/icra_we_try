"""
Robot integration and IK utilities for MuJoCo Fetch robot arm manipulation.
"""

import copy
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

import gymnasium_robotics

FETCH_ASSET_DIR = Path(gymnasium_robotics.__file__).parent / "envs" / "assets" / "fetch"

FETCH_BASE_POSES = {
    "home": {"pos": "0 -0.95 0", "quat": "0.7071068 0 0 0.7071068"},
    "right_side": {"pos": "1.025 -0.10 0", "quat": "0.7071068 0 0 -0.7071068"},
}

FETCH_HOME_QPOS = {
    "robot0:base_forward_joint": 0.0,
    "robot0:base_lateral_joint": 0.0,
    "robot0:base_yaw_joint": 0.0,
    "robot0:torso_lift_joint": 0.20,
    "robot0:head_pan_joint": -0.25,
    "robot0:head_tilt_joint": 0.35,
    "robot0:shoulder_pan_joint": 1.32,
    "robot0:shoulder_lift_joint": 1.40,
    "robot0:upperarm_roll_joint": -0.20,
    "robot0:elbow_flex_joint": 1.72,
    "robot0:forearm_roll_joint": 0.0,
    "robot0:wrist_flex_joint": 1.66,
    "robot0:wrist_roll_joint": 0.0,
    "robot0:r_gripper_finger_joint": 0.035,
    "robot0:l_gripper_finger_joint": 0.035,
}

HOME_ARM_SEED = np.array((1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0))

ARM_JOINTS = (
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:upperarm_roll_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
)

FETCH_ACTUATORS = (
    ("robot0:base_forward_actuator", "robot0:base_forward_joint", 6000, -1.0, 1.0),
    ("robot0:base_lateral_actuator", "robot0:base_lateral_joint", 6000, -1.5, 1.5),
    ("robot0:base_yaw_actuator", "robot0:base_yaw_joint", 3500, -3.14, 3.14),
    ("robot0:torso_lift_actuator", "robot0:torso_lift_joint", 3000, 0.0386, 0.3861),
    ("robot0:head_pan_actuator", "robot0:head_pan_joint", 100, -1.57, 1.57),
    ("robot0:head_tilt_actuator", "robot0:head_tilt_joint", 100, -0.76, 1.45),
    ("robot0:shoulder_pan_actuator", "robot0:shoulder_pan_joint", 650, -1.6056, 1.6056),
    ("robot0:shoulder_lift_actuator", "robot0:shoulder_lift_joint", 650, -1.221, 1.518),
    ("robot0:upperarm_roll_actuator", "robot0:upperarm_roll_joint", 350, -3.14, 3.14),
    ("robot0:elbow_flex_actuator", "robot0:elbow_flex_joint", 650, -2.251, 2.251),
    ("robot0:forearm_roll_actuator", "robot0:forearm_roll_joint", 350, -3.14, 3.14),
    ("robot0:wrist_flex_actuator", "robot0:wrist_flex_joint", 350, -2.16, 2.16),
    ("robot0:wrist_roll_actuator", "robot0:wrist_roll_joint", 250, -3.14, 3.14),
    ("robot0:r_gripper_finger_actuator", "robot0:r_gripper_finger_joint", 1000, 0.0, 0.05),
    ("robot0:l_gripper_finger_actuator", "robot0:l_gripper_finger_joint", 1000, 0.0, 0.05),
)

TOP_DOWN_ROTATION = np.array(
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
)


def _remove_named_body(parent: ET.Element, body_name: str) -> None:
    for body in list(parent.findall("body")):
        if body.get("name") == body_name:
            parent.remove(body)


def inject_fetch_robot(
    root: ET.Element,
    base_pose_name: str = "home",
    spawn_welds: bool = True,
    weld_target_body: str = "coffee_can",
) -> None:
    """Inject Fetch mobile manipulator into kitchen MJCF tree."""
    fetch_dir = FETCH_ASSET_DIR
    shared_root = ET.parse(fetch_dir / "shared.xml").getroot()
    robot_root = ET.parse(fetch_dir / "robot.xml").getroot()

    asset = root.find("asset")
    default = root.find("default")
    worldbody = root.find("worldbody")
    contact = root.find("contact")
    actuator = root.find("actuator")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    # Add shared assets
    shared_asset = shared_root.find("asset")
    for element in shared_asset:
        name = element.get("name", "")
        if element.tag == "mesh" or name.startswith("robot0:"):
            asset.append(copy.deepcopy(element))

    fetch_defaults = shared_root.find("default/default")
    default.append(copy.deepcopy(fetch_defaults))

    for exclusion in shared_root.findall("contact/exclude"):
        contact.append(copy.deepcopy(exclusion))

    robot_body = None
    for body in robot_root.findall("body"):
        if body.get("name") == "robot0:base_link":
            robot_body = copy.deepcopy(body)
            break
    if robot_body is None:
        raise RuntimeError("Fetch robot.xml does not contain robot0:base_link")

    base_pose = FETCH_BASE_POSES.get(base_pose_name, FETCH_BASE_POSES["home"])
    robot_body.set("pos", base_pose["pos"])
    robot_body.set("quat", base_pose["quat"])

    base_visual = robot_body.find("geom[@name='robot0:base_link']")
    if base_visual is not None:
        base_visual.set("contype", "0")
        base_visual.set("conaffinity", "0")
        base_visual.set("group", "1")
    ET.SubElement(
        robot_body,
        "geom",
        {
            "name": "robot0:base_collision_proxy",
            "type": "cylinder",
            "size": "0.27 0.18",
            "pos": "0 0 0.18",
            "condim": "1",
            "priority": "2",
            "friction": "0 0 0",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0 0 0 0",
            "group": "3",
        },
    )

    base_joints = robot_body.findall("joint")[:3]
    joint_specs = (
        ("robot0:base_forward_joint", "slide", "1 0 0", "-1 1", "500"),
        ("robot0:base_lateral_joint", "slide", "0 1 0", "-1.5 1.5", "500"),
        ("robot0:base_yaw_joint", "hinge", "0 0 1", "-3.14 3.14", "100"),
    )
    for joint, (name, joint_type, axis, joint_range, damping) in zip(base_joints, joint_specs):
        joint.attrib.clear()
        joint.set("name", name)
        joint.set("type", joint_type)
        joint.set("axis", axis)
        joint.set("range", joint_range)
        joint.set("limited", "true")
        joint.set("damping", damping)
        joint.set("armature", "0.1")

    _remove_named_body(robot_body, "robot0:external_camera_body_0")
    for camera in robot_body.iter("camera"):
        if camera.get("name") == "gripper_camera_rgb":
            camera.set("name", "robot0:gripper_camera_rgb_legacy")

    # Inject canonical egocentric camera into Fetch head optical frame
    head_optical_frame = None
    for body in robot_body.iter("body"):
        if body.get("name") == "robot0:head_camera_rgb_optical_frame":
            head_optical_frame = body
            break
    if head_optical_frame is not None:
        ET.SubElement(
            head_optical_frame,
            "camera",
            {
                "name": "robot0:ego_camera",
                "pos": "0 0 0",
                "euler": "2.85 0 0",
                "fovy": "65",
            },
        )

    gripper_body = None
    for body in robot_body.iter("body"):
        if body.get("name") == "robot0:gripper_link":
            gripper_body = body
            break

    if gripper_body is not None:
        for side, lateral in (("r", -0.008), ("l", 0.008)):
            finger_name = f"robot0:{side}_gripper_finger_link"
            finger_body = next(
                (b for b in gripper_body.iter("body") if b.get("name") == finger_name), None
            )
            if finger_body is not None:
                visual = finger_body.find(f"geom[@name='{finger_name}']")
                if visual is not None:
                    visual.set("name", f"{finger_name}_visual")
                    visual.set("contype", "0")
                    visual.set("conaffinity", "0")
                ET.SubElement(
                    finger_body,
                    "geom",
                    {
                        "name": finger_name,
                        "type": "box",
                        "pos": f"-0.00425 {lateral} 0",
                        "size": "0.03425 0.007 0.0135",
                        "condim": "4",
                        "friction": "1 0.05 0.01",
                        "rgba": "0 0 0 0",
                        "group": "3",
                    },
                )
        ET.SubElement(
            gripper_body,
            "camera",
            {
                "name": "wrist_camera",
                "pos": "0.03 0 0.04",
                "xyaxes": "0 -1 0 -0.7936 0 0.6085",
                "fovy": "65",
            },
        )

    for body in robot_body.iter("body"):
        if body.find("inertial") is not None:
            body.set("gravcomp", "1")
    for joint in robot_body.iter("joint"):
        name = joint.get("name", "")
        if name.startswith("robot0:") and "base_" not in name:
            if "torso_lift" in name:
                damping, armature = "50", "1"
            elif "gripper_finger" in name:
                damping, armature = "20", "0.2"
            elif "head_" in name:
                damping, armature = "5", "0.1"
            else:
                damping, armature = "10", "0.5"
            joint.set("damping", damping)
            joint.set("armature", armature)
            joint.attrib.pop("stiffness", None)

    _remove_named_body(worldbody, "wrist_camera_mount")
    worldbody.append(robot_body)

    for name, joint, kp, ctrl_min, ctrl_max in FETCH_ACTUATORS:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": joint,
                "kp": str(kp),
                "ctrlrange": f"{ctrl_min} {ctrl_max}",
            },
        )

    if spawn_welds:
        ET.SubElement(
            equality,
            "weld",
            {
                "name": "robot0:open_weld_B1_lid",
                "body1": "robot0:gripper_link",
                "body2": "B1_lid",
                "active": "false",
                "relpose": "0 0 0 1 0 0 0",
            },
        )
        if weld_target_body:
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": "robot0:grasp_weld_target",
                    "body1": "robot0:gripper_link",
                    "body2": weld_target_body,
                    "active": "false",
                    "relpose": "0 0 0 1 0 0 0",
                },
            )


def initialize_robot_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set default home qpos for Fetch robot joints."""
    for j_name, val in FETCH_HOME_QPOS.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id != -1:
            q_adr = model.jnt_qposadr[j_id]
            data.qpos[q_adr] = val
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j_name.replace("_joint", "_actuator"))
            if act_id != -1:
                data.ctrl[act_id] = val
    mujoco.mj_forward(model, data)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.ravel())
    if quat[0] < 0:
        quat = -quat
    norm = float(np.linalg.norm(quat[1:]))
    if norm < 1e-10:
        return np.zeros(3)
    return quat[1:] / norm * (2.0 * math.atan2(norm, float(quat[0])))


class VerticalIK:
    """Damped least-squares IK for Fetch 7-DOF arm."""

    def __init__(self, model: mujoco.MjModel, reference: mujoco.MjData):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = reference.qpos
        self.data.qvel[:] = 0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        )
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        limits = model.jnt_range[self.joint_ids]
        limited = model.jnt_limited[self.joint_ids].astype(bool)
        self.lower = np.where(limited, limits[:, 0] + 0.01, -math.pi + 0.01)
        self.upper = np.where(limited, limits[:, 1] - 0.01, math.pi - 0.01)

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        target_rotation: np.ndarray = TOP_DOWN_ROTATION,
    ) -> Tuple[np.ndarray, float, float]:
        self.data.qpos[self.qpos_addresses] = seed
        for _ in range(500):
            mujoco.mj_forward(self.model, self.data)
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            position_error = target - self.data.site_xpos[self.site_id]
            rotation_error = _rotation_vector(target_rotation @ current_rotation.T)
            error = np.concatenate((position_error, 0.40 * rotation_error))
            if np.linalg.norm(position_error) < 0.001 and np.linalg.norm(rotation_error) < math.radians(1.0):
                break

            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.site_id)
            jacobian = np.vstack(
                (
                    jac_pos[:, self.dof_addresses],
                    0.40 * jac_rot[:, self.dof_addresses],
                )
            )
            damping = 0.0015
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            current = self.data.qpos[self.qpos_addresses]
            self.data.qpos[self.qpos_addresses] = np.clip(
                current + np.clip(delta, -0.06, 0.06), self.lower, self.upper
            )

        mujoco.mj_forward(self.model, self.data)
        current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        position_error = float(np.linalg.norm(target - self.data.site_xpos[self.site_id]))
        angle_error = float(np.linalg.norm(_rotation_vector(target_rotation @ current_rotation.T)))
        return self.data.qpos[self.qpos_addresses].copy(), position_error, angle_error
