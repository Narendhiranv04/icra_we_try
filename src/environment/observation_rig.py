"""
Observation rig abstractions and task-aware camera gaze utilities.
Provides task-specific egocentric camera rigs and matched virtual camera extrinsics.
"""

from dataclasses import dataclass, asdict
import math
from typing import Dict, Any, List, Tuple, Optional
import mujoco
import numpy as np


@dataclass
class ObservationRig:
    """Explicit task-specific egocentric observation rig configuration."""
    rig_id: str
    task_id: str
    robot_base_pose: str
    head_pan: float
    head_tilt: float
    camera_name: str
    camera_mount_transform: Dict[str, Any]
    look_at_target: List[float]
    fovy: float = 65.0
    resolution: Tuple[int, int] = (640, 480)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservationRig":
        return cls(**data)


# Predefined task observation rigs
TASK_1_RIG = ObservationRig(
    rig_id="task1_open_box_ego",
    task_id="task1",
    robot_base_pose="home",
    head_pan=-0.50,      # Aimed toward box center/lid (x=0.52, y=0.18) on right side
    head_tilt=0.50,      # Tilted down ~28 deg for optimal workspace FOV
    camera_name="robot0:ego_camera",
    camera_mount_transform={"pos": [0.0, 0.0, 0.0], "euler": "3.1415 0 0"},
    look_at_target=[0.52, 0.18, 0.65],
    fovy=65.0,
    resolution=(640, 480),
)

TASK_2_RIG = ObservationRig(
    rig_id="task2_place_object_ego",
    task_id="task2",
    robot_base_pose="home",
    head_pan=0.05,       # Aimed at midpoint between pick (-0.25) and target (0.15)
    head_tilt=0.75,      # Tilted down ~43 deg to view pick object and target region
    camera_name="robot0:ego_camera",
    camera_mount_transform={"pos": [0.0, 0.0, 0.0], "euler": "3.1415 0 0"},
    look_at_target=[-0.05, 0.0, 0.50],
    fovy=65.0,
    resolution=(640, 480),
)

CONTEXT_CHALLENGE_RIG = ObservationRig(
    rig_id="context_challenge_ego",
    task_id="context",
    robot_base_pose="home",
    head_pan=0.25,
    head_tilt=0.60,
    camera_name="robot0:ego_camera",
    camera_mount_transform={"pos": [0.0, 0.0, 0.0], "euler": "3.1415 0 0"},
    look_at_target=[0.25, 0.1, 0.5],
    fovy=70.0,
    resolution=(640, 480),
)


def get_task_observation_rig(task_id: str) -> ObservationRig:
    """Retrieve default task-aware observation rig for task1 or task2."""
    if "1" in str(task_id):
        return TASK_1_RIG
    elif "2" in str(task_id):
        return TASK_2_RIG
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def compute_task_gaze(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    robot_base_pose: str = "home",
) -> Tuple[float, float]:
    """Compute analytical head pan and head tilt angles to point camera directly at target_pos."""
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_pan_joint")
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_tilt_joint")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot0:ego_camera")

    if pan_id == -1 or tilt_id == -1 or cam_id == -1:
        return 0.0, 0.45

    # Reset head joints to zero to find base camera origin
    data.qpos[model.jnt_qposadr[pan_id]] = 0.0
    data.qpos[model.jnt_qposadr[tilt_id]] = 0.0
    mujoco.mj_forward(model, data)

    cam_pos = data.cam_xpos[cam_id].copy()
    diff = target_pos - cam_pos
    dx, dy, dz = diff[0], diff[1], diff[2]

    # Fetch head_pan rotates negative for positive X (right)
    pan = math.atan2(-dx, dy)
    dist_xy = math.hypot(dx, dy)
    tilt = math.atan2(-dz, dist_xy)

    pan = float(np.clip(pan, -1.50, 1.50))
    tilt = float(np.clip(tilt, 0.0, 1.20))
    return pan, tilt


def get_camera_extrinsic_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = "robot0:ego_camera",
) -> np.ndarray:
    """Return 4x4 homogenous matrix representing camera world extrinsic pose."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        return np.eye(4)

    pos = data.cam_xpos[cam_id]
    rot = data.cam_xmat[cam_id].reshape(3, 3)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = pos
    return extrinsic


def apply_observation_rig(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rig: ObservationRig,
) -> Dict[str, Any]:
    """Apply an ObservationRig to a scene's model and data state.

    Sets exact head_pan and head_tilt, calls mj_forward, and returns measured camera metadata.
    """
    from src.environment.robot_integration import initialize_robot_qpos
    from src.environment.renderer import OffscreenRenderer

    initialize_robot_qpos(model, data, head_pan=rig.head_pan, head_tilt=rig.head_tilt)
    mujoco.mj_forward(model, data)

    renderer = OffscreenRenderer(model, width=rig.resolution[0], height=rig.resolution[1], camera_name=rig.camera_name)
    metadata = renderer.get_camera_metadata(data)
    renderer.close()

    metadata["rig_id"] = rig.rig_id
    metadata["task_id"] = rig.task_id
    metadata["name"] = rig.camera_name
    metadata["camera_name"] = rig.camera_name
    metadata["robot_base_pose"] = rig.robot_base_pose
    metadata["head_pan"] = rig.head_pan
    metadata["head_tilt"] = rig.head_tilt
    metadata["camera_extrinsic_matrix"] = get_camera_extrinsic_matrix(model, data, rig.camera_name).tolist()
    return metadata

