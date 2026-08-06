"""
Single source of truth for scene geometry queries and local frame transforms.

All task logic, occupancy checks, counterfactual generators, and demonstration
executors derive spatial positions from these utilities rather than
duplicating world-coordinate constants.
"""

from __future__ import annotations

from typing import Tuple, List

import mujoco
import numpy as np


def get_body_world_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Return world position of a named body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        raise KeyError(f"Body '{name}' not found in model")
    return data.xpos[bid].copy()


def get_body_world_mat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Return 3x3 world rotation matrix of a named body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        raise KeyError(f"Body '{name}' not found in model")
    return data.xmat[bid].reshape(3, 3).copy()


def get_geom_world_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Return world position of a named geom."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid == -1:
        raise KeyError(f"Geom '{name}' not found in model")
    return data.geom_xpos[gid].copy()


def get_geom_world_mat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Return 3x3 world rotation matrix of a named geom."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid == -1:
        raise KeyError(f"Geom '{name}' not found in model")
    return data.geom_xmat[gid].reshape(3, 3).copy()


def get_geom_half_extents(model: mujoco.MjModel, name: str) -> np.ndarray:
    """Return the size array of a named geom (half-extents for box, etc.)."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid == -1:
        raise KeyError(f"Geom '{name}' not found in model")
    return model.geom_size[gid].copy()


def get_site_world_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Return world position of a named site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid == -1:
        raise KeyError(f"Site '{name}' not found in model")
    return data.site_xpos[sid].copy()


def get_site_world_mat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Return 3x3 world rotation matrix of a named site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid == -1:
        raise KeyError(f"Site '{name}' not found in model")
    return data.site_xmat[sid].reshape(3, 3).copy()


def world_to_local(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_or_geom_name: str,
    world_pos: np.ndarray,
    *,
    use_geom: bool = False,
) -> np.ndarray:
    """Transform a world position into a body's or geom's local frame."""
    if use_geom:
        center = get_geom_world_pos(model, data, body_or_geom_name)
        rot = get_geom_world_mat(model, data, body_or_geom_name)
    else:
        center = get_body_world_pos(model, data, body_or_geom_name)
        rot = get_body_world_mat(model, data, body_or_geom_name)
    return rot.T @ (np.asarray(world_pos) - center)


def local_to_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_or_geom_name: str,
    local_pos: np.ndarray,
    *,
    use_geom: bool = False,
) -> np.ndarray:
    """Transform a local position back to world frame."""
    if use_geom:
        center = get_geom_world_pos(model, data, body_or_geom_name)
        rot = get_geom_world_mat(model, data, body_or_geom_name)
    else:
        center = get_body_world_pos(model, data, body_or_geom_name)
        rot = get_body_world_mat(model, data, body_or_geom_name)
    return center + rot @ np.asarray(local_pos)


# ── Task 1 helpers ──────────────────────────────────────────────────────

def get_lid_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World position of the B1 lid panel centre, read from the geom."""
    mujoco.mj_forward(model, data)
    return get_geom_world_pos(model, data, "B1_lid_panel")


def get_lid_frame(
    model: mujoco.MjModel, data: mujoco.MjData
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (center, rotation_3x3, half_extents) of the lid panel."""
    mujoco.mj_forward(model, data)
    center = get_geom_world_pos(model, data, "B1_lid_panel")
    rot = get_geom_world_mat(model, data, "B1_lid_panel")
    ext = get_geom_half_extents(model, "B1_lid_panel")
    return center, rot, ext


def get_handle_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World position of the B1 lid handle grasp site."""
    mujoco.mj_forward(model, data)
    return get_site_world_pos(model, data, "B1_lid_handle_grasp")


def get_hinge_axis(model: mujoco.MjModel) -> np.ndarray:
    """Hinge axis of B1_lid_joint in the joint's parent-body frame."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint")
    if jid == -1:
        raise KeyError("B1_lid_joint not found")
    return model.jnt_axis[jid].copy()


def get_hinge_qpos_adr(model: mujoco.MjModel) -> int:
    """Return qpos address of B1_lid_joint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint")
    if jid == -1:
        raise KeyError("B1_lid_joint not found")
    return int(model.jnt_qposadr[jid])


def get_lid_angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Current hinge angle of the B1 lid, in radians."""
    return float(data.qpos[get_hinge_qpos_adr(model)])


def get_box_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World position of the box_B1 body."""
    mujoco.mj_forward(model, data)
    return get_body_world_pos(model, data, "box_B1")


# ── Task 2 helpers ──────────────────────────────────────────────────────

def get_target_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World position of the target_region centre, read from geom."""
    mujoco.mj_forward(model, data)
    return get_geom_world_pos(model, data, "target_region_geom")


def get_target_frame(
    model: mujoco.MjModel, data: mujoco.MjData
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (center, rotation_3x3, half_extents_xy) of the target region."""
    mujoco.mj_forward(model, data)
    center = get_geom_world_pos(model, data, "target_region_geom")
    rot = get_geom_world_mat(model, data, "target_region_geom")
    ext = get_geom_half_extents(model, "target_region_geom")
    return center, rot, ext


def sample_position_on_lid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    x_frac: float = 0.0,
    y_frac: float = 0.0,
    height_above: float = 0.06,
) -> np.ndarray:
    """Sample a world position on the lid surface using the lid's local frame transform."""
    mujoco.mj_forward(model, data)
    center, rot, ext = get_lid_frame(model, data)
    local = np.array([x_frac * ext[0] * 0.7, y_frac * ext[1] * 0.7, ext[2] + height_above])
    return center + rot @ local


def sample_position_beside_box(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    offset_x: float = -0.30,
    offset_y: float = -0.15,
    height_above_table: float = 0.04,
) -> np.ndarray:
    """Sample a world position beside the box (off the lid) in the box's local frame."""
    mujoco.mj_forward(model, data)
    center = get_box_pos(model, data)
    rot = get_body_world_mat(model, data, "box_B1")
    local = np.array([offset_x, offset_y, height_above_table])
    return center + rot @ local


def sample_position_in_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    x_frac: float = 0.0,
    y_frac: float = 0.0,
    height_above: float = 0.07,
) -> np.ndarray:
    """Sample a world position inside the target region using the target's local frame transform."""
    mujoco.mj_forward(model, data)
    center, rot, ext = get_target_frame(model, data)
    local = np.array([x_frac * ext[0] * 0.5, y_frac * ext[1] * 0.5, ext[2] + height_above])
    return center + rot @ local


def sample_position_outside_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    offset_x: float = 0.30,
    offset_y: float = 0.0,
    height_above: float = 0.07,
) -> np.ndarray:
    """Sample a world position outside the target region in the target's local frame."""
    mujoco.mj_forward(model, data)
    center, rot, _ = get_target_frame(model, data)
    local = np.array([offset_x, offset_y, height_above])
    return center + rot @ local
