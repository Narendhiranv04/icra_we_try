"""
Privileged simulator state predicates for checking relational preconditions (occupancy, clearance, contact).
"""

from typing import Dict, List, Tuple
import numpy as np
import mujoco

from src.environment.scene_utils import (
    get_lid_center,
    get_lid_frame,
    get_target_center,
    get_target_frame,
)


def check_lid_occupancy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lid_geom_name: str = "B1_lid_panel",
    blocker_names: List[str] = None,
) -> Tuple[bool, List[str]]:
    """Evaluate whether B1_lid is occupied by any blocker object using local frame & contact predicates."""
    if blocker_names is None:
        blocker_names = ["coffee_can", "sugar_box", "mug", "cup", "bowl", "blocker1", "blocker2", "obj1", "obj2"]

    active_culprits = []

    # 1. Contact check
    lid_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, lid_geom_name)

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1]) or ""
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2]) or ""

        is_g1_lid = (g1 == lid_geom_id or "lid" in name1.lower() or "lid" in body1.lower())
        is_g2_lid = (g2 == lid_geom_id or "lid" in name2.lower() or "lid" in body2.lower())

        if is_g1_lid:
            for b_name in blocker_names:
                if b_name.lower() in name2.lower() or b_name.lower() in body2.lower():
                    if b_name not in active_culprits:
                        active_culprits.append(b_name)
        elif is_g2_lid:
            for b_name in blocker_names:
                if b_name.lower() in name1.lower() or b_name.lower() in body1.lower():
                    if b_name not in active_culprits:
                        active_culprits.append(b_name)

    # 2. Dynamic scene_utils local-frame footprint check
    try:
        lid_center, lid_rot, lid_half_extent = get_lid_frame(model, data)
    except KeyError:
        lid_center = np.array([0.52, 0.18, 0.74])
        lid_rot = np.eye(3)
        lid_half_extent = np.array([0.178, 0.093, 0.010])

    for b_name in blocker_names:
        if b_name in active_culprits:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b_name)
        if body_id != -1:
            obj_pos = data.xpos[body_id]
            rel_pos = lid_rot.T @ (obj_pos - lid_center)
            dx = abs(rel_pos[0])
            dy = abs(rel_pos[1])
            dz = rel_pos[2]
            if dx <= lid_half_extent[0] + 0.03 and dy <= lid_half_extent[1] + 0.03 and -0.02 <= dz <= 0.35:
                active_culprits.append(b_name)

    is_occupied = len(active_culprits) > 0
    return is_occupied, active_culprits


def check_target_occupancy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_region_geom_name: str = "target_region_geom",
    target_center: Tuple[float, float, float] = None,
    radius: float = None,
    candidate_objects: List[str] = None,
) -> Tuple[bool, List[str]]:
    """Evaluate whether single-capacity target_region is occupied using explicit local frame & contact."""
    if candidate_objects is None:
        candidate_objects = ["coffee_can", "sugar_box", "mug", "cup", "bowl", "occupant", "occupant1", "blocker1", "obj2"]

    active_culprits = []

    # 1. Contact check
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_region_geom_name)
    if target_geom_id != -1:
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = con.geom1, con.geom2
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1]) or ""
            body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2]) or ""

            if g1 == target_geom_id or "target" in name1 or "target" in body1:
                for obj_name in candidate_objects:
                    if obj_name.lower() in name2.lower() or obj_name.lower() in body2.lower():
                        if obj_name not in active_culprits:
                            active_culprits.append(obj_name)
            elif g2 == target_geom_id or "target" in name2 or "target" in body2:
                for obj_name in candidate_objects:
                    if obj_name.lower() in name1.lower() or obj_name.lower() in body1.lower():
                        if obj_name not in active_culprits:
                            active_culprits.append(obj_name)

    # 2. Dynamic scene_utils local-frame footprint check
    try:
        t_center, t_rot, t_extent = get_target_frame(model, data)
    except KeyError:
        t_center = np.array([-0.10, -0.20, 0.581]) if target_center is None else np.array(target_center)
        t_rot = np.eye(3)
        t_extent = np.array([0.10, 0.10])

    if target_center is not None:
        t_center = np.array(target_center)

    for obj_name in candidate_objects:
        if obj_name in active_culprits:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id != -1:
            obj_pos = data.xpos[body_id]
            rel_pos = t_rot.T @ (obj_pos - t_center)
            dx = abs(rel_pos[0])
            dy = abs(rel_pos[1])
            dz = rel_pos[2]
            if dx <= t_extent[0] + 0.03 and dy <= t_extent[1] + 0.03 and -0.02 <= dz <= 0.35:
                active_culprits.append(obj_name)

    is_occupied = len(active_culprits) > 0
    return is_occupied, active_culprits
