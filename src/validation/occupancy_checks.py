"""
Privileged simulator state predicates for checking relational preconditions (occupancy, clearance, contact).
"""

from typing import Dict, List, Tuple
import numpy as np
import mujoco


def check_lid_occupancy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lid_geom_name: str = "B1_lid_panel",
    blocker_names: List[str] = None,
) -> Tuple[bool, List[str]]:
    """Evaluate whether B1_lid is occupied by any blocker object using local frame & contact predicates.
    
    Args:
        model: MuJoCo MjModel instance.
        data: MuJoCo MjData instance.
        lid_geom_name: Name of lid surface geom.
        blocker_names: List of candidate blocker body/geom names.
        
    Returns:
        Tuple of (is_occupied: bool, active_culprits: List[str]).
    """
    if blocker_names is None:
        blocker_names = ["coffee_can", "sugar_box", "mug", "cup", "bowl", "blocker1", "blocker2", "obj1", "obj2"]

    lid_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, lid_geom_name)
    lid_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B1_lid")
    active_culprits = []

    if lid_geom_id == -1:
        for i in range(model.ngeom):
            g_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if g_name and "lid" in g_name.lower():
                lid_geom_id = i
                break

    # 1. Check direct contacts in MjData contact array
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

    # 2. Local-frame footprint & height check
    if lid_geom_id != -1:
        lid_center = data.geom_xpos[lid_geom_id]
        lid_rot = data.geom_xmat[lid_geom_id].reshape(3, 3)
        lid_half_extent = model.geom_size[lid_geom_id]
    elif lid_body_id != -1:
        lid_center = data.xpos[lid_body_id]
        lid_rot = data.xmat[lid_body_id].reshape(3, 3)
        lid_half_extent = np.array([0.178, 0.093, 0.010])
    else:
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
            # Overlap threshold relative to lid surface extents
            if dx <= lid_half_extent[0] + 0.04 and dy <= lid_half_extent[1] + 0.04 and -0.02 <= dz <= 0.35:
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
    """Evaluate whether single-capacity target_region is occupied using explicit local frame & contact.
    
    Args:
        model: MuJoCo MjModel instance.
        data: MuJoCo MjData instance.
        target_region_geom_name: Name of target region surface geom.
        target_center: Optional override (x, y, z) center coordinate of target region.
        radius: Optional footprint radius tolerance.
        candidate_objects: List of candidate occupant objects to check.
        
    Returns:
        Tuple of (is_occupied: bool, active_culprits: List[str]).
    """
    if candidate_objects is None:
        candidate_objects = ["coffee_can", "sugar_box", "mug", "cup", "bowl", "occupant1", "blocker1", "obj2"]

    active_culprits = []
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_region_geom_name)
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_region_body")

    if target_geom_id != -1:
        t_center = data.geom_xpos[target_geom_id]
        t_rot = data.geom_xmat[target_geom_id].reshape(3, 3)
        t_extent = model.geom_size[target_geom_id][:2]
    elif target_body_id != -1:
        t_center = data.xpos[target_body_id]
        t_rot = data.xmat[target_body_id].reshape(3, 3)
        t_extent = np.array([0.10, 0.10])
    else:
        t_center = np.array([-0.10, -0.20, 0.581]) if target_center is None else np.array(target_center)
        t_rot = np.eye(3)
        t_extent = np.array([0.10, 0.10])

    if target_center is not None:
        t_center = np.array(target_center)

    # 1. Contact check
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

    # 2. Local-frame footprint check
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
            # Single-capacity region footprint check
            if dx <= t_extent[0] + 0.04 and dy <= t_extent[1] + 0.04 and -0.02 <= dz <= 0.35:
                active_culprits.append(obj_name)

    is_occupied = len(active_culprits) > 0
    return is_occupied, active_culprits

