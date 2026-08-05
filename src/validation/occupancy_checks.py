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
    """Evaluate whether B1_lid is occupied by any blocker object.
    
    Args:
        model: MuJoCo MjModel instance.
        data: MuJoCo MjData instance.
        lid_geom_name: Name of lid surface geom.
        blocker_names: List of candidate blocker body/geom names.
        
    Returns:
        Tuple of (is_occupied: bool, active_culprits: List[str]).
    """
    if blocker_names is None:
        blocker_names = ["coffee_can", "sugar_box", "mug", "cup", "bowl"]

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

    # 2. Position fallback check: Z-height & XY footprint relative to lid center
    if not active_culprits:
        lid_pos = np.array([0.52, 0.18, 0.74]) # Lid center pos
        if lid_body_id != -1:
            lid_pos = data.xpos[lid_body_id]
        elif lid_geom_id != -1:
            lid_pos = data.geom_xpos[lid_geom_id]

        for b_name in blocker_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b_name)
            if body_id != -1:
                obj_pos = data.xpos[body_id]
                dx = abs(obj_pos[0] - lid_pos[0])
                dy = abs(obj_pos[1] - lid_pos[1])
                dz = obj_pos[2] - lid_pos[2]
                if dx < 0.22 and dy < 0.22 and -0.05 <= dz <= 0.30:
                    if b_name not in active_culprits:
                        active_culprits.append(b_name)

    is_occupied = len(active_culprits) > 0
    return is_occupied, active_culprits


def check_target_occupancy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_region_geom_name: str = "target_region_geom",
    target_center: Tuple[float, float, float] = (0.0, 0.20, 0.57),
    radius: float = 0.18,
    candidate_objects: List[str] = None,
) -> Tuple[bool, List[str]]:
    """Evaluate whether target_region is occupied by an unwanted object.
    
    Args:
        model: MuJoCo MjModel instance.
        data: MuJoCo MjData instance.
        target_region_geom_name: Name of target region surface geom.
        target_center: (x, y, z) center coordinate of target region.
        radius: Radius boundary of target region.
        candidate_objects: List of objects to check for target region occupancy.
        
    Returns:
        Tuple of (is_occupied: bool, active_culprits: List[str]).
    """
    if candidate_objects is None:
        candidate_objects = ["coffee_can", "sugar_box", "mug", "cup", "bowl"]

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

    # 2. Geometric distance check
    tc = np.array(target_center[:2])
    for obj_name in candidate_objects:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id != -1:
            obj_pos = data.xpos[body_id]
            dist = np.linalg.norm(obj_pos[:2] - tc)
            if dist <= radius and abs(obj_pos[2] - target_center[2]) < 0.25:
                if obj_name not in active_culprits:
                    active_culprits.append(obj_name)

    is_occupied = len(active_culprits) > 0
    return is_occupied, active_culprits
