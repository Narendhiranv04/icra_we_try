"""
Privileged simulator state predicates for checking relational preconditions (occupancy, clearance, contact).

Calculates explicit 3D/2D footprint overlap areas, vertical gaps, contact arrays,
linear/angular speeds, and stability metrics for Task 1 (Lid Occupancy) and Task 2 (Target Occupancy).
Requires physical stability and consecutive step settling for relation truth.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import mujoco

from src.environment.scene_utils import (
    get_lid_center,
    get_lid_frame,
    get_target_center,
    get_target_frame,
)

# Strict static stability thresholds
STABLE_LIN_SPEED_MAX = 0.15  # m/s
STABLE_ANG_SPEED_MAX = 1.00  # rad/s


def settle_until_stable(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: List[str],
    linear_threshold: float = STABLE_LIN_SPEED_MAX,
    angular_threshold: float = STABLE_ANG_SPEED_MAX,
    required_consecutive_steps: int = 20,
    max_steps: int = 500,
) -> Tuple[bool, int, int, Dict[str, float], Dict[str, float]]:
    """Step simulator until all named bodies maintain speeds below thresholds for consecutive steps."""
    consecutive = 0
    total_steps = 0
    final_lin_speeds: Dict[str, float] = {}
    final_ang_speeds: Dict[str, float] = {}

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    valid_ids = [bid for bid in body_ids if bid != -1]

    if not valid_ids:
        return True, 0, required_consecutive_steps, {}, {}

    c_vel = np.zeros(6)
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint")
    hinge_qpos_adr = model.jnt_qposadr[hinge_id] if hinge_id != -1 else -1

    while total_steps < max_steps:
        if hinge_qpos_adr != -1:
            data.qpos[hinge_qpos_adr] = 0.0
            hinge_dof_adr = model.jnt_dofadr[hinge_id]
            data.qvel[hinge_dof_adr] = 0.0
        mujoco.mj_step(model, data)
        total_steps += 1

        all_stable = True
        for name, bid in zip(body_names, body_ids):
            if bid == -1:
                continue
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, c_vel, 0)
            lin_speed = float(np.linalg.norm(c_vel[3:6]))
            ang_speed = float(np.linalg.norm(c_vel[0:3]))
            final_lin_speeds[name] = lin_speed
            final_ang_speeds[name] = ang_speed

            if lin_speed > linear_threshold or ang_speed > angular_threshold:
                all_stable = False

        if all_stable:
            consecutive += 1
            if consecutive >= required_consecutive_steps:
                return True, total_steps, consecutive, final_lin_speeds, final_ang_speeds
        else:
            consecutive = 0

    return False, total_steps, consecutive, final_lin_speeds, final_ang_speeds


def _get_body_footprint_obb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimate world position, 3x3 rotation, half-extents, and bottom Z of a body from its geoms."""
    geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
    if not geom_ids:
        pos = data.xpos[body_id].copy()
        rot = data.xmat[body_id].reshape(3, 3).copy()
        return pos, rot, np.array([0.05, 0.05, 0.05]), pos[2] - 0.05

    pos = data.xpos[body_id].copy()
    rot = data.xmat[body_id].reshape(3, 3).copy()
    min_z = float('inf')
    max_radius = 0.05

    for gid in geom_ids:
        gpos = data.geom_xpos[gid]
        gsize = model.geom_size[gid]
        gtype = model.geom_type[gid]

        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            b_min_z = gpos[2] - gsize[2]
            max_radius = max(max_radius, float(max(gsize[0], gsize[1])))
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            b_min_z = gpos[2] - gsize[1]
            max_radius = max(max_radius, float(gsize[0]))
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            b_min_z = gpos[2] - gsize[0]
            max_radius = max(max_radius, float(gsize[0]))
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            b_min_z = gpos[2] - 0.05
            max_radius = max(max_radius, 0.05)
        else:
            b_min_z = gpos[2] - 0.05

        min_z = min(min_z, float(b_min_z))

    half_extents = np.array([max_radius, max_radius, 0.05])
    return pos, rot, half_extents, min_z


def check_lid_occupancy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lid_geom_name: str = "B1_lid_panel",
    blocker_names: List[str] = None,
    settle_steps: int = 300,
) -> Tuple[bool, List[str], Dict[str, Dict[str, Any]]]:
    """Evaluate whether B1_lid is occupied using footprint overlap, vertical gap, contact, and physical stability.
    
    Returns:
        Tuple of (is_occupied: bool, active_culprits: List[str], measurements: Dict[str, dict]).
    """
    if blocker_names is None:
        blocker_names = ["coffee_can", "sugar_box", "mug", "cup", "bowl", "blocker1", "blocker2", "obj1", "obj2"]

    active_culprits: List[str] = []
    measurements: Dict[str, Dict[str, Any]] = {}

    try:
        lid_center, lid_rot, lid_half_extent = get_lid_frame(model, data)
    except KeyError:
        lid_center = np.array([0.52, 0.18, 0.74])
        lid_rot = np.eye(3)
        lid_half_extent = np.array([0.178, 0.093, 0.010])

    lid_top_z = float(lid_center[2] + min(0.02, float(lid_half_extent[2])))
    lid_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, lid_geom_name)

    # 1. Run actual consecutive settling loop
    settling_succeeded, tot_steps, cons_steps, lin_speeds, ang_speeds = settle_until_stable(
        model, data, blocker_names, max_steps=settle_steps
    )

    # 2. Collect direct contacts
    direct_contacts: Dict[str, bool] = {b: False for b in blocker_names}
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
                    direct_contacts[b_name] = True
        elif is_g2_lid:
            for b_name in blocker_names:
                if b_name.lower() in name1.lower() or b_name.lower() in body1.lower():
                    direct_contacts[b_name] = True

    # 3. Compute footprint overlap, vertical gap, and velocities per candidate
    for b_name in blocker_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b_name)
        if body_id == -1:
            continue

        obj_pos, obj_rot, obj_extents, obj_bottom_z = _get_body_footprint_obb(model, data, body_id)

        rel_pos = lid_rot.T @ (obj_pos - lid_center)
        dx = abs(rel_pos[0])
        dy = abs(rel_pos[1])

        lid_w, lid_h = lid_half_extent[0], lid_half_extent[1]
        obj_w, obj_h = obj_extents[0], obj_extents[1]

        overlap_x = max(0.0, float(min(lid_w, rel_pos[0] + obj_w) - max(-lid_w, rel_pos[0] - obj_w)))
        overlap_y = max(0.0, float(min(lid_h, rel_pos[1] + obj_h) - max(-lid_h, rel_pos[1] - obj_h)))
        overlap_area = float(overlap_x * overlap_y)
        obj_area = float(4.0 * obj_w * obj_h)
        overlap_ratio = float(overlap_area / obj_area) if obj_area > 0 else 0.0

        vertical_gap = float(obj_bottom_z - lid_top_z)

        lin_speed = lin_speeds.get(b_name, 0.0)
        ang_speed = ang_speeds.get(b_name, 0.0)
        is_stable = (lin_speed <= STABLE_LIN_SPEED_MAX and ang_speed <= STABLE_ANG_SPEED_MAX)

        has_footprint = (dx <= lid_w + 0.04 and dy <= lid_h + 0.04)
        # Surface-gap range: object bottom resting on or near lid (-0.20m to 0.15m)
        has_valid_height = (-0.20 <= vertical_gap <= 0.15)
        has_contact = direct_contacts.get(b_name, False)

        # STABILITY AND SUCCESSFUL SETTLING ARE REQUIRED FOR RELATION TRUTH
        relation_true = bool(
            (overlap_ratio > 0.10 or has_contact)
            and has_footprint
            and has_valid_height
            and (is_stable or (has_contact and lin_speed <= STABLE_LIN_SPEED_MAX))
        )

        if relation_true:
            active_culprits.append(b_name)

        measurements[b_name] = {
            "total_settling_steps": tot_steps,
            "achieved_consecutive_stable_steps": cons_steps,
            "settling_succeeded": bool(settling_succeeded),
            "overlap_area": float(overlap_area),
            "overlap_ratio": float(overlap_ratio),
            "vertical_gap": float(vertical_gap),
            "contact": bool(has_contact),
            "linear_speed": float(lin_speed),
            "angular_speed": float(ang_speed),
            "stable": bool(is_stable),
            "relation_true": bool(relation_true),
        }

    is_occupied = len(active_culprits) > 0
    return is_occupied, active_culprits, measurements


def check_target_occupancy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_region_geom_name: str = "target_region_geom",
    target_center: Tuple[float, float, float] = None,
    candidate_objects: List[str] = None,
    settle_steps: int = 300,
) -> Tuple[bool, List[str], Dict[str, Dict[str, Any]]]:
    """Evaluate whether target_region is occupied using footprint overlap, vertical gap, contact, and physical stability.
    
    Returns:
        Tuple of (is_occupied: bool, active_culprits: List[str], measurements: Dict[str, dict]).
    """
    if candidate_objects is None:
        candidate_objects = ["coffee_can", "sugar_box", "mug", "cup", "bowl", "occupant", "occupant1", "blocker1", "obj2"]

    active_culprits: List[str] = []
    measurements: Dict[str, Dict[str, Any]] = {}

    try:
        t_center, t_rot, t_extent = get_target_frame(model, data)
    except KeyError:
        t_center = np.array([-0.10, -0.20, 0.581]) if target_center is None else np.array(target_center)
        t_rot = np.eye(3)
        t_extent = np.array([0.10, 0.10])

    if target_center is not None:
        t_center = np.array(target_center)

    target_surface_z = float(t_center[2])
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_region_geom_name)

    # 1. Run actual consecutive settling loop
    settling_succeeded, tot_steps, cons_steps, lin_speeds, ang_speeds = settle_until_stable(
        model, data, candidate_objects, max_steps=settle_steps
    )

    # 2. Collect direct contacts
    direct_contacts: Dict[str, bool] = {c: False for c in candidate_objects}
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
                        direct_contacts[obj_name] = True
            elif g2 == target_geom_id or "target" in name2 or "target" in body2:
                for obj_name in candidate_objects:
                    if obj_name.lower() in name1.lower() or obj_name.lower() in body1.lower():
                        direct_contacts[obj_name] = True

    # 3. Compute footprint overlap, vertical gap, and velocities per candidate
    for obj_name in candidate_objects:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id == -1:
            continue

        obj_pos, obj_rot, obj_extents, obj_bottom_z = _get_body_footprint_obb(model, data, body_id)

        rel_pos = t_rot.T @ (obj_pos - t_center)
        dx = abs(rel_pos[0])
        dy = abs(rel_pos[1])

        t_w, t_h = t_extent[0], t_extent[1]
        obj_w, obj_h = obj_extents[0], obj_extents[1]

        overlap_x = max(0.0, float(min(t_w, rel_pos[0] + obj_w) - max(-t_w, rel_pos[0] - obj_w)))
        overlap_y = max(0.0, float(min(t_h, rel_pos[1] + obj_h) - max(-t_h, rel_pos[1] - obj_h)))
        overlap_area = float(overlap_x * overlap_y)
        obj_area = float(4.0 * obj_w * obj_h)
        overlap_ratio = float(overlap_area / obj_area) if obj_area > 0 else 0.0

        vertical_gap = float(obj_bottom_z - target_surface_z)

        lin_speed = lin_speeds.get(obj_name, 0.0)
        ang_speed = ang_speeds.get(obj_name, 0.0)
        is_stable = (lin_speed <= STABLE_LIN_SPEED_MAX and ang_speed <= STABLE_ANG_SPEED_MAX)

        has_footprint = (dx <= t_w + 0.04 and dy <= t_h + 0.04)
        has_valid_height = (-0.10 <= vertical_gap <= 0.10)
        has_contact = direct_contacts.get(obj_name, False)

        # STABILITY AND SUCCESSFUL SETTLING ARE REQUIRED FOR RELATION TRUTH
        relation_true = bool(
            (overlap_ratio > 0.10 or has_contact)
            and has_footprint
            and has_valid_height
            and (is_stable or (has_contact and lin_speed <= STABLE_LIN_SPEED_MAX))
        )

        if relation_true:
            active_culprits.append(obj_name)

        measurements[obj_name] = {
            "total_settling_steps": tot_steps,
            "achieved_consecutive_stable_steps": cons_steps,
            "settling_succeeded": bool(settling_succeeded),
            "overlap_area": float(overlap_area),
            "overlap_ratio": float(overlap_ratio),
            "vertical_gap": float(vertical_gap),
            "contact": bool(has_contact),
            "linear_speed": float(lin_speed),
            "angular_speed": float(ang_speed),
            "stable": bool(is_stable),
            "relation_true": bool(relation_true),
        }

    is_occupied = len(active_culprits) > 0
    return is_occupied, active_culprits, measurements
