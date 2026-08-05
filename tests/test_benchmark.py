"""
Comprehensive benchmark correctness test suite.

Tests all physical invariants, demonstration executors, occupancy predicates,
counterfactual pair symmetry, background profiles, and dataset validation.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import mujoco

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.environment.scene_utils import (
    get_lid_center,
    get_lid_frame,
    get_target_center,
    get_target_frame,
    get_handle_pos,
)
from src.environment.robot_integration import VerticalIK, TOP_DOWN_ROTATION, HOME_ARM_SEED
from src.tasks.open_box import BoxOpenExecutor
from src.tasks.place_object import PlaceObjectExecutor
from src.validation.occupancy_checks import check_lid_occupancy, check_target_occupancy
from src.validation.demonstration_validator import DemonstrationValidator
from src.generation.counterfactual_generator import CounterfactualPairGenerator
from src.generation.background_randomization import apply_background_profile, BACKGROUND_PROFILES
from src.generation.split_planner import SplitPlanner


# ── 1. Scene & Geometry Tests ─────────────────────────────────────────

def test_scene_utils_geometry_queries():
    sb = SceneBuilder()
    model, data = sb.create_environment(settle_steps=0)
    
    lid_center = get_lid_center(model, data)
    assert len(lid_center) == 3
    assert lid_center[2] > 0.5

    center, rot, ext = get_lid_frame(model, data)
    assert rot.shape == (3, 3)
    assert ext[0] > 0 and ext[1] > 0

    t_center = get_target_center(model, data)
    assert len(t_center) == 3
    assert t_center[2] > 0.5

    tc, trot, text = get_target_frame(model, data)
    assert text[0] > 0 and text[1] > 0


def test_robot_base_poses():
    sb = SceneBuilder()
    
    # Test 'home' pose
    m_home, d_home = sb.create_environment(include_robot=True, robot_base_pose="home")
    base_id = mujoco.mj_name2id(m_home, mujoco.mjtObj.mjOBJ_BODY, "robot0:base_link")
    assert base_id != -1
    assert abs(d_home.xpos[base_id][1] - (-0.95)) < 0.05

    # Test 'right_side' pose
    m_right, d_right = sb.create_environment(include_robot=True, robot_base_pose="right_side")
    base_id_r = mujoco.mj_name2id(m_right, mujoco.mjtObj.mjOBJ_BODY, "robot0:base_link")
    assert base_id_r != -1
    assert abs(d_right.xpos[base_id_r][0] - 1.025) < 0.05


# ── 2. IK & Reachability Tests ─────────────────────────────────────────

def test_ik_solver_reachability():
    sb = SceneBuilder()
    model, data = sb.create_environment(include_robot=True, robot_base_pose="home")
    ik = VerticalIK(model, data)
    
    target = np.array([-0.25, -0.30, 0.72])
    qpos, pos_err, ang_err = ik.solve(target, HOME_ARM_SEED, target_rotation=TOP_DOWN_ROTATION)
    
    assert pos_err < 0.01, f"IK position error too high: {pos_err:.4f}m"
    assert ang_err < 0.05, f"IK angle error too high: {ang_err:.4f}rad"


# ── 3. Occupancy Predicate Tests ───────────────────────────────────────

def test_lid_occupancy_stop_vs_proceed():
    sb = SceneBuilder()
    lid_center = [0.52, 0.18, 0.82]

    # STOP scene: blocker on top of lid
    stop_objs = [{"name": "blocker1", "type": "coffee_can", "pos": lid_center}]
    m_stop, d_stop = sb.create_environment(stop_objs, settle_steps=20)
    is_occ_stop, culprits_stop = check_lid_occupancy(m_stop, d_stop, blocker_names=["blocker1"])
    assert is_occ_stop is True
    assert "blocker1" in culprits_stop

    # PROCEED scene: blocker beside box
    proceed_objs = [{"name": "blocker1", "type": "coffee_can", "pos": [0.18, 0.18, 0.62]}]
    m_proc, d_proc = sb.create_environment(proceed_objs, settle_steps=20)
    is_occ_proc, culprits_proc = check_lid_occupancy(m_proc, d_proc, blocker_names=["blocker1"])
    assert is_occ_proc is False
    assert len(culprits_proc) == 0


def test_target_occupancy_stop_vs_proceed():
    sb = SceneBuilder()
    target_pos = [-0.10, -0.20, 0.65]

    # STOP scene: occupant inside target region
    stop_objs = [{"name": "occupant", "type": "sugar_box", "pos": target_pos}]
    m_stop, d_stop = sb.create_environment(stop_objs, settle_steps=20)
    is_occ_stop, culprits_stop = check_target_occupancy(m_stop, d_stop, candidate_objects=["occupant"])
    assert is_occ_stop is True
    assert "occupant" in culprits_stop

    # PROCEED scene: occupant outside target region
    proceed_objs = [{"name": "occupant", "type": "sugar_box", "pos": [0.25, -0.20, 0.65]}]
    m_proc, d_proc = sb.create_environment(proceed_objs, settle_steps=20)
    is_occ_proc, culprits_proc = check_target_occupancy(m_proc, d_proc, candidate_objects=["occupant"])
    assert is_occ_proc is False
    assert len(culprits_proc) == 0


# ── 4. Demonstration Executor Tests ────────────────────────────────────

def test_task1_open_box_demonstration():
    sb = SceneBuilder()
    model, data = sb.create_environment(include_robot=True, robot_base_pose="right_side")
    renderer = OffscreenRenderer(model, width=320, height=240)
    
    executor = BoxOpenExecutor(model, data)
    frames = executor.run_demonstration(renderer)
    renderer.close()

    assert len(frames) == 120
    is_valid, issues = DemonstrationValidator.validate_open_box(executor.state_log)
    assert is_valid is True, f"Open box validation failed: {issues}"


def test_task2_place_object_demonstration():
    sb = SceneBuilder()
    start_pos = (-0.30, -0.20, 0.65)
    target_pos = (-0.10, -0.20, 0.65)
    objects = [{"name": "coffee_can", "type": "coffee_can", "pos": list(start_pos)}]
    model, data = sb.create_environment(objects, include_robot=True, robot_base_pose="home", weld_target_body="coffee_can")
    renderer = OffscreenRenderer(model, width=320, height=240)

    executor = PlaceObjectExecutor(model, data, object_name="coffee_can", target_pos=target_pos)
    frames = executor.run_demonstration(renderer, start_pos=start_pos)
    renderer.close()

    assert len(frames) == 120
    is_valid, issues = DemonstrationValidator.validate_place_object(executor.state_log)
    assert is_valid is True, f"Place object validation failed: {issues}"


# ── 5. Background Profiles & Randomization Tests ────────────────────────

def test_background_profiles_application():
    sb = SceneBuilder()
    model, data = sb.create_environment()

    for name in BACKGROUND_PROFILES:
        apply_background_profile(model, profile_name=name)
        mujoco.mj_forward(model, data)
        # Ensure model runs without error
        assert model.nmat > 0


def test_split_planner():
    planner = SplitPlanner()
    
    id_assign = planner.get_assignment_for_split("id", 0)
    assert id_assign.background_id == "bg_neutral_wood"

    bg_assign = planner.get_assignment_for_split("unseen_background", 0)
    assert bg_assign.background_id == "bg_blue_counter"

    comp_assign = planner.get_assignment_for_split("compositional", 0)
    assert comp_assign.background_id == "bg_granite_dark"


# ── 6. Counterfactual Pair Generator & Masks Tests ──────────────────────

def test_counterfactual_pair_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))

        # Task 1 Pair
        meta1 = gen.generate_task1_pair("test_pair_t1", blocker_type="coffee_can")
        assert meta1["stop"]["is_occupied"] is True
        assert meta1["proceed"]["is_occupied"] is False
        assert Path(meta1["stop"]["rgb_path"]).exists()
        assert Path(meta1["stop"]["instance_uint16_path"]).exists()
        assert Path(meta1["proceed"]["rgb_path"]).exists()
        assert "instance_id_to_name_map" in meta1["stop"]

        # Task 2 Pair
        meta2 = gen.generate_task2_pair("test_pair_t2", target_occupant_type="sugar_box")
        assert meta2["stop"]["is_occupied"] is True
        assert meta2["proceed"]["is_occupied"] is False
        assert Path(meta2["stop"]["rgb_path"]).exists()
        assert Path(meta2["stop"]["instance_uint16_path"]).exists()
        assert Path(meta2["proceed"]["rgb_path"]).exists()
        assert "instance_id_to_name_map" in meta2["stop"]
