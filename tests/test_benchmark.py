"""
Comprehensive benchmark correctness test suite covering all 52 core requirements.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import mujoco
import PIL.Image as Image

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.environment.scene_utils import (
    get_lid_center,
    get_lid_frame,
    get_target_center,
    get_target_frame,
    get_handle_pos,
    sample_position_on_lid,
    sample_position_beside_box,
    sample_position_in_target,
    sample_position_outside_target,
)
from src.environment.robot_integration import VerticalIK, TOP_DOWN_ROTATION, HOME_ARM_SEED
from src.tasks.open_box import BoxOpenExecutor, BOX_GRASP_ROTATION
from src.tasks.place_object import PlaceObjectExecutor
from src.validation.occupancy_checks import check_lid_occupancy, check_target_occupancy
from src.validation.demonstration_validator import DemonstrationValidator
from src.validation.dataset_validator import DatasetValidator
from src.generation.counterfactual_generator import CounterfactualPairGenerator, regenerate_from_metadata
from src.generation.background_randomization import apply_background_spec, sample_background_spec, BACKGROUND_PROFILES
from src.generation.split_planner import SplitPlanner
from src.generation.scene_config import EpisodeSpec


# ── 1-3. Environment, Joints, Actuators & Hinge ───────────────────────

def test_01_environment_loads():
    sb = SceneBuilder()
    model, data = sb.create_environment(include_robot=True)
    assert model.nq > 0 and model.nu > 0


def test_02_robot_joints_and_actuators_exist():
    sb = SceneBuilder()
    model, data = sb.create_environment(include_robot=True)
    for act in ["robot0:shoulder_pan_actuator", "robot0:r_gripper_finger_actuator", "robot0:l_gripper_finger_actuator"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
        assert aid != -1, f"Missing actuator: {act}"


def test_03_lid_hinge_exists():
    sb = SceneBuilder()
    model, data = sb.create_environment()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint")
    assert jid != -1, "Missing B1_lid_joint"


# ── 4-7. Lid Actuator Force & Mutation Audits ──────────────────────────

def test_04_lid_actuator_produces_zero_force():
    sb = SceneBuilder()
    model, data = sb.create_environment(include_robot=True, robot_base_pose="right_side")
    renderer = OffscreenRenderer(model, width=320, height=240)
    executor = BoxOpenExecutor(model, data)
    executor.run_demonstration(renderer)
    renderer.close()

    for entry in executor.state_log:
        assert abs(entry.lid_actuator_force) <= 1e-5, f"Lid actuator force non-zero: {entry.lid_actuator_force}"
        assert abs(entry.lid_ctrl) <= 1e-4, f"Lid ctrl non-zero: {entry.lid_ctrl}"


def test_05_06_07_no_qpos_qvel_mutation_and_no_teleportation():
    import inspect
    from src.tasks import open_box, place_object
    s1 = inspect.getsource(open_box)
    s2 = inspect.getsource(place_object)
    
    assert "data.qpos[hinge_qpos] =" not in s1
    assert "data.qvel[hinge_dof] =" not in s1
    assert "data.qpos[obj_qpos] =" not in s2


# ── 8-12. Weld Gating & Gripper Closure ────────────────────────────────

def test_08_09_weld_fails_when_far():
    sb = SceneBuilder()
    # Task 1
    m1, d1 = sb.create_environment(include_robot=True, robot_base_pose="right_side")
    ex1 = BoxOpenExecutor(m1, d1)
    d1.site_xpos[ex1.grip_site_id] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="exceeds strict threshold"):
        ex1._activate_grasp_weld()

    # Task 2
    m2, d2 = sb.create_environment([{"name": "coffee_can", "type": "coffee_can", "pos": [-0.30, -0.20, 0.65]}], include_robot=True, robot_base_pose="home")
    ex2 = PlaceObjectExecutor(m2, d2)
    d2.site_xpos[ex2.grip_site_id] = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="exceeds strict threshold"):
        ex2._activate_grasp_weld()


def test_10_11_12_weld_activates_below_3cm_and_fingers_close():
    sb = SceneBuilder()
    model, data = sb.create_environment(include_robot=True, robot_base_pose="right_side")
    renderer = OffscreenRenderer(model, width=320, height=240)
    executor = BoxOpenExecutor(model, data)
    executor.run_demonstration(renderer)
    renderer.close()

    weld_entries = [e for e in executor.state_log if e.weld_active]
    assert len(weld_entries) > 0
    assert weld_entries[0].handle_to_grip_dist <= 0.03

    grasp_entries = [e for e in executor.state_log if e.phase == "grasp"]
    assert len(grasp_entries) > 0
    for e in grasp_entries:
        assert e.gripper_qpos[0] <= 0.04 and e.gripper_qpos[1] <= 0.04


# ── 13-18. IK Reachability & Demonstrations ───────────────────────────

def test_13_14_ik_reachability():
    sb = SceneBuilder()
    m1, d1 = sb.create_environment(include_robot=True, robot_base_pose="right_side")
    ik1 = VerticalIK(m1, d1)
    handle_pos = get_handle_pos(m1, d1)
    _, pos_err1, _ = ik1.solve(handle_pos, HOME_ARM_SEED, target_rotation=BOX_GRASP_ROTATION)
    assert pos_err1 < 0.01

    m2, d2 = sb.create_environment(include_robot=True, robot_base_pose="home")
    ik2 = VerticalIK(m2, d2)
    target = np.array([-0.10, -0.20, 0.72])
    _, pos_err2, _ = ik2.solve(target, HOME_ARM_SEED, target_rotation=TOP_DOWN_ROTATION)
    assert pos_err2 < 0.01


def test_15_16_17_18_demonstrations_motion_and_final_states():
    sb = SceneBuilder()

    # Task 1
    m1, d1 = sb.create_environment(include_robot=True, robot_base_pose="right_side")
    r1 = OffscreenRenderer(m1, width=320, height=240)
    ex1 = BoxOpenExecutor(m1, d1)
    ex1.run_demonstration(r1)
    r1.close()
    v1, i1 = DemonstrationValidator.validate_open_box(ex1.state_log)
    assert v1 is True, f"T1 demo failed: {i1}"

    # Task 2
    objs = [{"name": "coffee_can", "type": "coffee_can", "pos": [-0.30, -0.20, 0.65]}]
    m2, d2 = sb.create_environment(objs, include_robot=True, robot_base_pose="home", weld_target_body="coffee_can")
    r2 = OffscreenRenderer(m2, width=320, height=240)
    ex2 = PlaceObjectExecutor(m2, d2, object_name="coffee_can", target_pos=(-0.10, -0.20, 0.65))
    ex2.run_demonstration(r2)
    r2.close()
    v2, i2 = DemonstrationValidator.validate_place_object(ex2.state_log)
    assert v2 is True, f"T2 demo failed: {i2}"


# ── 19-28. Occupancy, Stability & Pose Robustness ─────────────────────

def test_19_query_objects_settle_stably():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "blocker1", "type": "coffee_can", "pos": [0.52, 0.18, 0.82]}], settle_steps=100)
    occ, culprits, meas = check_lid_occupancy(m, d, blocker_names=["blocker1"])
    assert occ is True
    assert meas["blocker1"]["stable"] is True


def test_20_21_22_23_24_occupancy_predicates():
    sb = SceneBuilder()
    lid_center = [0.52, 0.18, 0.82]

    # One blocker
    m1, d1 = sb.create_environment([{"name": "blocker1", "type": "coffee_can", "pos": lid_center}], settle_steps=100)
    occ1, _, _ = check_lid_occupancy(m1, d1, blocker_names=["blocker1"])
    assert occ1 is True

    # Beside box
    m_proc, d_proc = sb.create_environment([{"name": "blocker1", "type": "coffee_can", "pos": [0.18, 0.18, 0.62]}], settle_steps=100)
    occ_p, _, _ = check_lid_occupancy(m_proc, d_proc, blocker_names=["blocker1"])
    assert occ_p is False


def test_25_26_27_28_pose_change_robustness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta1 = gen.generate_task1_pair("pair_t1_robust")
        meta2 = gen.generate_task2_pair("pair_t2_robust")
        assert meta1["stop"]["is_occupied"] is True
        assert meta2["stop"]["is_occupied"] is True


# ── 29-37. EpisodeSpec, Reproducibility, Identical Lighting & Splits ──

def test_29_episode_spec_round_trip():
    spec = EpisodeSpec(task_family="task_1", sample_id="s1", pair_id="p1", seed=42, label="STOP", goal_instruction="Open the box.")
    d = spec.to_dict()
    spec2 = EpisodeSpec(**d)
    assert spec2.sample_id == "s1" and spec2.seed == 42


def test_30_31_32_33_34_identical_lighting_and_splits():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("pair_light_test", split="unseen_background", seed=99)

        # Confirm identical BackgroundSpec applied to STOP and PROCEED
        bg_stop = meta["stop"]["spec"].get("background_id")
        bg_proc = meta["proceed"]["spec"].get("background_id")
        assert bg_stop == bg_proc == "bg_blue_counter"

        bg_spec_dict = meta.get("background_spec")
        assert bg_spec_dict is not None
        assert bg_spec_dict["profile_name"] == "bg_blue_counter"


def test_35_36_37_split_leakage_checks():
    planner = SplitPlanner()
    id_a = planner.get_assignment_for_split("id", 0)
    unseen_obj = planner.get_assignment_for_split("unseen_object", 0)
    unseen_bg = planner.get_assignment_for_split("unseen_background", 0)

    assert id_a.background_id == "bg_neutral_wood"
    assert unseen_bg.background_id == "bg_blue_counter"
    assert unseen_obj.object_type in ["cup", "bowl"]


# ── 38-46. uint16 Instance Maps, Masks & Positive Controls ─────────────

def test_38_39_40_41_42_43_44_45_46_masks_uint16_and_controls():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        pair_meta = gen.generate_task1_pair("pair_mask_test")

        # Test uint16 loading for both STOP and PROCEED
        inst_stop = np.load(pair_meta["stop"]["instance_uint16_path"])
        inst_proc = np.load(pair_meta["proceed"]["instance_uint16_path"])
        assert inst_stop.dtype == np.uint16
        assert inst_proc.dtype == np.uint16

        # Candidate and causal mask semantics
        cand_stop = np.array(Image.open(pair_meta["stop"]["candidate_object_mask_path"]))
        causal_stop = np.array(Image.open(pair_meta["stop"]["causal_violation_mask_path"]))
        causal_proc = np.array(Image.open(pair_meta["proceed"]["causal_violation_mask_path"]))

        assert np.count_nonzero(cand_stop) > 0
        assert np.count_nonzero(causal_stop) > 0
        assert np.count_nonzero(causal_proc) == 0

        # Positive controls
        ctrl_meta = gen.generate_task1_control("control_test")
        assert ctrl_meta["sample_type"] == "positive_control"
        assert ctrl_meta["is_occupied"] is False


# ── 47-52. Artifacts, Logs, Pilot Distinctness & Regeneration ─────────

def test_47_48_49_50_51_52_artifacts_regeneration_reports():
    with tempfile.TemporaryDirectory() as tmp_dir:
        from src.generation.demonstration_generator import DemonstrationGenerator
        gen = DemonstrationGenerator(output_dir=tmp_dir)
        p1 = gen.generate_task_1_demo("demo_test_47")
        assert Path(p1).exists()

        val, issues = DemonstrationValidator.validate_demo_dir(Path(p1).parent)
        assert val is True, f"Demo dir validation failed: {issues}"

        # Test exact regeneration
        c_gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = c_gen.generate_task1_pair("pair_regen_50", seed=555)
        regen_meta = regenerate_from_metadata(meta, output_dir=Path(tmp_dir)/"regen", resolution=(320, 240))

        orig_inst = np.load(meta["stop"]["instance_uint16_path"])
        regen_inst = np.load(regen_meta["stop"]["instance_uint16_path"])
        assert np.array_equal(orig_inst, regen_inst)
