"""
Mandatory unit tests for egocentric observation rigs, camera extrinsics matching, and view quality.
"""

from pathlib import Path
import json
import numpy as np
import mujoco

from src.environment.observation_rig import TASK_1_RIG, TASK_2_RIG, get_task_observation_rig
from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.environment.robot_integration import initialize_robot_qpos


def test_01_task1_demo_and_query_share_rig_id():
    rig = get_task_observation_rig("task1")
    assert rig.rig_id == "task1_open_box_ego"
    assert rig.robot_base_pose == "home"


def test_02_task1_demo_query_camera_extrinsics_match():
    sb = SceneBuilder()
    m1, d1 = sb.create_environment(include_robot=True, robot_base_pose=TASK_1_RIG.robot_base_pose)
    initialize_robot_qpos(m1, d1, head_pan=TASK_1_RIG.head_pan, head_tilt=TASK_1_RIG.head_tilt)

    m2, d2 = sb.create_environment(include_robot=True, robot_base_pose=TASK_1_RIG.robot_base_pose)
    initialize_robot_qpos(m2, d2, head_pan=TASK_1_RIG.head_pan, head_tilt=TASK_1_RIG.head_tilt)

    r1 = OffscreenRenderer(m1, camera_name="robot0:ego_camera")
    r2 = OffscreenRenderer(m2, camera_name="robot0:ego_camera")

    e1 = r1.get_camera_metadata(d1)["camera_world_extrinsic"]["position"]
    e2 = r2.get_camera_metadata(d2)["camera_world_extrinsic"]["position"]

    assert np.allclose(e1, e2, atol=1e-3)
    r1.close()
    r2.close()


def test_03_task2_demo_query_camera_extrinsics_match():
    sb = SceneBuilder()
    m1, d1 = sb.create_environment(include_robot=True, robot_base_pose=TASK_2_RIG.robot_base_pose)
    initialize_robot_qpos(m1, d1, head_pan=TASK_2_RIG.head_pan, head_tilt=TASK_2_RIG.head_tilt)

    m2, d2 = sb.create_environment(include_robot=True, robot_base_pose=TASK_2_RIG.robot_base_pose)
    initialize_robot_qpos(m2, d2, head_pan=TASK_2_RIG.head_pan, head_tilt=TASK_2_RIG.head_tilt)

    r1 = OffscreenRenderer(m1, camera_name="robot0:ego_camera")
    r2 = OffscreenRenderer(m2, camera_name="robot0:ego_camera")

    e1 = r1.get_camera_metadata(d1)["camera_world_extrinsic"]["position"]
    e2 = r2.get_camera_metadata(d2)["camera_world_extrinsic"]["position"]

    assert np.allclose(e1, e2, atol=1e-3)
    r1.close()
    r2.close()


def test_04_task1_lid_handle_visible():
    sb = SceneBuilder()
    m, d = sb.create_environment(include_robot=True, robot_base_pose=TASK_1_RIG.robot_base_pose)
    initialize_robot_qpos(m, d, head_pan=TASK_1_RIG.head_pan, head_tilt=TASK_1_RIG.head_tilt)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    val = ren.validate_view_quality(d, target_geom_names=["B1_lid_panel"], candidate_geom_names=[])
    assert val["target_visible"] is True
    ren.close()


def test_05_task2_pick_and_target_visible():
    sb = SceneBuilder()
    objs = [{"name": "coffee_can", "type": "coffee_can", "pos": [-0.25, 0.0, 0.75]}]
    m, d = sb.create_environment(objs, include_robot=True, robot_base_pose=TASK_2_RIG.robot_base_pose)
    initialize_robot_qpos(m, d, head_pan=TASK_2_RIG.head_pan, head_tilt=TASK_2_RIG.head_tilt)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    from src.environment.scene_builder import get_instance_visual_geom_names
    cand_names = get_instance_visual_geom_names(m, "coffee_can")
    val = ren.validate_view_quality(d, target_geom_names=["target_region_geom"], candidate_geom_names=cand_names)
    assert val["target_visible"] is True
    assert val["candidate_visible"] is True
    ren.close()


def test_06_camera_target_near_centre():
    sb = SceneBuilder()
    m, d = sb.create_environment(include_robot=True, robot_base_pose=TASK_1_RIG.robot_base_pose)
    initialize_robot_qpos(m, d, head_pan=TASK_1_RIG.head_pan, head_tilt=TASK_1_RIG.head_tilt)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    seg = ren.render_segmentation(d)[:, :, 0]
    lid_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "B1_lid_panel")
    ys, xs = np.nonzero(seg == lid_gid)
    assert len(xs) > 0
    mean_x = float(np.mean(xs))
    # Horizontal center should be within middle half of width (160 to 480 px)
    assert 120 <= mean_x <= 520
    ren.close()


def test_07_torso_fraction_below_threshold():
    sb = SceneBuilder()
    m, d = sb.create_environment(include_robot=True, robot_base_pose=TASK_1_RIG.robot_base_pose)
    initialize_robot_qpos(m, d, head_pan=TASK_1_RIG.head_pan, head_tilt=TASK_1_RIG.head_tilt)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    val = ren.validate_view_quality(d, target_geom_names=["B1_lid_panel"], candidate_geom_names=[])
    assert val["torso_acceptable"] is True
    assert val["torso_fraction"] <= 0.25
    ren.close()


def test_08_bad_pantilt_fails_validation():
    sb = SceneBuilder()
    m, d = sb.create_environment(include_robot=True, robot_base_pose="home")
    # Set extreme pan/tilt pointing away into the ceiling
    initialize_robot_qpos(m, d, head_pan=-1.5, head_tilt=-0.7)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    val = ren.validate_view_quality(d, target_geom_names=["B1_lid_panel"], candidate_geom_names=[])
    assert val["target_visible"] is False
    assert val["is_valid"] is False
    ren.close()


def test_09_camera_calibration_files_exist():
    assert Path("camera_calibration/task1_grid.png").is_file()
    assert Path("camera_calibration/task2_grid.png").is_file()
    assert Path("camera_calibration/task1_best.png").is_file()
    assert Path("camera_calibration/task2_best.png").is_file()
    assert Path("camera_calibration/results.json").is_file()


def test_10_production_task1_matched_pair(tmp_path):
    from src.generation.counterfactual_generator import CounterfactualPairGenerator, check_camera_extrinsics_invariant
    gen = CounterfactualPairGenerator(output_dir=tmp_path, resolution=(640, 480))
    meta = gen.generate_task1_pair("test_pair_t1", seed=42)

    stop_cam = meta["stop"]["camera"]
    proc_cam = meta["proceed"]["camera"]
    check_camera_extrinsics_invariant(stop_cam, proc_cam, TASK_1_RIG)

    with open(tmp_path / "test_pair_t1" / "metadata.json", "r") as f:
        disk_meta = json.load(f)

    assert disk_meta["camera_name"] == "robot0:ego_camera"
    assert disk_meta["camera_configuration"] == "robot0:ego_camera"
    assert disk_meta["stop"]["resolved_scene_spec"]["robot"]["base_pose"] == "home"


def test_11_production_task2_matched_pair(tmp_path):
    from src.generation.counterfactual_generator import CounterfactualPairGenerator, check_camera_extrinsics_invariant
    gen = CounterfactualPairGenerator(output_dir=tmp_path, resolution=(640, 480))
    meta = gen.generate_task2_pair("test_pair_t2", seed=42)

    stop_cam = meta["stop"]["camera"]
    proc_cam = meta["proceed"]["camera"]
    check_camera_extrinsics_invariant(stop_cam, proc_cam, TASK_2_RIG)

    with open(tmp_path / "test_pair_t2" / "metadata.json", "r") as f:
        disk_meta = json.load(f)

    assert disk_meta["camera_name"] == "robot0:ego_camera"
    assert disk_meta["stop"]["resolved_scene_spec"]["robot"]["base_pose"] == "home"


def test_12_positive_controls_record_actual_camera_params(tmp_path):
    from src.generation.counterfactual_generator import CounterfactualPairGenerator
    gen = CounterfactualPairGenerator(output_dir=tmp_path, resolution=(640, 480))

    meta1 = gen.generate_task1_control("ctrl_t1", seed=42)
    meta2 = gen.generate_task2_control("ctrl_t2", seed=42)

    assert meta1["camera"]["name"] == "robot0:ego_camera"
    assert "camera_world_extrinsic" in meta1["camera"]
    assert meta1["camera"]["robot_base_pose"] == "home"

    assert meta2["camera"]["name"] == "robot0:ego_camera"
    assert "camera_world_extrinsic" in meta2["camera"]
    assert meta2["camera"]["robot_base_pose"] == "home"


def test_13_metadata_regression_no_front_camera(tmp_path):
    from src.generation.counterfactual_generator import CounterfactualPairGenerator
    gen = CounterfactualPairGenerator(output_dir=tmp_path, resolution=(640, 480))
    gen.generate_task1_pair("pair_t1_no_front", seed=42)

    meta_file = tmp_path / "pair_t1_no_front" / "metadata.json"
    content = meta_file.read_text(encoding="utf-8")
    assert "front_camera" not in content


def test_14_camera_invariant_failure_raises_value_error(tmp_path):
    import pytest
    from src.generation.counterfactual_generator import check_camera_extrinsics_invariant
    from copy import deepcopy

    sb = SceneBuilder()
    m, d = sb.create_environment(include_robot=True, robot_base_pose="home")
    initialize_robot_qpos(m, d, head_pan=TASK_1_RIG.head_pan, head_tilt=TASK_1_RIG.head_tilt)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    meta_stop = ren.get_camera_metadata(d)
    ren.close()

    meta_stop["rig_id"] = TASK_1_RIG.rig_id
    meta_stop["task_id"] = TASK_1_RIG.task_id
    meta_stop["robot_base_pose"] = TASK_1_RIG.robot_base_pose
    meta_stop["head_pan"] = TASK_1_RIG.head_pan
    meta_stop["head_tilt"] = TASK_1_RIG.head_tilt

    meta_proc = deepcopy(meta_stop)
    # Intentionally mutate position by 5 cm
    meta_proc["camera_world_extrinsic"]["position"][0] += 0.05

    with pytest.raises(ValueError, match="Camera world position mismatch"):
        check_camera_extrinsics_invariant(meta_stop, meta_proc, TASK_1_RIG)


def test_15_view_quality_production_validation(tmp_path):
    from src.generation.counterfactual_generator import CounterfactualPairGenerator
    gen = CounterfactualPairGenerator(output_dir=tmp_path, resolution=(640, 480))

    meta1 = gen.generate_task1_pair("vq_pair_t1", seed=42)
    meta2 = gen.generate_task2_pair("vq_pair_t2", seed=42)

    assert meta1["stop"]["is_occupied"] is True
    assert meta1["proceed"]["is_occupied"] is False
    assert meta2["stop"]["is_occupied"] is True
    assert meta2["proceed"]["is_occupied"] is False

