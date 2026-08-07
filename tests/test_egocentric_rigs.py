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
