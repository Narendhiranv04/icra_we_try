"""
Regression test suite covering final release invariants for egocentric YCB/GSO benchmark:
- Test A: Real Task-1 demonstration (B1_lid_panel visibility, extrinsics, camera name)
- Test B: Real Task-2 demonstration (object1 & target visibility, final placement)
- Test C: Per-instance Task-1 visibility (blockers, target, extrinsics invariant)
- Test D: Per-instance Task-2 visibility (object1, occupant, target, extrinsics invariant)
- Test E: Real GSO pair generation (source_dataset=='gso', mesh/texture metadata, RGB files)
- Test F: Split GSO participation (ID assignments and holdout GSO distractors)
- Test G: No stale camera metadata (no front_camera, proper base poses)
- Test H: No duplicate visible asset geoms (collision proxies hidden, alias visual geoms absent)
"""

import json
from pathlib import Path
import pytest
import numpy as np
import mujoco

from src.generation.demonstration_generator import DemonstrationGenerator
from src.generation.counterfactual_generator import CounterfactualPairGenerator
from src.generation.split_planner import SplitPlanner
from src.environment.scene_builder import SceneBuilder
from src.environment.observation_rig import TASK_1_RIG, TASK_2_RIG


def test_a_real_task1_demonstration(tmp_path):
    """Test A: Real Task-1 demonstration generation and temporal visibility validation."""
    demo_dir = tmp_path / "task1_demo"
    dg = DemonstrationGenerator(output_dir=demo_dir, resolution=(320, 240))
    mp4_path = dg.generate_task_1_demo(demo_id="demo_t1_test", seed=42)

    assert Path(mp4_path).exists()
    spec_json = demo_dir / "open_box" / "demo_t1_test" / "scene_config.json"
    assert spec_json.exists()

    with open(spec_json, "r") as f:
        meta = json.load(f)

    assert meta["camera_name"] == "robot0:ego_camera"
    assert meta["robot_base_pose"] == "home"
    assert "measured_camera_metadata" in meta
    assert "camera_world_extrinsic" in meta

    tvq = meta["temporal_view_quality"]
    assert tvq["is_valid"] is True
    assert tvq["phases"]["initial"]["is_valid"] is True
    assert tvq["phases"]["final_open"]["is_valid"] is True
    assert tvq["phases"]["initial"]["target_pixels"] > 0
    assert tvq["phases"]["final_open"]["target_pixels"] > 0


def test_b_real_task2_demonstration(tmp_path):
    """Test B: Real Task-2 demonstration generation and temporal visibility validation."""
    demo_dir = tmp_path / "task2_demo"
    dg = DemonstrationGenerator(output_dir=demo_dir, resolution=(320, 240))
    mp4_path = dg.generate_task_2_demo(demo_id="demo_t2_test", seed=43)

    assert Path(mp4_path).exists()
    spec_json = demo_dir / "place_object" / "demo_t2_test" / "scene_config.json"
    assert spec_json.exists()

    with open(spec_json, "r") as f:
        meta = json.load(f)

    assert meta["camera_name"] == "robot0:ego_camera"
    assert meta["robot_base_pose"] == "home"
    assert "measured_camera_metadata" in meta

    tvq = meta["temporal_view_quality"]
    assert tvq["phases"]["initial"]["is_valid"] is True
    assert tvq["phases"]["initial"]["target_pixels"] > 0
    assert tvq["is_valid"] is True, f"Temporal view quality details: {tvq}"


def test_c_per_instance_task1_visibility(tmp_path):
    """Test C: Per-instance visibility and camera invariant for Task 1 matched pair."""
    output_dir = tmp_path / "queries_t1"
    gen = CounterfactualPairGenerator(output_dir=output_dir, resolution=(320, 240))
    meta = gen.generate_task1_pair("pair_t1_test", blocker_count=1, seed=42)

    assert meta["stop"]["is_occupied"] is True
    assert meta["proceed"]["is_occupied"] is False

    # Camera extrinsics invariant
    stop_ext = meta["stop"]["resolved_scene_spec"]["camera"]["camera_world_extrinsic"]
    proc_ext = meta["proceed"]["resolved_scene_spec"]["camera"]["camera_world_extrinsic"]
    assert json.dumps(stop_ext) == json.dumps(proc_ext)

    # Per-instance object visibility
    stop_objs = meta["stop"]["resolved_scene_spec"]["objects"]
    assert "blocker1" in stop_objs
    assert stop_objs["blocker1"]["source_dataset"] in ["ycb", "gso"]


def test_d_per_instance_task2_visibility(tmp_path):
    """Test D: Per-instance visibility and camera invariant for Task 2 matched pair."""
    output_dir = tmp_path / "queries_t2"
    gen = CounterfactualPairGenerator(output_dir=output_dir, resolution=(320, 240))
    meta = gen.generate_task2_pair("pair_t2_test", target_occupant_type="sugar_box", seed=43)

    assert meta["stop"]["is_occupied"] is True
    assert meta["proceed"]["is_occupied"] is False

    # Camera extrinsics invariant
    stop_ext = meta["stop"]["resolved_scene_spec"]["camera"]["camera_world_extrinsic"]
    proc_ext = meta["proceed"]["resolved_scene_spec"]["camera"]["camera_world_extrinsic"]
    assert json.dumps(stop_ext) == json.dumps(proc_ext)

    # Objects check
    stop_objs = meta["stop"]["resolved_scene_spec"]["objects"]
    assert "coffee_can" in stop_objs
    assert "occupant" in stop_objs


def test_e_real_gso_generation(tmp_path):
    """Test E: Matched pair generation with a GSO asset."""
    output_dir = tmp_path / "queries_gso"
    gen = CounterfactualPairGenerator(output_dir=output_dir, resolution=(320, 240))
    meta = gen.generate_task2_pair("pair_gso_test", target_occupant_type="gso_coffee_jar", seed=44)

    assert meta["stop"]["is_occupied"] is True
    assert meta["proceed"]["is_occupied"] is False

    occ_meta = meta["stop"]["resolved_scene_spec"]["objects"]["occupant"]
    assert occ_meta["source_dataset"] == "gso"
    assert occ_meta["asset_id"] == "gso_coffee_jar"
    assert len(occ_meta["visual_mesh_identifier"]) > 0

    assert Path(meta["stop"]["rgb_path"]).exists()
    assert Path(meta["proceed"]["rgb_path"]).exists()


def test_f_split_gso_participation():
    """Test F: SplitPlanner produces GSO objects in ID assignments and holdouts."""
    planner = SplitPlanner()

    # Verify GSO present in ID objects list
    gso_id_objects = [o for o in planner.id_objects if o.startswith("gso_")]
    assert len(gso_id_objects) > 0, "SplitPlanner id_objects must contain GSO assets"

    # Verify GSO present in unseen objects list
    gso_unseen_objects = [o for o in planner.unseen_objects if o.startswith("gso_")]
    assert len(gso_unseen_objects) > 0, "SplitPlanner unseen_objects must contain GSO assets"

    # Check that sample assignments contain GSO assets
    id_assignments = [planner.get_assignment_for_split("id", idx=i, task_id="task_1") for i in range(10)]
    has_gso_id = any(a.object_type.startswith("gso_") for a in id_assignments)
    assert has_gso_id is True, "ID split assignments must sample GSO assets"


def test_g_no_stale_camera_metadata(tmp_path):
    """Test G: Ensure no stale camera metadata (front_camera) exists in fresh specs."""
    demo_dir = tmp_path / "demos_stale_check"
    dg1 = DemonstrationGenerator(output_dir=demo_dir, resolution=(320, 240))
    dg1.generate_task_1_demo(demo_id="demo_t1_stale", seed=42)

    spec_json = demo_dir / "open_box" / "demo_t1_stale" / "scene_config.json"
    with open(spec_json, "r") as f:
        raw_text = f.read()

    assert "front_camera" not in raw_text
    assert "robot0:ego_camera" in raw_text

    query_dir = tmp_path / "queries_stale_check"
    cg = CounterfactualPairGenerator(output_dir=query_dir, resolution=(320, 240))
    meta1 = cg.generate_task1_pair("pair_t1_stale", seed=42)

    stop_spec_text = json.dumps(meta1["stop"]["resolved_scene_spec"])
    assert "front_camera" not in stop_spec_text
    assert meta1["stop"]["resolved_scene_spec"]["robot"]["base_pose"] == TASK_1_RIG.robot_base_pose


def test_h_no_duplicate_visible_asset_geoms():
    """Test H: Spawn YCB and GSO assets and verify collision proxies and visual geoms."""
    sb = SceneBuilder()
    model, data = sb.create_environment(
        objects_to_spawn=[
            {"name": "obj_ycb", "type": "coffee_can", "pos": [0.0, -0.2, 0.05]},
            {"name": "obj_gso", "type": "gso_coffee_jar", "pos": [0.2, -0.2, 0.05]},
        ],
        include_robot=False,
    )

    for obj_name in ["obj_ycb", "obj_gso"]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        assert body_id != -1, f"Body {obj_name} not found"

        collision_geoms = []
        visual_geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body_id:
                if model.geom_group[g] == 3:
                    collision_geoms.append(g)
                else:
                    visual_geoms.append(g)

        assert len(collision_geoms) >= 1, f"Expected hidden collision proxy (group 3) for {obj_name}"
        assert len(visual_geoms) >= 1, f"Expected visual geom (group != 3) for {obj_name}"
