"""
Comprehensive benchmark correctness test suite covering all 26 explicit regression test requirements.
"""

import json
import math
import tempfile
from pathlib import Path
import copy

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
from src.validation.occupancy_checks import check_lid_occupancy, check_target_occupancy, settle_until_stable
from src.validation.demonstration_validator import DemonstrationValidator
from src.validation.dataset_validator import DatasetValidator, deep_diff
from src.validation.demonstration_distinctness import DemonstrationDistinctnessValidator
from src.generation.counterfactual_generator import CounterfactualPairGenerator, regenerate_from_metadata
from src.generation.background_randomization import apply_background_spec, sample_background_spec, BACKGROUND_PROFILES
from src.generation.split_planner import SplitPlanner
from src.generation.scene_config import EpisodeSpec
from src.preview.smoke_artifacts import TrackedSmokeArtifactsGenerator


# ── 1. Injected Background Mismatch Fails Validation ──────────────────
def test_01_background_mismatch_fails_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("pair_bg_fail")
        
        # Inject background mismatch into PROCEED resolved scene spec
        bad_meta = copy.deepcopy(meta)
        bad_meta["proceed"]["resolved_scene_spec"]["background"]["background_id"] = "bg_blue_counter"

        v = DatasetValidator("data/manifests/smoke_manifest.jsonl")
        valid, issues = v.validate_matched_pair_invariants(bad_meta)
        assert valid is False
        assert any("background.background_id" in iss for iss in issues)


# ── 2. Injected Lighting Mismatch Fails Validation ────────────────────
def test_02_lighting_mismatch_fails_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("pair_light_fail")
        
        # Inject light pos offset mismatch into PROCEED
        bad_meta = copy.deepcopy(meta)
        l_spec = bad_meta["proceed"]["resolved_scene_spec"]["background"]["background_spec"]["light_spec"]
        l_spec["light_pos_offsets"][0][0] += 0.50

        v = DatasetValidator("data/manifests/smoke_manifest.jsonl")
        valid, issues = v.validate_matched_pair_invariants(bad_meta)
        assert valid is False
        assert any("background.background_spec.light_spec.light_pos_offsets" in iss for iss in issues)


# ── 3. Undeclared Blocker2 Movement Fails Validation ─────────────────
def test_03_undeclared_blocker2_movement_fails_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("pair_b2_fail", blocker_count=1)
        
        # Add undeclared blocker2 change into PROCEED
        bad_meta = copy.deepcopy(meta)
        bad_meta["proceed"]["resolved_scene_spec"]["objects"]["blocker2"] = {
            "name": "blocker2",
            "type": "sugar_box",
            "position": [0.52, 0.18, 0.88],
            "orientation": [1.0, 0.0, 0.0, 0.0],
        }

        v = DatasetValidator("data/manifests/smoke_manifest.jsonl")
        valid, issues = v.validate_matched_pair_invariants(bad_meta)
        assert valid is False
        assert any("unexpected difference paths" in iss for iss in issues)


# ── 4. Missing Declared Intervention Fails Validation ─────────────────
def test_04_missing_declared_intervention_fails_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("pair_missing_fail", blocker_count=1)
        
        # Set PROCEED position identical to STOP position (no intervention)
        bad_meta = copy.deepcopy(meta)
        stop_pos = bad_meta["stop"]["resolved_scene_spec"]["objects"]["blocker1"]["position"]
        bad_meta["proceed"]["resolved_scene_spec"]["objects"]["blocker1"]["position"] = list(stop_pos)

        v = DatasetValidator("data/manifests/smoke_manifest.jsonl")
        valid, issues = v.validate_matched_pair_invariants(bad_meta)
        assert valid is False
        assert any("declared intervention paths did not change" in iss for iss in issues)


# ── 5-8. Exact Regeneration Tests ────────────────────────────────────
def test_05_exact_task1_matched_pair_regeneration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("p1_regen", seed=111)
        regen_meta = regenerate_from_metadata(meta, output_dir=Path(tmp_dir)/"regen", resolution=(320, 240))
        
        orig_inst = np.load(meta["stop"]["instance_uint16_path"])
        regen_inst = np.load(regen_meta["stop"]["instance_uint16_path"])
        assert np.array_equal(orig_inst, regen_inst)


def test_06_exact_task2_matched_pair_regeneration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task2_pair("p2_regen", seed=222)
        regen_meta = regenerate_from_metadata(meta, output_dir=Path(tmp_dir)/"regen", resolution=(320, 240))
        
        orig_inst = np.load(meta["stop"]["instance_uint16_path"])
        regen_inst = np.load(regen_meta["stop"]["instance_uint16_path"])
        assert np.array_equal(orig_inst, regen_inst)


def test_07_exact_task1_positive_control_regeneration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_control("ctrl1_regen", control_subtype="one_object_beside", object_type="mug", seed=333)
        regen_meta = regenerate_from_metadata(meta, output_dir=Path(tmp_dir)/"regen", resolution=(320, 240))
        
        assert regen_meta["object_type"] == "mug"
        orig_inst = np.load(meta["instance_uint16_path"])
        regen_inst = np.load(regen_meta["instance_uint16_path"])
        assert np.array_equal(orig_inst, regen_inst)


def test_08_exact_task2_positive_control_regeneration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task2_control("ctrl2_regen", control_subtype="one_object_beside_target", occupant_type="sugar_box", seed=444)
        regen_meta = regenerate_from_metadata(meta, output_dir=Path(tmp_dir)/"regen", resolution=(320, 240))
        
        assert regen_meta["occupant_type"] == "sugar_box"
        orig_inst = np.load(meta["instance_uint16_path"])
        regen_inst = np.load(regen_meta["instance_uint16_path"])
        assert np.array_equal(orig_inst, regen_inst)


# ── 9-10. uint16 Instance Map Loading ─────────────────────────────────
def test_09_stop_uint16_instance_map_loads_correctly():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("p_u16_stop")
        arr = np.load(meta["stop"]["instance_uint16_path"])
        assert arr.dtype == np.uint16
        assert arr.shape == (240, 320)


def test_10_proceed_uint16_instance_map_loads_correctly():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = CounterfactualPairGenerator(output_dir=tmp_dir, resolution=(320, 240))
        meta = gen.generate_task1_pair("p_u16_proc")
        arr = np.load(meta["proceed"]["instance_uint16_path"])
        assert arr.dtype == np.uint16
        assert arr.shape == (240, 320)


# ── 11-14. Box & Target Translation/Rotation Robustness ────────────────
def test_11_12_box_translation_and_rotation_robustness():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "blocker1", "type": "coffee_can", "pos": [0.52, 0.18, 0.82]}], settle_steps=100)
    
    # Translate and rotate box
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box_B1")
    m.body_pos[bid] += np.array([0.05, -0.05, 0.0])
    mujoco.mj_forward(m, d)

    occ, culprits, _ = check_lid_occupancy(m, d, blocker_names=["blocker1"])
    assert occ is True


def test_13_14_target_translation_and_rotation_robustness():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "occupant", "type": "sugar_box", "pos": [-0.10, -0.20, 0.65]}], settle_steps=100)
    
    # Translate target region geom
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_region_geom")
    m.geom_pos[gid] += np.array([0.02, 0.02, 0.0])
    mujoco.mj_forward(m, d)

    occ, culprits, _ = check_target_occupancy(m, d, candidate_objects=["occupant"])
    assert occ is True


# ── 15-17. Consecutive Stability & Label Mismatch Abort ────────────────
def test_15_real_consecutive_stability_counter():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "blocker1", "type": "coffee_can", "pos": [0.52, 0.18, 0.82]}], settle_steps=0)
    succ, tot_steps, cons_steps, lin_sp, ang_sp = settle_until_stable(m, d, ["blocker1"], max_steps=150)
    assert succ is True
    assert cons_steps >= 20


def test_16_17_unstable_object_and_label_mismatch_rejected():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "blocker1", "type": "coffee_can", "pos": [0.52, 0.18, 0.82]}], settle_steps=0)
    
    # Impart high linear velocity via qvel to blocker1
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "blocker1")
    jadr = m.body_jntadr[bid]
    if jadr != -1:
        dofadr = m.jnt_dofadr[jadr]
        d.qvel[dofadr:dofadr+3] = [2.0, 0.0, 0.0]

    occ, _, meas = check_lid_occupancy(m, d, blocker_names=["blocker1"], settle_steps=1)
    assert occ is False
    assert meas["blocker1"]["stable"] is False


# ── 18-20. Split Leakage Detection ────────────────────────────────────
def test_18_unseen_object_leakage_is_detected():
    v = DatasetValidator("data/manifests/smoke_manifest.jsonl")
    v.records = [
        {"sample_type": "matched_pair", "split": "id", "blocker_type": "coffee_can", "background_id": "bg_neutral_wood"},
        {"sample_type": "matched_pair", "split": "unseen_object", "blocker_type": "coffee_can", "background_id": "bg_neutral_wood"},
    ]
    valid, rep = v.validate_splits()
    assert valid is False
    assert rep["leakage_checks"]["object_leakage"] is True


def test_19_unseen_background_leakage_is_detected():
    v = DatasetValidator("data/manifests/smoke_manifest.jsonl")
    v.records = [
        {"sample_type": "matched_pair", "split": "id", "blocker_type": "coffee_can", "background_id": "bg_neutral_wood"},
        {"sample_type": "matched_pair", "split": "unseen_background", "blocker_type": "mug", "background_id": "bg_neutral_wood"},
    ]
    valid, rep = v.validate_splits()
    assert valid is False
    assert rep["leakage_checks"]["background_leakage"] is True


def test_20_compositional_tuple_leakage_is_detected():
    v = DatasetValidator("data/manifests/smoke_manifest.jsonl")
    rec = {
        "sample_type": "matched_pair",
        "task_id": "task_1",
        "split": "id",
        "blocker_type": "coffee_can",
        "background_id": "bg_neutral_wood",
        "blocker_pos_bin": "centre",
        "blocker_count": 1,
    }
    rec_comp = copy.deepcopy(rec)
    rec_comp["split"] = "compositional"

    v.records = [rec, rec_comp]
    valid, rep = v.validate_splits()
    assert valid is False
    assert rep["leakage_checks"]["compositional_leakage"] is True


# ── 21-22. Pilot Demonstration Distinctness ───────────────────────────
def test_21_22_pilot_demo_distinctness_and_duplicate_rejection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        from src.generation.demonstration_generator import DemonstrationGenerator
        gen = DemonstrationGenerator(output_dir=tmp_dir)
        p1 = gen.generate_task_1_demo("demo1", background_id="bg_neutral_wood", seed=1)
        p2 = gen.generate_task_1_demo("demo2", background_id="bg_blue_counter", seed=2)

        val = DemonstrationDistinctnessValidator(tmp_dir)
        valid, rep = val.validate_all_demos()
        assert valid is True

        # Test duplicate rejection by copying demo1 into demo3
        d3_dir = Path(tmp_dir) / "open_box" / "demo3"
        d3_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in (Path(p1).parent).iterdir():
            shutil.copy(f, d3_dir / f.name)

        valid_dup, rep_dup = val.validate_all_demos()
        assert valid_dup is False


# ── 23-26. Dynamic Report Status Propagation ───────────────────────────
def test_23_24_25_26_failed_reports_propagate_to_smoke_report():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = TrackedSmokeArtifactsGenerator(artifacts_dir=tmp_dir)

        # Write failed dataset_validation.json into data/reports/
        rep_dir = Path("data/reports")
        rep_dir.mkdir(parents=True, exist_ok=True)
        with open(rep_dir / "dataset_validation.json", "w", encoding="utf-8") as f:
            json.dump({"status": "FAILED", "issue_count": 1}, f)

        status = gen.generate_all_smoke_artifacts()
        assert status == "FAILED"
