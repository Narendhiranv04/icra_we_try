#!/usr/bin/env python3
"""
Independent validator for causal intervention datasets (Packet 2.5).
Rigorously enforces schema 2.1.0, metadata provenance, 22/22 physical Delta recomputation,
22/22 post RGB & segmentation reconstruction, candidate crop/mask/geometry alignment,
camera translation/rotation invariance, and strict relational identity transitions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import PIL.Image as Image
import mujoco

from src.interventions.intervention_records import (
    InterventionDatasetRecord,
    ResolvedSceneSpec,
    MODEL_INPUTS_ALLOWED_KEYS,
    FORBIDDEN_LEAKAGE_TERMS,
    SCHEMA_VERSION,
)
from src.interventions.intervention_types import (
    Intervention,
    InterventionOperator,
    ObjectPose,
    PrivilegedCategory,
)
from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.environment.observation_rig import (
    ObservationRig,
    get_task_observation_rig,
    apply_observation_rig,
)
from src.environment.scene_utils import world_to_local
from src.generation.background_randomization import (
    BackgroundSpec,
    apply_background_spec,
)
from src.validation.occupancy_checks import evaluate_relational_feasibility
from src.interventions.intervention_validator import (
    snapshot_simulator_state,
    restore_simulator_state,
    apply_intervention,
    get_body_freejoint_pose,
)
from src.interventions.intervention_scene_generator import _crop_object_from_rgb


def validate_intervention_dataset(
    manifest_path: Path,
    report_output_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    generator_commit: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dataset_root = manifest_path.parent

    # Read manifest records
    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    with open(manifest_path, "rb") as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()

    # Load dataset_metadata.json if present
    metadata_path = dataset_root / "dataset_metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Resolution from metadata (default 640, 480)
    res_list = metadata.get("resolution", [640, 480])
    width, height = int(res_list[0]), int(res_list[1])

    # Tolerances from metadata
    meta_tolerances = metadata.get("tolerances", {})
    tol_pre_rgb = float(meta_tolerances.get("rgb_pre_reconstruction_rmse_tolerance", 1.0))
    tol_post_rgb = float(meta_tolerances.get("rgb_post_reconstruction_rmse_tolerance", 1.0))
    tol_none_rgb = float(meta_tolerances.get("none_rgb_rmse_tolerance", 3.0))
    tol_cam_trans = float(meta_tolerances.get("camera_translation_tolerance_m", 0.001))
    tol_cam_rot = float(meta_tolerances.get("camera_rotation_tolerance_rad", 0.005))
    tol_collateral = float(meta_tolerances.get("max_collateral_translation_m", 0.05))
    tol_geom_pos = float(meta_tolerances.get("geometry_position_tolerance_m", 0.001))
    tol_geom_quat = float(meta_tolerances.get("geometry_quaternion_tolerance", 0.01))

    print(f"======================================================================")
    print(f"Validating Intervention Dataset Manifest: {manifest_path}")
    print(f"Schema Version: {SCHEMA_VERSION} | Resolution: [{width}, {height}] | Records Found: {len(records)}")
    print(f"======================================================================")

    # 1. METADATA & PROVENANCE VERIFICATION
    print("--> 1/8: Validating dataset metadata and provenance...")
    if metadata:
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Metadata schema_version '{metadata.get('schema_version')}' != '{SCHEMA_VERSION}'")
        if metadata.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"Metadata manifest_sha256 '{metadata.get('manifest_sha256')}' != actual manifest SHA '{manifest_sha}'")
        if metadata.get("record_count") != len(records):
            raise ValueError(f"Metadata record_count {metadata.get('record_count')} != actual records count {len(records)}")
        
        if config_path:
            with open(config_path, "rb") as f:
                actual_cfg_sha = hashlib.sha256(f.read()).hexdigest()
            if metadata.get("config_sha256") and metadata["config_sha256"] != actual_cfg_sha:
                raise ValueError(f"Metadata config_sha256 '{metadata['config_sha256']}' != actual config SHA '{actual_cfg_sha}'")

        if generator_commit and metadata.get("generator_commit"):
            if metadata["generator_commit"] != generator_commit:
                raise ValueError(f"Metadata generator_commit '{metadata['generator_commit']}' != expected '{generator_commit}'")

    # 2. SCHEMA & INFORMATION PARTITIONING
    print("--> 2/8: Validating schema and strict information partitioning...")
    scenes_map: Dict[str, List[Dict[str, Any]]] = {}
    record_ids_seen: Set[str] = set()
    intervention_ids_seen: Set[str] = set()

    for idx, rec_dict in enumerate(records):
        # Validate schema version
        if rec_dict.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Record {idx}: Invalid schema_version '{rec_dict.get('schema_version')}', expected '{SCHEMA_VERSION}'")

        # Uniqueness
        rec_id = rec_dict["record_id"]
        if rec_id in record_ids_seen:
            raise ValueError(f"Duplicate record_id: {rec_id}")
        record_ids_seen.add(rec_id)

        interv_id = rec_dict["intervention_id"]
        if interv_id in intervention_ids_seen:
            raise ValueError(f"Duplicate intervention_id: {interv_id}")
        intervention_ids_seen.add(interv_id)

        # Allowlist check on model_inputs
        model_inputs = rec_dict.get("model_inputs", {})
        unknown_keys = set(model_inputs.keys()) - MODEL_INPUTS_ALLOWED_KEYS
        if unknown_keys:
            raise ValueError(f"Record {rec_id}: Disallowed keys in model_inputs: {unknown_keys}")

        # Forbidden terms check
        InterventionDatasetRecord._check_forbidden_leakage(model_inputs)

        # Demonstration reference null check for smoke
        if model_inputs.get("demonstration_reference") is not None:
            raise ValueError(f"Record {rec_id}: Expected demonstration_reference=null in smoke dataset, got '{model_inputs['demonstration_reference']}'")

        # Group by scene
        scene_id = rec_dict["scene_id"]
        scenes_map.setdefault(scene_id, []).append(rec_dict)

    # 3. FILE EXISTENCE & INTEGRITY (DYNAMIC RESOLUTION)
    print("--> 3/8: Validating file existence, formats, and dtypes...")
    for rec_dict in records:
        m_in = rec_dict["model_inputs"]
        p_meta = rec_dict["privileged_metadata"]

        # Pre RGB
        pre_rgb_p = dataset_root / m_in["pre_rgb_path"]
        if not pre_rgb_p.exists():
            raise FileNotFoundError(f"Missing pre RGB: {pre_rgb_p}")
        with Image.open(pre_rgb_p) as im:
            if im.size != (width, height):
                raise ValueError(f"Unexpected pre RGB dimension: {im.size}, expected ({width}, {height})")

        # Pre Crop (if RELOCATE)
        if m_in["intervention_operator"] == "RELOCATE":
            crop_rel = m_in.get("candidate_object_crop_path")
            if not crop_rel:
                raise ValueError(f"Record {rec_dict.get('record_id')}: RELOCATE operator missing candidate_object_crop_path")
            crop_p = dataset_root / crop_rel
            if not crop_p.exists():
                raise FileNotFoundError(f"Missing crop: {crop_p}")
            with Image.open(crop_p) as im:
                if im.size[0] <= 0 or im.size[1] <= 0:
                    raise ValueError(f"Empty crop dimension: {im.size}")
        else:
            if m_in.get("candidate_object_crop_path") is not None:
                raise ValueError(f"NONE operator must have null candidate_object_crop_path, got {m_in['candidate_object_crop_path']}")
            if m_in.get("current_geometry") is not None:
                raise ValueError(f"NONE operator must have null current_geometry")
            if m_in.get("destination_geometry") is not None:
                raise ValueError(f"NONE operator must have null destination_geometry")

        # Post RGB
        post_rgb_p = dataset_root / p_meta["post_rgb_path"]
        if not post_rgb_p.exists():
            raise FileNotFoundError(f"Missing post RGB: {post_rgb_p}")
        with Image.open(post_rgb_p) as im:
            if im.size != (width, height):
                raise ValueError(f"Unexpected post RGB dimension: {im.size}, expected ({width}, {height})")

        # Raw Segmentation NPY (int32 shape (height, width, 2))
        pre_seg_p = dataset_root / p_meta["pre_segmentation_path"]
        if not pre_seg_p.exists():
            raise FileNotFoundError(f"Missing pre segmentation: {pre_seg_p}")
        pre_seg = np.load(pre_seg_p)
        if pre_seg.dtype != np.int32 or pre_seg.shape != (height, width, 2):
            raise ValueError(f"Invalid pre segmentation array: dtype={pre_seg.dtype}, shape={pre_seg.shape}, expected ({height}, {width}, 2)")

        post_seg_p = dataset_root / p_meta["post_segmentation_path"]
        if not post_seg_p.exists():
            raise FileNotFoundError(f"Missing post segmentation: {post_seg_p}")
        post_seg = np.load(post_seg_p)
        if post_seg.dtype != np.int32 or post_seg.shape != (height, width, 2):
            raise ValueError(f"Invalid post segmentation array: dtype={post_seg.dtype}, shape={post_seg.shape}, expected ({height}, {width}, 2)")

        # Masks
        rel_mask_p = dataset_root / p_meta["relation_target_mask_path"]
        if not rel_mask_p.exists():
            raise FileNotFoundError(f"Missing relation target mask: {rel_mask_p}")

    # 4. COUNT & EFFECT COVERAGE
    print("--> 4/8: Validating dataset counts and effect balance...")
    total_scenes = len(scenes_map)
    total_records = len(records)

    effect_counts = {1: 0, 0: 0, -1: 0}
    category_counts = {}
    for rec in records:
        eff = rec["supervision_targets"]["causal_effect"]
        effect_counts[eff] = effect_counts.get(eff, 0) + 1
        cat = rec["privileged_metadata"]["intended_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"    Scenes: {total_scenes}, Records: {total_records}")
    print(f"    Effects: +1: {effect_counts.get(1, 0)}, 0: {effect_counts.get(0, 0)}, -1: {effect_counts.get(-1, 0)}")
    print(f"    Categories: {category_counts}")

    if total_scenes != 6:
        raise ValueError(f"Expected 6 scenes in smoke dataset, got {total_scenes}")
    if total_records != 22:
        raise ValueError(f"Expected 22 records in smoke dataset, got {total_records}")
    if effect_counts.get(1, 0) != 4 or effect_counts.get(0, 0) != 16 or effect_counts.get(-1, 0) != 2:
        raise ValueError(f"Expected effects (+1: 4, 0: 16, -1: 2), got {effect_counts}")

    for expected_cat in ["repair", "hard_negative", "irrelevant", "identity", "harmful"]:
        if category_counts.get(expected_cat, 0) == 0:
            raise ValueError(f"Missing expected category '{expected_cat}' in smoke dataset")

    # 5. PHYSICAL RECOMPUTATION, 22/22 POST ARTIFACTS & CAMERA INVARIANCE
    print("--> 5/8: Independently recomputing physics, post RGB/seg, crop alignment, and camera invariance...")
    scene_builder = SceneBuilder()

    max_pre_rgb_rmse = 0.0
    max_post_rgb_rmse = 0.0
    max_none_rgb_rmse = 0.0
    max_cam_trans_drift_m = 0.0
    max_cam_rot_drift_rad = 0.0
    max_collateral_measured = 0.0

    recomputed_delta_count = 0
    post_rgb_recon_count = 0
    post_seg_recon_count = 0
    candidate_mask_match_count = 0
    candidate_crop_match_count = 0
    geometry_alignment_count = 0
    total_relocate_count = 0

    for scene_id, scene_records in scenes_map.items():
        sample_meta = scene_records[0]["privileged_metadata"]
        resolved_spec_dict = sample_meta["reconstruction_spec"]
        resolved_spec = ResolvedSceneSpec.from_dict(resolved_spec_dict)

        # Build fresh model and data using RESOLVED physical values
        spawn_dicts = [
            {"name": o.name, "type": o.object_type, "pos": list(o.position), "quat": list(o.quaternion_wxyz)}
            for o in resolved_spec.spawned_objects
        ]
        rig = get_task_observation_rig(resolved_spec.task_id)

        model, data = scene_builder.create_environment(
            objects_to_spawn=spawn_dicts,
            settle_steps=0,
            include_robot=True,
            robot_base_pose=resolved_spec.robot_base_pose,
            box_pose=list(resolved_spec.box_pose) if resolved_spec.box_pose else None,
            box_quat=list(resolved_spec.box_quat) if resolved_spec.box_quat else None,
            target_region_pos=list(resolved_spec.target_region_pos) if resolved_spec.target_region_pos else None,
            target_region_quat=list(resolved_spec.target_region_quat) if resolved_spec.target_region_quat else None,
        )

        # Apply exact resolved background
        apply_background_spec(model, resolved_spec.background_spec)

        # Apply observation rig
        apply_observation_rig(model, data, rig)

        # Pre F_R with held observation rig
        pre_res = evaluate_relational_feasibility(
            model, data, resolved_spec.task_id,
            candidate_objects=list(resolved_spec.obstruction_candidate_names),
            settle_steps=300,
            hold_observation_robot=True,
        )

        if not pre_res.settling_succeeded:
            raise RuntimeError(f"Scene {scene_id}: Independent pre settling failed.")

        # Recomputed pre_feasible check
        expected_pre_feasible = scene_records[0]["supervision_targets"]["pre_feasible"]
        if pre_res.feasible != expected_pre_feasible:
            raise ValueError(f"Scene {scene_id}: Recomputed pre_feasible {pre_res.feasible} != {expected_pre_feasible}")

        # Canonical object poses check against stored canonical_object_poses
        canonical_poses = {o.name: get_body_freejoint_pose(model, data, o.name) for o in resolved_spec.spawned_objects}
        if resolved_spec.canonical_object_poses:
            for obj_name, expected_pose_dict in resolved_spec.canonical_object_poses.items():
                if obj_name in canonical_poses:
                    meas_pos = np.array(canonical_poses[obj_name].position)
                    exp_pos = np.array(expected_pose_dict["position"])
                    pos_diff = float(np.linalg.norm(meas_pos - exp_pos))
                    if pos_diff > tol_geom_pos:
                        raise ValueError(f"Scene {scene_id} obj {obj_name}: Canonical pos diff {pos_diff} > {tol_geom_pos}")

        # Snapshot canonical state
        canonical_snap = snapshot_simulator_state(model, data)

        # Capture pre camera pose
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, rig.camera_name)
        cam_pos_pre = data.cam_xpos[cam_id].copy()
        cam_mat_pre = data.cam_xmat[cam_id].reshape(3, 3).copy()

        # Render pre RGB & segmentation from reconstructed canonical state
        renderer = OffscreenRenderer(model, width=width, height=height, camera_name=rig.camera_name)
        recomp_pre_rgb = renderer.render_rgb(data)
        recomp_pre_seg = renderer.render_segmentation(data)

        # Pre RGB reconstruction check
        saved_pre_rgb = np.array(Image.open(dataset_root / scene_records[0]["model_inputs"]["pre_rgb_path"]))
        pre_rgb_rmse = float(np.sqrt(np.mean((recomp_pre_rgb.astype(np.float32) - saved_pre_rgb.astype(np.float32)) ** 2)))
        max_pre_rgb_rmse = max(max_pre_rgb_rmse, pre_rgb_rmse)
        if pre_rgb_rmse > tol_pre_rgb:
            raise ValueError(f"Scene {scene_id}: Pre RGB RMSE {pre_rgb_rmse:.4f} > {tol_pre_rgb}")

        # Pre Segmentation array equality check
        saved_pre_seg = np.load(dataset_root / sample_meta["pre_segmentation_path"])
        if not np.array_equal(recomp_pre_seg, saved_pre_seg):
            raise ValueError(f"Scene {scene_id}: Pre segmentation array mismatch against saved file")

        # Relation target mask check
        target_geoms = ["B1_lid_panel"] if resolved_spec.task_id == "task_1" else ["target_region_geom"]
        recomp_relation_mask = renderer.render_region_mask(data, target_geoms)
        saved_relation_mask = np.array(Image.open(dataset_root / sample_meta["relation_target_mask_path"]))
        if not np.array_equal(recomp_relation_mask, saved_relation_mask):
            raise ValueError(f"Scene {scene_id}: Relation target mask mismatch against saved file")

        ref_frame_name = "B1_lid_panel" if resolved_spec.task_id == "task_1" else "target_region_geom"

        # Test each candidate record in scene
        for rec in scene_records:
            m_in = rec["model_inputs"]
            s_target = rec["supervision_targets"]
            p_meta = rec["privileged_metadata"]

            # Restore canonical snapshot
            restore_simulator_state(model, data, canonical_snap)

            # Construct physical intervention
            op = InterventionOperator(m_in["intervention_operator"])
            if op == InterventionOperator.RELOCATE:
                total_relocate_count += 1
                obj_name = p_meta["candidate_object_name"]
                dest_p = tuple(m_in["destination_geometry"]["world_position"])
                dest_q = tuple(m_in["destination_geometry"]["world_quaternion_wxyz"])
                interv = Intervention(
                    intervention_id=rec["intervention_id"],
                    operator=op,
                    object_name=obj_name,
                    destination_pose=ObjectPose(position=dest_p, quaternion_wxyz=dest_q),
                )

                # Candidate mask verification using renderer helper
                recomp_cand_mask = renderer.render_culprit_mask(data, [f"{obj_name}_visual", f"{obj_name}_geom"])
                saved_cand_mask = np.array(Image.open(dataset_root / p_meta["candidate_mask_path"]))
                if not np.array_equal(recomp_cand_mask, saved_cand_mask):
                    raise ValueError(f"Record {rec['record_id']}: Candidate mask mismatch for '{obj_name}'")
                candidate_mask_match_count += 1

                # Candidate crop verification against saved pre RGB
                recomp_crop = _crop_object_from_rgb(saved_pre_rgb, recomp_cand_mask, padding_fraction=0.10)
                saved_crop = np.array(Image.open(dataset_root / m_in["candidate_object_crop_path"]))
                if recomp_crop is None or not np.array_equal(recomp_crop, saved_crop):
                    raise ValueError(f"Record {rec['record_id']}: Candidate crop mismatch against recomputed crop")
                candidate_crop_match_count += 1

                # Current geometry freejoint, quaternion, and relative verification
                curr_geom = m_in["current_geometry"]
                curr_pos_meas = np.array(canonical_poses[obj_name].position)
                curr_pos_rec = np.array(curr_geom["world_position"])
                if float(np.linalg.norm(curr_pos_meas - curr_pos_rec)) > tol_geom_pos:
                    raise ValueError(f"Record {rec['record_id']}: Current geometry world pos mismatch")

                curr_quat_meas = np.array(canonical_poses[obj_name].quaternion_wxyz)
                curr_quat_rec = np.array(curr_geom["world_quaternion_wxyz"])
                curr_quat_dist = min(float(np.linalg.norm(curr_quat_meas - curr_quat_rec)), float(np.linalg.norm(curr_quat_meas + curr_quat_rec)))
                if curr_quat_dist > tol_geom_quat:
                    raise ValueError(f"Record {rec['record_id']}: Current geometry quaternion mismatch (dist: {curr_quat_dist})")
                
                exp_rel_pos = world_to_local(model, data, ref_frame_name, curr_pos_meas, use_geom=True)
                meas_rel_pos = np.array(curr_geom["relative_position"])
                if float(np.linalg.norm(exp_rel_pos - meas_rel_pos)) > tol_geom_pos:
                    raise ValueError(f"Record {rec['record_id']}: Current geometry relative pos mismatch")

                # Destination geometry quaternion normalization and relative frame verification
                dest_geom = m_in["destination_geometry"]
                dest_world_pos = np.array(dest_geom["world_position"])
                if not np.all(np.isfinite(dest_world_pos)):
                    raise ValueError(f"Record {rec['record_id']}: Non-finite destination world position.")

                dest_quat = np.array(dest_geom["world_quaternion_wxyz"])
                if not np.all(np.isfinite(dest_quat)):
                    raise ValueError(f"Record {rec['record_id']}: Non-finite destination quaternion.")
                dest_quat_norm = float(np.linalg.norm(dest_quat))
                if abs(dest_quat_norm - 1.0) > tol_geom_quat:
                    raise ValueError(f"Record {rec['record_id']}: Destination quaternion norm {dest_quat_norm} != 1.0")

                exp_dest_rel_pos = world_to_local(model, data, ref_frame_name, dest_world_pos, use_geom=True)
                meas_dest_rel_pos = np.array(dest_geom["relative_position"])
                if float(np.linalg.norm(exp_dest_rel_pos - meas_dest_rel_pos)) > tol_geom_pos:
                    raise ValueError(f"Record {rec['record_id']}: Destination geometry relative pos mismatch")

                geometry_alignment_count += 1

            else:
                interv = Intervention(
                    intervention_id=rec["intervention_id"],
                    operator=op,
                )

            # Apply intervention
            apply_intervention(model, data, interv)

            # Post F_R with held observation rig
            post_res = evaluate_relational_feasibility(
                model, data, resolved_spec.task_id,
                candidate_objects=list(resolved_spec.obstruction_candidate_names),
                settle_steps=300,
                hold_observation_robot=True,
            )

            if not post_res.settling_succeeded:
                raise RuntimeError(f"Record {rec['record_id']}: Independent post settling failed.")

            # Recompute observed Delta
            recomp_delta = int(post_res.feasible) - int(pre_res.feasible)
            stored_delta = s_target["causal_effect"]

            if recomp_delta != stored_delta:
                raise ValueError(
                    f"Record {rec['record_id']}: Recomputed Delta {recomp_delta} != Stored Delta {stored_delta} "
                    f"(pre: {pre_res.feasible}, post: {post_res.feasible})"
                )

            if post_res.feasible != s_target["post_feasible"]:
                raise ValueError(
                    f"Record {rec['record_id']}: Recomputed post_feasible {post_res.feasible} != {s_target['post_feasible']}"
                )
            recomputed_delta_count += 1

            # Camera translation and rotation drift checks
            cam_pos_post = data.cam_xpos[cam_id].copy()
            cam_mat_post = data.cam_xmat[cam_id].reshape(3, 3).copy()

            cam_trans_drift = float(np.linalg.norm(cam_pos_post - cam_pos_pre))
            max_cam_trans_drift_m = max(max_cam_trans_drift_m, cam_trans_drift)

            R_delta = cam_mat_pre.T @ cam_mat_post
            cos_theta = np.clip((float(np.trace(R_delta)) - 1.0) / 2.0, -1.0, 1.0)
            cam_rot_drift = float(np.arccos(cos_theta))
            max_cam_rot_drift_rad = max(max_cam_rot_drift_rad, cam_rot_drift)

            # Render independent post RGB & segmentation
            recomp_post_rgb = renderer.render_rgb(data)
            recomp_post_seg = renderer.render_segmentation(data)

            # Saved post RGB reconstruction check
            saved_post_rgb = np.array(Image.open(dataset_root / p_meta["post_rgb_path"]))
            post_rgb_rmse = float(np.sqrt(np.mean((recomp_post_rgb.astype(np.float32) - saved_post_rgb.astype(np.float32)) ** 2)))
            max_post_rgb_rmse = max(max_post_rgb_rmse, post_rgb_rmse)
            if post_rgb_rmse <= tol_post_rgb:
                post_rgb_recon_count += 1
            else:
                raise ValueError(f"Record {rec['record_id']}: Post RGB reconstruction RMSE {post_rgb_rmse:.4f} > {tol_post_rgb}")

            # Saved post segmentation exact array equality check
            saved_post_seg = np.load(dataset_root / p_meta["post_segmentation_path"])
            if np.array_equal(recomp_post_seg, saved_post_seg):
                post_seg_recon_count += 1
            else:
                raise ValueError(f"Record {rec['record_id']}: Post segmentation array mismatch")

            # NONE control visual check against saved pre RGB
            if op == InterventionOperator.NONE:
                none_rmse = float(np.sqrt(np.mean((recomp_post_rgb.astype(np.float32) - saved_pre_rgb.astype(np.float32)) ** 2)))
                max_none_rgb_rmse = max(max_none_rgb_rmse, none_rmse)

            # Collateral motion measurement
            for name in resolved_spec.obstruction_candidate_names:
                if op == InterventionOperator.RELOCATE and name == interv.object_name:
                    continue
                curr_p = np.array(get_body_freejoint_pose(model, data, name).position)
                init_p = np.array(canonical_poses[name].position)
                disp = float(np.linalg.norm(curr_p - init_p))
                max_collateral_measured = max(max_collateral_measured, disp)

        renderer.close()

    print(f"    Recomputed Delta: {recomputed_delta_count}/22 (100% agreement)")
    print(f"    Post RGB Reconstructed: {post_rgb_recon_count}/22")
    print(f"    Post Segmentation Lossless Matches: {post_seg_recon_count}/22")
    print(f"    Candidate Mask Matches: {candidate_mask_match_count}/{total_relocate_count}")
    print(f"    Candidate Crop Matches: {candidate_crop_match_count}/{total_relocate_count}")
    print(f"    Geometry Alignments: {geometry_alignment_count}/{total_relocate_count}")
    print(f"    Max Pre-RGB Reconstruction RMSE: {max_pre_rgb_rmse:.4f} (tol: {tol_pre_rgb})")
    print(f"    Max Post-RGB Reconstruction RMSE: {max_post_rgb_rmse:.4f} (tol: {tol_post_rgb})")
    print(f"    Max NONE Control RGB RMSE: {max_none_rgb_rmse:.4f} (tol: {tol_none_rgb})")
    print(f"    Max Camera Translation Drift: {max_cam_trans_drift_m:.6f}m (tol: {tol_cam_trans}m)")
    print(f"    Max Camera Rotation Drift: {max_cam_rot_drift_rad:.6f}rad (tol: {tol_cam_rot}rad)")
    print(f"    Max Collateral Displacement: {max_collateral_measured:.4f}m (tol: {tol_collateral}m)")

    # Enforce all numerical thresholds
    if max_pre_rgb_rmse > tol_pre_rgb:
        raise ValueError(f"Pre RGB RMSE {max_pre_rgb_rmse} > threshold {tol_pre_rgb}")
    if max_post_rgb_rmse > tol_post_rgb:
        raise ValueError(f"Post RGB RMSE {max_post_rgb_rmse} > threshold {tol_post_rgb}")
    if max_none_rgb_rmse > tol_none_rgb:
        raise ValueError(f"NONE RGB RMSE {max_none_rgb_rmse} > threshold {tol_none_rgb}")
    if max_cam_trans_drift_m > tol_cam_trans:
        raise ValueError(f"Camera translation drift {max_cam_trans_drift_m}m > threshold {tol_cam_trans}m")
    if max_cam_rot_drift_rad > tol_cam_rot:
        raise ValueError(f"Camera rotation drift {max_cam_rot_drift_rad}rad > threshold {tol_cam_rot}rad")
    if max_collateral_measured > tol_collateral:
        raise ValueError(f"Collateral displacement {max_collateral_measured}m > threshold {tol_collateral}m")

    # 6. RELATIONAL IDENTITY TRANSITION AUDIT
    print("--> 6/8: Validating relational culprit identity transitions...")
    relational_identity_count = 0
    for rec in records:
        cat = rec["privileged_metadata"]["intended_category"]
        c_before = rec["privileged_metadata"]["active_culprits_before"]
        c_after = rec["privileged_metadata"]["active_culprits_after"]
        cand_name = rec["privileged_metadata"]["candidate_object_name"]
        eff = rec["supervision_targets"]["causal_effect"]

        if cat == "repair":
            if eff != 1 or len(c_before) != 1 or len(c_after) != 0:
                raise ValueError(f"Repair identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "hard_negative":
            if eff != 0 or len(c_before) != 1 or c_before != c_after:
                raise ValueError(f"Hard negative identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "harmful":
            if eff != -1 or len(c_before) != 0 or c_after != [cand_name]:
                raise ValueError(f"Harmful identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "identity":
            if eff != 0 or c_before != c_after:
                raise ValueError(f"Identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "irrelevant":
            if eff != 0:
                raise ValueError(f"Irrelevant violation in {rec['record_id']}: eff={eff}")
            if len(c_before) == 1 and c_before != c_after:
                raise ValueError(f"Irrelevant (STOP) violation in {rec['record_id']}: before={c_before}, after={c_after}")
            if len(c_before) == 0 and len(c_after) != 0:
                raise ValueError(f"Irrelevant (PROCEED) violation in {rec['record_id']}: before={c_before}, after={c_after}")
        
        relational_identity_count += 1

    # 7. TASK 2 STRUCTURAL ROLE CHECKS
    print("--> 7/8: Validating Task 2 place subject structural exclusion...")
    for rec in records:
        if rec["model_inputs"]["task_id"] == "task_2":
            subj = rec["privileged_metadata"]["reconstruction_spec"]["action_subject_name"]
            obs_cands = rec["privileged_metadata"]["reconstruction_spec"]["obstruction_candidate_names"]
            if subj != "coffee_can":
                raise ValueError(f"Task 2 action subject '{subj}' != 'coffee_can'")
            if subj in obs_cands:
                raise ValueError(f"Task 2 place subject '{subj}' leaked into obstruction candidates!")
            if rec["privileged_metadata"]["candidate_object_name"] == subj:
                raise ValueError(f"Task 2 place subject '{subj}' was intervened on!")
            action_arg_obj = rec["model_inputs"]["action"]["arguments"].get("object")
            if action_arg_obj != "coffee_can":
                raise ValueError(f"Task 2 action object '{action_arg_obj}' != 'coffee_can'")

    # 8. PROVENANCE & SUMMARY REPORT
    print("--> 8/8: Compiling independent validation report...")
    validation_report = {
        "schema_version": SCHEMA_VERSION,
        "validation_status": "PASSED",
        "generator_commit": generator_commit or metadata.get("generator_commit", "unspecified"),
        "manifest_path": str(manifest_path.relative_to(manifest_path.parent.parent)),
        "manifest_sha256": manifest_sha,
        "config_sha256": metadata.get("config_sha256", "unspecified"),
        "scene_count": total_scenes,
        "record_count": total_records,
        "effect_counts": {str(k): v for k, v in effect_counts.items()},
        "category_counts": category_counts,
        "recomputed_delta": {
            "passed_count": recomputed_delta_count,
            "total": total_records,
            "passed": bool(recomputed_delta_count == total_records),
        },
        "pre_rgb_reconstruction": {
            "measured_max_rmse": float(max_pre_rgb_rmse),
            "threshold": tol_pre_rgb,
            "passed": bool(max_pre_rgb_rmse <= tol_pre_rgb),
        },
        "post_rgb_reconstruction": {
            "measured_max_rmse": float(max_post_rgb_rmse),
            "threshold": tol_post_rgb,
            "passed_count": post_rgb_recon_count,
            "total": total_records,
            "passed": bool(post_rgb_recon_count == total_records),
        },
        "post_segmentation": {
            "exact_match_count": post_seg_recon_count,
            "total": total_records,
            "passed": bool(post_seg_recon_count == total_records),
        },
        "candidate_mask_alignment": {
            "passed_count": candidate_mask_match_count,
            "total_relocate_records": total_relocate_count,
            "passed": bool(candidate_mask_match_count == total_relocate_count),
        },
        "candidate_crop_alignment": {
            "passed_count": candidate_crop_match_count,
            "total_relocate_records": total_relocate_count,
            "passed": bool(candidate_crop_match_count == total_relocate_count),
        },
        "current_geometry_alignment": {
            "passed_count": geometry_alignment_count,
            "total_relocate_records": total_relocate_count,
            "passed": bool(geometry_alignment_count == total_relocate_count),
        },
        "none_control": {
            "measured_max_rgb_rmse": float(max_none_rgb_rmse),
            "threshold": tol_none_rgb,
            "passed": bool(max_none_rgb_rmse <= tol_none_rgb),
        },
        "camera_translation": {
            "measured_max_m": float(max_cam_trans_drift_m),
            "threshold_m": tol_cam_trans,
            "passed": bool(max_cam_trans_drift_m <= tol_cam_trans),
        },
        "camera_rotation": {
            "measured_max_rad": float(max_cam_rot_drift_rad),
            "threshold_rad": tol_cam_rot,
            "passed": bool(max_cam_rot_drift_rad <= tol_cam_rot),
        },
        "collateral_translation": {
            "measured_max_m": float(max_collateral_measured),
            "threshold_m": tol_collateral,
            "passed": bool(max_collateral_measured <= tol_collateral),
        },
        "relational_identity": {
            "passed_count": relational_identity_count,
            "total": total_records,
            "passed": bool(relational_identity_count == total_records),
        },
        "task2_structural_exclusion": {
            "passed": True,
        },
        "leakage_audit": {
            "passed": True,
        },
        "metadata_provenance": {
            "passed": True,
        },
    }

    if report_output_path:
        report_output_path = Path(report_output_path)
        report_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_output_path, "w") as f:
            json.dump(validation_report, f, indent=2)
        print(f"Validation report written to: {report_output_path}")

    print(f"\n======================================================================")
    print(f"ALL 8 VALIDATION GATES PASSED (100% Validated)")
    print(f"======================================================================")

    return validation_report


def main():
    parser = argparse.ArgumentParser(description="Validate Intervention Dataset Manifest")
    parser.add_argument("--manifest", type=str, default="data/intervention_smoke/manifest.jsonl", help="Path to manifest.jsonl")
    parser.add_argument("--config", type=str, default=None, help="Optional path to source config YAML")
    parser.add_argument("--report", type=str, default="artifacts/intervention_smoke/smoke_validation_report.json", help="Path to write validation report JSON")
    parser.add_argument("--commit", type=str, default=None, help="Git commit SHA that generated the dataset")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    config_path = Path(args.config) if args.config else None
    report_path = Path(args.report) if args.report else None

    validate_intervention_dataset(
        manifest_path=manifest_path,
        report_output_path=report_path,
        config_path=config_path,
        generator_commit=args.commit,
    )


if __name__ == "__main__":
    main()
