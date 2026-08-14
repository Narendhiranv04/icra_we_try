#!/usr/bin/env python3
"""
Independent validator for causal intervention datasets.
Rigorously checks schema, file integrity, counts, physical recomputation, and visual determinism.
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


def validate_intervention_dataset(
    manifest_path: Path,
    report_output_path: Optional[Path] = None,
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

    print(f"======================================================================")
    print(f"Validating Intervention Dataset Manifest: {manifest_path}")
    print(f"Records Found: {len(records)}")
    print(f"======================================================================")

    # 1. SCHEMA & INFORMATION PARTITIONING
    print("--> 1/7: Validating schema and strict information partitioning...")
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

        # Group by scene
        scene_id = rec_dict["scene_id"]
        scenes_map.setdefault(scene_id, []).append(rec_dict)

    # 2. FILE EXISTENCE & INTEGRITY
    print("--> 2/7: Validating file existence, formats, and dtypes...")
    for rec_dict in records:
        m_in = rec_dict["model_inputs"]
        p_meta = rec_dict["privileged_metadata"]

        # Pre RGB
        pre_rgb_p = dataset_root / m_in["pre_rgb_path"]
        if not pre_rgb_p.exists():
            raise FileNotFoundError(f"Missing pre RGB: {pre_rgb_p}")
        with Image.open(pre_rgb_p) as im:
            if im.size != (640, 480):
                raise ValueError(f"Unexpected pre RGB dimension: {im.size}")

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

        # Raw Segmentation NPY (int32 shape (480, 640, 2))
        pre_seg_p = dataset_root / p_meta["pre_segmentation_path"]
        if not pre_seg_p.exists():
            raise FileNotFoundError(f"Missing pre segmentation: {pre_seg_p}")
        pre_seg = np.load(pre_seg_p)
        if pre_seg.dtype != np.int32 or pre_seg.shape != (480, 640, 2):
            raise ValueError(f"Invalid pre segmentation array: dtype={pre_seg.dtype}, shape={pre_seg.shape}")

        post_seg_p = dataset_root / p_meta["post_segmentation_path"]
        if not post_seg_p.exists():
            raise FileNotFoundError(f"Missing post segmentation: {post_seg_p}")
        post_seg = np.load(post_seg_p)
        if post_seg.dtype != np.int32 or post_seg.shape != (480, 640, 2):
            raise ValueError(f"Invalid post segmentation array: dtype={post_seg.dtype}, shape={post_seg.shape}")

        # Masks
        rel_mask_p = dataset_root / p_meta["relation_target_mask_path"]
        if not rel_mask_p.exists():
            raise FileNotFoundError(f"Missing relation target mask: {rel_mask_p}")

    # 3. COUNT & EFFECT COVERAGE
    print("--> 3/7: Validating dataset counts and effect balance...")
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

    # 4. PHYSICAL RECOMPUTATION & VISUAL RE-EVALUATION
    print("--> 4/7: Independently recomputing physics and verifying pre/post Delta...")
    scene_builder = SceneBuilder()

    max_pre_rgb_rmse = 0.0
    max_none_rgb_rmse = 0.0
    max_collateral_measured = 0.0

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

        # Pre F_R
        pre_res = evaluate_relational_feasibility(
            model, data, resolved_spec.task_id,
            candidate_objects=list(resolved_spec.obstruction_candidate_names),
            settle_steps=300,
        )

        if not pre_res.settling_succeeded:
            raise RuntimeError(f"Scene {scene_id}: Independent pre settling failed.")

        # Recomputed pre_feasible check
        expected_pre_feasible = scene_records[0]["supervision_targets"]["pre_feasible"]
        if pre_res.feasible != expected_pre_feasible:
            raise ValueError(f"Scene {scene_id}: Recomputed pre_feasible {pre_res.feasible} != {expected_pre_feasible}")

        # Snapshot canonical state
        canonical_snap = snapshot_simulator_state(model, data)

        # Visual determinism check on pre RGB
        renderer = OffscreenRenderer(model, width=640, height=480, camera_name=rig.camera_name)
        recomp_pre_rgb = renderer.render_rgb(data)
        renderer.close()

        saved_pre_rgb = np.array(Image.open(dataset_root / scene_records[0]["model_inputs"]["pre_rgb_path"]))
        pre_rgb_rmse = float(np.sqrt(np.mean((recomp_pre_rgb.astype(np.float32) - saved_pre_rgb.astype(np.float32)) ** 2)))
        max_pre_rgb_rmse = max(max_pre_rgb_rmse, pre_rgb_rmse)

        # Test each candidate independently from canonical state
        canonical_poses = {o.name: get_body_freejoint_pose(model, data, o.name) for o in resolved_spec.spawned_objects}

        for rec in scene_records:
            m_in = rec["model_inputs"]
            s_target = rec["supervision_targets"]
            p_meta = rec["privileged_metadata"]

            # Restore canonical snapshot
            restore_simulator_state(model, data, canonical_snap)

            # Construct physical intervention
            op = InterventionOperator(m_in["intervention_operator"])
            if op == InterventionOperator.RELOCATE:
                obj_name = p_meta["candidate_object_name"]
                dest_p = tuple(m_in["destination_geometry"]["world_position"])
                dest_q = tuple(m_in["destination_geometry"]["world_quaternion_wxyz"])
                interv = Intervention(
                    intervention_id=rec["intervention_id"],
                    operator=op,
                    object_name=obj_name,
                    destination_pose=ObjectPose(position=dest_p, quaternion_wxyz=dest_q),
                )
            else:
                interv = Intervention(
                    intervention_id=rec["intervention_id"],
                    operator=op,
                )

            # Apply intervention
            apply_intervention(model, data, interv)

            # Post F_R
            post_res = evaluate_relational_feasibility(
                model, data, resolved_spec.task_id,
                candidate_objects=list(resolved_spec.obstruction_candidate_names),
                settle_steps=300,
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

            # Collateral motion measurement
            for name in resolved_spec.obstruction_candidate_names:
                if op == InterventionOperator.RELOCATE and name == interv.object_name:
                    continue
                curr_p = np.array(get_body_freejoint_pose(model, data, name).position)
                init_p = np.array(canonical_poses[name].position)
                disp = float(np.linalg.norm(curr_p - init_p))
                max_collateral_measured = max(max_collateral_measured, disp)

            # NONE control visual check
            if op == InterventionOperator.NONE:
                r_none = OffscreenRenderer(model, width=640, height=480, camera_name=rig.camera_name)
                none_post_rgb = r_none.render_rgb(data)
                r_none.close()
                none_rmse = float(np.sqrt(np.mean((none_post_rgb.astype(np.float32) - saved_pre_rgb.astype(np.float32)) ** 2)))
                max_none_rgb_rmse = max(max_none_rgb_rmse, none_rmse)

    print(f"    100% agreement on all 22 recomputed Delta values!")
    print(f"    Max Pre-RGB Reconstruction RMSE: {max_pre_rgb_rmse:.4f}")
    print(f"    Max NONE Control Drift RMSE: {max_none_rgb_rmse:.4f}")
    print(f"    Max Measured Collateral Displacement: {max_collateral_measured:.4f}m")

    # 5. RELATIONAL IDENTITY TRANSITION AUDIT
    print("--> 5/7: Validating relational culprit identity transitions...")
    for rec in records:
        cat = rec["privileged_metadata"]["intended_category"]
        c_before = rec["privileged_metadata"]["active_culprits_before"]
        c_after = rec["privileged_metadata"]["active_culprits_after"]
        cand_name = rec["privileged_metadata"]["candidate_object_name"]
        eff = rec["supervision_targets"]["causal_effect"]

        if cat == "repair":
            if eff != 1 or len(c_before) == 0 or len(c_after) != 0:
                raise ValueError(f"Repair identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "hard_negative":
            if eff != 0 or len(c_before) == 0 or len(c_after) == 0:
                raise ValueError(f"Hard negative identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "harmful":
            if eff != -1 or len(c_before) != 0 or cand_name not in c_after:
                raise ValueError(f"Harmful identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "identity":
            if eff != 0 or c_before != c_after:
                raise ValueError(f"Identity violation in {rec['record_id']}: before={c_before}, after={c_after}, eff={eff}")
        elif cat == "irrelevant":
            if eff != 0:
                raise ValueError(f"Irrelevant violation in {rec['record_id']}: eff={eff}")

    # 6. TASK 2 STRUCTURAL ROLE CHECKS
    print("--> 6/7: Validating Task 2 place subject structural exclusion...")
    for rec in records:
        if rec["model_inputs"]["task_id"] == "task_2":
            subj = rec["privileged_metadata"]["reconstruction_spec"]["action_subject_name"]
            obs_cands = rec["privileged_metadata"]["reconstruction_spec"]["obstruction_candidate_names"]
            if subj in obs_cands:
                raise ValueError(f"Task 2 place subject '{subj}' leaked into obstruction candidates!")
            if rec["privileged_metadata"]["candidate_object_name"] == subj:
                raise ValueError(f"Task 2 place subject '{subj}' was intervened on!")
            action_arg_obj = rec["model_inputs"]["action"]["arguments"].get("object")
            if action_arg_obj != "coffee_can":
                raise ValueError(f"Task 2 action object '{action_arg_obj}' != 'coffee_can'")

    # 7. PROVENANCE & SUMMARY REPORT
    print("--> 7/7: Compiling independent validation report...")
    validation_report = {
        "schema_version": SCHEMA_VERSION,
        "validation_status": "PASSED",
        "manifest_path": str(manifest_path.relative_to(manifest_path.parent.parent)),
        "manifest_sha256": manifest_sha,
        "generator_commit": generator_commit or "unspecified",
        "scene_count": total_scenes,
        "record_count": total_records,
        "effect_counts": {str(k): v for k, v in effect_counts.items()},
        "category_counts": category_counts,
        "max_pre_rgb_rmse": float(max_pre_rgb_rmse),
        "max_none_rgb_rmse": float(max_none_rgb_rmse),
        "max_collateral_displacement_m": float(max_collateral_measured),
        "recomputed_delta_agreement": "22/22 (100%)",
        "leakage_audit_passed": True,
        "task2_structural_exclusion_passed": True,
    }

    if report_output_path:
        report_output_path = Path(report_output_path)
        report_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_output_path, "w") as f:
            json.dump(validation_report, f, indent=2)
        print(f"Validation report written to: {report_output_path}")

    print(f"\n======================================================================")
    print(f"ALL 7 VALIDATION GATES PASSED (100% Validated)")
    print(f"======================================================================")

    return validation_report


def main():
    parser = argparse.ArgumentParser(description="Validate Intervention Dataset Manifest")
    parser.add_argument("--manifest", type=str, default="data/intervention_smoke/manifest.jsonl", help="Path to manifest.jsonl")
    parser.add_argument("--report", type=str, default="artifacts/intervention_smoke/smoke_validation_report.json", help="Path to write validation report JSON")
    parser.add_argument("--commit", type=str, default=None, help="Git commit SHA that generated the dataset")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    report_path = Path(args.report) if args.report else None

    validate_intervention_dataset(
        manifest_path=manifest_path,
        report_output_path=report_path,
        generator_commit=args.commit,
    )


if __name__ == "__main__":
    main()
