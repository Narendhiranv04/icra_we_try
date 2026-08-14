"""
Scene generator and counterfactual renderer producing physically grounded intervention records.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import math
import numpy as np
import PIL.Image as Image
import mujoco

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.environment.observation_rig import (
    ObservationRig,
    get_task_observation_rig,
    apply_observation_rig,
)
from src.environment.scene_utils import (
    sample_position_on_lid,
    sample_position_beside_box,
    sample_position_in_target,
    sample_position_outside_target,
    world_to_local,
)
from src.generation.background_randomization import (
    BackgroundSpec,
    sample_background_spec,
    apply_background_spec,
)
from src.interventions.intervention_types import (
    Action,
    Intervention,
    InterventionOperator,
    InterventionOutcome,
    ObjectPose,
    PrivilegedCategory,
)
from src.interventions.intervention_validator import (
    InterventionValidator,
    get_body_freejoint_pose,
    snapshot_simulator_state,
    restore_simulator_state,
)
from src.interventions.intervention_generator import InterventionGenerator
from src.interventions.intervention_records import (
    GeometryRelativeSpec,
    InterventionDatasetRecord,
    InterventionSceneSpec,
    ResolvedSceneSpec,
    SpawnedObjectSpec,
    SCHEMA_VERSION,
)
from src.validation.occupancy_checks import evaluate_relational_feasibility


def _colorize_segmentation(seg_raw: np.ndarray) -> np.ndarray:
    """Create RGB 8-bit visualization array from raw int32 segmentation [geom_id, object_id]."""
    geom_ids = seg_raw[:, :, 0]
    # Simple deterministic hash coloring
    r = ((geom_ids * 53) % 255).astype(np.uint8)
    g = ((geom_ids * 97) % 255).astype(np.uint8)
    b = ((geom_ids * 193) % 255).astype(np.uint8)
    # Background (geom_id == -1) set to black
    mask_bg = (geom_ids == -1)
    r[mask_bg] = 0
    g[mask_bg] = 0
    b[mask_bg] = 0
    return np.stack([r, g, b], axis=-1)


def _crop_object_from_rgb(
    rgb_img: np.ndarray,
    obj_mask: np.ndarray,
    padding_fraction: float = 0.10,
) -> Optional[np.ndarray]:
    """Extract object crop from RGB using binary object mask with padding."""
    coords = np.argwhere(obj_mask > 0)
    if coords.size == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    h, w = rgb_img.shape[:2]
    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1

    pad_x = int(box_w * padding_fraction)
    pad_y = int(box_h * padding_fraction)

    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w, x_max + 1 + pad_x)
    y1 = min(h, y_max + 1 + pad_y)

    if (x1 - x0) <= 0 or (y1 - y0) <= 0:
        return None

    return rgb_img[y0:y1, x0:x1].copy()


class InterventionSceneGenerator:
    """Generates canonical base scenes, evaluates candidate interventions on live post-states,
    and produces fully validated, dataset-root-relative intervention records.
    """

    def __init__(
        self,
        scene_builder: Optional[SceneBuilder] = None,
        validator: Optional[InterventionValidator] = None,
        generator: Optional[InterventionGenerator] = None,
        resolution: Tuple[int, int] = (640, 480),
        min_relocation_m: float = 0.03,
        max_collateral_translation_m: float = 0.05,
        min_instance_pixels: int = 50,
        min_relation_target_pixels: int = 50,
        max_torso_fraction: float = 0.25,
    ):
        self.scene_builder = scene_builder if scene_builder is not None else SceneBuilder()
        self.validator = validator if validator is not None else InterventionValidator(
            settle_steps=300, max_collateral_translation_m=max_collateral_translation_m
        )
        self.generator = generator if generator is not None else InterventionGenerator(min_relocation_m=min_relocation_m)
        self.resolution = resolution
        self.min_relocation_m = min_relocation_m
        self.max_collateral_translation_m = max_collateral_translation_m
        self.min_instance_pixels = min_instance_pixels
        self.min_relation_target_pixels = min_relation_target_pixels
        self.max_torso_fraction = max_torso_fraction

    def generate_scene_dataset(
        self,
        spec: InterventionSceneSpec,
        dataset_root: Union[str, Path],
        max_base_retries: int = 10,
        max_candidate_set_retries: int = 5,
        demonstration_reference: Optional[str] = None,
    ) -> Tuple[List[InterventionDatasetRecord], ResolvedSceneSpec]:
        """Generate a complete accepted base scene and its set of candidate intervention records.
        
        Args:
            spec: Requested InterventionSceneSpec
            dataset_root: Root directory of the dataset (e.g. data/intervention_smoke)
            max_base_retries: Maximum attempts to build an acceptable base scene
            max_candidate_set_retries: Maximum attempts to generate an accepted candidate set
            demonstration_reference: Path or identifier to demonstration video/features
            
        Returns:
            Tuple of (accepted InterventionDatasetRecords, ResolvedSceneSpec)
        """
        dataset_root = Path(dataset_root)
        scene_dir = dataset_root / "scenes" / spec.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "masks").mkdir(parents=True, exist_ok=True)
        (scene_dir / "crops").mkdir(parents=True, exist_ok=True)
        (scene_dir / "post").mkdir(parents=True, exist_ok=True)

        rig = get_task_observation_rig(spec.task_id)
        is_stop = (spec.intended_base_state == "STOP")

        # --- STEP 1: BASE SCENE RETRY LOOP ---
        accepted_base = False
        resolved_model = None
        resolved_data = None
        resolved_bg_spec = None
        resolved_objects: List[SpawnedObjectSpec] = []
        actual_base_seed = spec.seed
        realized_base_attempt = 0
        canonical_snapshot = None
        pre_result = None
        observed_culprit: Optional[str] = None
        obstruction_candidates: List[str] = []
        all_visible_objects: List[str] = []

        for base_attempt in range(max_base_retries):
            realized_base_attempt = base_attempt
            actual_base_seed = spec.seed + base_attempt * 10000
            rng_base = np.random.default_rng(actual_base_seed)

            # Sample base scene layout
            ref_model, ref_data = self.scene_builder.create_environment(
                settle_steps=0, include_robot=True, robot_base_pose=rig.robot_base_pose,
                box_pose=list(spec.box_pose) if spec.box_pose else None,
                box_quat=list(spec.box_quat) if spec.box_quat else None,
                target_region_pos=list(spec.target_region_pos) if spec.target_region_pos else None,
                target_region_quat=list(spec.target_region_quat) if spec.target_region_quat else None,
            )

            objects_to_spawn_dicts = []
            spawned_specs = []
            obs_cands = []
            all_vis = []

            if spec.task_id == "task_1":
                if is_stop:
                    # Culprit on lid
                    culprit_name = "blocker1"
                    c_type = spec.intended_culprit_type or "coffee_can"
                    c_pos = sample_position_on_lid(ref_model, ref_data, rng_base, x_frac=0.0, y_frac=0.0, height_above=0.02).tolist()
                    c_quat = [1.0, 0.0, 0.0, 0.0]
                    objects_to_spawn_dicts.append({"name": culprit_name, "type": c_type, "pos": c_pos, "quat": c_quat})
                    spawned_specs.append(SpawnedObjectSpec(name=culprit_name, object_type=c_type, position=tuple(c_pos), quaternion_wxyz=tuple(c_quat), semantic_role="culprit"))
                    obs_cands.append(culprit_name)
                    all_vis.append(culprit_name)

                # Distractors beside box
                for idx, d_type in enumerate(spec.intended_distractor_types):
                    d_name = f"distractor{idx+1}"
                    offset_x = -0.30 - idx * 0.05
                    offset_y = -0.15 + idx * 0.20
                    d_pos = sample_position_beside_box(ref_model, ref_data, rng_base, offset_x=offset_x, offset_y=offset_y, height_above_table=0.04).tolist()
                    d_quat = [1.0, 0.0, 0.0, 0.0]
                    objects_to_spawn_dicts.append({"name": d_name, "type": d_type, "pos": d_pos, "quat": d_quat})
                    spawned_specs.append(SpawnedObjectSpec(name=d_name, object_type=d_type, position=tuple(d_pos), quaternion_wxyz=tuple(d_quat), semantic_role="distractor"))
                    obs_cands.append(d_name)
                    all_vis.append(d_name)

            elif spec.task_id == "task_2":
                # Action subject (e.g. coffee_can) placed outside target
                subj_name = spec.action_subject_name or "coffee_can"
                subj_type = "coffee_can"
                subj_pos = sample_position_outside_target(ref_model, ref_data, rng_base, offset_x=-0.25, offset_y=-0.10, height_above=0.07).tolist()
                subj_quat = [1.0, 0.0, 0.0, 0.0]
                objects_to_spawn_dicts.append({"name": subj_name, "type": subj_type, "pos": subj_pos, "quat": subj_quat})
                spawned_specs.append(SpawnedObjectSpec(name=subj_name, object_type=subj_type, position=tuple(subj_pos), quaternion_wxyz=tuple(subj_quat), semantic_role="action_subject"))
                all_vis.append(subj_name)
                # Note: subj_name is NOT in obs_cands!

                if is_stop:
                    # Occupant in target
                    occ_name = "occupant1"
                    occ_type = spec.intended_culprit_type or "sugar_box"
                    occ_pos = sample_position_in_target(ref_model, ref_data, rng_base, x_frac=0.0, y_frac=0.0, height_above=0.07).tolist()
                    occ_quat = [1.0, 0.0, 0.0, 0.0]
                    objects_to_spawn_dicts.append({"name": occ_name, "type": occ_type, "pos": occ_pos, "quat": occ_quat})
                    spawned_specs.append(SpawnedObjectSpec(name=occ_name, object_type=occ_type, position=tuple(occ_pos), quaternion_wxyz=tuple(occ_quat), semantic_role="culprit"))
                    obs_cands.append(occ_name)
                    all_vis.append(occ_name)

                # Distractors outside target
                for idx, d_type in enumerate(spec.intended_distractor_types):
                    d_name = f"distractor{idx+1}"
                    offset_x = 0.30 + idx * 0.05
                    offset_y = 0.05 + idx * 0.15
                    d_pos = sample_position_outside_target(ref_model, ref_data, rng_base, offset_x=offset_x, offset_y=offset_y, height_above=0.07).tolist()
                    d_quat = [1.0, 0.0, 0.0, 0.0]
                    objects_to_spawn_dicts.append({"name": d_name, "type": d_type, "pos": d_pos, "quat": d_quat})
                    spawned_specs.append(SpawnedObjectSpec(name=d_name, object_type=d_type, position=tuple(d_pos), quaternion_wxyz=tuple(d_quat), semantic_role="distractor"))
                    obs_cands.append(d_name)
                    all_vis.append(d_name)

            # 1. Build environment
            model, data = self.scene_builder.create_environment(
                objects_to_spawn=objects_to_spawn_dicts,
                settle_steps=0,
                include_robot=True,
                robot_base_pose=rig.robot_base_pose,
                box_pose=list(spec.box_pose) if spec.box_pose else None,
                box_quat=list(spec.box_quat) if spec.box_quat else None,
                target_region_pos=list(spec.target_region_pos) if spec.target_region_pos else None,
                target_region_quat=list(spec.target_region_quat) if spec.target_region_quat else None,
            )

            # 2. Sample and apply BackgroundSpec
            bg_spec = sample_background_spec(spec.background_profile, rng_base, n_lights=model.nlight)
            apply_background_spec(model, bg_spec)

            # 3. Apply task observation rig (sets robot posture, pan/tilt, mj_forward)
            apply_observation_rig(model, data, rig)

            # 4. Evaluate pre F_R on obstruction candidates with held observation rig
            pre_res = evaluate_relational_feasibility(
                model, data, spec.task_id, candidate_objects=obs_cands,
                settle_steps=self.validator.settle_steps, hold_observation_robot=True,
            )

            # 5. Enforce settling and state cardinality
            if not pre_res.settling_succeeded:
                continue

            if is_stop:
                if pre_res.feasible is not False or len(pre_res.active_culprits) != 1:
                    continue
                obs_culprit = pre_res.active_culprits[0]
            else:
                if pre_res.feasible is not True or len(pre_res.active_culprits) != 0:
                    continue
                obs_culprit = None

            # 6. Snapshot canonical state
            snap = snapshot_simulator_state(model, data)

            # 7. Validate instance visibility
            renderer = OffscreenRenderer(model, width=self.resolution[0], height=self.resolution[1], camera_name=rig.camera_name)
            target_geoms = ["B1_lid_panel"] if spec.task_id == "task_1" else ["target_region_geom"]
            req_instances = {name: [f"{name}_visual", f"{name}_geom"] for name in all_vis}
            vis_check = renderer.validate_instance_visibility(
                data, target_geom_names=target_geoms, required_instances=req_instances,
                minimum_pixels=self.min_instance_pixels, min_target_pixels=self.min_relation_target_pixels,
                max_torso_fraction=self.max_torso_fraction,
            )
            renderer.close()

            if not vis_check["is_valid"]:
                continue

            # Accepted base scene!
            accepted_base = True
            resolved_model = model
            resolved_data = data
            resolved_bg_spec = bg_spec
            resolved_objects = spawned_specs
            canonical_snapshot = snap
            pre_result = pre_res
            observed_culprit = obs_culprit
            obstruction_candidates = obs_cands
            all_visible_objects = all_vis
            break

        if not accepted_base or resolved_model is None or resolved_data is None or canonical_snapshot is None or pre_result is None:
            raise RuntimeError(f"Base scene '{spec.scene_id}' failed acceptance after {max_base_retries} attempts.")

        # --- STEP 2: RENDER PRE ARTIFACTS ---
        renderer = OffscreenRenderer(resolved_model, width=self.resolution[0], height=self.resolution[1], camera_name=rig.camera_name)
        
        # Pre RGB
        pre_rgb = renderer.render_rgb(resolved_data)
        pre_rgb_rel_path = f"scenes/{spec.scene_id}/pre_rgb.png"
        Image.fromarray(pre_rgb).save(dataset_root / pre_rgb_rel_path)

        # Pre Segmentation (int32 raw NPY)
        pre_seg_raw = renderer.render_segmentation(resolved_data)
        pre_seg_rel_path = f"scenes/{spec.scene_id}/pre_segmentation.npy"
        np.save(dataset_root / pre_seg_rel_path, pre_seg_raw)

        # Pre Segmentation Viz (8-bit colorized PNG)
        pre_seg_viz = _colorize_segmentation(pre_seg_raw)
        pre_seg_viz_rel_path = f"scenes/{spec.scene_id}/pre_segmentation_viz.png"
        Image.fromarray(pre_seg_viz).save(dataset_root / pre_seg_viz_rel_path)

        # Relation target mask (lid or target surface)
        target_geoms = ["B1_lid_panel"] if spec.task_id == "task_1" else ["target_region_geom"]
        relation_mask = renderer.render_region_mask(resolved_data, target_geoms)
        relation_mask_rel_path = f"scenes/{spec.scene_id}/masks/relation_target_mask.png"
        Image.fromarray(relation_mask).save(dataset_root / relation_mask_rel_path)

        # Object masks and Pre-state crops
        object_mask_paths = {}
        object_crop_paths = {}
        for obj_name in all_visible_objects:
            obj_geoms = [f"{obj_name}_visual", f"{obj_name}_geom"]
            mask = renderer.render_culprit_mask(resolved_data, obj_geoms)
            mask_rel_path = f"scenes/{spec.scene_id}/masks/object_{obj_name}.png"
            Image.fromarray(mask).save(dataset_root / mask_rel_path)
            object_mask_paths[obj_name] = mask_rel_path

            crop_img = _crop_object_from_rgb(pre_rgb, mask)
            if crop_img is not None:
                crop_rel_path = f"scenes/{spec.scene_id}/crops/object_{obj_name}.png"
                Image.fromarray(crop_img).save(dataset_root / crop_rel_path)
                object_crop_paths[obj_name] = crop_rel_path
            else:
                object_crop_paths[obj_name] = None

        cam_meta = renderer.get_camera_metadata(resolved_data)
        renderer.close()

        # Cache canonical initial poses
        canonical_poses = {name: get_body_freejoint_pose(resolved_model, resolved_data, name) for name in all_visible_objects}

        # --- STEP 3: CANDIDATE-SET RETRY LOOP ---
        accepted_candidate_set = False
        accepted_outcomes: List[InterventionOutcome] = []
        realized_candidate_set_attempt = 0
        actual_candidate_set_seed = actual_base_seed

        observed_distractors = [name for name in obstruction_candidates if name != observed_culprit]

        for cand_attempt in range(max_candidate_set_retries):
            realized_candidate_set_attempt = cand_attempt
            actual_candidate_set_seed = actual_base_seed + 100 + cand_attempt * 37
            rng_cand = np.random.default_rng(actual_candidate_set_seed)

            # Generate candidate set
            candidates = self.generator.generate_candidates(
                scene_id=spec.scene_id,
                task_id=spec.task_id,
                model=resolved_model,
                data=resolved_data,
                culprit_names=[observed_culprit] if observed_culprit else [],
                distractor_names=observed_distractors,
                rng=rng_cand,
                is_stop_scene=is_stop,
                min_relocation_m=self.min_relocation_m,
            )

            # Live post-rendering capture closure
            post_rendered_data = {}

            def _post_render_callback(post_model: mujoco.MjModel, post_data: mujoco.MjData, outcome: InterventionOutcome):
                r = OffscreenRenderer(post_model, width=self.resolution[0], height=self.resolution[1], camera_name=rig.camera_name)
                p_rgb = r.render_rgb(post_data)
                p_seg = r.render_segmentation(post_data)
                p_seg_viz = _colorize_segmentation(p_seg)
                r.close()
                post_rendered_data[outcome.intervention.intervention_id] = {
                    "rgb": p_rgb,
                    "seg": p_seg,
                    "seg_viz": p_seg_viz,
                }

            # Evaluate each candidate from canonical state
            outcomes = []
            candidate_set_valid = True

            for interv in candidates:
                outcome = self.validator.evaluate_candidate_from_canonical_state(
                    model=resolved_model,
                    data=resolved_data,
                    canonical_snapshot=canonical_snapshot,
                    pre_result=pre_result,
                    intervention=interv,
                    task_id=spec.task_id,
                    candidate_objects=obstruction_candidates,
                    initial_poses=canonical_poses,
                    on_post_settled_callback=_post_render_callback,
                )

                # Candidate acceptance checks
                if not outcome.validation_diagnostics.get("post_settling_succeeded", False):
                    candidate_set_valid = False
                    break
                if not outcome.intended_effect_matches:
                    candidate_set_valid = False
                    break
                if not outcome.validation_diagnostics.get("collateral_acceptable", True):
                    candidate_set_valid = False
                    break

                # Relational identity audit
                c_before = set(outcome.active_culprits_before)
                c_after = set(outcome.active_culprits_after)
                cat = interv.intended_category

                if is_stop:
                    if cat == PrivilegedCategory.REPAIR:
                        if len(c_after) != 0:
                            candidate_set_valid = False
                            break
                    elif cat == PrivilegedCategory.HARD_NEGATIVE:
                        if observed_culprit not in c_after:
                            candidate_set_valid = False
                            break
                    elif cat == PrivilegedCategory.IRRELEVANT:
                        if observed_culprit not in c_after:
                            candidate_set_valid = False
                            break
                    elif cat == PrivilegedCategory.IDENTITY:
                        if c_before != c_after:
                            candidate_set_valid = False
                            break
                else:
                    if cat == PrivilegedCategory.HARMFUL:
                        if interv.object_name not in c_after or len(c_after) != 1:
                            candidate_set_valid = False
                            break
                    elif cat in (PrivilegedCategory.IRRELEVANT, PrivilegedCategory.IDENTITY):
                        if len(c_after) != 0:
                            candidate_set_valid = False
                            break

                outcomes.append(outcome)

            if candidate_set_valid and len(outcomes) == len(candidates):
                # Candidate set fully accepted! Save post-rendered artifacts to disk
                for interv_id, p_data in post_rendered_data.items():
                    rgb_path = dataset_root / f"scenes/{spec.scene_id}/post/{interv_id}_rgb.png"
                    seg_path = dataset_root / f"scenes/{spec.scene_id}/post/{interv_id}_segmentation.npy"
                    viz_path = dataset_root / f"scenes/{spec.scene_id}/post/{interv_id}_segmentation_viz.png"
                    Image.fromarray(p_data["rgb"]).save(rgb_path)
                    np.save(seg_path, p_data["seg"])
                    Image.fromarray(p_data["seg_viz"]).save(viz_path)

                accepted_outcomes = outcomes
                accepted_candidate_set = True
                break

        if not accepted_candidate_set:
            raise RuntimeError(f"Scene '{spec.scene_id}' failed candidate-set acceptance after {max_candidate_set_retries} attempts.")

        # --- STEP 4: CONSTRUCT RESOLVED SCENE SPEC & MANIFEST RECORDS ---
        canonical_object_poses_dict = {
            name: {
                "position": list(pose.position),
                "quaternion_wxyz": list(pose.quaternion_wxyz),
            }
            for name, pose in canonical_poses.items()
        }

        resolved_spec = ResolvedSceneSpec(
            scene_id=spec.scene_id,
            task_id=spec.task_id,
            intended_base_state=spec.intended_base_state,
            instruction=spec.instruction,
            action=spec.action,
            action_subject_name=spec.action_subject_name,
            obstruction_candidate_names=tuple(obstruction_candidates),
            all_visible_object_names=tuple(all_visible_objects),
            spawned_objects=tuple(resolved_objects),
            background_spec=resolved_bg_spec,
            observation_rig_id=rig.rig_id,
            robot_base_pose=rig.robot_base_pose,
            requested_seed=spec.seed,
            realized_base_attempt=realized_base_attempt,
            actual_base_seed=actual_base_seed,
            canonical_object_poses=canonical_object_poses_dict,
            box_pose=spec.box_pose,
            box_quat=spec.box_quat,
            target_region_pos=spec.target_region_pos,
            target_region_quat=spec.target_region_quat,
        )

        ref_frame_name = "B1_lid_panel" if spec.task_id == "task_1" else "target_region_geom"

        dataset_records: List[InterventionDatasetRecord] = []
        for idx, outcome in enumerate(accepted_outcomes):
            interv = outcome.intervention
            record_id = f"rec_{spec.scene_id}_{idx:03d}"

            # Model inputs
            if interv.operator == InterventionOperator.RELOCATE:
                c_obj_name = interv.object_name
                c_crop_path = object_crop_paths.get(c_obj_name)
                
                curr_pose = canonical_poses[c_obj_name]
                dest_pose = interv.destination_pose

                # Task-relative coordinates
                curr_rel_pos = world_to_local(resolved_model, resolved_data, ref_frame_name, np.array(curr_pose.position), use_geom=True)
                dest_rel_pos = world_to_local(resolved_model, resolved_data, ref_frame_name, np.array(dest_pose.position), use_geom=True)

                curr_geom = GeometryRelativeSpec(
                    world_position=curr_pose.position,
                    world_quaternion_wxyz=curr_pose.quaternion_wxyz,
                    reference_frame=ref_frame_name,
                    relative_position=tuple(float(x) for x in curr_rel_pos),
                ).to_dict()

                dest_geom = GeometryRelativeSpec(
                    world_position=dest_pose.position,
                    world_quaternion_wxyz=dest_pose.quaternion_wxyz,
                    reference_frame=ref_frame_name,
                    relative_position=tuple(float(x) for x in dest_rel_pos),
                ).to_dict()

            else:
                # NONE control
                c_obj_name = None
                c_crop_path = None
                curr_geom = None
                dest_geom = None

            model_inputs = {
                "task_id": spec.task_id,
                "instruction": spec.instruction,
                "action": spec.action.to_dict(),
                "pre_rgb_path": pre_rgb_rel_path,
                "candidate_object_crop_path": c_crop_path,
                "intervention_operator": interv.operator.value,
                "current_geometry": curr_geom,
                "destination_geometry": dest_geom,
                "demonstration_reference": demonstration_reference,
            }

            supervision_targets = {
                "pre_feasible": bool(outcome.pre_feasible),
                "post_feasible": bool(outcome.post_feasible),
                "causal_effect": int(outcome.causal_effect),
            }

            privileged_metadata = {
                "intended_base_state": spec.intended_base_state,
                "intended_category": interv.intended_category.value if interv.intended_category else None,
                "destination_semantic_tag": interv.destination_semantic_tag,
                "intended_effect_matches": bool(outcome.intended_effect_matches),
                "intervention_idx": int(interv.intervention_idx if interv.intervention_idx is not None else idx),
                "candidate_object_name": c_obj_name,
                "candidate_object_type": next((obj.object_type for obj in resolved_objects if obj.name == c_obj_name), None) if c_obj_name else None,
                "is_culprit": bool(c_obj_name == observed_culprit) if c_obj_name else False,
                "active_culprits_before": list(outcome.active_culprits_before),
                "active_culprits_after": list(outcome.active_culprits_after),
                "post_rgb_path": f"scenes/{spec.scene_id}/post/{interv.intervention_id}_rgb.png",
                "pre_segmentation_path": pre_seg_rel_path,
                "post_segmentation_path": f"scenes/{spec.scene_id}/post/{interv.intervention_id}_segmentation.npy",
                "relation_target_mask_path": relation_mask_rel_path,
                "candidate_mask_path": object_mask_paths.get(c_obj_name) if c_obj_name else None,
                "all_scene_objects": [obj.to_dict() for obj in resolved_objects],
                "camera_metadata": cam_meta,
                "background_profile": spec.background_profile,
                "requested_base_seed": int(spec.seed),
                "actual_base_seed": int(actual_base_seed),
                "realized_base_attempt": int(realized_base_attempt),
                "actual_candidate_set_seed": int(actual_candidate_set_seed),
                "realized_candidate_set_attempt": int(realized_candidate_set_attempt),
                "reconstruction_spec": resolved_spec.to_dict(),
            }

            record = InterventionDatasetRecord(
                schema_version=SCHEMA_VERSION,
                record_id=record_id,
                scene_id=spec.scene_id,
                intervention_id=interv.intervention_id,
                model_inputs=model_inputs,
                supervision_targets=supervision_targets,
                privileged_metadata=privileged_metadata,
            )
            dataset_records.append(record)

        return dataset_records, resolved_spec
