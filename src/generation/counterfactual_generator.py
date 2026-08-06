"""
Matched counterfactual scene generator producing paired PROCEED / STOP query scenes and standalone positive controls.

Uses dynamic scene_utils geometry, SplitPlanner, episode RNG seeding, BackgroundSpec/LightSpec for 100% identical pair lighting,
spec diffs, detailed occupancy measurements, and lossless uint16 instance segmentation.
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
from src.validation.occupancy_checks import check_lid_occupancy, check_target_occupancy
from src.generation.scene_config import EpisodeSpec
from src.generation.background_randomization import (
    sample_background_spec,
    apply_background_spec,
    SPLIT_BACKGROUNDS,
    BackgroundSpec,
)
from src.generation.split_planner import SplitPlanner
from src.environment.scene_utils import (
    get_lid_center,
    get_lid_frame,
    get_target_center,
    get_target_frame,
    sample_position_on_lid,
    sample_position_beside_box,
    sample_position_in_target,
    sample_position_outside_target,
)


def _yaw_quat(yaw: float) -> List[float]:
    """Return quaternion [w, x, y, z] for a Z-axis yaw rotation."""
    half = yaw / 2.0
    return [float(math.cos(half)), 0.0, 0.0, float(math.sin(half))]


class CounterfactualPairGenerator:
    """Generator for producing matched STOP and PROCEED query image pairs with explicit masks."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/queries",
        resolution: Tuple[int, int] = (640, 480),
        camera_name: str = "front_camera",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width, self.height = resolution
        self.camera_name = camera_name
        self.scene_builder = SceneBuilder()
        self.split_planner = SplitPlanner()

    def _generate_masks_and_visualizations(
        self,
        renderer: OffscreenRenderer,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        candidate_geoms: List[str],
        target_geoms: List[str],
        is_stop: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        """Generate all required benchmark masks, lossless uint16 instance maps, and overlays."""
        # 1. Instance segmentation
        seg_mask = renderer.render_segmentation(data)
        geom_id_map = seg_mask[:, :, 0].astype(np.uint16)
        instance_map_8bit = (geom_id_map % 255).astype(np.uint8)

        # Build instance_id -> geom_name dictionary
        unique_ids = np.unique(geom_id_map)
        id_to_name: Dict[int, str] = {}
        for gid in unique_ids:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"
            id_to_name[int(gid)] = gname

        # 2. Candidate object mask (present in both STOP and PROCEED)
        candidate_mask = renderer.render_culprit_mask(data, candidate_geoms)

        # 3. Relation target mask (lid for Task 1, target region for Task 2; present in both)
        relation_target_mask = renderer.render_region_mask(data, target_geoms)

        # 4. Causal violation mask (union for STOP, all zeros for PROCEED)
        if is_stop:
            causal_violation_mask = np.maximum(candidate_mask, relation_target_mask)
        else:
            causal_violation_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # 5. Combined relation visualization overlay
        rgb = renderer.render_rgb(data)
        vis = rgb.copy()
        target_mask_bool = relation_target_mask > 0
        vis[target_mask_bool] = (0.6 * vis[target_mask_bool] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
        candidate_mask_bool = candidate_mask > 0
        vis[candidate_mask_bool] = (0.5 * vis[candidate_mask_bool] + 0.5 * np.array([255, 50, 50])).astype(np.uint8)

        return instance_map_8bit, geom_id_map, candidate_mask, relation_target_mask, causal_violation_mask, vis, id_to_name

    def generate_task1_pair(
        self,
        pair_id: str,
        blocker_type: str = "coffee_can",
        blocker_count: int = 1,
        blocker_pos_bin: str = "centre",
        split: str = "id",
        seed: int = 42,
    ) -> Dict[str, dict]:
        """Generate matched STOP/PROCEED counterfactual pair for Task 1 (Open Box)."""
        pair_dir = self.output_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        ref_model, ref_data = self.scene_builder.create_environment(settle_steps=0)
        lid_center = get_lid_center(ref_model, ref_data).tolist()

        bg_profile_name = SPLIT_BACKGROUNDS.get(split, "bg_neutral_wood")
        # Sample background and light spec ONCE per pair
        bg_spec = sample_background_spec(bg_profile_name, rng, n_lights=ref_model.nlight)

        pos_offsets = {
            "centre": [0.0, 0.0, 0.08],
            "front_left": [-0.05, -0.03, 0.08],
            "front_right": [0.05, -0.03, 0.08],
            "rear_left": [-0.05, 0.03, 0.08],
            "rear_right": [0.05, 0.03, 0.08],
            "opening_edge": [0.0, -0.04, 0.08],
            "hinge_side": [0.0, 0.04, 0.08],
        }
        base_offset = pos_offsets.get(blocker_pos_bin, [0.0, 0.0, 0.08])
        jitter_xy = rng.uniform(-0.015, 0.015, size=2)

        stop_b1_pos = [
            lid_center[0] + base_offset[0] + jitter_xy[0],
            lid_center[1] + base_offset[1] + jitter_xy[1],
            lid_center[2] + base_offset[2],
        ]
        stop_b1_quat = _yaw_quat(float(rng.uniform(-math.pi, math.pi)))

        stop_objects = [{"name": "blocker1", "type": blocker_type, "pos": stop_b1_pos, "quat": stop_b1_quat}]
        if blocker_count == 2:
            stop_b2_pos = [
                lid_center[0] - base_offset[0] + 0.04,
                lid_center[1] - base_offset[1] - 0.02,
                lid_center[2] + base_offset[2],
            ]
            stop_objects.append(
                {"name": "blocker2", "type": "sugar_box" if blocker_type != "sugar_box" else "mug", "pos": stop_b2_pos}
            )

        # PROCEED objects: placed beside the box (off the lid)
        proc_b1_pos = [lid_center[0] - 0.30, lid_center[1] - 0.15 + jitter_xy[1], 0.65]
        proceed_objects = [{"name": "blocker1", "type": blocker_type, "pos": proc_b1_pos, "quat": stop_b1_quat}]
        if blocker_count == 2:
            proceed_objects.append(
                {"name": "blocker2", "type": "sugar_box" if blocker_type != "sugar_box" else "mug", "pos": [lid_center[0] - 0.30, lid_center[1] + 0.05, 0.65]}
            )

        # ── Render STOP scene ──────────────────────────────────────────
        model_stop, data_stop = self.scene_builder.create_environment(stop_objects, settle_steps=100)
        apply_background_spec(model_stop, bg_spec)
        mujoco.mj_forward(model_stop, data_stop)

        renderer_stop = OffscreenRenderer(model_stop, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_stop = renderer_stop.render_rgb(data_stop)
        candidate_geoms = ["blocker1_geom"] + (["blocker2_geom"] if blocker_count == 2 else [])
        inst_stop_8, inst_stop_16, cand_stop, target_stop, causal_stop, vis_stop, id_map_stop = (
            self._generate_masks_and_visualizations(
                renderer_stop, model_stop, data_stop, candidate_geoms, ["B1_lid_panel"], is_stop=True
            )
        )

        blocker_names = ["blocker1"] + (["blocker2"] if blocker_count == 2 else [])
        is_occ_stop, active_culprits_stop, measurements_stop = check_lid_occupancy(
            model_stop, data_stop, blocker_names=blocker_names
        )
        renderer_stop.close()

        # ── Render PROCEED scene ───────────────────────────────────────
        model_proceed, data_proceed = self.scene_builder.create_environment(proceed_objects, settle_steps=100)
        apply_background_spec(model_proceed, bg_spec)
        mujoco.mj_forward(model_proceed, data_proceed)

        renderer_proceed = OffscreenRenderer(model_proceed, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_proceed = renderer_proceed.render_rgb(data_proceed)
        inst_proceed_8, inst_proceed_16, cand_proceed, target_proceed, causal_proceed, vis_proceed, id_map_proceed = (
            self._generate_masks_and_visualizations(
                renderer_proceed, model_proceed, data_proceed, candidate_geoms, ["B1_lid_panel"], is_stop=False
            )
        )

        is_occ_proceed, active_culprits_proceed, measurements_proceed = check_lid_occupancy(
            model_proceed, data_proceed, blocker_names=blocker_names
        )
        renderer_proceed.close()

        # Save query images & masks
        stop_rgb_path = pair_dir / "stop_rgb.png"
        stop_inst_path = pair_dir / "stop_instance_segmentation.png"
        stop_inst_npy = pair_dir / "stop_instance_uint16.npy"
        stop_cand_path = pair_dir / "stop_candidate_object_mask.png"
        stop_target_path = pair_dir / "stop_relation_target_mask.png"
        stop_causal_path = pair_dir / "stop_causal_violation_mask.png"
        stop_vis_path = pair_dir / "stop_combined_relation_visualization.png"

        proceed_rgb_path = pair_dir / "proceed_rgb.png"
        proceed_inst_path = pair_dir / "proceed_instance_segmentation.png"
        proceed_inst_npy = pair_dir / "proceed_instance_uint16.npy"
        proceed_cand_path = pair_dir / "proceed_candidate_object_mask.png"
        proceed_target_path = pair_dir / "proceed_relation_target_mask.png"
        proceed_causal_path = pair_dir / "proceed_causal_violation_mask.png"
        proceed_vis_path = pair_dir / "proceed_combined_relation_visualization.png"

        Image.fromarray(rgb_stop).save(stop_rgb_path)
        Image.fromarray(inst_stop_8).save(stop_inst_path)
        np.save(stop_inst_npy, inst_stop_16)
        Image.fromarray(cand_stop).save(stop_cand_path)
        Image.fromarray(target_stop).save(stop_target_path)
        Image.fromarray(causal_stop).save(stop_causal_path)
        Image.fromarray(vis_stop).save(stop_vis_path)

        Image.fromarray(rgb_proceed).save(proceed_rgb_path)
        Image.fromarray(inst_proceed_8).save(proceed_inst_path)
        # FIX Phase 3 uint16 saving bug: Save inst_proceed_16
        np.save(proceed_inst_npy, inst_proceed_16)
        Image.fromarray(cand_proceed).save(proceed_cand_path)
        Image.fromarray(target_proceed).save(proceed_target_path)
        Image.fromarray(causal_proceed).save(proceed_causal_path)
        Image.fromarray(vis_proceed).save(proceed_vis_path)

        # Legacy backward-compatible file copies
        Image.fromarray(cand_stop).save(pair_dir / "stop_culprit_mask.png")
        Image.fromarray(target_stop).save(pair_dir / "stop_lid_mask.png")
        Image.fromarray(cand_proceed).save(pair_dir / "proceed_culprit_mask.png")
        Image.fromarray(target_proceed).save(pair_dir / "proceed_lid_mask.png")

        intervention_vars = {
            "blocker1_pos": {"stop": stop_b1_pos, "proceed": proc_b1_pos},
        }
        invariant_vars = {
            "pair_id": pair_id,
            "task_id": "task_1",
            "instruction": "Open the box.",
            "blocker_type": blocker_type,
            "blocker_count": blocker_count,
            "blocker_pos_bin": blocker_pos_bin,
            "split": split,
            "background_id": bg_profile_name,
            "background_spec": bg_spec.to_dict(),
            "seed": seed,
            "lid_center": lid_center,
            "camera_name": self.camera_name,
            "resolution": [self.width, self.height],
        }
        spec_diff = {
            "intervention_variables": intervention_vars,
            "invariant_variables": invariant_vars,
            "observed_diff": ["blocker1_pos"],
        }

        stop_spec = EpisodeSpec(
            task_family="task_1",
            sample_id=f"{pair_id}_stop",
            pair_id=pair_id,
            seed=seed,
            label="STOP",
            goal_instruction="Open the box.",
            background_id=bg_profile_name,
            blocker_or_occupant_types=[blocker_type],
            blocker_count=blocker_count,
            blocker_position_bins=[blocker_pos_bin],
            split=split,
            relation_before=f"ON_TOP_OF({blocker_type}, B1_lid)",
            relation_after=f"BESIDE({blocker_type}, box_B1)",
            objects_to_spawn=stop_objects,
        )

        proceed_spec = EpisodeSpec(
            task_family="task_1",
            sample_id=f"{pair_id}_proceed",
            pair_id=pair_id,
            seed=seed,
            label="PROCEED",
            goal_instruction="Open the box.",
            background_id=bg_profile_name,
            blocker_or_occupant_types=[blocker_type],
            blocker_count=blocker_count,
            blocker_position_bins=[blocker_pos_bin],
            split=split,
            relation_before=f"ON_TOP_OF({blocker_type}, B1_lid)",
            relation_after=f"BESIDE({blocker_type}, box_B1)",
            objects_to_spawn=proceed_objects,
        )

        pair_meta = {
            "pair_id": pair_id,
            "sample_type": "matched_pair",
            "task_id": "task_1",
            "instruction": "Open the box.",
            "blocker_type": blocker_type,
            "blocker_count": blocker_count,
            "blocker_pos_bin": blocker_pos_bin,
            "split": split,
            "seed": seed,
            "background_id": bg_profile_name,
            "background_spec": bg_spec.to_dict(),
            "spec_diff": spec_diff,
            "stop": {
                "label": "STOP",
                "is_occupied": is_occ_stop,
                "culprits": active_culprits_stop,
                "measurements": measurements_stop,
                "rgb_path": str(stop_rgb_path),
                "instance_segmentation_path": str(stop_inst_path),
                "instance_uint16_path": str(stop_inst_npy),
                "candidate_object_mask_path": str(stop_cand_path),
                "relation_target_mask_path": str(stop_target_path),
                "causal_violation_mask_path": str(stop_causal_path),
                "combined_visualization_path": str(stop_vis_path),
                "culprit_mask_path": str(pair_dir / "stop_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "stop_lid_mask.png"),
                "instance_id_to_name_map": id_map_stop,
                "spec": stop_spec.to_dict(),
            },
            "proceed": {
                "label": "PROCEED",
                "is_occupied": is_occ_proceed,
                "culprits": active_culprits_proceed,
                "measurements": measurements_proceed,
                "rgb_path": str(proceed_rgb_path),
                "instance_segmentation_path": str(proceed_inst_path),
                "instance_uint16_path": str(proceed_inst_npy),
                "candidate_object_mask_path": str(proceed_cand_path),
                "relation_target_mask_path": str(proceed_target_path),
                "causal_violation_mask_path": str(proceed_causal_path),
                "combined_visualization_path": str(proceed_vis_path),
                "culprit_mask_path": str(pair_dir / "proceed_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "proceed_lid_mask.png"),
                "instance_id_to_name_map": id_map_proceed,
                "spec": proceed_spec.to_dict(),
            },
        }

        with open(pair_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(pair_meta, f, indent=2)

        return pair_meta

    def generate_task2_pair(
        self,
        pair_id: str,
        target_occupant_type: str = "sugar_box",
        occupant_pos_bin: str = "centre",
        split: str = "id",
        seed: int = 42,
    ) -> Dict[str, dict]:
        """Generate matched STOP/PROCEED counterfactual pair for Task 2 (Place Object)."""
        pair_dir = self.output_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        ref_model, ref_data = self.scene_builder.create_environment(settle_steps=0)
        target_center = get_target_center(ref_model, ref_data).tolist()

        bg_profile_name = SPLIT_BACKGROUNDS.get(split, "bg_neutral_wood")
        bg_spec = sample_background_spec(bg_profile_name, rng, n_lights=ref_model.nlight)

        pos_offsets = {
            "centre": [0.0, 0.0, 0.08],
            "left": [-0.04, 0.0, 0.08],
            "right": [0.04, 0.0, 0.08],
            "front": [0.0, -0.04, 0.08],
            "rear": [0.0, 0.04, 0.08],
        }
        base_offset = pos_offsets.get(occupant_pos_bin, [0.0, 0.0, 0.08])
        jitter_xy = rng.uniform(-0.015, 0.015, size=2)

        stop_occ_pos = [
            target_center[0] + base_offset[0] + jitter_xy[0],
            target_center[1] + base_offset[1] + jitter_xy[1],
            target_center[2] + base_offset[2],
        ]
        stop_occ_quat = _yaw_quat(float(rng.uniform(-math.pi, math.pi)))

        proc_occ_pos = [target_center[0] + 0.30, target_center[1] + jitter_xy[1], 0.65]

        stop_objects = [
            {"name": "coffee_can", "type": "coffee_can", "pos": [-0.30, -0.20, 0.65]},
            {"name": "occupant", "type": target_occupant_type, "pos": stop_occ_pos, "quat": stop_occ_quat},
        ]

        proceed_objects = [
            {"name": "coffee_can", "type": "coffee_can", "pos": [-0.30, -0.20, 0.65]},
            {"name": "occupant", "type": target_occupant_type, "pos": proc_occ_pos, "quat": stop_occ_quat},
        ]

        # ── Render STOP scene ──────────────────────────────────────────
        model_stop, data_stop = self.scene_builder.create_environment(stop_objects, settle_steps=100)
        apply_background_spec(model_stop, bg_spec)
        mujoco.mj_forward(model_stop, data_stop)

        renderer_stop = OffscreenRenderer(model_stop, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_stop = renderer_stop.render_rgb(data_stop)
        inst_stop_8, inst_stop_16, cand_stop, target_stop, causal_stop, vis_stop, id_map_stop = (
            self._generate_masks_and_visualizations(
                renderer_stop, model_stop, data_stop, ["occupant_geom"], ["target_region_geom"], is_stop=True
            )
        )

        is_occ_stop, active_culprits_stop, measurements_stop = check_target_occupancy(
            model_stop, data_stop, candidate_objects=["occupant"]
        )
        renderer_stop.close()

        # ── Render PROCEED scene ───────────────────────────────────────
        model_proceed, data_proceed = self.scene_builder.create_environment(proceed_objects, settle_steps=100)
        apply_background_spec(model_proceed, bg_spec)
        mujoco.mj_forward(model_proceed, data_proceed)

        renderer_proceed = OffscreenRenderer(model_proceed, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_proceed = renderer_proceed.render_rgb(data_proceed)
        inst_proceed_8, inst_proceed_16, cand_proceed, target_proceed, causal_proceed, vis_proceed, id_map_proceed = (
            self._generate_masks_and_visualizations(
                renderer_proceed, model_proceed, data_proceed, ["occupant_geom"], ["target_region_geom"], is_stop=False
            )
        )

        is_occ_proceed, active_culprits_proceed, measurements_proceed = check_target_occupancy(
            model_proceed, data_proceed, candidate_objects=["occupant"]
        )
        renderer_proceed.close()

        # Save query images & masks
        stop_rgb_path = pair_dir / "stop_rgb.png"
        stop_inst_path = pair_dir / "stop_instance_segmentation.png"
        stop_inst_npy = pair_dir / "stop_instance_uint16.npy"
        stop_cand_path = pair_dir / "stop_candidate_object_mask.png"
        stop_target_path = pair_dir / "stop_relation_target_mask.png"
        stop_causal_path = pair_dir / "stop_causal_violation_mask.png"
        stop_vis_path = pair_dir / "stop_combined_relation_visualization.png"

        proceed_rgb_path = pair_dir / "proceed_rgb.png"
        proceed_inst_path = pair_dir / "proceed_instance_segmentation.png"
        proceed_inst_npy = pair_dir / "proceed_instance_uint16.npy"
        proceed_cand_path = pair_dir / "proceed_candidate_object_mask.png"
        proceed_target_path = pair_dir / "proceed_relation_target_mask.png"
        proceed_causal_path = pair_dir / "proceed_causal_violation_mask.png"
        proceed_vis_path = pair_dir / "proceed_combined_relation_visualization.png"

        Image.fromarray(rgb_stop).save(stop_rgb_path)
        Image.fromarray(inst_stop_8).save(stop_inst_path)
        np.save(stop_inst_npy, inst_stop_16)
        Image.fromarray(cand_stop).save(stop_cand_path)
        Image.fromarray(target_stop).save(stop_target_path)
        Image.fromarray(causal_stop).save(stop_causal_path)
        Image.fromarray(vis_stop).save(stop_vis_path)

        Image.fromarray(rgb_proceed).save(proceed_rgb_path)
        Image.fromarray(inst_proceed_8).save(proceed_inst_path)
        # FIX Phase 3 uint16 saving bug: Save inst_proceed_16
        np.save(proceed_inst_npy, inst_proceed_16)
        Image.fromarray(cand_proceed).save(proceed_cand_path)
        Image.fromarray(target_proceed).save(proceed_target_path)
        Image.fromarray(causal_proceed).save(proceed_causal_path)
        Image.fromarray(vis_proceed).save(proceed_vis_path)

        # Legacy backward-compatible file copies
        Image.fromarray(cand_stop).save(pair_dir / "stop_culprit_mask.png")
        Image.fromarray(target_stop).save(pair_dir / "stop_target_mask.png")
        Image.fromarray(cand_proceed).save(pair_dir / "proceed_culprit_mask.png")
        Image.fromarray(target_proceed).save(pair_dir / "proceed_target_mask.png")

        intervention_vars = {
            "occupant_pos": {"stop": stop_occ_pos, "proceed": proc_occ_pos},
        }
        invariant_vars = {
            "pair_id": pair_id,
            "task_id": "task_2",
            "instruction": "Place object1 in the target region.",
            "target_occupant_type": target_occupant_type,
            "occupant_pos_bin": occupant_pos_bin,
            "split": split,
            "background_id": bg_profile_name,
            "background_spec": bg_spec.to_dict(),
            "seed": seed,
            "target_center": target_center,
            "camera_name": self.camera_name,
            "resolution": [self.width, self.height],
        }
        spec_diff = {
            "intervention_variables": intervention_vars,
            "invariant_variables": invariant_vars,
            "observed_diff": ["occupant_pos"],
        }

        stop_spec = EpisodeSpec(
            task_family="task_2",
            sample_id=f"{pair_id}_stop",
            pair_id=pair_id,
            seed=seed,
            label="STOP",
            goal_instruction="Place object1 in the target region.",
            background_id=bg_profile_name,
            object1_type="coffee_can",
            blocker_or_occupant_types=[target_occupant_type],
            occupant_position_bin=occupant_pos_bin,
            split=split,
            relation_before=f"OCCUPIES({target_occupant_type}, target_region)",
            relation_after=f"OUTSIDE({target_occupant_type}, target_region)",
            objects_to_spawn=stop_objects,
        )

        proceed_spec = EpisodeSpec(
            task_family="task_2",
            sample_id=f"{pair_id}_proceed",
            pair_id=pair_id,
            seed=seed,
            label="PROCEED",
            goal_instruction="Place object1 in the target region.",
            background_id=bg_profile_name,
            object1_type="coffee_can",
            blocker_or_occupant_types=[target_occupant_type],
            occupant_position_bin=occupant_pos_bin,
            split=split,
            relation_before=f"OCCUPIES({target_occupant_type}, target_region)",
            relation_after=f"OUTSIDE({target_occupant_type}, target_region)",
            objects_to_spawn=proceed_objects,
        )

        pair_meta = {
            "pair_id": pair_id,
            "sample_type": "matched_pair",
            "task_id": "task_2",
            "instruction": "Place object1 in the target region.",
            "target_occupant_type": target_occupant_type,
            "occupant_pos_bin": occupant_pos_bin,
            "split": split,
            "seed": seed,
            "background_id": bg_profile_name,
            "background_spec": bg_spec.to_dict(),
            "spec_diff": spec_diff,
            "stop": {
                "label": "STOP",
                "is_occupied": is_occ_stop,
                "culprits": active_culprits_stop,
                "measurements": measurements_stop,
                "rgb_path": str(stop_rgb_path),
                "instance_segmentation_path": str(stop_inst_path),
                "instance_uint16_path": str(stop_inst_npy),
                "candidate_object_mask_path": str(stop_cand_path),
                "relation_target_mask_path": str(stop_target_path),
                "causal_violation_mask_path": str(stop_causal_path),
                "combined_visualization_path": str(stop_vis_path),
                "culprit_mask_path": str(pair_dir / "stop_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "stop_target_mask.png"),
                "instance_id_to_name_map": id_map_stop,
                "spec": stop_spec.to_dict(),
            },
            "proceed": {
                "label": "PROCEED",
                "is_occupied": is_occ_proceed,
                "culprits": active_culprits_proceed,
                "measurements": measurements_proceed,
                "rgb_path": str(proceed_rgb_path),
                "instance_segmentation_path": str(proceed_inst_path),
                "instance_uint16_path": str(proceed_inst_npy),
                "candidate_object_mask_path": str(proceed_cand_path),
                "relation_target_mask_path": str(proceed_target_path),
                "causal_violation_mask_path": str(proceed_causal_path),
                "combined_visualization_path": str(proceed_vis_path),
                "culprit_mask_path": str(pair_dir / "proceed_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "proceed_target_mask.png"),
                "instance_id_to_name_map": id_map_proceed,
                "spec": proceed_spec.to_dict(),
            },
        }

        with open(pair_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(pair_meta, f, indent=2)

        return pair_meta

    def generate_task1_control(
        self,
        control_id: str,
        object_type: Optional[str] = None,
        split: str = "id",
        seed: int = 42,
    ) -> Dict[str, dict]:
        """Generate standalone positive control for Task 1 (Open Box)."""
        ctrl_dir = self.output_dir / control_id
        ctrl_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        ref_model, ref_data = self.scene_builder.create_environment(settle_steps=0)
        lid_center = get_lid_center(ref_model, ref_data).tolist()
        bg_profile_name = SPLIT_BACKGROUNDS.get(split, "bg_neutral_wood")
        bg_spec = sample_background_spec(bg_profile_name, rng, n_lights=ref_model.nlight)

        objects = []
        if object_type:
            objects.append({"name": "blocker1", "type": object_type, "pos": [lid_center[0] - 0.30, lid_center[1] - 0.15, 0.65]})

        model, data = self.scene_builder.create_environment(objects, settle_steps=50)
        apply_background_spec(model, bg_spec)
        mujoco.mj_forward(model, data)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height, camera_name=self.camera_name)
        rgb = renderer.render_rgb(data)

        candidate_geoms = ["blocker1_geom"] if object_type else []
        inst_8, inst_16, cand, target, causal, vis, id_map = self._generate_masks_and_visualizations(
            renderer, model, data, candidate_geoms, ["B1_lid_panel"], is_stop=False
        )

        is_occ, active_culprits, measurements = check_lid_occupancy(
            model, data, blocker_names=["blocker1"] if object_type else []
        )
        renderer.close()

        rgb_path = ctrl_dir / "control_rgb.png"
        inst_path = ctrl_dir / "control_instance_segmentation.png"
        inst_npy = ctrl_dir / "control_instance_uint16.npy"
        cand_path = ctrl_dir / "control_candidate_object_mask.png"
        target_path = ctrl_dir / "control_relation_target_mask.png"

        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(inst_8).save(inst_path)
        np.save(inst_npy, inst_16)
        Image.fromarray(cand).save(cand_path)
        Image.fromarray(target).save(target_path)

        meta = {
            "control_id": control_id,
            "sample_type": "positive_control",
            "task_id": "task_1",
            "instruction": "Open the box.",
            "label": "PROCEED",
            "split": split,
            "seed": seed,
            "background_id": bg_profile_name,
            "background_spec": bg_spec.to_dict(),
            "is_occupied": is_occ,
            "culprits": active_culprits,
            "measurements": measurements,
            "rgb_path": str(rgb_path),
            "instance_segmentation_path": str(inst_path),
            "instance_uint16_path": str(inst_npy),
            "candidate_object_mask_path": str(cand_path),
            "relation_target_mask_path": str(target_path),
            "instance_id_to_name_map": id_map,
        }

        with open(ctrl_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    def generate_task2_control(
        self,
        control_id: str,
        occupant_type: Optional[str] = None,
        split: str = "id",
        seed: int = 42,
    ) -> Dict[str, dict]:
        """Generate standalone positive control for Task 2 (Place Object)."""
        ctrl_dir = self.output_dir / control_id
        ctrl_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        ref_model, ref_data = self.scene_builder.create_environment(settle_steps=0)
        target_center = get_target_center(ref_model, ref_data).tolist()
        bg_profile_name = SPLIT_BACKGROUNDS.get(split, "bg_neutral_wood")
        bg_spec = sample_background_spec(bg_profile_name, rng, n_lights=ref_model.nlight)

        objects = [{"name": "coffee_can", "type": "coffee_can", "pos": [-0.30, -0.20, 0.65]}]
        if occupant_type:
            objects.append({"name": "occupant", "type": occupant_type, "pos": [target_center[0] + 0.30, target_center[1], 0.65]})

        model, data = self.scene_builder.create_environment(objects, settle_steps=50)
        apply_background_spec(model, bg_spec)
        mujoco.mj_forward(model, data)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height, camera_name=self.camera_name)
        rgb = renderer.render_rgb(data)

        candidate_geoms = ["occupant_geom"] if occupant_type else []
        inst_8, inst_16, cand, target, causal, vis, id_map = self._generate_masks_and_visualizations(
            renderer, model, data, candidate_geoms, ["target_region_geom"], is_stop=False
        )

        is_occ, active_culprits, measurements = check_target_occupancy(
            model, data, candidate_objects=["occupant"] if occupant_type else []
        )
        renderer.close()

        rgb_path = ctrl_dir / "control_rgb.png"
        inst_path = ctrl_dir / "control_instance_segmentation.png"
        inst_npy = ctrl_dir / "control_instance_uint16.npy"
        cand_path = ctrl_dir / "control_candidate_object_mask.png"
        target_path = ctrl_dir / "control_relation_target_mask.png"

        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(inst_8).save(inst_path)
        np.save(inst_npy, inst_16)
        Image.fromarray(cand).save(cand_path)
        Image.fromarray(target).save(target_path)

        meta = {
            "control_id": control_id,
            "sample_type": "positive_control",
            "task_id": "task_2",
            "instruction": "Place object1 in the target region.",
            "label": "PROCEED",
            "split": split,
            "seed": seed,
            "background_id": bg_profile_name,
            "background_spec": bg_spec.to_dict(),
            "is_occupied": is_occ,
            "culprits": active_culprits,
            "measurements": measurements,
            "rgb_path": str(rgb_path),
            "instance_segmentation_path": str(inst_path),
            "instance_uint16_path": str(inst_npy),
            "candidate_object_mask_path": str(cand_path),
            "relation_target_mask_path": str(target_path),
            "instance_id_to_name_map": id_map,
        }

        with open(ctrl_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta


def regenerate_from_metadata(
    meta: Dict[str, Any],
    output_dir: Union[str, Path],
    resolution: Tuple[int, int] = (640, 480),
) -> Dict[str, Any]:
    """Reconstruct exact scene RGB, masks, and privileged labels deterministically from saved metadata."""
    gen = CounterfactualPairGenerator(output_dir=output_dir, resolution=resolution)

    task_id = meta.get("task_id")
    sample_type = meta.get("sample_type", "matched_pair")
    split = meta.get("split", "id")
    seed = meta.get("seed", 42)

    if sample_type == "matched_pair":
        pair_id = meta.get("pair_id", "regen_pair")
        if task_id == "task_1":
            return gen.generate_task1_pair(
                pair_id=pair_id,
                blocker_type=meta.get("blocker_type", "coffee_can"),
                blocker_count=meta.get("blocker_count", 1),
                blocker_pos_bin=meta.get("blocker_pos_bin", "centre"),
                split=split,
                seed=seed,
            )
        else:
            return gen.generate_task2_pair(
                pair_id=pair_id,
                target_occupant_type=meta.get("target_occupant_type", "sugar_box"),
                occupant_pos_bin=meta.get("occupant_pos_bin", "centre"),
                split=split,
                seed=seed,
            )
    else:
        control_id = meta.get("control_id", "regen_control")
        if task_id == "task_1":
            return gen.generate_task1_control(control_id=control_id, split=split, seed=seed)
        else:
            return gen.generate_task2_control(control_id=control_id, split=split, seed=seed)
