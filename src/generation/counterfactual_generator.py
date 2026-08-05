"""
Matched counterfactual scene generator producing paired PROCEED / STOP benchmark query scenes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import PIL.Image as Image

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.validation.occupancy_checks import check_lid_occupancy, check_target_occupancy
from src.generation.scene_config import EpisodeSpec


class CounterfactualPairGenerator:
    """Generator for producing matched PROCEED and STOP query image pairs with explicit masks."""

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

    def _generate_masks_and_visualizations(
        self,
        renderer: OffscreenRenderer,
        data: any,
        candidate_geoms: List[str],
        target_geoms: List[str],
        is_stop: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate all required benchmark masks and overlays."""
        # 1. Instance segmentation
        seg_mask = renderer.render_segmentation(data)
        instance_map = seg_mask[:, :, 0].astype(np.uint8)

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
        # Cyan overlay for relation target
        target_mask_bool = relation_target_mask > 0
        vis[target_mask_bool] = (0.6 * vis[target_mask_bool] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
        # Red overlay for candidate culprit object
        candidate_mask_bool = candidate_mask > 0
        vis[candidate_mask_bool] = (0.5 * vis[candidate_mask_bool] + 0.5 * np.array([255, 50, 50])).astype(np.uint8)

        return instance_map, candidate_mask, relation_target_mask, causal_violation_mask, vis

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

        lid_center = [0.52, 0.18, 0.82]
        pos_offsets = {
            "centre": [0.0, 0.0, 0.0],
            "front_left": [-0.05, -0.03, 0.0],
            "front_right": [0.05, -0.03, 0.0],
            "rear_left": [-0.05, 0.03, 0.0],
            "rear_right": [0.05, 0.03, 0.0],
            "opening_edge": [0.0, -0.04, 0.0],
            "hinge_side": [0.0, 0.04, 0.0],
        }
        offset = pos_offsets.get(blocker_pos_bin, [0.0, 0.0, 0.0])

        stop_objects = [
            {
                "name": "blocker1",
                "type": blocker_type,
                "pos": [lid_center[0] + offset[0], lid_center[1] + offset[1], lid_center[2]],
            }
        ]
        if blocker_count == 2:
            stop_objects.append(
                {
                    "name": "blocker2",
                    "type": "sugar_box" if blocker_type != "sugar_box" else "mug",
                    "pos": [lid_center[0] - offset[0] + 0.04, lid_center[1] - offset[1] - 0.02, lid_center[2]],
                }
            )

        proceed_objects = [
            {
                "name": "blocker1",
                "type": blocker_type,
                "pos": [0.18, 0.18, 0.62], # Moved beside B1 box
            }
        ]
        if blocker_count == 2:
            proceed_objects.append(
                {
                    "name": "blocker2",
                    "type": "sugar_box" if blocker_type != "sugar_box" else "mug",
                    "pos": [0.18, 0.05, 0.62], # Beside box
                }
            )

        # Render STOP scene
        model_stop, data_stop = self.scene_builder.create_environment(stop_objects, settle_steps=20)
        renderer_stop = OffscreenRenderer(model_stop, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_stop = renderer_stop.render_rgb(data_stop)
        candidate_geoms = ["blocker1_geom"] + (["blocker2_geom"] if blocker_count == 2 else [])
        inst_stop, cand_stop, target_stop, causal_stop, vis_stop = self._generate_masks_and_visualizations(
            renderer_stop, data_stop, candidate_geoms, ["B1_lid_panel"], is_stop=True
        )

        blocker_names = ["blocker1"] + (["blocker2"] if blocker_count == 2 else [])
        is_occ_stop, active_culprits_stop = check_lid_occupancy(model_stop, data_stop, blocker_names=blocker_names)
        renderer_stop.close()

        # Render PROCEED scene
        model_proceed, data_proceed = self.scene_builder.create_environment(proceed_objects, settle_steps=20)

        renderer_proceed = OffscreenRenderer(model_proceed, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_proceed = renderer_proceed.render_rgb(data_proceed)
        inst_proceed, cand_proceed, target_proceed, causal_proceed, vis_proceed = self._generate_masks_and_visualizations(
            renderer_proceed, data_proceed, candidate_geoms, ["B1_lid_panel"], is_stop=False
        )

        is_occ_proceed, active_culprits_proceed = check_lid_occupancy(model_proceed, data_proceed, blocker_names=blocker_names)
        renderer_proceed.close()

        # Save query images & masks
        stop_rgb_path = pair_dir / "stop_rgb.png"
        stop_inst_path = pair_dir / "stop_instance_segmentation.png"
        stop_cand_path = pair_dir / "stop_candidate_object_mask.png"
        stop_target_path = pair_dir / "stop_relation_target_mask.png"
        stop_causal_path = pair_dir / "stop_causal_violation_mask.png"
        stop_vis_path = pair_dir / "stop_combined_relation_visualization.png"

        proceed_rgb_path = pair_dir / "proceed_rgb.png"
        proceed_inst_path = pair_dir / "proceed_instance_segmentation.png"
        proceed_cand_path = pair_dir / "proceed_candidate_object_mask.png"
        proceed_target_path = pair_dir / "proceed_relation_target_mask.png"
        proceed_causal_path = pair_dir / "proceed_causal_violation_mask.png"
        proceed_vis_path = pair_dir / "proceed_combined_relation_visualization.png"

        Image.fromarray(rgb_stop).save(stop_rgb_path)
        Image.fromarray(inst_stop).save(stop_inst_path)
        Image.fromarray(cand_stop).save(stop_cand_path)
        Image.fromarray(target_stop).save(stop_target_path)
        Image.fromarray(causal_stop).save(stop_causal_path)
        Image.fromarray(vis_stop).save(stop_vis_path)

        Image.fromarray(rgb_proceed).save(proceed_rgb_path)
        Image.fromarray(inst_proceed).save(proceed_inst_path)
        Image.fromarray(cand_proceed).save(proceed_cand_path)
        Image.fromarray(target_proceed).save(proceed_target_path)
        Image.fromarray(causal_proceed).save(proceed_causal_path)
        Image.fromarray(vis_proceed).save(proceed_vis_path)

        # Legacy backward-compatible file copies
        Image.fromarray(cand_stop).save(pair_dir / "stop_culprit_mask.png")
        Image.fromarray(target_stop).save(pair_dir / "stop_lid_mask.png")
        Image.fromarray(cand_proceed).save(pair_dir / "proceed_culprit_mask.png")
        Image.fromarray(target_proceed).save(pair_dir / "proceed_lid_mask.png")

        stop_spec = EpisodeSpec(
            task_family="task_1",
            sample_id=f"{pair_id}_stop",
            pair_id=pair_id,
            seed=seed,
            label="STOP",
            goal_instruction="Open the box.",
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
            "task_id": "task_1",
            "instruction": "Open the box.",
            "blocker_type": blocker_type,
            "blocker_count": blocker_count,
            "split": split,
            "stop": {
                "label": "STOP",
                "is_occupied": is_occ_stop,
                "culprits": active_culprits_stop,
                "rgb_path": str(stop_rgb_path),
                "instance_segmentation_path": str(stop_inst_path),
                "candidate_object_mask_path": str(stop_cand_path),
                "relation_target_mask_path": str(stop_target_path),
                "causal_violation_mask_path": str(stop_causal_path),
                "combined_visualization_path": str(stop_vis_path),
                "culprit_mask_path": str(pair_dir / "stop_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "stop_lid_mask.png"),
                "spec": stop_spec.to_dict(),
            },
            "proceed": {
                "label": "PROCEED",
                "is_occupied": is_occ_proceed,
                "culprits": active_culprits_proceed,
                "rgb_path": str(proceed_rgb_path),
                "instance_segmentation_path": str(proceed_inst_path),
                "candidate_object_mask_path": str(proceed_cand_path),
                "relation_target_mask_path": str(proceed_target_path),
                "causal_violation_mask_path": str(proceed_causal_path),
                "combined_visualization_path": str(proceed_vis_path),
                "culprit_mask_path": str(pair_dir / "proceed_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "proceed_lid_mask.png"),
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

        target_center = [-0.10, -0.20, 0.65]
        pos_offsets = {
            "centre": [0.0, 0.0, 0.0],
            "left": [-0.04, 0.0, 0.0],
            "right": [0.04, 0.0, 0.0],
            "front": [0.0, -0.04, 0.0],
            "rear": [0.0, 0.04, 0.0],
        }
        offset = pos_offsets.get(occupant_pos_bin, [0.0, 0.0, 0.0])

        stop_objects = [
            {
                "name": "coffee_can",
                "type": "coffee_can",
                "pos": [-0.35, -0.20, 0.65],
            },
            {
                "name": "occupant",
                "type": target_occupant_type,
                "pos": [target_center[0] + offset[0], target_center[1] + offset[1], target_center[2]],
            },
        ]

        proceed_objects = [
            {
                "name": "coffee_can",
                "type": "coffee_can",
                "pos": [-0.35, -0.20, 0.65],
            },
            {
                "name": "occupant",
                "type": target_occupant_type,
                "pos": [0.25, -0.20, 0.65], # Moved outside target_region
            },
        ]

        # Render STOP scene
        model_stop, data_stop = self.scene_builder.create_environment(stop_objects, settle_steps=20)
        renderer_stop = OffscreenRenderer(model_stop, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_stop = renderer_stop.render_rgb(data_stop)
        inst_stop, cand_stop, target_stop, causal_stop, vis_stop = self._generate_masks_and_visualizations(
            renderer_stop, data_stop, ["occupant_geom"], ["target_region_geom"], is_stop=True
        )

        is_occ_stop, active_culprits_stop = check_target_occupancy(
            model_stop, data_stop, candidate_objects=["occupant"]
        )
        renderer_stop.close()

        # Render PROCEED scene
        model_proceed, data_proceed = self.scene_builder.create_environment(proceed_objects, settle_steps=20)

        renderer_proceed = OffscreenRenderer(model_proceed, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_proceed = renderer_proceed.render_rgb(data_proceed)
        inst_proceed, cand_proceed, target_proceed, causal_proceed, vis_proceed = self._generate_masks_and_visualizations(
            renderer_proceed, data_proceed, ["occupant_geom"], ["target_region_geom"], is_stop=False
        )

        is_occ_proceed, active_culprits_proceed = check_target_occupancy(
            model_proceed, data_proceed, candidate_objects=["occupant"]
        )
        renderer_proceed.close()

        # Save query images & masks
        stop_rgb_path = pair_dir / "stop_rgb.png"
        stop_inst_path = pair_dir / "stop_instance_segmentation.png"
        stop_cand_path = pair_dir / "stop_candidate_object_mask.png"
        stop_target_path = pair_dir / "stop_relation_target_mask.png"
        stop_causal_path = pair_dir / "stop_causal_violation_mask.png"
        stop_vis_path = pair_dir / "stop_combined_relation_visualization.png"

        proceed_rgb_path = pair_dir / "proceed_rgb.png"
        proceed_inst_path = pair_dir / "proceed_instance_segmentation.png"
        proceed_cand_path = pair_dir / "proceed_candidate_object_mask.png"
        proceed_target_path = pair_dir / "proceed_relation_target_mask.png"
        proceed_causal_path = pair_dir / "proceed_causal_violation_mask.png"
        proceed_vis_path = pair_dir / "proceed_combined_relation_visualization.png"

        Image.fromarray(rgb_stop).save(stop_rgb_path)
        Image.fromarray(inst_stop).save(stop_inst_path)
        Image.fromarray(cand_stop).save(stop_cand_path)
        Image.fromarray(target_stop).save(stop_target_path)
        Image.fromarray(causal_stop).save(stop_causal_path)
        Image.fromarray(vis_stop).save(stop_vis_path)

        Image.fromarray(rgb_proceed).save(proceed_rgb_path)
        Image.fromarray(inst_proceed).save(proceed_inst_path)
        Image.fromarray(cand_proceed).save(proceed_cand_path)
        Image.fromarray(target_proceed).save(proceed_target_path)
        Image.fromarray(causal_proceed).save(proceed_causal_path)
        Image.fromarray(vis_proceed).save(proceed_vis_path)

        # Legacy backward-compatible file copies
        Image.fromarray(cand_stop).save(pair_dir / "stop_culprit_mask.png")
        Image.fromarray(target_stop).save(pair_dir / "stop_target_mask.png")
        Image.fromarray(cand_proceed).save(pair_dir / "proceed_culprit_mask.png")
        Image.fromarray(target_proceed).save(pair_dir / "proceed_target_mask.png")

        stop_spec = EpisodeSpec(
            task_family="task_2",
            sample_id=f"{pair_id}_stop",
            pair_id=pair_id,
            seed=seed,
            label="STOP",
            goal_instruction="Place object1 in the target region.",
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
            "task_id": "task_2",
            "instruction": "Place object1 in the target region.",
            "target_occupant_type": target_occupant_type,
            "split": split,
            "stop": {
                "label": "STOP",
                "is_occupied": is_occ_stop,
                "culprits": active_culprits_stop,
                "rgb_path": str(stop_rgb_path),
                "instance_segmentation_path": str(stop_inst_path),
                "candidate_object_mask_path": str(stop_cand_path),
                "relation_target_mask_path": str(stop_target_path),
                "causal_violation_mask_path": str(stop_causal_path),
                "combined_visualization_path": str(stop_vis_path),
                "culprit_mask_path": str(pair_dir / "stop_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "stop_target_mask.png"),
                "spec": stop_spec.to_dict(),
            },
            "proceed": {
                "label": "PROCEED",
                "is_occupied": is_occ_proceed,
                "culprits": active_culprits_proceed,
                "rgb_path": str(proceed_rgb_path),
                "instance_segmentation_path": str(proceed_inst_path),
                "candidate_object_mask_path": str(proceed_cand_path),
                "relation_target_mask_path": str(proceed_target_path),
                "causal_violation_mask_path": str(proceed_causal_path),
                "combined_visualization_path": str(proceed_vis_path),
                "culprit_mask_path": str(pair_dir / "proceed_culprit_mask.png"),
                "region_mask_path": str(pair_dir / "proceed_target_mask.png"),
                "spec": proceed_spec.to_dict(),
            },
        }

        with open(pair_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(pair_meta, f, indent=2)

        return pair_meta
