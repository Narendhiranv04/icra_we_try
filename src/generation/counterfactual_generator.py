"""
Matched counterfactual scene generator producing paired PROCEED / STOP benchmark query scenes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import cv2
import numpy as np
import PIL.Image as Image

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.validation.occupancy_checks import check_lid_occupancy, check_target_occupancy


class CounterfactualPairGenerator:
    """Generator for producing matched PROCEED and STOP query image pairs with minimal intervention."""

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

    def generate_task1_pair(
        self,
        pair_id: str,
        blocker_type: str = "coffee_can",
    ) -> Dict[str, dict]:
        """Generate matched STOP/PROCEED counterfactual pair for Task 1 (Open Box).
        
        STOP condition: blocker object on B1_lid.
        PROCEED condition: minimal intervention - move blocker object beside B1 box.
        
        Returns:
            Dict containing metadata for both PROCEED and STOP samples.
        """
        pair_dir = self.output_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build STOP Scene (Blocker on B1_lid)
        stop_objects = [
            {
                "name": "blocker1",
                "type": blocker_type,
                "pos": [0.52, 0.18, 0.82], # Placed on B1_lid
            }
        ]
        model_stop, data_stop = self.scene_builder.create_environment(stop_objects, settle_steps=100)
        renderer_stop = OffscreenRenderer(model_stop, width=self.width, height=self.height, camera_name=self.camera_name)
        
        rgb_stop = renderer_stop.render_rgb(data_stop)
        culprit_mask_stop = renderer_stop.render_culprit_mask(data_stop, ["blocker1_geom"])
        region_mask_stop = renderer_stop.render_region_mask(data_stop, ["B1_lid_panel"])
        
        is_occ_stop, active_culprits_stop = check_lid_occupancy(model_stop, data_stop, blocker_names=["blocker1"])
        renderer_stop.close()

        # 2. Build PROCEED Scene (Minimal intervention: move blocker beside box)
        proceed_objects = [
            {
                "name": "blocker1",
                "type": blocker_type,
                "pos": [0.15, 0.18, 0.62], # Moved beside B1 box
            }
        ]
        model_proceed, data_proceed = self.scene_builder.create_environment(proceed_objects, settle_steps=100)
        renderer_proceed = OffscreenRenderer(model_proceed, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_proceed = renderer_proceed.render_rgb(data_proceed)
        culprit_mask_proceed = renderer_proceed.render_culprit_mask(data_proceed, ["blocker1_geom"])
        region_mask_proceed = renderer_proceed.render_region_mask(data_proceed, ["B1_lid_panel"])
        
        is_occ_proceed, active_culprits_proceed = check_lid_occupancy(model_proceed, data_proceed, blocker_names=["blocker1"])
        renderer_proceed.close()

        # Save query images & masks
        stop_rgb_path = pair_dir / "stop_rgb.png"
        stop_mask_path = pair_dir / "stop_culprit_mask.png"
        stop_lid_mask_path = pair_dir / "stop_lid_mask.png"

        proceed_rgb_path = pair_dir / "proceed_rgb.png"
        proceed_mask_path = pair_dir / "proceed_culprit_mask.png"
        proceed_lid_mask_path = pair_dir / "proceed_lid_mask.png"

        Image.fromarray(rgb_stop).save(stop_rgb_path)
        Image.fromarray(culprit_mask_stop).save(stop_mask_path)
        Image.fromarray(region_mask_stop).save(stop_lid_mask_path)

        Image.fromarray(rgb_proceed).save(proceed_rgb_path)
        Image.fromarray(culprit_mask_proceed).save(proceed_mask_path)
        Image.fromarray(region_mask_proceed).save(proceed_lid_mask_path)

        pair_meta = {
            "pair_id": pair_id,
            "task_id": "task_1",
            "instruction": "Open the box.",
            "blocker_type": blocker_type,
            "stop": {
                "label": "STOP",
                "is_occupied": is_occ_stop,
                "culprits": active_culprits_stop,
                "rgb_path": str(stop_rgb_path),
                "culprit_mask_path": str(stop_mask_path),
                "region_mask_path": str(stop_lid_mask_path),
            },
            "proceed": {
                "label": "PROCEED",
                "is_occupied": is_occ_proceed,
                "culprits": active_culprits_proceed,
                "rgb_path": str(proceed_rgb_path),
                "culprit_mask_path": str(proceed_mask_path),
                "region_mask_path": str(proceed_lid_mask_path),
            },
        }

        with open(pair_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(pair_meta, f, indent=2)

        return pair_meta

    def generate_task2_pair(
        self,
        pair_id: str,
        target_occupant_type: str = "sugar_box",
    ) -> Dict[str, dict]:
        """Generate matched STOP/PROCEED counterfactual pair for Task 2 (Place Object).
        
        STOP condition: target_region is occupied by target_occupant object.
        PROCEED condition: minimal intervention - move target_occupant outside target_region.
        
        Returns:
            Dict containing metadata for both PROCEED and STOP samples.
        """
        pair_dir = self.output_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build STOP Scene (Occupant inside target_region)
        stop_objects = [
            {
                "name": "coffee_can", # Main item to place
                "type": "coffee_can",
                "pos": [-0.20, -0.20, 0.64],
            },
            {
                "name": "occupant",
                "type": target_occupant_type,
                "pos": [0.0, 0.20, 0.64], # Inside target_region
            },
        ]
        model_stop, data_stop = self.scene_builder.create_environment(stop_objects, settle_steps=100)
        renderer_stop = OffscreenRenderer(model_stop, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_stop = renderer_stop.render_rgb(data_stop)
        culprit_mask_stop = renderer_stop.render_culprit_mask(data_stop, ["occupant_geom"])
        region_mask_stop = renderer_stop.render_region_mask(data_stop, ["counter_surface"])

        is_occ_stop, active_culprits_stop = check_target_occupancy(
            model_stop, data_stop, candidate_objects=["occupant"]
        )
        renderer_stop.close()

        # 2. Build PROCEED Scene (Minimal intervention: move occupant outside target_region)
        proceed_objects = [
            {
                "name": "coffee_can",
                "type": "coffee_can",
                "pos": [-0.20, -0.20, 0.64],
            },
            {
                "name": "occupant",
                "type": target_occupant_type,
                "pos": [-0.35, 0.20, 0.64], # Moved outside target_region
            },
        ]
        model_proceed, data_proceed = self.scene_builder.create_environment(proceed_objects, settle_steps=100)
        renderer_proceed = OffscreenRenderer(model_proceed, width=self.width, height=self.height, camera_name=self.camera_name)

        rgb_proceed = renderer_proceed.render_rgb(data_proceed)
        culprit_mask_proceed = renderer_proceed.render_culprit_mask(data_proceed, ["occupant_geom"])
        region_mask_proceed = renderer_proceed.render_region_mask(data_proceed, ["counter_surface"])

        is_occ_proceed, active_culprits_proceed = check_target_occupancy(
            model_proceed, data_proceed, candidate_objects=["occupant"]
        )
        renderer_proceed.close()

        # Save query images & masks
        stop_rgb_path = pair_dir / "stop_rgb.png"
        stop_mask_path = pair_dir / "stop_culprit_mask.png"
        stop_target_mask_path = pair_dir / "stop_target_mask.png"

        proceed_rgb_path = pair_dir / "proceed_rgb.png"
        proceed_mask_path = pair_dir / "proceed_culprit_mask.png"
        proceed_target_mask_path = pair_dir / "proceed_target_mask.png"

        Image.fromarray(rgb_stop).save(stop_rgb_path)
        Image.fromarray(culprit_mask_stop).save(stop_mask_path)
        Image.fromarray(region_mask_stop).save(stop_target_mask_path)

        Image.fromarray(rgb_proceed).save(proceed_rgb_path)
        Image.fromarray(culprit_mask_proceed).save(proceed_mask_path)
        Image.fromarray(region_mask_proceed).save(proceed_target_mask_path)

        pair_meta = {
            "pair_id": pair_id,
            "task_id": "task_2",
            "instruction": "Place object1 in the target region.",
            "target_occupant_type": target_occupant_type,
            "stop": {
                "label": "STOP",
                "is_occupied": is_occ_stop,
                "culprits": active_culprits_stop,
                "rgb_path": str(stop_rgb_path),
                "culprit_mask_path": str(stop_mask_path),
                "region_mask_path": str(stop_target_mask_path),
            },
            "proceed": {
                "label": "PROCEED",
                "is_occupied": is_occ_proceed,
                "culprits": active_culprits_proceed,
                "rgb_path": str(proceed_rgb_path),
                "culprit_mask_path": str(proceed_mask_path),
                "region_mask_path": str(proceed_target_mask_path),
            },
        }

        with open(pair_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(pair_meta, f, indent=2)

        return pair_meta
