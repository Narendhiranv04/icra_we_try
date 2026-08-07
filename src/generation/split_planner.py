"""
Split planner for allocating factor combinations to benchmark splits.

Ensures strict partitioning across 'id', 'unseen_object', 'unseen_background', and 'compositional' splits.
Compositional split uses exclusively familiar factor components whose full factor tuple is absent from development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml
from pathlib import Path


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    object_type: str
    background_id: str
    position_bin: str
    blocker_count: int
    object1_start_bin: str
    lighting_family: str
    factor_tuple: Tuple[str, ...]


class SplitPlanner:
    """Manages benchmark split rules and factor allocations."""

    def __init__(self, splits_config_path: str = "configs/splits.yaml"):
        self.config_path = Path(splits_config_path)
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        id_factors = self.config.get("id_factors", {})
        self.id_objects: List[str] = id_factors.get("objects", ["coffee_can", "sugar_box", "mug", "gso_coffee_jar", "gso_sugar_jar"])
        self.id_backgrounds: List[str] = id_factors.get("backgrounds", ["bg_neutral_wood"])
        self.id_pos_t1: List[str] = id_factors.get("position_bins_task1", ["centre", "front_left", "front_right"])
        self.id_pos_t2: List[str] = id_factors.get("position_bins_task2", ["centre", "left", "right"])

        holdout_d = self.config.get("holdout_distractors", {})
        self.unseen_objects: List[str] = (holdout_d.get("val", ["cup"]) + holdout_d.get("test", ["bowl", "gso_canister_distractor"]))
        self.unseen_backgrounds: List[str] = self.config.get("holdout_backgrounds", {}).get("unseen", ["bg_blue_counter", "bg_granite_dark"])

        self.start_bins: List[str] = ["pick_left", "pick_right", "pick_far_left"]
        self.lighting_families: List[str] = id_factors.get("lighting_families", ["default_lighting"])

    def get_assignment_for_split(
        self,
        split: str,
        idx: int = 0,
        task_id: str = "task_1",
    ) -> SplitAssignment:
        """Return a complete SplitAssignment containing factor choices and explicit factor tuple."""
        lighting_fam = self.lighting_families[0]

        if task_id == "task_1":
            if split == "id":
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t1[idx % len(self.id_pos_t1)]
                b_count = 1 if (idx % 2 == 0) else 2
                start_bin = "pick_left" if (idx % 2 == 0) else "pick_right"
                factor_tuple = (task_id, obj, bg, pos_bin, b_count, start_bin, lighting_fam)
            elif split == "unseen_object":
                obj = self.unseen_objects[idx % len(self.unseen_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t1[idx % len(self.id_pos_t1)]
                b_count = 1
                start_bin = "pick_left"
                factor_tuple = (task_id, obj, bg, pos_bin, b_count, start_bin, lighting_fam)
            elif split == "unseen_background":
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.unseen_backgrounds[idx % len(self.unseen_backgrounds)]
                pos_bin = self.id_pos_t1[idx % len(self.id_pos_t1)]
                b_count = 1
                start_bin = "pick_left"
                factor_tuple = (task_id, obj, bg, pos_bin, b_count, start_bin, lighting_fam)
            elif split == "compositional":
                # Pure compositional split: 100% familiar components, novel factor combination absent from ID split!
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t1[idx % len(self.id_pos_t1)]
                b_count = 2 if (idx % 2 == 0) else 1                        # Swapped count relative to ID
                start_bin = "pick_left" if (idx % 2 == 0) else "pick_right" # Same start_bin sequence so (b_count, start_bin) is novel!
                factor_tuple = (task_id, obj, bg, pos_bin, b_count, start_bin, lighting_fam)
            else:
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t1[idx % len(self.id_pos_t1)]
                b_count = 1
                start_bin = "pick_left"
                factor_tuple = (task_id, obj, bg, pos_bin, b_count, start_bin, lighting_fam)

        else: # Task 2
            if split == "id":
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t2[idx % len(self.id_pos_t2)]
                b_count = 1
                start_bin = self.start_bins[idx % len(self.start_bins)]
                factor_tuple = (task_id, obj, bg, pos_bin, start_bin, lighting_fam)
            elif split == "unseen_object":
                obj = self.unseen_objects[idx % len(self.unseen_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t2[idx % len(self.id_pos_t2)]
                b_count = 1
                start_bin = self.start_bins[idx % len(self.start_bins)]
                factor_tuple = (task_id, obj, bg, pos_bin, start_bin, lighting_fam)
            elif split == "unseen_background":
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.unseen_backgrounds[idx % len(self.unseen_backgrounds)]
                pos_bin = self.id_pos_t2[idx % len(self.id_pos_t2)]
                b_count = 1
                start_bin = self.start_bins[idx % len(self.start_bins)]
                factor_tuple = (task_id, obj, bg, pos_bin, start_bin, lighting_fam)
            elif split == "compositional":
                # Pure compositional split: 100% familiar components, novel factor combination absent from ID split!
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t2[idx % len(self.id_pos_t2)]
                b_count = 1
                start_bin = self.start_bins[(idx + 1) % len(self.start_bins)] # Shifted start_bin relative to pos_bin!
                factor_tuple = (task_id, obj, bg, pos_bin, start_bin, lighting_fam)
            else:
                obj = self.id_objects[idx % len(self.id_objects)]
                bg = self.id_backgrounds[0]
                pos_bin = self.id_pos_t2[idx % len(self.id_pos_t2)]
                b_count = 1
                start_bin = self.start_bins[idx % len(self.start_bins)]
                factor_tuple = (task_id, obj, bg, pos_bin, start_bin, lighting_fam)

        return SplitAssignment(
            split=split,
            object_type=obj,
            background_id=bg,
            position_bin=pos_bin,
            blocker_count=b_count,
            object1_start_bin=start_bin,
            lighting_family=lighting_fam,
            factor_tuple=factor_tuple,
        )
