"""
Split planner for allocating factor combinations to benchmark splits.

Ensures holds out objects and background configurations are correctly assigned
to 'id', 'unseen_object', 'unseen_background', and 'compositional' splits.
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


class SplitPlanner:
    """Manages benchmark split rules and factor allocations."""

    def __init__(self, splits_config_path: str = "configs/splits.yaml"):
        self.config_path = Path(splits_config_path)
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "holdout_distractors": {
                    "train": ["coffee_can", "sugar_box", "mug"],
                    "val": ["cup"],
                    "test": ["bowl"],
                }
            }

        self.train_objects = self.config.get("holdout_distractors", {}).get("train", ["coffee_can", "sugar_box", "mug"])
        self.val_objects = self.config.get("holdout_distractors", {}).get("val", ["cup"])
        self.test_objects = self.config.get("holdout_distractors", {}).get("test", ["bowl"])

    def get_assignment_for_split(self, split: str, idx: int = 0) -> SplitAssignment:
        """Return object type and background profile for a given split and index."""
        if split == "id":
            obj = self.train_objects[idx % len(self.train_objects)]
            bg = "bg_neutral_wood"
        elif split == "unseen_object":
            unseen = self.val_objects + self.test_objects
            obj = unseen[idx % len(unseen)]
            bg = "bg_neutral_wood"
        elif split == "unseen_background":
            obj = self.train_objects[idx % len(self.train_objects)]
            bg = "bg_blue_counter"
        elif split == "compositional":
            unseen = self.val_objects + self.test_objects
            obj = unseen[idx % len(unseen)]
            bg = "bg_granite_dark"
        else:
            obj = self.train_objects[idx % len(self.train_objects)]
            bg = "bg_neutral_wood"

        return SplitAssignment(split=split, object_type=obj, background_id=bg)
