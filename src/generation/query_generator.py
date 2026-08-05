"""
Master query generator driving counterfactual pair creation, positive controls, and manifest production.
"""

from pathlib import Path
from typing import Dict, List, Union
import json
import yaml

from src.generation.counterfactual_generator import CounterfactualPairGenerator
from src.generation.split_planner import SplitPlanner


class QueryGenerator:
    """Generator for creating full dataset query pairs and positive controls according to profile configuration."""

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/smoke.yaml",
    ):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config.get("output_dir", "data"))
        self.queries_dir = self.output_dir / "queries"
        self.manifests_dir = self.output_dir / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        self.counterfactual_gen = CounterfactualPairGenerator(
            output_dir=self.queries_dir,
            resolution=tuple(self.config["query_generation"]["resolution"]),
            camera_name=self.config["query_generation"]["camera_name"],
        )
        self.split_planner = SplitPlanner()

    def run_generation(self) -> List[Dict[str, dict]]:
        """Run query pair and positive control generation for all configured tasks.
        
        Returns:
            List of generated metadata dictionary records.
        """
        all_records = []
        num_pairs = self.config["query_generation"].get("num_pairs_per_task", 4)
        num_controls = self.config["query_generation"].get("num_controls_per_task", 2)
        seed_base = self.config.get("seed", 42)

        task1_pos_bins = ["centre", "front_left", "front_right", "rear_left", "rear_right", "opening_edge", "hinge_side"]
        task2_pos_bins = ["centre", "left", "right", "front", "rear"]
        splits = ["id", "unseen_object", "unseen_background", "compositional"]

        for task_cfg in self.config.get("tasks", []):
            task_id = task_cfg["id"]

            # 1. Generate Matched Counterfactual Pairs using SplitPlanner
            for idx in range(num_pairs):
                pair_id = f"pair_{task_id}_{idx+1:03d}"
                split = splits[idx % len(splits)]
                ep_seed = seed_base + idx * 17

                assignment = self.split_planner.get_assignment_for_split(split, idx)
                blocker_obj = assignment.object_type

                if task_id == "task_1":
                    b_count = 2 if (idx % 3 == 2) else 1
                    pos_bin = task1_pos_bins[idx % len(task1_pos_bins)]
                    record = self.counterfactual_gen.generate_task1_pair(
                        pair_id=pair_id,
                        blocker_type=blocker_obj,
                        blocker_count=b_count,
                        blocker_pos_bin=pos_bin,
                        split=split,
                        seed=ep_seed,
                    )
                else:
                    pos_bin = task2_pos_bins[idx % len(task2_pos_bins)]
                    record = self.counterfactual_gen.generate_task2_pair(
                        pair_id=pair_id,
                        target_occupant_type=blocker_obj,
                        occupant_pos_bin=pos_bin,
                        split=split,
                        seed=ep_seed,
                    )
                all_records.append(record)

            # 2. Generate Standalone Positive Controls
            for c_idx in range(num_controls):
                control_id = f"control_{task_id}_{c_idx+1:03d}"
                split = splits[c_idx % len(splits)]
                ctrl_seed = seed_base + 1000 + c_idx * 13

                assignment = self.split_planner.get_assignment_for_split(split, c_idx)
                obj_type = assignment.object_type if (c_idx % 2 == 1) else None

                if task_id == "task_1":
                    ctrl_record = self.counterfactual_gen.generate_task1_control(
                        control_id=control_id,
                        object_type=obj_type,
                        split=split,
                        seed=ctrl_seed,
                    )
                else:
                    ctrl_record = self.counterfactual_gen.generate_task2_control(
                        control_id=control_id,
                        occupant_type=obj_type,
                        split=split,
                        seed=ctrl_seed,
                    )
                all_records.append(ctrl_record)

        # Write manifest file
        manifest_path = self.manifests_dir / f"{self.config['profile_name']}_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")

        return all_records
