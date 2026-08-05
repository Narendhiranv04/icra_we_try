"""
Master query generator for driving benchmark query pair creation and producing JSON manifests.
"""

from pathlib import Path
from typing import Dict, List, Union
import json
import yaml

from src.generation.counterfactual_generator import CounterfactualPairGenerator


class QueryGenerator:
    """Generator for creating full dataset query pairs according to profile configuration."""

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

    def run_generation(self) -> List[Dict[str, dict]]:
        """Run query pair generation for all configured tasks.
        
        Returns:
            List of generated metadata dictionary records.
        """
        all_records = []
        num_pairs = self.config["query_generation"].get("num_pairs_per_task", 2)

        for task_cfg in self.config.get("tasks", []):
            task_id = task_cfg["id"]
            blockers = task_cfg.get("blocker_objects", ["coffee_can", "sugar_box"])

            for idx in range(num_pairs):
                pair_id = f"pair_{task_id}_{idx+1:03d}"
                blocker_obj = blockers[idx % len(blockers)]

                if task_id == "task_1":
                    record = self.counterfactual_gen.generate_task1_pair(
                        pair_id, blocker_type=blocker_obj
                    )
                else:
                    record = self.counterfactual_gen.generate_task2_pair(
                        pair_id, target_occupant_type=blocker_obj
                    )
                all_records.append(record)

        # Write manifest file
        manifest_path = self.manifests_dir / f"{self.config['profile_name']}_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")

        return all_records
