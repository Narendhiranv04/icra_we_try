"""
Master query generator driving counterfactual pair creation, positive controls, and manifest production.
"""

from pathlib import Path
from typing import Dict, List, Union, Any
import json
import yaml

from src.generation.counterfactual_generator import CounterfactualPairGenerator
from src.generation.split_planner import SplitPlanner


def generate_control_distribution_report(
    records: List[Dict[str, Any]],
    output_report_path: Union[str, Path] = "data/reports/control_distribution.json",
) -> Dict[str, Any]:
    """Generate control distribution summary report and verify non-zero samples for all configured subtypes."""
    output_report_path = Path(output_report_path)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    controls = [r for r in records if r.get("sample_type") == "positive_control"]

    counts_by_task: Dict[str, int] = {}
    counts_by_subtype: Dict[str, int] = {}
    counts_by_object: Dict[str, int] = {}
    counts_by_bg: Dict[str, int] = {}
    cand_mask_empty_count = 0
    verified_occupancy_count = 0

    req_t1_subtypes = ["empty_lid", "one_object_beside", "two_objects_beside", "near_lid_outside_footprint"]
    req_t2_subtypes = ["empty_target", "one_object_beside_target", "one_object_near_target_outside", "multiple_distractors_outside"]

    for c in controls:
        t_id = c.get("task_id", "unknown")
        sub = c.get("control_subtype", "unknown")
        obj = c.get("object_type") or c.get("occupant_type") or "none"
        bg = c.get("background_id", "unknown")

        counts_by_task[t_id] = counts_by_task.get(t_id, 0) + 1
        counts_by_subtype[sub] = counts_by_subtype.get(sub, 0) + 1
        counts_by_object[obj] = counts_by_object.get(obj, 0) + 1
        counts_by_bg[bg] = counts_by_bg.get(bg, 0) + 1

        cand_p = c.get("candidate_object_mask_path")
        if not cand_p or not Path(cand_p).exists() or sub in ("empty_lid", "empty_target"):
            cand_mask_empty_count += 1

        if c.get("is_occupied", False):
            verified_occupancy_count += 1

    # Verify every subtype has at least 1 sample if controls generated
    missing_subtypes = []
    if len(controls) > 0:
        for sub in req_t1_subtypes + req_t2_subtypes:
            if counts_by_subtype.get(sub, 0) == 0:
                missing_subtypes.append(sub)

    overall_status = "PASSED" if (len(missing_subtypes) == 0 and verified_occupancy_count == 0) else "FAILED"

    report = {
        "status": overall_status,
        "total_controls": len(controls),
        "counts_by_task": counts_by_task,
        "counts_by_subtype": counts_by_subtype,
        "counts_by_object": counts_by_object,
        "counts_by_background": counts_by_bg,
        "candidate_mask_empty_count": cand_mask_empty_count,
        "verified_occupancy_count": verified_occupancy_count,
        "missing_subtypes": missing_subtypes,
    }

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


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
        self.queries_dir = Path(self.config.get("query_output_dir", self.output_dir / "queries"))
        self.manifests_dir = self.output_dir / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        self.counterfactual_gen = CounterfactualPairGenerator(
            output_dir=self.queries_dir,
            resolution=tuple(self.config["query_generation"]["resolution"]),
            camera_name=self.config["query_generation"]["camera_name"],
        )
        self.split_planner = SplitPlanner()

    def run_generation(self) -> List[Dict[str, dict]]:
        """Run query pair and positive control generation for all configured tasks."""
        all_records = []
        num_pairs = self.config["query_generation"].get("num_pairs_per_task", 4)
        num_controls = self.config["query_generation"].get("num_controls_per_task", 4)
        seed_base = self.config.get("seed", 42)

        splits = ["id", "unseen_object", "unseen_background", "compositional"]
        t1_control_subtypes = ["empty_lid", "one_object_beside", "two_objects_beside", "near_lid_outside_footprint"]
        t2_control_subtypes = ["empty_target", "one_object_beside_target", "one_object_near_target_outside", "multiple_distractors_outside"]

        for task_cfg in self.config.get("tasks", []):
            task_id = task_cfg["id"]

            # 1. Generate Matched Counterfactual Pairs using SplitPlanner
            for idx in range(num_pairs):
                pair_id = f"pair_{task_id}_{idx+1:03d}"
                split = splits[idx % len(splits)]
                ep_seed = seed_base + idx * 17

                split_idx = idx // len(splits)
                assignment = self.split_planner.get_assignment_for_split(split, split_idx, task_id=task_id)

                if task_id == "task_1":
                    record = self.counterfactual_gen.generate_task1_pair(
                        pair_id=pair_id,
                        blocker_type=assignment.object_type,
                        blocker_count=assignment.blocker_count,
                        blocker_pos_bin=assignment.position_bin,
                        split=split,
                        seed=ep_seed,
                    )
                else:
                    record = self.counterfactual_gen.generate_task2_pair(
                        pair_id=pair_id,
                        target_occupant_type=assignment.object_type,
                        occupant_pos_bin=assignment.position_bin,
                        object1_start_bin=assignment.object1_start_bin,
                        split=split,
                        seed=ep_seed,
                    )
                all_records.append(record)

            # 2. Generate Standalone Positive Controls
            for c_idx in range(num_controls):
                control_id = f"control_{task_id}_{c_idx+1:03d}"
                split = splits[c_idx % len(splits)]
                ctrl_seed = seed_base + 1000 + c_idx * 13

                assignment = self.split_planner.get_assignment_for_split(split, c_idx, task_id=task_id)

                if task_id == "task_1":
                    sub = t1_control_subtypes[c_idx % len(t1_control_subtypes)]
                    ctrl_record = self.counterfactual_gen.generate_task1_control(
                        control_id=control_id,
                        control_subtype=sub,
                        object_type=assignment.object_type if sub != "empty_lid" else None,
                        split=split,
                        seed=ctrl_seed,
                    )
                else:
                    sub = t2_control_subtypes[c_idx % len(t2_control_subtypes)]
                    ctrl_record = self.counterfactual_gen.generate_task2_control(
                        control_id=control_id,
                        control_subtype=sub,
                        occupant_type=assignment.object_type if sub != "empty_target" else None,
                        split=split,
                        seed=ctrl_seed,
                    )
                all_records.append(ctrl_record)

        # Write manifest file
        manifest_path = self.manifests_dir / f"{self.config['profile_name']}_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")

        # Generate control distribution report
        report_name = f"{self.config['profile_name']}_control_distribution.json" if self.config['profile_name'] != "smoke" else "control_distribution.json"
        generate_control_distribution_report(all_records, output_report_path=f"data/reports/{report_name}")

        return all_records
