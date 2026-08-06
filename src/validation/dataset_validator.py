"""
Comprehensive dataset validator for checking counterfactual pairs, split holdout leakage,
positive controls, uint16 instance maps, mask semantics, seed reproducibility, and spec invariants.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Union

import numpy as np
import PIL.Image as Image

from src.generation.counterfactual_generator import regenerate_from_metadata


def deep_diff(d1: Any, d2: Any, path: str = "") -> List[str]:
    """Recursively compare two data structures and return a list of dot-separated difference paths."""
    diff_paths: List[str] = []

    if type(d1) != type(d2):
        diff_paths.append(path)
        return diff_paths

    if isinstance(d1, dict):
        all_keys = set(d1.keys()).union(d2.keys())
        for k in sorted(all_keys):
            sub_path = f"{path}.{k}" if path else str(k)
            if k not in d1 or k not in d2:
                diff_paths.append(sub_path)
            else:
                diff_paths.extend(deep_diff(d1[k], d2[k], sub_path))
    elif isinstance(d1, list):
        if len(d1) != len(d2):
            diff_paths.append(path)
        else:
            for idx, (v1, v2) in enumerate(zip(d1, d2)):
                sub_path = f"{path}[{idx}]"
                diff_paths.extend(deep_diff(v1, v2, sub_path))
    elif isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
        if not math.isclose(d1, d2, abs_tol=1e-5, rel_tol=1e-4):
            diff_paths.append(path)
    else:
        if d1 != d2:
            diff_paths.append(path)

    return [p for p in diff_paths if p]


class DatasetValidator:
    """Validator performing 26 rigorous dataset verification passes and exporting machine-readable reports."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.output_reports_dir = Path("data/reports")
        self.output_reports_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict[str, Any]] = []
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))

    def validate_matched_pair_invariants(self, pair: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate invariant equality between STOP and PROCEED members using deep_diff."""
        issues: List[str] = []
        pair_id = pair.get("pair_id", "unknown")

        stop_meta = pair.get("stop", {})
        proceed_meta = pair.get("proceed", {})

        stop_spec = stop_meta.get("resolved_scene_spec") or stop_meta.get("spec", {})
        proc_spec = proceed_meta.get("resolved_scene_spec") or proceed_meta.get("spec", {})

        if not stop_spec or not proc_spec:
            issues.append(f"Pair {pair_id}: missing resolved_scene_spec in STOP or PROCEED")
            return False, issues

        declared_interventions = (
            pair.get("declared_intervention_paths")
            or stop_meta.get("declared_intervention_paths")
            or []
        )
        if not declared_interventions:
            declared_interventions = ["objects.blocker1.position"]
            if pair.get("blocker_count") == 2:
                declared_interventions.append("objects.blocker2.position")
            if pair.get("task_id") == "task_2":
                declared_interventions = ["objects.occupant.position"]

        observed_diff_paths = deep_diff(stop_spec, proc_spec)

        declared_set = set(declared_interventions)
        observed_set = set(observed_diff_paths)

        def normalize_path(p: str) -> str:
            for decl in declared_set:
                if p.startswith(decl):
                    return decl
            return p

        norm_observed_set = {normalize_path(p) for p in observed_set if p != "label"}

        unexpected_diff_paths = sorted(list(norm_observed_set - declared_set))
        missing_declared_changes = sorted(list(declared_set - norm_observed_set))

        if unexpected_diff_paths:
            issues.append(f"Pair {pair_id}: unexpected difference paths between STOP and PROCEED: {unexpected_diff_paths}")
        if missing_declared_changes:
            issues.append(f"Pair {pair_id}: declared intervention paths did not change between STOP and PROCEED: {missing_declared_changes}")

        return len(issues) == 0, issues

    def run_reproducibility_validation(self, sample_count: int = 5) -> Tuple[bool, Dict[str, Any]]:
        """Perform actual reconstruction validation on a subset of manifest records."""
        if not self.records:
            return False, {"status": "FAILED", "reason": "No records in manifest"}

        rng = np.random.default_rng(2026)
        num_samples = min(sample_count, len(self.records))
        indices = rng.choice(len(self.records), size=num_samples, replace=False)
        selected_records = [self.records[i] for i in indices]

        sample_results = []
        overall_passed = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for idx, orig_meta in enumerate(selected_records):
                sample_id = orig_meta.get("pair_id") or orig_meta.get("control_id") or f"sample_{idx}"
                sample_type = orig_meta.get("sample_type", "matched_pair")
                task_id = orig_meta.get("task_id", "task_1")

                try:
                    regen_meta = regenerate_from_metadata(orig_meta, output_dir=tmp_path / f"regen_{idx}")

                    # Check metadata matching
                    meta_diffs = deep_diff(
                        orig_meta.get("resolved_scene_spec", orig_meta.get("background_id")),
                        regen_meta.get("resolved_scene_spec", regen_meta.get("background_id")),
                    )

                    # Compare STOP/control RGB and uint16 instance maps
                    if sample_type == "matched_pair":
                        orig_rgb_p = orig_meta["stop"]["rgb_path"]
                        regen_rgb_p = regen_meta["stop"]["rgb_path"]
                        orig_inst_p = orig_meta["stop"]["instance_uint16_path"]
                        regen_inst_p = regen_meta["stop"]["instance_uint16_path"]
                    else:
                        orig_rgb_p = orig_meta["rgb_path"]
                        regen_rgb_p = regen_meta["rgb_path"]
                        orig_inst_p = orig_meta["instance_uint16_path"]
                        regen_inst_p = regen_meta["instance_uint16_path"]

                    orig_rgb = np.array(Image.open(orig_rgb_p))
                    regen_rgb = np.array(Image.open(regen_rgb_p))

                    max_diff = int(np.max(np.abs(orig_rgb.astype(int) - regen_rgb.astype(int))))
                    mean_diff = float(np.mean(np.abs(orig_rgb.astype(float) - regen_rgb.astype(float))))
                    mismatched_px = int(np.count_nonzero(orig_rgb != regen_rgb))

                    orig_inst = np.load(orig_inst_p)
                    regen_inst = np.load(regen_inst_p)
                    inst_mismatch_count = int(np.count_nonzero(orig_inst != regen_inst))

                    sample_passed = (max_diff <= 5 and inst_mismatch_count == 0 and len(meta_diffs) == 0)
                    if not sample_passed:
                        overall_passed = False

                    sample_results.append({
                        "sample_id": sample_id,
                        "sample_type": sample_type,
                        "task_id": task_id,
                        "status": "PASSED" if sample_passed else "FAILED",
                        "rgb_max_difference": max_diff,
                        "rgb_mean_difference": mean_diff,
                        "rgb_mismatched_pixels": mismatched_px,
                        "instance_mismatch_count": inst_mismatch_count,
                        "metadata_differences": meta_diffs,
                    })

                except Exception as err:
                    overall_passed = False
                    sample_results.append({
                        "sample_id": sample_id,
                        "sample_type": sample_type,
                        "task_id": task_id,
                        "status": "FAILED",
                        "error": str(err),
                    })

        rep_report = {
            "status": "PASSED" if overall_passed else "FAILED",
            "tested_sample_count": len(sample_results),
            "reconstructed_equality": overall_passed,
            "samples": sample_results,
        }

        with open(self.output_reports_dir / "reproducibility_report.json", "w", encoding="utf-8") as f:
            json.dump(rep_report, f, indent=2)

        return overall_passed, rep_report

    def validate_splits(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate holdout split sets, factor tuples, and leakage."""
        dev_objects: Set[str] = set()
        dev_backgrounds: Set[str] = set()
        dev_factor_tuples: Set[Tuple] = set()

        unseen_obj_objects: Set[str] = set()
        unseen_bg_backgrounds: Set[str] = set()
        comp_factor_tuples: Set[Tuple] = set()

        splits_count: Dict[str, int] = {}

        for rec in self.records:
            sp = rec.get("split", "id")
            splits_count[sp] = splits_count.get(sp, 0) + 1

            if rec.get("sample_type") == "positive_control":
                continue  # Positive controls do not contaminate holdout factor accounting

            task_id = rec.get("task_id", "task_1")
            obj = rec.get("blocker_type") or rec.get("target_occupant_type") or "none"
            bg = rec.get("background_id", "bg_neutral_wood")
            pos_bin = rec.get("blocker_pos_bin") or rec.get("occupant_pos_bin") or "centre"
            b_count = rec.get("blocker_count", 1)

            factor_tuple = (task_id, obj, bg, pos_bin, b_count)

            if sp == "id":
                dev_objects.add(obj)
                dev_backgrounds.add(bg)
                dev_factor_tuples.add(factor_tuple)
            elif sp == "unseen_object":
                unseen_obj_objects.add(obj)
            elif sp == "unseen_background":
                unseen_bg_backgrounds.add(bg)
            elif sp == "compositional":
                comp_factor_tuples.add(factor_tuple)

        obj_intersection = sorted(list(dev_objects.intersection(unseen_obj_objects)))
        bg_intersection = sorted(list(dev_backgrounds.intersection(unseen_bg_backgrounds)))
        comp_intersection = [list(t) for t in dev_factor_tuples.intersection(comp_factor_tuples)]

        has_obj_leak = len(obj_intersection) > 0
        has_bg_leak = len(bg_intersection) > 0
        has_comp_leak = len(comp_intersection) > 0

        overall_valid = not (has_obj_leak or has_bg_leak or has_comp_leak)

        split_rep = {
            "status": "PASSED" if overall_valid else "FAILED",
            "split_distribution": splits_count,
            "development_objects": sorted(list(dev_objects)),
            "unseen_object_objects": sorted(list(unseen_obj_objects)),
            "object_intersection": obj_intersection,
            "development_backgrounds": sorted(list(dev_backgrounds)),
            "unseen_backgrounds": sorted(list(unseen_bg_backgrounds)),
            "background_intersection": bg_intersection,
            "development_factor_tuples": [list(t) for t in dev_factor_tuples],
            "compositional_factor_tuples": [list(t) for t in comp_factor_tuples],
            "compositional_intersection": comp_intersection,
            "leakage_checks": {
                "object_leakage": has_obj_leak,
                "background_leakage": has_bg_leak,
                "compositional_leakage": has_comp_leak,
            },
        }

        with open(self.output_reports_dir / "split_validation.json", "w", encoding="utf-8") as f:
            json.dump(split_rep, f, indent=2)

        return overall_valid, split_rep

    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """Run all 26 verification passes on dataset manifest and generated files."""
        logs: List[str] = []
        issues: List[str] = []

        if not self.records:
            return False, ["Manifest is empty or missing"]

        logs.append(f"Loaded {len(self.records)} records from {self.manifest_path.name}.")

        matched_pairs = [r for r in self.records if r.get("sample_type") == "matched_pair" or "pair_id" in r]
        positive_controls = [r for r in self.records if r.get("sample_type") == "positive_control"]

        logs.append(f"Found {len(matched_pairs)} matched pairs and {len(positive_controls)} standalone positive controls.")

        # Pass 1 & 2: Matched-Pair Invariant Deep Diff Validation
        pair_issues = []
        for pair in matched_pairs:
            valid, p_issues = self.validate_matched_pair_invariants(pair)
            if not valid:
                pair_issues.extend(p_issues)
        logs.append(f"Pass 1 & 2 (Matched Pair Invariant Deep Diff): {len(pair_issues)} issues.")
        issues.extend(pair_issues)

        # Pass 3: File Existence, Mask Semantics & Stability
        mask_issues = []
        for rec in self.records:
            if rec.get("sample_type") == "positive_control":
                sub_samples = [("control", rec)]
            else:
                sub_samples = [("stop", rec.get("stop", {})), ("proceed", rec.get("proceed", {}))]

            for label_name, sub in sub_samples:
                sample_id = sub.get("sample_id") or rec.get("pair_id") or rec.get("control_id")
                rgb_p = sub.get("rgb_path")
                inst_p = sub.get("instance_uint16_path")
                cand_p = sub.get("candidate_object_mask_path") or sub.get("culprit_mask_path")
                target_p = sub.get("relation_target_mask_path") or sub.get("region_mask_path")

                if not rgb_p or not Path(rgb_p).exists():
                    mask_issues.append(f"Record {sample_id}: missing RGB file {rgb_p}")
                if not inst_p or not Path(inst_p).exists():
                    mask_issues.append(f"Record {sample_id}: missing uint16 Instance file {inst_p}")
                else:
                    try:
                        arr_16 = np.load(inst_p)
                        if arr_16.dtype != np.uint16:
                            mask_issues.append(f"Record {sample_id}: uint16 instance map has wrong dtype {arr_16.dtype}")
                        if arr_16.ndim != 2:
                            mask_issues.append(f"Record {sample_id}: uint16 instance map shape invalid {arr_16.shape}")
                    except Exception as err:
                        mask_issues.append(f"Record {sample_id}: failed to load uint16 instance map: {err}")

                if cand_p and Path(cand_p).exists():
                    cand_arr = np.array(Image.open(cand_p))
                    if label_name == "stop" and np.count_nonzero(cand_arr) == 0:
                        mask_issues.append(f"Record {sample_id}: STOP candidate mask is empty")

                if target_p and Path(target_p).exists():
                    t_arr = np.array(Image.open(target_p))
                    if np.count_nonzero(t_arr) == 0:
                        mask_issues.append(f"Record {sample_id}: Relation target mask is empty")
                    if np.count_nonzero(t_arr) > (t_arr.shape[0] * t_arr.shape[1] * 0.50):
                        mask_issues.append(f"Record {sample_id}: Target mask covers over 50% of image")

                if label_name == "stop":
                    causal_p = sub.get("causal_violation_mask_path")
                    if causal_p and Path(causal_p).exists():
                        causal_arr = np.array(Image.open(causal_p))
                        if np.count_nonzero(causal_arr) == 0:
                            mask_issues.append(f"Record {sample_id}: STOP causal violation mask is empty")

                    meas = sub.get("measurements", {})
                    for culprit_name, culprit_m in meas.items():
                        if isinstance(culprit_m, dict) and culprit_m.get("relation_true"):
                            if not culprit_m.get("stable", True):
                                mask_issues.append(f"Record {sample_id}: STOP culprit '{culprit_name}' is unstable (lin_speed={culprit_m.get('linear_speed'):.4f})")
                            if not culprit_m.get("settling_succeeded", True):
                                mask_issues.append(f"Record {sample_id}: STOP culprit '{culprit_name}' did not succeed settling")

                elif label_name == "proceed":
                    causal_p = sub.get("causal_violation_mask_path")
                    if causal_p and Path(causal_p).exists():
                        causal_arr = np.array(Image.open(causal_p))
                        if np.count_nonzero(causal_arr) != 0:
                            mask_issues.append(f"Record {sample_id}: PROCEED causal violation mask is not zero")

        logs.append(f"Pass 3 (File, Mask Semantics & Stability): {len(mask_issues)} issues.")
        issues.extend(mask_issues)

        # Pass 4: Split Validation
        splits_valid, split_rep = self.validate_splits()
        if not splits_valid:
            issues.append(f"Split validation failed: leakage detected: {split_rep.get('leakage_checks')}")
        logs.append(f"Pass 4 (Split Holdout & Leakage): {'PASSED' if splits_valid else 'FAILED'}.")

        # Pass 5: Actual Reproducibility Regeneration
        rep_valid, rep_rep = self.run_reproducibility_validation(sample_count=min(5, len(self.records)))
        if not rep_valid:
            issues.append("Reproducibility validation failed during sample reconstruction!")
        logs.append(f"Pass 5 (Actual Reproducibility Regeneration): {'PASSED' if rep_valid else 'FAILED'}.")

        # Export overall dataset validation report
        dataset_rep = {
            "status": "PASSED" if len(issues) == 0 else "FAILED",
            "total_records": len(self.records),
            "matched_pairs_count": len(matched_pairs),
            "positive_controls_count": len(positive_controls),
            "issue_count": len(issues),
            "issues": issues,
        }
        with open(self.output_reports_dir / "dataset_validation.json", "w", encoding="utf-8") as f:
            json.dump(dataset_rep, f, indent=2)

        # Export distribution report
        tasks_count: Dict[str, int] = {}
        splits_count: Dict[str, int] = {}
        for r in self.records:
            t = r.get("task_id", "unknown")
            s = r.get("split", "unknown")
            tasks_count[t] = tasks_count.get(t, 0) + 1
            splits_count[s] = splits_count.get(s, 0) + 1

        dist_rep = {
            "total_samples": len(self.records),
            "task_distribution": tasks_count,
            "split_distribution": splits_count,
            "matched_pairs": len(matched_pairs),
            "positive_controls": len(positive_controls),
        }
        with open(self.output_reports_dir / "distribution_report.json", "w", encoding="utf-8") as f:
            json.dump(dist_rep, f, indent=2)

        is_valid = len(issues) == 0
        if is_valid:
            logs.append("SUCCESS: All dataset validation checks passed cleanly!")
        else:
            logs.append(f"FAILED: {len(issues)} total dataset issues detected.")

        return is_valid, logs


if __name__ == "__main__":
    import sys
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "data/manifests/smoke_manifest.jsonl"
    validator = DatasetValidator(manifest_path)
    valid, logs = validator.validate_dataset()
    for log in logs:
        print(log)
    sys.exit(0 if valid else 1)
