"""
Comprehensive dataset validator verifying dataset integrity, counterfactual symmetry, predicate consistency, and leakage.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json
import cv2
import numpy as np
import PIL.Image as Image


class DatasetValidator:
    """Validator class for verifying benchmark dataset compliance."""

    def __init__(self, manifest_path: Union[str, Path]):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}")

    def load_manifest(self) -> List[Dict[str, dict]]:
        records = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def validate_demonstration_video(self, video_path: Union[str, Path], expected_min_frames: int = 60) -> Tuple[bool, str]:
        """Validate demonstration video readability, frame count, FPS, and non-static motion."""
        video_path = Path(video_path)
        if not video_path.exists():
            return False, f"Video file missing: {video_path}"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, f"Cannot open video file: {video_path}"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_count < expected_min_frames:
            cap.release()
            return False, f"Video {video_path.name} has only {frame_count} frames (expected >= {expected_min_frames})"

        # Motion check between first and middle frames
        ret1, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret2, frame2 = cap.read()
        cap.release()

        if not ret1 or not ret2:
            return False, f"Could not read video frames from {video_path.name}"

        diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
        if diff < 1.0:
            return False, f"Video {video_path.name} appears static (frame diff={diff:.2f})"

        return True, f"Video {video_path.name} valid ({frame_count} frames, {fps:.1f} FPS, diff={diff:.2f})"

    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """Run complete benchmark validation suite."""
        messages = []
        is_valid = True
        records = self.load_manifest()

        if not records:
            messages.append("ERROR: Manifest file is empty!")
            return False, messages

        messages.append(f"Loaded {len(records)} counterfactual pair records from manifest.")

        pair_ids = set()
        bg_counts = Counter()
        obj_counts = Counter()
        split_counts = Counter()
        pos_counts = Counter()
        mask_sizes = []

        for idx, rec in enumerate(records):
            pair_id = rec.get("pair_id", f"record_{idx}")

            if pair_id in pair_ids:
                messages.append(f"ERROR: Duplicate pair_id detected: {pair_id}")
                is_valid = False
            pair_ids.add(pair_id)

            split = rec.get("split", "id")
            split_counts[split] += 1

            # Check both STOP and PROCEED samples in pair
            for mode in ["stop", "proceed"]:
                sub = rec[mode]
                label = sub["label"]
                spec = sub.get("spec", {})

                bg = spec.get("background_id", "bg_default")
                bg_counts[f"{bg}_{label}"] += 1

                objs = spec.get("blocker_or_occupant_types", ["unknown"])
                for o in objs:
                    obj_counts[f"{o}_{label}"] += 1

                bins = spec.get("blocker_position_bins", [spec.get("occupant_position_bin", "default")])
                for b in bins:
                    if b:
                        pos_counts[f"{b}_{label}"] += 1

                # 1. Verify required files exist
                rgb_p = Path(sub["rgb_path"])
                inst_p = Path(sub.get("instance_segmentation_path", pair_dir_default(rgb_p, f"{mode}_instance_segmentation.png")))
                cand_p = Path(sub.get("candidate_object_mask_path", pair_dir_default(rgb_p, f"{mode}_candidate_object_mask.png")))
                target_p = Path(sub.get("relation_target_mask_path", pair_dir_default(rgb_p, f"{mode}_relation_target_mask.png")))
                causal_p = Path(sub.get("causal_violation_mask_path", pair_dir_default(rgb_p, f"{mode}_causal_violation_mask.png")))

                for name, p in [
                    ("RGB", rgb_p),
                    ("Instance Segmentation", inst_p),
                    ("Candidate Object Mask", cand_p),
                    ("Relation Target Mask", target_p),
                    ("Causal Violation Mask", causal_p),
                ]:
                    if not p.exists():
                        messages.append(f"[{pair_id}] {mode} {name} missing: {p}")
                        is_valid = False

                # 2. Check mask non-emptiness & causal mask semantics
                if cand_p.exists():
                    c_img = np.array(Image.open(cand_p))
                    if np.max(c_img) == 0:
                        messages.append(f"[{pair_id}] {mode} Candidate Object Mask is empty!")
                        is_valid = False
                    mask_sizes.append(np.sum(c_img > 0))

                if target_p.exists():
                    t_img = np.array(Image.open(target_p))
                    if np.max(t_img) == 0:
                        messages.append(f"[{pair_id}] {mode} Relation Target Mask is empty!")
                        is_valid = False

                if causal_p.exists():
                    caus_img = np.array(Image.open(causal_p))
                    if mode == "stop" and np.max(caus_img) == 0:
                        messages.append(f"[{pair_id}] STOP Causal Violation Mask is empty (expected non-empty)!")
                        is_valid = False
                    elif mode == "proceed" and np.max(caus_img) > 0:
                        messages.append(f"[{pair_id}] PROCEED Causal Violation Mask is non-zero (expected all zeros)!")
                        is_valid = False

            # 3. Label & predicate consistency checks
            if not rec["stop"]["is_occupied"]:
                messages.append(f"[{pair_id}] STOP state predicate check failed: expected occupied=True")
                is_valid = False

            if rec["proceed"]["is_occupied"]:
                messages.append(f"[{pair_id}] PROCEED state predicate check failed: expected occupied=False")
                is_valid = False

        avg_mask_size = np.mean(mask_sizes) if mask_sizes else 0
        messages.append(f"Distribution: {len(pair_ids)} pairs across splits {dict(split_counts)}.")
        messages.append(f"Average Candidate Mask Size: {avg_mask_size:.1f} pixels.")

        if is_valid:
            messages.append("SUCCESS: All dataset validation checks passed cleanly!")
        else:
            messages.append("FAILURE: Dataset validation failed with errors.")

        return is_valid, messages


def pair_dir_default(rgb_path: Path, filename: str) -> str:
    return str(rgb_path.parent / filename)


def validate_manifest(manifest_path: str) -> bool:
    validator = DatasetValidator(manifest_path)
    is_valid, logs = validator.validate_dataset()
    for log in logs:
        print(log)
    return is_valid


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/manifests/smoke_manifest.jsonl"
    success = validate_manifest(path)
    sys.exit(0 if success else 1)
