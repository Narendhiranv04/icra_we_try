"""
Demonstration distinctness validator.

Verifies that demonstration video files within each task family are genuinely distinct
by computing scene spec hashes, initial/middle/final frame SHA256 checksums,
and mean absolute pixel differences.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class DemonstrationDistinctnessValidator:
    """Validator ensuring generated demonstration videos are distinct and rejecting duplicates."""

    def __init__(self, demo_base_dir: str = "data/demos"):
        self.demo_base_dir = Path(demo_base_dir)
        self.output_reports_dir = Path("data/reports")
        self.output_reports_dir.mkdir(parents=True, exist_ok=True)

    def _hash_bytes(self, b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _extract_frames_and_hashes(self, video_path: Path) -> Tuple[List[np.ndarray], List[str], str]:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        hashes = []
        whole_hasher = hashlib.sha256()

        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                whole_hasher.update(frame.tobytes())
                frames.append(frame)
        cap.release()

        whole_video_hash = whole_hasher.hexdigest()

        if not frames:
            return [], [], whole_video_hash

        total = len(frames)
        selected_indices = [0, total // 2, total - 1]
        selected_frames = [frames[i] for i in selected_indices]
        frame_hashes = [self._hash_bytes(f.tobytes()) for f in selected_frames]

        return selected_frames, frame_hashes, whole_video_hash

    def validate_distinctness(
        self,
        task_family: str,
        demo_dirs: List[Path],
        min_mean_pixel_diff: float = 2.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that all demonstration directories within a task family are pairwise distinct."""
        issues: List[str] = []
        demo_data: Dict[str, dict] = {}

        for d_dir in demo_dirs:
            d_dir = Path(d_dir)
            demo_id = d_dir.name
            video_p = d_dir / "rgb.mp4"
            spec_p = d_dir / "scene_config.json"

            if not video_p.exists():
                issues.append(f"Missing video for demo {demo_id}: {video_p}")
                continue

            spec_dict = {}
            if spec_p.exists():
                with open(spec_p, "r", encoding="utf-8") as f:
                    spec_dict = json.load(f)

            spec_hash = self._hash_bytes(json.dumps(spec_dict, sort_keys=True).encode("utf-8"))
            sel_frames, frame_hashes, video_hash = self._extract_frames_and_hashes(video_p)

            demo_data[demo_id] = {
                "dir": str(d_dir),
                "spec": spec_dict,
                "spec_hash": spec_hash,
                "frames": sel_frames,
                "frame_hashes": frame_hashes,
                "video_hash": video_hash,
            }

        demo_ids = sorted(list(demo_data.keys()))
        pairwise_results = []
        has_duplicates = False

        for i in range(len(demo_ids)):
            for j in range(i + 1, len(demo_ids)):
                id1, id2 = demo_ids[i], demo_ids[j]
                d1, d2 = demo_data[id1], demo_data[id2]

                same_spec = (d1["spec_hash"] == d2["spec_hash"])
                same_video_hash = (d1["video_hash"] == d2["video_hash"])
                same_frame_hashes = (d1["frame_hashes"] == d2["frame_hashes"])

                # Compute mean pixel difference across selected frames
                px_diffs = []
                for f1, f2 in zip(d1["frames"], d2["frames"]):
                    if f1.shape == f2.shape:
                        diff = float(np.mean(np.abs(f1.astype(float) - f2.astype(float))))
                        px_diffs.append(diff)
                    else:
                        px_diffs.append(255.0)

                mean_px_diff = float(np.mean(px_diffs)) if px_diffs else 0.0

                is_duplicate = same_video_hash or (mean_px_diff < min_mean_pixel_diff)
                if is_duplicate:
                    has_duplicates = True
                    issues.append(f"Demos '{id1}' and '{id2}' are duplicates (mean pixel diff = {mean_px_diff:.2f} < {min_mean_pixel_diff})")

                pairwise_results.append({
                    "demo_1": id1,
                    "demo_2": id2,
                    "same_spec": same_spec,
                    "same_video_hash": same_video_hash,
                    "same_frame_hashes": same_frame_hashes,
                    "mean_pixel_difference": mean_px_diff,
                    "is_duplicate": is_duplicate,
                })

        overall_status = "PASSED" if (not has_duplicates and len(issues) == 0) else "FAILED"

        report = {
            "task_family": task_family,
            "status": overall_status,
            "demo_ids": demo_ids,
            "pairwise_comparisons": pairwise_results,
            "issues": issues,
        }

        return overall_status == "PASSED", report

    def validate_all_demos(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate distinctness for all task families in data/demos/."""
        all_issues = []
        all_reports = {}
        overall_valid = True

        for task_family in ["open_box", "place_object"]:
            t_dir = self.demo_base_dir / task_family
            if not t_dir.exists():
                continue

            d_dirs = [p for p in t_dir.iterdir() if p.is_dir()]
            if len(d_dirs) >= 2:
                valid, rep = self.validate_distinctness(task_family, d_dirs)
                all_reports[task_family] = rep
                if not valid:
                    overall_valid = False
                    all_issues.extend(rep.get("issues", []))

        final_report = {
            "status": "PASSED" if overall_valid else "FAILED",
            "task_families": all_reports,
            "issues": all_issues,
        }

        with open(self.output_reports_dir / "demonstration_distinctness.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        return overall_valid, final_report
