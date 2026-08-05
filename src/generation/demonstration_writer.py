"""
Structured demonstration writer and artifact exporter.

Saves complete demonstration data directories containing:
- rgb.mp4
- state_log.jsonl
- metadata.json
- scene_config.json
- initial_rgb.png
- final_rgb.png
- validation_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import PIL.Image as Image
import imageio


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


class DemonstrationWriter:
    """Writer for saving complete structured demonstration directories and metadata."""

    def __init__(self, base_dir: Union[str, Path] = "data/demos"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_demonstration(
        self,
        task_family: str,
        demo_id: str,
        frames: List[np.ndarray],
        state_log: List[Any],
        scene_spec: Dict[str, Any],
        validation_result: Dict[str, Any],
        fps: int = 15,
    ) -> Path:
        """Save a complete demonstration directory with video, logs, images, and validation report."""
        demo_dir = self.base_dir / task_family / demo_id
        demo_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save MP4 video
        video_path = demo_dir / "rgb.mp4"
        imageio.mimsave(str(video_path), frames, fps=fps, quality=8)

        # 2. Save initial & final RGB PNGs
        if frames:
            Image.fromarray(frames[0]).save(demo_dir / "initial_rgb.png")
            Image.fromarray(frames[-1]).save(demo_dir / "final_rgb.png")

        # 3. Save state_log.jsonl
        state_log_path = demo_dir / "state_log.jsonl"
        with open(state_log_path, "w", encoding="utf-8") as f:
            for entry in state_log:
                if hasattr(entry, "__dict__"):
                    d = entry.__dict__
                elif isinstance(entry, dict):
                    d = entry
                else:
                    d = {"raw": str(entry)}
                clean_d = _to_json_serializable(d)
                f.write(json.dumps(clean_d) + "\n")

        # 4. Save metadata.json
        meta_path = demo_dir / "metadata.json"
        metadata = {
            "demo_id": demo_id,
            "task_family": task_family,
            "frame_count": len(frames),
            "fps": fps,
            "resolution": [frames[0].shape[1], frames[0].shape[0]] if frames else [640, 480],
            "video_path": str(video_path),
            "state_log_path": str(state_log_path),
            "is_valid": validation_result.get("is_valid", False),
            "validation_issues": validation_result.get("issues", []),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_serializable(metadata), f, indent=2)

        # 5. Save scene_config.json
        scene_config_path = demo_dir / "scene_config.json"
        with open(scene_config_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_serializable(scene_spec), f, indent=2)

        # 6. Save validation_report.json
        val_report_path = demo_dir / "validation_report.json"
        with open(val_report_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_serializable(validation_result), f, indent=2)

        return demo_dir
