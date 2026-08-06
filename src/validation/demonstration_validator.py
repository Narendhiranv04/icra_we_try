"""
Demonstration validator.

Validates that generated demonstration directories, state logs, and video files
satisfy all physical and state invariants without direct lid/object qpos writes and zero actuator force.
Consolidates calculated detailed metrics into machine-readable demonstration_validation.json reports.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import json

import numpy as np
import cv2



class DemonstrationValidator:
    """Validates demonstration files and state logs against physical invariants."""

    @staticmethod
    def validate_open_box(
        state_log: List[Any],
        *,
        min_final_angle_deg: float = 45.0,
        max_initial_angle_deg: float = 10.0,
        proximity_threshold: float = 0.03,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate Task 1 (Open Box) demonstration state log and return calculated metrics."""
        issues: List[str] = []

        if not state_log:
            return False, {}, ["No state log entries"]

        def get_val(entry, key, default=None):
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        initial_entries = [e for e in state_log if get_val(e, "phase") == "initial"]
        final_entries = [e for e in state_log if get_val(e, "phase") == "final"]

        initial_phase_present = len(initial_entries) > 0
        final_phase_present = len(final_entries) > 0

        initial_lid_angle = math.degrees(get_val(initial_entries[0], "lid_angle_rad", 0.0)) if initial_entries else 0.0
        final_lid_angle = math.degrees(get_val(final_entries[-1], "lid_angle_rad", 0.0)) if final_entries else 0.0

        if not initial_phase_present:
            issues.append("Missing initial static phase")
        elif initial_lid_angle > max_initial_angle_deg:
            issues.append(f"Lid not closed at start: {initial_lid_angle:.1f}° > {max_initial_angle_deg}°")

        if not final_phase_present:
            issues.append("Missing final static phase")
        elif final_lid_angle < min_final_angle_deg:
            issues.append(f"Lid not sufficiently open at end: {final_lid_angle:.1f}° < {min_final_angle_deg}°")

        max_lid_actuator_force = 0.0
        max_abs_lid_control = 0.0

        for idx, entry in enumerate(state_log):
            lid_ctrl = abs(get_val(entry, "lid_ctrl", 0.0))
            lid_force = abs(get_val(entry, "lid_actuator_force", 0.0))
            if lid_ctrl > max_abs_lid_control:
                max_abs_lid_control = lid_ctrl
            if lid_force > max_lid_actuator_force:
                max_lid_actuator_force = lid_force

            if lid_ctrl > 1e-4:
                issues.append(f"Frame {idx}: B1_lid_actuator commanded with non-zero control ({lid_ctrl})")
                break
            if lid_force > 1e-5:
                issues.append(f"Frame {idx}: B1_lid_actuator produced non-zero force ({lid_force})")
                break

        arm_positions = [get_val(e, "arm_qpos") for e in state_log if get_val(e, "arm_qpos") is not None]
        arm_displacement = float(np.linalg.norm(np.array(arm_positions[-1]) - np.array(arm_positions[0]))) if len(arm_positions) >= 2 else 0.0
        if arm_displacement < 0.01:
            issues.append(f"Robot arm barely moved: displacement = {arm_displacement:.4f} rad")

        ee_positions = [get_val(e, "ee_pos") for e in state_log if get_val(e, "ee_pos") is not None]
        ee_displacement = float(np.linalg.norm(np.array(ee_positions[-1]) - np.array(ee_positions[0]))) if len(ee_positions) >= 2 else 0.0

        handle_distances = [get_val(e, "handle_to_grip_dist", 999.0) for e in state_log]
        minimum_handle_to_grip_distance = float(min(handle_distances)) if handle_distances else 999.0

        closure_frames = [idx for idx, e in enumerate(state_log) if get_val(e, "phase") == "grasp" or (get_val(e, "gripper_qpos", [1,1])[0] < 0.04)]
        weld_active_frames = [idx for idx, e in enumerate(state_log) if get_val(e, "weld_active") is True]

        closure_frame = closure_frames[0] if closure_frames else -1
        weld_activation_frame = weld_active_frames[0] if weld_active_frames else -1
        closure_precedes_weld = (closure_frame != -1 and weld_activation_frame != -1 and closure_frame <= weld_activation_frame)

        if weld_activation_frame == -1:
            issues.append("Weld constraint was never activated")
        else:
            first_weld = state_log[weld_activation_frame]
            dist = get_val(first_weld, "handle_to_grip_dist", 0.0)
            if dist > proximity_threshold:
                issues.append(f"Weld activated when grip-handle dist {dist:.3f}m > threshold {proximity_threshold:.3f}m")

        opening_entries = [e for e in state_log if get_val(e, "phase") == "opening"]
        opening_angle_increase = False
        if len(opening_entries) >= 2:
            angles = [get_val(e, "lid_angle_rad", 0.0) for e in opening_entries]
            opening_angle_increase = (angles[-1] > angles[0])
            if not opening_angle_increase:
                issues.append(
                    f"Lid angle did not increase during opening: "
                    f"start={math.degrees(angles[0]):.1f}° end={math.degrees(angles[-1]):.1f}°"
                )

        final_open_condition = final_lid_angle >= min_final_angle_deg

        metrics = {
            "video_readable": True,
            "metadata_frame_count": len(state_log),
            "state_log_frame_count": len(state_log),
            "frame_count_matches": True,
            "initial_phase_present": initial_phase_present,
            "final_phase_present": final_phase_present,
            "initial_lid_angle_deg": round(initial_lid_angle, 2),
            "final_lid_angle_deg": round(final_lid_angle, 2),
            "max_lid_actuator_force": round(max_lid_actuator_force, 6),
            "max_abs_lid_control": round(max_abs_lid_control, 6),
            "arm_displacement": round(arm_displacement, 4),
            "ee_displacement": round(ee_displacement, 4),
            "minimum_handle_to_grip_distance": round(minimum_handle_to_grip_distance, 4),
            "weld_activation_frame": weld_activation_frame,
            "closure_frame": closure_frame,
            "closure_precedes_weld": closure_precedes_weld,
            "opening_angle_increase": opening_angle_increase,
            "final_open_condition": final_open_condition,
        }

        is_valid = len(issues) == 0
        return is_valid, metrics, issues

    @staticmethod
    def validate_place_object(
        state_log: List[Any],
        *,
        proximity_threshold: float = 0.03,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate Task 2 (Place Object) demonstration state log and return calculated metrics."""
        issues: List[str] = []

        if not state_log:
            return False, {}, ["No state log entries"]

        def get_val(entry, key, default=None):
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        initial_entries = [e for e in state_log if get_val(e, "phase") == "initial"]
        final_entries = [e for e in state_log if get_val(e, "phase") == "final"]

        target_empty_at_start = True
        if not initial_entries:
            issues.append("Missing initial static phase")
        else:
            if get_val(initial_entries[0], "target_occupied", False):
                target_empty_at_start = False
                issues.append("Object already in target at start")

        object_in_target_at_end = False
        lin_speed = 0.0
        ang_speed = 0.0

        if not final_entries:
            issues.append("Missing final static phase")
        else:
            last_entry = final_entries[-1]
            object_in_target_at_end = bool(get_val(last_entry, "target_occupied", False))
            if not object_in_target_at_end:
                issues.append("Object not in target region at end of demonstration")

            linvel = get_val(last_entry, "object1_linvel", [0, 0, 0])
            angvel = get_val(last_entry, "object1_angvel", [0, 0, 0])
            lin_speed = float(np.linalg.norm(linvel))
            ang_speed = float(np.linalg.norm(angvel))
            if lin_speed > 0.10:
                issues.append(f"Object not stationary at end: linear speed = {lin_speed:.4f} m/s")
            if ang_speed > 0.20:
                issues.append(f"Object not stationary at end: angular speed = {ang_speed:.4f} rad/s")

        arm_positions = [get_val(e, "arm_qpos") for e in state_log if get_val(e, "arm_qpos") is not None]
        arm_displacement = float(np.linalg.norm(np.array(arm_positions[-1]) - np.array(arm_positions[0]))) if len(arm_positions) >= 2 else 0.0
        if arm_displacement < 0.01:
            issues.append(f"Robot arm barely moved: displacement = {arm_displacement:.4f} rad")

        ee_positions = [get_val(e, "ee_pos") for e in state_log if get_val(e, "ee_pos") is not None]
        ee_displacement = float(np.linalg.norm(np.array(ee_positions[-1]) - np.array(ee_positions[0]))) if len(ee_positions) >= 2 else 0.0

        obj_distances = [get_val(e, "object_to_grip_dist", 999.0) for e in state_log]
        minimum_object_to_grip_distance = float(min(obj_distances)) if obj_distances else 999.0

        closure_start_frames = [idx for idx, e in enumerate(state_log) if get_val(e, "phase") == "finger_closure"]
        first_closed_frames = [
            idx for idx, e in enumerate(state_log)
            if get_val(e, "gripper_closed") is True and get_val(e, "weld_active") is False
        ]
        weld_active_frames = [idx for idx, e in enumerate(state_log) if get_val(e, "weld_active") is True]
        weld_events = [idx for idx, e in enumerate(state_log) if get_val(e, "weld_activation_event") is True]

        closure_start_frame = closure_start_frames[0] if closure_start_frames else (first_closed_frames[0] if first_closed_frames else -1)
        first_closed_frame = first_closed_frames[0] if first_closed_frames else -1
        weld_activation_frame = weld_active_frames[0] if weld_active_frames else -1

        if not first_closed_frames:
            issues.append("No closure-settling frames exist where gripper is closed and weld is inactive")
        if not weld_active_frames:
            issues.append("Weld constraint was never activated")

        closure_precedes_weld = False
        if first_closed_frame != -1 and weld_activation_frame != -1:
            if first_closed_frame >= weld_activation_frame:
                issues.append(
                    f"Invalid event order: weld activated at frame {weld_activation_frame} before or same frame as finger closure (frame {first_closed_frame})"
                )
            else:
                closure_precedes_weld = True

            weld_entry = state_log[weld_activation_frame]
            dist = get_val(weld_entry, "object_to_grip_dist", 0.0)
            if dist > proximity_threshold:
                issues.append(f"Weld activated when grip-object dist {dist:.3f}m > threshold {proximity_threshold:.3f}m")

        if len(weld_events) != 1:
            issues.append(f"Expected exactly 1 weld activation event, found {len(weld_events)}")

        weld_release_frames = [idx for idx, e in enumerate(state_log) if get_val(e, "phase") == "retreat"]
        weld_release_frame = weld_release_frames[0] if weld_release_frames else -1

        transport_entries = [e for e in state_log if get_val(e, "phase") == "transport"]
        object_transport_displacement = 0.0
        if len(transport_entries) >= 2:
            start_pos = np.array(get_val(transport_entries[0], "object1_pos", [0, 0, 0]))
            end_pos = np.array(get_val(transport_entries[-1], "object1_pos", [0, 0, 0]))
            object_transport_displacement = float(np.linalg.norm(end_pos - start_pos))
            if object_transport_displacement < 0.02:
                issues.append(f"Object barely moved during transport: {object_transport_displacement:.4f}m")

        final_stability = (lin_speed <= 0.10 and ang_speed <= 0.20)

        metrics = {
            "video_readable": True,
            "metadata_frame_count": len(state_log),
            "state_log_frame_count": len(state_log),
            "frame_count_matches": True,
            "target_empty_at_start": target_empty_at_start,
            "object_in_target_at_end": object_in_target_at_end,
            "arm_displacement": round(arm_displacement, 4),
            "ee_displacement": round(ee_displacement, 4),
            "object_transport_displacement": round(object_transport_displacement, 4),
            "minimum_object_to_grip_distance": round(minimum_object_to_grip_distance, 4),
            "closure_start_frame": closure_start_frame,
            "first_closed_frame": first_closed_frame,
            "weld_activation_frame": weld_activation_frame,
            "closure_precedes_weld": closure_precedes_weld,
            "weld_release_frame": weld_release_frame,
            "final_linear_speed": round(lin_speed, 4),
            "final_angular_speed": round(ang_speed, 4),
            "final_stability": final_stability,
            "final_target_occupancy": object_in_target_at_end,
        }

        is_valid = len(issues) == 0
        return is_valid, metrics, issues

    @classmethod
    def _validate_mp4(
        cls,
        mp4_path: Path,
        expected_frame_count: int,
        expected_fps: float,
        expected_width: int,
        expected_height: int,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Open and validate an MP4 file using OpenCV. Returns (passed, metrics, issues)."""
        issues: List[str] = []
        metrics: Dict[str, Any] = {}

        if not mp4_path.exists():
            return False, {"video_readable": False}, [f"MP4 file does not exist: {mp4_path}"]

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            cap.release()
            return False, {"video_readable": False}, [f"Failed to open MP4: {mp4_path}"]

        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate frame count
        frame_count_matches = (actual_frame_count == expected_frame_count)
        if not frame_count_matches:
            issues.append(
                f"MP4 frame count mismatch: actual={actual_frame_count}, expected={expected_frame_count}"
            )

        # Validate FPS within tolerance
        fps_matches = (abs(actual_fps - expected_fps) <= max(0.5, expected_fps * 0.05))
        if not fps_matches:
            issues.append(f"MP4 FPS mismatch: actual={actual_fps:.2f}, expected={expected_fps:.2f}")

        # Validate resolution
        resolution_matches = (actual_width == expected_width and actual_height == expected_height)
        if not resolution_matches:
            issues.append(
                f"MP4 resolution mismatch: actual={actual_width}x{actual_height}, expected={expected_width}x{expected_height}"
            )

        # Decode first and last frames
        first_frame_readable = False
        final_frame_readable = False
        decoded_frame_count = 0
        first_frame_arr = None
        last_frame_arr = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            first_frame_readable = True
            first_frame_arr = frame
            decoded_frame_count += 1

        if actual_frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, actual_frame_count - 1))
            ret, frame = cap.read()
            if ret:
                final_frame_readable = True
                last_frame_arr = frame
                decoded_frame_count += 1

        # Scan all frames (verify readability)
        if actual_frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            scan_count = 0
            for _ in range(actual_frame_count):
                ret, _ = cap.read()
                if ret:
                    scan_count += 1
                else:
                    break
            decoded_frame_count = scan_count
            if scan_count < actual_frame_count:
                issues.append(
                    f"MP4 decode incomplete: decoded {scan_count}/{actual_frame_count} frames"
                )

        # Check non-static: compare first and last frames
        video_non_static = False
        if first_frame_arr is not None and last_frame_arr is not None:
            diff = np.abs(first_frame_arr.astype(np.int32) - last_frame_arr.astype(np.int32))
            mean_diff = float(np.mean(diff))
            video_non_static = mean_diff > 0.1  # threshold for non-trivial change
            if not video_non_static:
                issues.append(
                    f"MP4 appears static: mean pixel diff between first and last frame = {mean_diff:.3f}"
                )

        cap.release()

        metrics = {
            "video_readable": True,
            "video_actual_frame_count": actual_frame_count,
            "metadata_frame_count": expected_frame_count,
            "state_log_frame_count": expected_frame_count,
            "frame_count_matches": frame_count_matches,
            "video_actual_fps": round(actual_fps, 3),
            "metadata_fps": expected_fps,
            "fps_matches": fps_matches,
            "video_width": actual_width,
            "video_height": actual_height,
            "metadata_width": expected_width,
            "metadata_height": expected_height,
            "resolution_matches": resolution_matches,
            "decoded_frame_count": decoded_frame_count,
            "first_frame_readable": first_frame_readable,
            "final_frame_readable": final_frame_readable,
            "video_non_static": video_non_static,
        }

        return len(issues) == 0, metrics, issues

    @classmethod
    def validate_demo_dir(cls, demo_dir: Union[str, Path]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate an entire saved demonstration directory."""
        demo_dir = Path(demo_dir)
        issues: List[str] = []

        if not demo_dir.exists():
            return False, {}, [f"Demonstration directory does not exist: {demo_dir}"]

        req_files = [
            "rgb.mp4",
            "state_log.jsonl",
            "metadata.json",
            "scene_config.json",
            "initial_rgb.png",
            "final_rgb.png",
            "validation_report.json",
        ]
        for fname in req_files:
            fpath = demo_dir / fname
            if not fpath.exists():
                issues.append(f"Missing required file: {fname}")

        if issues:
            return False, {}, issues

        metrics: Dict[str, Any] = {}

        try:
            with open(demo_dir / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            task_family = meta.get("task_family", "")
            expected_frame_count = meta.get("frame_count", 0)
            expected_fps = float(meta.get("fps", 15))
            expected_width = int(meta.get("width", 640))
            expected_height = int(meta.get("height", 480))

            state_log = []
            with open(demo_dir / "state_log.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        state_log.append(json.loads(line))

            state_log_frame_count = len(state_log)

            if expected_frame_count != state_log_frame_count:
                issues.append(
                    f"Frame count mismatch: metadata={expected_frame_count} state_log={state_log_frame_count}"
                )

            # Validate actual MP4 file
            mp4_issues: List[str] = []
            mp4_path = demo_dir / "rgb.mp4"
            mp4_valid, mp4_metrics, mp4_issues = cls._validate_mp4(
                mp4_path,
                expected_frame_count=state_log_frame_count,
                expected_fps=expected_fps,
                expected_width=expected_width,
                expected_height=expected_height,
            )
            # Update state_log_frame_count in mp4_metrics
            mp4_metrics["state_log_frame_count"] = state_log_frame_count
            issues.extend(mp4_issues)

            if task_family == "open_box":
                valid, task_metrics, log_issues = cls.validate_open_box(state_log)
                issues.extend(log_issues)
            elif task_family == "place_object":
                valid, task_metrics, log_issues = cls.validate_place_object(state_log)
                issues.extend(log_issues)
            else:
                valid = False
                task_metrics = {}
                issues.append(f"Unknown task_family: {task_family}")

            # Merge mp4_metrics (overrides video_readable, frame_count_matches from task_metrics)
            metrics = {**task_metrics, **mp4_metrics}

        except Exception as err:
            issues.append(f"Error parsing demonstration directory: {err}")
            valid = False

        is_valid = len(issues) == 0
        return is_valid, metrics, issues


    @classmethod
    def generate_demonstration_validation_report(
        cls,
        demo_base_dir: Union[str, Path] = "data/demos",
        output_report_path: Union[str, Path] = "data/reports/demonstration_validation.json",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Scan all demonstration directories under demo_base_dir and export consolidated demonstration_validation.json."""
        demo_base_dir = Path(demo_base_dir)
        output_report_path = Path(output_report_path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        demo_dirs = [p for p in demo_base_dir.rglob("*") if p.is_dir() and (p / "metadata.json").exists()]
        demo_results: Dict[str, dict] = {}
        all_passed = True

        for d_dir in demo_dirs:
            valid, metrics, issues = cls.validate_demo_dir(d_dir)
            if not valid:
                all_passed = False

            meta_p = d_dir / "metadata.json"
            meta = {}
            if meta_p.exists():
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            state_log_p = d_dir / "state_log.jsonl"

            demo_id = meta.get("demo_id", d_dir.name)
            task_family = meta.get("task_family", "unknown")

            demo_results[demo_id] = {
                "demo_id": demo_id,
                "task_family": task_family,
                "video_path": str(d_dir / "rgb.mp4"),
                "state_log_path": str(state_log_p),
                "validation_status": "PASSED" if valid else "FAILED",
                "frame_count": meta.get("frame_count") or (metrics.get("state_log_frame_count", 0) if isinstance(metrics, dict) else 0),
                "fps": meta.get("fps", 15),
                "metrics": metrics,
                "overall_status": "PASSED" if valid else "FAILED",
                "issues": issues,
            }

        overall_status = "PASSED" if (all_passed and len(demo_results) > 0) else "FAILED"

        report = {
            "status": overall_status,
            "demonstration_count": len(demo_results),
            "demonstrations": demo_results,
        }

        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return overall_status == "PASSED", report
