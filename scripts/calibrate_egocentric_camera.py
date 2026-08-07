"""
Camera calibration script for calculating and evaluating optimal task-specific egocentric camera gaze.
Generates grid evaluation images and saves optimal rig configurations to camera_calibration/results.json.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import mujoco
import numpy as np

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer


def compute_head_gaze(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
) -> Tuple[float, float]:
    """Calculate analytical head pan and head tilt angles to align camera with target point."""
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_pan_joint")
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_tilt_joint")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot0:ego_camera")

    # Set neutral head orientation
    data.qpos[model.jnt_qposadr[pan_id]] = 0.0
    data.qpos[model.jnt_qposadr[tilt_id]] = 0.0
    mujoco.mj_forward(model, data)

    cam_pos = data.cam_xpos[cam_id].copy()
    diff = target_pos - cam_pos
    dx, dy, dz = diff[0], diff[1], diff[2]

    # In Fetch robot base frame (+X right, +Y forward, +Z up):
    pan = math.atan2(dx, dy)
    dist_xy = math.hypot(dx, dy)
    tilt = math.atan2(-dz, dist_xy)

    # Clamp to physical joint limits
    pan = float(np.clip(pan, -1.50, 1.50))
    tilt = float(np.clip(tilt, 0.0, 1.20))
    return pan, tilt


def score_camera_pose(
    renderer: OffscreenRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geoms: List[str],
    candidate_geoms: List[str],
) -> Dict[str, Any]:
    """Score a camera pose based on target visibility, candidate visibility, ROI distance, and torso occlusion."""
    seg = renderer.render_segmentation(data)
    geom_map = seg[:, :, 0]

    # Target pixels
    target_mask = np.zeros_like(geom_map, dtype=bool)
    for gname in target_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid != -1:
            target_mask |= (geom_map == gid)
    target_pixels = int(np.count_nonzero(target_mask))

    # Candidate pixels
    cand_mask = np.zeros_like(geom_map, dtype=bool)
    for gname in candidate_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid != -1:
            cand_mask |= (geom_map == gid)
    candidate_pixels = int(np.count_nonzero(cand_mask))

    # Torso occlusion pixels
    torso_mask = np.zeros_like(geom_map, dtype=bool)
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "torso" in gname or "base" in gname or "shoulder" in gname:
            torso_mask |= (geom_map == i)
    torso_pixels = int(np.count_nonzero(torso_mask))
    total_pixels = geom_map.shape[0] * geom_map.shape[1]
    torso_fraction = float(torso_pixels / total_pixels)

    # ROI center distance
    roi_mask = target_mask | cand_mask
    if np.any(roi_mask):
        ys, xs = np.nonzero(roi_mask)
        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))
        center_x = geom_map.shape[1] / 2.0
        center_y = geom_map.shape[0] / 2.0
        roi_dist = float(np.hypot(mean_x - center_x, mean_y - center_y) / np.hypot(center_x, center_y))
    else:
        roi_dist = 1.0

    # Composite score
    score = (target_pixels + candidate_pixels) - (1000.0 * roi_dist) - (5000.0 * torso_fraction)

    return {
        "score": float(score),
        "target_pixels": target_pixels,
        "candidate_pixels": candidate_pixels,
        "torso_fraction": torso_fraction,
        "roi_distance": roi_dist,
    }


def calibrate_task(
    task_id: str,
    target_pos: np.ndarray,
    target_geoms: List[str],
    candidate_geoms: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Perform calibration grid search for a given task and save visualization artifacts."""
    sb = SceneBuilder()
    model, data = sb.create_environment(include_robot=True, robot_base_pose="home", settle_steps=50)
    renderer = OffscreenRenderer(model, width=640, height=480, camera_name="robot0:ego_camera")

    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_pan_joint")
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_tilt_joint")

    analytical_pan, analytical_tilt = compute_head_gaze(model, data, target_pos)

    # Grid search around analytical gaze
    pan_grid = np.linspace(analytical_pan - 0.20, analytical_pan + 0.20, 5)
    tilt_grid = np.linspace(analytical_tilt - 0.15, analytical_tilt + 0.15, 5)

    best_score = -1e9
    best_config = None
    best_rgb = None
    grid_renders = []

    for p in pan_grid:
        row_renders = []
        for t in tilt_grid:
            data.qpos[model.jnt_qposadr[pan_id]] = float(p)
            data.qpos[model.jnt_qposadr[tilt_id]] = float(t)
            mujoco.mj_forward(model, data)

            metrics = score_camera_pose(renderer, model, data, target_geoms, candidate_geoms)
            rgb = renderer.render_rgb(data)

            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_config = {
                    "task_id": task_id,
                    "head_pan": float(p),
                    "head_tilt": float(t),
                    "analytical_pan": float(analytical_pan),
                    "analytical_tilt": float(analytical_tilt),
                    "metrics": metrics,
                }
                best_rgb = rgb.copy()

            small_rgb = cv2.resize(rgb, (160, 120))
            row_renders.append(small_rgb)
        grid_renders.append(np.hstack(row_renders))

    grid_img = np.vstack(grid_renders)
    cv2.imwrite(str(out_dir / f"{task_id}_grid.png"), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    if best_rgb is not None:
        cv2.imwrite(str(out_dir / f"{task_id}_best.png"), cv2.cvtColor(best_rgb, cv2.COLOR_RGB2BGR))

    renderer.close()
    return best_config


def main():
    out_dir = Path("camera_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Calibrating Task 1 camera rig...")
    t1_target = np.array([0.52, 0.18, 0.65])  # Box lid midpoint
    t1_config = calibrate_task("task1", t1_target, ["B1_lid_panel"], ["blocker1_geom"], out_dir)

    print("Calibrating Task 2 camera rig...")
    t2_target = np.array([-0.05, 0.0, 0.50])  # Pick and place midpoint
    t2_config = calibrate_task("task2", t2_target, ["target_region_geom"], ["occupant_geom"], out_dir)

    results = {
        "task1_open_box_ego": t1_config,
        "task2_place_object_ego": t2_config,
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Calibration completed! Results saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
