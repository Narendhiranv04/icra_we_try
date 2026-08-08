import os
import json
import math
import random
import numpy as np
from pathlib import Path
import PIL.Image as Image
import mujoco
import hashlib

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.validation.occupancy_checks import check_lid_occupancy, check_target_occupancy
from src.environment.observation_rig import CONTEXT_CHALLENGE_RIG, apply_observation_rig
from src.generation.background_randomization import sample_background_spec, apply_background_spec, SPLIT_BACKGROUNDS
from src.environment.scene_utils import (
    sample_position_on_lid,
    sample_position_in_target,
    sample_position_beside_box,
    sample_position_outside_target
)

VALID_BLOCKERS = ["tea_box", "coffee_can", "mug", "sugar_box"]
VALID_OCCUPANTS = ["sugar_box", "mug", "bowl", "cup"]
VALID_BGS = [SPLIT_BACKGROUNDS["id"]]

def _yaw_quat(yaw: float) -> list:
    half = yaw / 2.0
    return [float(math.cos(half)), 0.0, 0.0, float(math.sin(half))]

class ContextChallengeGenerator:
    def __init__(self, output_dir="data/context_challenge", resolution=(640, 480)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width, self.height = resolution
        self.scene_builder = SceneBuilder()
        self.camera_name = "robot0:ego_camera"

    def _get_body_geom_names(self, model: mujoco.MjModel, body_name: str) -> list:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return [f"{body_name}_visual", f"{body_name}_geom"]
        geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body_id and model.geom_group[g] != 3:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                if name:
                    geoms.append(name)
        return geoms if geoms else [f"{body_name}_visual"]

    def generate_scene(self, scene_id: str, state: str, seed: int = 42):
        """
        States:
        A: Lid blocked, Target clear (Task 1: STOP, Task 2: PROCEED)
        B: Lid clear, Target blocked (Task 1: PROCEED, Task 2: STOP)
        C: Lid blocked, Target blocked (Task 1: STOP, Task 2: STOP)
        D: Lid clear, Target clear (Task 1: PROCEED, Task 2: PROCEED)
        """
        scene_dir = self.output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        while True:
            # Randomize assets
            blocker_type = rng.choice(VALID_BLOCKERS)
            occupant_type = rng.choice(VALID_OCCUPANTS)
            bg = rng.choice(VALID_BGS)

            box_pose = [0.0, -0.2, 0.58]
            target_pos = [0.0, 0.3, 0.0]

            ref_model, ref_data = self.scene_builder.create_environment(
                settle_steps=0, include_robot=True, robot_base_pose="home",
                box_pose=box_pose, target_region_pos=target_pos
            )

            objects = [
                {"name": "coffee_can", "type": "coffee_can",
                 "pos": sample_position_outside_target(ref_model, ref_data, rng, offset_x=-0.25, offset_y=0.1, height_above=0.07).tolist()}
            ]

            if state in ["A", "C"]: # Lid blocked
                lid_pos = sample_position_on_lid(ref_model, ref_data, rng, x_frac=0.0, y_frac=0.0, height_above=0.02).tolist()
                objects.append({"name": "blocker", "type": blocker_type, "pos": lid_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})
            else: # Lid clear
                beside_pos = sample_position_beside_box(ref_model, ref_data, rng, offset_x=-0.3, offset_y=-0.15, height_above_table=0.04).tolist()
                objects.append({"name": "blocker", "type": blocker_type, "pos": beside_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})

            if state in ["B", "C"]: # Target blocked
                occ_pos = sample_position_in_target(ref_model, ref_data, rng, x_frac=0.0, y_frac=0.0, height_above=0.07).tolist()
                objects.append({"name": "occupant", "type": occupant_type, "pos": occ_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})
            else: # Target clear
                out_pos = sample_position_outside_target(ref_model, ref_data, rng, offset_x=0.3, offset_y=0.0, height_above=0.07).tolist()
                objects.append({"name": "occupant", "type": occupant_type, "pos": out_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})

            # Render Scene
            rig = CONTEXT_CHALLENGE_RIG
            model, data = self.scene_builder.create_environment(
                objects, settle_steps=300, include_robot=True, robot_base_pose=rig.robot_base_pose,
                box_pose=box_pose, target_region_pos=target_pos
            )

            # Physics validation
            is_lid_occ, _, _ = check_lid_occupancy(model, data, blocker_names=["blocker"])
            is_targ_occ, _, _ = check_target_occupancy(model, data, candidate_objects=["occupant"])

            valid = True
            if state == "A" and (not is_lid_occ or is_targ_occ): valid = False
            if state == "B" and (is_lid_occ or not is_targ_occ): valid = False
            if state == "C" and (not is_lid_occ or not is_targ_occ): valid = False
            if state == "D" and (is_lid_occ or is_targ_occ): valid = False

            if valid:
                break

        bg_spec = sample_background_spec(bg, rng, n_lights=model.nlight)
        apply_background_spec(model, bg_spec)
        cam_meta = apply_observation_rig(model, data, rig)
        mujoco.mj_forward(model, data)

        # Save actual occupancy for provenance
        actual_lid_occ = bool(is_lid_occ)
        actual_targ_occ = bool(is_targ_occ)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height, camera_name=self.camera_name)
        rgb = renderer.render_rgb(data)

        # Generate Task 1 Masks
        cand_geoms_1 = self._get_body_geom_names(model, "blocker")
        cand_mask_1 = renderer.render_culprit_mask(data, cand_geoms_1)
        targ_mask_1 = renderer.render_region_mask(data, ["B1_lid_panel"])

        # Visibility validation
        if targ_mask_1.sum() < 500:
            renderer.close()
            del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
            return self.generate_scene(scene_id, state, seed=seed+1)
        if state in ["A", "C"] and cand_mask_1.sum() < 50:
            renderer.close()
            del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
            return self.generate_scene(scene_id, state, seed=seed+1)
        causal_mask_1 = np.maximum(cand_mask_1, targ_mask_1) if state in ["A", "C"] else np.zeros_like(cand_mask_1)

        # Validate task 1 mask
        if state in ["A", "C"] and causal_mask_1.sum() == 0:
            renderer.close()
            del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
            return self.generate_scene(scene_id, state, seed=seed+1)

        # Generate Task 2 Masks
        cand_geoms_2 = self._get_body_geom_names(model, "occupant")
        cand_mask_2 = renderer.render_culprit_mask(data, cand_geoms_2)
        targ_mask_2 = renderer.render_region_mask(data, ["target_region_geom"])

        if targ_mask_2.sum() < 500:
            renderer.close()
            del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
            return self.generate_scene(scene_id, state, seed=seed+1)
        if state in ["B", "C"] and cand_mask_2.sum() < 50:
            renderer.close()
            del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
            return self.generate_scene(scene_id, state, seed=seed+1)
        causal_mask_2 = np.maximum(cand_mask_2, targ_mask_2) if state in ["B", "C"] else np.zeros_like(cand_mask_2)

        # Validate task 2 mask
        if state in ["B", "C"] and causal_mask_2.sum() == 0:
            renderer.close()
            del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
            return self.generate_scene(scene_id, state, seed=seed+1)

        renderer.close()
        
        rgb_path = scene_dir / "rgb.png"
        mask1_path = scene_dir / "task1_causal_mask.png"
        mask2_path = scene_dir / "task2_causal_mask.png"

        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(causal_mask_1).save(mask1_path)
        Image.fromarray(causal_mask_2).save(mask2_path)
        # Calculate SHA256 for RGB
        with open(rgb_path, "rb") as f:
            rgb_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Read pilot demos from index
        t1_demos = []
        t2_demos = []
        index_path = Path("learning_data/index.jsonl")
        if index_path.exists():
            with open(index_path) as f:
                for line in f:
                    rec = json.loads(line)
                    # Use only real valid demos that exist on disk
                    demo_path = rec["demonstration_video_path"]
                    if os.path.exists(demo_path):
                        if rec["task_id"] == "task_1" and rec["demonstration_id"] not in [d[0] for d in t1_demos]:
                            t1_demos.append((rec["demonstration_id"], demo_path))
                        if rec["task_id"] == "task_2" and rec["demonstration_id"] not in [d[0] for d in t2_demos]:
                            t2_demos.append((rec["demonstration_id"], demo_path))

        # We need EXACTLY 3 valid demos for each task according to prompt requirements
        # Sort to ensure determinism
        t1_demos = sorted(t1_demos, key=lambda x: x[0])[:3]
        t2_demos = sorted(t2_demos, key=lambda x: x[0])[:3]

        if len(t1_demos) < 3 or len(t2_demos) < 3:
            raise RuntimeError(f"Missing required real pilot demos. Found {len(t1_demos)} Task 1 and {len(t2_demos)} Task 2 demos.")

        t1_demo = t1_demos[rng.integers(0, len(t1_demos))]
        t2_demo = t2_demos[rng.integers(0, len(t2_demos))]

        records = []

        # Task 1 Record
        records.append({
            "pair_id": f"context_{scene_id}",
            "sample_id": f"{scene_id}_t1",
            "task_id": "task_1",
            "instruction": "Open the box.",
            "label": "STOP" if state in ["A", "C"] else "PROCEED",
            "split": "context_challenge",
            "query_rgb_path": str(rgb_path),
            "query_rgb_sha256": rgb_sha256,
            "causal_mask_path": str(mask1_path),
            "demonstration_id": t1_demo[0],
            "demonstration_video_path": t1_demo[1],
            "state": state,
            "measured_lid_occupancy": actual_lid_occ,
            "measured_target_occupancy": actual_targ_occ,
            "camera_rig_id": rig.rig_id
        })

        # Task 2 Record
        records.append({
            "pair_id": f"context_{scene_id}",
            "sample_id": f"{scene_id}_t2",
            "task_id": "task_2",
            "instruction": "Place object1 in the target region.",
            "label": "STOP" if state in ["B", "C"] else "PROCEED",
            "split": "context_challenge",
            "query_rgb_path": str(rgb_path),
            "query_rgb_sha256": rgb_sha256,
            "causal_mask_path": str(mask2_path),
            "demonstration_id": t2_demo[0],
            "demonstration_video_path": t2_demo[1],
            "state": state,
            "measured_lid_occupancy": actual_lid_occ,
            "measured_target_occupancy": actual_targ_occ,
            "camera_rig_id": rig.rig_id
        })

        return records

if __name__ == "__main__":
    generator = ContextChallengeGenerator()
    manifest_path = Path("data/manifests/context_challenge_manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []

    idx = 0
    for state in ["A", "B", "C", "D"]:
        for i in range(10): # 10 physical scenes per state
            seed = 1000 + idx
            scene_id = f"scene_{idx:03d}_{state}"
            records = generator.generate_scene(scene_id, state, seed=seed)
            all_records.extend(records)
            idx += 1
            print(f"Generated {scene_id}")

    with open(manifest_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
