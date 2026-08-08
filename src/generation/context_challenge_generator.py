import os
import json
import math
import numpy as np
from pathlib import Path
import PIL.Image as Image
import mujoco

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.environment.observation_rig import TASK_1_RIG, apply_observation_rig
from src.generation.background_randomization import sample_background_spec, apply_background_spec
from src.environment.scene_utils import (
    sample_position_on_lid,
    sample_position_in_target,
    sample_position_beside_box,
    sample_position_outside_target
)

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

        box_pose = [0.0, -0.2, 0.0]
        target_pos = [0.0, 0.3, 0.0]

        # Initial pass to get frame data
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
            objects.append({"name": "blocker", "type": "tea_box", "pos": lid_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})
        else: # Lid clear
            beside_pos = sample_position_beside_box(ref_model, ref_data, rng, offset_x=-0.3, offset_y=-0.15, height_above_table=0.04).tolist()
            objects.append({"name": "blocker", "type": "tea_box", "pos": beside_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})

        if state in ["B", "C"]: # Target blocked
            occ_pos = sample_position_in_target(ref_model, ref_data, rng, x_frac=0.0, y_frac=0.0, height_above=0.07).tolist()
            objects.append({"name": "occupant", "type": "sugar_box", "pos": occ_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})
        else: # Target clear
            out_pos = sample_position_outside_target(ref_model, ref_data, rng, offset_x=0.3, offset_y=0.0, height_above=0.07).tolist()
            objects.append({"name": "occupant", "type": "sugar_box", "pos": out_pos, "quat": _yaw_quat(float(rng.uniform(-math.pi, math.pi)))})

        # Render Scene
        rig = TASK_1_RIG
        model, data = self.scene_builder.create_environment(
            objects, settle_steps=300, include_robot=True, robot_base_pose=rig.robot_base_pose,
            box_pose=box_pose, target_region_pos=target_pos
        )
        
        bg_spec = sample_background_spec("bg_neutral_wood", rng, n_lights=model.nlight)
        apply_background_spec(model, bg_spec)
        cam_meta = apply_observation_rig(model, data, rig)
        mujoco.mj_forward(model, data)

        renderer = OffscreenRenderer(model, width=self.width, height=self.height, camera_name=self.camera_name)
        rgb = renderer.render_rgb(data)

        # Generate Task 1 Masks
        cand_geoms_1 = self._get_body_geom_names(model, "blocker")
        cand_mask_1 = renderer.render_culprit_mask(data, cand_geoms_1)
        targ_mask_1 = renderer.render_region_mask(data, ["B1_lid_panel"])
        causal_mask_1 = np.maximum(cand_mask_1, targ_mask_1) if state in ["A", "C"] else np.zeros_like(cand_mask_1)

        # Generate Task 2 Masks
        cand_geoms_2 = self._get_body_geom_names(model, "occupant")
        cand_mask_2 = renderer.render_culprit_mask(data, cand_geoms_2)
        targ_mask_2 = renderer.render_region_mask(data, ["target_region_geom"])
        causal_mask_2 = np.maximum(cand_mask_2, targ_mask_2) if state in ["B", "C"] else np.zeros_like(cand_mask_2)

        renderer.close()

        rgb_path = scene_dir / "rgb.png"
        mask1_path = scene_dir / "task1_causal_mask.png"
        mask2_path = scene_dir / "task2_causal_mask.png"
        
        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(causal_mask_1).save(mask1_path)
        Image.fromarray(causal_mask_2).save(mask2_path)

        records = []
        
        # Task 1 Record
        records.append({
            "pair_id": f"context_{scene_id}",
            "sample_id": f"{scene_id}_t1",
            "task_id": "task_1",
            "instruction": "Open the box.",
            "label": "STOP" if state in ["A", "C"] else "PROCEED",
            "split": "id_val",
            "query_rgb_path": str(rgb_path),
            "causal_mask_path": str(mask1_path),
            "demonstration_id": "demo_t1_open_box",
            "demonstration_video_path": "data/demonstrations/demo_t1_open_box.mp4",
            "state": state
        })

        # Task 2 Record
        records.append({
            "pair_id": f"context_{scene_id}",
            "sample_id": f"{scene_id}_t2",
            "task_id": "task_2",
            "instruction": "Place object1 in the target region.",
            "label": "STOP" if state in ["B", "C"] else "PROCEED",
            "split": "id_val",
            "query_rgb_path": str(rgb_path),
            "causal_mask_path": str(mask2_path),
            "demonstration_id": "demo_t2_place_object",
            "demonstration_video_path": "data/demonstrations/demo_t2_place_object.mp4",
            "state": state
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
