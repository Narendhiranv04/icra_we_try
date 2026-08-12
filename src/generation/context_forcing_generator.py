import os
import json
import math
import random
import numpy as np
from pathlib import Path
import PIL.Image as Image
import mujoco
import hashlib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

class ContextForcingGenerator:
    def __init__(self, output_dir="data/context_forcing_v1", resolution=(640, 480)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width, self.height = resolution
        self.scene_builder = SceneBuilder()
        self.camera_name = "robot0:ego_camera"
        
        # Load Benchmark B blocklist
        freeze_path = Path("artifacts/context_forcing_v1/benchmark_b_freeze.json")
        if freeze_path.exists():
            with open(freeze_path) as f:
                freeze_data = json.load(f)
                self.blocklist_hashes = set(freeze_data.get("rgb_sha256_hashes", []))
        else:
            print("WARNING: benchmark_b_freeze.json not found! Contamination checks will fail.")
            self.blocklist_hashes = set()

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

    def generate_scene(self, scene_id: str, state: str, seed: int = 42, max_attempts: int = 50):
        scene_dir = self.output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_attempts):
            current_seed = seed + attempt
            rng = np.random.default_rng(current_seed)

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

            if not valid:
                del model, data, ref_model, ref_data
                import gc; gc.collect()
                continue

            bg_spec = sample_background_spec(bg, rng, n_lights=model.nlight)
            apply_background_spec(model, bg_spec)
            cam_meta = apply_observation_rig(model, data, rig)
            mujoco.mj_forward(model, data)

            # Save actual occupancy for provenance
            actual_lid_occ = bool(is_lid_occ)
            actual_targ_occ = bool(is_targ_occ)

            renderer = OffscreenRenderer(model, width=self.width, height=self.height, camera_name=self.camera_name)
            rgb = renderer.render_rgb(data)
            
            # Check for Benchmark B leakage early
            # Actually, to be safe, encode to PNG and then hash
            import io
            img_byte_arr = io.BytesIO()
            Image.fromarray(rgb).save(img_byte_arr, format='PNG')
            rgb_sha256 = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()
            
            if rgb_sha256 in self.blocklist_hashes:
                print(f"Warning: Hash collision with Benchmark B! Retrying scene {scene_id}.")
                renderer.close()
                del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
                continue

            # Generate Task 1 Masks
            cand_geoms_1 = self._get_body_geom_names(model, "blocker")
            cand_mask_1 = renderer.render_culprit_mask(data, cand_geoms_1)
            targ_mask_1 = renderer.render_region_mask(data, ["B1_lid_panel"])

            if targ_mask_1.sum() < 500:
                renderer.close()
                del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
                continue
            if state in ["A", "C"] and cand_mask_1.sum() < 50:
                renderer.close()
                del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
                continue
            causal_mask_1 = np.maximum(cand_mask_1, targ_mask_1) if state in ["A", "C"] else np.zeros_like(cand_mask_1)

            if state in ["A", "C"] and causal_mask_1.sum() == 0:
                renderer.close()
                del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
                continue

            # Generate Task 2 Masks
            cand_geoms_2 = self._get_body_geom_names(model, "occupant")
            cand_mask_2 = renderer.render_culprit_mask(data, cand_geoms_2)
            targ_mask_2 = renderer.render_region_mask(data, ["target_region_geom"])

            if targ_mask_2.sum() < 500:
                renderer.close()
                del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
                continue
            if state in ["B", "C"] and cand_mask_2.sum() < 50:
                renderer.close()
                del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
                continue
            causal_mask_2 = np.maximum(cand_mask_2, targ_mask_2) if state in ["B", "C"] else np.zeros_like(cand_mask_2)

            if state in ["B", "C"] and causal_mask_2.sum() == 0:
                renderer.close()
                del renderer, model, data, ref_model, ref_data; import gc; gc.collect()
                continue

            renderer.close()
        
            rgb_path = scene_dir / "rgb.png"
            mask1_path = scene_dir / "task1_causal_mask.png"
            mask2_path = scene_dir / "task2_causal_mask.png"

            with open(rgb_path, "wb") as f:
                f.write(img_byte_arr.getvalue())
            Image.fromarray(causal_mask_1).save(mask1_path)
            Image.fromarray(causal_mask_2).save(mask2_path)

            # Read pilot demos from index
            t1_demos = []
            t2_demos = []
            index_path = Path("learning_data/index.jsonl")
            if index_path.exists():
                with open(index_path) as f:
                    for line in f:
                        rec = json.loads(line)
                        demo_path = rec.get("demonstration_video_path")
                        if demo_path and os.path.exists(demo_path):
                            if rec["task_id"] == "task_1" and rec["demonstration_id"] not in [d[0] for d in t1_demos]:
                                t1_demos.append((rec["demonstration_id"], demo_path))
                            if rec["task_id"] == "task_2" and rec["demonstration_id"] not in [d[0] for d in t2_demos]:
                                t2_demos.append((rec["demonstration_id"], demo_path))

            t1_demos = sorted(t1_demos, key=lambda x: x[0])[:3]
            t2_demos = sorted(t2_demos, key=lambda x: x[0])[:3]

            if len(t1_demos) < 3 or len(t2_demos) < 3:
                raise RuntimeError(f"Missing required real pilot demos. Found {len(t1_demos)} Task 1 and {len(t2_demos)} Task 2 demos.")

            t1_demo = t1_demos[rng.integers(0, len(t1_demos))]
            t2_demo = t2_demos[rng.integers(0, len(t2_demos))]

            records = []
            
            # The split (context_forcing_v1_train/val) will be populated externally by the main function

            # Task 1 Record
            records.append({
                "pair_id": scene_id,
                "sample_id": f"{scene_id}_t1",
                "task_id": "task_1",
                "instruction": "Open the box.",
                "label": "STOP" if state in ["A", "C"] else "PROCEED",
                "dataset_role": "context_forcing_v1",
                "split": "TBD",
                "query_rgb_path": str(rgb_path),
                "query_rgb_sha256": rgb_sha256,
                "causal_mask_path": str(mask1_path),
                "demonstration_id": t1_demo[0],
                "demonstration_video_path": t1_demo[1],
                "state": state,
                "measured_lid_occupancy": actual_lid_occ,
                "measured_target_occupancy": actual_targ_occ,
                "camera_rig_id": rig.rig_id,
                "generation_seed": int(current_seed)
            })

            # Task 2 Record
            records.append({
                "pair_id": scene_id,
                "sample_id": f"{scene_id}_t2",
                "task_id": "task_2",
                "instruction": "Place object1 in the target region.",
                "label": "STOP" if state in ["B", "C"] else "PROCEED",
                "dataset_role": "context_forcing_v1",
                "split": "TBD",
                "query_rgb_path": str(rgb_path),
                "query_rgb_sha256": rgb_sha256,
                "causal_mask_path": str(mask2_path),
                "demonstration_id": t2_demo[0],
                "demonstration_video_path": t2_demo[1],
                "state": state,
                "measured_lid_occupancy": actual_lid_occ,
                "measured_target_occupancy": actual_targ_occ,
                "camera_rig_id": rig.rig_id,
                "generation_seed": int(current_seed)
            })

            return records

        raise RuntimeError(
            f"Exceeded max_attempts ({max_attempts}) generating scene {scene_id} in state {state}. "
            f"Initial seed: {seed}, last attempted seed: {current_seed}."
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="Generate tiny 8-scene subset")
    args = parser.parse_args()

    # Generate 160 pairs (40 per state), or 8 pairs if tiny
    n_per_state = 2 if args.tiny else 40
    out_dir = "data/context_forcing_v1_tiny" if args.tiny else "data/context_forcing_v1"
    manifest_name = "context_forcing_v1_tiny_manifest.jsonl" if args.tiny else "context_forcing_v1_manifest.jsonl"
    
    generator = ContextForcingGenerator(output_dir=out_dir)
    manifest_path = Path("data/manifests") / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    
    # Need disjoint seeds from context_challenge. It used 1000..1039. Let's use 20000+
    # We want reproducible seeds that are disjoint from benchmark B.
    idx = 0
    for state in ["A", "B", "C", "D"]:
        for i in range(n_per_state):
            seed = 20000 + idx
            scene_id = f"cf_scene_{idx:03d}_{state}"
            if args.tiny:
                scene_id = f"cftiny_scene_{idx:03d}_{state}"
            records = generator.generate_scene(scene_id, state, seed=seed)
            all_pairs.append({"pair_id": scene_id, "records": records})
            idx += 1
            print(f"Generated {scene_id}")

    # Split into train/val (80/20) based on pair_ids
    # "context_forcing_split_seed = 42"
    split_rng = random.Random(42)
    pair_ids = sorted([p["pair_id"] for p in all_pairs])
    split_rng.shuffle(pair_ids)
    
    train_count = int(len(pair_ids) * 0.8)
    train_pairs = set(pair_ids[:train_count])
    val_pairs = set(pair_ids[train_count:])
    
    # Save train/val pair id JSONs
    with open(manifest_path.parent / ("train_pair_ids.json" if not args.tiny else "tiny_train_pair_ids.json"), "w") as f:
        json.dump(sorted(list(train_pairs)), f, indent=2)
    with open(manifest_path.parent / ("val_pair_ids.json" if not args.tiny else "tiny_val_pair_ids.json"), "w") as f:
        json.dump(sorted(list(val_pairs)), f, indent=2)

    # Assign split label and flatten
    final_records = []
    for pair in all_pairs:
        pid = pair["pair_id"]
        split_name = "context_forcing_v1_train" if pid in train_pairs else "context_forcing_v1_val"
        for rec in pair["records"]:
            rec["split"] = split_name
            final_records.append(rec)

    with open(manifest_path, "w") as f:
        for r in final_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated {len(final_records)} records ({len(all_pairs)} physical pairs) to {manifest_path}")
