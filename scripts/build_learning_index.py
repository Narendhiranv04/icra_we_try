import json
import os
import glob
from pathlib import Path

def main():
    manifests_dir = Path("data/manifests")
    output_dir = Path("learning_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "index.jsonl"
    
    # Map tasks to a default demonstration video.
    # We will pick the first available pilot demo or smoke demo for each task.
    task_to_demo = {}
    pilot_demos = glob.glob("data/pilot_demos/*.mp4")
    for demo_path in sorted(pilot_demos):
        if "task1" in demo_path and "task_1" not in task_to_demo:
            task_to_demo["task_1"] = demo_path
        elif "task2" in demo_path and "task_2" not in task_to_demo:
            task_to_demo["task_2"] = demo_path
            
    # Fallback to smoke demos if pilot demos not found
    if "task_1" not in task_to_demo:
        task_to_demo["task_1"] = "data/demos/open_box/demo_task1_smoke/rgb.mp4"
    if "task_2" not in task_to_demo:
        task_to_demo["task_2"] = "data/demos/place_object/demo_task2_smoke/rgb.mp4"

    manifest_files = glob.glob(str(manifests_dir / "*.jsonl"))
    
    total_queries = 0
    total_pairs = 0

    with open(output_file, 'w') as out_f:
        for manifest_path in manifest_files:
            source_dataset = os.path.basename(manifest_path).replace("_manifest.jsonl", "")
            with open(manifest_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    pair_data = json.loads(line)
                    
                    pair_id = None
                    if "stop" in pair_data:
                        pair_id = pair_data["stop"].get("spec", {}).get("pair_id")
                    if not pair_id and "proceed" in pair_data:
                        pair_id = pair_data["proceed"].get("spec", {}).get("pair_id")
                        
                    if not pair_id:
                        continue
                        
                    total_pairs += 1
                    
                    # Process both mates
                    for mate_type, mate_data in pair_data.items():
                        if mate_type not in ["stop", "proceed"]:
                            continue
                            
                        label = mate_data.get("label", "").upper()
                        spec = mate_data.get("spec", {})
                        resolved_spec = mate_data.get("resolved_scene_spec", {})
                        
                        task_id = spec.get("task_family", resolved_spec.get("task_family"))
                        if not task_id:
                            continue
                            
                        instruction = resolved_spec.get("instruction", spec.get("goal_instruction"))
                        split = spec.get("split", resolved_spec.get("split"))
                        
                        query_rgb_path = mate_data.get("rgb_path")
                        causal_mask_path = mate_data.get("causal_violation_mask_path")
                        
                        # Proceed samples may not have a causal mask path in some older formats.
                        # But as per prompt, PROCEED has an all-zero mask, which we handle in the dataset.
                        # But if it is given in the manifest, we use it.
                        
                        sample_id = spec.get("sample_id", f"{pair_id}_{mate_type}")
                        demo_path = task_to_demo.get(task_id)
                        
                        record = {
                            "sample_id": sample_id,
                            "pair_id": pair_id,
                            "task_id": task_id,
                            "instruction": instruction,
                            "demonstration_id": os.path.basename(demo_path) if demo_path else None,
                            "demonstration_video_path": demo_path,
                            "query_rgb_path": query_rgb_path,
                            "label": label,
                            "causal_mask_path": causal_mask_path,
                            "split": split,
                            "asset_metadata": {}, # Could be populated with asset IDs
                            "source_dataset": source_dataset,
                            "mate_id": f"{pair_id}_proceed" if mate_type == "stop" else f"{pair_id}_stop"
                        }
                        
                        out_f.write(json.dumps(record) + "\n")
                        total_queries += 1

    print(f"Built learning index with {total_pairs} pairs ({total_queries} queries).")
    print(f"Saved to {output_file}")
    
if __name__ == "__main__":
    main()
