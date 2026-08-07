import json
import os
import glob
from pathlib import Path
import argparse

def get_demo_id(demo_path):
    p = Path(demo_path)
    if p.name == "rgb.mp4":
        return p.parent.name
    return p.stem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/pilot_manifest.jsonl")
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    output_dir = Path("learning_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "index.jsonl"
    
    # Discover all successful pilot demos
    # They might be .mp4 files or nested rgb.mp4 files
    pilot_demos = glob.glob("data/pilot_demos/**/*.mp4", recursive=True)
    
    # If not found under pilot_demos, just find pilot in name
    if not pilot_demos:
        pilot_demos = glob.glob("data/pilot_demos/*.mp4")
    
    task_demos = {"task_1": [], "task_2": []}
    for dp in sorted(pilot_demos):
        if "task1" in dp or "task_1" in dp:
            task_demos["task_1"].append(dp)
        elif "task2" in dp or "task_2" in dp:
            task_demos["task_2"].append(dp)
            
    if not task_demos["task_1"]:
        raise RuntimeError("No pilot demos found for task_1")
    if not task_demos["task_2"]:
        raise RuntimeError("No pilot demos found for task_2")

    total_queries = 0
    total_pairs = 0

    with open(output_file, 'w') as out_f:
        source_dataset = os.path.basename(manifest_path).replace("_manifest.jsonl", "")
        with open(manifest_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                pair_data = json.loads(line)
                
                try:
                
                    # Validation: exactly stop + proceed
                    if "stop" not in pair_data or "proceed" not in pair_data:
                        raise ValueError(f"Pair does not have both stop and proceed mates: {pair_data.keys()}")
                    
                    pair_id = pair_data["stop"].get("spec", {}).get("pair_id")
                    if not pair_id:
                        raise ValueError("Missing pair_id in stop mate")
                    
                    # Check match
                    proceed_pair_id = pair_data["proceed"].get("spec", {}).get("pair_id")
                    if proceed_pair_id != pair_id:
                        raise ValueError(f"Mismatched pair IDs: {pair_id} vs {proceed_pair_id}")
                        
                    task_id = pair_data["stop"].get("spec", {}).get("task_family")
                    if not task_id:
                        task_id = pair_data["stop"].get("resolved_scene_spec", {}).get("task_family")
                    
                    if task_id not in task_demos:
                        raise ValueError(f"Unknown task_id: {task_id}")
                        
                    # Deterministic demo assignment based on pair_id
                    demo_pool = task_demos[task_id]
                    demo_idx = hash(pair_id) % len(demo_pool)
                    demo_path = demo_pool[demo_idx]
                    demo_id = get_demo_id(demo_path)
                    
                    if not os.path.exists(demo_path):
                        raise FileNotFoundError(f"Missing demo: {demo_path}")
                    
                    # Process both mates
                    seen_sample_ids = set()
                    for mate_type in ["stop", "proceed"]:
                        mate_data = pair_data[mate_type]
                        
                        label = mate_data.get("label", "").upper()
                        spec = mate_data.get("spec", {})
                        resolved_spec = mate_data.get("resolved_scene_spec", {})
                        
                        instruction = resolved_spec.get("instruction", spec.get("goal_instruction"))
                        split = spec.get("split", resolved_spec.get("split"))
                        if not split:
                            raise ValueError(f"Missing split for pair {pair_id}")
                        
                        query_rgb_path = mate_data.get("rgb_path")
                        if not query_rgb_path or not os.path.exists(query_rgb_path):
                            raise FileNotFoundError(f"Missing query RGB for {pair_id} {mate_type}: {query_rgb_path}")
                        
                        causal_mask_path = mate_data.get("causal_violation_mask_path")
                        if label == "STOP":
                            if not causal_mask_path or not os.path.exists(causal_mask_path):
                                raise FileNotFoundError(f"Missing mask for STOP {pair_id}: {causal_mask_path}")
                        
                        sample_id = spec.get("sample_id", f"{pair_id}_{mate_type}")
                        if sample_id in seen_sample_ids:
                            raise ValueError(f"Duplicate sample ID in pair: {sample_id}")
                        seen_sample_ids.add(sample_id)
                        
                        record = {
                            "sample_id": sample_id,
                            "pair_id": pair_id,
                            "task_id": task_id,
                            "instruction": instruction,
                            "demonstration_id": demo_id,
                            "demonstration_video_path": demo_path,
                            "query_rgb_path": query_rgb_path,
                            "label": label,
                            "causal_mask_path": causal_mask_path,
                            "split": split,
                            "asset_metadata": {},
                            "source_dataset": source_dataset,
                            "mate_id": f"{pair_id}_proceed" if mate_type == "stop" else f"{pair_id}_stop"
                        }
                        
                        out_f.write(json.dumps(record) + "\n")
                        total_queries += 1
                        
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping broken record {pair_data.get('stop', {}).get('spec', {}).get('pair_id', 'unknown')}: {e}")
                    continue
                    
                total_pairs += 1

    print(f"Built learning index with {total_pairs} pairs ({total_queries} queries).")
    print(f"Saved to {output_file}")
    
if __name__ == "__main__":
    main()
