import json
import os
import glob
import hashlib
from pathlib import Path
import argparse
from collections import Counter

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
    
    pilot_demos = glob.glob("data/pilot_demos/**/*.mp4", recursive=True)
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
    controls_ignored = 0
    task_counts = Counter()
    split_counts = Counter()

    with open(output_file, 'w') as out_f:
        source_dataset = os.path.basename(manifest_path).replace("_manifest.jsonl", "")
        with open(manifest_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                pair_data = json.loads(line)
                
                if pair_data.get("sample_type") == "positive_control":
                    controls_ignored += 1
                    continue
                
                if "stop" not in pair_data or "proceed" not in pair_data:
                    raise ValueError(f"Pair does not have both stop and proceed mates: {pair_data.keys()}")
                
                pair_id = pair_data["stop"].get("spec", {}).get("pair_id")
                if not pair_id:
                    raise ValueError("Missing pair_id in stop mate")
                
                proceed_pair_id = pair_data["proceed"].get("spec", {}).get("pair_id")
                if proceed_pair_id != pair_id:
                    raise ValueError(f"Mismatched pair IDs: {pair_id} vs {proceed_pair_id}")
                    
                task_id = pair_data["stop"].get("spec", {}).get("task_family")
                if not task_id:
                    task_id = pair_data["stop"].get("resolved_scene_spec", {}).get("task_family")
                
                if task_id not in task_demos:
                    raise ValueError(f"Unknown task_id: {task_id}")
                    
                # Stable deterministic demo assignment based on pair_id
                demo_pool = task_demos[task_id]
                demo_idx = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % len(demo_pool)
                demo_path = demo_pool[demo_idx]
                demo_id = get_demo_id(demo_path)
                
                if not os.path.exists(demo_path):
                    raise FileNotFoundError(f"Missing demo: {demo_path}")
                
                seen_sample_ids = set()
                for mate_type in ["stop", "proceed"]:
                    mate_data = pair_data[mate_type]
                    
                    label = mate_data.get("label", "").upper()
                    if label not in ["STOP", "PROCEED"]:
                        raise ValueError(f"Invalid label: {label}")
                        
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
                    
                total_pairs += 1
                task_counts[task_id] += 1
                split_counts[split] += 1

    print(f"Built learning index with {total_pairs} pairs ({total_queries} queries).")
    print(f"Controls ignored: {controls_ignored}")
    print(f"Demos discovered: {len(pilot_demos)}")
    print(f"Counts by task: {dict(task_counts)}")
    print(f"Counts by split: {dict(split_counts)}")
    print(f"Saved to {output_file}")
    
if __name__ == "__main__":
    main()
