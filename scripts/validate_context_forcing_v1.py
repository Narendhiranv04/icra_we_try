import json
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

def validate(manifest_path, freeze_path, out_path):
    print(f"Validating manifest: {manifest_path}")
    
    with open(manifest_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    with open(freeze_path, "r") as f:
        freeze_data = json.load(f)
        
    blocklist = set(freeze_data.get("rgb_sha256_hashes", []))
    
    pairs = {}
    states = {"A": 0, "B": 0, "C": 0, "D": 0}
    splits = {"context_forcing_v1_train": set(), "context_forcing_v1_val": set()}
    
    for r in records:
        pid = r["pair_id"]
        if pid not in pairs:
            pairs[pid] = []
        pairs[pid].append(r)
        
    errors = []
    
    if len(pairs) != 160:
        errors.append(f"Expected 160 pairs, found {len(pairs)}")
        
    if len(records) != 320:
        errors.append(f"Expected 320 records, found {len(records)}")

    # Check pair constraints
    for pid, pair_recs in pairs.items():
        if len(pair_recs) != 2:
            errors.append(f"Pair {pid} has {len(pair_recs)} records instead of 2")
            continue
            
        t1 = next((r for r in pair_recs if r["task_id"] == "task_1"), None)
        t2 = next((r for r in pair_recs if r["task_id"] == "task_2"), None)
        
        if not t1 or not t2:
            errors.append(f"Pair {pid} is missing task_1 or task_2")
            continue
            
        if t1["query_rgb_sha256"] != t2["query_rgb_sha256"]:
            errors.append(f"Pair {pid} has mismatched RGB hashes!")
            
        if t1["query_rgb_sha256"] in blocklist:
            errors.append(f"Pair {pid} hash {t1['query_rgb_sha256']} collides with Benchmark B!")
            
        if t1["state"] != t2["state"]:
            errors.append(f"Pair {pid} has mismatched states!")
            
        if t1["split"] != t2["split"]:
            errors.append(f"Pair {pid} has mismatched splits!")
            
        if t1["split"] not in splits:
            errors.append(f"Pair {pid} has invalid split {t1['split']}")
        else:
            splits[t1["split"]].add(pid)
            
        states[t1["state"]] += 1
        
        # Check files and masks
        for r in pair_recs:
            if not os.path.exists(r["query_rgb_path"]):
                errors.append(f"Missing RGB for {r['sample_id']}: {r['query_rgb_path']}")
            if not os.path.exists(r["demonstration_video_path"]):
                errors.append(f"Missing Demo for {r['sample_id']}: {r['demonstration_video_path']}")
            
            mask_path = r["causal_mask_path"]
            if not os.path.exists(mask_path):
                errors.append(f"Missing Mask for {r['sample_id']}: {mask_path}")
            else:
                mask_arr = np.array(Image.open(mask_path))
                mask_sum = mask_arr.sum()
                if r["label"] == "STOP" and mask_sum == 0:
                    errors.append(f"STOP sample {r['sample_id']} has empty mask")
                elif r["label"] == "PROCEED" and mask_sum > 0:
                    errors.append(f"PROCEED sample {r['sample_id']} has non-empty mask (sum={mask_sum})")

    for state, count in states.items():
        if count != 40:
            errors.append(f"Expected 40 pairs for state {state}, found {count}")
            
    train_count = len(splits["context_forcing_v1_train"])
    val_count = len(splits["context_forcing_v1_val"])
    
    if train_count != 128:
        errors.append(f"Expected 128 train pairs, found {train_count}")
    if val_count != 32:
        errors.append(f"Expected 32 val pairs, found {val_count}")

    passed = len(errors) == 0

    results = {
        "passed": passed,
        "total_records": len(records),
        "total_pairs": len(pairs),
        "train_pairs": train_count,
        "val_pairs": val_count,
        "state_distribution": states,
        "errors": errors
    }
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    if passed:
        print("Validation PASSED!")
    else:
        print("Validation FAILED!")
        for e in errors[:10]:
            print(f" - {e}")
        if len(errors) > 10:
            print(f" ... and {len(errors) - 10} more errors.")
            
    return passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/context_forcing_v1_manifest.jsonl")
    parser.add_argument("--freeze", default="artifacts/context_forcing_v1/benchmark_b_freeze.json")
    parser.add_argument("--out", default="artifacts/context_forcing_v1/dataset_validation.json")
    args = parser.parse_args()
    
    passed = validate(args.manifest, args.freeze, args.out)
    if not passed:
        exit(1)
