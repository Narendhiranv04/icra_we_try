import json
import os
import hashlib
from collections import defaultdict
import cv2
import numpy as np

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    manifest_path = "data/manifests/context_challenge_manifest.jsonl"
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        return 1

    records = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 80:
        print(f"Expected 80 records, got {len(records)}")
        return 1

    scenes = defaultdict(list)
    for r in records:
        scenes[r["pair_id"]].append(r)

    if len(scenes) != 40:
        print(f"Expected 40 scenes, got {len(scenes)}")
        return 1

    state_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for pid, pairs in scenes.items():
        if len(pairs) != 2:
            print(f"Scene {pid} has {len(pairs)} records")
            return 1

        t1, t2 = pairs[0], pairs[1]
        state = t1["state"]
        if t2["state"] != state:
            print(f"Mismatched state in scene {pid}")
            return 1
            
        state_counts[state] += 1

        # Check RGB
        if t1["query_rgb_path"] != t2["query_rgb_path"]:
            print(f"Mismatched RGB in scene {pid}")
            return 1
        rgb_path = t1["query_rgb_path"]
        if not os.path.exists(rgb_path):
            print(f"Missing RGB {rgb_path}")
            return 1
        actual_sha = compute_sha256(rgb_path)
        if t1["query_rgb_sha256"] != actual_sha or t2["query_rgb_sha256"] != actual_sha:
            print(f"SHA mismatch for {rgb_path}")
            return 1
            
        # Check occupancy
        occ_reqs = {
            "A": (True, False),
            "B": (False, True),
            "C": (True, True),
            "D": (False, False)
        }
        req_lid, req_targ = occ_reqs[state]
        for r in pairs:
            if r["measured_lid_occupancy"] != req_lid or r["measured_target_occupancy"] != req_targ:
                print(f"Occupancy mismatch in {pid} state {state}")
                return 1

        # Check masks
        for r in pairs:
            mask_path = r["causal_mask_path"]
            if not os.path.exists(mask_path):
                print(f"Missing mask {mask_path}")
                return 1
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_sum = np.sum(mask > 0)
            
            if r["label"] == "STOP":
                if mask_sum == 0:
                    print(f"Zero mask for STOP record {r['sample_id']}")
                    return 1
            elif r["label"] == "PROCEED":
                if mask_sum != 0:
                    print(f"Non-zero mask for PROCEED record {r['sample_id']}")
                    return 1
                    
        # Check demos
        for r in pairs:
            demo_path = r["demonstration_video_path"]
            if not os.path.exists(demo_path):
                print(f"Missing demo {demo_path}")
                return 1
                
    t1_demos = set()
    t2_demos = set()
    for state, count in state_counts.items():
        if count != 10:
            print(f"Expected 10 scenes for state {state}, got {count}")
            return 1
            
    for r in records:
        if r["task_id"] == "task_1":
            t1_demos.add(r["demonstration_id"])
        elif r["task_id"] == "task_2":
            t2_demos.add(r["demonstration_id"])

    if len(t1_demos) != 3:
        print(f"Expected exactly 3 unique Task 1 demos, found {len(t1_demos)}: {t1_demos}")
        return 1
        
    if len(t2_demos) != 3:
        print(f"Expected exactly 3 unique Task 2 demos, found {len(t2_demos)}: {t2_demos}")
        return 1

    print("Benchmark-B dataset validated successfully.")
    return 0

if __name__ == "__main__":
    exit(main())
