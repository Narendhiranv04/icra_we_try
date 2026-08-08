import json
import numpy as np
import os
import torch
import hashlib
import sys

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest_path = "data/manifests/context_challenge_manifest_gate.jsonl"
if not os.path.exists(manifest_path):
    print(f"Error: {manifest_path} not found.")
    sys.exit(1)

scenes = {}
with open(manifest_path, "r") as f:
    for line in f:
        if not line.strip(): continue
        rec = json.loads(line)
        pid = rec["pair_id"]
        if pid not in scenes: scenes[pid] = []
        scenes[pid].append(rec)

# Validate EXACTLY 4 physical scenes, 8 records
if len(scenes) != 4:
    print(f"Error: Expected 4 physical scenes, found {len(scenes)}")
    sys.exit(1)

states_found = set()
for pid, pairs in scenes.items():
    if len(pairs) != 2:
        print(f"Error: Scene {pid} has {len(pairs)} records, expected 2.")
        sys.exit(1)
        
    t1, t2 = pairs[0], pairs[1]
    
    tasks = {t1["task_id"], t2["task_id"]}
    if tasks != {"task_1", "task_2"}:
        print(f"Error: Scene {pid} tasks are {tasks}, expected {{'task_1', 'task_2'}}")
        sys.exit(1)
        
    if t1["state"] != t2["state"]:
        print(f"Error: Mismatched states in scene {pid}")
        sys.exit(1)
        
    states_found.add(t1["state"])
    
    # Check RGB paths and hashes
    if t1["query_rgb_path"] != t2["query_rgb_path"]:
        print(f"Error: Mismatched RGB paths in scene {pid}")
        sys.exit(1)
        
    rgb_path = t1["query_rgb_path"]
    if not os.path.exists(rgb_path):
        print(f"Error: RGB path {rgb_path} does not exist.")
        sys.exit(1)
        
    actual_sha = compute_sha256(rgb_path)
    if t1["query_rgb_sha256"] != actual_sha or t2["query_rgb_sha256"] != actual_sha:
        print(f"Error: Manifest SHA mismatch for {rgb_path}")
        sys.exit(1)

    # Check query features
    feat1_path = f"data/features_context/query_{t1['sample_id']}_features.pt"
    feat2_path = f"data/features_context/query_{t2['sample_id']}_features.pt"
    if not os.path.exists(feat1_path) or not os.path.exists(feat2_path):
        print(f"Error: Missing query features for {pid}")
        sys.exit(1)
        
    feat1 = torch.load(feat1_path, map_location="cpu", weights_only=True)
    feat2 = torch.load(feat2_path, map_location="cpu", weights_only=True)
    
    if not torch.allclose(feat1["global"], feat2["global"], atol=1e-5):
        print(f"Error: query_global mismatch for {pid}")
        sys.exit(1)
    if not torch.allclose(feat1["patch"], feat2["patch"], atol=1e-5):
        print(f"Error: query_patch mismatch for {pid}")
        sys.exit(1)
        
    # Check demo features are genuinely different
    dfeat1_path = f"data/features_context/demo_{t1['demonstration_id']}_features.pt"
    dfeat2_path = f"data/features_context/demo_{t2['demonstration_id']}_features.pt"
    if not os.path.exists(dfeat1_path) or not os.path.exists(dfeat2_path):
        print(f"Error: Missing demo features for {pid}")
        sys.exit(1)
        
    dfeat1 = torch.load(dfeat1_path, map_location="cpu", weights_only=True)
    dfeat2 = torch.load(dfeat2_path, map_location="cpu", weights_only=True)
    
    if torch.allclose(dfeat1["global"], dfeat2["global"], atol=1e-5):
        print(f"Error: demo_global unexpectedly identical for {pid}")
        sys.exit(1)

if states_found != {"A", "B", "C", "D"}:
    print(f"Error: Expected states {{'A', 'B', 'C', 'D'}}, found {states_found}")
    sys.exit(1)
    
print("Dataset invariants verified.")

print("Running Query-only gate checks...")
qo_raw_path = "learning_outputs/query_only_gate_seed42/context_challenge_results_raw.json"
if not os.path.exists(qo_raw_path):
    print(f"Error: {qo_raw_path} not found.")
    sys.exit(1)

with open(qo_raw_path, "r") as f:
    qo_raw = json.load(f)["standard"]

qo_logits = qo_raw["_raw_logits"]
qo_pairs = qo_raw["_raw_pair_ids"]

qo_pair_dict = {}
for i, pid in enumerate(qo_pairs):
    if pid not in qo_pair_dict:
        qo_pair_dict[pid] = []
    qo_pair_dict[pid].append(qo_logits[i])

for pid, logits in qo_pair_dict.items():
    if len(logits) == 2:
        if np.abs(logits[0] - logits[1]) >= 1e-4:
            print(f"Error: Query-only logit mismatch! {logits[0]} != {logits[1]}")
            sys.exit(1)
print("Query-only gate checks passed!")

print("Running Relational v1 gate checks...")
rel_raw_path = "learning_outputs/relational_heatmap_gate_seed42/context_challenge_results_raw.json"
if not os.path.exists(rel_raw_path):
    print(f"Error: {rel_raw_path} not found.")
    sys.exit(1)

with open(rel_raw_path, "r") as f:
    rel_raw = json.load(f)["standard"]

rel_logits = rel_raw["_raw_logits"]
rel_preds = rel_raw["_raw_preds"]

for l in rel_logits:
    if not np.isfinite(l):
        print("Error: Relational logit is not finite!")
        sys.exit(1)
        
for p in rel_preds:
    if not np.isfinite(p):
        print("Error: Relational probability is not finite!")
        sys.exit(1)

if "_raw_latents" not in rel_raw or len(rel_raw["_raw_latents"]) == 0:
    print("Error: Relational model missing latents in output!")
    sys.exit(1)
    
for l in rel_raw["_raw_latents"]:
    if not np.all(np.isfinite(l)):
        print("Error: Relational latent is not finite!")
        sys.exit(1)
        
if "_raw_scores" not in rel_raw or len(rel_raw["_raw_scores"]) == 0:
    print("Error: Relational model missing compatibility scores in output!")
    sys.exit(1)
    
for s in rel_raw["_raw_scores"]:
    if not np.isfinite(s):
        print("Error: Relational compatibility score is not finite!")
        sys.exit(1)

print("Relational v1 gate checks passed!")

