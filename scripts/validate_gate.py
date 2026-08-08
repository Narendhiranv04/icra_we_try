import json
import numpy as np

with open("learning_outputs/query_only_gate_seed42/context_challenge_results_raw.json", "r") as f:
    qo_raw = json.load(f)["standard"]

qo_preds = qo_raw["_raw_preds"]
qo_logits = qo_raw["_raw_logits"]
qo_pairs = qo_raw["_raw_pair_ids"]

qo_pair_dict = {}
for i, pid in enumerate(qo_pairs):
    if pid not in qo_pair_dict:
        qo_pair_dict[pid] = []
    qo_pair_dict[pid].append(qo_logits[i])

for pid, logits in qo_pair_dict.items():
    if len(logits) == 2:
        assert np.abs(logits[0] - logits[1]) < 1e-4, f"Query-only logit mismatch! {logits[0]} != {logits[1]}"

import os
import torch
# Check feature identicality for same-RGB pairs
manifest_path = "data/manifests/context_challenge_manifest_gate.jsonl"
if os.path.exists(manifest_path):
    scenes = {}
    with open(manifest_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["pair_id"]
            if pid not in scenes: scenes[pid] = []
            scenes[pid].append(rec)

    for pid, pairs in scenes.items():
        if len(pairs) == 2:
            t1, t2 = pairs[0], pairs[1]
            assert t1["query_rgb_sha256"] == t2["query_rgb_sha256"], "RGB SHA must be identical"
            feat1 = torch.load(f"data/features_context/query_{t1['sample_id']}_features.pt", weights_only=True)
            feat2 = torch.load(f"data/features_context/query_{t2['sample_id']}_features.pt", weights_only=True)
            assert torch.allclose(feat1["global"], feat2["global"]), "query_global must be identical"
            assert torch.allclose(feat1["patch"], feat2["patch"]), "query_patch must be identical"

print("Query-only gate checks passed!")

with open("learning_outputs/relational_heatmap_gate_seed42/context_challenge_results_raw.json", "r") as f:
    rel_raw = json.load(f)["standard"]

rel_logits = rel_raw["_raw_logits"]
for l in rel_logits:
    assert np.isfinite(l), "Relational output is not finite!"
print("Relational v1 gate checks passed!")

