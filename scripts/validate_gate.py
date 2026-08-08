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
print("Query-only gate checks passed!")

with open("learning_outputs/relational_heatmap_gate_seed42/context_challenge_results_raw.json", "r") as f:
    rel_raw = json.load(f)["standard"]

rel_logits = rel_raw["_raw_logits"]
for l in rel_logits:
    assert np.isfinite(l), "Relational output is not finite!"
print("Relational v1 gate checks passed!")

