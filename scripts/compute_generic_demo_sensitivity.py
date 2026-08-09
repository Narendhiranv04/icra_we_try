import json
import os
import numpy as np
from pathlib import Path

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

models = [
    "query_only", "language_query", "demo_query", 
    "full_no_ranking", "pooled_multimodal", "relational_heatmap"
]
seeds = [11, 23, 42, 67, 101]

out_dir = Path("artifacts/learning_stage1/benchmark_b")
out_dir.mkdir(parents=True, exist_ok=True)

results = {}

for m in models:
    results[m] = {}
    
    # query_only does not receive demo, but generic_text ablation might have been run for all? 
    # Actually, evaluate_context_challenge.py runs it.
    
    seed_deltas = []
    for s in seeds:
        raw_path = Path(f"learning_outputs/{m}_seed{s}/context_challenge_results_raw.json")
        if not raw_path.exists():
            continue
            
        with open(raw_path) as f:
            raw = json.load(f)
            
        if "generic_text" not in raw:
            continue
            
        g = raw["generic_text"]
        if not g.get("_raw_pair_ids"):
            continue
            
        pairs = {}
        for i, pid in enumerate(g["_raw_pair_ids"]):
            tid = g["_raw_task_ids"][i]
            if pid not in pairs:
                pairs[pid] = {}
            pairs[pid][tid] = {
                "score": g.get("_raw_scores", g["_raw_preds"])[i],
                "logit": g["_raw_logits"][i],
                "pred": g["_raw_preds"][i],
                "latent": g.get("_raw_latents", [None]*len(g["_raw_preds"]))[i]
            }
            
        # Compare task_1 vs task_2
        deltas = {
            "abs_delta_prob": [],
            "abs_delta_logit": [],
            "flip_rate": [],
            "latent_cosine_sim": [],
            "latent_cosine_change": []
        }
        
        for pid, tasks in pairs.items():
            if "task_1" in tasks and "task_2" in tasks:
                t1 = tasks["task_1"]
                t2 = tasks["task_2"]
                
                deltas["abs_delta_prob"].append(abs(t1["score"] - t2["score"]))
                deltas["abs_delta_logit"].append(abs(t1["logit"] - t2["logit"]))
                c1 = 1 if t1["score"] > 0.5 else 0
                c2 = 1 if t2["score"] > 0.5 else 0
                deltas["flip_rate"].append(1.0 if c1 != c2 else 0.0)
                
                if t1["latent"] is not None and t2["latent"] is not None:
                    sim = cosine_similarity(t1["latent"], t2["latent"])
                    deltas["latent_cosine_sim"].append(sim)
                    deltas["latent_cosine_change"].append(1.0 - sim)
                    
        if not deltas["abs_delta_prob"]:
            continue
            
        seed_res = {
            "seed": s,
            "abs_delta_prob": float(np.mean(deltas["abs_delta_prob"])),
            "abs_delta_logit": float(np.mean(deltas["abs_delta_logit"])),
            "flip_rate": float(np.mean(deltas["flip_rate"]))
        }
        if deltas["latent_cosine_sim"]:
            seed_res["latent_cosine_sim"] = float(np.mean(deltas["latent_cosine_sim"]))
            seed_res["latent_cosine_change"] = float(np.mean(deltas["latent_cosine_change"]))
            
        seed_deltas.append(seed_res)
        
    if seed_deltas:
        results[m] = {
            "per_seed": seed_deltas,
            "mean_abs_delta_prob": float(np.mean([x["abs_delta_prob"] for x in seed_deltas])),
            "mean_abs_delta_logit": float(np.mean([x["abs_delta_logit"] for x in seed_deltas])),
            "mean_flip_rate": float(np.mean([x["flip_rate"] for x in seed_deltas]))
        }
        if "latent_cosine_sim" in seed_deltas[0]:
            results[m]["mean_latent_cosine_sim"] = float(np.mean([x["latent_cosine_sim"] for x in seed_deltas]))
            results[m]["mean_latent_cosine_change"] = float(np.mean([x["latent_cosine_change"] for x in seed_deltas]))

with open(out_dir / "generic_demo_pair_sensitivity.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved generic demo sensitivity to", out_dir / "generic_demo_pair_sensitivity.json")
