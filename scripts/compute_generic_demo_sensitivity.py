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

    seed_deltas = []
    for s in seeds:
        raw_path = Path(f"learning_outputs/{m}_seed{s}/context_challenge_results_raw.json")
        if not raw_path.exists():
            raise RuntimeError(f"Missing file: {raw_path}")

        with open(raw_path) as f:
            raw = json.load(f)

        if "generic_text" not in raw:
            raise RuntimeError(f"Missing generic_text in {raw_path}")

        g = raw["generic_text"]
        if not g.get("_raw_pair_ids"):
            raise RuntimeError(f"Missing or empty _raw_pair_ids in {raw_path}")

        if "_raw_task_ids" not in g or "_raw_preds" not in g or "_raw_logits" not in g:
            raise RuntimeError(f"Missing required raw arrays in {raw_path}")

        expected_len = len(g["_raw_pair_ids"])
        if len(g["_raw_task_ids"]) != expected_len or len(g["_raw_preds"]) != expected_len or len(g["_raw_logits"]) != expected_len:
            raise RuntimeError(f"Raw array lengths do not match in {raw_path}")

        pairs = {}
        for i, pid in enumerate(g["_raw_pair_ids"]):
            tid = g["_raw_task_ids"][i]
            if pid not in pairs:
                pairs[pid] = {}

            entry = {
                "prob": g["_raw_preds"][i],
                "logit": g["_raw_logits"][i],
            }
            if "_raw_scores" in g and len(g["_raw_scores"]) > i:
                entry["compatibility"] = g["_raw_scores"][i]
            if "_raw_latents" in g and len(g["_raw_latents"]) > i:
                entry["latent"] = g["_raw_latents"][i]

            pairs[pid][tid] = entry

        if len(pairs) != 40:
            raise RuntimeError(f"Expected 40 physical pairs for {m} seed {s}, found {len(pairs)}")

        # Compare task_1 vs task_2
        deltas = {
            "abs_delta_prob": [],
            "abs_delta_logit": [],
            "flip_rate": [],
            "abs_delta_compatibility": [],
            "latent_cosine_sim": [],
            "latent_cosine_change": []
        }

        for pid, tasks in pairs.items():
            if "task_1" not in tasks or "task_2" not in tasks:
                raise RuntimeError(f"Missing task_1 or task_2 for pair {pid} in {m} seed {s}")

            t1 = tasks["task_1"]
            t2 = tasks["task_2"]

            deltas["abs_delta_prob"].append(abs(t1["prob"] - t2["prob"]))
            deltas["abs_delta_logit"].append(abs(t1["logit"] - t2["logit"]))

            c1 = 1 if t1["prob"] >= 0.5 else 0
            c2 = 1 if t2["prob"] >= 0.5 else 0
            deltas["flip_rate"].append(1.0 if c1 != c2 else 0.0)

            if "compatibility" in t1 and "compatibility" in t2:
                deltas["abs_delta_compatibility"].append(abs(t1["compatibility"] - t2["compatibility"]))

            if "latent" in t1 and "latent" in t2 and t1["latent"] is not None and t2["latent"] is not None:
                sim = cosine_similarity(t1["latent"], t2["latent"])
                deltas["latent_cosine_sim"].append(sim)
                deltas["latent_cosine_change"].append(1.0 - sim)

        if not deltas["abs_delta_prob"]:
            raise RuntimeError(f"No valid delta probabilities computed for {m} seed {s}")

        seed_res = {
            "seed": s,
            "abs_delta_prob": float(np.mean(deltas["abs_delta_prob"])),
            "abs_delta_logit": float(np.mean(deltas["abs_delta_logit"])),
            "flip_rate": float(np.mean(deltas["flip_rate"]))
        }
        if deltas["abs_delta_compatibility"]:
            seed_res["abs_delta_compatibility"] = float(np.mean(deltas["abs_delta_compatibility"]))
        if deltas["latent_cosine_sim"]:
            seed_res["latent_cosine_sim"] = float(np.mean(deltas["latent_cosine_sim"]))
            seed_res["latent_cosine_change"] = float(np.mean(deltas["latent_cosine_change"]))

        seed_deltas.append(seed_res)

    if len(seed_deltas) != len(seeds):
        raise RuntimeError(f"Expected {len(seeds)} seed results for {m}, but got {len(seed_deltas)}")

    results[m] = {
        "per_seed": seed_deltas,
        "mean_abs_delta_prob": float(np.mean([x["abs_delta_prob"] for x in seed_deltas])),
        "mean_abs_delta_logit": float(np.mean([x["abs_delta_logit"] for x in seed_deltas])),
        "mean_flip_rate": float(np.mean([x["flip_rate"] for x in seed_deltas]))
    }
    if "abs_delta_compatibility" in seed_deltas[0]:
        results[m]["mean_abs_delta_compatibility"] = float(np.mean([x["abs_delta_compatibility"] for x in seed_deltas]))
    if "latent_cosine_sim" in seed_deltas[0]:
        results[m]["mean_latent_cosine_sim"] = float(np.mean([x["latent_cosine_sim"] for x in seed_deltas]))
        results[m]["mean_latent_cosine_change"] = float(np.mean([x["latent_cosine_change"] for x in seed_deltas]))

with open(out_dir / "generic_demo_pair_sensitivity.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved generic demo sensitivity to", out_dir / "generic_demo_pair_sensitivity.json")
