import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def bootstrap_confidence_interval(metrics_list, n_iterations=2000, ci=0.95):
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0].keys() if isinstance(metrics_list[0][k], (int, float))]
    bootstrapped_means = {k: [] for k in keys}
    n = len(metrics_list)
    
    for _ in tqdm(range(n_iterations), desc="Bootstrapping", leave=False):
        indices = np.random.randint(0, n, n)
        for k in keys:
            values = [metrics_list[i][k] for i in indices if k in metrics_list[i]]
            if values:
                sample_mean = np.mean(values)
                bootstrapped_means[k].append(sample_mean)
            
    results = {}
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    
    for k in keys:
        arr = np.array(bootstrapped_means[k])
        results[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "lower": float(np.percentile(arr, lower_pct)),
            "upper": float(np.percentile(arr, upper_pct))
        }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--n-iterations", type=int, default=2000)
    args = parser.parse_args()
    
    out_dir = Path(args.experiment_dir)
    results = {}
    
    # 1. Benchmark A (Pair-level)
    for split in ["id_val", "compositional", "unseen_object", "unseen_background"]:
        raw_path = out_dir / f"metrics_raw_{split}.json"
        if not raw_path.exists():
            continue
            
        with open(raw_path, "r") as f:
            raw = json.load(f)
            
        preds = raw.get("_raw_preds", [])
        targets = raw.get("_raw_targets", [])
        pair_ids = raw.get("_raw_pair_ids", [])
        scores = raw.get("_raw_scores", [])
        
        if not pair_ids:
            continue
            
        # Group by pair
        pairs = {}
        for i in range(len(preds)):
            pid = pair_ids[i]
            if pid not in pairs:
                pairs[pid] = {"preds": [], "targets": [], "scores": []}
            pairs[pid]["preds"].append(preds[i])
            pairs[pid]["targets"].append(targets[i])
            if scores:
                pairs[pid]["scores"].append(scores[i])
                
        metrics_list = []
        for pid, data in pairs.items():
            if len(data["preds"]) != 2:
                continue
            
            p1, p2 = data["preds"][0] >= 0.5, data["preds"][1] >= 0.5
            t1, t2 = data["targets"][0], data["targets"][1]
            
            pair_metric = {
                "accuracy": ((p1 == t1) + (p2 == t2)) / 2.0,
                "classification_pair_consistency": 1.0 if (p1 == t1 and p2 == t2) else 0.0
            }
            
            if scores:
                s1, s2 = data["scores"][0], data["scores"][1]
                latent_correct = 0.0
                if t1 == 0 and t2 == 1: latent_correct = 1.0 if s1 > s2 else 0.0
                elif t1 == 1 and t2 == 0: latent_correct = 1.0 if s2 > s1 else 0.0
                pair_metric["latent_pla"] = latent_correct
                
            metrics_list.append(pair_metric)
            
        print(f"Bootstrapping {split} ({len(metrics_list)} pairs)")
        results[split] = bootstrap_confidence_interval(metrics_list, args.n_iterations)
        
    # 2. Benchmark B (Scene-level)
    ctx_raw_path = out_dir / "context_challenge_results_raw.json"
    if ctx_raw_path.exists():
        with open(ctx_raw_path, "r") as f:
            ctx_raw = json.load(f)
            
        for mode in ["standard", "generic_text", "generic_wrong_demo"]:
            if mode not in ctx_raw: continue
            raw = ctx_raw[mode]
            preds = raw.get("_raw_preds", [])
            targets = raw.get("_raw_targets", [])
            pair_ids = raw.get("_raw_pair_ids", [])
            states = raw.get("_raw_states", [])
            
            if not pair_ids: continue
            
            scenes = {}
            for i in range(len(preds)):
                pid = pair_ids[i]
                scene_id = pid.rsplit("_", 1)[0] # Extract scene_id from context_scene_XXX_A
                state = states[i]
                
                if scene_id not in scenes:
                    scenes[scene_id] = {"preds": [], "targets": [], "states": []}
                scenes[scene_id]["preds"].append(preds[i])
                scenes[scene_id]["targets"].append(targets[i])
                scenes[scene_id]["states"].append(state)
                
            metrics_list = []
            for scene_id, data in scenes.items():
                p = data["preds"]
                t = data["targets"]
                st = data["states"]
                
                if len(p) != 2: continue # 2 predictions per scene (Task 1, Task 2)
                p1, p2 = p[0] >= 0.5, p[1] >= 0.5
                t1, t2 = t[0], t[1]
                state = st[0]
                
                is_correct = (p1 == t1 and p2 == t2)
                
                scene_metric = {
                    "context_pair_consistency": 1.0 if is_correct else 0.0,
                    "context_flip_rate": 1.0 if p1 != p2 else 0.0
                }
                
                if state in ["A", "B"]: # Reversal states
                    scene_metric["reversal_accuracy"] = 1.0 if is_correct else 0.0
                    
                metrics_list.append(scene_metric)
                
            print(f"Bootstrapping context challenge {mode} ({len(metrics_list)} scenes)")
            results[f"context_{mode}"] = bootstrap_confidence_interval(metrics_list, args.n_iterations)
            
    with open(out_dir / "bootstrap_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved bootstrap results to {out_dir / 'bootstrap_results.json'}")

if __name__ == "__main__":
    main()
