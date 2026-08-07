import json
import numpy as np
from pathlib import Path
import glob
import sys

def main():
    base_dirs = [
        "query_only", "language_query", "demo_query", 
        "full_no_ranking", "pooled_multimodal", "relational_heatmap"
    ]
    
    seeds = [11, 23, 42, 67, 101]
    
    for model_dir in base_dirs:
        seed_metrics = []
        out_base = Path(f"artifacts/learning_stage1/{model_dir}")
        out_base.mkdir(parents=True, exist_ok=True)
        
        for seed in seeds:
            exp_dir = Path(f"learning_outputs/{model_dir}_seed{seed}")
            metrics_path = exp_dir / "metrics_by_split.json"
            if not metrics_path.exists():
                print(f"Warning: {metrics_path} missing.")
                continue
            with open(metrics_path) as f:
                seed_metrics.append({"seed": seed, "metrics": json.load(f)})
                
        if not seed_metrics:
            continue
            
        with open(out_base / "seed_metrics.json", "w") as f:
            json.dump(seed_metrics, f, indent=2)
            
        # Aggregate
        splits = seed_metrics[0]["metrics"].keys()
        agg = {}
        for split in splits:
            agg[split] = {}
            for metric in seed_metrics[0]["metrics"][split].keys():
                vals = [s["metrics"][split].get(metric, 0) for s in seed_metrics]
                if isinstance(vals[0], (int, float)):
                    agg[split][metric] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals))
                    }
        with open(out_base / "aggregate_metrics.json", "w") as f:
            json.dump(agg, f, indent=2)
            
        print(f"Aggregated metrics for {model_dir}")
        
        # Aggregate condition sensitivity if it exists
        cond_paths = glob.glob(f"learning_outputs/{model_dir}_seed*/conditioning_sensitivity.json")
        if cond_paths:
            cond_metrics = []
            for cp in cond_paths:
                with open(cp) as f:
                    cond_metrics.append(json.load(f))
                    
            agg_cond = {}
            conditions = cond_metrics[0].keys()
            for c in conditions:
                agg_cond[c] = {}
                for m in cond_metrics[0][c].keys():
                    vals = [cm[c].get(m, 0) for cm in cond_metrics]
                    if isinstance(vals[0], (int, float)):
                        agg_cond[c][m] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals))
                        }
            with open(out_base / "conditioning_sensitivity.json", "w") as f:
                json.dump(agg_cond, f, indent=2)
                
if __name__ == "__main__":
    main()
