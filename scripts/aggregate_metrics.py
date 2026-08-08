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
                raise RuntimeError(f"Error: {metrics_path} missing for model {model_dir} seed {seed}.")
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
                vals = []
                for s in seed_metrics:
                    if metric not in s["metrics"][split]:
                        raise RuntimeError(f"Metric {metric} missing in split {split} for seed {s['seed']}")
                    vals.append(s["metrics"][split][metric])
                    
                if isinstance(vals[0], (int, float)):
                    agg[split][metric] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals))
                    }
        with open(out_base / "aggregate_metrics.json", "w") as f:
            json.dump(agg, f, indent=2)

        print(f"Aggregated metrics for {model_dir}")

        # Aggregate condition sensitivity if it exists
        cond_paths = [Path(f"learning_outputs/{model_dir}_seed{s}/conditioning_sensitivity.json") for s in seeds]
        if all(p.exists() for p in cond_paths):
            cond_metrics = []
            for cp, s in zip(cond_paths, seeds):
                with open(cp) as f:
                    cond_metrics.append({"seed": s, "metrics": json.load(f)})

            agg_cond = {}
            conditions = cond_metrics[0]["metrics"].keys()
            for c in conditions:
                agg_cond[c] = {}
                splits_in_cond = cond_metrics[0]["metrics"][c].keys()
                for split in splits_in_cond:
                    agg_cond[c][split] = {}
                    for m in cond_metrics[0]["metrics"][c][split].keys():
                        vals = []
                        for cm in cond_metrics:
                            if m not in cm["metrics"][c][split]:
                                raise RuntimeError(f"Metric {m} missing in cond {c} split {split} for seed {cm['seed']}")
                            vals.append(cm["metrics"][c][split][m])
                            
                        if vals and isinstance(vals[0], (int, float)):
                            agg_cond[c][split][m] = {
                                "mean": float(np.mean(vals)),
                                "std": float(np.std(vals))
                            }
            with open(out_base / "conditioning_sensitivity.json", "w") as f:
                json.dump(agg_cond, f, indent=2)

        # Aggregate Context Challenge results
        ctx_paths = [Path(f"learning_outputs/{model_dir}_seed{s}/context_challenge_results.json") for s in seeds]
        if all(p.exists() for p in ctx_paths):
            ctx_metrics = []
            for cp, s in zip(ctx_paths, seeds):
                with open(cp) as f:
                    ctx_metrics.append({"seed": s, "metrics": json.load(f)})

            agg_ctx = {}
            ctx_conditions = ctx_metrics[0]["metrics"].keys()
            for c in ctx_conditions:
                agg_ctx[c] = {}
                for m in ctx_metrics[0]["metrics"][c].keys():
                    vals = []
                    for cm in ctx_metrics:
                        if m not in cm["metrics"][c]:
                            raise RuntimeError(f"Metric {m} missing in ctx {c} for seed {cm['seed']}")
                        vals.append(cm["metrics"][c][m])
                        
                    if vals and isinstance(vals[0], (int, float)):
                        agg_ctx[c][m] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals))
                        }
            with open(out_base / "context_challenge_results.json", "w") as f:
                json.dump(agg_ctx, f, indent=2)

        # Aggregate Bootstrap results
        boot_paths = [Path(f"learning_outputs/{model_dir}_seed{s}/bootstrap_results.json") for s in seeds]
        if all(p.exists() for p in boot_paths):
            boot_metrics = []
            for bp, s in zip(boot_paths, seeds):
                with open(bp) as f:
                    boot_metrics.append({"seed": s, "metrics": json.load(f)})

            agg_boot = {}
            boot_splits = boot_metrics[0]["metrics"].keys()
            for split in boot_splits:
                agg_boot[split] = {}
                for m in boot_metrics[0]["metrics"][split].keys():
                    seed_vals = []
                    for i in range(len(boot_metrics)):
                        if split not in boot_metrics[i]["metrics"] or m not in boot_metrics[i]["metrics"][split]:
                            raise RuntimeError(f"Metric {m} missing in bootstrap split {split} for seed {boot_metrics[i]['seed']}")
                        bm = boot_metrics[i]["metrics"][split][m]
                        seed_vals.append({
                            "seed": boot_metrics[i]["seed"],
                            "mean": bm["mean"],
                            "lower": bm["lower"],
                            "upper": bm["upper"]
                        })

                    if seed_vals:
                        agg_boot[split][m] = {
                            "per_seed": seed_vals
                        }
            with open(out_base / "bootstrap_results.json", "w") as f:
                json.dump(agg_boot, f, indent=2)

if __name__ == "__main__":
    main()
