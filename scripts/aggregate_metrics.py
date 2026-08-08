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
        cond_paths = [f"learning_outputs/{model_dir}_seed{s}/conditioning_sensitivity.json" for s in seeds]
        cond_paths = [p for p in cond_paths if Path(p).exists()]
        if cond_paths:
            cond_metrics = []
            for cp in cond_paths:
                with open(cp) as f:
                    cond_metrics.append(json.load(f))

            agg_cond = {}
            conditions = cond_metrics[0].keys()
            for c in conditions:
                agg_cond[c] = {}
                splits_in_cond = cond_metrics[0][c].keys()
                for split in splits_in_cond:
                    agg_cond[c][split] = {}
                    for m in cond_metrics[0][c][split].keys():
                        vals = [cm[c][split].get(m, 0) for cm in cond_metrics if c in cm and split in cm[c]]
                        if vals and isinstance(vals[0], (int, float)):
                            agg_cond[c][split][m] = {
                                "mean": float(np.mean(vals)),
                                "std": float(np.std(vals))
                            }
            with open(out_base / "conditioning_sensitivity.json", "w") as f:
                json.dump(agg_cond, f, indent=2)

        # Aggregate Context Challenge results
        ctx_paths = [f"learning_outputs/{model_dir}_seed{s}/context_challenge_results.json" for s in seeds]
        ctx_paths = [p for p in ctx_paths if Path(p).exists()]
        if ctx_paths:
            ctx_metrics = []
            for cp in ctx_paths:
                with open(cp) as f:
                    ctx_metrics.append(json.load(f))

            agg_ctx = {}
            ctx_conditions = ctx_metrics[0].keys()
            for c in ctx_conditions:
                agg_ctx[c] = {}
                for m in ctx_metrics[0][c].keys():
                    vals = [cm[c].get(m, 0) for cm in ctx_metrics if c in cm]
                    if vals and isinstance(vals[0], (int, float)):
                        agg_ctx[c][m] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals))
                        }
            with open(out_base / "context_challenge_results.json", "w") as f:
                json.dump(agg_ctx, f, indent=2)

        # Aggregate Bootstrap results
        boot_paths = [f"learning_outputs/{model_dir}_seed{s}/bootstrap_results.json" for s in seeds]
        boot_paths = [p for p in boot_paths if Path(p).exists()]
        if boot_paths:
            boot_metrics = []
            for bp in boot_paths:
                with open(bp) as f:
                    boot_metrics.append(json.load(f))

            agg_boot = {}
            boot_splits = boot_metrics[0].keys()
            for split in boot_splits:
                agg_boot[split] = {}
                for m in boot_metrics[0][split].keys():
                    # Do not average the bounds across seeds, keep them separate per-seed
                    seed_vals = [{"seed": seeds[i], "mean": boot_metrics[i][split][m]["mean"],
                                  "lower": boot_metrics[i][split][m]["lower"],
                                  "upper": boot_metrics[i][split][m]["upper"]} 
                                 for i in range(len(boot_metrics)) if split in boot_metrics[i] and m in boot_metrics[i][split]]

                    if seed_vals:
                        agg_boot[split][m] = {
                            "per_seed": seed_vals
                        }
            with open(out_base / "bootstrap_results.json", "w") as f:
                json.dump(agg_boot, f, indent=2)

if __name__ == "__main__":
    main()
