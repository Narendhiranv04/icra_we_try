import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def bootstrap_confidence_interval(metrics_list, n_iterations=2000, ci=0.95):
    """
    Given a list of N metrics dictionaries (each representing one test sample or pair),
    sample N items with replacement n_iterations times and compute the mean of each metric.
    Returns the lower and upper bounds of the confidence interval.
    """
    if not metrics_list:
        return {}
        
    keys = [k for k in metrics_list[0].keys() if isinstance(metrics_list[0][k], (int, float))]
    
    bootstrapped_means = {k: [] for k in keys}
    n = len(metrics_list)
    
    for _ in tqdm(range(n_iterations), desc="Bootstrapping"):
        indices = np.random.randint(0, n, n)
        for k in keys:
            sample_mean = np.mean([metrics_list[i][k] for i in indices])
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

def compute_raw_metrics(targets, preds, logits, scores=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    preds_cls = np.array(preds) >= 0.5
    
    acc = accuracy_score(targets, preds_cls)
    b_acc = balanced_accuracy_score(targets, preds_cls)
    f1 = f1_score(targets, preds_cls, zero_division=0)
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "f1": float(f1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--n-iterations", type=int, default=2000)
    args = parser.parse_args()
    
    out_dir = Path(args.experiment_dir)
    val_metrics_path = out_dir / "val_metrics.json"
    
    if not val_metrics_path.exists():
        print(f"Metrics not found at {val_metrics_path}")
        return
        
    with open(val_metrics_path, "r") as f:
        metrics = json.load(f)
        
    # Standard bootstrap on id_val
    if "id_val" in metrics:
        print("Bootstrapping requires raw pair-level data. This will be implemented in run_all_experiments.sh aggregation.")

if __name__ == "__main__":
    main()
