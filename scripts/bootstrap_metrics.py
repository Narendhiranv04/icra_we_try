import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score

def compute_metrics(preds, targets, scores=None, state_info=None, is_context=False):
    preds_cls = (np.array(preds) >= 0.5).astype(int)
    targets = np.array(targets)
    
    metrics = {}
    if len(preds_cls) > 0:
        metrics["accuracy"] = float((preds_cls == targets).mean())
        metrics["f1"] = float(f1_score(targets, preds_cls, zero_division=0))
        # Handle cases where only one class is present in the bootstrap sample
        if len(np.unique(targets)) > 1:
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds_cls))
        else:
            metrics["balanced_accuracy"] = metrics["accuracy"]
            
    if not is_context:
        # Pair-level consistency (Benchmark A)
        # Assumes preds and targets are ordered such that every 2 elements form a pair
        consistencies = []
        latent_correct = []
        for i in range(0, len(preds_cls), 2):
            if i + 1 >= len(preds_cls): break
            p1, p2 = preds_cls[i], preds_cls[i+1]
            t1, t2 = targets[i], targets[i+1]
            consistencies.append(1.0 if (p1 == t1 and p2 == t2) else 0.0)
            
            if scores is not None and len(scores) > i+1:
                s1, s2 = scores[i], scores[i+1]
                if t1 == 0 and t2 == 1: latent_correct.append(1.0 if s1 > s2 else 0.0)
                elif t1 == 1 and t2 == 0: latent_correct.append(1.0 if s2 > s1 else 0.0)
                
        if consistencies:
            metrics["classification_pair_consistency"] = float(np.mean(consistencies))
        if latent_correct:
            metrics["latent_pla"] = float(np.mean(latent_correct))
    else:
        # Scene-level consistency (Benchmark B)
        consistencies = []
        flips = []
        rev_accs = []
        ab_acc = []
        cd_acc = []
        for i in range(0, len(preds_cls), 2):
            if i + 1 >= len(preds_cls): break
            p1, p2 = preds_cls[i], preds_cls[i+1]
            t1, t2 = targets[i], targets[i+1]
            is_correct = (p1 == t1 and p2 == t2)
            consistencies.append(1.0 if is_correct else 0.0)
            flips.append(1.0 if p1 != p2 else 0.0)
            
            if state_info:
                st = state_info[i//2]
                if st in ["A", "B"]:
                    rev_accs.append(1.0 if is_correct else 0.0)
                    ab_acc.extend([1.0 if p1 == t1 else 0.0, 1.0 if p2 == t2 else 0.0])
                else:
                    cd_acc.extend([1.0 if p1 == t1 else 0.0, 1.0 if p2 == t2 else 0.0])
                    
        if consistencies:
            metrics["context_pair_consistency"] = float(np.mean(consistencies))
            metrics["context_flip_rate"] = float(np.mean(flips))
        if rev_accs:
            metrics["reversal_accuracy"] = float(np.mean(rev_accs))
        if ab_acc:
            metrics["reversal_states_acc"] = float(np.mean(ab_acc))
        if cd_acc:
            metrics["non_reversal_states_acc"] = float(np.mean(cd_acc))
            
    return metrics

def bootstrap_confidence_interval(units, rng, n_iterations=2000, ci=0.95, is_context=False):
    if not units:
        return {}
    
    n = len(units)
    bootstrapped_metrics = {}
    
    for _ in tqdm(range(n_iterations), desc="Bootstrapping", leave=False):
        indices = rng.integers(0, n, n)
        
        b_preds = []
        b_targets = []
        b_scores = [] if "scores" in units[0] else None
        b_states = [] if "state" in units[0] else None
        
        for i in indices:
            unit = units[i]
            b_preds.extend(unit["preds"])
            b_targets.extend(unit["targets"])
            if b_scores is not None:
                b_scores.extend(unit["scores"])
            if b_states is not None:
                b_states.append(unit["state"])
                
        metrics = compute_metrics(b_preds, b_targets, scores=b_scores, state_info=b_states, is_context=is_context)
        
        for k, v in metrics.items():
            if k not in bootstrapped_metrics:
                bootstrapped_metrics[k] = []
            bootstrapped_metrics[k].append(v)
            
    results = {}
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    
    for k, arr in bootstrapped_metrics.items():
        arr = np.array(arr)
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
    
    rng = np.random.default_rng(42)
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
                
        units = []
        for pid, data in pairs.items():
            if len(data["preds"]) != 2:
                continue
            units.append(data)
            
        print(f"Bootstrapping {split} ({len(units)} pairs)")
        results[split] = bootstrap_confidence_interval(units, rng, args.n_iterations, is_context=False)
        
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
                state = states[i]
                
                if pid not in scenes:
                    scenes[pid] = {"preds": [], "targets": [], "state": state}
                scenes[pid]["preds"].append(preds[i])
                scenes[pid]["targets"].append(targets[i])
                
            units = []
            for scene_id, data in scenes.items():
                if len(data["preds"]) != 2: continue
                units.append(data)
                
            print(f"Bootstrapping context challenge {mode} ({len(units)} scenes)")
            results[f"context_{mode}"] = bootstrap_confidence_interval(units, rng, args.n_iterations, is_context=True)
            
    with open(out_dir / "bootstrap_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved bootstrap results to {out_dir / 'bootstrap_results.json'}")

if __name__ == "__main__":
    main()
