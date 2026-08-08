import os
import json
import torch
from pathlib import Path
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.dataset import LearningDataset
from src.learning.metrics import compute_metrics
from scripts.train_model import get_model

def evaluate_context(model, dataset, device, desc="Context Challenge"):
    model.eval()
    all_preds, all_targets, all_logits = [], [], []
    all_scores = []
    
    # Store predictions by pair_id to check consistency
    pair_preds = {}
    pair_targets = {}
    pair_states = {}
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            text_feat = batch["text_feat"].to(device)
            query_global = batch["query_global"].to(device)
            query_patch = batch["query_patch"].to(device)
            demo_global = batch["demo_global"].to(device)
            targets = batch["label"].to(device)
            pair_ids = batch["pair_id"]
            states = batch.get("state", [""] * len(targets)) # Dataset must return state!
            
            with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu"):
                import inspect
                sig = inspect.signature(model.forward)
                kwargs = {}
                if "text_feat" in sig.parameters: kwargs["text_feat"] = text_feat
                if "demo_global" in sig.parameters: kwargs["demo_global"] = demo_global
                if "query_global" in sig.parameters: kwargs["query_global"] = query_global
                if "query_patch" in sig.parameters: kwargs["query_patch"] = query_patch
                if "x" in sig.parameters: kwargs["x"] = query_global
                
                out = model(**kwargs)
                
                logits, s = None, None
                if isinstance(out, tuple):
                    if len(out) == 3:
                        logits, s, _ = out
                    else:
                        logits, s = out
                else:
                    logits = out
                    
            preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(len(preds)):
                pid = pair_ids[i]
                if pid not in pair_preds:
                    pair_preds[pid] = []
                    pair_targets[pid] = []
                    pair_states[pid] = states[i]
                pair_preds[pid].append(preds[i])
                pair_targets[pid].append(targets_np[i])
                
            all_preds.extend(preds)
            all_targets.extend(targets_np)
            all_logits.extend(logits.squeeze(-1).cpu().numpy())
            
    # Calculate metrics
    preds_cls = np.array(all_preds) >= 0.5
    targets_np = np.array(all_targets)
    acc = (preds_cls == targets_np).mean()
    
    # Calculate reversal (A, B) vs non-reversal (C, D) accuracy
    rev_preds, rev_targets = [], []
    non_preds, non_targets = [], []
    
    context_pair_consistency = 0
    reversal_acc = 0
    rev_pairs = 0
    
    for pid, p_list in pair_preds.items():
        if len(p_list) == 2:
            p1, p2 = p_list[0] >= 0.5, p_list[1] >= 0.5
            t1, t2 = pair_targets[pid][0], pair_targets[pid][1]
            state = pair_states[pid]
            
            is_correct = (p1 == t1) and (p2 == t2)
            if is_correct:
                context_pair_consistency += 1
                
            if state in ["A", "B"]:
                rev_pairs += 1
                if p1 != p2:
                    reversal_acc += 1
                rev_preds.extend([p1, p2])
                rev_targets.extend([t1, t2])
            else:
                non_preds.extend([p1, p2])
                non_targets.extend([t1, t2])
                
    total_pairs = len([p for p in pair_preds.values() if len(p) == 2])
    
    metrics = {
        "accuracy": float(acc),
        "reversal_accuracy": float(reversal_acc / max(1, rev_pairs)),
        "context_pair_consistency": float(context_pair_consistency / max(1, total_pairs)),
        "reversal_states_acc": float((np.array(rev_preds) == np.array(rev_targets)).mean()) if rev_preds else 0.0,
        "non_reversal_states_acc": float((np.array(non_preds) == np.array(non_targets)).mean()) if non_preds else 0.0,
        "_raw_preds": all_preds,
        "_raw_logits": all_logits,
        "_raw_targets": all_targets
    }
    
    print(f"[{desc}] Acc: {metrics['accuracy']:.4f} | Reversal Acc: {metrics['reversal_accuracy']:.4f} | Pair Cons: {metrics['context_pair_consistency']:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precompute", action="store_true", help="Precompute context challenge features")
    args = parser.parse_args()
    
    if args.precompute:
        os.system("python3 scripts/precompute_features.py --index data/manifests/context_challenge_manifest.jsonl --out_dir data/features_context --force")
        
    out_dir = Path(args.experiment_dir)
    with open(out_dir / "resolved_config.yaml") as f:
        config = yaml.safe_load(f)
        
    model = get_model(config).to(args.device)
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=args.device))
    
    # Needs a way to pass states in dataset. We will inject it!
    class ContextDataset(LearningDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["state"] = self.records[idx].get("state", "")
            return item
            
    dataset = ContextDataset(
        index_path="data/manifests/context_challenge_manifest.jsonl",
        features_dir="data/features_context",
        split="id_val", # everything is id_val in this manifest for simplicity
        return_masks=False,
        seed=42,
        split_seed=42
    )
    
    # 1. Standard evaluation
    standard_metrics = evaluate_context(model, dataset, args.device, desc="Context Challenge (Standard)")
    
    # 2. Generic Demo Diagnostic
    # Force instruction to "Perform the demonstrated task."
    ds_generic = ContextDataset(
        index_path="data/manifests/context_challenge_manifest.jsonl",
        features_dir="data/features_context",
        split="id_val",
        return_masks=False
    )
    
    orig_getitem = ds_generic.__getitem__
    # We need a text feature for "Perform the demonstrated task."
    import sys
    # Actually, we can just use zero_text for the diagnostic, or if the feature is not precomputed, we use zero.
    # Let's use zero_text ablation mode!
    ds_generic.ablation_mode = "zero_text"
    
    generic_metrics = evaluate_context(model, ds_generic, args.device, desc="Context Challenge (Zero Text)")
    
    ds_generic_wrong_demo = ContextDataset(
        index_path="data/manifests/context_challenge_manifest.jsonl",
        features_dir="data/features_context",
        split="id_val",
        return_masks=False
    )
    ds_generic_wrong_demo.ablation_mode = "wrong_demo"
    wrong_demo_metrics = evaluate_context(model, ds_generic_wrong_demo, args.device, desc="Context Challenge (Wrong Demo)")
    
    # Cleanup raw arrays for JSON
    for m in [standard_metrics, generic_metrics, wrong_demo_metrics]:
        for k in ["_raw_preds", "_raw_logits", "_raw_targets"]:
            m.pop(k, None)
            
    results = {
        "standard": standard_metrics,
        "zero_text": generic_metrics,
        "wrong_demo": wrong_demo_metrics
    }
    
    with open(out_dir / "context_challenge_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
