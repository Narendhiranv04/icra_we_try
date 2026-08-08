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
from src.learning.utils import resolve_checkpoint
from scripts.train_model import get_model
from sklearn.metrics import f1_score, balanced_accuracy_score

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
            task_ids = batch["task_id"]
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
                pair_preds[pid].append((preds[i], task_ids[i]))
                pair_targets[pid].append(targets_np[i])

            all_preds.extend(preds)
            all_targets.extend(targets_np)
            all_logits.extend(logits.squeeze(-1).cpu().numpy())
            
            if s is not None:
                if "all_scores" not in locals(): all_scores = []
                all_scores.extend(s.cpu().numpy())
                
            if isinstance(out, tuple) and len(out) == 3:
                latents = out[2]
                if latents is not None:
                    if "all_latents" not in locals(): all_latents = []
                    # Average over spatial dimensions for simpler storage
                    if latents.dim() == 4:
                        latents = latents.mean(dim=(2, 3))
                    all_latents.extend(latents.cpu().numpy())

    # Re-collect ordered pairs and states
    ordered_pair_ids = []
    ordered_states = []
    for pid, p_list in pair_preds.items():
        if len(p_list) == 2:
            ordered_pair_ids.extend([pid, pid])
            ordered_states.extend([pair_states[pid], pair_states[pid]])

    # Calculate metrics
    preds_cls = np.array(all_preds) >= 0.5
    targets_np = np.array(all_targets)
    acc = (preds_cls == targets_np).mean()

    # Calculate reversal (A, B) vs non-reversal (C, D) accuracy
    rev_preds, rev_targets = [], []
    non_preds, non_targets = [], []

    context_pair_consistency = 0
    reversal_acc_sum = 0
    rev_pairs = 0
    flip_count = 0

    state_correct = {"A": 0, "B": 0, "C": 0, "D": 0}
    state_total = {"A": 0, "B": 0, "C": 0, "D": 0}
    task_correct = {"task_1": 0, "task_2": 0}
    task_total = {"task_1": 0, "task_2": 0}

    for pid, p_list in pair_preds.items():
        if len(p_list) == 2:
            p1, p2 = p_list[0][0] >= 0.5, p_list[1][0] >= 0.5
            t1, t2 = pair_targets[pid][0], pair_targets[pid][1]
            tid1, tid2 = p_list[0][1], p_list[1][1]
            state = pair_states[pid]

            is_correct = (p1 == t1) and (p2 == t2)
            if is_correct:
                context_pair_consistency += 1

            if p1 != p2:
                flip_count += 1

            state_correct[state] += int(p1 == t1) + int(p2 == t2)
            state_total[state] += 2

            task_correct[tid1] += int(p1 == t1)
            task_total[tid1] += 1
            task_correct[tid2] += int(p2 == t2)
            task_total[tid2] += 1

            if state in ["A", "B"]:
                rev_pairs += 1
                if is_correct:
                    reversal_acc_sum += 1
                rev_preds.extend([p1, p2])
                rev_targets.extend([t1, t2])
            else:
                non_preds.extend([p1, p2])
                non_targets.extend([t1, t2])

    total_pairs = len([p for p in pair_preds.values() if len(p) == 2])

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy_score(targets_np, preds_cls)),
        "f1": float(f1_score(targets_np, preds_cls)),
        "reversal_accuracy": float(reversal_acc_sum / max(1, rev_pairs)),
        "context_pair_consistency": float(context_pair_consistency / max(1, total_pairs)),
        "context_flip_rate": float(flip_count / max(1, total_pairs)),
        "reversal_states_acc": float((np.array(rev_preds) == np.array(rev_targets)).mean()) if rev_preds else 0.0,
        "non_reversal_states_acc": float((np.array(non_preds) == np.array(non_targets)).mean()) if non_preds else 0.0,
        "state_A_acc": float(state_correct["A"] / max(1, state_total["A"])),
        "state_B_acc": float(state_correct["B"] / max(1, state_total["B"])),
        "state_C_acc": float(state_correct["C"] / max(1, state_total["C"])),
        "state_D_acc": float(state_correct["D"] / max(1, state_total["D"])),
        "task_1_acc": float(task_correct["task_1"] / max(1, task_total["task_1"])),
        "task_2_acc": float(task_correct["task_2"] / max(1, task_total["task_2"])),
        "_raw_preds": all_preds,
        "_raw_logits": all_logits,
        "_raw_targets": all_targets,
        "_raw_pair_ids": ordered_pair_ids,
        "_raw_states": ordered_states
    }

    if "all_scores" in locals() and len(all_scores) > 0:
        metrics["_raw_scores"] = all_scores
    if "all_latents" in locals() and len(all_latents) > 0:
        metrics["_raw_latents"] = all_latents

    print(f"[{desc}] Acc: {metrics['accuracy']:.4f} | Reversal Acc: {metrics['reversal_accuracy']:.4f} | Pair Cons: {metrics['context_pair_consistency']:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precompute", action="store_true", help="Precompute context challenge features")
    parser.add_argument("--index", default="data/manifests/context_challenge_manifest.jsonl")
    args = parser.parse_args()

    if args.precompute:
        os.system(f"python3 scripts/precompute_features.py --index {args.index} --out_dir data/features_context --force")

    out_dir = Path(args.experiment_dir)
    with open(out_dir / "resolved_config.yaml") as f:
        config = yaml.safe_load(f)

    model = get_model(config).to(args.device)

    ckpt_path = resolve_checkpoint(out_dir)
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True))

    # Needs a way to pass states in dataset. We will inject it!
    class ContextDataset(LearningDataset):
        def __init__(self, *args_tuple, generic_text=False, **kwargs):
            super().__init__(*args_tuple, **kwargs)
            self.generic_text = generic_text

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["state"] = self.records[idx].get("state", "")
            if self.generic_text:
                if "Perform the demonstrated task." in self.text_features:
                    item["text_feat"] = self.text_features["Perform the demonstrated task."]
                else:
                    raise RuntimeError("Missing generic text feature: 'Perform the demonstrated task.'")
            return item

    dataset = ContextDataset(
        index_path=args.index,
        features_dir="data/features_context",
        split="context_challenge",
        return_masks=False,
        seed=42,
        split_seed=42
    )

    # 1. Standard evaluation
    standard_metrics = evaluate_context(model, dataset, args.device, desc="Context Challenge (Standard)")

    # 2. Generic Demo Diagnostic
    # Force instruction to "Perform the demonstrated task."
    ds_generic = ContextDataset(
        index_path=args.index,
        features_dir="data/features_context",
        split="context_challenge",
        return_masks=False,
        generic_text=True
    )
    generic_metrics = evaluate_context(model, ds_generic, args.device, desc="Context Challenge (Generic Text)")

    ds_generic_wrong_demo = ContextDataset(
        index_path=args.index,
        features_dir="data/features_context",
        split="context_challenge",
        return_masks=False,
        generic_text=True
    )
    ds_generic_wrong_demo.ablation_mode = "wrong_demo"
    wrong_demo_metrics = evaluate_context(model, ds_generic_wrong_demo, args.device, desc="Context Challenge (Generic + Wrong Demo)")

    # Save raw arrays for bootstrapping
    raw_results = {
        "standard": {k: v for k, v in standard_metrics.items() if k.startswith("_raw")},
        "generic_text": {k: v for k, v in generic_metrics.items() if k.startswith("_raw")},
        "generic_wrong_demo": {k: v for k, v in wrong_demo_metrics.items() if k.startswith("_raw")}
    }
    
    # Convert numpy arrays/types to float for JSON
    def to_float_list(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, list):
                out[k] = [float(x) if isinstance(x, np.generic) else (x.tolist() if isinstance(x, np.ndarray) else x) for x in v]
        return out
    
    raw_results["standard"] = to_float_list(raw_results["standard"])
    raw_results["generic_text"] = to_float_list(raw_results["generic_text"])
    raw_results["generic_wrong_demo"] = to_float_list(raw_results["generic_wrong_demo"])

    with open(out_dir / "context_challenge_results_raw.json", "w") as f:
        json.dump(raw_results, f)

    # Compute Phase O delta metrics
    def compute_deltas(baseline, perturbed):
        b_preds = np.array(baseline.get("_raw_preds", []))
        p_preds = np.array(perturbed.get("_raw_preds", []))
        b_logits = np.array(baseline.get("_raw_logits", []))
        p_logits = np.array(perturbed.get("_raw_logits", []))
        b_scores = np.array(baseline.get("_raw_scores", []))
        p_scores = np.array(perturbed.get("_raw_scores", []))
        b_lat = baseline.get("_raw_latents", [])
        p_lat = perturbed.get("_raw_latents", [])
        
        res = {}
        if len(b_logits) > 0 and len(b_logits) == len(p_logits):
            res["mean_delta_logit"] = float(np.abs(p_logits - b_logits).mean())
        if len(b_preds) > 0 and len(b_preds) == len(p_preds):
            res["mean_delta_prob"] = float(np.abs(p_preds - b_preds).mean())
            b_class = (b_preds > 0.5).astype(int)
            p_class = (p_preds > 0.5).astype(int)
            res["flip_rate"] = float((b_class != p_class).mean())
        if len(b_scores) > 0 and len(b_scores) == len(p_scores):
            res["mean_delta_compatibility"] = float(np.abs(p_scores - b_scores).mean())
        if len(b_lat) > 0 and len(b_lat) == len(p_lat):
            b_lat, p_lat = np.array(b_lat), np.array(p_lat)
            norm_b = np.linalg.norm(b_lat, axis=1)
            norm_p = np.linalg.norm(p_lat, axis=1)
            norm_b[norm_b == 0] = 1e-8
            norm_p[norm_p == 0] = 1e-8
            cos_sim = np.sum(b_lat * p_lat, axis=1) / (norm_b * norm_p)
            res["mean_latent_cosine_change"] = float(1.0 - cos_sim.mean())
        return res

    generic_metrics["deltas_vs_wrong_demo"] = compute_deltas(generic_metrics, wrong_demo_metrics)

    # Cleanup raw arrays for JSON
    for m in [standard_metrics, generic_metrics, wrong_demo_metrics]:
        keys = list(m.keys())
        for k in keys:
            if k.startswith("_raw"):
                m.pop(k, None)

    results = {
        "standard": standard_metrics,
        "generic_text": generic_metrics,
        "generic_wrong_demo": wrong_demo_metrics
    }

    with open(out_dir / "context_challenge_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
