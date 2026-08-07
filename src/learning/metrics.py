import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

def compute_metrics(preds, targets, scores=None, pair_ids=None, heat_preds=None, heat_targets=None):
    preds_np = np.array(preds) >= 0.5
    targets_np = np.array(targets)
    
    acc = accuracy_score(targets_np, preds_np)
    b_acc = balanced_accuracy_score(targets_np, preds_np)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    prec = precision_score(targets_np, preds_np, zero_division=0)
    rec = recall_score(targets_np, preds_np, zero_division=0)
    
    cm = confusion_matrix(targets_np, preds_np, labels=[0, 1]).tolist()
    
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm
    }
    
    if scores is not None and pair_ids is not None:
        pid_to_idx = {}
        for i, pid in enumerate(pair_ids):
            if pid not in pid_to_idx:
                pid_to_idx[pid] = []
            pid_to_idx[pid].append(i)
            
        correct_pairs = 0
        total_pairs = 0
        
        for pid, indices in pid_to_idx.items():
            if len(indices) == 2:
                i1, i2 = indices
                t1, t2 = targets[i1], targets[i2]
                s1, s2 = scores[i1], scores[i2]
                
                if t1 == 0 and t2 == 1:
                    total_pairs += 1
                    if s1 > s2:
                        correct_pairs += 1
                elif t1 == 1 and t2 == 0:
                    total_pairs += 1
                    if s2 > s1:
                        correct_pairs += 1
                        
        metrics["pla"] = correct_pairs / max(1, total_pairs)
        metrics["total_pairs"] = total_pairs
        
    if heat_preds is not None and heat_targets is not None:
        heat_preds_np = np.array(heat_preds) > 0.5
        heat_targets_np = np.array(heat_targets) > 0.5
        
        stop_iou = []
        stop_dice = []
        proceed_fp = []
        proceed_act = []
        
        for i in range(len(targets_np)):
            hp = heat_preds_np[i]
            ht = heat_targets_np[i]
            if targets_np[i] == 1: # STOP
                intersection = np.logical_and(hp, ht).sum()
                union = np.logical_or(hp, ht).sum()
                iou = intersection / max(1, union)
                dice = (2.0 * intersection) / max(1, hp.sum() + ht.sum())
                stop_iou.append(iou)
                stop_dice.append(dice)
            else: # PROCEED
                fp_frac = hp.sum() / max(1, hp.size)
                proceed_fp.append(fp_frac)
                proceed_act.append(np.array(heat_preds[i]).mean())
                
        metrics["stop_iou"] = float(np.mean(stop_iou)) if stop_iou else 0.0
        metrics["stop_dice"] = float(np.mean(stop_dice)) if stop_dice else 0.0
        metrics["proceed_fp_frac"] = float(np.mean(proceed_fp)) if proceed_fp else 0.0
        metrics["proceed_mean_act"] = float(np.mean(proceed_act)) if proceed_act else 0.0
        
    return metrics
