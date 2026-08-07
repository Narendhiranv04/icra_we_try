import argparse
import yaml
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.dataset import LearningDataset, PairBatchSampler
from src.learning.trainer import validate_epoch
from src.learning.losses import LearningLoss
from scripts.train_model import get_model

def run_evaluation(model, dataset, criterion, device, batch_size=8, desc=""):
    sampler = PairBatchSampler(dataset, batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    metrics = validate_epoch(model, loader, criterion, device)
    print(f"[{desc}] Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f} | PLA: {metrics.get('pla', 0):.4f}")
    return metrics

def compare_metrics(baseline, perturbed):
    b_preds = np.array(baseline.get("_raw_preds", []))
    p_preds = np.array(perturbed.get("_raw_preds", []))
    
    b_scores = np.array(baseline.get("_raw_scores", []))
    p_scores = np.array(perturbed.get("_raw_scores", []))
    
    res = {
        "accuracy": perturbed["accuracy"],
        "f1": perturbed.get("f1", 0),
        "pla": perturbed.get("pla", 0),
    }
    
    if len(b_preds) > 0 and len(b_preds) == len(p_preds):
        b_class = (b_preds > 0.5).astype(int)
        p_class = (p_preds > 0.5).astype(int)
        flips = (b_class != p_class).mean()
        delta_logit = np.abs(np.log(p_preds / (1 - p_preds + 1e-9)) - np.log(b_preds / (1 - b_preds + 1e-9))).mean()
        res["flip_rate"] = float(flips)
        res["mean_delta_logit"] = float(delta_logit)
        res["mean_confidence"] = float(np.abs(p_preds - 0.5).mean() * 2)
        
    if len(b_scores) > 0 and len(b_scores) == len(p_scores):
        delta_compat = np.abs(p_scores - b_scores).mean()
        res["mean_compatibility"] = float(p_scores.mean())
        res["mean_delta_compatibility"] = float(delta_compat)
        
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to learning_outputs/<experiment_name>")
    args = parser.parse_args()
    
    out_dir = Path(args.dir)
    with open(out_dir / "resolved_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)
    
    ckpt_path = out_dir / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = out_dir / "last.ckpt"
        if not ckpt_path.exists():
            print(f"Skipping {out_dir} - no checkpoints.")
            return
            
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    criterion = LearningLoss(
        margin=config.get("margin", 0.2),
        lambda_cls=config.get("lambda_cls", 1.0),
        lambda_rank=config.get("lambda_rank", 0.0),
        lambda_heat=config.get("lambda_heat", 0.0)
    )
    
    splits = ["id_train", "id_val", "unseen_object", "unseen_background", "compositional"]
    all_metrics = {}
    
    for split in splits:
        dataset = LearningDataset(
            index_path=config.get("index_path", "learning_data/index.jsonl"),
            features_dir=config.get("feature_cache_path", "learning_data/features"),
            split=split,
            return_masks=config.get("heatmap", False)
        )
        if len(dataset) > 0:
            desc = f"[TRAIN] Split {split}" if split == "id_train" else f"Split {split}"
            m = run_evaluation(model, dataset, criterion, device, config.get("batch_size", 8), desc=desc)
            # Remove raw arrays before saving
            m_clean = {k: v for k, v in m.items() if not k.startswith("_raw")}
            if split != "id_train":
                all_metrics[split] = m_clean
            if split == "id_val":
                baseline_id_val = m
            
    with open(out_dir / "metrics_by_split.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"Saved evaluation metrics to {out_dir / 'metrics_by_split.json'}")
    
    # Ablation logic
    if "relational" in str(out_dir) or "pooled_multimodal" in str(out_dir) or "demo_query" in str(out_dir) or "language_query" in str(out_dir):
        print("Running Conditioning Diagnostics...")
        ablation_results = {}
        
        def run_ablation(dataset_mod_func, name):
            ds = LearningDataset(
                index_path=config.get("index_path", "learning_data/index.jsonl"),
                features_dir=config.get("feature_cache_path", "learning_data/features"),
                split="id_val",
                return_masks=config.get("heatmap", False)
            )
            dataset_mod_func(ds)
            m = run_evaluation(model, ds, criterion, device, batch_size=8, desc=name)
            res = compare_metrics(baseline_id_val, m)
            ablation_results[name] = res

        # Wrong Instruction
        def mod_wrong_instr(ds):
            for rec in ds.records:
                if rec["task_id"] == "task_1":
                    rec["instruction"] = "Place object1 in the target region."
                else:
                    rec["instruction"] = "Open the box."
        run_ablation(mod_wrong_instr, "Wrong Instruction")
        
        # Held-out Paraphrase
        def mod_paraphrase(ds):
            for rec in ds.records:
                if rec["task_id"] == "task_1":
                    rec["instruction"] = "Uncover the box by opening its lid."
                else:
                    rec["instruction"] = "Place the object into the indicated area."
        run_ablation(mod_paraphrase, "Held-out Paraphrase")
        
        # Zero Text
        def mod_zero_text(ds):
            for inst in ds.text_features:
                ds.text_features[inst] = torch.zeros_like(ds.text_features[inst])
        run_ablation(mod_zero_text, "Zero Text")
        
        # Wrong Demo
        def mod_wrong_demo(ds):
            t1 = [r["demonstration_id"] for r in ds.records if r["task_id"] == "task_1"]
            t2 = [r["demonstration_id"] for r in ds.records if r["task_id"] == "task_2"]
            if t1 and t2:
                for rec in ds.records:
                    if rec["task_id"] == "task_1":
                        rec["demonstration_id"] = t2[0]
                    else:
                        rec["demonstration_id"] = t1[0]
        run_ablation(mod_wrong_demo, "Wrong Demo")
        
        # Zero Demo
        def mod_zero_demo(ds):
            import glob
            for f in glob.glob(str(Path(ds.features_dir) / "demo_*_features.pt")):
                d = torch.load(f, weights_only=True)
                d["global"] = torch.zeros_like(d["global"])
                d["patch"] = torch.zeros_like(d["patch"])
                torch.save(d, f)
            # Restore later? Actually better to just zero them in memory but dataset loads on the fly.
            # Instead we can zero it inside trainer... wait.
            # Zeroing in file is bad because it affects parallel runs. Let's just modify the `__getitem__` temporarily!
        
        # For zero demo, let's inject a zeroing transform.
        ds_zero_demo = LearningDataset(
            index_path=config.get("index_path", "learning_data/index.jsonl"),
            features_dir=config.get("feature_cache_path", "learning_data/features"),
            split="id_val",
            return_masks=config.get("heatmap", False)
        )
        orig_getitem = ds_zero_demo.__getitem__
        def zero_demo_getitem(idx):
            item = orig_getitem(idx)
            item["demo_global"] = torch.zeros_like(item["demo_global"])
            item["demo_patch"] = torch.zeros_like(item["demo_patch"])
            return item
        ds_zero_demo.__getitem__ = zero_demo_getitem
        
        m = run_evaluation(model, ds_zero_demo, criterion, device, batch_size=8, desc="Zero Demo")
        ablation_results["Zero Demo"] = compare_metrics(baseline_id_val, m)
        
        with open(out_dir / "conditioning_sensitivity.json", "w") as f:
            json.dump(ablation_results, f, indent=2)
            
if __name__ == "__main__":
    main()
