import argparse
import yaml
import os
import json
import torch
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
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    criterion = LearningLoss(
        margin=config.get("margin", 0.2),
        lambda_cls=config.get("lambda_cls", 1.0),
        lambda_rank=config.get("lambda_rank", 0.5),
        lambda_heat=config.get("lambda_heat", 0.5)
    )
    
    splits = ["id", "unseen_object", "unseen_background", "compositional"]
    all_metrics = {}
    
    for split in splits:
        dataset = LearningDataset(
            index_path=config.get("index_path", "learning_data/index.jsonl"),
            features_dir=config.get("feature_cache_path", "learning_data/features"),
            split=split,
            return_masks=config.get("heatmap", False)
        )
        if len(dataset) > 0:
            m = run_evaluation(model, dataset, criterion, device, config.get("batch_size", 8), desc=f"Split {split}")
            all_metrics[split] = m
            
    with open(out_dir / "metrics_by_split.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"Saved evaluation metrics to {out_dir / 'metrics_by_split.json'}")
    
    # Ablation logic
    if "relational" in str(out_dir) or "pooled_multimodal" in str(out_dir):
        print("Running wrong-instruction ablation...")
        id_dataset = LearningDataset(
            index_path=config.get("index_path", "learning_data/index.jsonl"),
            features_dir=config.get("feature_cache_path", "learning_data/features"),
            split="id_val",
            return_masks=config.get("heatmap", False)
        )
        
        # Modify instructions to wrong task
        for rec in id_dataset.records:
            if rec["task_id"] == "task_1":
                rec["instruction"] = "Place object1 in the target region."
            elif rec["task_id"] == "task_2":
                rec["instruction"] = "Open the box."
                
        wrong_instr_metrics = run_evaluation(model, id_dataset, criterion, device, batch_size=8, desc="Wrong Instruction")
        with open(out_dir / "wrong_instruction_results.json", "w") as f:
            json.dump(wrong_instr_metrics, f, indent=2)
            
        print("Running held-out paraphrase ablation...")
        paraphrase_dataset = LearningDataset(
            index_path=config.get("index_path", "learning_data/index.jsonl"),
            features_dir=config.get("feature_cache_path", "learning_data/features"),
            split="id_val",
            return_masks=config.get("heatmap", False)
        )
        
        # Modify instructions to held-out paraphrase
        for rec in paraphrase_dataset.records:
            if rec["task_id"] == "task_1":
                rec["instruction"] = "Uncover the box by opening its lid." # The 4th paraphrase
            elif rec["task_id"] == "task_2":
                rec["instruction"] = "Place the object into the indicated area." # The 4th paraphrase
                
        paraphrase_metrics = run_evaluation(model, paraphrase_dataset, criterion, device, batch_size=8, desc="Held-out Paraphrase")
        with open(out_dir / "paraphrase_results.json", "w") as f:
            json.dump(paraphrase_metrics, f, indent=2)
            
        print("Running wrong-demo ablation...")
        demo_dataset = LearningDataset(
            index_path=config.get("index_path", "learning_data/index.jsonl"),
            features_dir=config.get("feature_cache_path", "learning_data/features"),
            split="id_val",
            return_masks=config.get("heatmap", False)
        )
        
        # Find valid demo IDs for each task
        task_demos = {"task_1": set(), "task_2": set()}
        for rec in demo_dataset.records:
            task_demos[rec["task_id"]].add(rec["demonstration_id"])
            
        task1_demos = list(task_demos["task_1"])
        task2_demos = list(task_demos["task_2"])
        
        if task1_demos and task2_demos:
            for rec in demo_dataset.records:
                if rec["task_id"] == "task_1":
                    rec["demonstration_id"] = task2_demos[0]
                elif rec["task_id"] == "task_2":
                    rec["demonstration_id"] = task1_demos[0]
                    
            wrong_demo_metrics = run_evaluation(model, demo_dataset, criterion, device, batch_size=8, desc="Wrong Demo")
            with open(out_dir / "wrong_demo_results.json", "w") as f:
                json.dump(wrong_demo_metrics, f, indent=2)
            
if __name__ == "__main__":
    main()
