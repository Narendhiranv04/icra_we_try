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
from src.learning.losses import LearningLoss
from src.learning.trainer import train_epoch, validate_epoch
from src.learning.models.query_only import QueryOnlyBaseline
from src.learning.models.pooled_multimodal import PooledMultimodalBaseline
from src.learning.models.relational_model import DemoLanguageConditionedRelationalModel

def get_model(config):
    model_type = config.get("model", "query_only")
    if model_type == "query_only":
        return QueryOnlyBaseline(in_features=768, hidden_dim=config.get("latent_dim", 256))
    elif model_type == "pooled_multimodal":
        return PooledMultimodalBaseline(
            text_dim=384, vision_dim=768, latent_dim=config.get("latent_dim", 256),
            use_text=config.get("use_text", True),
            use_demo=config.get("use_demo", True)
        )
    elif model_type == "relational":
        model = DemoLanguageConditionedRelationalModel(text_dim=384, vision_dim=768, latent_dim=config.get("latent_dim", 256), num_demo_frames=config.get("demo_frames", 4))
        
        if config.get("heatmap", False):
            from src.learning.models.heatmap_decoder import CausalHeatmapDecoder
            model.heatmap_decoder = CausalHeatmapDecoder(latent_dim=config.get("latent_dim", 256))
            
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    out_dir = Path(config["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save resolved config
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config, f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get("seed", 42))
    
    train_dataset = LearningDataset(
        index_path=config.get("index_path", "learning_data/index.jsonl"),
        features_dir=config.get("feature_cache_path", "learning_data/features"),
        split="id_train",
        return_masks=config.get("heatmap", False),
        train_ratio=config.get("train_ratio", 0.8),
        seed=config.get("seed", 42)
    )
    
    val_dataset = LearningDataset(
        index_path=config.get("index_path", "learning_data/index.jsonl"),
        features_dir=config.get("feature_cache_path", "learning_data/features"),
        split="id_val",
        return_masks=config.get("heatmap", False),
        train_ratio=config.get("train_ratio", 0.8),
        seed=config.get("seed", 42)
    )
    
    batch_size = config.get("batch_size", 8)
    
    train_sampler = PairBatchSampler(train_dataset, batch_size)
    val_sampler = PairBatchSampler(val_dataset, batch_size)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2)
    
    model = get_model(config).to(device)
    
    criterion = LearningLoss(
        margin=config.get("margin", 0.2),
        lambda_cls=config.get("lambda_cls", 1.0),
        lambda_rank=config.get("lambda_rank", 0.5),
        lambda_heat=config.get("lambda_heat", 0.5)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    
    scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", True) and device.type == "cuda" else None
    
    epochs = config.get("epochs", 30)
    best_val_loss = float('inf')
    
    history = {"train": [], "val": []}
    
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, config.get("accumulation_steps", 1))
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val PLA: {val_metrics.get('pla', 0):.4f}")
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), out_dir / "best.ckpt")
            
        torch.save(model.state_dict(), out_dir / "last.ckpt")
        
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"Training completed. Results saved in {out_dir}")

if __name__ == "__main__":
    main()
