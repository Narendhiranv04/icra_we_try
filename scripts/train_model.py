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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_gradients", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if args.seed is not None:
        config["model_seed"] = args.seed
        config["output_path"] = f"{config['output_path']}_seed{args.seed}"
        
    out_dir = Path(config["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save resolved config
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config, f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_seed = config.get("model_seed", 42)
    split_seed = config.get("split_seed", 42)
    sampler_seed = config.get("sampler_seed", 42)
    
    torch.manual_seed(model_seed)
    
    train_dataset = LearningDataset(
        index_path=config.get("index_path", "learning_data/index.jsonl"),
        features_dir=config.get("feature_cache_path", "learning_data/features"),
        split=config.get("train_split", "id_train"),
        return_masks=config.get("heatmap", False),
        train_ratio=config.get("train_ratio", 0.8),
        seed=model_seed,
        split_seed=split_seed
    )
    
    val_dataset = LearningDataset(
        index_path=config.get("index_path", "learning_data/index.jsonl"),
        features_dir=config.get("feature_cache_path", "learning_data/features"),
        split=config.get("val_split", "id_val"),
        return_masks=config.get("heatmap", False),
        train_ratio=config.get("train_ratio", 0.8),
        seed=model_seed,
        split_seed=split_seed
    )
    
    batch_size = config.get("batch_size", 8)
    
    train_sampler = PairBatchSampler(train_dataset, batch_size, seed=sampler_seed)
    val_sampler = PairBatchSampler(val_dataset, batch_size, seed=sampler_seed)
    
    # Log and assert train/val pair memberships
    train_pairs = sorted(list(set(r["pair_id"] for r in train_dataset.records)))
    val_pairs = sorted(list(set(r["pair_id"] for r in val_dataset.records)))
    assert not set(train_pairs).intersection(set(val_pairs)), "Train and Val pairs overlap!"
    
    with open(out_dir / "train_pairs.json", "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(out_dir / "val_pairs.json", "w") as f:
        json.dump(val_pairs, f, indent=2)
    
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
    
    epochs = config.get("epochs", 50)
    best_val_loss = float('inf')
    best_epoch = -1
    patience = config.get("patience", 7)
    epochs_no_improve = 0
    
    history = {"train": [], "val": []}
    
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, config.get("accumulation_steps", 1), log_gradients=args.log_gradients)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Clean up raw arrays before saving to history to avoid JSON errors
        train_clean = {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v for k, v in train_metrics.items() if not k.startswith('_raw')}
        val_clean = {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v for k, v in val_metrics.items() if not k.startswith('_raw')}
        
        history["train"].append(train_clean)
        history["val"].append(val_clean)
        
        if args.log_gradients and hasattr(model, '_first_batch_grad_norms'):
            with open(out_dir / f"gradient_norms_epoch{epoch+1}.json", "w") as f:
                json.dump(model._first_batch_grad_norms, f, indent=2)
            delattr(model, '_first_batch_grad_norms')

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val PLA: {val_metrics.get('pla', 0):.4f}")
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "best.ckpt")
        else:
            epochs_no_improve += 1
            
        torch.save(model.state_dict(), out_dir / "last.ckpt")
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    history["best_epoch"] = best_epoch
        
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"Training completed. Results saved in {out_dir}")

if __name__ == "__main__":
    main()
