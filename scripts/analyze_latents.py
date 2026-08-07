import argparse
import yaml
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.dataset import LearningDataset, PairBatchSampler
from scripts.train_model import get_model

def run_latent_analysis(model, dataset, device, out_dir):
    sampler = PairBatchSampler(dataset, 8)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    
    model.eval()
    
    all_z_S = []
    all_z_Q = []
    all_s = []
    all_targets = []
    all_pair_ids = []
    
    with torch.no_grad():
        for batch in loader:
            text_feat = batch["text_feat"].to(device)
            query_global = batch["query_global"].to(device)
            query_patch = batch["query_patch"].to(device)
            demo_global = batch["demo_global"].to(device)
            targets = batch["label"].to(device)
            pair_ids = batch["pair_id"]
            
            with torch.cuda.amp.autocast():
                # We need z_S and z_Q from relational model. 
                # Let's rebuild the forward pass manually or use the model if it returns Z_R.
                if hasattr(model, "temporal_encoder"):
                    B, K, _ = demo_global.shape
                    z_t = model.proj_t(text_feat).unsqueeze(1)
                    z_d = model.proj_v_global(demo_global)
                    seq = torch.cat([z_t, z_d], dim=1)
                    seq = seq + model.temporal_pos_embed[:, :1+K, :]
                    
                    Z_S = seq
                    for layer in model.temporal_encoder:
                        Z_S = layer(Z_S)
                    z_S = Z_S.mean(dim=1)
                    
                    Z_Q = model.proj_v_patch(query_patch)
                    Z_R = Z_Q
                    for layer in model.cross_attention:
                        Z_R = layer(Z_R, Z_S)
                        
                    z_R = Z_R.mean(dim=1)
                    z_Q = z_R
                    
                    z_S_norm = F.normalize(z_S, dim=-1)
                    z_Q_norm = F.normalize(z_Q, dim=-1)
                    s = (z_S_norm * z_Q_norm).sum(dim=-1)
                    
                    all_z_S.extend(z_S.cpu().numpy())
                    all_z_Q.extend(z_Q.cpu().numpy())
                    all_s.extend(s.cpu().numpy())
                else:
                    return # not relational model
                
            all_targets.extend(targets.cpu().numpy())
            all_pair_ids.extend(pair_ids)
            
    # Metrics
    if not all_s:
        return
        
    targets_np = np.array(all_targets)
    s_np = np.array(all_s)
    z_S_np = np.array(all_z_S)
    z_Q_np = np.array(all_z_Q)
    
    proceed_s = s_np[targets_np == 0]
    stop_s = s_np[targets_np == 1]
    
    mean_s_proceed = float(proceed_s.mean()) if len(proceed_s) > 0 else 0
    mean_s_stop = float(stop_s.mean()) if len(stop_s) > 0 else 0
    margin = mean_s_proceed - mean_s_stop
    
    # PLA
    pid_to_idx = {}
    for i, pid in enumerate(all_pair_ids):
        if pid not in pid_to_idx:
            pid_to_idx[pid] = []
        pid_to_idx[pid].append(i)
        
    correct_pairs = 0
    total_pairs = 0
    
    for pid, indices in pid_to_idx.items():
        if len(indices) == 2:
            i1, i2 = indices
            t1, t2 = targets_np[i1], targets_np[i2]
            s1, s2 = s_np[i1], s_np[i2]
            
            if t1 == 0 and t2 == 1:
                total_pairs += 1
                if s1 > s2:
                    correct_pairs += 1
            elif t1 == 1 and t2 == 0:
                total_pairs += 1
                if s2 > s1:
                    correct_pairs += 1
                    
    pla = correct_pairs / max(1, total_pairs)
    
    # Norm statistics
    z_S_norms = np.linalg.norm(z_S_np, axis=1)
    z_Q_norms = np.linalg.norm(z_Q_np, axis=1)
    
    # Collapse check (variance)
    z_S_var = float(np.var(z_S_np, axis=0).mean())
    z_Q_var = float(np.var(z_Q_np, axis=0).mean())
    
    metrics = {
        "mean_s_proceed": mean_s_proceed,
        "mean_s_stop": mean_s_stop,
        "margin": margin,
        "pla": pla,
        "z_S_mean_norm": float(z_S_norms.mean()),
        "z_S_std_norm": float(z_S_norms.std()),
        "z_Q_mean_norm": float(z_Q_norms.mean()),
        "z_Q_std_norm": float(z_Q_norms.std()),
        "z_S_mean_variance": z_S_var,
        "z_Q_mean_variance": z_Q_var,
        "collapse_detected": z_S_var < 1e-4 or z_Q_var < 1e-4
    }
    
    with open(out_dir / "latent_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Latent analysis completed. Metrics saved to {out_dir / 'latent_metrics.json'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to learning_outputs/<experiment_name>")
    args = parser.parse_args()
    
    out_dir = Path(args.dir)
    with open(out_dir / "resolved_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    if "relational" not in config.get("model", ""):
        print("Model is not relational, skipping latent analysis.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)
    
    ckpt_path = out_dir / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = out_dir / "last.ckpt"
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    dataset = LearningDataset(
        index_path=config.get("index_path", "learning_data/index.jsonl"),
        features_dir=config.get("feature_cache_path", "learning_data/features"),
        split="id_val",
        return_masks=config.get("heatmap", False)
    )
    
    run_latent_analysis(model, dataset, device, out_dir)

if __name__ == "__main__":
    main()
