import torch
import numpy as np
import yaml
import os
from pathlib import Path
from PIL import Image
import cv2
import torchvision.transforms as T

from src.learning.dataset import LearningDataset
from scripts.train_model import get_model

def generate_heatmaps():
    out_dir = Path("learning_outputs/relational_heatmap")
    with open(out_dir / "resolved_config.yaml") as f:
        config = yaml.safe_load(f)

    model = get_model(config)
    
    if config.get("heatmap", False):
        from src.learning.models.heatmap_decoder import CausalHeatmapDecoder
        model.heatmap_decoder = CausalHeatmapDecoder(latent_dim=config.get("latent_dim", 256))
        
    ckpt_path = out_dir / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = out_dir / "last.ckpt"
        if not ckpt_path.exists():
            print("No checkpoints found, skipping heatmap extraction.")
            return

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    ds = LearningDataset(
        index_path=config.get("index_path", "learning_data/index.jsonl"),
        features_dir=config.get("feature_cache_path", "learning_data/features"),
        split="id_val",
        return_masks=True
    )

    out_heat = Path("artifacts/learning_stage1/heatmaps")
    out_heat.mkdir(parents=True, exist_ok=True)

    rgb_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224)
    ])
    mask_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(224)
    ])

    saved = 0
    for i in range(len(ds)):
        item = ds[i]
        if saved >= 4: break
        
        t = item["text_feat"].unsqueeze(0)
        d = item["demo_global"].unsqueeze(0)
        q = item["query_patch"].unsqueeze(0)
        
        with torch.no_grad():
            _, _, z_R = model(t, d, q)
            heat_logits = model.heatmap_decoder(z_R)
            heat_preds = torch.sigmoid(heat_logits)
        
        h_img = heat_preds[0].numpy()
        h_img = np.clip(h_img, 0, 1)
        
        heatmap = (h_img * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        record = ds.records[i]
        rgb_path = record["query_rgb_path"]
        
        if os.path.exists(rgb_path):
            img_pil = Image.open(rgb_path).convert("RGB")
            img_cropped = rgb_transform(img_pil)
            orig = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR)
            
            overlay = cv2.addWeighted(orig, 0.5, heatmap_colored, 0.5, 0)
            
            label = record["label"]
            prefix = str(out_heat / f"sample_{i}_{label}")
            cv2.imwrite(f"{prefix}_heatmap.jpg", heatmap_colored)
            cv2.imwrite(f"{prefix}_overlay.jpg", overlay)
            cv2.imwrite(f"{prefix}_query.jpg", orig)
            
            mask_path = record["causal_mask_path"]
            if mask_path and os.path.exists(mask_path):
                m_pil = Image.open(mask_path).convert("L")
                m_cropped = mask_transform(m_pil)
                mask = np.array(m_cropped)
                cv2.imwrite(f"{prefix}_gt_mask.jpg", mask)
                
            saved += 1

if __name__ == "__main__":
    generate_heatmaps()
