import torch
import json
import numpy as np
import yaml
import os
from pathlib import Path
from PIL import Image
import cv2
import torchvision.transforms as T
import subprocess

from src.learning.dataset import LearningDataset
from src.learning.utils import resolve_checkpoint
from scripts.train_model import get_model

import argparse

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def generate_heatmaps():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="learning_outputs/relational_heatmap_seed42")
    parser.add_argument("--index", default="data/manifests/context_challenge_manifest.jsonl")
    parser.add_argument("--out_dir", default="artifacts/learning_stage1/heatmaps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.experiment_dir)
    with open(out_dir / "resolved_config.yaml") as f:
        config = yaml.safe_load(f)

    model = get_model(config)

    if config.get("heatmap", False):
        from src.learning.models.heatmap_decoder import CausalHeatmapDecoder
        model.heatmap_decoder = CausalHeatmapDecoder(latent_dim=config.get("latent_dim", 256))

    ckpt_path = resolve_checkpoint(out_dir)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    out_heat = Path(args.out_dir)
    out_heat.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tested_code_commit": get_git_commit(),
        "model_seed": args.seed,
        "checkpoint_path": str(ckpt_path),
        "samples": []
    }

    rgb_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224)
    ])
    mask_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(224)
    ])

    splits_to_evaluate = ["id_val", "compositional", "context_challenge"]

    for split in splits_to_evaluate:
        try:
            index_p = args.index if split == "context_challenge" else config.get("index_path", "learning_data/index.jsonl")
            feat_p = "data/features_context" if split == "context_challenge" else config.get("feature_cache_path", "learning_data/features")
            ds = LearningDataset(
                index_path=index_p,
                features_dir=feat_p,
                split=split,
                return_masks=True,
                seed=42,
                split_seed=42
            )
        except Exception as e:
            print(f"Skipping split {split}: {e}")
            continue

        saved_task1 = 0
        saved_task2 = 0
        for i in range(len(ds)):
            record = ds.records[i]
            task_id = record["task_id"]
            if task_id == "task_1" and saved_task1 >= 4: continue
            if task_id == "task_2" and saved_task2 >= 4: continue

            item = ds[i]

            t = item["text_feat"].unsqueeze(0)
            d = item["demo_global"].unsqueeze(0)
            q = item["query_patch"].unsqueeze(0)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cpu"):
                    import inspect
                    sig = inspect.signature(model.forward)
                    kwargs = {}
                    if "text_feat" in sig.parameters: kwargs["text_feat"] = t
                    if "demo_global" in sig.parameters: kwargs["demo_global"] = d
                    if "query_patch" in sig.parameters: kwargs["query_patch"] = q
                    if "x" in sig.parameters: kwargs["x"] = item["query_global"].unsqueeze(0)

                    out = model(**kwargs)
                    if isinstance(out, tuple) and len(out) >= 3:
                        logits, s, z_R = out[:3]
                        heat_logits = model.heatmap_decoder(z_R)
                        heat_preds = torch.sigmoid(heat_logits)
                    else:
                        continue # No heatmap

            h_img = heat_preds[0].numpy()
            h_img = np.clip(h_img, 0, 1)

            heatmap = (h_img * 255).astype(np.uint8)
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            rgb_path = record["query_rgb_path"]

            if os.path.exists(rgb_path):
                img_pil = Image.open(rgb_path).convert("RGB")
                img_cropped = rgb_transform(img_pil)
                orig = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR)

                overlay = cv2.addWeighted(orig, 0.5, heatmap_colored, 0.5, 0)

                label = record["label"]
                prefix = str(out_heat / f"{split}_sample_{i}_{task_id}_{label}")
                cv2.imwrite(f"{prefix}_heatmap.jpg", heatmap_colored)
                cv2.imwrite(f"{prefix}_overlay.jpg", overlay)
                cv2.imwrite(f"{prefix}_query.jpg", orig)

                mask_path = record.get("causal_mask_path")
                if mask_path and os.path.exists(mask_path):
                    m_pil = Image.open(mask_path).convert("L")
                    m_cropped = mask_transform(m_pil)
                    mask = np.array(m_cropped)
                    cv2.imwrite(f"{prefix}_gt_mask.jpg", mask)

                metadata["samples"].append({
                    "sample_id": record["sample_id"],
                    "task": task_id,
                    "split": split,
                    "label": label,
                    "prefix": f"{split}_sample_{i}_{task_id}_{label}"
                })
                if task_id == "task_1": saved_task1 += 1
                if task_id == "task_2": saved_task2 += 1

    with open(out_heat / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    generate_heatmaps()
