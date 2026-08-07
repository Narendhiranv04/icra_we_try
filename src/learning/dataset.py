import json
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from PIL import Image
import numpy as np

class LearningDataset(Dataset):
    def __init__(self, index_path, features_dir, split="id", return_masks=False):
        self.features_dir = Path(features_dir)
        self.return_masks = return_masks
        
        with open(index_path, "r") as f:
            all_records = [json.loads(line) for line in f if line.strip()]
            
        # Filter by split
        if split:
            self.records = [r for r in all_records if r["split"] == split]
        else:
            self.records = all_records
            
        self.text_features = torch.load(self.features_dir / "text_features.pt", weights_only=True)
        
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        
        # Labels: y = 0 -> PROCEED, y = 1 -> STOP
        label = 1 if rec["label"].upper() == "STOP" else 0
        
        sample_id = rec["sample_id"]
        pair_id = rec["pair_id"]
        task_id = rec["task_id"]
        instruction = rec["instruction"]
        
        text_feat = self.text_features.get(instruction, torch.zeros(384))
        
        query_feat_path = self.features_dir / f"query_{sample_id}_features.pt"
        if query_feat_path.exists():
            q_dict = torch.load(query_feat_path, weights_only=True)
            q_global = q_dict["global"]
            q_patch = q_dict["patch"]
        else:
            q_global = torch.zeros(768)
            q_patch = torch.zeros(256, 768) # 16x16=256
            
        demo_id = rec["demonstration_id"]
        demo_feat_path = self.features_dir / f"demo_{demo_id}_features.pt"
        if demo_feat_path.exists():
            d_dict = torch.load(demo_feat_path, weights_only=True)
            d_global = d_dict["global"]
            d_patch = d_dict["patch"]
        else:
            d_global = torch.zeros(4, 768)
            d_patch = torch.zeros(4, 256, 768)
            
        mask = torch.zeros(224, 224)
        if self.return_masks:
            if label == 1 and rec.get("causal_mask_path"):
                mask_path = rec["causal_mask_path"]
                if Path(mask_path).exists():
                    mask_img = Image.open(mask_path).convert("L")
                    mask_img = mask_img.resize((224, 224), Image.Resampling.NEAREST)
                    mask = torch.from_numpy(np.array(mask_img)) > 127
                    mask = mask.float()
        
        return {
            "sample_id": sample_id,
            "pair_id": pair_id,
            "task_id": task_id,
            "instruction": instruction,
            "label": torch.tensor(label, dtype=torch.float32),
            "text_feat": text_feat,
            "query_global": q_global,
            "query_patch": q_patch,
            "demo_global": d_global,
            "demo_patch": d_patch,
            "mask": mask
        }

class PairBatchSampler(Sampler):
    """
    Ensures that STOP and PROCEED pairs are in the same batch.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size - (batch_size % 2) # Ensure even
        
        self.pair_to_indices = {}
        for idx, rec in enumerate(dataset.records):
            pid = rec["pair_id"]
            if pid not in self.pair_to_indices:
                self.pair_to_indices[pid] = []
            self.pair_to_indices[pid].append(idx)
            
        # Only keep full pairs
        self.pairs = [v for v in self.pair_to_indices.values() if len(v) == 2]

    def __iter__(self):
        import random
        random.shuffle(self.pairs)
        batch = []
        for pair in self.pairs:
            batch.extend(pair)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.pairs) * 2 + self.batch_size - 1) // self.batch_size
