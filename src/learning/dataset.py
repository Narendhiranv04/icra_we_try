import json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from PIL import Image
import numpy as np
import random

class LearningDataset(Dataset):
    def __init__(self, index_path, features_dir, split="id", return_masks=False, train_ratio=0.8, seed=42, split_seed=42, ablation_mode="none"):
        self.features_dir = Path(features_dir)
        self.return_masks = return_masks
        self.seed = seed
        self.ablation_mode = ablation_mode

        with open(index_path, "r") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        # Handle splitting for id
        if split in ["id_train", "id_val"]:
            # Get all unique pairs in "id"
            id_records = [r for r in all_records if r["split"] == "id"]
            pair_ids = sorted(list(set(r["pair_id"] for r in id_records)))

            # Deterministic split independent of model seed
            rng = random.Random(split_seed)
            rng.shuffle(pair_ids)
            split_idx = int(len(pair_ids) * train_ratio)

            train_pairs = set(pair_ids[:split_idx])
            val_pairs = set(pair_ids[split_idx:])

            if split == "id_train":
                self.records = [r for r in id_records if r["pair_id"] in train_pairs]
            else:
                self.records = [r for r in id_records if r["pair_id"] in val_pairs]
        elif split == "context_challenge":
            self.records = [r for r in all_records if r.get("split") == "context_challenge"]
        elif split:
            self.records = [r for r in all_records if r["split"] == split]
        else:
            self.records = all_records

        text_features_path = self.features_dir / "text_features.pt"
        if not text_features_path.exists():
            raise RuntimeError(f"Missing text features at {text_features_path}")
        self.text_features = torch.load(text_features_path, weights_only=True)

        self.demo_pool = {"task_1": [], "task_2": []}
        for r in all_records:
            if r["task_id"] in self.demo_pool and r["demonstration_id"] not in self.demo_pool[r["task_id"]]:
                self.demo_pool[r["task_id"]].append(r["demonstration_id"])

        self.mask_transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(224),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        label = 1 if rec["label"].upper() == "STOP" else 0
        sample_id = rec["sample_id"]
        pair_id = rec["pair_id"]
        task_id = rec["task_id"]
        instruction = rec["instruction"]

        if instruction not in self.text_features:
            raise RuntimeError(f"Missing text feature for instruction: '{instruction}' (sample_id={sample_id})")
        text_feat = self.text_features[instruction]

        if self.ablation_mode == "zero_text":
            text_feat = torch.zeros_like(text_feat)
        elif self.ablation_mode == "wrong_instruction":
            wrong_inst = "Place object1 in the target region." if task_id == "task_1" else "Open the box."
            if wrong_inst not in self.text_features:
                raise RuntimeError(f"Missing wrong instruction text feature: '{wrong_inst}'")
            text_feat = self.text_features[wrong_inst]
        elif self.ablation_mode == "heldout_paraphrase":
            para = "Uncover the box by opening its lid." if task_id == "task_1" else "Place the object into the indicated area."
            if para not in self.text_features:
                raise RuntimeError(f"Missing held-out paraphrase text feature: '{para}'")
            text_feat = self.text_features[para]

        query_feat_path = self.features_dir / f"query_{sample_id}_features.pt"
        if not query_feat_path.exists():
            raise RuntimeError(f"Missing query features at {query_feat_path} (sample_id={sample_id}, pair_id={pair_id})")

        q_dict = torch.load(query_feat_path, weights_only=True)
        q_global = q_dict["global"]
        q_patch = q_dict["patch"]

        demo_id = rec["demonstration_id"]
        if self.ablation_mode == "wrong_demo":
            other_task = "task_2" if task_id == "task_1" else "task_1"
            if self.demo_pool[other_task]:
                # Deterministically pick first demo of opposite task
                demo_id = self.demo_pool[other_task][0]
            else:
                raise RuntimeError(f"No demo found for opposite task '{other_task}'")

        demo_feat_path = self.features_dir / f"demo_{demo_id}_features.pt"
        if not demo_feat_path.exists():
            raise RuntimeError(f"Missing demo features at {demo_feat_path} (sample_id={sample_id}, pair_id={pair_id})")

        d_dict = torch.load(demo_feat_path, weights_only=True)
        d_global = d_dict["global"]
        d_patch = d_dict["patch"]

        if self.ablation_mode == "zero_demo":
            d_global = torch.zeros_like(d_global)
            d_patch = torch.zeros_like(d_patch)

        mask = torch.zeros(224, 224)
        if self.return_masks:
            if label == 1 and rec.get("causal_mask_path"):
                mask_path = rec["causal_mask_path"]
                if Path(mask_path).exists():
                    mask_img = Image.open(mask_path).convert("L")
                    # Same deterministic spatial transform as RGB, but with NEAREST
                    mask_tensor = self.mask_transform(mask_img).squeeze(0) # (224, 224)
                    mask = (mask_tensor > 0.5).float()
                else:
                    raise RuntimeError(f"Missing mask at {mask_path} for STOP sample (sample_id={sample_id})")

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
    def __init__(self, dataset, batch_size, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size - (batch_size % 2)
        self.seed = seed

        self.pair_to_indices = {}
        for idx, rec in enumerate(dataset.records):
            pid = rec["pair_id"]
            if pid not in self.pair_to_indices:
                self.pair_to_indices[pid] = []
            self.pair_to_indices[pid].append(idx)

        self.pairs = [v for v in self.pair_to_indices.values() if len(v) == 2]

    def __iter__(self):
        g = torch.Generator(device='cpu')
        g.manual_seed(self.seed)

        pairs_copy = list(self.pairs)
        indices = torch.randperm(len(pairs_copy), generator=g).tolist()
        pairs_copy = [pairs_copy[i] for i in indices]

        batch = []
        for pair in pairs_copy:
            batch.extend(pair)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.pairs) * 2 + self.batch_size - 1) // max(1, self.batch_size)
