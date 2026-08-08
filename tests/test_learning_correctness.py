import pytest
import torch
import json
from pathlib import Path
from src.learning.dataset import LearningDataset, PairBatchSampler

def test_split_seed_isolation(tmp_path):
    # Dummy manifest
    manifest_path = tmp_path / "index.jsonl"
    with open(manifest_path, "w") as f:
        for i in range(20):
            f.write(json.dumps({
                "pair_id": f"pair_{i}",
                "sample_id": f"samp_{i}a",
                "split": "id",
                "task_id": "task_1",
                "demonstration_id": "demo_t1",
                "instruction": "Open the box."
            }) + "\n")
            f.write(json.dumps({
                "pair_id": f"pair_{i}",
                "sample_id": f"samp_{i}b",
                "split": "id",
                "task_id": "task_1",
                "demonstration_id": "demo_t1",
                "instruction": "Open the box."
            }) + "\n")
            
    # Dummy text features
    text_feat_path = tmp_path / "text_features.pt"
    torch.save({"Open the box.": torch.zeros(384)}, text_feat_path)
            
    ds1 = LearningDataset(manifest_path, tmp_path, split="id_train", seed=11, split_seed=42)
    ds2 = LearningDataset(manifest_path, tmp_path, split="id_train", seed=101, split_seed=42)
    
    assert len(ds1) > 0
    # The pairs should be identical despite different seeds
    pairs1 = [r["pair_id"] for r in ds1.records]
    pairs2 = [r["pair_id"] for r in ds2.records]
    assert pairs1 == pairs2, "Train split should be identical regardless of model seed"
    
def test_pair_sampler_determinism(tmp_path):
    # Just mock a dataset with a records attribute
    class MockDataset:
        def __init__(self):
            self.records = [{"pair_id": f"pair_{i//2}"} for i in range(20)]
            
    ds = MockDataset()
    s1 = PairBatchSampler(ds, batch_size=4, seed=42)
    s2 = PairBatchSampler(ds, batch_size=4, seed=42)
    s3 = PairBatchSampler(ds, batch_size=4, seed=101)
    
    batches1 = list(s1)
    batches2 = list(s2)
    batches3 = list(s3)
    
    assert batches1 == batches2, "Samplers with same seed must yield same batches"
    assert batches1 != batches3, "Samplers with different seeds must yield different batches"
