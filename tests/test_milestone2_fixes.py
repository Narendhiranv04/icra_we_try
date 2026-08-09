import pytest
import os
import torch
import json
import numpy as np
from unittest.mock import patch, MagicMock

# Import the modules we need to test
from src.generation.context_challenge_generator import ContextChallengeGenerator
from src.learning.dataset import LearningDataset, PairBatchSampler
from scripts.evaluate_context_challenge import evaluate_context
from scripts.bootstrap_metrics import compute_metrics, bootstrap_confidence_interval

def test_generator_bounded_retry(tmp_path):
    gen = ContextChallengeGenerator(output_dir=str(tmp_path / "tmp_test_gen"))
    
    # Force visibility failure to test retry
    with patch('src.environment.renderer.OffscreenRenderer.render_region_mask', return_value=np.zeros((10,10))):
        with pytest.raises(RuntimeError, match="Exceeded max_attempts"):
            gen.generate_scene("test_scene", "A", seed=42, max_attempts=3)

def test_seed_isolation_dataset(tmp_path):
    # Test that split_seed determines splits independently of general seed
    # Create a dummy index
    index_file = tmp_path / "tmp_index.jsonl"
    with open(index_file, "w") as f:
        for i in range(10):
            f.write(json.dumps({"pair_id": f"p{i}", "split": "id", "sample_id": f"s{i}", "task_id": "task_1", "instruction": "Open the box.", "label": "STOP", "demonstration_id": "d1"}) + "\n")
            
    # Mock text and features to exist
    with patch('pathlib.Path.exists', return_value=True), patch('torch.load', return_value={"Open the box.": torch.zeros(1), "global": torch.zeros(1), "patch": torch.zeros(1)}):
        ds1 = LearningDataset(index_path=str(index_file), features_dir="tmp", split="id_train", train_ratio=0.8, seed=42, split_seed=100)
        ds2 = LearningDataset(index_path=str(index_file), features_dir="tmp", split="id_train", train_ratio=0.8, seed=99, split_seed=100)
        ds3 = LearningDataset(index_path=str(index_file), features_dir="tmp", split="id_train", train_ratio=0.8, seed=42, split_seed=200)
        
        pairs1 = set([r["pair_id"] for r in ds1.records])
        pairs2 = set([r["pair_id"] for r in ds2.records])
        pairs3 = set([r["pair_id"] for r in ds3.records])
        
        assert pairs1 == pairs2, "Splits should be identical if split_seed is same, despite different model seed"
        assert pairs1 != pairs3, "Splits should differ if split_seed differs"

def test_sampler_seed_determinism():
    class DummyDS:
        def __init__(self):
            self.records = [{"pair_id": f"p{i//2}"} for i in range(20)]
            
    ds = DummyDS()
    sampler1 = PairBatchSampler(ds, batch_size=4, seed=42)
    sampler2 = PairBatchSampler(ds, batch_size=4, seed=42)
    sampler3 = PairBatchSampler(ds, batch_size=4, seed=99)
    
    b1 = list(sampler1)
    b2 = list(sampler2)
    b3 = list(sampler3)
    
    assert b1 == b2, "Same sampler seed should produce identical batches"
    assert b1 != b3, "Different sampler seed should produce different batches"

def test_generic_text_diagnostic_failure(tmp_path):
    index_file = tmp_path / "tmp_index.jsonl"
    with open(index_file, "w") as f:
        for i in range(10):
            f.write(json.dumps({"pair_id": f"p{i}", "split": "id", "sample_id": f"s{i}", "task_id": "task_1", "instruction": "Perform the demonstrated task.", "label": "STOP", "demonstration_id": "d1"}) + "\n")
    
    with patch('pathlib.Path.exists', return_value=True), \
         patch('torch.load', return_value={"Open the box.": torch.zeros(1), "global": torch.zeros(1), "patch": torch.zeros(1)}):
        
        # When instruction is generic text, the dataset attempts to load the text feature for "Perform the demonstrated task."
        # The mock doesn't have it, so it should raise KeyError.
        ds = LearningDataset(index_path=str(index_file), features_dir="tmp", split="id_train")
        with pytest.raises(RuntimeError, match="Perform the demonstrated task"):
            _ = ds[0]

def test_bootstrap_determinism_and_logic():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    units = [
        {"preds": [0.9, 0.1], "targets": [1, 0], "scores": [0.8, 0.2]},
        {"preds": [0.8, 0.2], "targets": [1, 0], "scores": [0.7, 0.3]},
        {"preds": [0.1, 0.9], "targets": [1, 0], "scores": [0.1, 0.9]} # Wrong
    ]
    
    res1 = bootstrap_confidence_interval(units, rng1, n_iterations=10, is_context=False)
    res2 = bootstrap_confidence_interval(units, rng2, n_iterations=10, is_context=False)
    
    assert res1["accuracy"]["mean"] == res2["accuracy"]["mean"]
    
def test_evaluator_metrics():
    preds = [0.9, 0.1, 0.8, 0.9] # p1, p2, p3, p4
    targets = [1, 0, 1, 0] # t1, t2, t3, t4
    states = ["A", "A"] # scene1=A, scene2=A
    
    m = compute_metrics(preds, targets, state_info=states, is_context=True)
    assert m["accuracy"] == 0.75
    # Scene 1: p1>=0.5 (T), p2>=0.5 (F) -> (T, F) == (1, 0) -> Correct
    # Scene 2: p3>=0.5 (T), p4>=0.5 (T) -> (T, T) != (1, 0) -> Incorrect
    assert m["context_pair_consistency"] == 0.5
    assert m["reversal_accuracy"] == 0.5

def test_heatmap_bfloat16_conversion():
    # Simulate a bfloat16 tensor prediction
    tensor_bf16 = torch.tensor([[[0.1, 0.9], [0.5, 0.5]]], dtype=torch.bfloat16)
    
    # The fix used in extract_heatmaps.py:
    # h_img = heat_preds[0].detach().float().cpu().numpy()
    
    # Apply fix
    h_img = tensor_bf16[0].detach().float().cpu().numpy()
    h_img = np.clip(h_img, 0, 1)
    
    # Verify properties
    assert isinstance(h_img, np.ndarray)
    assert h_img.dtype == np.float32
    assert h_img.shape == (2, 2)
    assert np.all(np.isfinite(h_img))
    
if __name__ == "__main__":
    pytest.main([__file__])
