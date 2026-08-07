import pytest
import torch
import os
import json
from pathlib import Path
from src.learning.dataset import LearningDataset
from src.learning.models.relational_model import DemoLanguageConditionedRelationalModel
from src.learning.losses import LearningLoss
from src.learning.metrics import compute_metrics
import hashlib
from src.learning.metrics import compute_metrics

def test_deterministic_demo_assignment():
    pair_id = "test_pair_123"
    demo_pool = ["demo1.mp4", "demo2.mp4", "demo3.mp4"]
    idx1 = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % len(demo_pool)
    idx2 = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % len(demo_pool)
    assert idx1 == idx2, "Demo assignment must be deterministic across calls"

def test_dataset_no_silent_fallback(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    index_path = tmp_path / "index.jsonl"
    
    # Missing text feature
    record = {
        "sample_id": "test_sample_1",
        "pair_id": "pair_1",
        "task_id": "task_1",
        "instruction": "Open the box.",
        "demonstration_id": "demo_1",
        "demonstration_video_path": "dummy.mp4",
        "label": "STOP",
        "split": "id",
        "query_rgb_path": "dummy.jpg",
    }
    with open(index_path, "w") as f:
        f.write(json.dumps(record) + "\n")
        
    torch.save({"dummy": torch.zeros(384)}, features_dir / "text_features.pt")
    
    with pytest.raises(RuntimeError, match="Missing text feature"):
        ds = LearningDataset(index_path, features_dir, split="id", return_masks=False)
        _ = ds[0]
        
def test_ranking_loss_prefers_proceed():
    loss_fn = LearningLoss(margin=0.2, lambda_rank=1.0)
    
    # s_pos (proceed) > s_neg (stop)
    logits = torch.zeros(2, 1)
    targets = torch.tensor([0.0, 1.0]) # t=0 is proceed, t=1 is stop
    s = torch.tensor([0.9, 0.1])
    pair_ids = ["p1", "p1"]
    
    loss, metrics = loss_fn(logits, targets, s=s, pair_ids=pair_ids)
    assert metrics["l_rank"] == 0.0 # margin is 0.2, difference is 0.8
    
    # s_pos (proceed) < s_neg (stop) -> loss should be > 0
    s_bad = torch.tensor([0.1, 0.9])
    loss_bad, metrics_bad = loss_fn(logits, targets, s=s_bad, pair_ids=pair_ids)
    assert metrics_bad["l_rank"] > 0.0

def test_pla_correctness():
    # Correct pair
    preds = [0.1, 0.9] # preds for stop vs proceed doesn't matter for PLA directly
    targets = [0, 1]   # proceed, stop
    scores = [0.8, 0.2] # s(proceed) > s(stop)
    pair_ids = ["p1", "p1"]
    
    m = compute_metrics(preds, targets, scores=scores, pair_ids=pair_ids)
    assert m["pla"] == 1.0
    
    # Incorrect pair
    scores_bad = [0.2, 0.8]
    m_bad = compute_metrics(preds, targets, scores=scores_bad, pair_ids=pair_ids)
    assert m_bad["pla"] == 0.0

def test_relational_forward_shapes():
    model = DemoLanguageConditionedRelationalModel(text_dim=384, vision_dim=768, latent_dim=256, num_demo_frames=4)
    
    text_feat = torch.randn(2, 384)
    demo_global = torch.randn(2, 4, 768)
    query_patch = torch.randn(2, 256, 768)
    
    logits, s, Z_R = model(text_feat, demo_global, query_patch)
    
    assert logits.shape == (2, 1)
    assert s.shape == (2,)
    assert Z_R.shape == (2, 256, 256) # (B, N, latent_dim)

def test_heatmap_metrics():
    # Mock targets and preds
    heat_preds = [
        torch.ones(224, 224), # proceed
        torch.ones(224, 224)  # stop
    ]
    heat_preds[0][0, 0] = 0.0 # almost all 1s
    heat_targets = [
        torch.zeros(224, 224), # proceed
        torch.ones(224, 224)   # stop
    ]
    targets = [0, 1]
    
    m = compute_metrics([0, 1], targets, heat_preds=heat_preds, heat_targets=heat_targets)
    assert m["stop_iou"] == 1.0
    assert m["stop_dice"] == 1.0
    assert m["proceed_fp_frac"] > 0.99
