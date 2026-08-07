import torch
from src.learning.models.query_only import QueryOnlyBaseline
from src.learning.models.pooled_multimodal import PooledMultimodalBaseline
from src.learning.models.relational_model import DemoLanguageConditionedRelationalModel
from src.learning.models.heatmap_decoder import CausalHeatmapDecoder

def test_query_only():
    model = QueryOnlyBaseline()
    q = torch.randn(2, 768)
    out = model(q)
    assert out.shape == (2, 1)
    
def test_pooled_multimodal():
    model = PooledMultimodalBaseline()
    t = torch.randn(2, 384)
    d = torch.randn(2, 4, 768)
    q = torch.randn(2, 768)
    logits, s = model(t, d, q)
    assert logits.shape == (2, 1)
    assert s.shape == (2, 1)

def test_relational_model():
    model = DemoLanguageConditionedRelationalModel(num_demo_frames=4)
    t = torch.randn(2, 384)
    d = torch.randn(2, 4, 768)
    q_patch = torch.randn(2, 256, 768)
    logits, s, z_R = model(t, d, q_patch)
    assert logits.shape == (2, 1)
    assert s.shape == (2, 1)
    assert z_R.shape == (2, 256, 256) # (B, N, latent_dim)

def test_heatmap_decoder():
    decoder = CausalHeatmapDecoder(latent_dim=256, output_size=224, patch_grid_size=16)
    z_R = torch.randn(2, 256, 256)
    out = decoder(z_R)
    assert out.shape == (2, 224, 224)
