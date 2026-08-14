"""Comprehensive Architecture and Information-Flow Tests for Intervention Models."""

from pathlib import Path
import pytest
import torch

from src.learning.intervention_feature_spec import InterventionFeatureSpec
from src.learning.models.intervention_outputs import InterventionModelOutput
from src.learning.models.intervention_query_only import InterventionQueryOnlyBaseline
from src.learning.models.simple_intervention import SimpleInterventionBaseline
from src.learning.models.intervention_relational import InterventionRelationalModel
from src.learning.models.intervention_model_registry import (
    get_intervention_model,
    INTERVENTION_MODEL_REGISTRY,
)
from src.learning.intervention_dataset import (
    InterventionLearningDataset,
    InterventionGroupBatchSampler,
    collate_intervention_group,
)


def get_mock_model_inputs(batch_size: int = 4) -> dict:
    """Helper creating a valid batch of model_inputs tensors."""
    spec = InterventionFeatureSpec()
    return {
        "text_feat": torch.randn(batch_size, spec.text_dim),
        "scene_global": torch.randn(batch_size, spec.scene_global_dim),
        "scene_patch": torch.randn(batch_size, spec.scene_patch_count, spec.scene_patch_dim),
        "candidate_visual": torch.randn(batch_size, spec.candidate_visual_dim),
        "current_geometry": torch.randn(batch_size, spec.current_geom_dim),
        "destination_geometry": torch.randn(batch_size, spec.dest_geom_dim),
        "operator_idx": torch.tensor([1, 1, 0, 1], dtype=torch.long)[:batch_size],
    }


def test_b0_forward_shape():
    """Test B0 baseline forward pass shapes and outputs."""
    model = InterventionQueryOnlyBaseline()
    inputs = get_mock_model_inputs(batch_size=4)
    out = model(inputs)

    assert isinstance(out, InterventionModelOutput)
    assert out.post_logit.shape == (4,)
    assert out.ranking_score.shape == (4,)
    assert out.relational_embedding is None
    assert out.relational_tokens is None


def test_b0_within_scene_candidate_invariance():
    """Test that B0 logits are bitwise identical across candidates from the same scene."""
    model = InterventionQueryOnlyBaseline()
    model.eval()

    inputs = get_mock_model_inputs(batch_size=4)
    # Give all rows identical scene_global but different candidate inputs
    inputs["scene_global"] = inputs["scene_global"][0:1].expand(4, -1).clone()

    with torch.no_grad():
        out = model(inputs)

    # All 4 logits must be identical
    first_logit = out.post_logit[0]
    for i in range(1, 4):
        torch.testing.assert_close(out.post_logit[i], first_logit)


def test_b0b_forward_shape_and_scene_patch_independence():
    """Test B0b baseline forward shapes and verify sentinel scene_patch independence."""
    model = SimpleInterventionBaseline()
    inputs = get_mock_model_inputs(batch_size=4)

    # Supply sentinel/NaN scene_patch
    inputs["scene_patch"] = torch.full_like(inputs["scene_patch"], float("nan"))

    out = model(inputs)

    assert isinstance(out, InterventionModelOutput)
    assert out.post_logit.shape == (4,)
    assert out.ranking_score.shape == (4,)
    assert not torch.isnan(out.post_logit).any()
    assert out.relational_embedding is None
    assert out.relational_tokens is None


def test_b0b_candidate_gradient_connectivity():
    """Test that candidate visual, current geom, dest geom, and operator embed receive gradients in B0b."""
    model = SimpleInterventionBaseline()
    model.train()

    inputs = get_mock_model_inputs(batch_size=4)
    inputs["candidate_visual"].requires_grad_(True)
    inputs["current_geometry"].requires_grad_(True)
    inputs["destination_geometry"].requires_grad_(True)

    out = model(inputs)
    loss = out.post_logit.sum()
    loss.backward()

    assert inputs["candidate_visual"].grad is not None
    assert torch.count_nonzero(inputs["candidate_visual"].grad).item() > 0
    assert inputs["current_geometry"].grad is not None
    assert torch.count_nonzero(inputs["current_geometry"].grad).item() > 0
    assert inputs["destination_geometry"].grad is not None
    assert torch.count_nonzero(inputs["destination_geometry"].grad).item() > 0
    assert model.operator_embed.weight.grad is not None
    assert torch.count_nonzero(model.operator_embed.weight.grad).item() > 0


def test_v2_forward_shapes():
    """Test V2 InterventionRelationalModel forward output shapes."""
    model = InterventionRelationalModel()
    inputs = get_mock_model_inputs(batch_size=4)
    out = model(inputs)

    assert isinstance(out, InterventionModelOutput)
    assert out.post_logit.shape == (4,)
    assert out.ranking_score.shape == (4,)
    assert out.relational_embedding is not None
    assert out.relational_embedding.shape == (4, 256)
    assert out.relational_tokens is not None
    assert out.relational_tokens.shape == (4, 256, 256)


def test_v2_none_forward():
    """Test V2 forward pass on neutral NONE record (zero tensors, operator_idx=0)."""
    model = InterventionRelationalModel()
    inputs = get_mock_model_inputs(batch_size=2)
    spec = InterventionFeatureSpec()

    # Row 0: NONE operator
    inputs["candidate_visual"][0] = 0.0
    inputs["current_geometry"][0] = 0.0
    inputs["destination_geometry"][0] = 0.0
    inputs["operator_idx"][0] = 0

    out = model(inputs)

    assert not torch.isnan(out.post_logit).any()
    assert not torch.isinf(out.post_logit).any()
    assert not torch.isnan(out.relational_embedding).any()


def test_v2_all_declared_gradient_paths_active():
    """Test structural gradient connectivity through all declared V2 input pathways and trainable modules."""
    model = InterventionRelationalModel()
    model.train()

    inputs = get_mock_model_inputs(batch_size=4)
    inputs["text_feat"].requires_grad_(True)
    inputs["scene_global"].requires_grad_(True)
    inputs["scene_patch"].requires_grad_(True)
    inputs["candidate_visual"].requires_grad_(True)
    inputs["current_geometry"].requires_grad_(True)
    inputs["destination_geometry"].requires_grad_(True)

    out = model(inputs)
    loss = out.post_logit.sum()
    loss.backward()

    assert inputs["text_feat"].grad is not None
    assert torch.count_nonzero(inputs["text_feat"].grad).item() > 0

    assert inputs["scene_global"].grad is not None
    assert torch.count_nonzero(inputs["scene_global"].grad).item() > 0

    assert inputs["scene_patch"].grad is not None
    assert torch.count_nonzero(inputs["scene_patch"].grad).item() > 0

    assert inputs["candidate_visual"].grad is not None
    assert torch.count_nonzero(inputs["candidate_visual"].grad).item() > 0

    assert inputs["current_geometry"].grad is not None
    assert torch.count_nonzero(inputs["current_geometry"].grad).item() > 0

    assert inputs["destination_geometry"].grad is not None
    assert torch.count_nonzero(inputs["destination_geometry"].grad).item() > 0

    assert model.proj_text.weight.grad is not None
    assert model.proj_scene_global.weight.grad is not None
    assert model.proj_patch.weight.grad is not None
    assert model.proj_intervention.weight.grad is not None
    assert model.operator_embed.weight.grad is not None


def test_v2_matched_destination_conditioning():
    """Test that synthetic matched candidates with differing destination geometry produce distinct intervention latents."""
    model = InterventionRelationalModel()
    model.eval()

    c_vis = torch.randn(1, 768)
    curr_g = torch.tensor([[0.0, 0.0, 0.05]])
    dest_g1 = torch.tensor([[-0.25, 0.0, -0.12]])
    dest_g2 = torch.tensor([[0.10, 0.15, 0.05]])
    op_idx = torch.tensor([1])

    with torch.no_grad():
        h1, z1 = model.encode_intervention(c_vis, curr_g, dest_g1, op_idx)
        h2, z2 = model.encode_intervention(c_vis, curr_g, dest_g2, op_idx)

    assert not torch.equal(h1, h2)
    assert not torch.equal(z1, z2)


def test_batch_permutation_equivariance():
    """Test batch permutation equivariance across all three architectures in eval mode."""
    b0 = InterventionQueryOnlyBaseline().eval()
    b0b = SimpleInterventionBaseline().eval()
    v2 = InterventionRelationalModel().eval()

    inputs = get_mock_model_inputs(batch_size=4)
    perm = [2, 0, 3, 1]

    inputs_perm = {k: v[perm] for k, v in inputs.items()}

    with torch.no_grad():
        out_b0 = b0(inputs)
        out_b0_perm = b0(inputs_perm)

        out_b0b = b0b(inputs)
        out_b0b_perm = b0b(inputs_perm)

        out_v2 = v2(inputs)
        out_v2_perm = v2(inputs_perm)

    torch.testing.assert_close(out_b0.post_logit[perm], out_b0_perm.post_logit)
    torch.testing.assert_close(out_b0b.post_logit[perm], out_b0b_perm.post_logit)
    torch.testing.assert_close(out_v2.post_logit[perm], out_v2_perm.post_logit)


def test_model_output_score_consistency():
    """Test that ranking_score is identical to post_logit across all models."""
    models = [
        InterventionQueryOnlyBaseline(),
        SimpleInterventionBaseline(),
        InterventionRelationalModel(),
    ]
    inputs = get_mock_model_inputs(batch_size=4)

    for m in models:
        m.eval()
        with torch.no_grad():
            out = m(inputs)
        torch.testing.assert_close(out.ranking_score, out.post_logit)


def test_registry_dispatch_and_rejection():
    """Test registry dispatch for valid model names and explicit loud rejection for B3 / unknown names."""
    m_b0 = get_intervention_model("query_only")
    assert isinstance(m_b0, InterventionQueryOnlyBaseline)

    m_b0b = get_intervention_model("simple_intervention")
    assert isinstance(m_b0b, SimpleInterventionBaseline)

    m_v2 = get_intervention_model("intervention_relational")
    assert isinstance(m_v2, InterventionRelationalModel)

    with pytest.raises(ValueError, match="not a separate architecture"):
        get_intervention_model("relational_feasibility_only")

    with pytest.raises(ValueError, match="Unknown intervention model architecture"):
        get_intervention_model("unknown_future_model")


def test_parameter_counts_reported():
    """Test parameter count calculation and logging for B0, B0b, and V2."""
    b0 = get_intervention_model("query_only")
    b0b = get_intervention_model("simple_intervention")
    v2 = get_intervention_model("intervention_relational")

    p_b0 = sum(p.numel() for p in b0.parameters() if p.requires_grad)
    p_b0b = sum(p.numel() for p in b0b.parameters() if p.requires_grad)
    p_v2 = sum(p.numel() for p in v2.parameters() if p.requires_grad)

    assert p_b0 > 0
    assert p_b0b > p_b0
    assert p_v2 > p_b0b

    print(f"\n[Packet 4 Parameter Report] B0: {p_b0:,} | B0b: {p_b0b:,} | V2: {p_v2:,}")


def test_model_constructors_reject_unknown_kwargs():
    """Verify that model constructors and registry dispatch loudly reject unknown kwargs."""
    with pytest.raises(TypeError):
        InterventionQueryOnlyBaseline(unknown_param=123)

    with pytest.raises(TypeError):
        SimpleInterventionBaseline(unknown_param=123)

    with pytest.raises(TypeError):
        InterventionRelationalModel(unknown_param=123)

    with pytest.raises(TypeError):
        get_intervention_model("intervention_relational", num_cross_layer=4)

    with pytest.raises(TypeError):
        get_intervention_model("query_only", invalid_arg=99)


def test_b0_requires_dict_input():
    """Verify that B0 forward rejects non-dict inputs."""
    model = InterventionQueryOnlyBaseline()
    raw_tensor = torch.randn(4, 768)
    with pytest.raises(TypeError, match="requires a dict input"):
        model(raw_tensor)


def test_real_smoke_forward_pass_all_architectures():
    """Integration test processing all 22 real smoke dataset records through B0, B0b, and V2."""
    manifest_path = Path("data/intervention_smoke/manifest.jsonl")
    features_dir = Path("data/intervention_smoke/features")

    if not manifest_path.exists() or not features_dir.exists():
        pytest.skip("Smoke dataset or features directory not present.")

    ds = InterventionLearningDataset(manifest_path=manifest_path, features_dir=features_dir, split="all")
    assert len(ds) == 22

    sampler = InterventionGroupBatchSampler(dataset=ds, scenes_per_batch=1, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 6

    b0 = get_intervention_model("query_only").eval()
    b0b = get_intervention_model("simple_intervention").eval()
    v2 = get_intervention_model("intervention_relational").eval()

    total_records_processed = 0

    with torch.no_grad():
        for batch_indices in batches:
            items = [ds[i] for i in batch_indices]
            collated = collate_intervention_group(items)
            m_inputs = collated["model_inputs"]

            out_b0 = b0(m_inputs)
            out_b0b = b0b(m_inputs)
            out_v2 = v2(m_inputs)

            b_sz = collated["batch_size"]
            total_records_processed += b_sz

            assert out_b0.post_logit.shape == (b_sz,)
            assert out_b0b.post_logit.shape == (b_sz,)
            assert out_v2.post_logit.shape == (b_sz,)

            assert torch.isfinite(out_b0.post_logit).all()
            assert torch.isfinite(out_b0.ranking_score).all()
            assert torch.isfinite(out_b0b.post_logit).all()
            assert torch.isfinite(out_b0b.ranking_score).all()
            assert torch.isfinite(out_v2.post_logit).all()
            assert torch.isfinite(out_v2.ranking_score).all()
            assert torch.isfinite(out_v2.relational_embedding).all()
            assert torch.isfinite(out_v2.relational_tokens).all()

    assert total_records_processed == 22
