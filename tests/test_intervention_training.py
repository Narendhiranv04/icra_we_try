"""Unit and integration tests for metrics, sampler, training config, checkpointing, and training loop."""

import hashlib
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.learning.intervention_config import InterventionTrainingConfig
from src.learning.intervention_dataset import (
    InterventionLearningDataset,
    InterventionGroupBatchSampler,
    collate_intervention_group,
)
from src.learning.intervention_loss import InterventionLoss
from src.learning.intervention_metrics import InterventionMetrics
from src.learning.models.intervention_model_registry import get_intervention_model
from src.learning.models.intervention_outputs import InterventionModelOutput
from scripts.train_intervention_model import (
    _compute_model_fingerprint,
    save_checkpoint,
    load_checkpoint,
    train,
)


# ─── Metrics Tests ─────────────────────────────────────────────────────────────

def test_metrics_tie_aware_and_permutation_invariance():
    """Test that tie-aware metrics give expected 0.5 pair credit and are invariant to row permutations (B0-case)."""
    metrics = InterventionMetrics(tie_atol=1e-6)

    # 1 scene group with 4 candidates: 2 positive, 2 negative.
    # Case: Model outputs exactly tied scores (like B0)
    scores = torch.tensor([0.42, 0.42, 0.42, 0.42])
    post_feasible = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pre_feasible = torch.tensor([0.0, 0.0, 0.0, 0.0])
    causal_effect = torch.tensor([1, 0, 1, 0])
    ptrs = torch.tensor([0, 4])

    output = InterventionModelOutput(post_logit=scores, ranking_score=scores)
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": causal_effect}

    res = metrics.compute(output, supervision, ptrs)

    # 4 pairs, each tied -> each gets 0.5 -> average 0.5
    assert res.pairwise_ranking_pair_count == 4
    assert res.pairwise_ranking_accuracy == 0.5
    # Tied max set = all 4 candidates -> expected success = 2 / 4 = 0.5
    assert res.top1_post_feasible_rate == 0.5
    assert res.stop_top1_feasible_rate == 0.5
    assert res.proceed_top1_safe_rate is None

    # Permute candidate order within the scene
    perm = [3, 0, 2, 1]
    scores_perm = scores[perm]
    post_perm = post_feasible[perm]
    pre_perm = pre_feasible[perm]
    causal_perm = causal_effect[perm]

    output_perm = InterventionModelOutput(post_logit=scores_perm, ranking_score=scores_perm)
    supervision_perm = {"post_feasible": post_perm, "pre_feasible": pre_perm, "causal_effect": causal_perm}

    res_perm = metrics.compute(output_perm, supervision_perm, ptrs)

    assert res_perm.pairwise_ranking_accuracy == res.pairwise_ranking_accuracy
    assert res_perm.top1_post_feasible_rate == res.top1_post_feasible_rate
    assert res_perm.stop_top1_feasible_rate == res.stop_top1_feasible_rate


def test_metrics_zero_ranking_pairs_returns_none():
    """Test that zero valid ranking pairs results in pairwise_ranking_accuracy=None and pair_count=0 (not 1.0)."""
    metrics = InterventionMetrics()

    # All positive
    scores = torch.tensor([2.0, 1.0])
    post_feasible = torch.tensor([1.0, 1.0])
    pre_feasible = torch.tensor([1.0, 1.0])
    ptrs = torch.tensor([0, 2])

    output = InterventionModelOutput(post_logit=scores, ranking_score=scores)
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": torch.tensor([0, 0])}

    res = metrics.compute(output, supervision, ptrs)

    assert res.pairwise_ranking_pair_count == 0
    assert res.pairwise_ranking_accuracy is None


def test_metrics_counterfactual_probability_delta():
    """Test counterfactual probability delta MAE diagnostic."""
    metrics = InterventionMetrics()

    # 1 scene group with 3 candidates:
    # 0: NONE candidate (score = 0.0, sigmoid = 0.5, causal_effect = 0)
    # 1: REPAIR candidate (score = 2.0, sigmoid = 0.8808, causal_effect = +1)
    # 2: HARD_NEG candidate (score = -2.0, sigmoid = 0.1192, causal_effect = 0)
    scores = torch.tensor([0.0, 2.0, -2.0])
    causal_effect = torch.tensor([0, 1, 0])
    operator_idx = torch.tensor([0, 1, 1])
    ptrs = torch.tensor([0, 3])

    mae = metrics.compute_counterfactual_delta(scores, causal_effect, operator_idx, ptrs)
    assert mae is not None

    p_none = 0.5
    d1 = torch.sigmoid(torch.tensor(2.0)).item() - p_none   # ~0.3808; GT = 1.0 -> err ~ 0.6192
    d2 = torch.sigmoid(torch.tensor(-2.0)).item() - p_none  # ~-0.3808; GT = 0.0 -> err ~ 0.3808
    d0 = 0.0 - 0.0                                          # 0.0; GT = 0.0 -> err = 0.0
    expected_mae = (abs(d0 - 0) + abs(d1 - 1.0) + abs(d2 - 0.0)) / 3.0

    assert abs(mae - expected_mae) < 1e-4


# ─── Sampler Tests ─────────────────────────────────────────────────────────────

def test_epoch_aware_sampler_reproducibility():
    """Test that setting epoch controls shuffle order deterministically."""
    # Build a synthetic dataset stub
    class MockDataset:
        def __init__(self):
            # 6 scenes
            self.records = [{"scene_id": f"scene_{i}"} for i in range(6) for _ in range(3)]

    ds = MockDataset()

    sampler = InterventionGroupBatchSampler(dataset=ds, scenes_per_batch=1, shuffle=True, seed=42)

    # Epoch 0
    sampler.set_epoch(0)
    order_ep0_run1 = list(sampler)

    sampler.set_epoch(0)
    order_ep0_run2 = list(sampler)

    # Identical across same (seed, epoch)
    assert order_ep0_run1 == order_ep0_run2

    # Epoch 1 must produce a different order
    sampler.set_epoch(1)
    order_ep1 = list(sampler)

    assert order_ep0_run1 != order_ep1


# ─── Config Validation Tests ───────────────────────────────────────────────────

def test_config_typo_and_irrelevant_key_rejection(tmp_path):
    """Test loud rejection of unknown keys, typos, and architecture-irrelevant parameters."""
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text("model_name: intervention_relational\nunknown_param_typo: 123\n")

    with pytest.raises(ValueError, match="Unknown configuration keys"):
        InterventionTrainingConfig.from_yaml(str(bad_yaml))

    # Typo: num_cross_layer instead of num_cross_layers
    with pytest.raises(TypeError):
        InterventionTrainingConfig(model_name="intervention_relational", num_cross_layer=4)

    # query_only with lambda_rank > 0 must fail
    with pytest.raises(ValueError, match="candidate-invariant and cannot use ranking loss"):
        InterventionTrainingConfig(model_name="query_only", lambda_rank=1.0)

    # query_only setting non-default transformer params must fail
    with pytest.raises(ValueError, match="does not use transformer parameters"):
        InterventionTrainingConfig(model_name="query_only", nhead=4)

    # intervention_relational with latent_dim % nhead != 0 must fail
    with pytest.raises(ValueError, match="latent_dim % nhead == 0"):
        InterventionTrainingConfig(model_name="intervention_relational", latent_dim=250, nhead=8)

    # Negative learning rate
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        InterventionTrainingConfig(learning_rate=-0.01)


def test_canonical_smoke_configs_load_and_validate():
    """Test that all 4 canonical smoke config YAML files load and validate properly."""
    cfg_b0 = InterventionTrainingConfig.from_yaml("configs/intervention/learning/b0_smoke.yaml")
    assert cfg_b0.model_name == "query_only"
    assert cfg_b0.lambda_rank == 0.0

    cfg_b0b = InterventionTrainingConfig.from_yaml("configs/intervention/learning/b0b_smoke.yaml")
    assert cfg_b0b.model_name == "simple_intervention"
    assert cfg_b0b.lambda_rank == 1.0

    cfg_b3 = InterventionTrainingConfig.from_yaml("configs/intervention/learning/b3_smoke.yaml")
    assert cfg_b3.model_name == "intervention_relational"
    assert cfg_b3.lambda_rank == 0.0

    cfg_v2 = InterventionTrainingConfig.from_yaml("configs/intervention/learning/v2_smoke.yaml")
    assert cfg_v2.model_name == "intervention_relational"
    assert cfg_v2.lambda_rank == 1.0

    # B3 and V2 must have IDENTICAL architecture hyperparameters and seed
    assert cfg_b3.latent_dim == cfg_v2.latent_dim
    assert cfg_b3.operator_embed_dim == cfg_v2.operator_embed_dim
    assert cfg_b3.nhead == cfg_v2.nhead
    assert cfg_b3.num_context_layers == cfg_v2.num_context_layers
    assert cfg_b3.num_cross_layers == cfg_v2.num_cross_layers
    assert cfg_b3.dropout == cfg_v2.dropout
    assert cfg_b3.seed == cfg_v2.seed


# ─── B3 / V2 Initialization Fingerprint Equality ──────────────────────────────

def test_b3_and_v2_same_seed_identical_initial_parameters():
    """Verify that same seed + same architecture config produces bitwise identical initial model parameters."""
    cfg_b3 = InterventionTrainingConfig.from_yaml("configs/intervention/learning/b3_smoke.yaml")
    cfg_v2 = InterventionTrainingConfig.from_yaml("configs/intervention/learning/v2_smoke.yaml")

    # Instantiate B3
    torch.manual_seed(cfg_b3.seed)
    m_b3 = get_intervention_model(
        model_name=cfg_b3.model_name,
        latent_dim=cfg_b3.latent_dim,
        operator_embed_dim=cfg_b3.operator_embed_dim,
        nhead=cfg_b3.nhead,
        num_context_layers=cfg_b3.num_context_layers,
        num_cross_layers=cfg_b3.num_cross_layers,
        dropout=cfg_b3.dropout,
    )
    fp_b3 = _compute_model_fingerprint(m_b3)

    # Instantiate V2
    torch.manual_seed(cfg_v2.seed)
    m_v2 = get_intervention_model(
        model_name=cfg_v2.model_name,
        latent_dim=cfg_v2.latent_dim,
        operator_embed_dim=cfg_v2.operator_embed_dim,
        nhead=cfg_v2.nhead,
        num_context_layers=cfg_v2.num_context_layers,
        num_cross_layers=cfg_v2.num_cross_layers,
        dropout=cfg_v2.dropout,
    )
    fp_v2 = _compute_model_fingerprint(m_v2)

    assert fp_b3 == fp_v2, f"Initial parameter fingerprints mismatch: {fp_b3} != {fp_v2}"

    # Verify bitwise equality on every single parameter tensor
    b3_params = dict(m_b3.named_parameters())
    v2_params = dict(m_v2.named_parameters())

    assert b3_params.keys() == v2_params.keys()
    for name in b3_params:
        torch.testing.assert_close(b3_params[name], v2_params[name])


# ─── Training Step & Finiteness ───────────────────────────────────────────────

def test_optimizer_step_finite_gradients_all_models():
    """Verify 1 optimizer step on synthetic batch produces finite gradients and finite parameters for all 4 models."""
    from tests.test_intervention_models import get_mock_model_inputs

    models = [
        ("query_only", InterventionLoss(lambda_feas=1.0, lambda_rank=0.0)),
        ("simple_intervention", InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)),
        ("intervention_relational", InterventionLoss(lambda_feas=1.0, lambda_rank=0.0)),  # B3
        ("intervention_relational", InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)),  # V2
    ]

    inputs = get_mock_model_inputs(batch_size=4)
    supervision = {
        "post_feasible": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "pre_feasible": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "causal_effect": torch.tensor([1, 0, 1, 0]),
    }
    group_ptrs = torch.tensor([0, 4])

    for model_name, criterion in models:
        model = get_intervention_model(model_name)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, supervision, group_ptrs)

        assert torch.isfinite(loss.total_loss)
        loss.total_loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert torch.isfinite(p.grad).all()

        optimizer.step()

        for p in model.parameters():
            assert torch.isfinite(p).all()


# ─── Checkpoint Roundtrip ──────────────────────────────────────────────────────

def test_checkpoint_roundtrip(tmp_path):
    """Verify that saved checkpoint preserves all provenance and reloads identical model tensors and eval output."""
    from tests.test_intervention_models import get_mock_model_inputs

    cfg = InterventionTrainingConfig.from_yaml("configs/intervention/learning/v2_smoke.yaml")
    cfg.checkpoint_dir = str(tmp_path)

    torch.manual_seed(42)
    model = get_intervention_model(cfg.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    fp = _compute_model_fingerprint(model)
    ckpt_file = tmp_path / "v2_test_ckpt.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        global_step=30,
        config=cfg,
        initial_fingerprint=fp,
        manifest_sha="test_manifest_sha",
        cache_index_sha="test_cache_sha",
        out_path=ckpt_file,
    )

    fresh_model = get_intervention_model(cfg.model_name)
    load_checkpoint(
        checkpoint_path=ckpt_file,
        model=fresh_model,
        optimizer=None,
        config=cfg,
        manifest_sha="test_manifest_sha",
        cache_index_sha="test_cache_sha",
    )

    # All parameters identical
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh_model.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)

    # Eval output identical
    model.eval()
    fresh_model.eval()
    inputs = get_mock_model_inputs(batch_size=4)
    with torch.no_grad():
        out1 = model(inputs)
        out2 = fresh_model(inputs)

    torch.testing.assert_close(out1.post_logit, out2.post_logit)
