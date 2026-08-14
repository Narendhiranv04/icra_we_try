"""Unit and integration tests for Scene-Balanced Grouped Pairwise Intervention Loss."""

import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from src.learning.intervention_loss import InterventionLoss, InterventionLossOutput
from src.learning.models.intervention_outputs import InterventionModelOutput


def test_bce_exact_expected_value():
    """Verify that when lambda_rank=0, loss exactly matches manual BCE value."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=0.0)

    # 4 items in 1 scene group
    post_logit = torch.tensor([2.0, -1.0, 0.5, -2.0])
    post_feasible = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pre_feasible = torch.tensor([0.0, 0.0, 0.0, 0.0])
    group_ptrs = torch.tensor([0, 4])

    output = InterventionModelOutput(post_logit=post_logit, ranking_score=post_logit)
    supervision = {
        "post_feasible": post_feasible,
        "pre_feasible": pre_feasible,
        "causal_effect": torch.tensor([1, 0, 1, 0]),
    }

    loss_out = criterion(output, supervision, group_ptrs)

    expected_bce = F.binary_cross_entropy_with_logits(post_logit, post_feasible)
    torch.testing.assert_close(loss_out.total_loss, expected_bce)
    torch.testing.assert_close(loss_out.post_feasibility_loss, expected_bce)
    # When lambda_rank=0, total loss is unaffected by ranking loss
    torch.testing.assert_close(loss_out.total_loss, criterion.lambda_feas * loss_out.post_feasibility_loss)


def test_group_pointer_validation():
    """Verify assertions on malformed scene group pointers."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)
    post_logit = torch.tensor([1.0, 0.0, 1.0])
    output = InterventionModelOutput(post_logit=post_logit, ranking_score=post_logit)
    supervision = {
        "post_feasible": torch.tensor([1.0, 0.0, 1.0]),
        "pre_feasible": torch.tensor([0.0, 0.0, 0.0]),
        "causal_effect": torch.tensor([1, 0, 1]),
    }

    # First pointer != 0
    with pytest.raises(AssertionError, match="First pointer must be 0"):
        criterion(output, supervision, torch.tensor([1, 3]))

    # Final pointer != batch_size
    with pytest.raises(AssertionError, match="Final pointer must equal batch_size"):
        criterion(output, supervision, torch.tensor([0, 2]))

    # Non-increasing (empty group)
    with pytest.raises(AssertionError, match="strictly increasing"):
        criterion(output, supervision, torch.tensor([0, 2, 2, 3]))


def test_pre_feasible_constant_per_group():
    """Verify assertion that pre_feasible must be constant within each scene group."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)
    post_logit = torch.tensor([1.0, 0.0])
    output = InterventionModelOutput(post_logit=post_logit, ranking_score=post_logit)
    supervision = {
        "post_feasible": torch.tensor([1.0, 0.0]),
        "pre_feasible": torch.tensor([0.0, 1.0]),  # Conflicting pre_feasible in 1 scene!
        "causal_effect": torch.tensor([1, 0]),
    }
    group_ptrs = torch.tensor([0, 2])

    with pytest.raises(AssertionError, match="pre_feasible not constant in group"):
        criterion(output, supervision, group_ptrs)


def test_positive_negative_pairs_only_and_ignoring_same_label():
    """Verify that ranking loss is constructed strictly across positive vs negative candidates."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)

    # Group: 2 pos (scores 2.0, 1.0), 1 neg (score 0.0)
    # Pairs: (2.0 - 0.0) and (1.0 - 0.0)
    scores = torch.tensor([2.0, 1.0, 0.0])
    post_feasible = torch.tensor([1.0, 1.0, 0.0])
    pre_feasible = torch.tensor([0.0, 0.0, 0.0])
    group_ptrs = torch.tensor([0, 3])

    output = InterventionModelOutput(post_logit=scores, ranking_score=scores)
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": torch.tensor([1, 1, 0])}

    loss_out = criterion(output, supervision, group_ptrs)

    assert loss_out.ranking_pair_count == 2
    assert loss_out.valid_ranking_group_count == 1

    expected_l_rank = (math.log(1.0 + math.exp(-2.0)) + math.log(1.0 + math.exp(-1.0))) / 2.0
    torch.testing.assert_close(loss_out.ranking_loss, torch.tensor(expected_l_rank, dtype=torch.float32))


def test_all_positive_and_all_negative_groups_contribute_zero_pairs():
    """Verify that all-positive or all-negative groups contribute 0 ranking pairs and are excluded from ranking average."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)

    # Group 0: all positive (2 items) -> 0 pairs
    # Group 1: all negative (2 items) -> 0 pairs
    # Group 2: mixed (1 pos, 1 neg) -> 1 pair
    scores = torch.tensor([1.0, 2.0, -1.0, -2.0, 3.0, -1.0])
    post_feasible = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    pre_feasible = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    group_ptrs = torch.tensor([0, 2, 4, 6])

    output = InterventionModelOutput(post_logit=scores, ranking_score=scores)
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": torch.zeros(6, dtype=torch.long)}

    loss_out = criterion(output, supervision, group_ptrs)

    assert loss_out.ranking_pair_count == 1
    assert loss_out.valid_ranking_group_count == 1

    # Loss is determined solely by the single mixed group
    expected_rank = math.log(1.0 + math.exp(-(3.0 - (-1.0))))
    torch.testing.assert_close(loss_out.ranking_loss, torch.tensor(expected_rank, dtype=torch.float32))


def test_scene_normalized_ranking_weight_equality():
    """Verify scene normalization: a scene with 2 pairs and a scene with 10 pairs have EQUAL scene weight."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)

    # Scene 1: 1 pos (score 1.0), 2 negs (scores 0.0, 0.0) -> 2 pairs, loss = softplus(-1.0)
    # Scene 2: 1 pos (score 2.0), 10 negs (scores 0.0) -> 10 pairs, loss = softplus(-2.0)
    s1_scores = [1.0, 0.0, 0.0]
    s1_post = [1.0, 0.0, 0.0]
    s1_pre = [0.0, 0.0, 0.0]

    s2_scores = [2.0] + [0.0] * 10
    s2_post = [1.0] + [0.0] * 10
    s2_pre = [0.0] * 11

    scores = torch.tensor(s1_scores + s2_scores)
    post_feasible = torch.tensor(s1_post + s2_post)
    pre_feasible = torch.tensor(s1_pre + s2_pre)
    group_ptrs = torch.tensor([0, 3, 14])

    output = InterventionModelOutput(post_logit=scores, ranking_score=scores)
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": torch.zeros(14, dtype=torch.long)}

    loss_out = criterion(output, supervision, group_ptrs)

    assert loss_out.ranking_pair_count == 12  # 2 + 10
    assert loss_out.valid_ranking_group_count == 2

    l_g1 = math.log(1.0 + math.exp(-1.0))
    l_g2 = math.log(1.0 + math.exp(-2.0))
    expected_rank = (l_g1 + l_g2) / 2.0  # Equal weight 0.5 each!

    torch.testing.assert_close(loss_out.ranking_loss, torch.tensor(expected_rank, dtype=torch.float32))


def test_monotonicity_of_logistic_ranking_loss():
    """Verify that increasing positive score relative to negative score strictly decreases ranking loss."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)
    group_ptrs = torch.tensor([0, 2])
    post_feasible = torch.tensor([1.0, 0.0])
    pre_feasible = torch.tensor([0.0, 0.0])
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": torch.tensor([1, 0])}

    # Small margin
    out_small = InterventionModelOutput(post_logit=torch.tensor([0.5, 0.0]), ranking_score=torch.tensor([0.5, 0.0]))
    loss_small = criterion(out_small, supervision, group_ptrs).ranking_loss

    # Large margin
    out_large = InterventionModelOutput(post_logit=torch.tensor([5.0, -5.0]), ranking_score=torch.tensor([5.0, -5.0]))
    loss_large = criterion(out_large, supervision, group_ptrs).ranking_loss

    assert loss_large < loss_small


def test_zero_pair_ranking_loss_is_differentiable_and_device_safe():
    """Verify zero valid pairs case returns a differentiable zero loss on correct device."""
    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)

    # All positive
    scores = torch.tensor([1.0, 2.0], requires_grad=True)
    post_feasible = torch.tensor([1.0, 1.0])
    pre_feasible = torch.tensor([1.0, 1.0])
    group_ptrs = torch.tensor([0, 2])

    output = InterventionModelOutput(post_logit=scores, ranking_score=scores)
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": torch.tensor([0, 0])}

    loss_out = criterion(output, supervision, group_ptrs)

    assert loss_out.ranking_pair_count == 0
    assert loss_out.valid_ranking_group_count == 0
    assert loss_out.ranking_loss.item() == 0.0

    # Must be differentiable without error
    loss_out.total_loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_b3_vs_v2_objective_ablation_difference():
    """Verify that B3 (lambda_rank=0) and V2 (lambda_rank=1.0) on identical inputs differ only by the ranking loss."""
    crit_b3 = InterventionLoss(lambda_feas=1.0, lambda_rank=0.0)
    crit_v2 = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)

    scores = torch.tensor([1.0, -1.0])
    post_feasible = torch.tensor([1.0, 0.0])
    pre_feasible = torch.tensor([0.0, 0.0])
    group_ptrs = torch.tensor([0, 2])

    output = InterventionModelOutput(post_logit=scores, ranking_score=scores)
    supervision = {"post_feasible": post_feasible, "pre_feasible": pre_feasible, "causal_effect": torch.tensor([1, 0])}

    l_b3 = crit_b3(output, supervision, group_ptrs)
    l_v2 = crit_v2(output, supervision, group_ptrs)

    torch.testing.assert_close(l_b3.post_feasibility_loss, l_v2.post_feasibility_loss)
    torch.testing.assert_close(l_v2.total_loss, l_b3.total_loss + l_v2.ranking_loss)


def test_smoke_dataset_ranking_pair_count():
    """Integration test verifying exact 16 ranking pairs across the 6 smoke dataset scenes."""
    manifest_path = Path("data/intervention_smoke/manifest.jsonl")
    features_dir = Path("data/intervention_smoke/features")

    if not manifest_path.exists() or not features_dir.exists():
        pytest.skip("Smoke dataset or features directory not present.")

    from src.learning.intervention_dataset import (
        InterventionLearningDataset,
        InterventionGroupBatchSampler,
        collate_intervention_group,
    )

    ds = InterventionLearningDataset(manifest_path=manifest_path, features_dir=features_dir, split="all")
    sampler = InterventionGroupBatchSampler(dataset=ds, scenes_per_batch=len(ds), shuffle=False)
    batch_indices = next(iter(sampler))
    batch = collate_intervention_group([ds[i] for i in batch_indices])

    criterion = InterventionLoss(lambda_feas=1.0, lambda_rank=1.0)

    # Dummy logits
    dummy_logits = torch.zeros(len(ds))
    output = InterventionModelOutput(post_logit=dummy_logits, ranking_score=dummy_logits)

    loss_out = criterion(output, batch["supervision_targets"], batch["scene_group_ptrs"])

    assert loss_out.ranking_pair_count == 16, f"Expected 16 smoke pairs, got {loss_out.ranking_pair_count}"
    assert loss_out.valid_ranking_group_count == 6
