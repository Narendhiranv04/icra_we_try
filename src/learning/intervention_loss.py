"""Scene-Balanced Grouped Pairwise Logistic Intervention Loss.

Implements the canonical Packet-5 intervention training objective:
    L_total = lambda_feas * L_post + lambda_rank * L_rank

where:
    L_post = BCEWithLogitsLoss(post_logit, post_feasible)
    L_rank = scene-normalized pairwise logistic ranking loss

The ranking loss is normalized per scene before group averaging, ensuring
every valid scene contributes equal ranking weight regardless of candidate
pair count.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.learning.models.intervention_outputs import InterventionModelOutput


@dataclass
class InterventionLossOutput:
    """Structured loss output with diagnostic fields."""
    total_loss: torch.Tensor
    post_feasibility_loss: torch.Tensor
    ranking_loss: torch.Tensor
    ranking_pair_count: int
    valid_ranking_group_count: int


class InterventionLoss(nn.Module):
    """Scene-balanced grouped pairwise logistic loss for intervention learning.

    Args:
        lambda_feas: Weight for post-feasibility BCE loss (must be > 0).
        lambda_rank: Weight for pairwise ranking loss (>= 0; 0 for BCE-only).
    """

    def __init__(self, lambda_feas: float = 1.0, lambda_rank: float = 0.0):
        super().__init__()
        if lambda_feas <= 0:
            raise ValueError(f"lambda_feas must be > 0, got {lambda_feas}")
        if lambda_rank < 0:
            raise ValueError(f"lambda_rank must be >= 0, got {lambda_rank}")
        self.lambda_feas = lambda_feas
        self.lambda_rank = lambda_rank
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        output: InterventionModelOutput,
        supervision: dict,
        scene_group_ptrs: torch.Tensor,
    ) -> InterventionLossOutput:
        """Compute the total intervention loss.

        Args:
            output: Model output containing post_logit and ranking_score.
            supervision: Dict with 'post_feasible', 'pre_feasible', 'causal_effect'.
            scene_group_ptrs: 1-D tensor of scene group boundary pointers.

        Returns:
            InterventionLossOutput with total_loss, component losses, and diagnostics.
        """
        post_logit = output.post_logit
        ranking_score = output.ranking_score
        post_feasible = supervision["post_feasible"].float()
        pre_feasible = supervision["pre_feasible"].float()

        batch_size = post_logit.shape[0]

        # Validate group pointers
        self._validate_group_pointers(scene_group_ptrs, batch_size)

        # Validate supervision tensors
        assert post_feasible.shape == (batch_size,), f"post_feasible shape {post_feasible.shape} != ({batch_size},)"
        assert pre_feasible.shape == (batch_size,), f"pre_feasible shape {pre_feasible.shape} != ({batch_size},)"

        # Post-feasibility BCE loss
        l_post = self.bce(post_logit, post_feasible)

        # Scene-normalized grouped pairwise logistic ranking loss
        l_rank, pair_count, valid_group_count = self._compute_ranking_loss(
            ranking_score, post_feasible, pre_feasible, scene_group_ptrs
        )

        # Total loss
        total = self.lambda_feas * l_post + self.lambda_rank * l_rank

        return InterventionLossOutput(
            total_loss=total,
            post_feasibility_loss=l_post,
            ranking_loss=l_rank,
            ranking_pair_count=pair_count,
            valid_ranking_group_count=valid_group_count,
        )

    def _validate_group_pointers(self, ptrs: torch.Tensor, batch_size: int) -> None:
        """Validate scene group pointer invariants."""
        assert ptrs.dim() == 1, f"scene_group_ptrs must be 1-D, got {ptrs.dim()}-D"
        assert ptrs[0].item() == 0, f"First pointer must be 0, got {ptrs[0].item()}"
        assert ptrs[-1].item() == batch_size, f"Final pointer must equal batch_size {batch_size}, got {ptrs[-1].item()}"

        diffs = ptrs[1:] - ptrs[:-1]
        assert (diffs > 0).all(), "Pointers must be strictly increasing (no empty groups)"
        assert (ptrs >= 0).all() and (ptrs <= batch_size).all(), "Pointers out of range"

    def _compute_ranking_loss(
        self,
        scores: torch.Tensor,
        post_feasible: torch.Tensor,
        pre_feasible: torch.Tensor,
        ptrs: torch.Tensor,
    ):
        """Compute scene-normalized grouped pairwise logistic ranking loss.

        For each scene group g:
            P_g = {i | post_feasible_i = 1}
            N_g = {j | post_feasible_j = 0}
            Q_g = P_g x N_g (all positive-negative pairs)

            L_rank_g = (1/|Q_g|) * sum_{(i,j) in Q_g} softplus(-(s_i - s_j))

        Then: L_rank = (1/|G_valid|) * sum_{g in G_valid} L_rank_g

        Returns:
            (ranking_loss, total_pair_count, valid_group_count)
        """
        num_groups = ptrs.shape[0] - 1
        group_losses = []
        total_pairs = 0

        for g in range(num_groups):
            start = ptrs[g].item()
            end = ptrs[g + 1].item()

            group_scores = scores[start:end]
            group_labels = post_feasible[start:end]
            group_pre = pre_feasible[start:end]

            # Validate pre_feasible constant within group
            assert (group_pre == group_pre[0]).all(), \
                f"pre_feasible not constant in group {g}: {group_pre.tolist()}"

            # Find positive and negative indices
            pos_mask = group_labels == 1.0
            neg_mask = group_labels == 0.0

            pos_scores = group_scores[pos_mask]
            neg_scores = group_scores[neg_mask]

            n_pos = pos_scores.shape[0]
            n_neg = neg_scores.shape[0]

            if n_pos == 0 or n_neg == 0:
                continue  # No valid pairs in this group

            n_pairs = n_pos * n_neg
            total_pairs += n_pairs

            # Compute pairwise differences: softplus(-(s_pos - s_neg))
            # pos_scores: (P,), neg_scores: (N,)
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # (P, N)
            pair_losses = F.softplus(-diff)  # (P, N)

            # Scene-normalized loss for this group
            group_loss = pair_losses.sum() / n_pairs
            group_losses.append(group_loss)

        if len(group_losses) == 0:
            # Zero-pair case: return differentiable device-safe zero
            zero_loss = scores.sum() * 0.0
            return zero_loss, 0, 0

        # Average over valid groups (scene-balanced)
        ranking_loss = torch.stack(group_losses).mean()
        return ranking_loss, total_pairs, len(group_losses)
