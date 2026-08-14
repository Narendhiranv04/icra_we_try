"""Tie-Aware Non-Privileged Training Metrics for Intervention Learning.

All metrics are computed without privileged metadata (no semantic categories,
no culprit labels). Tie-aware formulations ensure permutation invariance,
critical for B0 which produces identical scores within each scene.

Metric naming avoids overclaiming:
- post_bce / post_accuracy (not "CFR")
- pairwise_ranking_accuracy (not "AUROC")
- top1_post_feasible_rate
- stop_top1_feasible_rate
- proceed_top1_safe_rate
- counterfactual_probability_delta_mae (diagnostic only)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.learning.models.intervention_outputs import InterventionModelOutput


# Numerical tolerance for tie detection
_TIE_ATOL = 1e-6


@dataclass
class InterventionMetricsOutput:
    """Structured metrics output."""
    post_bce: float
    post_accuracy: float
    pairwise_ranking_accuracy: Optional[float]
    pairwise_ranking_pair_count: int
    top1_post_feasible_rate: Optional[float]
    top1_group_count: int
    stop_top1_feasible_rate: Optional[float]
    stop_group_count: int
    proceed_top1_safe_rate: Optional[float]
    proceed_group_count: int
    counterfactual_probability_delta_mae: Optional[float] = None
    counterfactual_delta_group_count: int = 0


class InterventionMetrics:
    """Tie-aware non-privileged metrics computer for intervention learning.

    Args:
        tie_atol: Absolute tolerance for tie detection in scores.
    """

    def __init__(self, tie_atol: float = _TIE_ATOL):
        self.tie_atol = tie_atol

    @torch.no_grad()
    def compute(
        self,
        output: InterventionModelOutput,
        supervision: dict,
        scene_group_ptrs: torch.Tensor,
    ) -> InterventionMetricsOutput:
        """Compute all non-privileged intervention metrics.

        Args:
            output: Model output with post_logit and ranking_score.
            supervision: Dict with 'post_feasible', 'pre_feasible', 'causal_effect'.
            scene_group_ptrs: 1-D group boundary pointers.

        Returns:
            InterventionMetricsOutput with all computed metrics.
        """
        post_logit = output.post_logit
        ranking_score = output.ranking_score
        post_feasible = supervision["post_feasible"].float()
        pre_feasible = supervision["pre_feasible"].float()
        causal_effect = supervision["causal_effect"]

        batch_size = post_logit.shape[0]
        num_groups = scene_group_ptrs.shape[0] - 1

        # 1. Post-feasibility BCE
        post_bce = F.binary_cross_entropy_with_logits(post_logit, post_feasible).item()

        # 2. Post accuracy (threshold at 0.5 probability = 0.0 logit)
        preds = (post_logit >= 0.0).float()
        post_accuracy = (preds == post_feasible).float().mean().item()

        # 3. Scene-balanced pairwise ranking accuracy (tie-aware)
        group_ranking_accs = []
        total_pairs = 0
        for g in range(num_groups):
            start = scene_group_ptrs[g].item()
            end = scene_group_ptrs[g + 1].item()
            g_scores = ranking_score[start:end]
            g_labels = post_feasible[start:end]

            pos_scores = g_scores[g_labels == 1.0]
            neg_scores = g_scores[g_labels == 0.0]

            if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                continue

            n_pairs = pos_scores.numel() * neg_scores.numel()
            total_pairs += n_pairs

            # Pairwise comparison
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # (P, N)
            pair_acc = torch.zeros_like(diff)
            pair_acc[diff > self.tie_atol] = 1.0
            pair_acc[diff.abs() <= self.tie_atol] = 0.5
            # diff < -tie_atol stays 0.0

            group_ranking_accs.append(pair_acc.mean().item())

        pairwise_ranking_accuracy = None
        if group_ranking_accs:
            pairwise_ranking_accuracy = sum(group_ranking_accs) / len(group_ranking_accs)

        # 4. Tie-aware top-1 metrics
        top1_scores = []
        stop_scores = []
        proceed_scores = []

        for g in range(num_groups):
            start = scene_group_ptrs[g].item()
            end = scene_group_ptrs[g + 1].item()
            g_scores = ranking_score[start:end]
            g_labels = post_feasible[start:end]
            g_pre = pre_feasible[start:end]

            s_max = g_scores.max()
            # Tied maximum set M_g = {i | |s_i - s_max| <= atol}
            tied_mask = (g_scores - s_max).abs() <= self.tie_atol
            tied_labels = g_labels[tied_mask]

            # Expected uniform-tie success: mean(post_feasible_i for i in M_g)
            top1_success = tied_labels.mean().item()
            top1_scores.append(top1_success)

            # STOP group: pre_feasible == 0
            if g_pre[0].item() == 0.0:
                stop_scores.append(top1_success)
            # PROCEED group: pre_feasible == 1
            elif g_pre[0].item() == 1.0:
                proceed_scores.append(top1_success)

        top1_post_feasible_rate = sum(top1_scores) / len(top1_scores) if top1_scores else None
        stop_top1_feasible_rate = sum(stop_scores) / len(stop_scores) if stop_scores else None
        proceed_top1_safe_rate = sum(proceed_scores) / len(proceed_scores) if proceed_scores else None

        # 5. Counterfactual probability delta MAE (diagnostic only)
        cf_delta_maes = []
        cf_delta_group_count = 0
        for g in range(num_groups):
            start = scene_group_ptrs[g].item()
            end = scene_group_ptrs[g + 1].item()
            g_scores = ranking_score[start:end]
            g_labels = post_feasible[start:end]
            g_ce = causal_effect[start:end]

            # Find NONE candidate (operator_idx == 0 → causal_effect should be 0 for NONE)
            # We identify NONE by looking for the zero-geometry candidate
            # In practice, there should be exactly one per group
            # Use supervision to find it: NONE has causal_effect=0 and is typically at a known position
            # For robustness, skip if not exactly one NONE candidate identifiable
            # Since we don't have operator_idx in supervision, we use the convention
            # that NONE candidates have causal_effect == 0 and post_feasible matching pre_feasible
            # Actually, we need operator_idx from model_inputs which isn't passed here.
            # So we skip this metric if we can't identify NONE reliably.
            # The training script should pass operator_idx if this metric is needed.
            pass  # Will be computed separately if operator_idx is available

        cf_delta_mae = None

        return InterventionMetricsOutput(
            post_bce=post_bce,
            post_accuracy=post_accuracy,
            pairwise_ranking_accuracy=pairwise_ranking_accuracy,
            pairwise_ranking_pair_count=total_pairs,
            top1_post_feasible_rate=top1_post_feasible_rate,
            top1_group_count=len(top1_scores),
            stop_top1_feasible_rate=stop_top1_feasible_rate,
            stop_group_count=len(stop_scores),
            proceed_top1_safe_rate=proceed_top1_safe_rate,
            proceed_group_count=len(proceed_scores),
            counterfactual_probability_delta_mae=cf_delta_mae,
            counterfactual_delta_group_count=cf_delta_group_count,
        )

    @torch.no_grad()
    def compute_counterfactual_delta(
        self,
        ranking_score: torch.Tensor,
        causal_effect: torch.Tensor,
        operator_idx: torch.Tensor,
        scene_group_ptrs: torch.Tensor,
    ) -> Optional[float]:
        """Compute counterfactual probability-delta MAE diagnostic.

        For each group with exactly one NONE candidate (operator_idx == 0):
            p_none = sigmoid(score_none)
            delta_prob_hat_i = sigmoid(score_i) - p_none
            GT: causal_effect_i in {-1, 0, +1}

        Returns MAE over all defined groups, or None if no valid groups.
        """
        from src.learning.intervention_feature_spec import OPERATOR_NONE_IDX

        num_groups = scene_group_ptrs.shape[0] - 1
        all_errors = []

        for g in range(num_groups):
            start = scene_group_ptrs[g].item()
            end = scene_group_ptrs[g + 1].item()
            g_scores = ranking_score[start:end]
            g_ce = causal_effect[start:end].float()
            g_ops = operator_idx[start:end]

            none_mask = g_ops == OPERATOR_NONE_IDX
            if none_mask.sum().item() != 1:
                continue  # Skip groups without exactly one NONE

            none_idx = none_mask.nonzero(as_tuple=True)[0][0]
            p_none = torch.sigmoid(g_scores[none_idx])

            probs = torch.sigmoid(g_scores)
            delta_hat = probs - p_none
            errors = (delta_hat - g_ce).abs()
            all_errors.append(errors.mean().item())

        if not all_errors:
            return None
        return sum(all_errors) / len(all_errors)
