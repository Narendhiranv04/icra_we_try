"""Structured Model Output Contract for Intervention-Conditioned Models."""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class InterventionModelOutput:
    """Canonical structured output for intervention-conditioned learning models."""

    post_logit: torch.Tensor
    """Unbounded post-intervention relational feasibility logit. Shape: (B,)"""

    ranking_score: torch.Tensor
    """Candidate ranking score (identical to post_logit in Packet 4). Shape: (B,)"""

    relational_embedding: Optional[torch.Tensor] = None
    """Mean-pooled relational embedding (for V2). Shape: (B, latent_dim) or None"""

    relational_tokens: Optional[torch.Tensor] = None
    """Cross-attended patch relational tokens (for V2). Shape: (B, N, latent_dim) or None"""
