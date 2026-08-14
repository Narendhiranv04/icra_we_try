"""B0b — Candidate-Aware Non-Relational Baseline Model for Intervention Learning."""

from typing import Dict, Optional
import torch
import torch.nn as nn

from src.learning.intervention_feature_spec import InterventionFeatureSpec
from src.learning.models.intervention_outputs import InterventionModelOutput


class SimpleInterventionBaseline(nn.Module):
    """B0b baseline: candidate-aware baseline without patch-level relational cross-attention."""

    def __init__(
        self,
        spec: Optional[InterventionFeatureSpec] = None,
        operator_embed_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spec = spec or InterventionFeatureSpec()
        self.operator_embed_dim = operator_embed_dim

        self.operator_embed = nn.Embedding(
            num_embeddings=self.spec.operator_count,
            embedding_dim=operator_embed_dim,
        )

        in_dim = (
            self.spec.text_dim
            + self.spec.scene_global_dim
            + self.spec.candidate_visual_dim
            + self.spec.current_geom_dim
            + self.spec.dest_geom_dim
            + operator_embed_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> InterventionModelOutput:
        """Forward pass over candidate-aware non-patch features.

        Args:
            model_inputs: Dictionary containing exact seven model-visible input fields.
        """
        required_keys = [
            "text_feat", "scene_global", "candidate_visual",
            "current_geometry", "destination_geometry", "operator_idx"
        ]
        for k in required_keys:
            if k not in model_inputs:
                raise KeyError(f"model_inputs dict missing required key '{k}'")

        text_feat = model_inputs["text_feat"]
        scene_global = model_inputs["scene_global"]
        cand_vis = model_inputs["candidate_visual"]
        curr_geom = model_inputs["current_geometry"]
        dest_geom = model_inputs["destination_geometry"]
        op_idx = model_inputs["operator_idx"]

        op_emb = self.operator_embed(op_idx)

        h_cat = torch.cat([
            text_feat,
            scene_global,
            cand_vis,
            curr_geom,
            dest_geom,
            op_emb,
        ], dim=-1)

        logits = self.mlp(h_cat).squeeze(-1)  # (B,)

        return InterventionModelOutput(
            post_logit=logits,
            ranking_score=logits,
            relational_embedding=None,
            relational_tokens=None,
        )
