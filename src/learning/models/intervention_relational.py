"""V2 — Intervention-Conditioned Relational Model."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.learning.intervention_feature_spec import InterventionFeatureSpec
from src.learning.models.intervention_outputs import InterventionModelOutput
from src.learning.models.relational_model import (
    TransformerEncoderLayer,
    CrossAttentionLayer,
)


class InterventionRelationalModel(nn.Module):
    """V2 Intervention-Conditioned Relational Model.

    Conditions pre-scene patch representations on instruction text, global scene context,
    and a neutral physical candidate intervention descriptor via context self-attention
    and patch cross-attention.
    """

    def __init__(
        self,
        spec: Optional[InterventionFeatureSpec] = None,
        latent_dim: int = 256,
        operator_embed_dim: int = 64,
        nhead: int = 8,
        num_context_layers: int = 2,
        num_cross_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spec = spec or InterventionFeatureSpec()
        self.latent_dim = latent_dim
        self.operator_embed_dim = operator_embed_dim

        # Operator Embedding
        self.operator_embed = nn.Embedding(
            num_embeddings=self.spec.operator_count,
            embedding_dim=operator_embed_dim,
        )

        # Projections
        self.proj_text = nn.Linear(self.spec.text_dim, latent_dim)
        self.proj_scene_global = nn.Linear(self.spec.scene_global_dim, latent_dim)

        self.interv_input_dim = (
            self.spec.candidate_visual_dim
            + self.spec.current_geom_dim
            + self.spec.dest_geom_dim
            + operator_embed_dim
        )
        self.proj_intervention = nn.Linear(self.interv_input_dim, latent_dim)

        self.proj_patch = nn.Linear(self.spec.scene_patch_dim, latent_dim)

        # 3 Context Tokens Positional / Role Embedding [text, scene_global, intervention]
        self.context_pos_embed = nn.Parameter(torch.zeros(1, 3, latent_dim))
        nn.init.trunc_normal_(self.context_pos_embed, std=0.02)

        # Context Self-Attention Encoder (2 layers)
        self.context_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dropout=dropout)
            for _ in range(num_context_layers)
        ])

        # Patch Cross-Attention (2 layers: query=patch, key/value=context)
        self.cross_attention = nn.ModuleList([
            CrossAttentionLayer(d_model=latent_dim, nhead=nhead, dropout=dropout)
            for _ in range(num_cross_layers)
        ])

        # Post-feasibility Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def encode_intervention(
        self,
        candidate_visual: torch.Tensor,
        current_geometry: torch.Tensor,
        destination_geometry: torch.Tensor,
        operator_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode raw intervention inputs into descriptor h_rho and projected latent z_interv.

        Returns:
            (h_rho, z_interv) where:
            h_rho has shape (B, interv_input_dim)
            z_interv has shape (B, latent_dim)
        """
        op_emb = self.operator_embed(operator_idx)
        h_rho = torch.cat([
            candidate_visual,
            current_geometry,
            destination_geometry,
            op_emb,
        ], dim=-1)
        z_interv = self.proj_intervention(h_rho)
        return h_rho, z_interv

    def forward(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> InterventionModelOutput:
        """Forward pass over model_inputs dictionary.

        Args:
            model_inputs: Dictionary containing exact seven model-visible input fields:
                          text_feat, scene_global, scene_patch, candidate_visual,
                          current_geometry, destination_geometry, operator_idx.
        """
        required_keys = [
            "text_feat", "scene_global", "scene_patch", "candidate_visual",
            "current_geometry", "destination_geometry", "operator_idx"
        ]
        for k in required_keys:
            if k not in model_inputs:
                raise KeyError(f"model_inputs dict missing required key '{k}'")

        text_feat = model_inputs["text_feat"]
        scene_global = model_inputs["scene_global"]
        scene_patch = model_inputs["scene_patch"]
        cand_vis = model_inputs["candidate_visual"]
        curr_geom = model_inputs["current_geometry"]
        dest_geom = model_inputs["destination_geometry"]
        op_idx = model_inputs["operator_idx"]

        # 1. Project Context Inputs
        z_text = self.proj_text(text_feat).unsqueeze(1)            # (B, 1, latent_dim)
        z_scene = self.proj_scene_global(scene_global).unsqueeze(1) # (B, 1, latent_dim)
        _, z_int = self.encode_intervention(cand_vis, curr_geom, dest_geom, op_idx)
        z_int = z_int.unsqueeze(1)                                 # (B, 1, latent_dim)

        # 2. Assemble Context Sequence (B, 3, latent_dim)
        Z_C = torch.cat([z_text, z_scene, z_int], dim=1)
        Z_C = Z_C + self.context_pos_embed

        for layer in self.context_encoder:
            Z_C = layer(Z_C)

        # 3. Project Query Pre-Scene Patches (B, 256, latent_dim)
        Z_R = self.proj_patch(scene_patch)

        # 4. Relational Patch Cross-Attention
        for layer in self.cross_attention:
            Z_R = layer(Z_R, Z_C)

        # 5. Mean Pooling & Classification
        z_R = Z_R.mean(dim=1)                                      # (B, latent_dim)
        logits = self.classifier(z_R).squeeze(-1)                   # (B,)

        return InterventionModelOutput(
            post_logit=logits,
            ranking_score=logits,
            relational_embedding=z_R,
            relational_tokens=Z_R,
        )
