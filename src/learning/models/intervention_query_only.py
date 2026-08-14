"""B0 — Query / Pre-Scene Only Baseline Model for Intervention Learning."""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from src.learning.intervention_feature_spec import InterventionFeatureSpec
from src.learning.models.intervention_outputs import InterventionModelOutput


class InterventionQueryOnlyBaseline(nn.Module):
    """B0 baseline: predicts feasibility strictly from pre-scene global observation.

    Has zero candidate intervention inputs by construction, guaranteeing candidate
    invariance within the same scene.
    """

    def __init__(
        self,
        spec: Optional[InterventionFeatureSpec] = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.spec = spec or InterventionFeatureSpec()
        in_features = self.spec.scene_global_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        model_inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> InterventionModelOutput:
        """Forward pass over pre-scene global feature.

        Args:
            model_inputs: Either standard model_inputs dict containing 'scene_global' (B, 768)
                          or raw scene_global tensor (B, 768).
        """
        if isinstance(model_inputs, dict):
            if "scene_global" not in model_inputs:
                raise KeyError("model_inputs dict missing required key 'scene_global'")
            x = model_inputs["scene_global"]
        else:
            x = model_inputs

        logits = self.net(x).squeeze(-1)  # (B,)

        return InterventionModelOutput(
            post_logit=logits,
            ranking_score=logits,
            relational_embedding=None,
            relational_tokens=None,
        )
