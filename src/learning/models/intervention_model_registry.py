"""Model Registry and Dispatch for Intervention-Conditioned Architectures."""

from typing import Dict, Optional, Type
import torch.nn as nn

from src.learning.intervention_feature_spec import InterventionFeatureSpec
from src.learning.models.intervention_outputs import InterventionModelOutput
from src.learning.models.intervention_query_only import InterventionQueryOnlyBaseline
from src.learning.models.simple_intervention import SimpleInterventionBaseline
from src.learning.models.intervention_relational import InterventionRelationalModel


INTERVENTION_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "query_only": InterventionQueryOnlyBaseline,
    "simple_intervention": SimpleInterventionBaseline,
    "intervention_relational": InterventionRelationalModel,
}


def get_intervention_model(
    model_name: str,
    spec: Optional[InterventionFeatureSpec] = None,
    latent_dim: int = 256,
    operator_embed_dim: int = 64,
    hidden_dim: Optional[int] = None,
    nhead: int = 8,
    num_context_layers: int = 2,
    num_cross_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Instantiate an intervention learning model from registered architectural identities.

    Args:
        model_name: Identity string ('query_only', 'simple_intervention', 'intervention_relational')
        spec: InterventionFeatureSpec instance
        latent_dim: Latent dimension for V2 projections and transformer layers
        operator_embed_dim: Embedding dimension for physical operators
        hidden_dim: Hidden MLP dimension (defaults to 256 for B0, 512 for B0b)
        nhead: Transformer attention heads for V2
        num_context_layers: Transformer context encoder layers for V2
        num_cross_layers: Transformer cross attention layers for V2
        dropout: Dropout probability

    Raises:
        ValueError: If model_name is unknown or 'relational_feasibility_only'.
        TypeError: If unexpected keyword arguments are passed (no **kwargs).
    """
    if model_name == "relational_feasibility_only":
        raise ValueError(
            "Model 'relational_feasibility_only' (B3) is not a separate architecture. "
            "Use model_name='intervention_relational' and set lambda_rank=0 in training configuration."
        )

    if model_name not in INTERVENTION_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown intervention model architecture: '{model_name}'. "
            f"Supported architectures: {list(INTERVENTION_MODEL_REGISTRY.keys())}"
        )

    spec_inst = spec or InterventionFeatureSpec()

    if model_name == "query_only":
        h_dim = hidden_dim if hidden_dim is not None else 256
        return InterventionQueryOnlyBaseline(
            spec=spec_inst,
            hidden_dim=h_dim,
            dropout=dropout,
        )
    elif model_name == "simple_intervention":
        h_dim = hidden_dim if hidden_dim is not None else 512
        return SimpleInterventionBaseline(
            spec=spec_inst,
            operator_embed_dim=operator_embed_dim,
            hidden_dim=h_dim,
            dropout=dropout,
        )
    elif model_name == "intervention_relational":
        return InterventionRelationalModel(
            spec=spec_inst,
            latent_dim=latent_dim,
            operator_embed_dim=operator_embed_dim,
            nhead=nhead,
            num_context_layers=num_context_layers,
            num_cross_layers=num_cross_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unhandled registered model name: {model_name}")
