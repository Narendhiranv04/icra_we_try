"""Architecture-Specific Training Configuration Validation for Intervention Models.

Provides a strict configuration schema with per-architecture validation rules,
ensuring only valid hyperparameter combinations can reach the optimizer.
"""

from dataclasses import dataclass, field
from typing import Optional

import yaml

from src.learning.models.intervention_model_registry import INTERVENTION_MODEL_REGISTRY


@dataclass
class InterventionTrainingConfig:
    """Strict training configuration for intervention learning.

    All model-specific and loss-specific parameters are validated at construction.
    """

    # Run Identity
    run_label: str = "default_smoke"

    # Model Identity
    model_name: str = "intervention_relational"

    # Architecture Hyperparameters
    latent_dim: int = 256
    operator_embed_dim: int = 64
    hidden_dim: Optional[int] = None  # Model-specific default (256 for B0, 512 for B0b)
    nhead: int = 8
    num_context_layers: int = 2
    num_cross_layers: int = 2
    dropout: float = 0.1

    # Loss Hyperparameters
    lambda_feas: float = 1.0
    lambda_rank: float = 0.0

    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Training Schedule
    num_epochs: int = 10
    scenes_per_batch: int = 1
    grad_clip_max_norm: Optional[float] = None

    # Reproducibility
    seed: int = 42
    device: str = "cpu"

    # Data Paths
    manifest_path: str = "data/intervention_smoke/manifest.jsonl"
    features_dir: str = "data/intervention_smoke/features"

    # Checkpoint
    checkpoint_dir: Optional[str] = None
    checkpoint_every_n_epochs: int = 5

    # Logging
    log_every_n_batches: int = 1

    def __post_init__(self):
        """Validate configuration consistency."""
        self.validate()

    def validate(self) -> None:
        """Enforce architecture-specific configuration constraints.

        Raises:
            ValueError: On invalid configuration.
        """
        # Model must be registered
        if self.model_name not in INTERVENTION_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_name '{self.model_name}'. "
                f"Must be one of: {list(INTERVENTION_MODEL_REGISTRY.keys())}"
            )

        # Optimizer validation
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. Must be one of: adam, adamw, sgd"
            )

        # Loss weight constraints
        if self.lambda_feas <= 0:
            raise ValueError(f"lambda_feas must be > 0, got {self.lambda_feas}")
        if self.lambda_rank < 0:
            raise ValueError(f"lambda_rank must be >= 0, got {self.lambda_rank}")

        # Architecture-specific constraints
        if self.model_name == "query_only":
            # B0 MUST NOT use ranking loss (candidate-invariant)
            if self.lambda_rank > 0:
                raise ValueError(
                    f"query_only (B0) is candidate-invariant and cannot use ranking loss. "
                    f"Set lambda_rank=0 (got {self.lambda_rank})."
                )

        if self.model_name in ("query_only", "simple_intervention"):
            # Transformer params are irrelevant for B0 and B0b.
            # Check for obviously wrong values that indicate user intent
            if self.nhead != 8 or self.num_context_layers != 2 or self.num_cross_layers != 2:
                raise ValueError(
                    f"Architecture '{self.model_name}' does not use transformer parameters "
                    f"(nhead, num_context_layers, num_cross_layers). "
                    f"Do not set these for B0/B0b configs. "
                    f"Got nhead={self.nhead}, num_context_layers={self.num_context_layers}, "
                    f"num_cross_layers={self.num_cross_layers}."
                )

        if self.model_name == "intervention_relational":
            # V2/B3: latent_dim must be divisible by nhead
            if self.latent_dim % self.nhead != 0:
                raise ValueError(
                    f"intervention_relational requires latent_dim % nhead == 0. "
                    f"Got latent_dim={self.latent_dim}, nhead={self.nhead} "
                    f"(remainder={self.latent_dim % self.nhead})"
                )
            if self.num_context_layers < 1:
                raise ValueError(
                    f"num_context_layers must be >= 1, got {self.num_context_layers}"
                )
            if self.num_cross_layers < 1:
                raise ValueError(
                    f"num_cross_layers must be >= 1, got {self.num_cross_layers}"
                )

        # Positive-dimension sanity
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")
        if self.operator_embed_dim <= 0:
            raise ValueError(f"operator_embed_dim must be > 0, got {self.operator_embed_dim}")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0 if specified, got {self.hidden_dim}")
        if self.nhead <= 0:
            raise ValueError(f"nhead must be > 0, got {self.nhead}")

        # Training sanity checks
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.scenes_per_batch < 1:
            raise ValueError(f"scenes_per_batch must be >= 1, got {self.scenes_per_batch}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @classmethod
    def from_yaml(cls, path: str) -> "InterventionTrainingConfig":
        """Load configuration from a YAML file.

        Raises ValueError on unknown keys to catch typos early.
        """
        import dataclasses as dc
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML config at '{path}' must be a mapping, got {type(data)}")

        # Flatten nested structure if present (sections like model:, loss:, etc.)
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v

        # Enumerate known field names
        known_fields = {f.name for f in dc.fields(cls)}

        # Reject unknown keys
        unknown = set(flat.keys()) - known_fields
        if unknown:
            raise ValueError(
                f"Unknown configuration keys in '{path}': {sorted(unknown)}. "
                f"Valid keys: {sorted(known_fields)}"
            )

        return cls(**{k: v for k, v in flat.items() if k in known_fields})

    def to_dict(self) -> dict:
        """Serialize to dict for logging / checkpointing."""
        import dataclasses
        return dataclasses.asdict(self)
