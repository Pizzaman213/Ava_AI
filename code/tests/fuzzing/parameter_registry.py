"""
Parameter Registry for Config Fuzzer

Defines all fuzzable parameters with their types, constraints, and validation stages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ParameterSpec:
    """Specification for a fuzzable parameter."""

    name: str  # Full path e.g., "model.hidden_size"
    param_type: type  # int, float, str, bool
    valid_range: Optional[Tuple[Optional[float], Optional[float]]] = None  # (min, max)
    valid_values: Optional[List[Any]] = None  # For enums
    constraints: List[str] = field(default_factory=list)  # Cross-field constraints
    default_value: Any = None
    validation_stage: str = "config"  # Where validation SHOULD occur
    description: str = ""

    @property
    def short_name(self) -> str:
        """Get the parameter name without the section prefix."""
        return self.name.split(".")[-1]

    @property
    def section(self) -> str:
        """Get the config section (model, training, data, etc.)."""
        parts = self.name.split(".")
        return parts[0] if len(parts) > 1 else ""


class ParameterRegistry:
    """Central registry of all fuzzable configuration parameters."""

    # Model architecture parameters
    MODEL_PARAMS: Dict[str, ParameterSpec] = {
        "vocab_size": ParameterSpec(
            name="model.vocab_size",
            param_type=int,
            valid_range=(1, None),
            default_value=50680,
            validation_stage="config",
            description="Vocabulary size for token embeddings"
        ),
        "hidden_size": ParameterSpec(
            name="model.hidden_size",
            param_type=int,
            valid_range=(1, None),
            constraints=["divisible_by:num_attention_heads"],
            default_value=1024,
            validation_stage="config",
            description="Hidden dimension size"
        ),
        "num_layers": ParameterSpec(
            name="model.num_layers",
            param_type=int,
            valid_range=(1, None),
            default_value=6,
            validation_stage="config",
            description="Number of transformer layers"
        ),
        "num_attention_heads": ParameterSpec(
            name="model.num_attention_heads",
            param_type=int,
            valid_range=(1, None),
            default_value=16,
            validation_stage="config",
            description="Number of attention heads"
        ),
        "intermediate_size": ParameterSpec(
            name="model.intermediate_size",
            param_type=int,
            valid_range=(1, None),
            default_value=4096,
            validation_stage="config",
            description="FFN intermediate dimension"
        ),
        "num_experts": ParameterSpec(
            name="model.num_experts",
            param_type=int,
            valid_range=(1, None),
            default_value=4,
            validation_stage="config",
            description="Number of MoE experts"
        ),
        "num_experts_per_token": ParameterSpec(
            name="model.num_experts_per_token",
            param_type=int,
            valid_range=(1, None),
            constraints=["le:num_experts"],
            default_value=2,
            validation_stage="config",
            description="Number of experts activated per token (top-k)"
        ),
        "capacity_factor": ParameterSpec(
            name="model.capacity_factor",
            param_type=float,
            valid_range=(0.0, None),  # Should be > 0
            default_value=1.25,
            validation_stage="config",
            description="Expert capacity factor for load balancing"
        ),
        "dropout": ParameterSpec(
            name="model.dropout",
            param_type=float,
            valid_range=(0.0, 1.0),
            default_value=0.1,
            validation_stage="config",
            description="Dropout probability"
        ),
        "attention_dropout": ParameterSpec(
            name="model.attention_dropout",
            param_type=float,
            valid_range=(0.0, 1.0),
            default_value=0.1,
            validation_stage="config",
            description="Attention dropout probability"
        ),
        "expert_dropout": ParameterSpec(
            name="model.expert_dropout",
            param_type=float,
            valid_range=(0.0, 1.0),
            default_value=0.0,
            validation_stage="config",
            description="Expert dropout probability"
        ),
        "router_type": ParameterSpec(
            name="model.router_type",
            param_type=str,
            valid_values=["mixtral", "deepseek", "switch"],
            default_value="mixtral",
            validation_stage="config",
            description="MoE router type"
        ),
        "activation": ParameterSpec(
            name="model.activation",
            param_type=str,
            valid_values=["swiglu", "geglu", "gelu", "relu", "silu"],
            default_value="swiglu",
            validation_stage="config",
            description="Activation function"
        ),
        "max_position_embeddings": ParameterSpec(
            name="model.max_position_embeddings",
            param_type=int,
            valid_range=(1, None),
            default_value=2048,
            validation_stage="config",
            description="Maximum sequence length"
        ),
        "layer_norm_eps": ParameterSpec(
            name="model.layer_norm_eps",
            param_type=float,
            valid_range=(0.0, None),  # Should be > 0
            default_value=1e-5,
            validation_stage="config",
            description="Layer norm epsilon for numerical stability"
        ),
        "rope_theta": ParameterSpec(
            name="model.rope_theta",
            param_type=float,
            valid_range=(0.0, None),
            default_value=10000.0,
            validation_stage="config",
            description="RoPE base frequency"
        ),
        "router_z_loss_coef": ParameterSpec(
            name="model.router_z_loss_coef",
            param_type=float,
            valid_range=(0.0, None),
            default_value=0.0001,
            validation_stage="config",
            description="Router z-loss coefficient"
        ),
        "load_balance_loss_coef": ParameterSpec(
            name="model.load_balance_loss_coef",
            param_type=float,
            valid_range=(0.0, None),
            default_value=0.01,
            validation_stage="config",
            description="Load balancing loss coefficient"
        ),
        "use_flash_attention": ParameterSpec(
            name="model.use_flash_attention",
            param_type=bool,
            valid_values=[True, False],
            default_value=True,
            validation_stage="config",
            description="Use Flash Attention"
        ),
        "gradient_checkpointing": ParameterSpec(
            name="model.gradient_checkpointing",
            param_type=bool,
            valid_values=[True, False],
            default_value=True,
            validation_stage="config",
            description="Enable gradient checkpointing"
        ),
    }

    # Training parameters
    TRAINING_PARAMS: Dict[str, ParameterSpec] = {
        "batch_size": ParameterSpec(
            name="training.batch_size",
            param_type=int,
            valid_range=(1, None),
            default_value=32,
            validation_stage="config",
            description="Training batch size"
        ),
        "learning_rate": ParameterSpec(
            name="training.learning_rate",
            param_type=float,
            valid_range=(0.0, None),  # Should be > 0
            default_value=1e-4,
            validation_stage="config",
            description="Learning rate"
        ),
        "gradient_accumulation_steps": ParameterSpec(
            name="training.gradient_accumulation_steps",
            param_type=int,
            valid_range=(1, None),
            default_value=1,
            validation_stage="config",
            description="Gradient accumulation steps"
        ),
        "warmup_steps": ParameterSpec(
            name="training.warmup_steps",
            param_type=int,
            valid_range=(0, None),
            default_value=2000,
            validation_stage="config",
            description="Learning rate warmup steps"
        ),
        "max_gradient_norm": ParameterSpec(
            name="training.max_gradient_norm",
            param_type=float,
            valid_range=(0.0, None),
            default_value=1.0,
            validation_stage="config",
            description="Maximum gradient norm for clipping"
        ),
        "weight_decay": ParameterSpec(
            name="training.weight_decay",
            param_type=float,
            valid_range=(0.0, 1.0),
            default_value=0.01,
            validation_stage="config",
            description="Weight decay coefficient"
        ),
        "max_steps": ParameterSpec(
            name="training.max_steps",
            param_type=int,
            valid_range=(1, None),
            default_value=None,
            validation_stage="config",
            description="Maximum training steps"
        ),
        "epochs": ParameterSpec(
            name="training.epochs",
            param_type=int,
            valid_range=(1, None),
            default_value=5,
            validation_stage="config",
            description="Number of training epochs"
        ),
    }

    # Data loading parameters
    DATA_PARAMS: Dict[str, ParameterSpec] = {
        "max_length": ParameterSpec(
            name="data.max_length",
            param_type=int,
            valid_range=(1, None),
            default_value=512,
            validation_stage="config",
            description="Maximum sequence length for data"
        ),
        "num_workers": ParameterSpec(
            name="data.num_workers",
            param_type=int,
            valid_range=(0, None),
            default_value=4,
            validation_stage="config",
            description="Number of data loading workers"
        ),
        "prefetch_factor": ParameterSpec(
            name="data.prefetch_factor",
            param_type=int,
            valid_range=(1, None),
            default_value=2,
            validation_stage="config",
            description="Prefetch factor for data loading"
        ),
        "buffer_size": ParameterSpec(
            name="data.buffer_size",
            param_type=int,
            valid_range=(1, None),
            default_value=1000,
            validation_stage="config",
            description="Shuffle buffer size"
        ),
    }

    def __init__(self):
        """Initialize the registry with all parameters."""
        self._all_params: Dict[str, ParameterSpec] = {}
        self._all_params.update(self.MODEL_PARAMS)
        self._all_params.update(self.TRAINING_PARAMS)
        self._all_params.update(self.DATA_PARAMS)

    @property
    def all_params(self) -> Dict[str, ParameterSpec]:
        """Get all registered parameters."""
        return self._all_params

    @property
    def model_params(self) -> Dict[str, ParameterSpec]:
        """Get model parameters only."""
        return self.MODEL_PARAMS

    @property
    def training_params(self) -> Dict[str, ParameterSpec]:
        """Get training parameters only."""
        return self.TRAINING_PARAMS

    @property
    def data_params(self) -> Dict[str, ParameterSpec]:
        """Get data parameters only."""
        return self.DATA_PARAMS

    def get_param(self, name: str) -> Optional[ParameterSpec]:
        """Get a parameter specification by name."""
        # Try direct lookup first
        if name in self._all_params:
            return self._all_params[name]
        # Try with section prefix
        for key, spec in self._all_params.items():
            if spec.name == name or spec.short_name == name:
                return spec
        return None

    def get_params_by_type(self, param_type: type) -> List[ParameterSpec]:
        """Get all parameters of a specific type."""
        return [p for p in self._all_params.values() if p.param_type == param_type]

    def get_params_by_section(self, section: str) -> List[ParameterSpec]:
        """Get all parameters in a config section."""
        return [p for p in self._all_params.values() if p.section == section]

    def get_params_with_constraints(self) -> List[ParameterSpec]:
        """Get all parameters that have cross-field constraints."""
        return [p for p in self._all_params.values() if p.constraints]


# Global registry instance
REGISTRY = ParameterRegistry()


def get_baseline_config() -> Dict[str, Any]:
    """
    Get a minimal valid baseline configuration.
    Uses small values to ensure fast testing on CPU.
    """
    return {
        "model": {
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "capacity_factor": 1.25,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "expert_dropout": 0.0,
            "router_type": "mixtral",
            "activation": "gelu",
            "max_position_embeddings": 512,
            "layer_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "router_z_loss_coef": 0.0001,
            "load_balance_loss_coef": 0.01,
            "use_flash_attention": False,  # CPU compatible
            "gradient_checkpointing": False,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 100,
            "max_gradient_norm": 1.0,
            "weight_decay": 0.01,
            "epochs": 1,
        },
        "data": {
            "max_length": 128,
            "num_workers": 0,
            "prefetch_factor": 2,
            "buffer_size": 100,
        },
    }
