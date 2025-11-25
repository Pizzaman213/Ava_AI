"""
Configuration Validation Schema for Ava Training

This module provides comprehensive validation for training configurations,
including parameter bounds, interdependencies, and conflict detection.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParameterConstraint:
    """Defines constraints for a configuration parameter."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = False
    depends_on: Optional[Dict[str, Any]] = None
    conflicts_with: Optional[List[str]] = None
    warning_message: Optional[str] = None


@dataclass
class ValidationRule:
    """Defines a validation rule for configuration parameters."""

    path: str  # Dot-separated path to parameter (e.g., "model.num_experts")
    constraint: ParameterConstraint
    description: str = ""


class ConfigValidator:
    """Validates training configurations against defined schema."""

    def __init__(self):
        """Initialize with default validation rules."""
        self.rules = self._get_default_rules()
        self.warnings = []
        self.errors = []

    def _get_default_rules(self) -> List[ValidationRule]:
        """Get default validation rules for Ava configurations."""

        return [
            # Model parameters
            ValidationRule(
                path="model.hidden_size",
                constraint=ParameterConstraint(
                    min_value=64,
                    max_value=32768,
                    required=True
                ),
                description="Model hidden dimension"
            ),
            ValidationRule(
                path="model.num_layers",
                constraint=ParameterConstraint(
                    min_value=1,
                    max_value=200,
                    required=True
                ),
                description="Number of transformer layers"
            ),
            ValidationRule(
                path="model.num_attention_heads",
                constraint=ParameterConstraint(
                    min_value=1,
                    max_value=256,
                    required=True
                ),
                description="Number of attention heads"
            ),
            ValidationRule(
                path="model.max_position_embeddings",
                constraint=ParameterConstraint(
                    min_value=128,
                    max_value=32768,
                    required=True,
                    warning_message="Values > 16384 may require significant memory"
                ),
                description="Maximum sequence length"
            ),
            ValidationRule(
                path="model.vocab_size",
                constraint=ParameterConstraint(
                    min_value=100,
                    max_value=500000,
                    required=True
                ),
                description="Vocabulary size"
            ),

            # MoE parameters
            ValidationRule(
                path="model.num_experts",
                constraint=ParameterConstraint(
                    min_value=2,
                    max_value=256,
                    required=False
                ),
                description="Number of MoE experts"
            ),
            ValidationRule(
                path="model.num_experts_per_token",
                constraint=ParameterConstraint(
                    min_value=1,
                    max_value=8,
                    required=False,
                    depends_on={"model.num_experts": lambda x: x is not None}
                ),
                description="Experts activated per token"
            ),
            ValidationRule(
                path="model.capacity_factor",
                constraint=ParameterConstraint(
                    min_value=1.0,
                    max_value=10.0,
                    required=False
                ),
                description="Expert capacity multiplier"
            ),

            # Training parameters
            ValidationRule(
                path="training.batch_size",
                constraint=ParameterConstraint(
                    min_value=1,
                    max_value=4096,
                    required=True,
                    warning_message="Batch size > 256 may require gradient accumulation"
                ),
                description="Training batch size per GPU"
            ),
            ValidationRule(
                path="training.learning_rate",
                constraint=ParameterConstraint(
                    min_value=1e-8,
                    max_value=1e-1,
                    required=True,
                    warning_message="LR > 1e-2 may cause instability"
                ),
                description="Initial learning rate"
            ),
            ValidationRule(
                path="training.gradient_accumulation_steps",
                constraint=ParameterConstraint(
                    min_value=1,
                    max_value=128,
                    required=False,
                    warning_message="Values > 16 may lead to stale gradients"
                ),
                description="Gradient accumulation steps"
            ),
            ValidationRule(
                path="training.gradient_clipping",
                constraint=ParameterConstraint(
                    min_value=0.0,
                    max_value=10.0,
                    required=False
                ),
                description="Gradient clipping value"
            ),
            ValidationRule(
                path="training.warmup_steps",
                constraint=ParameterConstraint(
                    min_value=0,
                    max_value=100000,
                    required=False
                ),
                description="Learning rate warmup steps"
            ),

            # Optimizer parameters
            ValidationRule(
                path="optimizer.type",
                constraint=ParameterConstraint(
                    allowed_values=["adam", "adamw", "sgd", "lion", "sophia", "adafactor"],
                    required=True
                ),
                description="Optimizer type"
            ),
            ValidationRule(
                path="optimizer.weight_decay",
                constraint=ParameterConstraint(
                    min_value=0.0,
                    max_value=1.0,
                    required=False
                ),
                description="Weight decay coefficient"
            ),

            # Mixed precision
            ValidationRule(
                path="training.mixed_precision",
                constraint=ParameterConstraint(
                    allowed_values=["fp32", "fp16", "bf16", "fp8"],
                    required=False
                ),
                description="Mixed precision training mode"
            ),

            # Attention parameters
            ValidationRule(
                path="model.use_flash_attention",
                constraint=ParameterConstraint(
                    allowed_values=[True, False],
                    conflicts_with=["model.attention_dropout"],
                    required=False
                ),
                description="Use Flash Attention"
            ),
            ValidationRule(
                path="model.attention_dropout",
                constraint=ParameterConstraint(
                    min_value=0.0,
                    max_value=0.5,
                    required=False,
                    depends_on={"model.use_flash_attention": lambda x: x != True}
                ),
                description="Attention dropout rate"
            ),
        ]

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from config using dot-separated path."""

        keys = path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _check_constraint(
        self,
        value: Any,
        constraint: ParameterConstraint,
        config: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check if a value satisfies a constraint."""

        # Check required
        if constraint.required and value is None:
            return False, "Required parameter is missing"

        # Skip further checks if value is None and not required
        if value is None:
            return True, None

        # Check min/max bounds
        if constraint.min_value is not None and value < constraint.min_value:
            return False, f"Value {value} is below minimum {constraint.min_value}"

        if constraint.max_value is not None and value > constraint.max_value:
            return False, f"Value {value} exceeds maximum {constraint.max_value}"

        # Check allowed values
        if constraint.allowed_values is not None and value not in constraint.allowed_values:
            return False, f"Value {value} not in allowed values: {constraint.allowed_values}"

        # Check dependencies
        if constraint.depends_on:
            for dep_path, dep_check in constraint.depends_on.items():
                dep_value = self._get_nested_value(config, dep_path)
                if not dep_check(dep_value):
                    return False, f"Dependency not satisfied: {dep_path}"

        # Check conflicts
        if constraint.conflicts_with:
            for conflict_path in constraint.conflicts_with:
                conflict_value = self._get_nested_value(config, conflict_path)
                if conflict_value is not None and conflict_value != 0 and conflict_value != False:
                    return False, f"Conflicts with {conflict_path}={conflict_value}"

        return True, None

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration against the schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid, False otherwise

        Side effects:
            Populates self.warnings and self.errors lists
        """

        self.warnings = []
        self.errors = []

        for rule in self.rules:
            value = self._get_nested_value(config, rule.path)

            is_valid, error_msg = self._check_constraint(value, rule.constraint, config)

            if not is_valid:
                self.errors.append(f"{rule.path}: {error_msg} ({rule.description})")
            elif rule.constraint.warning_message and value is not None:
                # Check if warning conditions are met
                should_warn = False

                if rule.constraint.max_value and value > rule.constraint.max_value * 0.8:
                    should_warn = True
                elif rule.path == "training.batch_size" and value > 256:
                    should_warn = True
                elif rule.path == "training.learning_rate" and value > 1e-2:
                    should_warn = True
                elif rule.path == "training.gradient_accumulation_steps" and value > 16:
                    should_warn = True
                elif rule.path == "model.max_position_embeddings" and value > 16384:
                    should_warn = True

                if should_warn:
                    self.warnings.append(f"{rule.path}: {rule.constraint.warning_message}")

        # Additional interdependency checks
        self._check_interdependencies(config)

        return len(self.errors) == 0

    def _check_interdependencies(self, config: Dict[str, Any]):
        """Check complex interdependencies between parameters."""

        # Flash attention requires attention_dropout = 0
        use_flash = self._get_nested_value(config, "model.use_flash_attention")
        attn_dropout = self._get_nested_value(config, "model.attention_dropout")
        if use_flash and attn_dropout and attn_dropout > 0:
            self.errors.append("Flash attention requires attention_dropout=0.0")

        # Gradient checkpointing conflicts with torch.compile
        grad_checkpoint = self._get_nested_value(config, "training.gradient_checkpointing")
        torch_compile = self._get_nested_value(config, "training.use_torch_compile")
        if grad_checkpoint and torch_compile:
            self.errors.append("Gradient checkpointing conflicts with torch.compile")

        # Expert offloading conflicts with full LoRA
        expert_offload = self._get_nested_value(config, "model.use_expert_offloading")
        lora_experts = self._get_nested_value(config, "model.use_lora_experts")
        if expert_offload and not lora_experts:
            self.warnings.append("Expert offloading works best with LoRA experts to reduce memory")

        # num_experts_per_token must be <= num_experts
        num_experts = self._get_nested_value(config, "model.num_experts")
        experts_per_token = self._get_nested_value(config, "model.num_experts_per_token")
        if num_experts and experts_per_token and experts_per_token > num_experts:
            self.errors.append(f"num_experts_per_token ({experts_per_token}) cannot exceed num_experts ({num_experts})")

        # Hidden size must be divisible by num_attention_heads
        hidden_size = self._get_nested_value(config, "model.hidden_size")
        num_heads = self._get_nested_value(config, "model.num_attention_heads")
        if hidden_size and num_heads and hidden_size % num_heads != 0:
            self.errors.append(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})")

        # Intermediate size recommendations
        intermediate_size = self._get_nested_value(config, "model.intermediate_size")
        if hidden_size and intermediate_size:
            ratio = intermediate_size / hidden_size
            if ratio < 2.0 or ratio > 5.0:
                self.warnings.append(f"intermediate_size/hidden_size ratio ({ratio:.1f}) should be between 2-5 (typically 3.5-4)")

    def print_report(self):
        """Print validation report."""

        if self.errors:
            logger.error("Configuration validation failed:")
            for error in self.errors:
                logger.error(f"   {error}")

        if self.warnings:
            logger.warning("Configuration warnings:")
            for warning in self.warnings:
                logger.warning(f"    {warning}")

        if not self.errors and not self.warnings:
            logger.info(" Configuration validation passed")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Convenience function to validate a configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """

    validator = ConfigValidator()
    is_valid = validator.validate(config)
    validator.print_report()
    return is_valid


# Export key components
__all__ = [
    'ConfigValidator',
    'ValidationRule',
    'ParameterConstraint',
    'validate_config'
]