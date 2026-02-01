"""
Training Configuration Manager

This module handles all training configuration management including
enhanced feature flags, parameter validation, and configuration inheritance.

Configuration Schema v2.0 - 8-Section Structure:
    1. model       - Architecture, MoE, tokens, regularization
    2. training    - Optimizer, schedule, batching, validation, generation
    3. data        - Loading, streaming, packing, splits
    4. compute     - Device, precision, CUDA, kernels, memory
    5. distributed - Multi-GPU, DeepSpeed, load balancing
    6. logging     - Console, WandB, TensorBoard, diagnostics
    7. checkpoints - Save/load, model selection
    8. experimental- FP8, MTP, caching, advanced features
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Set
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def get_mixed_precision(config: Dict[str, Any]) -> str:
    """
    Get mixed_precision setting with backward compatibility.

    Supports both:
    - training.precision.mixed_precision (new, recommended)
    - training.mixed_precision (legacy fallback)

    Args:
        config: Configuration dictionary

    Returns:
        Mixed precision mode: 'bf16', 'fp16', or 'fp32'
    """
    training = config.get('training', {})
    precision = training.get('precision', {})

    # Try nested path first (new structure)
    if 'mixed_precision' in precision:
        return precision['mixed_precision']

    # Fallback to legacy flat path
    return training.get('mixed_precision', 'bf16')


# Always import from centralized paths module
from ava.core.paths import get_project_root, get_data_dir, get_outputs_dir


# =============================================================================
# DYNAMIC CONFIG CLASS
# =============================================================================

class DynamicConfig:
    """
    Dynamic configuration class that accepts any fields from YAML.

    Provides both dictionary-style and attribute-style access to configuration values.
    Automatically converts nested dictionaries to nested DynamicConfig objects.

    Backward Compatibility:
        The config system has been reorganized from 38 sections into 8 categories.
        Old config paths (e.g., 'hardware.device') are automatically resolved to
        new paths (e.g., 'compute.device.type') with deprecation warnings.

    Example:
        config = DynamicConfig({'training': {'batch_size': 32}})
        config.training.batch_size  # Returns 32
        config['training']['batch_size']  # Also returns 32

    Strict Mode:
        By default, missing attributes return None. Enable strict mode to raise
        AttributeError for missing keys (helps catch typos):

        DynamicConfig.enable_strict_mode()
        config.typo_key  # Raises AttributeError instead of returning None
    """

    # Class-level settings for strict mode and tracking
    _strict_mode: bool = False
    _accessed_missing: Set[str] = set()
    _enable_path_migration: bool = True  # Enable deprecated path resolution

    @classmethod
    def enable_strict_mode(cls) -> None:
        """Enable strict mode - raises AttributeError for missing config keys."""
        cls._strict_mode = True
        logger.info("DynamicConfig strict mode enabled - missing keys will raise AttributeError")

    @classmethod
    def disable_strict_mode(cls) -> None:
        """Disable strict mode - missing keys return None (default behavior)."""
        cls._strict_mode = False
        logger.debug("DynamicConfig strict mode disabled")

    @classmethod
    def enable_path_migration(cls) -> None:
        """Enable deprecated path resolution (default behavior)."""
        cls._enable_path_migration = True

    @classmethod
    def disable_path_migration(cls) -> None:
        """Disable deprecated path resolution."""
        cls._enable_path_migration = False

    @classmethod
    def get_accessed_missing_keys(cls) -> Set[str]:
        """Get set of missing keys that were accessed."""
        return cls._accessed_missing.copy()

    @classmethod
    def clear_accessed_missing_keys(cls) -> None:
        """Clear the set of accessed missing keys."""
        cls._accessed_missing.clear()

    @classmethod
    def report_missing_keys(cls) -> None:
        """Log a warning about any missing keys that were accessed."""
        if cls._accessed_missing:
            keys_list = sorted(cls._accessed_missing)
            logger.warning(
                f"Config accessed {len(keys_list)} missing key(s): {keys_list[:10]}"
                + (f"... and {len(keys_list) - 10} more" if len(keys_list) > 10 else "")
            )

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """
        Initialize DynamicConfig from a dictionary.

        Args:
            data: Dictionary of configuration values
        """
        # Initialize instance attributes dict directly to avoid __setattr__ issues
        object.__setattr__(self, '_config_name', '')
        object.__setattr__(self, '_raw_data', data or {})  # Store raw data for path resolution
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    # Recursively convert nested dicts to DynamicConfig
                    nested = DynamicConfig(value)
                    object.__setattr__(nested, '_config_name', key)
                    setattr(self, key, nested)
                else:
                    setattr(self, key, value)

    def _try_resolve_deprecated_path(self, name: str) -> Optional[Any]:
        """
        Try to resolve a deprecated config path to its new location.

        This provides backward compatibility for the config reorganization.
        Old paths (e.g., 'hardware') are transparently resolved to new paths
        (e.g., 'compute.device') with a deprecation warning.

        Args:
            name: The attribute name being accessed

        Returns:
            The resolved value, or None if no mapping exists
        """
        if not DynamicConfig._enable_path_migration:
            return None

        try:
            from ava.config.path_mapping import (
                is_deprecated_path,
                get_new_path,
                warn_deprecated_path,
            )
        except ImportError:
            # path_mapping module not available yet
            return None

        # Build full path for checking
        config_name = object.__getattribute__(self, '_config_name')
        full_path = f"{config_name}.{name}" if config_name else name

        # Check if this is a deprecated top-level section
        if is_deprecated_path(name) or is_deprecated_path(full_path):
            new_path = get_new_path(name) or get_new_path(full_path)
            if new_path:
                warn_deprecated_path(full_path, new_path)
                # Try to resolve the new path
                return self._resolve_path(new_path)

        return None

    def _resolve_path(self, path: str) -> Optional[Any]:
        """
        Resolve a dot-separated path to its value.

        Args:
            path: Dot-separated path like 'compute.device.type'

        Returns:
            The value at that path, or None if not found
        """
        parts = path.split('.')
        current = self

        # Navigate to root if we're a nested config
        while hasattr(current, '_config_name'):
            parent = object.__getattribute__(current, '_config_name')
            if not parent:
                break
            # We can't navigate up, so start from root data
            raw_data = object.__getattribute__(self, '_raw_data')
            if raw_data:
                # Try to resolve from raw data
                result = raw_data
                for part in parts:
                    if isinstance(result, dict) and part in result:
                        result = result[part]
                    else:
                        return None
                return result
            break

        # Try to resolve from current position
        for part in parts:
            if current is None:
                return None
            if isinstance(current, DynamicConfig):
                if hasattr(current, part) and part in current.__dict__:
                    current = getattr(current, part)
                else:
                    return None
            elif isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    return None
            else:
                return None

        return current

    def __getattr__(self, name: str) -> Any:
        """
        Allow accessing any attribute dynamically.

        In default mode, returns None for missing attributes to allow safe access
        to optional config fields. In strict mode, raises AttributeError.

        Deprecated paths are automatically resolved to new paths with a warning.

        Args:
            name: Attribute name to access

        Returns:
            None if attribute not found (in non-strict mode)

        Raises:
            AttributeError: If strict mode enabled and attribute not found
        """
        # Avoid recursion for internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Try to resolve deprecated path first
        resolved = self._try_resolve_deprecated_path(name)
        if resolved is not None:
            return resolved

        # Track the missing key access
        config_name = object.__getattribute__(self, '_config_name') if hasattr(self, '_config_name') else ''
        full_key = f"{config_name}.{name}" if config_name else name
        DynamicConfig._accessed_missing.add(full_key)

        if DynamicConfig._strict_mode:
            raise AttributeError(
                f"Config key '{full_key}' not found (strict mode enabled). "
                f"Check for typos or add the key to your config file."
            )

        # Log at debug level for troubleshooting
        logger.debug(f"Config accessed missing key: {full_key}")
        return None

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting any attribute dynamically."""
        super().__setattr__(name, value)

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access: config['key']"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment: config['key'] = value"""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with a default fallback.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return getattr(self, key, default)

    def to_dict(self, _visited: Optional[set] = None) -> Dict[str, Any]:
        """
        Convert DynamicConfig back to a dictionary with circular reference protection.

        Args:
            _visited: Internal set to track visited objects (prevents infinite recursion)

        Returns:
            Dictionary representation of configuration
        """
        if _visited is None:
            _visited = set()

        # Check for circular reference
        obj_id = id(self)
        if obj_id in _visited:
            return {"_circular_reference": True}

        _visited.add(obj_id)

        result = {}
        for key, value in self.__dict__.items():
            # Skip internal attributes
            if key.startswith('_'):
                continue
            if isinstance(value, DynamicConfig):
                result[key] = value.to_dict(_visited)
            else:
                result[key] = value

        _visited.remove(obj_id)
        return result

    def validate(self) -> bool:
        """
        Validate the configuration for required fields and circular references.

        Returns:
            True if valid, raises ValueError if invalid
        """
        try:
            # Check for circular references by attempting to convert to dict
            self.to_dict()
            return True
        except RecursionError:
            raise ValueError("Configuration contains circular references")

    def validate_required_fields(self, required: List[str]) -> List[str]:
        """
        Issue #18 fix: Validate that required fields exist.

        Args:
            required: List of dot-notation paths like 'training.batch_size'

        Returns:
            List of missing field paths
        """
        missing = []
        for path in required:
            parts = path.split('.')
            current = self
            for part in parts:
                if not hasattr(current, part) or getattr(current, part) is None:
                    missing.append(path)
                    break
                current = getattr(current, part)
        return missing

    def validate_schema(self, schema: Dict[str, Any]) -> List[str]:
        """
        Issue #18 fix: Validate config against a schema definition.

        Args:
            schema: Dict with field names and expected types/constraints.
                    Each value can be:
                    - A type (e.g., int, str, float)
                    - A dict with 'type', 'required', 'min', 'max' keys

        Returns:
            List of validation error messages

        Example:
            errors = config.validate_schema({
                'training.batch_size': {'type': int, 'required': True, 'min': 1},
                'training.learning_rate': {'type': float, 'min': 0.0},
                'model.hidden_size': int,  # Simple type check
            })
        """
        errors = []
        for field, constraints in schema.items():
            # Get value using dot notation
            parts = field.split('.')
            value = self
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    value = None
                    break

            if isinstance(constraints, type):
                # Simple type check
                if value is not None and not isinstance(value, constraints):
                    errors.append(f"{field}: expected {constraints.__name__}, got {type(value).__name__}")
            elif isinstance(constraints, dict):
                # Complex constraint
                if 'required' in constraints and constraints['required'] and value is None:
                    errors.append(f"{field}: required field is missing")
                if value is not None:
                    if 'type' in constraints and not isinstance(value, constraints['type']):
                        errors.append(f"{field}: expected {constraints['type'].__name__}, got {type(value).__name__}")
                    if 'min' in constraints and value < constraints['min']:
                        errors.append(f"{field}: value {value} is below minimum {constraints['min']}")
                    if 'max' in constraints and value > constraints['max']:
                        errors.append(f"{field}: value {value} exceeds maximum {constraints['max']}")
        return errors

    def __repr__(self) -> str:
        """String representation of DynamicConfig"""
        return f"DynamicConfig({self.to_dict()})"


# =============================================================================
# SECTION 1: MODEL
# Architecture, MoE, tokens, regularization
# =============================================================================

# -----------------------------------------------------------------------------
# MoE Sub-Configurations (model.moe.*)
# -----------------------------------------------------------------------------

@dataclass
class MoEArchitectureConfig:
    """Configuration for MoE architecture settings.

    YAML Path: model.moe.architecture.*
    """
    num_experts: int = 4                      # Number of experts
    num_experts_per_token: int = 2            # Top-k experts per token
    router_type: str = 'mixtral'              # 'mixtral', 'deepseek', 'switch'
    capacity_factor: float = 1.25             # Expert capacity factor (1.0-2.0)
    expert_dropout: float = 0.0               # Expert dropout rate

    def __post_init__(self):
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}")
        if self.num_experts_per_token <= 0:
            raise ValueError(f"num_experts_per_token must be > 0, got {self.num_experts_per_token}")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError(
                f"num_experts_per_token ({self.num_experts_per_token}) cannot exceed "
                f"num_experts ({self.num_experts})"
            )
        if self.capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {self.capacity_factor}")


@dataclass
class MoELossesConfig:
    """Configuration for MoE auxiliary losses.

    YAML Path: model.moe.losses.*
    """
    router_z_loss_coef: float = 0.001         # Router z-loss coefficient
    load_balance_loss_coef: float = 0.01      # Load balance loss coefficient
    diversity_loss_coef: float = 0.001        # Diversity loss coefficient
    expert_dropout_loss_coef: float = 0.0     # Expert dropout loss coefficient
    router_jitter_noise: float = 0.01         # Router jitter noise for exploration
    aux_loss_frequency: int = 500             # Compute aux losses every N steps
    diversity_loss_frequency: int = 100       # Compute diversity loss every N steps

    def __post_init__(self):
        if self.router_z_loss_coef < 0:
            raise ValueError(f"router_z_loss_coef cannot be negative, got {self.router_z_loss_coef}")
        if self.load_balance_loss_coef < 0:
            raise ValueError(f"load_balance_loss_coef cannot be negative, got {self.load_balance_loss_coef}")
        if self.diversity_loss_coef < 0:
            raise ValueError(f"diversity_loss_coef cannot be negative, got {self.diversity_loss_coef}")


@dataclass
class MoEOptimizationConfig:
    """Configuration for MoE optimization flags.

    YAML Path: model.moe.optimization.*
    """
    use_optimized_moe: bool = True            # Use optimized MoE implementation
    use_grouped_gemm: bool = True             # Use grouped GEMM kernels for experts
    use_compile_friendly_dispatch: bool = True  # Use compile-friendly expert dispatch
    use_fused_moe_kernel: bool = False        # Mega kernel for MoE (advanced)
    use_sparse_expert_dispatch: bool = False  # Sparse dispatch for efficiency
    use_selective_expert_loading: bool = False  # Load experts on demand
    use_vectorized_capacity: bool = True      # Vectorized capacity limiting
    use_fused_softmax_topk: bool = True       # Fused softmax + top-k kernel
    router_kernel_mode: str = 'auto'          # 'auto', 'triton', 'pytorch'
    router_block_size: int = 4                # Tokens per thread block


@dataclass
class MoEPrefetchConfig:
    """Configuration for MoE expert prefetching.

    YAML Path: model.moe.prefetch.*
    """
    enabled: bool = True                      # Enable expert prefetching
    lookahead: int = 2                        # Number of experts to prefetch ahead
    use_multiple_streams: bool = True         # Use multiple CUDA streams


@dataclass
class MoECacheConfig:
    """Configuration for MoE expert caching.

    YAML Path: model.moe.cache.*
    """
    auto_limit: bool = True                   # Auto-limit cache size
    use_lru_eviction: bool = True             # Use LRU eviction policy
    clear_after_optimizer_step: bool = True   # Clear cache after optimizer step


@dataclass
class MoERouterCompileConfig:
    """Configuration for MoE router compilation.

    YAML Path: model.moe.router_compile.*
    """
    cache_hash_on_gpu: bool = True            # Cache hash computation on GPU
    compile_routers: bool = True              # Compile routers with torch.compile
    compile_mode: str = 'default'             # 'default', 'reduce-overhead', 'max-autotune'
    compile_dynamic: bool = True              # Allow dynamic shapes


@dataclass
class MoEMetricsConfig:
    """Configuration for MoE metrics tracking.

    YAML Path: model.moe.metrics.*
    """
    routing_metrics_freq: int = 100           # Routing metrics logging frequency
    moe_metrics_freq: int = 5000              # MoE metrics logging frequency
    log_moe: bool = True                      # Enable MoE logging
    log_routing_diagnostics: bool = False     # Log detailed routing diagnostics
    track_expert_utilization: bool = True     # Track expert utilization
    track_routing_decisions: bool = False     # Track routing decisions
    track_load_balance: bool = True           # Track load balance


@dataclass
class StableMoEAdaptiveConfig:
    """Configuration for Stable-MoE adaptive routing.

    YAML Path: model.moe.stable_moe.*
    """
    enabled: bool = False                     # Enable stable MoE
    target_utilization: float = 0.0           # Target expert utilization
    capacity_min: float = 1.0                 # Minimum capacity factor
    capacity_max: float = 2.0                 # Maximum capacity factor
    utilization_tolerance: float = 0.1        # Tolerance for utilization
    adaptation_rate: float = 0.01             # Adaptation rate
    temperature_init: float = 1.0             # Initial temperature
    temperature_min: float = 0.1              # Minimum temperature
    temperature_decay: float = 0.9999         # Temperature decay rate
    log_utilization_histogram: bool = True    # Log utilization histogram
    log_capacity_factors: bool = True         # Log capacity factors
    log_temperature: bool = True              # Log temperature


@dataclass
class MoEConfig:
    """Composite configuration for all MoE settings.

    YAML Path: model.moe.*

    This consolidates all MoE-related configuration into a single nested structure.
    """
    architecture: MoEArchitectureConfig = field(default_factory=MoEArchitectureConfig)
    losses: MoELossesConfig = field(default_factory=MoELossesConfig)
    optimization: MoEOptimizationConfig = field(default_factory=MoEOptimizationConfig)
    prefetch: MoEPrefetchConfig = field(default_factory=MoEPrefetchConfig)
    cache: MoECacheConfig = field(default_factory=MoECacheConfig)
    router_compile: MoERouterCompileConfig = field(default_factory=MoERouterCompileConfig)
    metrics: MoEMetricsConfig = field(default_factory=MoEMetricsConfig)
    stable_moe: StableMoEAdaptiveConfig = field(default_factory=StableMoEAdaptiveConfig)


# -----------------------------------------------------------------------------
# Main Model Configuration
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for model architecture and optimizations.

    YAML Path: model.*

    This dataclass maps to the 'model' section in YAML configs and includes:
    - Core architecture (vocab, hidden size, layers, attention heads)
    - MoE settings (experts, routing, capacity)
    - Performance optimizations (flash attention, GEMM, Triton)
    - Auxiliary losses (router z-loss, load balance, diversity)
    - Regularization (dropout, layer norm)
    """
    # Architecture
    vocab_size: int = 50680
    hidden_size: int = 1024
    num_layers: int = 6
    num_attention_heads: int = 16
    intermediate_size: int = 8192
    max_position_embeddings: int = 512

    # MoE settings
    num_experts: int = 4
    num_experts_per_token: int = 1
    router_type: str = 'mixtral'              # 'mixtral' or 'deepseek'
    capacity_factor: float = 1.25
    expert_dropout: float = 0.0
    activation: str = 'swiglu'                # 'swiglu', 'geglu', 'gelu', 'relu'

    # Performance optimizations
    use_grouped_gemm: bool = True             # Use grouped GEMM kernels for experts
    use_triton_kernels: bool = True           # Use Triton fused kernels
    torch_compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    torch_compile_dynamic: bool = False       # Allow dynamic shapes (slower but flexible)
    torch_compile_fullgraph: bool = False     # Require full graph (faster but stricter)
    use_compile_friendly_dispatch: bool = True  # Use compile-friendly expert dispatch
    enable_cudagraphs_safe_routing: bool = False  # Enable CUDA graphs safe routing (deprecated)
    use_flash_attention: bool = True          # Use flash attention
    gradient_checkpointing: bool = True       # Enable gradient checkpointing
    use_optimized_moe: bool = True            # Use optimized MoE implementation
    quantize_kv_cache: bool = False           # Enable KV cache quantization for memory savings

    # Phase 1 Optimizations (low-risk, high-impact)
    use_fused_qkv: bool = True                # Fused Q/K/V projection (5-10% attention speedup)
    use_fused_norm: bool = True               # Fused LayerNorm + residual (8-15% per layer)
    diversity_loss_frequency: int = 100       # Compute diversity loss every N steps (was 10)

    # Auxiliary losses
    router_z_loss_coef: float = 0.0001        # Router z-loss coefficient
    load_balance_loss_coef: float = 0.01      # Load balance loss coefficient
    diversity_loss_coef: float = 0.0001       # Diversity loss coefficient
    expert_dropout_loss_coef: float = 0.0     # Expert dropout loss coefficient
    router_jitter_noise: float = 0.01         # Router jitter noise for exploration
    aux_loss_frequency: int = 500             # Compute aux losses every N steps

    # Regularization
    attention_dropout: float = 0.0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.01

    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2

    # Positional encoding
    use_alibi: bool = False                   # Use ALiBi positional encoding instead of absolute
    rope_theta: float = 10000.0               # RoPE base theta for rotary positional embeddings
    rope_scaling: Optional[Dict[str, float]] = None  # RoPE scaling configuration

    # Consolidated MoE Configuration (model.moe.*)
    # This is the new canonical location for all MoE settings
    # The flat fields above are kept for backward compatibility
    moe: Optional[MoEConfig] = None

    def __post_init__(self):
        """Validate configuration values to catch invalid configs early."""
        # Initialize MoE config if not provided
        if self.moe is None:
            self.moe = MoEConfig()
        # Basic positive value checks
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be > 0, got {self.intermediate_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be > 0, got {self.num_attention_heads}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}")

        # Divisibility check
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # MoE-specific validations
        if self.num_experts_per_token > self.num_experts:
            raise ValueError(
                f"num_experts_per_token ({self.num_experts_per_token}) cannot exceed "
                f"num_experts ({self.num_experts})"
            )
        if self.num_experts_per_token <= 0:
            raise ValueError(f"num_experts_per_token must be > 0, got {self.num_experts_per_token}")
        if self.capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {self.capacity_factor}")

        # Auxiliary loss coefficient validations (must be non-negative)
        if self.router_z_loss_coef < 0:
            raise ValueError(f"router_z_loss_coef cannot be negative, got {self.router_z_loss_coef}")
        if self.load_balance_loss_coef < 0:
            raise ValueError(f"load_balance_loss_coef cannot be negative, got {self.load_balance_loss_coef}")
        if self.diversity_loss_coef < 0:
            raise ValueError(f"diversity_loss_coef cannot be negative, got {self.diversity_loss_coef}")

        # torch.compile configuration validation
        valid_compile_modes = {'default', 'reduce-overhead', 'max-autotune'}
        if self.torch_compile_mode not in valid_compile_modes:
            raise ValueError(
                f"torch_compile_mode must be one of {valid_compile_modes}, "
                f"got '{self.torch_compile_mode}'"
            )


# =============================================================================
# SECTION 2: TRAINING
# Optimizer, schedule, batching, validation, generation, coherence
# =============================================================================

@dataclass
class PrecisionConfig:
    """Configuration for training precision.

    YAML Path: training.precision.*
    """
    mixed_precision: str = 'bf16'             # 'fp32', 'fp16', 'bf16'


@dataclass
class BatchingConfig:
    """Configuration for batching parameters.

    YAML Path: training.batching.*
    """
    batch_size: int = 32                      # Per-GPU batch size
    gradient_accumulation_steps: int = 1      # Gradient accumulation steps


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings.

    YAML Path: training.optimizer.*
    """
    type: str = 'adamw'                       # 'adamw', 'adam', 'sgd', 'lion'
    learning_rate: float = 0.0001             # Learning rate
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01                # Weight decay
    max_grad_norm: float = 1.0                # Maximum gradient norm for clipping
    use_fused: bool = True                    # Use fused optimizer (5-10% speedup)


@dataclass
class ScheduleConfig:
    """Configuration for learning rate schedule.

    YAML Path: training.schedule.*
    """
    num_epochs: int = 3                       # Number of training epochs
    max_steps: Optional[int] = None           # Maximum training steps (overrides epochs)
    steps_per_epoch: Optional[int] = None     # Steps per epoch for scheduler
    warmup_steps: int = 2000                  # Number of warmup steps
    scheduler_type: str = 'cosine'            # 'cosine', 'linear', 'constant'
    num_cycles: int = 1                       # Number of cosine cycles
    min_lr: float = 0.0                       # Minimum learning rate


@dataclass
class ValidationConfig:
    """Configuration for validation during training.

    YAML Path: training.validation.*
    """
    enabled: bool = True                      # Enable validation
    batch_size: int = 16                      # Validation batch size
    max_batches: int = 20                     # Maximum validation batches
    compute_train_ratio: bool = True          # Compute train/val loss ratio


@dataclass
class GenerationConfig:
    """Configuration for text generation during training.

    YAML Path: training.generation.*
    """
    enabled: bool = True                      # Enable generation
    generate_every_n_steps: int = 1000        # Generation frequency
    num_per_step: int = 1                     # Samples per generation step
    num_return_sequences: int = 1             # Sequences per prompt
    during_eval: bool = True                  # Generate during evaluation
    test_quality: bool = True                 # Test generation quality
    max_length: int = 512                     # Maximum generation length
    min_length: int = 10                      # Minimum generation length
    temperature: float = 0.8                  # Sampling temperature
    top_p: float = 0.9                        # Nucleus sampling threshold
    top_k: Optional[int] = 50                 # Top-k sampling
    repetition_penalty: float = 1.2           # Repetition penalty
    no_repeat_ngram_size: int = 3             # Block n-gram repetitions
    do_sample: bool = True                    # Enable sampling
    num_beams: int = 1                        # Beam search width
    early_stopping: bool = False              # Stop when all beams finish
    skip_special_tokens: bool = True          # Skip special tokens in output
    prompt: str = "Once upon a time"          # Default generation prompt
    alternative_prompts: List[str] = field(default_factory=list)


@dataclass
class CoherenceConfig:
    """Configuration for coherence measurement during training.

    YAML Path: training.coherence.*
    """
    enabled: bool = True                      # Enable coherence measurement
    eval_every_n_steps: int = 500             # Measure coherence every N steps
    num_samples: int = 10                     # Number of samples for evaluation
    max_generation_length: int = 256          # Max tokens to generate

    # Metric weights for aggregate score
    perplexity_weight: float = 0.3
    repetition_weight: float = 0.2
    flow_weight: float = 0.25
    topic_weight: float = 0.25

    # Thresholds
    max_perplexity: float = 100.0             # Cap perplexity for scoring
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    min_sentence_length: int = 5              # Min tokens for a sentence

    # Generation parameters
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50

    # Logging
    log_to_wandb: bool = True
    log_to_console: bool = True
    use_fast: bool = True
    use_bf16: bool = True
    micro_batch_size: int = 16

    def __post_init__(self):
        """Validate configuration values."""
        weights_sum = (
            self.perplexity_weight +
            self.repetition_weight +
            self.flow_weight +
            self.topic_weight
        )
        if not (0.99 <= weights_sum <= 1.01):
            logger.warning(
                f"CoherenceConfig weights sum to {weights_sum:.4f}, not 1.0. "
                f"Coherence scores may be outside [0,1] range."
            )


@dataclass
class TrainingLoggingConfig:
    """Configuration for training-specific logging settings.

    YAML Path: training.logging.*
    """
    save_steps: int = 500                     # Save checkpoint every N steps
    eval_steps: int = 500                     # Evaluate every N steps
    logging_steps: int = 10                   # Log metrics every N steps
    use_tensorboard: bool = False             # Enable TensorBoard
    use_wandb: bool = True                    # Enable WandB


@dataclass
class ProgressiveTrainingConfig:
    """Configuration for progressive training features.

    YAML Path: training.progressive.*
    """
    enable_progressive_training: bool = False    # Enable progressive training

    # Sequence length scaling
    enable_sequence_scaling: bool = False
    initial_seq_length: int = 128
    final_seq_length: int = 2048
    length_schedule: str = "linear"
    length_growth_epochs: int = 10
    enable_length_bucketing: bool = False

    # Difficulty scoring
    enable_curriculum: bool = False
    curriculum_metric: str = "loss"
    enable_score_caching: bool = False
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".cache" / "ava_difficulty"))
    cache_version: str = "v1.0"


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive learning rate settings.

    YAML Path: training.adaptive_lr.*
    """
    enabled: bool = False
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    patience: int = 5
    factor: float = 0.5


@dataclass
class TrainingSectionConfig:
    """Composite configuration for the training section.

    YAML Path: training.*

    Contains all training-related sub-configs organized to match YAML structure.
    """
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    coherence: CoherenceConfig = field(default_factory=CoherenceConfig)
    logging: TrainingLoggingConfig = field(default_factory=TrainingLoggingConfig)
    progressive: ProgressiveTrainingConfig = field(default_factory=ProgressiveTrainingConfig)
    adaptive_lr: AdaptiveLRConfig = field(default_factory=AdaptiveLRConfig)

    # Legacy flat fields for backward compatibility
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    gradient_accumulation: int = 1
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: Optional[int] = None

    def __post_init__(self):
        """Validate training configuration values."""
        # Batch size validation (if specified)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        # Gradient accumulation must be positive
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}")

        # Warmup steps must be non-negative
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps cannot be negative, got {self.warmup_steps}")


# Legacy alias for backward compatibility
TrainingConfig = TrainingSectionConfig


# =============================================================================
# SECTION 3: DATA
# Loading, streaming, packing, splits
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data handling.

    YAML Path: data.*
    """
    data_dir: str = field(default_factory=lambda: str(get_data_dir("processed")))
    max_length: int = 512                     # Max sequence length
    tokenizer_name: Optional[str] = None      # Tokenizer name or path
    max_samples: Optional[int] = None         # Max samples (testing)
    streaming: bool = False                   # Streaming loader
    buffer_size: int = 50000                  # Streaming buffer size
    num_workers: int = 0                      # Data loading workers
    prefetch_factor: int = 4                  # Batches to prefetch per worker
    persistent_workers: bool = True           # Keep workers alive between epochs
    padding_side: str = 'right'               # Tokenizer padding side
    truncation: bool = True                   # Enable truncation
    max_train_examples: Optional[int] = None  # Max training examples
    max_eval_examples: Optional[int] = None   # Max evaluation examples
    dataloader_drop_last: bool = False        # Drop last incomplete batch
    dataloader_pin_memory: bool = True        # Pin memory for faster GPU transfer
    default_tokenizer_name: str = 'Qwen/Qwen2.5-0.5B'  # Default tokenizer

    # Sequence packing
    use_sequence_packing: bool = False        # Enable sequence packing
    packing_strategy: str = 'greedy'          # 'greedy' or 'adaptive'
    max_tokens_per_batch: Optional[int] = None
    packing_target_ratio: float = 0.95

    # Dataset splits
    train_split: str = 'train'
    eval_split: str = 'validation'
    auto_create_validation_split: bool = True
    validation_split_ratio: float = 0.1
    val_split_ratio: float = 0.1
    val_max_samples: Optional[int] = None

    # Additional settings
    dataloader_persistent_workers: bool = False
    dataloader_samples_per_file: int = 64
    samples_per_file: int = 64
    use_streaming_tokenization: bool = False
    enable_bucketing: bool = True
    dataset_name: Optional[str] = None

    # Randomization control
    shuffle_seed: Optional[int] = None
    enable_length_sorting: bool = True
    disable_packing_length_sort: bool = False
    examples_per_random_select: int = 100

    # Indexed loader
    use_indexed_loader: bool = False
    indexed_num_bins: int = 8
    indexed_cache_size: int = 50
    indexed_index_workers: Optional[int] = None

    # DataLoader worker settings
    worker_timeout: float = 300.0             # Training worker timeout in seconds
    val_num_workers: Optional[int] = None     # Validation workers (None = min(2, num_workers))
    val_timeout: float = 0.0                  # Validation timeout (0 = disabled, prevents spurious timeouts)
    val_persistent_workers: bool = False      # Validation persistent workers (False reduces resource contention)

    # Fast startup options
    fast_startup: bool = False
    skip_sequence_count: bool = True
    skip_dataloader_validation: bool = True


@dataclass
class MultiColumnDataConfig:
    """Configuration for multi-column data.

    YAML Path: data.multi_column.* (or legacy multi_column_data.*)
    """
    use_multi_column: bool = False
    dataset_config: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_dataset_config: Optional[str] = None
    column_names: Optional[str] = None
    column_types: Optional[str] = None
    column_roles: Optional[str] = None
    combine_strategy: str = 'concatenate'
    column_template: Optional[str] = None


# =============================================================================
# SECTION 4: COMPUTE
# Device, precision, CUDA, kernels, memory
# =============================================================================

@dataclass
class DeviceConfig:
    """Configuration for device settings.

    YAML Path: compute.device.*
    """
    type: str = 'cuda'                        # 'cuda', 'cpu', or 'mps'
    compile: bool = False                     # Enable torch.compile
    num_gpus: int = 1                         # Number of GPUs


@dataclass
class MemoryConfig:
    """Configuration for memory management.

    YAML Path: compute.memory.*
    """
    headroom_gb: float = 3.0                  # Reserved memory headroom
    cleanup_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'warning': 0.85,
        'critical': 0.90,
        'emergency': 0.95
    })
    proactive_cleanup_enabled: bool = True
    fragmentation_threshold: float = 0.30
    cleanup_frequency: int = 2000
    cleanup_after_validation: bool = True
    cleanup_after_generation: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance modes and hardware optimizations.

    YAML Path: compute.performance.*
    """
    ultra_fast_mode: bool = False
    fast_progress: bool = False
    minimal_progress: bool = False
    no_sync: bool = False
    express_mode: bool = False

    # TF32 and hardware optimizations
    enable_tf32: bool = True                  # Enable TF32 on Ampere+ GPUs
    float32_matmul_precision: str = 'high'    # 'highest', 'high', 'medium'
    enable_cudnn_benchmark: bool = True       # Auto-tune cuDNN kernels
    cudagraph_skip_dynamic_shapes: bool = True
    cudagraph_dynamic_shape_warn_limit: Optional[int] = None
    torchinductor_max_autotune: int = 0       # TorchInductor autotune level


@dataclass
class KernelsConfig:
    """Configuration for low-level kernel optimizations.

    YAML Path: compute.kernels.*
    """
    router_kernel_mode: str = 'auto'          # 'auto', 'triton', 'pytorch'
    use_fused_softmax_topk: bool = True       # Fused softmax + top-k kernel
    router_block_size: int = 4                # Tokens per thread block

    # Expert computation optimizations
    use_sparse_expert_dispatch: bool = False
    use_fused_activations: bool = True        # Fused SwiGLU/GeGLU kernels
    use_selective_expert_loading: bool = False

    # Capacity limiting
    use_vectorized_capacity: bool = True

    # Advanced optimizations
    use_fused_moe_kernel: bool = False        # Mega kernel
    enable_kernel_profiling: bool = False


@dataclass
class CudaStreamsConfig:
    """Configuration for CUDA stream optimizations.

    YAML Path: compute.cuda.streams.*
    """
    enabled: bool = False
    num_streams: int = 4
    use_event_timing: bool = True
    use_stream_pool: bool = True
    high_priority_transfers: bool = True


@dataclass
class CudaGraphsConfig:
    """Configuration for CUDA graph capture.

    YAML Path: compute.cuda.graphs.*
    """
    enabled: bool = False
    capture_backward: bool = True
    capture_optimizer_step: bool = True
    max_cached_graphs: int = 4
    use_memory_pool: bool = True
    warmup_steps: int = 3


@dataclass
class CudaConfig:
    """Composite configuration for CUDA settings.

    YAML Path: compute.cuda.*
    """
    streams: CudaStreamsConfig = field(default_factory=CudaStreamsConfig)
    graphs: CudaGraphsConfig = field(default_factory=CudaGraphsConfig)


@dataclass
class CalibrationConfig:
    """Configuration for enhanced calibration system.

    YAML Path: compute.calibration.*
    """
    enabled: bool = True
    run_memory_profiling: bool = True
    run_backward_profiling: bool = True
    run_throughput_profiling: bool = True
    batch_sizes_to_profile: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])
    seq_lengths_to_profile: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048])
    num_samples_per_config: int = 3
    cache_enabled: bool = True
    cache_dir: str = "~/.cache/ava_calibration"
    save_to_run_dir: bool = True
    cache_ttl_hours: int = 168
    max_calibration_time_seconds: int = 300


@dataclass
class ComputeSectionConfig:
    """Composite configuration for the compute section.

    YAML Path: compute.*
    """
    device: DeviceConfig = field(default_factory=DeviceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    kernels: KernelsConfig = field(default_factory=KernelsConfig)
    cuda: CudaConfig = field(default_factory=CudaConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


# Legacy aliases for backward compatibility
HardwareConfig = DeviceConfig
KernelOptimizationConfig = KernelsConfig
CUDAGraphConfig = CudaGraphsConfig


# =============================================================================
# SECTION 5: DISTRIBUTED
# Multi-GPU, DeepSpeed, load balancing
# =============================================================================

@dataclass
class GPULoadBalancingConfig:
    """Configuration for GPU load balancing.

    YAML Path: distributed.load_balancing.*
    """
    enabled: bool = False
    strategy: str = 'adaptive'                # 'round_robin', 'memory_aware', 'compute_aware', 'adaptive'
    rebalance_interval: int = 1000
    enable_expert_migration: bool = True
    migration_threshold: float = 0.2
    log_gpu_metrics: bool = True


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed distributed training.

    YAML Path: distributed.deepspeed.*
    """
    use_deepspeed: bool = False
    config_file: Optional[str] = None
    zero_stage: int = 2                       # ZeRO stage (0, 1, 2, 3)
    cpu_offload: bool = False
    nvme_offload: bool = False
    gradient_accumulation_steps: int = 1
    train_batch_size: Optional[int] = None
    micro_batch_size: Optional[int] = None
    enable_mixed_precision: bool = False
    precision_type: str = 'fp16'              # 'fp16', 'bf16', 'fp32'

    # ZeRO-specific settings
    zero_allow_untested_optimizer: bool = False
    zero_force_ds_cpu_optimizer: bool = False
    zero_reduce_scatter: bool = False
    zero_overlap_comm: bool = False
    zero_contiguous_gradients: bool = False
    zero_reduce_bucket_size: int = 500000000
    zero_allgather_bucket_size: int = 500000000
    zero_stage3_prefetch_bucket_size: int = 500000000
    zero_stage3_param_persistence_threshold: int = 1000000

    # Communication settings
    communication_data_type: str = 'fp32'
    allreduce_partitions: bool = False
    allgather_partitions: bool = False
    overlap_comm: bool = False
    wall_clock_breakdown: bool = False

    # Advanced features
    activation_checkpointing: bool = False
    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = False
    synchronize_dp_processes: bool = False

    # Pipeline parallelism
    pipeline_parallel_size: int = 1
    gradient_clipping: Optional[float] = None

    # Monitoring
    monitor_config: Dict[str, Any] = field(default_factory=dict)
    tensorboard: Dict[str, Any] = field(default_factory=dict)
    wandb_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradientSyncConfig:
    """Configuration for distributed gradient synchronization.

    YAML Path: distributed.gradient_sync.*
    """
    mode: str = 'standard'                    # 'standard', 'overlapped', 'fused'
    fused_bucket_mb: float = 100.0
    fusion_factor: int = 4
    async_allreduce: bool = True
    bucket_size_mb: float = 25.0
    use_coalesced_ops: bool = True


@dataclass
class LayerwiseOptimizerConfig:
    """Configuration for layer-wise optimizer updates.

    YAML Path: distributed.layerwise_optimizer.*
    """
    enabled: bool = False
    bucket_size_mb: float = 25.0
    release_gradients_early: bool = True
    align_with_ddp_buckets: bool = True


@dataclass
class DistributedSectionConfig:
    """Composite configuration for the distributed section.

    YAML Path: distributed.*
    """
    load_balancing: GPULoadBalancingConfig = field(default_factory=GPULoadBalancingConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    gradient_sync: GradientSyncConfig = field(default_factory=GradientSyncConfig)
    layerwise_optimizer: LayerwiseOptimizerConfig = field(default_factory=LayerwiseOptimizerConfig)

    # Barrier optimization
    minimize_distributed_barriers: bool = True
    sync_loss_across_ranks: bool = False


# =============================================================================
# SECTION 6: LOGGING
# Console, WandB, TensorBoard, diagnostics
# =============================================================================

@dataclass
class ConsoleLoggingConfig:
    """Configuration for console logging.

    YAML Path: logging.console.*
    """
    enabled: bool = True
    level: str = 'info'                       # 'debug', 'info', 'warning', 'error'
    progress_bar_enabled: bool = True
    tqdm_update_interval: int = 10
    step_logging_enabled: bool = True
    epoch_logging_enabled: bool = True


@dataclass
class FileLoggingConfig:
    """Configuration for file logging.

    YAML Path: logging.file.*
    """
    enabled: bool = True
    level: str = 'debug'
    format: str = 'default'                   # 'default', 'json', 'structured'


@dataclass
class MetricsLoggingConfig:
    """Configuration for metrics logging categories.

    YAML Path: logging.metrics.*
    """
    log_training_metrics: bool = True
    log_gradient_metrics: bool = True
    log_moe_metrics: bool = True
    log_expert_metrics: bool = True  # Per-expert utilization, load, capacity
    log_validation_metrics: bool = True
    log_coherence_metrics: bool = True
    log_quality_metrics: bool = True
    log_memory_metrics: bool = True
    log_timing_metrics: bool = True
    log_generation_samples: bool = True


@dataclass
class WandBLoggingConfig:
    """Configuration for Weights & Biases logging.

    YAML Path: logging.wandb.*
    """
    enabled: bool = True
    project: str = 'Ava'
    name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ['moe', 'training'])
    log_freq: int = 10
    offline: bool = False
    cache_size: int = 2000
    cache_flush_interval: int = 50

    # Fine-grained logging controls
    log_training: bool = True
    log_gradients: bool = True
    log_moe: bool = True
    log_experts: bool = True  # Per-expert metrics (utilization, load, capacity)
    log_validation: bool = True
    log_coherence: bool = True
    log_quality: bool = True
    log_memory: bool = True
    log_timing: bool = True
    log_generations: bool = True
    log_per_layer_grads: bool = False
    log_routing_diagnostics: bool = False
    log_model_topology: bool = False


@dataclass
class TensorBoardLoggingConfig:
    """Configuration for TensorBoard logging.

    YAML Path: logging.tensorboard.*
    """
    enabled: bool = False
    log_training: bool = True
    log_gradients: bool = True
    log_moe: bool = True
    log_experts: bool = True  # Per-expert metrics (utilization, load, capacity)
    log_validation: bool = True


@dataclass
class FrequenciesConfig:
    """Configuration for logging frequencies.

    YAML Path: logging.frequencies.*
    """
    log_interval: int = 100
    verbose_log_interval: int = 500
    metrics_log_freq: int = 500
    memory_check_freq: int = 2000
    health_summary_freq: int = 500
    moe_metrics_freq: int = 5000
    routing_metrics_freq: int = 0


@dataclass
class FeaturesLoggingConfig:
    """Configuration for logging feature flags.

    YAML Path: logging.features.*
    """
    enable_timing_breakdown: bool = True
    enable_memory_profiling: bool = True
    enable_health_summaries: bool = True
    log_tensor_shapes: bool = True
    log_checkpoint_validation: bool = True
    save_sample_generations: bool = True
    use_structured_logging: bool = True


@dataclass
class DiagnosticsConfig:
    """Configuration for detailed diagnostic logging.

    YAML Path: logging.diagnostics.*
    """
    enabled: bool = False

    # Per-layer gradient statistics
    enable_per_layer_gradients: bool = False
    per_layer_log_freq: int = 500
    layer_name_patterns: List[str] = field(default_factory=lambda: ['layers', 'experts'])

    # Expert routing diagnostics
    enable_routing_diagnostics: bool = False
    routing_log_freq: int = 100
    track_per_expert_load: bool = True
    track_routing_entropy: bool = True
    track_expert_capacity_usage: bool = True

    # Memory breakdown
    enable_memory_breakdown: bool = False
    memory_log_freq: int = 500
    track_activation_memory: bool = True
    track_gradient_memory: bool = True
    track_optimizer_state_memory: bool = True
    track_parameter_memory: bool = True

    # Timing profiling
    enable_timing_profiling: bool = False
    timing_log_freq: int = 500
    profile_forward: bool = True
    profile_backward: bool = True
    profile_optimizer_step: bool = True
    profile_data_loading: bool = True


@dataclass
class DevLogConfig:
    """Configuration for development logging.

    YAML Path: logging.dev.*
    """
    enabled: bool = False
    show_file_timings: bool = True
    show_batch_timings: bool = True
    show_step_breakdown: bool = True
    report_interval: int = 100


@dataclass
class LoggingSectionConfig:
    """Composite configuration for the logging section.

    YAML Path: logging.*
    """
    console: ConsoleLoggingConfig = field(default_factory=ConsoleLoggingConfig)
    file: FileLoggingConfig = field(default_factory=FileLoggingConfig)
    metrics: MetricsLoggingConfig = field(default_factory=MetricsLoggingConfig)
    wandb: WandBLoggingConfig = field(default_factory=WandBLoggingConfig)
    tensorboard: TensorBoardLoggingConfig = field(default_factory=TensorBoardLoggingConfig)
    frequencies: FrequenciesConfig = field(default_factory=FrequenciesConfig)
    features: FeaturesLoggingConfig = field(default_factory=FeaturesLoggingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    dev: DevLogConfig = field(default_factory=DevLogConfig)

    # Master disable flags
    disabled: bool = False
    silent_mode: bool = False
    log_mode: str = 'tqdm'                    # 'tqdm' or 'verbose'
    verbosity: str = 'info'


# Legacy alias - keep LoggingConfig as an alias
LoggingConfig = LoggingSectionConfig


# =============================================================================
# SECTION 7: CHECKPOINTS
# Save/load, model selection
# =============================================================================

@dataclass
class SelectionConfig:
    """Configuration for multi-metric model selection.

    YAML Path: checkpoints.selection.*
    """
    enabled: bool = True
    val_loss_weight: float = 0.5
    coherence_score_weight: float = 0.3
    perplexity_weight: float = 0.2
    perplexity_cap: float = 100.0
    val_loss_cap: float = 10.0
    higher_is_better: bool = True
    require_all_metrics: bool = False
    fallback_to_val_loss: bool = True


@dataclass
class CheckpointsSectionConfig:
    """Composite configuration for the checkpoints section.

    YAML Path: checkpoints.*
    """
    output_dir: str = field(default_factory=lambda: str(get_outputs_dir()))
    save_every: int = 500                     # Save frequency in steps
    resume: Optional[str] = None              # Resume checkpoint path
    fresh_start: bool = False                 # Force fresh start
    async_saving: bool = True                 # Save checkpoints in background
    selection: SelectionConfig = field(default_factory=SelectionConfig)


# Legacy aliases
ModelSelectionConfig = SelectionConfig
OutputConfig = CheckpointsSectionConfig


# =============================================================================
# SECTION 8: EXPERIMENTAL
# FP8, MTP, caching, advanced features
# =============================================================================

@dataclass
class FP8Config:
    """Configuration for FP8 training (Hopper/Ada GPUs only).

    YAML Path: experimental.fp8.*
    """
    enabled: bool = False
    use_transformer_engine: bool = False
    format: str = 'e4m3'                      # 'e4m3' or 'e5m2'
    margin: int = 0
    backward_enabled: bool = False
    backward_format: str = 'e5m2'
    exclude_layers: List[str] = field(default_factory=lambda: ['embedding', 'lm_head'])
    gradient_scaling_strategy: str = 'per_tensor'
    amax_history_len: int = 1024
    amax_compute_algo: str = 'max'


@dataclass
class MTPConfig:
    """Configuration for Adaptive Multi-Token Prediction.

    YAML Path: experimental.mtp.*
    """
    enabled: bool = False
    num_prediction_heads: int = 3
    confidence_threshold_train: float = 0.6
    confidence_threshold_inference: float = 0.7
    gate_hidden_dims: str = "512,256"
    gate_dropout: float = 0.1
    gate_activation: str = 'gelu'
    use_attention_pooling: bool = False
    head_type: str = 'linear'
    head_intermediate_size: Optional[int] = None
    head_dropout: float = 0.1
    share_projections: bool = False
    mtp_warmup_epochs: int = 2
    confidence_reg_strength: float = 0.01
    use_confidence_weighting: bool = False
    primary_loss_weight: float = 1.0
    additional_loss_base_weight: float = 0.1
    enable_dynamic_prediction: bool = False
    min_confidence_for_computation: float = 0.3


@dataclass
class HybridCachingConfig:
    """Configuration for hybrid KV + activation caching.

    YAML Path: experimental.caching.*
    """
    enabled: bool = False
    max_cache_size_gb: float = 1.0
    kv_cache_ratio: float = 0.7
    eviction_policy: str = 'hybrid'           # 'lru', 'lfu', 'hybrid'
    prefetch_enabled: bool = True
    prefetch_lookahead: int = 2
    min_score_threshold: float = 0.1


@dataclass
class OverlappedCheckpointingConfig:
    """Configuration for overlapped gradient checkpointing.

    YAML Path: experimental.checkpointing.overlapped.*
    """
    enabled: bool = False
    stream_overlap: bool = True
    target_layers: str = 'layers'


@dataclass
class DoubleCheckpointingConfig:
    """Configuration for double (nested) gradient checkpointing.

    YAML Path: experimental.checkpointing.double.*
    """
    enabled: bool = False
    coarse_checkpoint_interval: int = 8
    fine_checkpoint_interval: int = 2
    use_cuda_streams: bool = True


@dataclass
class CheckpointingOptConfig:
    """Configuration for checkpointing optimizations.

    YAML Path: experimental.checkpointing.*
    """
    overlapped: OverlappedCheckpointingConfig = field(default_factory=OverlappedCheckpointingConfig)
    double: DoubleCheckpointingConfig = field(default_factory=DoubleCheckpointingConfig)


@dataclass
class SYMIConfig:
    """Configuration for SYMI optimizer decoupling.

    YAML Path: experimental.symi.*
    """
    enabled: bool = False
    num_partitions: int = 0
    sync_frequency: int = 100
    use_async_sync: bool = True
    gradient_averaging: str = 'partition'
    state_precision: str = 'fp32'
    enable_checkpointing: bool = True


@dataclass
class Sparse24Config:
    """Configuration for 2:4 activation sparsity.

    YAML Path: experimental.sparse24.*
    """
    enabled: bool = False
    warmup_steps: int = 1000
    apply_to_gate: bool = True
    apply_to_up: bool = True
    apply_to_down: bool = False
    use_ste_scaling: bool = True
    ste_scale_factor: float = 1.0
    sparsity_granularity: str = 'activation'
    use_triton_kernel: bool = True
    log_sparsity_stats: bool = False


@dataclass
class StableMoEConfig:
    """Configuration for Stable-MoE routing.

    YAML Path: experimental.stable_moe.*
    """
    enabled: bool = False
    target_utilization: float = 0.0
    utilization_tolerance: float = 0.1
    adaptation_rate: float = 0.01
    temperature_init: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.9999
    capacity_min: float = 1.0
    capacity_max: float = 2.0
    log_utilization_histogram: bool = True
    log_capacity_factors: bool = True
    log_temperature: bool = True


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory.

    YAML Path: experimental.episodic_memory.*
    """
    use_episodic_memory: bool = False
    memory_capacity: int = 1000
    memory_selection_strategy: str = 'importance'
    memory_importance_threshold: float = 0.5
    memory_retrieval_method: str = 'cosine'
    memory_replay_ratio: float = 0.2
    memory_replay_strategy: str = 'importance'
    memory_adaptation_rate: float = 0.01
    memory_performance_window: int = 100
    task_id: int = 0
    silent_mode: bool = False
    enable_auto_grad_accumulation: bool = False
    priority_exponent: float = 0.6
    importance_weight_exponent: float = 0.4
    buffer_warmup_steps: int = 100
    store_aux_info: bool = False


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval Augmented Generation).

    YAML Path: experimental.rag.*
    """
    use_rag: bool = False
    knowledge_base_path: Optional[str] = None
    max_retrieved_docs: int = 5
    rag_fusion_type: str = 'attention'
    retriever_type: str = 'faiss'
    index_path: Optional[str] = None
    embedding_dim: int = 768
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    normalize_embeddings: bool = True
    chromadb_persist_dir: Optional[str] = None
    chromadb_collection: str = 'ava_rag'
    fusion_dropout: float = 0.1
    fusion_num_heads: int = 8
    fusion_layer: str = 'all'
    retrieval_mode: str = 'once'
    query_strategy: str = 'first_token'


@dataclass
class LossConfig:
    """Configuration for advanced loss functions.

    YAML Path: experimental.losses.*
    """
    use_focal_loss: bool = False
    use_contrastive_loss: bool = False
    use_diversity_loss: bool = False
    adaptive_loss_scaling: bool = False
    use_multi_token_prediction: bool = False
    num_future_tokens: int = 3
    mtp_weight: float = 0.1
    initial_temperature: float = 1.0
    adaptive_temperature: bool = False
    label_smoothing: float = 0.0
    use_moe_balancing: bool = False
    gradient_balance_weight: float = 0.0
    use_auxiliary_loss: bool = False
    use_ngram_penalty: bool = False
    ngram_size: int = 3
    ngram_penalty_weight: float = 0.0
    use_immediate_repetition_detector: bool = False
    immediate_repetition_weight: float = 0.0


@dataclass
class ArchitectureEnhancementsConfig:
    """Configuration for architecture enhancements.

    YAML Path: experimental.architecture.*
    """
    use_moh: bool = False                     # Mixture of Heads
    use_moa: bool = False                     # Mixture of Activations
    use_cross_attention: bool = False         # Multi-modal cross-attention
    use_alibi: bool = False                   # ALiBi positional encoding
    expert_routing_type: str = 'switch'


@dataclass
class QuantizationConfig:
    """Configuration for quantization.

    YAML Path: experimental.quantization.*
    """
    quantization_aware: bool = False
    bit_width: int = 8
    use_nvfp4: bool = False
    nvfp4_block_size: int = 16
    stochastic_rounding: bool = False
    use_hadamard_transform: bool = False
    use_torchao_nvfp4: bool = False


@dataclass
class ExperimentalSectionConfig:
    """Composite configuration for the experimental section.

    YAML Path: experimental.*
    """
    fp8: FP8Config = field(default_factory=FP8Config)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    caching: HybridCachingConfig = field(default_factory=HybridCachingConfig)
    checkpointing: CheckpointingOptConfig = field(default_factory=CheckpointingOptConfig)
    symi: SYMIConfig = field(default_factory=SYMIConfig)
    sparse24: Sparse24Config = field(default_factory=Sparse24Config)
    stable_moe: StableMoEConfig = field(default_factory=StableMoEConfig)
    episodic_memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    architecture: ArchitectureEnhancementsConfig = field(default_factory=ArchitectureEnhancementsConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


# =============================================================================
# LEGACY DATACLASSES (Backward Compatibility)
# These are kept for code that still references the old structure
# =============================================================================

# Legacy HardwareConfig with full fields
@dataclass
class LegacyHardwareConfig:
    """Legacy configuration for hardware settings (backward compatibility)."""
    device: str = 'cuda'
    mixed_precision: str = 'fp32'
    compile: bool = False
    num_gpus: int = 1
    use_gpu_load_balancing: bool = False
    balancing_strategy: str = 'adaptive'
    rebalance_interval: int = 1000
    enable_expert_migration: bool = True
    migration_threshold: float = 0.2
    log_gpu_metrics: bool = True


@dataclass
class GradientConfig:
    """Configuration for gradient surgery."""
    gradient_surgery: bool = False
    adaptive_gradient_surgery: bool = False
    gradient_surgery_method: str = 'pcgrad'


@dataclass
class RetryConfig:
    """Configuration for retry logic in pipeline error handling."""
    max_retries: int = 3
    initial_backoff: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff: float = 60.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_on_oom: bool = True
    reduce_batch_on_oom: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation during training."""
    eval_during_training: bool = False
    eval_metrics: Optional[str] = None
    eval_frequency: int = 500


@dataclass
class MoEMetricsConfig:
    """Configuration for MoE-specific metrics tracking."""
    track_expert_utilization: bool = True
    log_frequency: int = 50
    track_routing_decisions: bool = False
    track_load_balance: bool = True


@dataclass
class LRFinderConfig:
    """Configuration for Learning Rate Finder."""
    run_lr_finder: bool = False
    start_lr: float = 1e-8
    end_lr: float = 1.0
    num_iterations: int = 100
    suggestion_method: str = 'steepest'
    use_suggested_lr: bool = False
    plot_path: Optional[str] = None
    smooth_beta: float = 0.98
    stop_div_threshold: float = 4.0


@dataclass
class RunManagementConfig:
    """Configuration for run management."""
    run_name: Optional[str] = None
    run_tags: Optional[str] = None
    run_description: Optional[str] = None
    disable_run_manager: bool = False


@dataclass
class WandBConfig:
    """Legacy configuration for Weights & Biases."""
    use_wandb: bool = False
    disable_wandb: bool = False
    wandb_offline: bool = False
    wandb_project: str = 'Ava'
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ['moe', 'training'])
    wandb_log_freq: int = 10
    wandb_cache_size: int = 2000
    wandb_cache_flush_interval: int = 50


@dataclass
class OptimizationsConfig:
    """Configuration for all training optimizations."""
    torchinductor_autotune: int = 1
    memory_headroom_gb: float = 3.0
    memory_cleanup_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'warning': 0.85,
        'critical': 0.90,
        'emergency': 0.95
    })
    expert_prefetch: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'lookahead': 2,
        'use_multiple_streams': True
    })
    expert_cache: Dict[str, Any] = field(default_factory=lambda: {
        'auto_limit': True,
        'use_lru_eviction': True,
        'clear_after_optimizer_step': True
    })
    checkpoint: Dict[str, Any] = field(default_factory=lambda: {
        'async_saving': True
    })
    memory_cleanup: Dict[str, Any] = field(default_factory=lambda: {
        'fast_mode': True,
        'remove_sleep': True
    })
    proactive_memory_cleanup: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'fragmentation_threshold': 0.30,
        'cleanup_frequency': 2000,
        'cleanup_after_validation': True,
        'cleanup_after_generation': True
    })
    dataloader: Dict[str, Any] = field(default_factory=lambda: {
        'adaptive_file_reading': True,
        'adaptive_multipliers': {
            'large_files': 4,
            'medium_files': 2,
            'small_files': 1
        }
    })
    gradient_checkpointing: Dict[str, Any] = field(default_factory=lambda: {
        'selective': False,
        'checkpoint_attention': True
    })
    router: Dict[str, Any] = field(default_factory=lambda: {
        'cache_hash_on_gpu': True,
        'compile_routers': True,
        'compile_mode': 'default',
        'compile_dynamic': True
    })


# =============================================================================
# ROOT CONFIGURATION
# =============================================================================

@dataclass
class EnhancedTrainingConfig:
    """Main configuration class combining all sub-configs.

    Organized into 8 sections matching the YAML schema:
    1. model       - Architecture, MoE, tokens, regularization
    2. training    - Optimizer, schedule, batching, validation, generation
    3. data        - Loading, streaming, packing, splits
    4. compute     - Device, precision, CUDA, kernels, memory
    5. distributed - Multi-GPU, DeepSpeed, load balancing
    6. logging     - Console, WandB, TensorBoard, diagnostics
    7. checkpoints - Save/load, model selection
    8. experimental- FP8, MTP, caching, advanced features
    """
    config_file: str                          # Required config file path

    # Section 1: MODEL
    model: ModelConfig = field(default_factory=ModelConfig)

    # Section 2: TRAINING
    training: TrainingSectionConfig = field(default_factory=TrainingSectionConfig)

    # Section 3: DATA
    data: DataConfig = field(default_factory=DataConfig)
    multi_column_data: MultiColumnDataConfig = field(default_factory=MultiColumnDataConfig)

    # Section 4: COMPUTE
    compute: ComputeSectionConfig = field(default_factory=ComputeSectionConfig)

    # Section 5: DISTRIBUTED
    distributed: DistributedSectionConfig = field(default_factory=DistributedSectionConfig)

    # Section 6: LOGGING
    logging: LoggingSectionConfig = field(default_factory=LoggingSectionConfig)

    # Section 7: CHECKPOINTS
    checkpoints: CheckpointsSectionConfig = field(default_factory=CheckpointsSectionConfig)

    # Section 8: EXPERIMENTAL
    experimental: ExperimentalSectionConfig = field(default_factory=ExperimentalSectionConfig)

    # =========================================================================
    # LEGACY FIELDS (Backward Compatibility)
    # These fields are kept for code that still references the old structure
    # =========================================================================
    hardware: LegacyHardwareConfig = field(default_factory=LegacyHardwareConfig)
    architecture: ArchitectureEnhancementsConfig = field(default_factory=ArchitectureEnhancementsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)
    memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    adaptive_mtp: MTPConfig = field(default_factory=MTPConfig)
    dev_log: DevLogConfig = field(default_factory=DevLogConfig)
    optimizations: OptimizationsConfig = field(default_factory=OptimizationsConfig)
    model_selection: SelectionConfig = field(default_factory=SelectionConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    output: CheckpointsSectionConfig = field(default_factory=CheckpointsSectionConfig)
    run_management: RunManagementConfig = field(default_factory=RunManagementConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    kernel_optimization: KernelsConfig = field(default_factory=KernelsConfig)

    # Enhanced features
    enhanced_features: Optional[Dict[str, Any]] = None

    # Special flags
    enable_all_features: bool = False
    multi_task: bool = False


# =============================================================================
# TRAINING CONFIG MANAGER
# =============================================================================

class TrainingConfigManager:
    """Manager for training configuration with validation and feature compatibility."""

    def __init__(self):
        self.config = None
        self._feature_dependencies = self._build_feature_dependencies()

    def load_yaml_config(self, config_path: str) -> DynamicConfig:
        """
        Load YAML configuration file into a dynamic structure.

        Automatically resolves relative paths in the configuration based on
        the project root directory.

        Args:
            config_path: Path to YAML configuration file (can be relative or absolute)

        Returns:
            DynamicConfig object with all YAML fields accessible via dot notation
            and paths resolved to absolute paths

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        # Try to use the enhanced YAML loader with path resolution
        try:
            from ava.config.yaml_loader import load_yaml_with_path_resolution
            config_dict = load_yaml_with_path_resolution(config_path)
        except (ImportError, Exception):
            # Fallback to manual loading if loader not available
            config_path_obj = Path(config_path)

            # If path doesn't exist, try different relative paths
            if not config_path_obj.exists():
                # Try relative to current directory
                alt_path = Path.cwd() / config_path
                if alt_path.exists():
                    config_path_obj = alt_path
                else:
                    raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path_obj, "r") as f:
                config_dict = yaml.safe_load(f)

        # Return config exactly as written in YAML - no auto-sync overrides
        return DynamicConfig(config_dict)

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser for training."""
        parser = argparse.ArgumentParser(
            description='Enhanced LLM Training with Advanced Features',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic enhanced training
  python train.py --config configs/gpu/small.yaml --enable-all-features

  # RAG-enabled training
  python train.py --config configs/gpu/small.yaml --use-rag --knowledge-base-path data/kb/

  # Multi-task training with gradient surgery
  python train.py --config configs/gpu/small.yaml --multi-task --gradient-surgery

  # Quantization-aware training
  python train.py --config configs/gpu/small.yaml --quantization-aware --bit-width 8
            """
        )

        # Configuration file (required)
        parser.add_argument('--config', type=str, required=True,
                          help='Path to configuration file')

        # === ENHANCED FEATURE FLAGS ===
        parser.add_argument('--enable-all-features', action='store_true',
                          help='Enable all enhanced features (overrides individual flags)')

        # Learning Rate Finder
        lr_finder_group = parser.add_argument_group('Learning Rate Finder')
        lr_finder_group.add_argument('--run-lr-finder', action='store_true',
                                    help='Run LR Finder before training')
        lr_finder_group.add_argument('--lr-finder-start', type=float, default=1e-8,
                                    help='Starting LR for LR finder')
        lr_finder_group.add_argument('--lr-finder-end', type=float, default=1.0,
                                    help='Ending LR for LR finder')
        lr_finder_group.add_argument('--lr-finder-iterations', type=int, default=100,
                                    help='Number of iterations for LR finder')
        lr_finder_group.add_argument('--lr-finder-method', type=str, default='steepest',
                                    choices=['steepest', 'minimum', 'valley'],
                                    help='Method for suggesting LR')
        lr_finder_group.add_argument('--lr-finder-use-suggested', action='store_true',
                                    help='Automatically use the suggested LR')
        lr_finder_group.add_argument('--lr-finder-plot-path', type=str, default=None,
                                    help='Path to save LR finder plot')

        # === DATA ARGUMENTS ===
        data_group = parser.add_argument_group('Data Configuration')
        data_group.add_argument('--data-dir', type=str,
                               default=str(get_data_dir("processed")),
                               help='Directory containing preprocessed training data')
        data_group.add_argument('--max-length', type=int, default=512,
                               help='Maximum sequence length')
        data_group.add_argument('--max-samples', type=int, default=None,
                               help='Maximum number of training samples')
        data_group.add_argument('--streaming', action='store_true', default=True,
                               help='Use streaming data loader')
        data_group.add_argument('--no-streaming', dest='streaming', action='store_false',
                               help='Disable streaming')
        data_group.add_argument('--buffer-size', type=int, default=50000,
                               help='Buffer size for streaming')
        data_group.add_argument('--num-workers', type=int, default=8,
                               help='Number of parallel data loading workers')
        data_group.add_argument('--prefetch-factor', type=int, default=4,
                               help='Number of batches to prefetch per worker')
        data_group.add_argument('--no-persistent-workers', dest='persistent_workers',
                               action='store_false', default=True,
                               help='Disable persistent workers')

        # === MULTI-COLUMN DATA ARGUMENTS ===
        multi_col_group = parser.add_argument_group('Multi-Column Data')
        multi_col_group.add_argument('--use-multi-column', action='store_true',
                                    help='Enable multi-column data loading')
        multi_col_group.add_argument('--dataset-config', type=str, default=None,
                                    help='Path to dataset configuration file')
        multi_col_group.add_argument('--hf-dataset', type=str, default=None,
                                    help='HuggingFace dataset name')
        multi_col_group.add_argument('--hf-dataset-config', type=str, default=None,
                                    help='HuggingFace dataset configuration')
        multi_col_group.add_argument('--column-names', type=str, default=None,
                                    help='Comma-separated list of column names')
        multi_col_group.add_argument('--column-types', type=str, default=None,
                                    help='Comma-separated list of column types')
        multi_col_group.add_argument('--column-roles', type=str, default=None,
                                    help='Comma-separated list of column roles')
        multi_col_group.add_argument('--combine-strategy', type=str, default='concatenate',
                                    choices=['concatenate', 'separate', 'template'],
                                    help='Strategy for combining columns')
        multi_col_group.add_argument('--column-template', type=str, default=None,
                                    help='Template string for combining columns')

        # === TRAINING ARGUMENTS ===
        training_group = parser.add_argument_group('Training Parameters')
        training_group.add_argument('--batch-size', type=int, default=None,
                                   help='Batch size')
        training_group.add_argument('--epochs', type=int, default=None,
                                   help='Number of epochs')
        training_group.add_argument('--learning-rate', type=float, default=None,
                                   help='Learning rate')
        training_group.add_argument('--gradient-accumulation', type=int, default=1,
                                   help='Gradient accumulation steps')

        # === OUTPUT ARGUMENTS ===
        output_group = parser.add_argument_group('Output Configuration')
        output_group.add_argument('--output-dir', type=str, default=str(get_outputs_dir()),
                                 help='Output directory for checkpoints')
        output_group.add_argument('--save-every', type=int, default=100,
                                 help='Save checkpoint every N steps')
        output_group.add_argument('--resume', type=str, default=None,
                                 help='Resume from checkpoint')
        output_group.add_argument('--fresh-start', action='store_true',
                                 help='Force fresh start')

        # === RUN MANAGEMENT ARGUMENTS ===
        run_group = parser.add_argument_group('Run Management')
        run_group.add_argument('--run-name', type=str, default=None,
                              help='Custom name for this training run')
        run_group.add_argument('--run-tags', type=str, default=None,
                              help='Comma-separated tags for this run')
        run_group.add_argument('--run-description', type=str, default=None,
                              help='Description of this experiment')
        run_group.add_argument('--disable-run-manager', action='store_true',
                              help='Disable the run manager')

        # Weights & Biases arguments
        wandb_group = parser.add_argument_group('Weights & Biases')
        wandb_group.add_argument('--use-wandb', action='store_true', default=True,
                                help='Enable Weights & Biases logging')
        wandb_group.add_argument('--disable-wandb', action='store_true',
                                help='Disable Weights & Biases logging')
        wandb_group.add_argument('--wandb-offline', action='store_true',
                                help='Force Weights & Biases offline mode')
        wandb_group.add_argument('--wandb-project', type=str, default='Ava',
                                help='Weights & Biases project name')
        wandb_group.add_argument('--wandb-name', type=str, default=None,
                                help='Weights & Biases run name')
        wandb_group.add_argument('--wandb-tags', type=str, default=None,
                                help='Comma-separated Weights & Biases tags')
        wandb_group.add_argument('--wandb-log-freq', type=int, default=10,
                                help='Weights & Biases logging frequency')
        wandb_group.add_argument('--wandb-cache-size', type=int, default=2000,
                                help='Weights & Biases cache size')
        wandb_group.add_argument('--wandb-cache-flush-interval', type=int, default=50,
                                help='Weights & Biases cache flush interval')

        # DeepSpeed arguments
        ds_group = parser.add_argument_group('DeepSpeed Distributed Training')
        ds_group.add_argument('--use-deepspeed', action='store_true',
                             help='Enable DeepSpeed distributed training')
        ds_group.add_argument('--deepspeed-config', type=str, default=None,
                             help='Path to DeepSpeed JSON configuration file')
        ds_group.add_argument('--zero-stage', type=int, default=2, choices=[0, 1, 2, 3],
                             help='ZeRO optimization stage')
        ds_group.add_argument('--cpu-offload', action='store_true',
                             help='Enable CPU offloading')
        ds_group.add_argument('--nvme-offload', action='store_true',
                             help='Enable NVMe offloading')
        ds_group.add_argument('--ds-gradient-accumulation', type=int, default=1,
                             help='DeepSpeed gradient accumulation steps')
        ds_group.add_argument('--train-batch-size', type=int, default=None,
                             help='Global training batch size')
        ds_group.add_argument('--micro-batch-size', type=int, default=None,
                             help='Micro batch size per GPU')
        ds_group.add_argument('--ds-precision', type=str, default='fp16',
                             choices=['fp16', 'bf16', 'fp32'],
                             help='Mixed precision type for DeepSpeed')
        ds_group.add_argument('--activation-checkpointing', action='store_true',
                             help='Enable activation checkpointing')
        ds_group.add_argument('--partition-activations', action='store_true',
                             help='Partition activations across GPUs')
        ds_group.add_argument('--cpu-checkpointing', action='store_true',
                             help='Store activation checkpoints on CPU')
        ds_group.add_argument('--pipeline-parallel-size', type=int, default=1,
                             help='Pipeline parallelism size')
        ds_group.add_argument('--wall-clock-breakdown', action='store_true',
                             help='Enable DeepSpeed wall clock breakdown')

        # Performance mode arguments
        perf_group = parser.add_argument_group('Performance Modes')
        perf_group.add_argument('--ultra-fast-mode', action='store_true',
                               help='Ultra-fast mode: disable all logging')
        perf_group.add_argument('--fast-progress', action='store_true',
                               help='Fast-progress mode: enhanced progress bar')
        perf_group.add_argument('--minimal-progress', action='store_true',
                               help='Minimal-progress mode: compact display')
        perf_group.add_argument('--no-sync', action='store_true',
                               help='No-sync mode: disable CUDA synchronization')
        perf_group.add_argument('--express-mode', action='store_true',
                               help='Express mode: optimized async logging')

        return parser

    def parse_args_to_config(self, args: argparse.Namespace) -> EnhancedTrainingConfig:
        """Convert parsed arguments to structured configuration."""

        # Handle enable-all-features flag
        if args.enable_all_features:
            self._enable_all_features(args)

        # Load YAML to extract config sections
        yaml_config = self.load_yaml_config(args.config)
        yaml_dict = yaml_config.to_dict()

        # Extract hardware/device config
        hardware_dict = yaml_dict.get('hardware', yaml_dict.get('compute', {}).get('device', {}))
        hardware_config = LegacyHardwareConfig(
            device=hardware_dict.get('device', hardware_dict.get('type', 'cuda')),
            mixed_precision=hardware_dict.get('mixed_precision', 'fp32'),
            compile=hardware_dict.get('compile', False)
        )

        # Load dev_log config from YAML
        dev_log_dict = yaml_dict.get('dev_log', yaml_dict.get('logging', {}).get('dev', {}))
        dev_log_config = DevLogConfig(
            enabled=dev_log_dict.get('enabled', False),
            show_file_timings=dev_log_dict.get('show_file_timings', True),
            show_batch_timings=dev_log_dict.get('show_batch_timings', True),
            show_step_breakdown=dev_log_dict.get('show_step_breakdown', True),
            report_interval=dev_log_dict.get('report_interval', 100)
        )

        # Create structured config
        config = EnhancedTrainingConfig(
            config_file=args.config,
            enable_all_features=args.enable_all_features,
            multi_task=False,

            # Hardware configuration
            hardware=hardware_config,

            # Use default configs for other sections
            architecture=ArchitectureEnhancementsConfig(),
            rag=RAGConfig(),
            gradient=GradientConfig(),
            evaluation=EvaluationConfig(),
            quantization=QuantizationConfig(),
            memory=EpisodicMemoryConfig(),

            lr_finder=LRFinderConfig(
                run_lr_finder=args.run_lr_finder,
                start_lr=args.lr_finder_start,
                end_lr=args.lr_finder_end,
                num_iterations=args.lr_finder_iterations,
                suggestion_method=args.lr_finder_method,
                use_suggested_lr=args.lr_finder_use_suggested,
                plot_path=args.lr_finder_plot_path
            ),

            data=DataConfig(
                data_dir=args.data_dir,
                max_length=args.max_length,
                max_samples=args.max_samples,
                streaming=args.streaming,
                buffer_size=args.buffer_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers
            ),

            multi_column_data=MultiColumnDataConfig(
                use_multi_column=args.use_multi_column,
                dataset_config=args.dataset_config,
                hf_dataset=args.hf_dataset,
                hf_dataset_config=args.hf_dataset_config,
                column_names=args.column_names,
                column_types=args.column_types,
                column_roles=args.column_roles,
                combine_strategy=args.combine_strategy,
                column_template=args.column_template
            ),

            training=TrainingSectionConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                gradient_accumulation=args.gradient_accumulation
            ),

            output=CheckpointsSectionConfig(
                output_dir=args.output_dir,
                save_every=args.save_every,
                resume=args.resume,
                fresh_start=args.fresh_start
            ),

            run_management=RunManagementConfig(
                run_name=args.run_name,
                run_tags=args.run_tags,
                run_description=args.run_description,
                disable_run_manager=args.disable_run_manager
            ),

            wandb=WandBConfig(
                use_wandb=args.use_wandb and not args.disable_wandb,
                disable_wandb=args.disable_wandb,
                wandb_offline=args.wandb_offline,
                wandb_project=args.wandb_project,
                wandb_name=args.wandb_name,
                wandb_tags=args.wandb_tags.split(',') if args.wandb_tags else ['moe', 'training'],
                wandb_log_freq=args.wandb_log_freq,
                wandb_cache_size=args.wandb_cache_size,
                wandb_cache_flush_interval=args.wandb_cache_flush_interval
            ),

            performance=PerformanceConfig(
                ultra_fast_mode=args.ultra_fast_mode,
                fast_progress=args.fast_progress,
                minimal_progress=args.minimal_progress,
                no_sync=args.no_sync,
                express_mode=args.express_mode
            ),

            deepspeed=DeepSpeedConfig(
                use_deepspeed=args.use_deepspeed,
                config_file=args.deepspeed_config,
                zero_stage=args.zero_stage,
                cpu_offload=args.cpu_offload,
                nvme_offload=args.nvme_offload,
                gradient_accumulation_steps=args.ds_gradient_accumulation,
                train_batch_size=args.train_batch_size,
                micro_batch_size=args.micro_batch_size,
                precision_type=args.ds_precision,
                activation_checkpointing=args.activation_checkpointing,
                partition_activations=args.partition_activations,
                cpu_checkpointing=args.cpu_checkpointing,
                pipeline_parallel_size=args.pipeline_parallel_size,
                wall_clock_breakdown=args.wall_clock_breakdown
            ),

            # Development logging config loaded from YAML
            dev_log=dev_log_config
        )

        self.config = config
        return config

    def create_unified_config(self, args: argparse.Namespace) -> DynamicConfig:
        """
        Create a unified configuration by loading YAML and merging command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            DynamicConfig object with merged YAML + CLI configuration
        """
        # Load YAML configuration dynamically
        yaml_config = self.load_yaml_config(args.config)

        # Apply command-line overrides
        # Only override if the argument was explicitly provided (not None)

        # Training overrides
        if hasattr(yaml_config, 'training'):
            if args.batch_size is not None:
                yaml_config.training.batch_size = args.batch_size
            if args.learning_rate is not None:
                yaml_config.training.learning_rate = args.learning_rate
            if args.epochs is not None:
                yaml_config.training.epochs = args.epochs
            if args.gradient_accumulation != 1:
                yaml_config.training.gradient_accumulation_steps = args.gradient_accumulation
        else:
            training_dict = {}
            if args.batch_size is not None:
                training_dict['batch_size'] = args.batch_size
            if args.learning_rate is not None:
                training_dict['learning_rate'] = args.learning_rate
            if args.epochs is not None:
                training_dict['epochs'] = args.epochs
            if args.gradient_accumulation != 1:
                training_dict['gradient_accumulation_steps'] = args.gradient_accumulation
            if training_dict:
                yaml_config.training = DynamicConfig(training_dict)

        # Data overrides
        default_data_dir = str(get_data_dir("processed"))
        if hasattr(yaml_config, 'data'):
            if args.data_dir != default_data_dir:
                yaml_config.data.data_dir = args.data_dir
            if args.max_length != 512:
                yaml_config.data.max_length = args.max_length
            if args.max_samples is not None:
                yaml_config.data.max_samples = args.max_samples
        else:
            data_dict = {}
            if args.data_dir != default_data_dir:
                data_dict['data_dir'] = args.data_dir
            if args.max_length != 512:
                data_dict['max_length'] = args.max_length
            if args.max_samples is not None:
                data_dict['max_samples'] = args.max_samples
            if data_dict:
                yaml_config.data = DynamicConfig(data_dict)

        # Output overrides
        default_output_dir = str(get_outputs_dir())
        if hasattr(yaml_config, 'output'):
            if args.output_dir != default_output_dir:
                yaml_config.output.output_dir = args.output_dir
            if args.resume is not None:
                yaml_config.output.resume = args.resume
        else:
            output_dict = {}
            if args.output_dir != default_output_dir:
                output_dict['output_dir'] = args.output_dir
            if args.resume is not None:
                output_dict['resume'] = args.resume
            if output_dict:
                yaml_config.output = DynamicConfig(output_dict)

        # Store for later access
        self.config = yaml_config
        return yaml_config

    def _enable_all_features(self, args: argparse.Namespace) -> None:
        """Enable all enhanced features when --enable-all-features is set."""
        # Currently no features to enable beyond what's in YAML configs.
        pass

    def _build_feature_dependencies(self) -> Dict[str, List[str]]:
        """Build feature dependency mapping."""
        return {}

    def validate_dynamic_config(self, config: DynamicConfig) -> List[str]:
        """
        Validate dynamic configuration and return list of warnings/errors.

        Args:
            config: DynamicConfig object to validate

        Returns:
            List of validation messages
        """
        messages = []

        def safe_get(obj, path, default=None):
            """Safely get nested attribute using dot notation"""
            parts = path.split('.')
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return default
            return obj

        # Check DeepSpeed settings
        if safe_get(config, 'deepspeed.use_deepspeed', False) or safe_get(config, 'distributed.deepspeed.use_deepspeed', False):
            config_file = safe_get(config, 'deepspeed.config_file') or safe_get(config, 'distributed.deepspeed.config_file')
            if config_file and isinstance(config_file, (str, Path)) and not Path(config_file).exists():
                messages.append(f"DeepSpeed config file not found: {config_file}")

            zero_stage = safe_get(config, 'deepspeed.zero_stage', 0) or safe_get(config, 'distributed.deepspeed.zero_stage', 0)
            cpu_offload = safe_get(config, 'deepspeed.cpu_offload', False) or safe_get(config, 'distributed.deepspeed.cpu_offload', False)
            nvme_offload = safe_get(config, 'deepspeed.nvme_offload', False) or safe_get(config, 'distributed.deepspeed.nvme_offload', False)

            if zero_stage == 3 and not cpu_offload:
                messages.append("Warning: ZeRO stage 3 without CPU offload may cause OOM")

            if nvme_offload and not cpu_offload:
                messages.append("Warning: NVMe offload requires CPU offload to be enabled")

        # Check data directory exists
        data_dir = safe_get(config, 'data.data_dir')
        if data_dir and isinstance(data_dir, (str, Path)) and not Path(data_dir).exists():
            messages.append(f"Warning: Data directory not found: {data_dir}")

        # Check performance mode conflicts
        perf_modes = [
            safe_get(config, 'performance.ultra_fast_mode', False) or safe_get(config, 'compute.performance.ultra_fast_mode', False),
            safe_get(config, 'performance.fast_progress', False) or safe_get(config, 'compute.performance.fast_progress', False),
            safe_get(config, 'performance.minimal_progress', False) or safe_get(config, 'compute.performance.minimal_progress', False),
            safe_get(config, 'performance.express_mode', False) or safe_get(config, 'compute.performance.express_mode', False)
        ]
        if sum(1 for mode in perf_modes if mode) > 1:
            messages.append("Warning: Multiple performance modes enabled, may conflict")

        return messages

    def validate_config(self, config: EnhancedTrainingConfig) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages
        """
        messages = []

        # Check file paths exist
        if not Path(config.config_file).exists():
            messages.append(f"Config file not found: {config.config_file}")

        if config.rag.knowledge_base_path and not Path(config.rag.knowledge_base_path).exists():
            messages.append(f"Knowledge base path not found: {config.rag.knowledge_base_path}")

        # Check feature dependencies
        if config.gradient.gradient_surgery and not config.multi_task:
            messages.append("Warning: Gradient surgery enabled but multi-task is disabled")

        if config.rag.use_rag and not config.rag.knowledge_base_path:
            messages.append("Warning: RAG enabled but no knowledge base path provided")

        # Check performance mode conflicts
        perf_modes = [
            config.performance.ultra_fast_mode,
            config.performance.fast_progress,
            config.performance.minimal_progress,
            config.performance.express_mode
        ]
        if sum(1 for mode in perf_modes if mode) > 1:
            messages.append("Warning: Multiple performance modes enabled, may conflict")

        # Check quantization settings
        if config.quantization.quantization_aware and config.quantization.use_nvfp4:
            messages.append("Warning: Both standard quantization and NVFP4 enabled")

        # Check DeepSpeed settings
        if config.deepspeed.use_deepspeed:
            if config.deepspeed.config_file and not Path(config.deepspeed.config_file).exists():
                messages.append(f"DeepSpeed config file not found: {config.deepspeed.config_file}")

            if config.deepspeed.zero_stage == 3 and not config.deepspeed.cpu_offload:
                messages.append("Warning: ZeRO stage 3 without CPU offload may cause OOM")

            if config.deepspeed.nvme_offload and not config.deepspeed.cpu_offload:
                messages.append("Warning: NVMe offload requires CPU offload to be enabled")

        return messages

    def get_feature_summary(self, config: EnhancedTrainingConfig) -> Dict[str, Any]:
        """Get summary of enabled features."""
        enabled_features = []

        # Architecture features
        if config.architecture.use_moh:
            enabled_features.append("Mixture of Heads")
        if config.architecture.use_moa:
            enabled_features.append("Mixture of Activations")
        if config.architecture.use_cross_attention:
            enabled_features.append("Cross-Attention")
        if config.architecture.use_alibi:
            enabled_features.append("ALiBi Positioning")

        # Advanced features
        if config.rag.use_rag:
            enabled_features.append("RAG System")
        if config.gradient.gradient_surgery:
            enabled_features.append("Gradient Surgery")
        if config.quantization.quantization_aware or config.quantization.use_nvfp4:
            enabled_features.append("Quantization")
        if config.memory.use_episodic_memory:
            enabled_features.append("Episodic Memory")
        if config.deepspeed.use_deepspeed:
            enabled_features.append(f"DeepSpeed ZeRO-{config.deepspeed.zero_stage}")

        return {
            'total_features': len(enabled_features),
            'enabled_features': enabled_features,
            'expert_routing': config.architecture.expert_routing_type,
            'performance_mode': self._get_active_performance_mode(config.performance)
        }

    def _get_active_performance_mode(self, perf_config: PerformanceConfig) -> str:
        """Get the active performance mode."""
        if perf_config.ultra_fast_mode:
            return "Ultra Fast"
        elif perf_config.fast_progress:
            return "Fast Progress"
        elif perf_config.minimal_progress:
            return "Minimal Progress"
        elif perf_config.express_mode:
            return "Express Mode"
        elif perf_config.no_sync:
            return "No Sync"
        else:
            return "Standard"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

# Keep AdaptiveMTPConfig as an alias for MTPConfig (backward compatibility)
AdaptiveMTPConfig = MTPConfig

# Keep ArchitectureConfig as an alias (backward compatibility)
ArchitectureConfig = ArchitectureEnhancementsConfig
