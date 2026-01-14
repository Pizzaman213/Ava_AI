"""
Training Configuration Manager

This module handles all training configuration management including
enhanced feature flags, parameter validation, and configuration inheritance.
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


# Import path utilities for relative path resolution
try:
    from ava.core.paths import get_project_root, get_data_dir, get_outputs_dir
except ImportError:
    # Fallback for when utils.paths is not available
    def get_project_root() -> Path:
        from pathlib import Path
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists() or (parent / ".project").exists():
                return parent
        return current.parents[4] if len(current.parts) > 4 else Path.cwd()

    def get_data_dir(data_type: str = "processed") -> Path:
        return get_project_root() / "code" / "data" / data_type

    def get_outputs_dir() -> Path:
        return get_project_root() / "code" / "outputs"


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


@dataclass
class HardwareConfig:
    """Configuration for hardware settings."""
    device: str = 'cuda'                      # Device: 'cuda', 'cpu', or 'mps'
    mixed_precision: str = 'fp32'             # Mixed precision: 'fp32', 'fp16', 'bf16'
    compile: bool = False                     # Enable torch.compile
    num_gpus: int = 1                         # Number of GPUs to use for training

    # GPU Load Balancing Settings
    use_gpu_load_balancing: bool = False      # Enable GPU load balancing for multi-GPU MoE
    balancing_strategy: str = 'adaptive'      # Load balancing strategy: 'round_robin', 'memory_aware', 'compute_aware', 'adaptive'
    rebalance_interval: int = 1000            # Steps between load rebalancing checks
    enable_expert_migration: bool = True      # Allow expert migration between GPUs
    migration_threshold: float = 0.2          # Load imbalance threshold for migration (0.2 = 20%)
    log_gpu_metrics: bool = True              # Log per-GPU metrics (memory, compute, etc.)


@dataclass
class ModelConfig:
    """Configuration for model architecture and optimizations."""
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
    # torch.compile settings moved to compute.performance.enable_torch_compile
    torch_compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    torch_compile_dynamic: bool = False       # Allow dynamic shapes (slower but flexible)
    torch_compile_fullgraph: bool = False     # Require full graph (faster but stricter)
    use_compile_friendly_dispatch: bool = True  # Use compile-friendly expert dispatch
    enable_cudagraphs_safe_routing: bool = False  # Enable CUDA graphs safe routing (deprecated)
    use_flash_attention: bool = True          # Use flash attention
    gradient_checkpointing: bool = True       # Enable gradient checkpointing
    use_optimized_moe: bool = True            # Use optimized MoE implementation
    quantize_kv_cache: bool = False           # Enable KV cache quantization for memory savings

    # Auxiliary losses
    router_z_loss_coef: float = 0.0001        # Router z-loss coefficient
    load_balance_loss_coef: float = 0.01      # Load balance loss coefficient
    diversity_loss_coef: float = 0.0001       # Diversity loss coefficient
    expert_dropout_loss_coef: float = 0.0     # Expert dropout loss coefficient
    router_jitter_noise: float = 0.01         # Router jitter noise for exploration
    aux_loss_frequency: int = 500             # Compute aux losses every N steps (500=fast, 10=detailed monitoring, 1=every step is slow)

    # Regularization
    attention_dropout: float = 0.0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.01

    # Positional encoding
    use_alibi: bool = False                   # Use ALiBi positional encoding instead of absolute
    rope_theta: float = 10000.0               # RoPE base theta for rotary positional embeddings
    rope_scaling: Optional[Dict[str, float]] = None  # RoPE scaling configuration

    def __post_init__(self):
        """Validate configuration values to catch invalid configs early.

        FIX: Added comprehensive validations to catch errors at config load time
        rather than deep in training with cryptic errors.
        """
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

        # Warn about incompatible compile configurations
        if self.use_torch_compile:
            import warnings
            if self.enable_cudagraphs_safe_routing:
                warnings.warn(
                    "torch.compile + enable_cudagraphs_safe_routing may conflict. "
                    "torch.compile already uses CUDA graphs internally with 'reduce-overhead' mode.",
                    UserWarning
                )
            if self.torch_compile_dynamic and self.torch_compile_fullgraph:
                warnings.warn(
                    "torch_compile_dynamic=True with torch_compile_fullgraph=True may cause "
                    "frequent recompilation. Consider setting fullgraph=False for dynamic shapes.",
                    UserWarning
                )


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 512                     # Maximum generation length
    min_length: int = 10                      # Minimum generation length
    temperature: float = 0.8                  # Sampling temperature (lower = more coherent)
    top_p: float = 0.9                        # Nucleus sampling threshold
    top_k: Optional[int] = 50                 # Top-k sampling (None = disabled)
    repetition_penalty: float = 1.2           # Repetition penalty (>1.0 discourages)
    no_repeat_ngram_size: int = 3             # Block n-gram repetitions
    do_sample: bool = True                    # Enable sampling (vs greedy)
    num_beams: int = 1                        # Beam search width (1 = no beam search)
    early_stopping: bool = False              # Stop when all beams finish


@dataclass
class ArchitectureConfig:
    """Configuration for architecture enhancements.

    IMPORTANT: Defaults should be False/minimal to allow YAML config to control features.
    Only enable features when explicitly requested in YAML config.
    """
    use_moh: bool = False                     # Mixture of Heads (disabled by default)
    use_moa: bool = False                     # Mixture of Activations (disabled by default)
    use_cross_attention: bool = False         # Multi-modal cross-attention (disabled by default)
    use_alibi: bool = False                   # ALiBi positional encoding (disabled by default)
    expert_routing_type: str = 'switch'       # Expert routing type


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval Augmented Generation) system.

    RAG enhances the model by retrieving relevant documents from a
    knowledge base and incorporating them into the generation process.

    Supported retrievers:
        - faiss: Fast similarity search using Facebook's FAISS library
        - chromadb: Persistent vector store with optional embedding functions
        - memory: Simple in-memory retriever for testing

    Supported fusion strategies:
        - attention: Cross-attention over retrieved documents
        - concat: Concatenate and project
        - gated: Learnable gate for context mixing
        - adaptive: Combines multiple strategies with learned routing
    """
    use_rag: bool = False                     # Enable RAG (disabled by default - YAML controls)
    knowledge_base_path: Optional[str] = None  # KB path (alias for index_path)
    max_retrieved_docs: int = 5               # Max retrieved documents (alias for top_k)
    rag_fusion_type: str = 'attention'        # Fusion strategy: attention, concat, gated, adaptive

    # Retriever settings
    retriever_type: str = 'faiss'             # Retriever backend: faiss, chromadb, memory
    index_path: Optional[str] = None          # Path to pre-built vector index
    embedding_dim: int = 768                  # Dimension of document embeddings
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'  # Model for encoding
    normalize_embeddings: bool = True         # L2-normalize for cosine similarity

    # ChromaDB-specific settings
    chromadb_persist_dir: Optional[str] = None  # Persistent storage directory
    chromadb_collection: str = 'ava_rag'        # Collection name

    # Fusion settings
    fusion_dropout: float = 0.1               # Dropout in fusion layers
    fusion_num_heads: int = 8                 # Attention heads for cross-attention fusion
    fusion_layer: str = 'all'                 # Where to apply fusion: 'all', 'last', or layer index

    # Retrieval settings
    retrieval_mode: str = 'once'              # 'once' (at start) or 'per_layer'
    query_strategy: str = 'first_token'       # How to create query: 'first_token', 'mean', 'cls'


@dataclass
class AdaptiveMTPConfig:
    """Configuration for Adaptive Multi-Token Prediction."""
    # Enable/disable adaptive MTP
    use_adaptive_mtp: bool = False            # Enable adaptive MTP system

    # Core MTP settings
    num_prediction_heads: int = 3             # Number of future tokens to predict (2-4)
    confidence_threshold_train: float = 0.6   # Confidence threshold during training
    confidence_threshold_inference: float = 0.7  # Threshold during inference (higher)

    # Confidence gate settings
    gate_hidden_dims: str = "512,256"         # Hidden dims for gate MLP (comma-separated)
    gate_dropout: float = 0.1                 # Dropout in confidence gate
    gate_activation: str = 'gelu'             # Activation function
    use_attention_pooling: bool = False       # Use attention pooling in gate

    # Prediction head settings
    head_type: str = 'linear'                 # 'linear' or 'mlp'
    head_intermediate_size: Optional[int] = None  # Intermediate size for MLP heads
    head_dropout: float = 0.1                 # Dropout in prediction heads
    share_projections: bool = False           # Share weights across heads

    # Training settings
    mtp_warmup_epochs: int = 2                # Train only primary head for first N epochs
    confidence_reg_strength: float = 0.01     # Regularization for confident predictions

    # Loss weighting
    use_confidence_weighting: bool = False    # Weight losses by confidence (YAML controls)
    primary_loss_weight: float = 1.0          # Primary token always gets full weight
    additional_loss_base_weight: float = 0.1  # Base weight for additional tokens

    # Efficiency settings
    enable_dynamic_prediction: bool = False   # Skip MTP when low confidence (YAML controls)
    min_confidence_for_computation: float = 0.3  # Don't compute heads below this


@dataclass
class LossConfig:
    """Configuration for advanced loss functions."""
    use_focal_loss: bool = False              # Focal loss (disabled by default - YAML controls)
    use_contrastive_loss: bool = False        # Contrastive loss (disabled by default)
    use_diversity_loss: bool = False          # Diversity loss (disabled by default)
    adaptive_loss_scaling: bool = False       # Adaptive loss scaling (disabled by default)

    # Multi-token prediction settings (DeepSeek-style)
    use_multi_token_prediction: bool = False  # Enable MTP loss (disabled by default - YAML controls)
    num_future_tokens: int = 3                # Number of future tokens to predict
    mtp_weight: float = 0.1                   # Weight for MTP loss

    # Temperature scaling settings
    initial_temperature: float = 1.0          # Initial temperature for scaling
    adaptive_temperature: bool = False        # Adapt temperature based on training (disabled by default)
    label_smoothing: float = 0.0              # Label smoothing factor (0 = disabled by default)

    # MoE balancing settings
    use_moe_balancing: bool = False           # Enable auxiliary-free MoE balancing (YAML controls)
    gradient_balance_weight: float = 0.0      # Weight for gradient-based balancing
    use_auxiliary_loss: bool = False          # Use traditional auxiliary loss (YAML controls)

    # N-gram repetition blocking (disabled by default for speed, YAML controls)
    use_ngram_penalty: bool = False           # Enable n-gram repetition detection (YAML controls)
    ngram_size: int = 3                       # Size of n-grams to detect
    ngram_penalty_weight: float = 0.0         # Weight for n-gram repetition penalty
    use_immediate_repetition_detector: bool = False  # Detect consecutive token repetition (YAML controls)
    immediate_repetition_weight: float = 0.0  # Weight for immediate repetition penalty


@dataclass
class GradientConfig:
    """Configuration for gradient surgery."""
    gradient_surgery: bool = False            # Enable gradient surgery (disabled by default)
    adaptive_gradient_surgery: bool = False   # Adaptive method selection (disabled by default)
    gradient_surgery_method: str = 'pcgrad'   # Surgery method


@dataclass
class RetryConfig:
    """Configuration for retry logic in pipeline error handling."""
    max_retries: int = 3                      # Maximum retry attempts per component
    initial_backoff: float = 1.0              # Initial backoff delay in seconds
    backoff_multiplier: float = 2.0           # Multiplier for exponential backoff
    max_backoff: float = 60.0                 # Maximum backoff delay in seconds
    jitter: bool = True                       # Add random jitter to backoff
    jitter_factor: float = 0.1                # Jitter as fraction of backoff (0.0-1.0)
    retry_on_oom: bool = True                 # Retry on CUDA OOM errors
    reduce_batch_on_oom: bool = True          # Reduce batch size on OOM retry


@dataclass
class EvaluationConfig:
    """Configuration for evaluation during training."""
    eval_during_training: bool = False        # Enable evaluation (disabled by default - YAML controls)
    eval_metrics: Optional[str] = None        # Comma-separated metrics
    eval_frequency: int = 500                 # Evaluation frequency (steps)


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    quantization_aware: bool = False          # QAT training
    bit_width: int = 8                        # Quantization bits
    use_nvfp4: bool = False                   # NVFP4 training
    nvfp4_block_size: int = 16               # NVFP4 block size
    stochastic_rounding: bool = False         # Stochastic rounding
    use_hadamard_transform: bool = False      # Hadamard transforms
    use_torchao_nvfp4: bool = False          # TorchAO NVFP4


@dataclass
class HybridCachingConfig:
    """Configuration for hybrid KV + activation caching."""
    enabled: bool = False
    max_cache_size_gb: float = 1.0
    kv_cache_ratio: float = 0.7               # Ratio for KV vs activation cache
    eviction_policy: str = 'hybrid'           # 'lru', 'lfu', 'hybrid'
    prefetch_enabled: bool = True
    prefetch_lookahead: int = 2
    min_score_threshold: float = 0.1


@dataclass
class CudaStreamsConfig:
    """Configuration for CUDA stream optimizations."""
    enabled: bool = False
    num_streams: int = 4                      # Stream pool size
    use_event_timing: bool = True             # Use CUDA events for timing
    use_stream_pool: bool = True              # Reuse streams
    high_priority_transfers: bool = True      # Priority for CPU<->GPU transfers


@dataclass
class OverlappedCheckpointingConfig:
    """Configuration for overlapped gradient checkpointing."""
    enabled: bool = False
    stream_overlap: bool = True               # Use CUDA streams for overlapping
    target_layers: str = 'layers'             # Which layers to apply to


@dataclass
class DoubleCheckpointingConfig:
    """Configuration for double (nested) gradient checkpointing."""
    enabled: bool = False
    coarse_checkpoint_interval: int = 8       # Outer checkpoint interval
    fine_checkpoint_interval: int = 2         # Inner checkpoint interval
    use_cuda_streams: bool = True


@dataclass
class FP8Config:
    """Configuration for FP8 training (Hopper/Ada GPUs only).

    FP8 provides 2-3x speedup on supported hardware (H100, L40, RTX 4090+).

    Backward pass optimization (new):
    - backward_enabled: Enable FP8 for gradient computation
    - backward_format: Use e5m2 for gradients (higher dynamic range)
    - exclude_layers: Skip FP8 for sensitive layers
    """
    enabled: bool = False
    use_transformer_engine: bool = False
    format: str = 'e4m3'                      # 'e4m3' or 'e5m2' for forward
    margin: int = 0                           # Scale margin

    # Backward pass FP8 optimization
    backward_enabled: bool = False            # Enable FP8 for backward pass
    backward_format: str = 'e5m2'             # 'e5m2' recommended for gradients (higher range)
    exclude_layers: List[str] = field(default_factory=lambda: ['embedding', 'lm_head'])
    gradient_scaling_strategy: str = 'per_tensor'  # 'per_tensor' or 'per_channel'
    amax_history_len: int = 1024              # History for dynamic scaling
    amax_compute_algo: str = 'max'            # 'max' or 'most_recent'


@dataclass
class SYMIConfig:
    """Configuration for SYMI optimizer decoupling (arXiv 2504.19925).

    SYMI (State-Yield Method for MoE Inference/training) decouples optimizer
    states (momentum, variance) from expert parameters for ~30% training speedup.

    Key insight: Static partitioning of optimizer state across workers while
    allowing dynamic expert parameter placement reduces memory and sync overhead.

    Best for multi-GPU training. Limited benefit on single GPU.
    """
    enabled: bool = False                      # Enable SYMI decoupling
    num_partitions: int = 0                    # State partitions (0 = auto = num_gpus)
    sync_frequency: int = 100                  # Steps between full state sync
    use_async_sync: bool = True                # Async state transfer during backward
    gradient_averaging: str = 'partition'      # 'partition' or 'global' averaging
    state_precision: str = 'fp32'              # Optimizer state dtype
    enable_checkpointing: bool = True          # Checkpoint partitioned states


@dataclass
class Sparse24Config:
    """Configuration for 2:4 activation sparsity (arXiv 2503.16672).

    2:4 structured sparsity leverages NVIDIA Tensor Cores on Ampere+ GPUs.
    Every 4 contiguous elements have exactly 2 zeros, enabling hardware-
    accelerated sparse matmul with 2x throughput and near-lossless accuracy.

    Provides 1.2-1.3x speedup on Ampere+ GPUs (SM >= 8.0: A100, RTX 3090, etc.)
    Falls back to dense computation on older hardware.
    """
    enabled: bool = False                      # Enable 2:4 sparsity
    warmup_steps: int = 1000                   # Steps before enabling sparsity
    apply_to_gate: bool = True                 # Apply to gate projection
    apply_to_up: bool = True                   # Apply to up projection
    apply_to_down: bool = False                # Apply to down projection (usually dense)
    use_ste_scaling: bool = True               # Gradient scaling in STE
    ste_scale_factor: float = 1.0              # STE gradient multiplier
    sparsity_granularity: str = 'activation'   # 'activation' or 'weight' sparsity
    use_triton_kernel: bool = True             # Use Triton kernel (fallback: cuSPARSELt)
    log_sparsity_stats: bool = False           # Log sparsity statistics


@dataclass
class StableMoEConfig:
    """Configuration for Stable-MoE routing (arXiv 2512.06784).

    Uses Lyapunov-based load balancing with adaptive capacity factors and
    temperature annealing for 40% throughput improvement over fixed capacity.

    Key innovation: Control-theoretic approach maintains expert utilization
    within target bounds with stability guarantees, replacing fixed aux losses.
    """
    enabled: bool = False                      # Enable Stable-MoE routing
    target_utilization: float = 0.0            # Target per-expert util (0 = auto = 1/E)
    utilization_tolerance: float = 0.1         # Allowed deviation from target
    adaptation_rate: float = 0.01              # Lyapunov controller gain

    # Temperature annealing for exploration/exploitation
    temperature_init: float = 1.0              # Initial routing temperature
    temperature_min: float = 0.1               # Minimum temperature
    temperature_decay: float = 0.9999          # Per-step temperature decay

    # Adaptive capacity bounds
    capacity_min: float = 1.0                  # Minimum capacity factor
    capacity_max: float = 2.0                  # Maximum capacity factor

    # Metrics
    log_utilization_histogram: bool = True     # Log per-expert utilization
    log_capacity_factors: bool = True          # Log adaptive capacities
    log_temperature: bool = True               # Log temperature schedule


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
    run_lr_finder: bool = False              # Run LR Finder before training
    start_lr: float = 1e-8                   # Starting LR for search
    end_lr: float = 1.0                      # Ending LR for search
    num_iterations: int = 100                # Number of iterations to test
    suggestion_method: str = 'steepest'      # Method for suggesting LR
    use_suggested_lr: bool = False           # Automatically use suggested LR
    plot_path: Optional[str] = None          # Path to save plot
    smooth_beta: float = 0.98                # Loss smoothing factor
    stop_div_threshold: float = 4.0          # Stop if loss diverges


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory.

    Implements prioritized experience replay for continual learning.
    Samples with higher loss are replayed more frequently, with
    importance sampling weights to correct for the sampling bias.

    Priority Calculation:
        P(i) = priority_i^alpha / sum(priority_j^alpha)
        where alpha = priority_exponent

    Importance Sampling Weights:
        w_i = (N * P(i))^(-beta) / max(w)
        where beta = importance_weight_exponent
    """
    use_episodic_memory: bool = False         # Enable episodic memory (disabled by default)
    memory_capacity: int = 1000               # Memory capacity (buffer size)
    memory_selection_strategy: str = 'importance'  # Selection strategy: 'importance' or 'uniform'
    memory_importance_threshold: float = 0.5  # Importance threshold for filtering
    memory_retrieval_method: str = 'cosine'   # Retrieval method for similarity
    memory_replay_ratio: float = 0.2          # Ratio of replay samples per batch (0.2 = 20%)
    memory_replay_strategy: str = 'importance'  # Replay strategy: 'importance' or 'uniform'
    memory_adaptation_rate: float = 0.01      # Adaptation rate for priority updates
    memory_performance_window: int = 100      # Window for performance tracking
    task_id: int = 0                          # Task ID for multi-task learning
    silent_mode: bool = False                 # Suppress memory warnings in console
    enable_auto_grad_accumulation: bool = False  # Enable auto gradient accumulation adjustment

    # Prioritized replay parameters (Schaul et al., 2015)
    priority_exponent: float = 0.6            # Alpha: controls prioritization (0=uniform, 1=full priority)
    importance_weight_exponent: float = 0.4   # Beta: controls IS weight correction (0=none, 1=full)
    buffer_warmup_steps: int = 100            # Steps before replay starts (let buffer fill)
    store_aux_info: bool = False              # Store MoE routing info per sample (memory intensive)


@dataclass
class DataLoadingConfig:
    """Configuration for data loading parameters."""
    format_detection_samples: int = 10         # Number of files to sample for format detection
    fallback_data_paths: list = field(default_factory=lambda: [  # Fallback paths to search for data
        str(get_data_dir("processed")),
        str(get_data_dir("combined")),
        str(get_data_dir()),
        "./data/processed",
        "./data/combined",
        "./data",
        "../data/processed",
        "../data",
        "../../data"
    ])


@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_dir: str = field(default_factory=lambda: str(get_data_dir("processed")))  # Data directory
    max_length: int = 512                     # Max sequence length
    tokenizer_name: Optional[str] = None      # Tokenizer name or path
    max_samples: Optional[int] = None         # Max samples (testing)
    streaming: bool = False                   # Streaming loader (YAML controls)
    buffer_size: int = 50000                  # Streaming buffer size (optimized for LLM pretraining)
    num_workers: int = 0                      # Default 0 (safe). Set to 8+ in config for better throughput
    prefetch_factor: int = 4                  # Batches to prefetch per worker (4 is optimal)
    persistent_workers: bool = True           # Keep workers alive between epochs (avoids spawn overhead)
    padding_side: str = 'right'               # Tokenizer padding side
    truncation: bool = True                   # Enable truncation
    max_train_examples: Optional[int] = None  # Max training examples
    max_eval_examples: Optional[int] = None   # Max evaluation examples
    dataloader_drop_last: bool = False        # Drop last incomplete batch
    dataloader_pin_memory: bool = True        # Pin memory for faster GPU transfer (recommended)
    default_tokenizer_name: str = 'Qwen/Qwen2.5-0.5B'  # Default tokenizer if none specified

    # Sequence packing for improved throughput
    use_sequence_packing: bool = False        # Enable sequence packing
    packing_strategy: str = 'greedy'          # Packing strategy: 'greedy' or 'adaptive'
    max_tokens_per_batch: Optional[int] = None  # Max tokens per batch

    # Dataset splits and validation
    train_split: str = 'train'                # Training split name
    eval_split: str = 'validation'            # Evaluation split name
    auto_create_validation_split: bool = True # Auto-create validation split
    validation_split_ratio: float = 0.1       # Validation split ratio
    val_split_ratio: float = 0.1              # Alias for validation_split_ratio
    val_max_samples: Optional[int] = None     # Max validation samples

    # Additional dataloader settings
    dataloader_persistent_workers: bool = False  # Persistent workers
    dataloader_samples_per_file: int = 64     # Samples per file rotation
    samples_per_file: int = 64                # Alias for dataloader_samples_per_file
    use_streaming_tokenization: bool = False  # Use streaming tokenization
    enable_bucketing: bool = True             # Enable sequence bucketing
    dataset_name: Optional[str] = None        # Dataset name

    # Randomization control (NEW)
    shuffle_seed: Optional[int] = None        # Global shuffle seed (None = non-deterministic)
    enable_length_sorting: bool = True        # Enable length sorting in distributed mode
    disable_packing_length_sort: bool = False # Disable length sorting in packing
    examples_per_random_select: int = 100     # Examples per file selection (higher = better I/O locality, 50-200 recommended)

    # Indexed loader (map-style with true random shuffling)
    use_indexed_loader: bool = False          # Enable IndexedArrowDataset (true random access)
    indexed_num_bins: int = 8                 # Number of length bins for sampling
    indexed_cache_size: int = 50              # Arrow table LRU cache size per worker
    indexed_index_workers: Optional[int] = None  # Parallel workers for indexing (None = auto)

    # Fast startup options (reduce overhead) - PERF: defaults optimized for speed
    fast_startup: bool = False                # Skip all validation and stats for fastest startup
    skip_sequence_count: bool = True          # Skip dataset stats logging at startup (saves 10-30s)
    skip_dataloader_validation: bool = True   # Skip 5-batch validation at startup (saves 1-2s)


@dataclass
class MultiColumnDataConfig:
    """Configuration for multi-column data."""
    use_multi_column: bool = False            # Enable multi-column
    dataset_config: Optional[str] = None      # Dataset config file
    hf_dataset: Optional[str] = None          # HuggingFace dataset
    hf_dataset_config: Optional[str] = None   # HF dataset config
    column_names: Optional[str] = None        # Column names
    column_types: Optional[str] = None        # Column types
    column_roles: Optional[str] = None        # Column roles
    combine_strategy: str = 'concatenate'     # Combination strategy
    column_template: Optional[str] = None     # Column template


@dataclass
class ProgressiveTrainingConfig:
    """Configuration for progressive training features."""
    enable_progressive_training: bool = False    # Enable progressive training

    # Sequence length scaling (5.1 fixes)
    enable_sequence_scaling: bool = False
    initial_seq_length: int = 128
    final_seq_length: int = 2048
    length_schedule: str = "linear"
    length_growth_epochs: int = 10
    enable_length_bucketing: bool = False     # YAML controls

    # Difficulty scoring (5.2 fixes)
    enable_curriculum: bool = False
    curriculum_metric: str = "loss"
    enable_score_caching: bool = False        # YAML controls
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".cache" / "ava_difficulty"))
    cache_version: str = "v1.0"



@dataclass
class CalibrationConfig:
    """Configuration for enhanced calibration system for dynamic batching.

    The calibration system profiles GPU memory and throughput at training start
    to enable more accurate batch size predictions and better GPU utilization.
    """
    # Enable/disable calibration
    enabled: bool = True

    # Calibration phases
    run_memory_profiling: bool = True           # Profile memory at different batch/seq configs
    run_backward_profiling: bool = True         # Measure actual backward/forward memory ratio
    run_throughput_profiling: bool = True       # Profile throughput to find optimal batch size

    # Thorough profiling grid (7x7 = 49 combinations)
    batch_sizes_to_profile: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])
    seq_lengths_to_profile: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048])
    num_samples_per_config: int = 3             # Measurements per configuration

    # Persistence - both global cache AND run-local
    cache_enabled: bool = True
    cache_dir: str = "~/.cache/ava_calibration"  # Global cache for fast reuse
    save_to_run_dir: bool = True                # Also save to outputs/runs/<run>/calibration/
    cache_ttl_hours: int = 168                  # 1 week TTL

    # Safety
    max_calibration_time_seconds: int = 300     # 5 min max for thorough profiling


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: Optional[int] = None          # Batch size
    epochs: Optional[int] = None              # Number of epochs
    learning_rate: Optional[float] = None     # Learning rate
    gradient_accumulation: int = 1            # Gradient accumulation (legacy)
    gradient_accumulation_steps: int = 1      # Gradient accumulation steps (preferred)
    max_gradient_norm: float = 1.0            # Maximum gradient norm for clipping

    # Learning rate schedule
    warmup_steps: int = 2000                  # Number of warmup steps
    max_steps: Optional[int] = None           # Maximum training steps

    # Adaptive LR configuration
    adaptive_lr: dict = field(default_factory=dict)  # Adaptive learning rate settings

    # Progressive training
    progressive: ProgressiveTrainingConfig = field(default_factory=ProgressiveTrainingConfig)

    def __post_init__(self):
        """Validate training configuration values.

        FIX: Added comprehensive validations to catch config errors early.
        """
        # Batch size validation (if specified)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        # Gradient accumulation must be positive
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}")
        if self.gradient_accumulation <= 0:
            raise ValueError(f"gradient_accumulation must be > 0, got {self.gradient_accumulation}")

        # Learning rate validation (if specified)
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

        # Warmup steps must be non-negative
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps cannot be negative, got {self.warmup_steps}")

        # Max steps validation (if specified)
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0 if specified, got {self.max_steps}")

        # Epochs validation (if specified)
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError(f"epochs must be > 0 if specified, got {self.epochs}")

        # Max gradient norm must be positive
        if self.max_gradient_norm <= 0:
            raise ValueError(f"max_gradient_norm must be > 0, got {self.max_gradient_norm}")


@dataclass
class OutputConfig:
    """Configuration for output handling."""
    output_dir: str = field(default_factory=lambda: str(get_outputs_dir()))  # Output directory
    save_every: int = 100                     # Save frequency
    resume: Optional[str] = None              # Resume checkpoint
    fresh_start: bool = False                 # Force fresh start, ignore checkpoints


@dataclass
class RunManagementConfig:
    """Configuration for run management."""
    run_name: Optional[str] = None            # Custom run name
    run_tags: Optional[str] = None            # Run tags
    run_description: Optional[str] = None     # Run description
    disable_run_manager: bool = False         # Disable run manager


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases."""
    use_wandb: bool = False                   # Enable WandB (YAML controls)
    disable_wandb: bool = False               # Disable WandB
    wandb_offline: bool = False               # Force WandB offline mode
    wandb_project: str = 'Ava'                # WandB project
    wandb_name: Optional[str] = None          # WandB run name
    wandb_tags: List[str] = field(default_factory=lambda: ['moe', 'training'])
    wandb_log_freq: int = 10                  # Log frequency
    wandb_cache_size: int = 2000             # Cache size
    wandb_cache_flush_interval: int = 50     # Cache flush interval


@dataclass
class CoherenceConfig:
    """Configuration for coherence measurement during training evaluation."""
    enabled: bool = True                      # Enable coherence measurement (detect issues early)
    eval_every_n_steps: int = 500             # Measure coherence every N steps
    num_samples: int = 10                     # Number of samples to generate for evaluation
    max_generation_length: int = 256          # Max tokens to generate

    # Metric weights for aggregate score
    perplexity_weight: float = 0.3
    repetition_weight: float = 0.2
    flow_weight: float = 0.25
    topic_weight: float = 0.25

    # Thresholds
    max_perplexity: float = 100.0             # Cap perplexity for scoring
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    min_sentence_length: int = 5              # Min tokens to consider a sentence

    # Generation parameters
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50

    # Logging
    log_to_wandb: bool = True                 # Log coherence metrics to WandB
    log_to_console: bool = True               # Log coherence metrics to console

    def __post_init__(self):
        """Validate configuration values."""
        weights_sum = (
            self.perplexity_weight +
            self.repetition_weight +
            self.flow_weight +
            self.topic_weight
        )
        if not (0.99 <= weights_sum <= 1.01):  # Allow small floating point tolerance
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"CoherenceConfig weights sum to {weights_sum:.4f}, not 1.0. "
                f"Coherence scores may be outside [0,1] range."
            )


@dataclass
class ModelSelectionConfig:
    """Configuration for multi-metric model selection.

    Combines val_loss, coherence_score, and perplexity into a weighted
    quality score for determining the best model checkpoint.
    """
    # Enable/disable multi-metric selection
    enabled: bool = True

    # Metric weights (should sum to 1.0 for normalized scoring)
    val_loss_weight: float = 0.5              # Weight for validation loss (lower is better)
    coherence_score_weight: float = 0.3       # Weight for coherence (higher is better)
    perplexity_weight: float = 0.2            # Weight for perplexity (lower is better)

    # Normalization settings
    perplexity_cap: float = 100.0             # Cap perplexity for normalization
    val_loss_cap: float = 10.0                # Cap val_loss for normalization

    # Selection behavior
    higher_is_better: bool = True             # Quality score interpretation
    require_all_metrics: bool = False         # Require all metrics present
    fallback_to_val_loss: bool = True         # Use val_loss alone if others missing


@dataclass
class DiagnosticsConfig:
    """Configuration for detailed diagnostic logging.

    Enables per-layer gradients, expert routing stats, memory breakdown,
    and timing profiling for in-depth training analysis.
    """
    # Master enable
    enabled: bool = False

    # Per-layer gradient statistics
    enable_per_layer_gradients: bool = False
    per_layer_log_freq: int = 500             # Steps between per-layer logs
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

    # Timing profiling (each timing log requires cuda.synchronize - use sparingly)
    enable_timing_profiling: bool = False
    timing_log_freq: int = 500  # Increased from 100 to reduce sync overhead
    profile_forward: bool = True
    profile_backward: bool = True
    profile_optimizer_step: bool = True
    profile_data_loading: bool = True


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed distributed training."""
    use_deepspeed: bool = False               # Enable DeepSpeed
    config_file: Optional[str] = None         # DeepSpeed JSON config file path
    zero_stage: int = 2                       # ZeRO optimization stage (0, 1, 2, 3)
    cpu_offload: bool = False                 # Enable CPU offloading
    nvme_offload: bool = False                # Enable NVMe offloading
    gradient_accumulation_steps: int = 1      # Gradient accumulation
    train_batch_size: Optional[int] = None    # Global batch size
    micro_batch_size: Optional[int] = None    # Micro batch size
    enable_mixed_precision: bool = False      # Enable FP16/BF16 (YAML controls)
    precision_type: str = 'fp16'              # Precision type: fp16, bf16, fp32

    # ZeRO-specific settings
    zero_allow_untested_optimizer: bool = False  # YAML controls
    zero_force_ds_cpu_optimizer: bool = False
    zero_reduce_scatter: bool = False         # YAML controls
    zero_overlap_comm: bool = False           # YAML controls
    zero_contiguous_gradients: bool = False   # YAML controls
    zero_reduce_bucket_size: int = 500000000       # 500MB
    zero_allgather_bucket_size: int = 500000000    # 500MB
    zero_stage3_prefetch_bucket_size: int = 500000000  # 500MB
    zero_stage3_param_persistence_threshold: int = 1000000

    # Communication settings
    communication_data_type: str = 'fp32'     # Communication data type
    allreduce_partitions: bool = False        # YAML controls
    allgather_partitions: bool = False        # YAML controls
    overlap_comm: bool = False                # YAML controls
    wall_clock_breakdown: bool = False        # Enable timing breakdown

    # Advanced features
    activation_checkpointing: bool = False    # Enable activation checkpointing
    partition_activations: bool = False       # Partition activations
    cpu_checkpointing: bool = False          # CPU activation checkpointing
    contiguous_memory_optimization: bool = False
    synchronize_dp_processes: bool = False    # YAML controls

    # Pipeline parallelism
    pipeline_parallel_size: int = 1          # Pipeline parallel size
    gradient_clipping: Optional[float] = None # Gradient clipping value

    # Monitoring and debugging
    monitor_config: Dict[str, Any] = field(default_factory=dict)
    tensorboard: Dict[str, Any] = field(default_factory=dict)
    wandb_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradientSyncConfig:
    """Configuration for distributed gradient synchronization.

    Provides three modes for gradient all-reduce:
    - 'standard': Default DDP bucket-based all-reduce
    - 'overlapped': Overlap gradient sync with backward computation
    - 'fused': Combine multiple buckets into fewer all-reduce operations

    Fused mode provides 5-15% speedup in multi-GPU training by reducing
    communication overhead and network round trips.
    """
    mode: str = 'standard'                    # 'standard', 'overlapped', 'fused'

    # Fused all-reduce settings (when mode='fused')
    fused_bucket_mb: float = 100.0            # Maximum fused bucket size in MB
    fusion_factor: int = 4                    # Number of DDP buckets to fuse
    async_allreduce: bool = True              # Use async all-reduce

    # Overlapped sync settings (when mode='overlapped')
    bucket_size_mb: float = 25.0              # Bucket size for overlapped sync

    # Common settings
    use_coalesced_ops: bool = True            # Use coalesced tensor operations


@dataclass
class LayerwiseOptimizerConfig:
    """Configuration for layer-wise optimizer updates.

    Enables overlapping optimizer updates with backward computation by
    beginning updates as soon as gradients for each layer become available.

    Benefits:
    - 10-20% speedup by overlapping optimizer with backward
    - 10-30% memory savings with early gradient release
    """
    enabled: bool = False                     # Enable layer-wise updates
    bucket_size_mb: float = 25.0              # Update bucket size in MB
    release_gradients_early: bool = True      # Free gradients after update
    align_with_ddp_buckets: bool = True       # Align with DDP gradient buckets


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA graph capture of training steps.

    CUDA graphs eliminate kernel launch overhead by capturing the entire
    forward+backward+optimizer sequence as a single graph.

    Benefits:
    - 15-25% speedup for small batch sizes
    - Reduced CPU overhead

    Limitations:
    - Incompatible with dynamic shapes
    - Incompatible with layer-wise optimizer
    - Incompatible with fused all-reduce
    """
    enabled: bool = False                     # Enable CUDA graphs
    capture_backward: bool = True             # Include backward in graph
    capture_optimizer_step: bool = True       # Include optimizer in graph
    max_cached_graphs: int = 4                # Max graphs for different shapes
    use_memory_pool: bool = True              # Pre-allocate memory pool
    warmup_steps: int = 3                     # Warmup steps before capture


@dataclass
class PerformanceConfig:
    """Configuration for performance modes and hardware optimizations."""
    ultra_fast_mode: bool = False             # Ultra fast mode
    fast_progress: bool = False               # Fast progress mode
    minimal_progress: bool = False            # Minimal progress mode
    no_sync: bool = False                     # No CUDA sync mode
    express_mode: bool = False                # Express mode

    # TF32 and hardware optimizations (NEW)
    enable_tf32: bool = True                  # Enable TF32 on Ampere+ GPUs for faster matmul
    float32_matmul_precision: str = 'high'    # Options: 'highest', 'high', 'medium'
    enable_cudnn_benchmark: bool = True       # Auto-tune cuDNN kernels
    cudagraph_skip_dynamic_shapes: bool = True   # Skip dynamic shapes in CUDAGraph
    cudagraph_dynamic_shape_warn_limit: Optional[int] = None  # Warning limit for dynamic shapes
    torchinductor_max_autotune: int = 0       # TorchInductor autotune level (0-4)


@dataclass
class OptimizationsConfig:
    """Configuration for all training optimizations (Phase 1, 2, 3)."""

    # Phase 1: Quick Wins
    torchinductor_autotune: int = 1            # 0=off, 1=basic, 2=aggressive

    # Memory Management
    memory_headroom_gb: float = 3.0            # Reserve headroom for safety
    memory_cleanup_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'warning': 0.85,   # Trigger warning cleanup
        'critical': 0.90,  # Trigger critical cleanup
        'emergency': 0.95  # Trigger emergency cleanup
    })

    # Phase 2: Expert Offloading Optimizations
    expert_prefetch: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                # Multi-stage async prefetch
        'lookahead': 2,                 # Prefetch lookahead
        'use_multiple_streams': True    # Use multiple CUDA streams
    })

    expert_cache: Dict[str, Any] = field(default_factory=lambda: {
        'auto_limit': True,             # Auto-limit cache size
        'use_lru_eviction': True,       # LRU eviction policy
        'clear_after_optimizer_step': True  # Clear after optimizer step
    })

    # Phase 2: Checkpoint Optimizations
    checkpoint: Dict[str, Any] = field(default_factory=lambda: {
        'async_saving': True            # Save checkpoints in background thread
    })

    # Phase 2: GPU Memory Cleanup
    memory_cleanup: Dict[str, Any] = field(default_factory=lambda: {
        'fast_mode': True,              # Reduced cleanup rounds
        'remove_sleep': True            # Remove sleep between cleanup
    })

    # Proactive Memory Fragmentation Cleanup
    proactive_memory_cleanup: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                 # Enable proactive fragmentation cleanup
        'fragmentation_threshold': 0.30, # Trigger cleanup when fragmentation exceeds this (0.0-1.0)
        'cleanup_frequency': 2000,       # Check fragmentation every N steps
        'cleanup_after_validation': True, # Also cleanup after validation runs
        'cleanup_after_generation': True  # Also cleanup after generation runs
    })

    # Phase 2: Data Loading
    dataloader: Dict[str, Any] = field(default_factory=lambda: {
        'adaptive_file_reading': True,  # Adjust samples per file
        'adaptive_multipliers': {
            'large_files': 4,           # Multiplier for files >10MB
            'medium_files': 2,          # Multiplier for files >1MB
            'small_files': 1            # Multiplier for files <1MB
        }
    })

    # Phase 2: Gradient Checkpointing
    gradient_checkpointing: Dict[str, Any] = field(default_factory=lambda: {
        'selective': False,             # Checkpoint everything for max memory savings
        'checkpoint_attention': True    # Checkpoint attention layers
    })

    # Router Optimizations
    router: Dict[str, Any] = field(default_factory=lambda: {
        'cache_hash_on_gpu': True,      # Compute routing cache hash on GPU
        'compile_routers': True,        # torch.compile routers
        'compile_mode': 'default',      # 'default' or 'reduce-overhead'
        'compile_dynamic': True         # Handle variable sequence lengths
    })


@dataclass
class LoggingConfig:
    """Configuration for logging and observability.

    For fastest training, disable all logging:
        logging:
          disabled: true

    Or selectively disable specific logging types:
        logging:
          console_enabled: false
          file_enabled: false
          progress_bar_enabled: false
          wandb:
            enabled: false
          tensorboard:
            enabled: false
    """
    # =========================================================================
    # Master disable flags (for faster training)
    # =========================================================================
    disabled: bool = False                    # Master disable ALL logging (fastest mode)
    silent_mode: bool = False                 # Only critical errors (minimal output)

    # =========================================================================
    # Console/File logging controls
    # =========================================================================
    console_enabled: bool = True              # Enable console output
    file_enabled: bool = True                 # Enable file logging
    progress_bar_enabled: bool = True         # Enable tqdm progress bar
    tqdm_update_interval: int = 10            # Update progress bar every N batches (reduces I/O overhead)
    step_logging_enabled: bool = True         # Enable per-step console output
    epoch_logging_enabled: bool = True        # Enable epoch summary logging

    # =========================================================================
    # Metric category controls (affects both console and external logging)
    # =========================================================================
    log_training_metrics: bool = True         # train/loss, train/lr, train/batch_size
    log_gradient_metrics: bool = True         # gradients/norm, gradients/avg, etc.
    log_moe_metrics: bool = True              # moe/expert_load, moe/routing_entropy
    log_validation_metrics: bool = True       # validation/loss, validation/epoch
    log_coherence_metrics: bool = True        # coherence/* metrics
    log_quality_metrics: bool = True          # model_selection/* quality scores
    log_memory_metrics: bool = True           # memory/* breakdown stats
    log_timing_metrics: bool = True           # timing/* profiling stats
    log_generation_samples: bool = True       # Text generation samples

    # =========================================================================
    # WandB logging controls (fine-grained)
    # =========================================================================
    wandb_enabled: bool = True                # Master WandB enable (overrides wandb.enabled)
    wandb_log_training: bool = True           # Log train/* metrics to WandB
    wandb_log_gradients: bool = True          # Log gradients/* to WandB
    wandb_log_moe: bool = True                # Log moe/* metrics to WandB
    wandb_log_validation: bool = True         # Log validation/* to WandB
    wandb_log_coherence: bool = True          # Log coherence/* to WandB
    wandb_log_quality: bool = True            # Log model_selection/* to WandB
    wandb_log_memory: bool = True             # Log memory/* breakdown to WandB
    wandb_log_timing: bool = True             # Log timing/* profile to WandB
    wandb_log_generations: bool = True        # Log generation table to WandB
    wandb_log_per_layer_grads: bool = False   # Log per-layer gradients (expensive)
    wandb_log_routing_diagnostics: bool = False  # Log routing/* (expensive)
    wandb_log_model_topology: bool = False    # Log model graph (one-time)

    # =========================================================================
    # TensorBoard logging controls
    # =========================================================================
    tensorboard_enabled: bool = True          # Master TensorBoard enable
    tensorboard_log_training: bool = True     # Log train/* to TensorBoard
    tensorboard_log_gradients: bool = True    # Log gradients/* to TensorBoard
    tensorboard_log_moe: bool = True          # Log moe/* to TensorBoard
    tensorboard_log_validation: bool = True   # Log validation/* to TensorBoard

    # =========================================================================
    # Verbosity settings
    # =========================================================================
    verbosity: str = 'info'                   # Log level: 'debug', 'info', 'warning', 'error'
    console_level: str = 'info'               # Console log level (can be different from file)
    file_level: str = 'debug'                 # File log level (more detailed)

    # =========================================================================
    # Monitoring frequencies (in steps) - higher values = less overhead
    # =========================================================================
    log_interval: int = 100                   # Main logging interval (GPU->CPU sync for loss)
    verbose_log_interval: int = 500           # Show detailed INFO logs every N steps (0 = never)
    log_mode: str = 'tqdm'                    # 'tqdm' (progress bar only) or 'verbose' (tqdm + INFO logs)
    metrics_log_freq: int = 500               # Reduced frequency to minimize sync overhead
    memory_check_freq: int = 2000             # Reduced frequency to minimize sync overhead
    health_summary_freq: int = 500            # How often to log training health summary
    moe_metrics_freq: int = 5000              # Reduced frequency to minimize sync overhead
    routing_metrics_freq: int = 0             # Per-layer routing metrics (0=disabled, 1000+ recommended if enabled)

    # =========================================================================
    # Feature flags
    # =========================================================================
    enable_timing_breakdown: bool = True      # Log step-level timing (data, forward, backward, optimizer)
    enable_memory_profiling: bool = True      # Enable detailed memory profiling
    enable_health_summaries: bool = True      # Enable periodic health summary logs
    log_tensor_shapes: bool = True            # Log tensor shapes on errors (OOM, NaN)
    log_checkpoint_validation: bool = True    # Validate checkpoints after save
    save_sample_generations: bool = True      # Save validation generation samples to file

    # Structured logging
    use_structured_logging: bool = True       # Use structured logs with contextual fields
    log_format: str = 'default'               # Log format: 'default', 'json', 'structured'

    # Legacy fields (kept for backward compatibility)
    log_gradients_to_wandb: bool = False      # Deprecated: use wandb_log_per_layer_grads
    log_model_topology: bool = False          # Deprecated: use wandb_log_model_topology


@dataclass
class DevLogConfig:
    """Configuration for development logging to identify performance bottlenecks."""
    enabled: bool = False                     # Enable dev logging
    show_file_timings: bool = True            # Show timing for each file read
    show_batch_timings: bool = True           # Show timing for each batch operation
    show_step_breakdown: bool = True          # Show breakdown of step components
    report_interval: int = 100                # Log timing every N steps


@dataclass
class KernelOptimizationConfig:
    """Configuration for low-level kernel optimizations.

    Controls Triton kernel usage, dispatch strategies, and GPU optimizations.
    These settings can provide significant throughput improvement when properly tuned.
    """
    # Router/Gating kernel optimizations
    router_kernel_mode: str = 'auto'          # 'auto', 'triton', 'pytorch'
    use_fused_softmax_topk: bool = True       # Fused softmax + top-k kernel
    router_block_size: int = 4                # Tokens per thread block (1-8)

    # Expert computation optimizations
    use_sparse_expert_dispatch: bool = False  # Sparse vs dense dispatch (sparse=memory efficient)
    use_fused_activations: bool = True        # Fused SwiGLU/GeGLU kernels
    use_selective_expert_loading: bool = False # On-demand expert loading for large E

    # Capacity limiting
    use_vectorized_capacity: bool = True      # Vectorized vs loop-based capacity limiting

    # Advanced optimizations (experimental)
    use_fused_moe_kernel: bool = False        # Mega kernel (routing + expert in one)
    enable_kernel_profiling: bool = False     # Profile kernel execution times


@dataclass
class BasicModelConfig:
    """Configuration for model architecture parameters (basic version for legacy compatibility)."""
    vocab_size: int = 32000                   # Vocabulary size
    hidden_size: int = 4096                   # Hidden dimension
    num_experts: Optional[int] = None         # Number of experts for MoE
    num_layers: int = 32                      # Number of layers
    num_attention_heads: int = 32             # Number of attention heads
    intermediate_size: int = 11008            # FFN intermediate size
    dropout: float = 0.1                      # Dropout rate


@dataclass
class EnhancedTrainingConfig:
    """Main configuration class combining all sub-configs."""
    config_file: str                          # Required config file

    # Hardware configuration
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Feature configurations
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)
    memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    model: BasicModelConfig = field(default_factory=BasicModelConfig)
    adaptive_mtp: AdaptiveMTPConfig = field(default_factory=AdaptiveMTPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dev_log: DevLogConfig = field(default_factory=DevLogConfig)
    optimizations: OptimizationsConfig = field(default_factory=OptimizationsConfig)

    # Model selection and diagnostics
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # Enhanced features (supports both losses and enhanced_features.losses paths)
    enhanced_features: Optional[Dict[str, Any]] = None  # type: ignore[assignment]

    # Data configurations
    data: DataConfig = field(default_factory=DataConfig)
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)
    multi_column_data: MultiColumnDataConfig = field(default_factory=MultiColumnDataConfig)

    # Training configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run_management: RunManagementConfig = field(default_factory=RunManagementConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)  # type: ignore[call-overload]
    kernel_optimization: KernelOptimizationConfig = field(default_factory=KernelOptimizationConfig)

    # Special flags
    enable_all_features: bool = False         # Enable all features
    multi_task: bool = False                  # Multi-task learning


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
                                    help='Run LR Finder before training to find optimal learning rate')
        lr_finder_group.add_argument('--lr-finder-start', type=float, default=1e-8,
                                    help='Starting LR for LR finder (default: 1e-8)')
        lr_finder_group.add_argument('--lr-finder-end', type=float, default=1.0,
                                    help='Ending LR for LR finder (default: 1.0)')
        lr_finder_group.add_argument('--lr-finder-iterations', type=int, default=100,
                                    help='Number of iterations for LR finder (default: 100)')
        lr_finder_group.add_argument('--lr-finder-method', type=str, default='steepest',
                                    choices=['steepest', 'minimum', 'valley'],
                                    help='Method for suggesting LR from results (default: steepest)')
        lr_finder_group.add_argument('--lr-finder-use-suggested', action='store_true',
                                    help='Automatically use the suggested LR from LR finder')
        lr_finder_group.add_argument('--lr-finder-plot-path', type=str, default=None,
                                    help='Path to save LR finder plot (default: auto-generated in run dir)')

        # === DATA ARGUMENTS ===
        data_group = parser.add_argument_group('Data Configuration')
        data_group.add_argument('--data-dir', type=str,
                               default=str(get_data_dir("processed")),
                               help='Directory containing preprocessed training data')
        data_group.add_argument('--max-length', type=int, default=512,
                               help='Maximum sequence length')
        data_group.add_argument('--max-samples', type=int, default=None,
                               help='Maximum number of training samples to load (for testing)')
        data_group.add_argument('--streaming', action='store_true', default=True,
                               help='Use streaming data loader for large datasets (default: True)')
        data_group.add_argument('--no-streaming', dest='streaming', action='store_false',
                               help='Disable streaming and load all data into memory')
        data_group.add_argument('--buffer-size', type=int, default=50000,
                               help='Buffer size for streaming data loader (default: 50000, optimized for LLM pretraining)')
        data_group.add_argument('--num-workers', type=int, default=8,
                               help='Number of parallel data loading workers (default: 8)')
        data_group.add_argument('--prefetch-factor', type=int, default=4,
                               help='Number of batches to prefetch per worker (default: 4)')
        data_group.add_argument('--no-persistent-workers', dest='persistent_workers', action='store_false', default=True,
                               help='Disable persistent workers (workers restart each epoch)')

        # === MULTI-COLUMN DATA ARGUMENTS ===
        multi_col_group = parser.add_argument_group('Multi-Column Data')
        multi_col_group.add_argument('--use-multi-column', action='store_true',
                                    help='Enable multi-column data loading')
        multi_col_group.add_argument('--dataset-config', type=str, default=None,
                                    help='Path to dataset configuration file for multi-column loading')
        multi_col_group.add_argument('--hf-dataset', type=str, default=None,
                                    help='HuggingFace dataset name (e.g., HuggingFaceM4/FineVision)')
        multi_col_group.add_argument('--hf-dataset-config', type=str, default=None,
                                    help='HuggingFace dataset configuration/subset')
        multi_col_group.add_argument('--column-names', type=str, default=None,
                                    help='Comma-separated list of column names to use')
        multi_col_group.add_argument('--column-types', type=str, default=None,
                                    help='Comma-separated list of column types (text,numeric,image,etc)')
        multi_col_group.add_argument('--column-roles', type=str, default=None,
                                    help='Comma-separated list of column roles (input,target,auxiliary)')
        multi_col_group.add_argument('--combine-strategy', type=str, default='concatenate',
                                    choices=['concatenate', 'separate', 'template'],
                                    help='Strategy for combining multiple input columns')
        multi_col_group.add_argument('--column-template', type=str, default=None,
                                    help='Template string for combining columns (e.g., "Question: {question}\\nAnswer: {answer}")')

        # === TRAINING ARGUMENTS ===
        training_group = parser.add_argument_group('Training Parameters')
        training_group.add_argument('--batch-size', type=int, default=None,
                                   help='Batch size (overrides config)')
        training_group.add_argument('--epochs', type=int, default=None,
                                   help='Number of epochs (overrides config)')
        training_group.add_argument('--learning-rate', type=float, default=None,
                                   help='Learning rate (overrides config)')
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
                                 help='Force fresh start, ignore any existing checkpoints')

        # === RUN MANAGEMENT ARGUMENTS ===
        run_group = parser.add_argument_group('Run Management')
        run_group.add_argument('--run-name', type=str, default=None,
                              help='Custom name for this training run')
        run_group.add_argument('--run-tags', type=str, default=None,
                              help='Comma-separated tags for this run (e.g., experiment,baseline,ablation)')
        run_group.add_argument('--run-description', type=str, default=None,
                              help='Description of this experiment')
        run_group.add_argument('--disable-run-manager', action='store_true',
                              help='Disable the run manager and use legacy output structure')

        # Weights & Biases arguments
        wandb_group = parser.add_argument_group('Weights & Biases')
        wandb_group.add_argument('--use-wandb', action='store_true', default=True,
                                help='Enable Weights & Biases logging (enabled by default)')
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
                                help='Weights & Biases logging frequency (steps)')
        wandb_group.add_argument('--wandb-cache-size', type=int, default=2000,
                                help='Weights & Biases cache size for offline resilience')
        wandb_group.add_argument('--wandb-cache-flush-interval', type=int, default=50,
                                help='Weights & Biases cache flush interval')

        # DeepSpeed arguments
        ds_group = parser.add_argument_group('DeepSpeed Distributed Training')
        ds_group.add_argument('--use-deepspeed', action='store_true',
                             help='Enable DeepSpeed distributed training')
        ds_group.add_argument('--deepspeed-config', type=str, default=None,
                             help='Path to DeepSpeed JSON configuration file')
        ds_group.add_argument('--zero-stage', type=int, default=2, choices=[0, 1, 2, 3],
                             help='ZeRO optimization stage (0=disabled, 1=optimizer, 2=optimizer+gradients, 3=all)')
        ds_group.add_argument('--cpu-offload', action='store_true',
                             help='Enable CPU offloading for optimizer states')
        ds_group.add_argument('--nvme-offload', action='store_true',
                             help='Enable NVMe offloading for large models')
        ds_group.add_argument('--ds-gradient-accumulation', type=int, default=1,
                             help='DeepSpeed gradient accumulation steps')
        ds_group.add_argument('--train-batch-size', type=int, default=None,
                             help='Global training batch size (for DeepSpeed)')
        ds_group.add_argument('--micro-batch-size', type=int, default=None,
                             help='Micro batch size per GPU (for DeepSpeed)')
        ds_group.add_argument('--ds-precision', type=str, default='fp16',
                             choices=['fp16', 'bf16', 'fp32'],
                             help='Mixed precision type for DeepSpeed')
        ds_group.add_argument('--activation-checkpointing', action='store_true',
                             help='Enable activation checkpointing to save memory')
        ds_group.add_argument('--partition-activations', action='store_true',
                             help='Partition activations across GPUs')
        ds_group.add_argument('--cpu-checkpointing', action='store_true',
                             help='Store activation checkpoints on CPU')
        ds_group.add_argument('--pipeline-parallel-size', type=int, default=1,
                             help='Pipeline parallelism size')
        ds_group.add_argument('--wall-clock-breakdown', action='store_true',
                             help='Enable DeepSpeed wall clock breakdown for profiling')

        # Performance mode arguments
        perf_group = parser.add_argument_group('Performance Modes')
        perf_group.add_argument('--ultra-fast-mode', action='store_true',
                               help='Ultra-fast mode: disable all logging for maximum training speed')
        perf_group.add_argument('--fast-progress', action='store_true',
                               help='Fast-progress mode: enhanced progress bar with real-time loss')
        perf_group.add_argument('--minimal-progress', action='store_true',
                               help='Minimal-progress mode: ultra-compact progress display')
        perf_group.add_argument('--no-sync', action='store_true',
                               help='No-sync mode: disable CUDA synchronization for maximum speed')
        perf_group.add_argument('--express-mode', action='store_true',
                               help='Express mode: optimized async logging with reduced frequency')

        return parser

    def parse_args_to_config(self, args: argparse.Namespace) -> EnhancedTrainingConfig:
        """Convert parsed arguments to structured configuration."""

        # Handle enable-all-features flag
        if args.enable_all_features:
            self._enable_all_features(args)

        # Load YAML to extract hardware config and dev_log config
        yaml_config = self.load_yaml_config(args.config)
        yaml_dict = yaml_config.to_dict()

        hardware_dict = yaml_dict.get('hardware', {})
        hardware_config = HardwareConfig(
            device=hardware_dict.get('device', 'cuda'),
            mixed_precision=hardware_dict.get('mixed_precision', 'fp32'),
            compile=hardware_dict.get('compile', False)
        )

        # Load dev_log config from YAML
        dev_log_dict = yaml_dict.get('dev_log', {})
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
            multi_task=False,  # Multi-task learning arguments removed

            # Hardware configuration
            hardware=hardware_config,

            # Use default configs for removed features
            architecture=ArchitectureConfig(),
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

            training=TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                gradient_accumulation=args.gradient_accumulation
            ),

            output=OutputConfig(
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

        This is the new recommended way to load configuration that:
        - Loads all YAML fields dynamically (no predefined structure needed)
        - Merges command-line argument overrides on top
        - Returns a DynamicConfig object with dot notation access

        Args:
            args: Parsed command-line arguments

        Returns:
            DynamicConfig object with merged YAML + CLI configuration

        Example:
            config = manager.create_unified_config(args)
            batch_size = config.training.batch_size
            learning_rate = config.training.learning_rate
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
            if args.gradient_accumulation != 1:  # 1 is the default
                yaml_config.training.gradient_accumulation_steps = args.gradient_accumulation
        else:
            # Create training section if it doesn't exist
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
            if args.data_dir != default_data_dir:  # Not default
                yaml_config.data.data_dir = args.data_dir
            if args.max_length != 512:  # Not default
                yaml_config.data.max_length = args.max_length
            if args.max_samples is not None:
                yaml_config.data.max_samples = args.max_samples
        else:
            # Create data section if it doesn't exist
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
            if args.output_dir != default_output_dir:  # Not default
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
        # Note: Most enhanced features have been removed. This function is kept for compatibility.
        # Currently no features to enable beyond what's in YAML configs.
        pass

    def _build_feature_dependencies(self) -> Dict[str, List[str]]:
        """Build feature dependency mapping."""
        # Most features have been removed. Keeping empty dict for compatibility.
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

        # Helper function to safely get nested attributes
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
        if safe_get(config, 'deepspeed.use_deepspeed', False):
            config_file = safe_get(config, 'deepspeed.config_file')
            if config_file and isinstance(config_file, (str, Path)) and not Path(config_file).exists():
                messages.append(f"DeepSpeed config file not found: {config_file}")

            zero_stage = safe_get(config, 'deepspeed.zero_stage', 0)
            cpu_offload = safe_get(config, 'deepspeed.cpu_offload', False)
            nvme_offload = safe_get(config, 'deepspeed.nvme_offload', False)

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
            safe_get(config, 'performance.ultra_fast_mode', False),
            safe_get(config, 'performance.fast_progress', False),
            safe_get(config, 'performance.minimal_progress', False),
            safe_get(config, 'performance.express_mode', False)
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