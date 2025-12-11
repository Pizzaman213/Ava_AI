"""
Training Configuration Manager

This module handles all training configuration management including
enhanced feature flags, parameter validation, and configuration inheritance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml

# Import path utilities for relative path resolution
try:
    from src.Ava.utils.paths import get_project_root, get_data_dir, get_outputs_dir
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

    Example:
        config = DynamicConfig({'training': {'batch_size': 32}})
        config.training.batch_size  # Returns 32
        config['training']['batch_size']  # Also returns 32
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """
        Initialize DynamicConfig from a dictionary.

        Args:
            data: Dictionary of configuration values
        """
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    # Recursively convert nested dicts to DynamicConfig
                    setattr(self, key, DynamicConfig(value))
                else:
                    setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """
        Allow accessing any attribute dynamically.

        Returns None for missing attributes to allow safe access to optional
        config fields. Use .get() with a default or hasattr() if you need
        to distinguish between None values and missing attributes.
        """
        # Return None for missing attributes - allows safe optional config access
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
    use_torch_compile: bool = False           # Enable torch.compile
    enable_cudagraphs_safe_routing: bool = False  # Enable CUDA graphs safe routing (20-30% speedup)
    use_flash_attention: bool = True          # Use flash attention
    gradient_checkpointing: bool = True       # Enable gradient checkpointing
    use_optimized_moe: bool = True            # Use optimized MoE implementation

    # Auxiliary losses
    router_z_loss_coef: float = 0.0001        # Router z-loss coefficient
    load_balance_loss_coef: float = 0.01      # Load balance loss coefficient
    diversity_loss_coef: float = 0.0001       # Diversity loss coefficient
    expert_dropout_loss_coef: float = 0.0     # Expert dropout loss coefficient
    router_jitter_noise: float = 0.01         # Router jitter noise for exploration

    # LoRA settings
    use_lora_experts: bool = False            # Use LoRA for experts
    lora_rank: int = 4                        # LoRA rank
    lora_alpha: int = 8                       # LoRA alpha
    freeze_lora_base: bool = False            # Freeze LoRA base weights

    # Expert offloading
    use_expert_offloading: bool = False       # Enable CPU offloading
    max_active_experts_gpu: int = 2           # Max experts on GPU
    offload_eviction_policy: str = 'lru'      # 'lru' or 'random'

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
        """Validate configuration values to prevent division by zero."""
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be > 0, got {self.num_attention_heads}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 512                     # Maximum generation length
    min_length: int = 10                      # Minimum generation length
    temperature: float = 1.2                  # Sampling temperature
    top_p: float = 0.95                       # Nucleus sampling threshold
    top_k: Optional[int] = 50                 # Top-k sampling (None = disabled)
    repetition_penalty: float = 1.1           # Repetition penalty (>1.0 discourages)
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
    """Configuration for RAG system."""
    use_rag: bool = False                     # Enable RAG (disabled by default - YAML controls)
    knowledge_base_path: Optional[str] = None  # KB path
    max_retrieved_docs: int = 5               # Max retrieved documents
    rag_fusion_type: str = 'attention'        # Fusion strategy


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
    """Configuration for FP8 training (Hopper/Ada GPUs only)."""
    enabled: bool = False
    use_transformer_engine: bool = False
    format: str = 'e4m3'                      # 'e4m3' or 'e5m2'
    margin: int = 0                           # Scale margin


@dataclass
class MoEMetricsConfig:
    """Configuration for MoE-specific metrics tracking."""
    track_expert_utilization: bool = True
    log_frequency: int = 50
    track_routing_decisions: bool = False
    track_load_balance: bool = True


@dataclass
class MoEMemoryOptimizationConfig:
    """
    Configuration for MoE memory optimization techniques.

    Enables advanced memory reduction strategies for sparse MoE models:
    - LoRA expert sharing: Shared base + low-rank deltas (40-60% savings)
    - CPU expert offloading: Keep inactive experts on CPU (50-80% savings)
    - Hierarchical loading: Cluster-based expert organization (30-50% savings)
    - Quantization: INT8/INT4 for inactive experts (50-75% savings)

    Combined savings: Up to 85-90% memory reduction!

    HYBRID MODE (NEW):
    - All three optimizations can now work together!
    - LoRA + Offloading + Quantization = 99.9%+ memory savings
    - Simply enable use_lora_experts, use_expert_offloading, and use_expert_quantization
    """
    # === LoRA Expert Sharing ===
    use_lora_experts: bool = False           # Enable LoRA-based expert parameter sharing
    lora_rank: int = 8                       # Rank of LoRA matrices (4-16, lower=more savings)
    lora_alpha: int = 16                     # LoRA scaling parameter (typically 2*rank)
    freeze_lora_base: bool = False           # Freeze shared base parameters

    # === CPU Expert Offloading ===
    use_expert_offloading: bool = False      # Enable CPU expert offloading
    max_active_experts_gpu: int = 4          # Max experts to keep on GPU
    offload_prefetch_lookahead: int = 2      # Number of experts to prefetch
    offload_eviction_policy: str = 'lru'     # Eviction policy: 'lru', 'frequency', 'hybrid'
    offload_pin_memory: bool = True          # Use pinned memory for faster transfers
    offload_async_transfers: bool = True     # Enable async GPU-CPU transfers

    # === Hierarchical Expert Loading ===
    use_hierarchical_experts: bool = False   # Enable hierarchical expert clustering
    num_expert_clusters: int = 4             # Number of expert clusters
    expert_clustering_method: str = 'random' # Clustering method: 'random', 'kmeans', 'functional'
    load_only_active_cluster: bool = True    # Load only active cluster to GPU

    # === Expert Quantization ===
    # NOTE: Can now be combined with LoRA and offloading for hybrid mode!
    use_expert_quantization: bool = False    # Enable expert quantization
    quantize_inactive_experts: bool = True   # Quantize only inactive experts
    expert_quantization_bits: int = 8        # Quantization bits (8 or 4)
    expert_quantization_method: str = 'per_channel'  # 'per_channel' or 'per_tensor'
    use_bitsandbytes: bool = False           # Use bitsandbytes library for quantization


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
    """Configuration for episodic memory."""
    use_episodic_memory: bool = False         # Enable episodic memory (disabled by default)
    memory_capacity: int = 1000               # Memory capacity
    memory_selection_strategy: str = 'importance'  # Selection strategy
    memory_importance_threshold: float = 0.5  # Importance threshold
    memory_retrieval_method: str = 'cosine'  # Retrieval method
    memory_replay_ratio: float = 0.2         # Replay ratio
    memory_replay_strategy: str = 'importance' # Replay strategy
    memory_adaptation_rate: float = 0.01     # Adaptation rate
    memory_performance_window: int = 100     # Performance window
    task_id: int = 0                         # Task ID
    silent_mode: bool = False                # Suppress memory warnings in console
    enable_auto_grad_accumulation: bool = False  # Enable auto gradient accumulation adjustment


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
    num_workers: int = 0                      # CRITICAL FIX: Default 0 to avoid multiprocessing deadlocks with Arrow files
    prefetch_factor: int = 4                  # Batches to prefetch per worker
    persistent_workers: bool = False          # Keep workers alive between epochs (YAML controls)
    padding_side: str = 'right'               # Tokenizer padding side
    truncation: bool = True                   # Enable truncation
    max_train_examples: Optional[int] = None  # Max training examples
    max_eval_examples: Optional[int] = None   # Max evaluation examples
    dataloader_drop_last: bool = False        # Drop last incomplete batch
    dataloader_pin_memory: bool = False       # Pin memory for faster GPU transfer
    default_tokenizer_name: str = 'Qwen/Qwen2.5-0.5B'  # Default tokenizer if none specified

    # SPEED OPTIMIZATION: Sequence packing for 20-35% speedup
    use_sequence_packing: bool = False        # Enable sequence packing (20-35% speedup)
    packing_strategy: str = 'greedy'          # Packing strategy: 'greedy' or 'adaptive'
    use_dynamic_batching: bool = False        # Enable dynamic batching
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

    # Dynamic batch sizing (5.3 fixes)
    enable_dynamic_batch: bool = False
    enable_binary_search_oom: bool = False    # YAML controls
    enable_dry_run_mode: bool = False         # YAML controls
    min_batch_size: int = 1
    max_batch_size: int = 64
    target_gpu_utilization: float = 0.85
    batch_size_adaptation_steps: int = 100


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
class DynamicBatchingConfig:
    """
    Unified configuration for dynamic batch sizing based on GPU memory.

    Consolidates all 8+ features for improved GPU utilization and stability.
    All advanced features default to disabled for backward compatibility.
    """
    # Core settings
    enabled: bool = False                      # Enable dynamic batching
    initial_batch_size: int = 128              # Starting batch size
    min_batch_size: int = 1                    # Minimum batch size
    max_batch_size: int = 64                   # Maximum batch size

    # Memory thresholds (conservative for stability)
    target_memory_utilization: float = 0.85    # Target GPU memory usage (legacy name)
    low_memory_threshold: float = 0.55         # Below this, increase batch size
    target_memory_threshold: float = 0.70      # Target utilization (conservative)
    high_memory_threshold: float = 0.80        # Above this, decrease batch size
    critical_memory_threshold: float = 0.88    # Emergency decrease (safe margin)

    # Adjustment parameters
    adjustment_frequency: int = 100            # Check every N steps
    adjustment_factor: float = 1.25            # Scale factor for adjustments (legacy)
    adjustment_strategy: str = 'adaptive'      # 'adaptive', 'geometric', 'linear'
    increase_factor: float = 1.1               # Multiply by this when increasing
    decrease_factor: float = 0.9               # Multiply by this when decreasing
    warmup_steps: int = 500                    # Don't adjust during first N steps
    smooth_transitions: bool = False           # Use gradual adjustments (YAML controls)

    # Safety parameters
    max_adjustments_per_session: int = 100     # Prevent oscillation
    cooldown_steps: int = 30                   # Steps to wait after adjustment
    hysteresis_margin: float = 0.05            # Don't adjust unless memory differs by >5%
    min_stable_steps: int = 30                 # Require N stable steps before change

    # Feature 1: Token Budget Batching
    token_budget_enabled: bool = False
    target_tokens_per_batch: int = 4096
    max_tokens_per_batch: int = 8192
    min_tokens_per_batch: int = 512

    # Feature 2: Sequence-Length Aware Batching
    sequence_aware: bool = False
    base_sequence_length: int = 512            # Reference length for scaling
    sequence_scaling_factor: float = 1.0       # How aggressively to scale (0.5-1.5)

    # Feature 3: Multi-GPU Synchronization
    sync_across_gpus: bool = True              # Enable for distributed training
    sync_strategy: str = 'min'                 # 'min', 'max', or 'mean'

    # Feature 4: Gradient Accumulation Integration
    coordinate_with_grad_accum: bool = False
    target_effective_batch_size: int = 512     # batch_size * grad_accum_steps
    dynamic_grad_accum: bool = False           # Adjust grad_accum instead of batch_size
    original_grad_accum_steps: int = 1         # Store original value

    # Feature 5: Predictive Memory Estimation
    predictive_enabled: bool = False
    calibration_steps: int = 50
    memory_model: str = 'linear'               # 'linear' or 'quadratic'
    backward_safety_margin: float = 1.20       # Multiplier for backward pass memory

    # Feature 7: Geometric Warmup Strategy
    warmup_strategy: str = 'none'              # 'none', 'linear', 'geometric'
    warmup_growth_rate: float = 1.15           # For geometric: multiply each adjustment
    warmup_initial_fraction: float = 0.25      # Start at 25% of min_batch_size

    # Feature 8: Memory Trend Detection
    trend_detection_enabled: bool = False
    trend_window: int = 20                     # Steps to analyze for trends
    oscillation_threshold: int = 5             # Direction changes before dampening
    auto_tune_smoothing: bool = True           # Auto-adjust EMA alpha

    # Feature 9: Adaptive backward margin (auto-tunes over time)
    adaptive_margin_enabled: bool = False      # Disabled by default for stability
    adaptive_margin_min: float = 1.05          # Minimum margin (backward >= 5% more)
    adaptive_margin_max: float = 2.0           # Maximum margin (backward <= 2x forward)
    adaptive_margin_learning_rate: float = 0.02  # How fast to adapt (lower = stable)
    adaptive_margin_ema_alpha: float = 0.1     # EMA smoothing for observations

    # GPU SYNC: Memory stats cache interval (reduces cudaStreamSynchronize calls)
    memory_stats_cache_interval_sec: float = 0.5  # Query GPU memory at most every N seconds

    # Feature 10: Aggressive Growth Mode (probes for optimal batch size)
    aggressive_growth_enabled: bool = False       # Enable aggressive batch size exploration
    aggressive_growth_exploration_steps: int = 500  # Steps to explore larger batches
    aggressive_growth_factor: int = 2             # Multiplier for batch increases during exploration
    aggressive_growth_min_headroom: float = 0.10  # Min memory headroom to trigger growth

    # Enhanced calibration system
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def __post_init__(self):
        """Validate configuration values to prevent division by zero."""
        if self.min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be > 0, got {self.min_batch_size}")
        if self.max_batch_size < self.min_batch_size:
            raise ValueError(
                f"max_batch_size ({self.max_batch_size}) must be >= "
                f"min_batch_size ({self.min_batch_size})"
            )


# Backward compatibility alias
DynamicBatchConfig = DynamicBatchingConfig


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

    # Dynamic batching
    dynamic_batching: Optional[DynamicBatchingConfig] = None


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
    enabled: bool = False                     # Enable coherence measurement
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
class PerformanceConfig:
    """Configuration for performance modes and hardware optimizations."""
    ultra_fast_mode: bool = False             # Ultra fast mode
    fast_progress: bool = False               # Fast progress mode
    minimal_progress: bool = False            # Minimal progress mode
    no_sync: bool = False                     # No CUDA sync mode
    express_mode: bool = False                # Express mode

    # TF32 and hardware optimizations (NEW)
    enable_tf32: bool = True                  # Enable TF32 on Ampere+ GPUs (8x faster matmul)
    float32_matmul_precision: str = 'high'    # Options: 'highest', 'high', 'medium'
    enable_cudnn_benchmark: bool = True       # Auto-tune cuDNN kernels
    cudagraph_skip_dynamic_shapes: bool = True   # Skip dynamic shapes in CUDAGraph
    cudagraph_dynamic_shape_warn_limit: Optional[int] = None  # Warning limit for dynamic shapes
    torchinductor_max_autotune: int = 0       # TorchInductor autotune level (0-4)


@dataclass
class OptimizationsConfig:
    """Configuration for all training optimizations (Phase 1, 2, 3)."""

    # Phase 1: Quick Wins
    torchinductor_autotune: int = 1            # 0=off, 1=basic, 2=aggressive (10-15% speedup)

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
    """Configuration for logging and observability."""
    # Verbosity settings
    verbosity: str = 'info'                   # Log level: 'debug', 'info', 'warning', 'error'
    console_level: str = 'info'               # Console log level (can be different from file)
    file_level: str = 'debug'                 # File log level (more detailed)

    # Monitoring frequencies (in steps)
    metrics_log_freq: int = 500               # GPU UTIL FIX: Reduced frequency to minimize .item() sync overhead (was 100)
    memory_check_freq: int = 2000             # GPU UTIL FIX: Reduced frequency to minimize mem_get_info() sync overhead (was 50)
    health_summary_freq: int = 500            # How often to log training health summary
    moe_metrics_freq: int = 5000              # GPU UTIL FIX: Reduced frequency to minimize expert metric sync overhead (was 2000)

    # Feature flags
    enable_timing_breakdown: bool = True      # Log step-level timing (data, forward, backward, optimizer)
    enable_memory_profiling: bool = True      # Enable detailed memory profiling
    enable_health_summaries: bool = True      # Enable periodic health summary logs
    log_tensor_shapes: bool = True            # Log tensor shapes on errors (OOM, NaN)
    log_checkpoint_validation: bool = True    # Validate checkpoints after save
    save_sample_generations: bool = True      # Save validation generation samples to file

    # Structured logging
    use_structured_logging: bool = True       # Use structured logs with contextual fields
    log_format: str = 'default'               # Log format: 'default', 'json', 'structured'

    # WandB/external logging
    log_gradients_to_wandb: bool = False      # Log gradient histograms to WandB (expensive)
    log_model_topology: bool = False          # Log model graph to WandB (one-time)


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
    These settings can provide 50-80% throughput improvement when properly tuned.
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
    moe_memory_optimization: MoEMemoryOptimizationConfig = field(default_factory=MoEMemoryOptimizationConfig)
    lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)
    memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    model: BasicModelConfig = field(default_factory=BasicModelConfig)
    adaptive_mtp: AdaptiveMTPConfig = field(default_factory=AdaptiveMTPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dev_log: DevLogConfig = field(default_factory=DevLogConfig)
    optimizations: OptimizationsConfig = field(default_factory=OptimizationsConfig)

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
            from src.Ava.config.yaml_loader import load_yaml_with_path_resolution
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