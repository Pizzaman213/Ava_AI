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
        """Allow accessing any attribute dynamically for type checkers."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert DynamicConfig back to a dictionary.

        Returns:
            Dictionary representation of configuration
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DynamicConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """String representation of DynamicConfig"""
        return f"DynamicConfig({self.to_dict()})"


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
class GradientHealthConfig:
    """Configuration for gradient health monitoring."""
    grad_norm_history: List[float] = field(default_factory=list)  # History of gradient norms
    grad_norm_pre_clip_history: List[float] = field(default_factory=list)  # Pre-clip norms
    total_steps: int = 0                      # Total steps tracked
    explosion_threshold: float = 10.0         # Threshold for gradient explosions
    recent_explosions: List[int] = field(default_factory=list)  # Recent explosion steps
    total_explosions: int = 0                 # Total explosions detected


@dataclass
class GradientConfig:
    """Configuration for gradient surgery."""
    gradient_surgery: bool = False            # Enable gradient surgery (disabled by default)
    adaptive_gradient_surgery: bool = False   # Adaptive method selection (disabled by default)
    gradient_surgery_method: str = 'pcgrad'   # Surgery method
    health_monitoring: GradientHealthConfig = field(default_factory=GradientHealthConfig)  # Gradient health monitoring


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
        "/project/code/data/processed",
        "/project/code/data/combined",
        "/project/code/data",
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
    data_dir: str = '/project/code/data/Testing'  # Data directory
    max_length: int = 512                     # Max sequence length
    tokenizer_name: Optional[str] = None      # Tokenizer name or path
    max_samples: Optional[int] = None         # Max samples (testing)
    streaming: bool = False                   # Streaming loader (YAML controls)
    buffer_size: int = 50000                  # Streaming buffer size (optimized for LLM pretraining)
    num_workers: int = 8                      # Parallel data loading workers
    prefetch_factor: int = 4                  # Batches to prefetch per worker
    persistent_workers: bool = False          # Keep workers alive between epochs (YAML controls)
    padding_side: str = 'right'               # Tokenizer padding side
    truncation: bool = True                   # Enable truncation
    max_train_examples: Optional[int] = None  # Max training examples
    max_eval_examples: Optional[int] = None   # Max evaluation examples
    dataloader_drop_last: bool = False        # Drop last incomplete batch
    dataloader_pin_memory: bool = False       # Pin memory for faster GPU transfer
    default_tokenizer_name: str = 'Qwen/Qwen2.5-0.5B'  # Default tokenizer if none specified


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
    cache_dir: str = "/tmp/difficulty_cache"
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
class DynamicBatchingConfig:
    """Configuration for dynamic batch sizing based on GPU memory."""
    enabled: bool = False                      # Enable dynamic batching
    min_batch_size: int = 1                    # Minimum batch size
    max_batch_size: int = 64                   # Maximum batch size
    target_memory_utilization: float = 0.85    # Target GPU memory usage (0.85 = 85%)
    adjustment_frequency: int = 100            # Check every N steps
    adjustment_factor: float = 1.25            # Scale factor for adjustments
    warmup_steps: int = 500                    # Don't adjust during first N steps
    smooth_transitions: bool = False           # Use gradual adjustments (YAML controls)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: Optional[int] = None          # Batch size
    epochs: Optional[int] = None              # Number of epochs
    learning_rate: Optional[float] = None     # Learning rate
    gradient_accumulation: int = 1            # Gradient accumulation
    max_gradient_norm: float = 1.0            # Maximum gradient norm for clipping

    # Adaptive LR configuration
    adaptive_lr: dict = field(default_factory=dict)  # Adaptive learning rate settings

    # Progressive training
    progressive: ProgressiveTrainingConfig = field(default_factory=ProgressiveTrainingConfig)

    # Dynamic batching
    dynamic_batching: Optional[DynamicBatchingConfig] = None


@dataclass
class OutputConfig:
    """Configuration for output handling."""
    output_dir: str = '/project/code/outputs'  # Output directory
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
    """Configuration for performance modes."""
    ultra_fast_mode: bool = False             # Ultra fast mode
    fast_progress: bool = False               # Fast progress mode
    minimal_progress: bool = False            # Minimal progress mode
    no_sync: bool = False                     # No CUDA sync mode
    express_mode: bool = False                # Express mode


@dataclass
class ModelConfig:
    """Configuration for model architecture parameters."""
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

    # Feature configurations
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)
    memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adaptive_mtp: AdaptiveMTPConfig = field(default_factory=AdaptiveMTPConfig)

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

        Args:
            config_path: Path to YAML configuration file

        Returns:
            DynamicConfig object with all YAML fields accessible via dot notation

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
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

        # AUTO-SYNC: Ensure gradient_accumulation_steps is consistent across all sections
        # This prevents the common bug where training.gradient_accumulation_steps differs
        # from deepspeed.gradient_accumulation_steps or lr_finder.gradient_accumulation_steps
        if 'training' in config_dict and 'gradient_accumulation_steps' in config_dict['training']:
            master_grad_accum = config_dict['training']['gradient_accumulation_steps']

            # Sync deepspeed section
            if 'deepspeed' in config_dict:
                if config_dict['deepspeed'].get('gradient_accumulation_steps') != master_grad_accum:
                    print(f"⚙️  Auto-syncing deepspeed.gradient_accumulation_steps: "
                          f"{config_dict['deepspeed'].get('gradient_accumulation_steps', 'not set')} → {master_grad_accum}")
                    config_dict['deepspeed']['gradient_accumulation_steps'] = master_grad_accum

            # Sync lr_finder section
            if 'lr_finder' in config_dict:
                if config_dict['lr_finder'].get('gradient_accumulation_steps') != master_grad_accum:
                    print(f"⚙️  Auto-syncing lr_finder.gradient_accumulation_steps: "
                          f"{config_dict['lr_finder'].get('gradient_accumulation_steps', 'not set')} → {master_grad_accum}")
                    config_dict['lr_finder']['gradient_accumulation_steps'] = master_grad_accum

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

        # Architecture enhancements
        arch_group = parser.add_argument_group('Architecture Features')
        arch_group.add_argument('--use-moh', action='store_true', default=False,
                               help='Enable Mixture of Heads (MoH)')
        arch_group.add_argument('--use-moa', action='store_true', default=False,
                               help='Enable Mixture of Activations (MoA)')
        arch_group.add_argument('--use-cross-attention', action='store_true', default=False,
                               help='Enable multi-modal cross-attention')
        arch_group.add_argument('--use-alibi', action='store_true', default=False,
                               help='Use ALiBi positional encoding instead of RoPE')
        arch_group.add_argument('--expert-routing-type', type=str, default='switch',
                               choices=['base', 'switch', 'gshard', 'hash', 'stochastic'],
                               help='Type of expert routing to use')

        # RAG system
        rag_group = parser.add_argument_group('RAG System')
        rag_group.add_argument('--use-rag', action='store_true', default=False,
                              help='Enable Retrieval-Augmented Generation')
        rag_group.add_argument('--knowledge-base-path', type=str,
                              help='Path to knowledge base for RAG')
        rag_group.add_argument('--max-retrieved-docs', type=int, default=5,
                              help='Maximum number of documents to retrieve')
        rag_group.add_argument('--rag-fusion-type', type=str, default='attention',
                              choices=['attention', 'gate', 'concat', 'weighted'],
                              help='RAG fusion strategy')

        # Advanced loss functions
        loss_group = parser.add_argument_group('Loss Functions')
        loss_group.add_argument('--use-focal-loss', action='store_true', default=False,
                               help='Enable focal loss for hard example mining')
        loss_group.add_argument('--use-contrastive-loss', action='store_true', default=False,
                               help='Enable contrastive learning loss')
        loss_group.add_argument('--use-diversity-loss', action='store_true', default=False,
                               help='Enable expert diversity loss')
        loss_group.add_argument('--adaptive-loss-scaling', action='store_true', default=False,
                               help='Enable adaptive loss scaling')

        # Gradient surgery
        grad_group = parser.add_argument_group('Gradient Surgery')
        grad_group.add_argument('--gradient-surgery', action='store_true', default=False,
                               help='Enable gradient surgery for multi-task learning')
        grad_group.add_argument('--adaptive-gradient-surgery', action='store_true', default=False,
                               help='Use adaptive gradient surgery method selection')
        grad_group.add_argument('--gradient-surgery-method', type=str, default='pcgrad',
                               choices=['pcgrad', 'graddrop', 'gradnorm', 'cagrad', 'mgda'],
                               help='Gradient surgery method')
        grad_group.add_argument('--multi-task', action='store_true',
                               help='Enable multi-task learning mode')

        # Evaluation during training
        eval_group = parser.add_argument_group('Evaluation')
        eval_group.add_argument('--eval-during-training', action='store_true', default=False,
                               help='Run comprehensive evaluation during training')
        eval_group.add_argument('--eval-metrics', type=str,
                               help='Comma-separated list of evaluation metrics')
        eval_group.add_argument('--eval-frequency', type=int, default=500,
                               help='Evaluation frequency (steps)')

        # Quantization
        quant_group = parser.add_argument_group('Quantization')
        quant_group.add_argument('--quantization-aware', action='store_true',
                                help='Enable quantization-aware training')
        quant_group.add_argument('--bit-width', type=int, default=8, choices=[4, 8],
                                help='Quantization bit width')
        quant_group.add_argument('--use-nvfp4', action='store_true',
                                help='Enable NVFP4 4-bit floating-point training')
        quant_group.add_argument('--nvfp4-block-size', type=int, default=16,
                                help='NVFP4 micro-block size (default: 16)')
        quant_group.add_argument('--stochastic-rounding', action='store_true',
                                help='Enable stochastic rounding for NVFP4 training')
        quant_group.add_argument('--use-hadamard-transform', action='store_true',
                                help='Apply Hadamard transforms to reshape tensor distributions')
        quant_group.add_argument('--use-torchao-nvfp4', action='store_true',
                                help='Use TorchAO native NVFP4 implementation if available')

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

        # Episodic memory for continual learning
        memory_group = parser.add_argument_group('Episodic Memory')
        memory_group.add_argument('--use-episodic-memory', action='store_true', default=False,
                                 help='Enable episodic memory for continual learning')
        memory_group.add_argument('--memory-capacity', type=int, default=1000,
                                 help='Episodic memory bank capacity')
        memory_group.add_argument('--memory-selection-strategy', type=str, default='importance',
                                 choices=['importance', 'random', 'task_balanced'],
                                 help='Memory selection strategy')
        memory_group.add_argument('--memory-importance-threshold', type=float, default=0.5,
                                 help='Threshold for memory importance scoring')
        memory_group.add_argument('--memory-retrieval-method', type=str, default='cosine',
                                 choices=['cosine', 'euclidean', 'dot'],
                                 help='Memory retrieval similarity method')
        memory_group.add_argument('--memory-replay-ratio', type=float, default=0.2,
                                 help='Ratio of replay samples to current batch')
        memory_group.add_argument('--memory-replay-strategy', type=str, default='importance',
                                 choices=['random', 'importance', 'similarity'],
                                 help='Experience replay sampling strategy')
        memory_group.add_argument('--memory-adaptation-rate', type=float, default=0.01,
                                 help='Adaptation rate for memory parameters')
        memory_group.add_argument('--memory-performance-window', type=int, default=100,
                                 help='Window size for performance-based adaptation')
        memory_group.add_argument('--task-id', type=int, default=0,
                                 help='Task ID for multi-task continual learning')

        # === DATA ARGUMENTS ===
        data_group = parser.add_argument_group('Data Configuration')
        data_group.add_argument('--data-dir', type=str,
                               default='/project/code/data/processed',
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
        output_group.add_argument('--output-dir', type=str, default='/project/code/outputs',
                                 help='Output directory for checkpoints')
        output_group.add_argument('--save-every', type=int, default=100,
                                 help='Save checkpoint every N steps')
        output_group.add_argument('--resume', type=str, default=None,
                                 help='Resume from checkpoint')
        output_group.add_argument('--fresh-start', action='store_true',
                                 help='Force fresh start, ignore any existing checkpoints')
        output_group.add_argument('--reset-step-counter', action='store_true',
                                 help='Reset global step counter to 0 (for debugging)')

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

        # Progressive training arguments
        prog_group = parser.add_argument_group('Progressive Training (Phase 5 Fixes)')
        prog_group.add_argument('--enable-progressive-training', action='store_true',
                               help='Enable progressive training with Phase 5 fixes')

        # Sequence length scaling (5.1)
        prog_group.add_argument('--enable-sequence-scaling', action='store_true',
                               help='Enable progressive sequence length scaling (5.1)')
        prog_group.add_argument('--initial-seq-length', type=int, default=128,
                               help='Initial sequence length for progressive scaling')
        prog_group.add_argument('--final-seq-length', type=int, default=2048,
                               help='Final sequence length for progressive scaling')
        prog_group.add_argument('--length-schedule', type=str, default='linear',
                               choices=['linear', 'exponential', 'step'],
                               help='Sequence length growth schedule')
        prog_group.add_argument('--length-growth-epochs', type=int, default=10,
                               help='Number of epochs to grow sequence length')
        prog_group.add_argument('--enable-length-bucketing', action='store_true', default=True,
                               help='Enable length-based bucketing for efficiency')

        # Difficulty scoring (5.2)
        prog_group.add_argument('--enable-curriculum', action='store_true',
                               help='Enable curriculum learning with streaming batches (5.2)')
        prog_group.add_argument('--curriculum-metric', type=str, default='loss',
                               choices=['loss', 'perplexity', 'attention_entropy'],
                               help='Metric for difficulty scoring')
        prog_group.add_argument('--enable-score-caching', action='store_true', default=True,
                               help='Enable difficulty score disk caching')
        prog_group.add_argument('--cache-dir', type=str, default='/tmp/difficulty_cache',
                               help='Directory for difficulty score cache')

        # Dynamic batch sizing (5.3)
        prog_group.add_argument('--enable-dynamic-batch', action='store_true',
                               help='Enable dynamic batch sizing with binary search OOM handling (5.3)')
        prog_group.add_argument('--enable-binary-search-oom', action='store_true', default=True,
                               help='Use binary search for OOM handling instead of simple halving')
        prog_group.add_argument('--enable-dry-run-mode', action='store_true', default=True,
                               help='Enable dry-run mode for safe batch size testing')
        prog_group.add_argument('--progressive-min-batch-size', type=int, default=1,
                               help='Minimum batch size for progressive training')
        prog_group.add_argument('--progressive-max-batch-size', type=int, default=64,
                               help='Maximum batch size for progressive training')
        prog_group.add_argument('--target-gpu-utilization', type=float, default=0.85,
                               help='Target GPU utilization for dynamic batch sizing')

        return parser

    def parse_args_to_config(self, args: argparse.Namespace) -> EnhancedTrainingConfig:
        """Convert parsed arguments to structured configuration."""

        # Handle enable-all-features flag
        if args.enable_all_features:
            self._enable_all_features(args)

        # Create structured config
        config = EnhancedTrainingConfig(
            config_file=args.config,
            enable_all_features=args.enable_all_features,
            multi_task=args.multi_task,

            architecture=ArchitectureConfig(
                use_moh=args.use_moh,
                use_moa=args.use_moa,
                use_cross_attention=args.use_cross_attention,
                use_alibi=args.use_alibi,
                expert_routing_type=args.expert_routing_type
            ),

            rag=RAGConfig(
                use_rag=args.use_rag,
                knowledge_base_path=args.knowledge_base_path,
                max_retrieved_docs=args.max_retrieved_docs,
                rag_fusion_type=args.rag_fusion_type
            ),

            losses=LossConfig(
                use_focal_loss=args.use_focal_loss,
                use_contrastive_loss=args.use_contrastive_loss,
                use_diversity_loss=args.use_diversity_loss,
                adaptive_loss_scaling=args.adaptive_loss_scaling
            ),

            gradient=GradientConfig(
                gradient_surgery=args.gradient_surgery,
                adaptive_gradient_surgery=args.adaptive_gradient_surgery,
                gradient_surgery_method=args.gradient_surgery_method
            ),

            evaluation=EvaluationConfig(
                eval_during_training=args.eval_during_training,
                eval_metrics=args.eval_metrics,
                eval_frequency=args.eval_frequency
            ),

            quantization=QuantizationConfig(
                quantization_aware=args.quantization_aware,
                bit_width=args.bit_width,
                use_nvfp4=args.use_nvfp4,
                nvfp4_block_size=args.nvfp4_block_size,
                stochastic_rounding=args.stochastic_rounding,
                use_hadamard_transform=args.use_hadamard_transform,
                use_torchao_nvfp4=args.use_torchao_nvfp4
            ),

            lr_finder=LRFinderConfig(
                run_lr_finder=args.run_lr_finder,
                start_lr=args.lr_finder_start,
                end_lr=args.lr_finder_end,
                num_iterations=args.lr_finder_iterations,
                suggestion_method=args.lr_finder_method,
                use_suggested_lr=args.lr_finder_use_suggested,
                plot_path=args.lr_finder_plot_path
            ),

            memory=EpisodicMemoryConfig(
                use_episodic_memory=args.use_episodic_memory,
                memory_capacity=args.memory_capacity,
                memory_selection_strategy=args.memory_selection_strategy,
                memory_importance_threshold=args.memory_importance_threshold,
                memory_retrieval_method=args.memory_retrieval_method,
                memory_replay_ratio=args.memory_replay_ratio,
                memory_replay_strategy=args.memory_replay_strategy,
                memory_adaptation_rate=args.memory_adaptation_rate,
                memory_performance_window=args.memory_performance_window,
                task_id=args.task_id
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
                gradient_accumulation=args.gradient_accumulation,
                progressive=ProgressiveTrainingConfig(
                    enable_progressive_training=args.enable_progressive_training,
                    enable_sequence_scaling=args.enable_sequence_scaling,
                    initial_seq_length=args.initial_seq_length,
                    final_seq_length=args.final_seq_length,
                    length_schedule=args.length_schedule,
                    length_growth_epochs=args.length_growth_epochs,
                    enable_length_bucketing=args.enable_length_bucketing,
                    enable_curriculum=args.enable_curriculum,
                    curriculum_metric=args.curriculum_metric,
                    enable_score_caching=args.enable_score_caching,
                    cache_dir=args.cache_dir,
                    enable_dynamic_batch=args.enable_dynamic_batch,
                    enable_binary_search_oom=args.enable_binary_search_oom,
                    enable_dry_run_mode=args.enable_dry_run_mode,
                    min_batch_size=args.progressive_min_batch_size,
                    max_batch_size=args.progressive_max_batch_size,
                    target_gpu_utilization=args.target_gpu_utilization
                )
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
            )
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
        if hasattr(yaml_config, 'data'):
            if args.data_dir != '/project/code/data/processed':  # Not default
                yaml_config.data.data_dir = args.data_dir
            if args.max_length != 512:  # Not default
                yaml_config.data.max_length = args.max_length
            if args.max_samples is not None:
                yaml_config.data.max_samples = args.max_samples
        else:
            # Create data section if it doesn't exist
            data_dict = {}
            if args.data_dir != '/project/code/data/processed':
                data_dict['data_dir'] = args.data_dir
            if args.max_length != 512:
                data_dict['max_length'] = args.max_length
            if args.max_samples is not None:
                data_dict['max_samples'] = args.max_samples
            if data_dict:
                yaml_config.data = DynamicConfig(data_dict)

        # Output overrides
        if hasattr(yaml_config, 'output'):
            if args.output_dir != '/project/code/outputs':  # Not default
                yaml_config.output.output_dir = args.output_dir
            if args.resume is not None:
                yaml_config.output.resume = args.resume
        else:
            output_dict = {}
            if args.output_dir != '/project/code/outputs':
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
        # Architecture features
        args.use_moh = True
        args.use_moa = True
        args.use_cross_attention = True
        args.use_alibi = True

        # RAG features
        args.use_rag = True

        # Loss features
        args.use_focal_loss = True
        args.use_contrastive_loss = True
        args.use_diversity_loss = True
        args.adaptive_loss_scaling = True

        # Gradient features
        args.gradient_surgery = True
        args.adaptive_gradient_surgery = True

        # Evaluation features
        args.eval_during_training = True

        # Memory features
        args.use_episodic_memory = True

        # DeepSpeed features (optional - only enable if distributed training is desired)
        # args.use_deepspeed = True  # Comment out by default as it requires multi-GPU setup

    def _build_feature_dependencies(self) -> Dict[str, List[str]]:
        """Build feature dependency mapping."""
        return {
            'gradient_surgery': ['multi_task'],
            'rag': ['knowledge_base_path'],
            'episodic_memory': ['task_id'],
            'quantization_aware': ['bit_width'],
            'nvfp4': ['nvfp4_block_size']
        }

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