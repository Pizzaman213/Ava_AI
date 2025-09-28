"""
Enhanced Trainer using Modular Components

This module provides the core Enhanced Trainer that integrates all the
modular training components for maximum flexibility and maintainability.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path

# Optional imports with fallbacks
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# Import all the new modular components
from ..utils.gpu_memory import GPUMemoryManager
from ..utils.async_logging import AsyncLogger, AsyncLoggingConfig
from ..config.training_config import EnhancedTrainingConfig
from .advanced_warmup import AdvancedWarmupScheduler, WarmupConfig
from .adaptive_lr import AdaptiveLearningRateManager, AdaptiveLRConfig
from .performance_modes import PerformanceModeManager, PerformanceModeConfig
from .metrics import TrainingMetricsCollector, MetricConfig

# Import existing components
from ..losses.advanced_losses import CompositeLoss, AdaptiveLossScaling
from .gradient_surgery import GradientSurgeon, AdaptiveGradientSurgeon
from .gradient_health import GradientHealthMonitor, LossHealthMonitor
from .memory_monitor import MemoryMonitor
from .lr_manager import IntelligentLRManager, LRConfig
from .distributed_manager import DistributedManager, DistributedConfig, get_distributed_manager
from .rank_aware_error_handler import RankAwareErrorHandler, get_error_handler, ErrorType, ErrorSeverity
from .distributed_health_checker import DistributedHealthChecker, get_health_checker, record_training_metrics
from ..retrieval.rag_system import RAGSystem, KnowledgeBase
from ..evaluation.comprehensive_eval import ComprehensiveEvaluator
from ..optimization.quantization import ModelQuantizer, QuantizationConfig
from ..memory.episodic_memory import EpisodicMemoryBank, AdaptiveMemoryManager, ExperienceReplay

# Import Phase 7 observability components
from ..observability.training_integration import (
    ObservabilityIntegration,
    ObservabilityConfig,
    create_observability_integration,
    create_lightweight_observability
)


class EnhancedModularTrainer:
    """
    Enhanced trainer built with modular components.

    This trainer integrates all the advanced features using the new modular
    architecture for better maintainability and reusability.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        config: EnhancedTrainingConfig,
        run_manager=None
    ):
        """
        Initialize enhanced modular trainer.

        Args:
            model: PyTorch model
            tokenizer: Model tokenizer
            device: Training device
            config: Enhanced training configuration
            run_manager: Optional run manager
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.run_manager = run_manager

        # DeepSpeed state
        self.deepspeed_engine = None
        self.deepspeed_config = None
        self.is_distributed = False

        # Distributed training state
        self.distributed_manager = None
        self.error_handler = None
        self.health_checker = None

        # Observability integration (Phase 7)
        self.observability = None

        # Initialize all modular components
        self._init_gpu_memory_manager()
        self._init_performance_manager()
        self._init_distributed_manager()
        self._init_error_handler()
        self._init_health_checker()
        self._init_deepspeed()
        self._init_async_logger()
        self._init_metrics_collector()
        self._init_loss_functions()
        self._init_gradient_surgery()
        self._init_rag_system()
        self._init_evaluator()
        self._init_quantization()
        self._init_episodic_memory()
        self._init_observability()

        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.best_loss = float('inf')

        # Mixed precision gradient scaler with health monitoring
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self.scaler_reset_interval = 1000  # Reset scaler every N steps to prevent error accumulation
        self.scaler_last_reset = 0

        # Initialize gradient and loss health monitors
        self.gradient_health = GradientHealthMonitor(
            initial_clip_value=5.0,
            final_clip_value=1.0,
            warmup_steps=1000,
            history_size=100,
            explosion_threshold=10.0
        )
        self.loss_health = LossHealthMonitor(
            history_size=100,
            spike_threshold_sigma=3.0,
            divergence_threshold=2.0
        )

        # Initialize memory monitor for proactive OOM prevention
        self.memory_monitor = MemoryMonitor(
            target_utilization=0.85,
            warning_threshold=0.90,
            critical_threshold=0.95,
            emergency_threshold=0.98,
            history_size=100,
            memory_headroom_gb=1.0
        )
        print("Gradient, loss, and memory monitors initialized")

    def _init_gpu_memory_manager(self):
        """Initialize GPU memory manager."""
        self.gpu_manager = GPUMemoryManager(auto_cleanup=True)
        print("GPU memory manager initialized")

    def _init_performance_manager(self):
        """Initialize performance mode manager."""
        # Convert PerformanceConfig to PerformanceModeConfig
        from .performance_modes import PerformanceModeConfig, PerformanceMode

        # Determine mode from boolean flags
        if self.config.performance.ultra_fast_mode:
            mode = PerformanceMode.ULTRA_FAST
        elif self.config.performance.fast_progress:
            mode = PerformanceMode.FAST_PROGRESS
        elif self.config.performance.minimal_progress:
            mode = PerformanceMode.MINIMAL_PROGRESS
        elif self.config.performance.express_mode:
            mode = PerformanceMode.EXPRESS_MODE
        elif self.config.performance.no_sync:
            mode = PerformanceMode.NO_SYNC
        else:
            mode = PerformanceMode.STANDARD

        # Create PerformanceModeConfig
        perf_mode_config = PerformanceModeConfig(
            mode=mode,
            disable_wandb=self.config.performance.ultra_fast_mode,
            disable_progress_bar=self.config.performance.ultra_fast_mode,
            disable_cuda_sync=self.config.performance.no_sync,
            minimal_logging=self.config.performance.ultra_fast_mode,
            async_logging=not self.config.performance.ultra_fast_mode
        )

        self.performance_manager = PerformanceModeManager(perf_mode_config)
        summary = self.performance_manager.get_performance_summary()
        print(f"Performance mode: {summary['mode']}")
        print(f"Optimizations: {', '.join(summary['active_optimizations'])}")

    def _init_distributed_manager(self):
        """Initialize distributed training manager."""
        from .distributed_manager import is_distributed, get_distributed_manager, DistributedConfig

        # Check if we're in a distributed environment
        if not is_distributed():
            print("Single-node training mode")
            return

        print("🚀 Initializing distributed training manager...")

        # Create distributed configuration from training config
        distributed_config = DistributedConfig(
            backend=getattr(self.config.deepspeed, 'backend', 'nccl'),
            timeout_seconds=getattr(self.config.training, 'distributed_timeout', 1800),
            enable_barriers=getattr(self.config.training, 'enable_barriers', True),
            enable_heartbeat=getattr(self.config.training, 'enable_heartbeat', True),
            enable_rank_failure_recovery=getattr(self.config.training, 'enable_rank_failure_recovery', True),
            max_failed_ranks=getattr(self.config.training, 'max_failed_ranks', 1)
        )

        # Get global distributed manager
        self.distributed_manager = get_distributed_manager(distributed_config)

        # Initialize distributed training
        success = self.distributed_manager.initialize()
        if success:
            self.is_distributed = True
            stats = self.distributed_manager.get_stats()
            print(f"✅ Distributed training initialized successfully:")
            print(f"   Rank: {stats['rank']}/{stats['world_size']}")
            print(f"   Backend: {stats['backend']}")
            print(f"   Health monitoring: {stats['health_monitoring']}")

            # Register cleanup handler
            self.distributed_manager.register_cleanup_handler(self._distributed_cleanup_handler)
        else:
            print("❌ Failed to initialize distributed training")
            self.distributed_manager = None

    def _distributed_cleanup_handler(self):
        """Cleanup handler for distributed training."""
        print("🧹 Distributed trainer cleanup handler called")
        # Add any trainer-specific distributed cleanup here

    def _init_error_handler(self):
        """Initialize rank-aware error handler."""
        from .distributed_manager import is_distributed

        if not is_distributed():
            print("Single-node training: error handler disabled")
            return

        print("🛡️ Initializing rank-aware error handler...")

        try:
            # Get error handler with configuration
            self.error_handler = get_error_handler(
                max_retries=getattr(self.config.training, 'max_error_retries', 3),
                heartbeat_interval=getattr(self.config.training, 'error_heartbeat_interval', 10.0),
                failure_timeout=getattr(self.config.training, 'rank_failure_timeout', 60.0),
                enable_recovery=getattr(self.config.training, 'enable_error_recovery', True)
            )

            # Register recovery callbacks
            self.error_handler.register_recovery_callback(self._handle_rank_failure)

            print(f"✅ Error handler initialized for rank {self.error_handler.rank}")
            print(f"   Max retries: {self.error_handler.max_retries}")
            print(f"   Heartbeat interval: {self.error_handler.heartbeat_interval}s")
            print(f"   Failure timeout: {self.error_handler.failure_timeout}s")

        except Exception as e:
            print(f"❌ Failed to initialize error handler: {e}")
            self.error_handler = None

    def _handle_rank_failure(self, failed_rank: int, error_info):
        """Handle rank failure callback."""
        if failed_rank == -1:
            # Global failure - initiate emergency shutdown
            print(f"🚨 Emergency shutdown initiated due to: {error_info.message}")
            # Could trigger model saving, cleanup, etc.
        else:
            print(f"🔧 Handling failure of rank {failed_rank}: {error_info.message}")
            # Could implement rank replacement, model redistribution, etc.

        # Log the failure event
        if hasattr(self, 'async_logger') and self.async_logger:
            self.async_logger.log_metric(
                f"rank_failure_{failed_rank}",
                1,
                {"error_type": error_info.error_type.value, "severity": error_info.severity.value}
            )

    def _init_health_checker(self):
        """Initialize distributed health checker."""
        from .distributed_manager import is_distributed

        if not is_distributed():
            print("Single-node training: health checker disabled")
            return

        print("🩺 Initializing distributed health checker...")

        try:
            # Get health checker with configuration
            self.health_checker = get_health_checker(
                check_interval=getattr(self.config.training, 'health_check_interval', 30.0),
                loss_history_size=getattr(self.config.training, 'health_history_size', 100),
                anomaly_threshold=getattr(self.config.training, 'health_anomaly_threshold', 2.5),
                enable_gradient_sync=getattr(self.config.training, 'enable_gradient_health_sync', True),
                enable_performance_sync=getattr(self.config.training, 'enable_performance_health_sync', True)
            )

            print(f"✅ Health checker initialized for rank {self.health_checker.rank}")
            print(f"   Check interval: {self.health_checker.check_interval}s")
            print(f"   Anomaly threshold: {self.health_checker.anomaly_threshold}")
            print(f"   Gradient sync: {self.health_checker.enable_gradient_sync}")

        except Exception as e:
            print(f"❌ Failed to initialize health checker: {e}")
            self.health_checker = None

    def _init_deepspeed(self):
        """Initialize DeepSpeed distributed training."""
        if not self.config.deepspeed.use_deepspeed or not DEEPSPEED_AVAILABLE:
            print("DeepSpeed disabled or not available")
            return

        print("Initializing DeepSpeed...")

        # Check if we're in a distributed environment
        import os
        self.is_distributed = (
            'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
        ) or (
            'LOCAL_RANK' in os.environ
        )

        if not self.is_distributed:
            print(" DeepSpeed enabled but not in distributed environment")
            print("   Set WORLD_SIZE and LOCAL_RANK environment variables for multi-GPU training")

        # Load or generate DeepSpeed configuration
        self.deepspeed_config = self._create_deepspeed_config()

        print("DeepSpeed configuration prepared")

    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration dictionary."""
        ds_config = self.config.deepspeed

        config = {
            "train_batch_size": ds_config.train_batch_size or 32,
            "train_micro_batch_size_per_gpu": ds_config.micro_batch_size or 4,
            "gradient_accumulation_steps": ds_config.gradient_accumulation_steps,

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "weight_decay": "auto",
                    "beta1": "auto",
                    "beta2": "auto",
                    "eps": "auto"
                }
            },

            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto"
                }
            },

            "zero_optimization": {
                "stage": ds_config.zero_stage,
                "allgather_partitions": ds_config.allgather_partitions,
                "allgather_bucket_size": ds_config.zero_allgather_bucket_size,
                "overlap_comm": ds_config.overlap_comm,
                "reduce_scatter": ds_config.zero_reduce_scatter,
                "reduce_bucket_size": ds_config.zero_reduce_bucket_size,
                "contiguous_gradients": ds_config.zero_contiguous_gradients
            },

            "gradient_clipping": ds_config.gradient_clipping or 1.0,

            "wall_clock_breakdown": ds_config.wall_clock_breakdown,

            "data_types": {
                "grad_accum_dtype": "fp32",
                "params_dtype": "fp32"
            }
        }

        # Add mixed precision configuration
        if ds_config.enable_mixed_precision:
            if ds_config.precision_type == "fp16":
                config["fp16"] = {
                    "enabled": True,
                    "auto_cast": False,
                    "loss_scale": 0,
                    "initial_scale_power": 16,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "consecutive_hysteresis": False,
                    "min_loss_scale": 1
                }
            elif ds_config.precision_type == "bf16":
                config["bf16"] = {
                    "enabled": True
                }

        # Add ZeRO stage-specific configurations
        if ds_config.zero_stage == 3:
            config["zero_optimization"].update({
                "stage3_prefetch_bucket_size": ds_config.zero_stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": ds_config.zero_stage3_param_persistence_threshold,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            })

        # Add CPU offloading
        if ds_config.cpu_offload:
            if ds_config.zero_stage == 2:
                config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            elif ds_config.zero_stage == 3:
                config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
                config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }

        # Add NVMe offloading
        if ds_config.nvme_offload and ds_config.cpu_offload:
            config["zero_optimization"]["offload_optimizer"]["nvme_path"] = "/local_nvme"
            if ds_config.zero_stage == 3:
                config["zero_optimization"]["offload_param"]["nvme_path"] = "/local_nvme"

        # Add activation checkpointing
        if ds_config.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": ds_config.partition_activations,
                "cpu_checkpointing": ds_config.cpu_checkpointing,
                "contiguous_memory_optimization": ds_config.contiguous_memory_optimization,
                "synchronize_checkpoint_boundary": ds_config.synchronize_dp_processes
            }

        return config

    def _init_async_logger(self):
        """Initialize async logging system."""
        if self.performance_manager.should_use_async_logging():
            logging_config = AsyncLoggingConfig(
                enable_system_metrics=True,
                wandb_cache_size=self.config.wandb.wandb_cache_size,
                wandb_flush_interval=self.config.wandb.wandb_cache_flush_interval
            )
            self.async_logger = AsyncLogger(
                logging_config,
                wandb_available=not self.config.wandb.disable_wandb,
                wandb_offline=self.config.wandb.wandb_offline,
                disable_wandb=self.config.wandb.disable_wandb
            )
            self.async_logger.start()
            print("Async logging initialized")
        else:
            self.async_logger = None

    def _init_metrics_collector(self):
        """Initialize training metrics collector."""
        if self.performance_manager.config.mode.value != "ultra_fast":
            metrics_config = MetricConfig(
                collect_gradients=True,
                collect_memory=True,
                collect_system=self.performance_manager.config.mode.value != "minimal_progress",
                gradient_freq=100,
                memory_freq=50
            )
            self.metrics_collector = TrainingMetricsCollector(metrics_config)
            print("Training metrics collector initialized")
        else:
            self.metrics_collector = None

    def _init_loss_functions(self):
        """Initialize advanced loss functions."""
        if any([self.config.losses.use_focal_loss, self.config.losses.use_contrastive_loss,
                self.config.losses.use_diversity_loss]):
            loss_config = {}
            if self.config.losses.use_focal_loss:
                loss_config['focal'] = {'type': 'focal', 'alpha': 1.0, 'gamma': 2.0, 'weight': 0.1}
            if self.config.losses.use_contrastive_loss:
                loss_config['contrastive'] = {'type': 'contrastive', 'temperature': 0.07, 'weight': 0.1}
            if self.config.losses.use_diversity_loss:
                loss_config['diversity'] = {'type': 'diversity', 'weight': 0.01}

            # Add auxiliary loss only if enabled (for MoE stability)
            if getattr(self.config.losses, 'use_auxiliary_loss', True):
                loss_config['auxiliary'] = {
                    'type': 'auxiliary',
                    'load_balancing_weight': 0.0001,
                    'router_z_weight': 0.00001,
                    'weight': 0.001
                }

            self.composite_loss = CompositeLoss(loss_config)
            print(f" Advanced loss functions: {list(loss_config.keys())}")
        else:
            self.composite_loss = None

        # Adaptive loss scaling
        if self.config.losses.adaptive_loss_scaling:
            num_losses = len(loss_config) if loss_config else 1
            self.adaptive_scaler = AdaptiveLossScaling(num_losses=num_losses)
        else:
            self.adaptive_scaler = None

    def _init_gradient_surgery(self):
        """Initialize gradient surgery."""
        if self.config.gradient.gradient_surgery:
            if self.config.gradient.adaptive_gradient_surgery:
                self.gradient_surgeon = AdaptiveGradientSurgeon(
                    methods=[self.config.gradient.gradient_surgery_method]
                )
                print(" Adaptive gradient surgery initialized")
            else:
                self.gradient_surgeon = GradientSurgeon(
                    method=self.config.gradient.gradient_surgery_method
                )
                print(" Gradient surgery initialized")
        else:
            self.gradient_surgeon = None

    def _init_rag_system(self):
        """Initialize RAG system."""
        if self.config.rag.use_rag:
            try:
                self.rag_system = RAGSystem(
                    encoder_dim=768,  # Default, should be from model config
                    retrieval_dim=256,
                    max_retrieved=self.config.rag.max_retrieved_docs,
                    fusion_type=self.config.rag.rag_fusion_type
                )

                if self.config.rag.knowledge_base_path:
                    kb_path = Path(self.config.rag.knowledge_base_path)
                    if kb_path.exists():
                        self.knowledge_base = KnowledgeBase(embedding_dim=256)
                        self.knowledge_base.load(kb_path)
                        self.rag_system.set_knowledge_base(self.knowledge_base)
                        print(f" RAG system with KB: {kb_path.name}")
                    else:
                        print(f" Knowledge base not found: {kb_path}")
                        self.rag_system = None
            except Exception as e:
                print(f" RAG initialization failed: {e}")
                self.rag_system = None
        else:
            self.rag_system = None

    def _init_evaluator(self):
        """Initialize comprehensive evaluator."""
        if self.config.evaluation.eval_during_training:
            eval_config = {
                'tokenizer_name': 'gpt2',  # Default
                'bleu_max_n': 4,
                'rouge_types': ['rouge-1', 'rouge-2', 'rouge-l']
            }
            self.evaluator = ComprehensiveEvaluator(eval_config)

            if self.config.evaluation.eval_metrics:
                self.eval_metrics = self.config.evaluation.eval_metrics.split(',')
            else:
                self.eval_metrics = ['perplexity']

            print(f" Evaluator: {', '.join(self.eval_metrics)}")
        else:
            self.evaluator = None

    def _init_quantization(self):
        """Initialize quantization."""
        if self.config.quantization.quantization_aware or self.config.quantization.use_nvfp4:
            if self.config.quantization.use_nvfp4:
                quant_config = QuantizationConfig(
                    bit_width=4,
                    use_nvfp4=True,
                    nvfp4_block_size=self.config.quantization.nvfp4_block_size,
                    stochastic_rounding=self.config.quantization.stochastic_rounding,
                    use_hadamard_transform=self.config.quantization.use_hadamard_transform,
                    symmetric=True,
                    per_channel=True
                )
                print(" NVFP4 4-bit quantization enabled")
            else:
                quant_config = QuantizationConfig(
                    bit_width=self.config.quantization.bit_width,
                    symmetric=True,
                    per_channel=True
                )
                print(f" {self.config.quantization.bit_width}-bit quantization enabled")

            self.quantizer = ModelQuantizer(quant_config)
        else:
            self.quantizer = None

    def _init_episodic_memory(self):
        """Initialize episodic memory."""
        if self.config.memory.use_episodic_memory:
            try:
                self.memory_bank = EpisodicMemoryBank(
                    capacity=self.config.memory.memory_capacity,
                    hidden_size=768,  # Default, should be from model config
                    selection_strategy=self.config.memory.memory_selection_strategy,
                    importance_threshold=self.config.memory.memory_importance_threshold
                )

                self.memory_manager = AdaptiveMemoryManager(
                    memory_bank=self.memory_bank,
                    adaptation_rate=self.config.memory.memory_adaptation_rate,
                    performance_window=self.config.memory.memory_performance_window
                )

                self.experience_replay = ExperienceReplay(
                    memory_bank=self.memory_bank,
                    replay_ratio=self.config.memory.memory_replay_ratio,
                    replay_strategy=self.config.memory.memory_replay_strategy
                )

                print(f" Episodic memory: {self.config.memory.memory_capacity} capacity")
            except Exception as e:
                print(f" Memory initialization failed: {e}")
                self.memory_bank = None
                self.memory_manager = None
                self.experience_replay = None
        else:
            self.memory_bank = None
            self.memory_manager = None
            self.experience_replay = None

    def _init_observability(self):
        """Initialize Phase 7 observability integration."""
        try:
            # Determine observability level based on performance mode
            if self.performance_manager.config.mode.value == "ultra_fast":
                # Lightweight observability for maximum speed
                self.observability = create_lightweight_observability()
                print("🔍 Lightweight observability initialized (ultra_fast mode)")
            else:
                # Full observability for normal training
                obs_config = ObservabilityConfig(
                    # Enable based on performance settings
                    enable_hierarchical_logging=not self.performance_manager.config.minimal_logging,
                    enable_health_dashboard=not self.performance_manager.config.disable_progress_bar,
                    enable_optimized_metrics=True,
                    enable_training_validation=True,
                    enable_post_mortem=True,
                    enable_monitoring_api=not self.performance_manager.config.disable_wandb,

                    # Configure based on performance mode
                    log_level="DEBUG" if self.performance_manager.config.mode.value == "standard" else "INFO",
                    dashboard_update_interval=5.0 if self.performance_manager.config.mode.value == "standard" else 10.0,
                    metric_sampling_rate=0.2 if self.performance_manager.config.mode.value == "standard" else 0.1,
                    enable_async_metrics=self.performance_manager.config.async_logging,

                    # Integration settings
                    export_interval=100,
                    checkpoint_observability=True
                )

                self.observability = ObservabilityIntegration(obs_config)
                print("🔍 Full observability integration initialized")

            # Prepare training context for initialization
            training_context = {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'device': self.device,
                'config': self.config,
                'distributed': self.is_distributed,
                'deepspeed': self.deepspeed_engine is not None
            }

            # Initialize observability with context
            self.observability.initialize(training_context)
            print("✓ Phase 7 observability integration ready")

        except Exception as e:
            print(f"❌ Failed to initialize observability: {e}")
            print("   Training will continue without observability")
            self.observability = None

    def setup_training(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Set up training with warmup and adaptive LR, optionally with DeepSpeed.

        Args:
            optimizer: PyTorch optimizer

        Returns:
            Dictionary with training setup information
        """
        setup_info = {}

        # Initialize DeepSpeed engine if enabled
        if self.config.deepspeed.use_deepspeed and DEEPSPEED_AVAILABLE:
            setup_info.update(self._setup_deepspeed_training(optimizer))
        else:
            setup_info.update(self._setup_standard_training(optimizer))

        setup_info.update({
            'performance_mode': self.performance_manager.config.mode.value,
            'async_logging': self.async_logger is not None,
            'metrics_collection': self.metrics_collector is not None,
            'deepspeed_enabled': self.deepspeed_engine is not None,
            'observability_enabled': self.observability is not None
        })

        # Start observability observation if enabled
        if self.observability:
            # Enhanced training context with optimizer information
            enhanced_context = {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'device': self.device,
                'config': self.config,
                'optimizer': optimizer,
                'distributed': self.is_distributed,
                'deepspeed': self.deepspeed_engine is not None,
                'performance_mode': self.performance_manager.config.mode.value
            }
            self.observability.start_training_observation(enhanced_context)

        print(" Training setup completed")
        return setup_info

    def _setup_deepspeed_training(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Set up DeepSpeed training with robust initialization."""
        try:
            # Validate distributed training environment
            import torch.distributed as dist

            if not dist.is_available():
                raise RuntimeError("PyTorch distributed not available for DeepSpeed")

            # More robust distributed initialization check
            initialization_success = False

            # Check if distributed is already initialized
            if dist.is_initialized():
                print(f"Distributed already initialized: rank {dist.get_rank()}/{dist.get_world_size()}")
                initialization_success = True
            else:
                # Try to initialize with comprehensive environment variable checking
                required_vars = ['RANK', 'WORLD_SIZE']
                optional_vars = ['MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK']

                missing_vars = [var for var in required_vars if var not in os.environ]
                if missing_vars:
                    raise RuntimeError(
                        f"DeepSpeed requires distributed training environment variables. "
                        f"Missing: {missing_vars}. "
                        f"Please set RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT."
                    )

                # Log environment variables for debugging
                env_info = {var: os.environ.get(var, 'Not set') for var in required_vars + optional_vars}
                print(f"DeepSpeed environment: {env_info}")

                # Try initialization with error handling
                try:
                    print("Initializing torch.distributed for DeepSpeed...")
                    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                    dist.init_process_group(backend=backend)
                    print(f"Distributed initialized successfully: rank {dist.get_rank()}/{dist.get_world_size()}")
                    initialization_success = True
                except Exception as init_error:
                    raise RuntimeError(f"Failed to initialize distributed training: {init_error}")

            if not initialization_success:
                raise RuntimeError("Could not initialize distributed training for DeepSpeed")

            # Validate world size
            world_size = dist.get_world_size()
            if world_size < 1:
                raise RuntimeError(f"Invalid world size: {world_size}")

            # Load and merge DeepSpeed configs more carefully
            final_config = self.deepspeed_config.copy()  # Start with programmatic config

            if self.config.deepspeed.config_file:
                import json
                config_file_path = self.config.deepspeed.config_file
                print(f"Loading DeepSpeed config from: {config_file_path}")

                try:
                    with open(config_file_path, 'r') as f:
                        file_config = json.load(f)

                    # Smart merge: keep our feature flags, but allow file to override training settings
                    protected_keys = ['train_batch_size', 'train_micro_batch_size_per_gpu', 'gradient_accumulation_steps']
                    for key, value in file_config.items():
                        if key in protected_keys:
                            print(f"  File config overriding {key}: {final_config.get(key)} -> {value}")
                        final_config[key] = value

                except Exception as config_error:
                    print(f"Warning: Failed to load DeepSpeed config file: {config_error}")
                    print("  Using programmatic config only")

            # Validate config before initialization
            required_config_keys = ['train_batch_size', 'zero_optimization']
            missing_config = [key for key in required_config_keys if key not in final_config]
            if missing_config:
                raise RuntimeError(f"DeepSpeed config missing required keys: {missing_config}")

            print(f"Final DeepSpeed config keys: {list(final_config.keys())}")

            # Initialize DeepSpeed engine with better error handling
            try:
                self.deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=self.model,
                    optimizer=optimizer,
                    config=final_config,
                    lr_scheduler=None  # We'll handle LR scheduling manually
                )
            except Exception as ds_init_error:
                print(f"DeepSpeed initialize() failed: {ds_init_error}")
                print("  This might be due to:")
                print("  - Incompatible config settings")
                print("  - Insufficient GPU memory")
                print("  - Model architecture incompatibility")
                raise

            # Store references safely
            self.model = self.deepspeed_engine.module  # Get the wrapped model
            self.optimizer = optimizer  # DeepSpeed-managed optimizer
            self.lr_scheduler = lr_scheduler  # May be None

            # Validate the engine was created properly
            if not hasattr(self.deepspeed_engine, 'backward'):
                raise RuntimeError("DeepSpeed engine missing required methods")

            print(f" DeepSpeed initialized successfully:")
            print(f"   ZeRO stage: {self.config.deepspeed.zero_stage}")
            print(f"   World size: {world_size}")
            print(f"   Rank: {dist.get_rank()}")
            print(f"   LR scheduler: {'Yes' if lr_scheduler else 'No'}")

            return {
                'deepspeed_engine': True,
                'zero_stage': self.config.deepspeed.zero_stage,
                'world_size': world_size,
                'rank': dist.get_rank(),
                'warmup_enabled': lr_scheduler is not None,  # If DeepSpeed provides scheduler
                'adaptive_lr_enabled': lr_scheduler is not None,
            }

        except Exception as e:
            print(f" DeepSpeed initialization failed: {e}")
            print("   Falling back to standard training")

            # Cleanup any partial initialization
            if hasattr(self, 'deepspeed_engine'):
                self.deepspeed_engine = None

            return self._setup_standard_training(optimizer)

    def _setup_standard_training(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Set up standard (non-DeepSpeed) training with intelligent LR management."""
        # Get training parameters
        num_epochs = self.config.training.epochs if self.config.training.epochs else 3
        batch_size = getattr(self.config.training, 'batch_size', 8) or 8
        gradient_accumulation_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1) or 1

        # Create intelligent LR configuration
        lr_config = LRConfig(
            warmup_ratio=getattr(self.config.training, 'warmup_ratio', 0.03),  # 3% default
            warmup_min_ratio=0.01,
            warmup_schedule="linear",
            main_schedule="cosine",
            min_lr_ratio=0.01,
            enable_adaptive=getattr(self.config.training, 'enable_adaptive_lr', True),
            plateau_patience=getattr(self.config.training, 'plateau_patience', 10),
            plateau_threshold=0.01,
            plateau_factor=0.5,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_lr_recovery=True
        )

        # Create intelligent LR manager (will calculate total steps intelligently)
        self.lr_manager = IntelligentLRManager(
            optimizer=optimizer,
            config=lr_config,
            total_steps=None,  # Will be calculated
            steps_per_epoch=None  # Will be calculated
        )

        # Store for later dataset size calculation
        self.training_params = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps
        }

        # Legacy schedulers set to None (replaced by lr_manager)
        self.warmup_scheduler = None
        self.lr_scheduler = None
        self.adaptive_lr_manager = self.lr_manager  # For compatibility

        print(f"✓ Intelligent LR management enabled:")
        print(f"   Warmup ratio: {lr_config.warmup_ratio:.1%}")
        print(f"   Adaptive LR: {'Enabled' if lr_config.enable_adaptive else 'Disabled'}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")

        return {
            'deepspeed_engine': False,
            'warmup_enabled': True,
            'adaptive_lr_enabled': lr_config.enable_adaptive,
            'intelligent_lr_enabled': True,
            'gradient_accumulation_aware': True,
        }

    def setup_dataset_aware_lr(self, train_loader, dataset_size: Optional[int] = None):
        """
        Configure LR manager with actual dataset information.

        Args:
            train_loader: Training dataloader
            dataset_size: Optional dataset size override
        """
        if not hasattr(self, 'lr_manager') or self.lr_manager is None:
            return  # No LR manager to configure

        # Estimate dataset size if not provided
        if dataset_size is None:
            try:
                if hasattr(train_loader.dataset, '__len__'):
                    dataset_size = len(train_loader.dataset)
                else:
                    # For streaming datasets, estimate based on first few batches
                    print("⏳ Estimating dataset size for streaming dataset...")
                    sample_batches = 10
                    total_samples = 0

                    temp_iter = iter(train_loader)
                    for i in range(min(sample_batches, 50)):  # Don't sample too many
                        try:
                            batch = next(temp_iter)
                            if isinstance(batch, dict) and 'input_ids' in batch:
                                total_samples += batch['input_ids'].size(0)
                        except StopIteration:
                            break

                    if total_samples > 0:
                        avg_batch_size = total_samples / min(sample_batches, i + 1)
                        # Estimate total dataset size (rough approximation)
                        dataset_size = int(avg_batch_size * 1000)  # Assume 1000 batches minimum
                        print(f"📊 Estimated dataset size: ~{dataset_size:,} samples")
                    else:
                        dataset_size = 100000  # Fallback
                        print(f"⚠️  Could not estimate dataset size, using fallback: {dataset_size:,}")

            except Exception as e:
                dataset_size = 100000  # Fallback
                print(f"⚠️  Error estimating dataset size: {e}, using fallback: {dataset_size:,}")

        # Calculate total steps with actual dataset information
        total_steps = self.lr_manager.calculate_total_steps(
            num_epochs=self.training_params['num_epochs'],
            dataset_size=dataset_size,
            batch_size=self.training_params['batch_size'],
            gradient_accumulation_steps=self.training_params['gradient_accumulation_steps']
        )

        print(f"✓ LR schedule configured with actual dataset size:")
        print(f"   Dataset size: {dataset_size:,} samples")
        print(f"   Total steps: {total_steps:,}")
        print(f"   Warmup steps: {self.lr_manager.warmup_steps:,}")

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        Perform a single training step with all enhancements.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            optimizer: Optimizer
            epoch: Current epoch
            batch_idx: Current batch index

        Returns:
            Dictionary with step results
        """
        # Wrap entire training step with error handling
        if self.error_handler:
            try:
                return self._train_step_impl(input_ids, attention_mask, labels, optimizer, epoch, batch_idx)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM errors with collective coordination if distributed
                    oom_info = {
                        "rank": getattr(self.distributed_manager, 'rank', 0) if self.distributed_manager else 0,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "batch_size": input_ids.size(0),
                        "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                        "memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                        "error_message": str(e)
                    }

                    # Coordinate OOM handling across all ranks if distributed
                    if self.distributed_manager and self.distributed_manager.is_initialized():
                        print(f"🔥 OOM detected on rank {oom_info['rank']} - coordinating with other ranks...")

                        # Broadcast OOM signal to all ranks
                        broadcast_success = self.distributed_manager.broadcast_oom_signal(oom_info)

                        if broadcast_success:
                            # Coordinate recovery action across ranks
                            recovery_success = self.distributed_manager.coordinate_oom_recovery("reduce_batch_size")

                            if recovery_success:
                                print(f"✅ Collective OOM recovery coordinated across all ranks")
                            else:
                                print(f"❌ Failed to coordinate OOM recovery - falling back to local handling")

                    # Handle OOM with error handler and observability
                    handled = self.error_handler.handle_error(
                        e, ErrorType.MEMORY, ErrorSeverity.CRITICAL,
                        context=oom_info,
                        recoverable=True
                    )

                    # Notify observability of OOM error
                    if self.observability:
                        self.observability.handle_out_of_memory(self.step_count, oom_info)

                    if not handled:
                        raise
                    # Return dummy results to continue training
                    return {
                        'loss': float('inf'),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'skipped': True,
                        'skip_reason': 'memory_error'
                    }
                else:
                    # Handle other runtime errors
                    error_context = {"epoch": epoch, "batch_idx": batch_idx}
                    handled = self.error_handler.handle_error(
                        e, ErrorType.COMPUTE, ErrorSeverity.ERROR,
                        context=error_context,
                        recoverable=True
                    )

                    # Notify observability of runtime error
                    if self.observability:
                        self.observability.handle_training_error(e, self.step_count, error_context)

                    if not handled:
                        raise
                    # Return dummy results
                    return {
                        'loss': float('inf'),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'skipped': True,
                        'skip_reason': 'compute_error'
                    }
            except Exception as e:
                # Handle unexpected errors
                error_context = {"epoch": epoch, "batch_idx": batch_idx}
                handled = self.error_handler.handle_error(
                    e, ErrorType.UNKNOWN, ErrorSeverity.ERROR,
                    context=error_context,
                    recoverable=False
                )

                # Notify observability of unexpected error
                if self.observability:
                    self.observability.handle_training_error(e, self.step_count, error_context)

                if not handled:
                    raise
                return {
                    'loss': float('inf'),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'skipped': True,
                    'skip_reason': 'unknown_error'
                }
        else:
            # No error handler, proceed normally
            return self._train_step_impl(input_ids, attention_mask, labels, optimizer, epoch, batch_idx)

    def _train_step_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        Internal implementation of training step.
        """
        # Start metrics collection
        if self.metrics_collector:
            self.metrics_collector.start_step(self.step_count, epoch, batch_idx)

        # Check memory health at start of step - estimate batch size from input
        current_batch_size = input_ids.size(0) if torch.is_tensor(input_ids) else 8
        memory_health = self.memory_monitor.check_memory_health(current_batch_size)

        # Check collective memory health across all ranks if distributed
        collective_memory_health = None
        if self.distributed_manager and self.distributed_manager.is_initialized():
            # Periodic collective memory health checks (every 50 steps to avoid overhead)
            if batch_idx % 50 == 0:
                collective_memory_health = self.distributed_manager.check_collective_memory_health()

                if collective_memory_health["status"] in ["warning", "critical"]:
                    print(f"⚠️  Collective memory health: {collective_memory_health['status']}")
                    print(f"   Max utilization: {collective_memory_health['max_utilization']:.1%}")
                    print(f"   Min available: {collective_memory_health['min_available_gb']:.1f}GB")
                    print(f"   Problematic ranks: {len(collective_memory_health['problematic_ranks'])}/{collective_memory_health['total_ranks']}")

                    # If collective memory is critical, coordinate preventive action
                    if collective_memory_health["status"] == "critical":
                        print(f"🚨 Critical collective memory situation - coordinating preventive action...")
                        self.distributed_manager.coordinate_oom_recovery("reduce_batch_size")

            # Periodic rank failure detection (every 100 steps to avoid overhead)
            if batch_idx % 100 == 0:
                failure_status = self.distributed_manager.detect_rank_failures()

                if failure_status["status"] == "success":
                    if not failure_status["all_ranks_healthy"]:
                        failed_ranks = failure_status["failed_ranks"]
                        print(f"💥 Detected failed ranks: {failed_ranks}")
                        print(f"   Healthy ranks: {len(failure_status['healthy_ranks'])}/{failure_status['total_ranks']}")

                        # Coordinate recovery for failed ranks
                        print(f"🔄 Coordinating recovery for failed ranks...")
                        recovery_success = self.distributed_manager.coordinate_rank_replacement(
                            failed_ranks, ""
                        )

                        if recovery_success:
                            print(f"✅ Successfully coordinated recovery for failed ranks")

                            # Coordinate data resharding if dataloader supports it
                            if hasattr(self, 'train_dataloader') and self.train_dataloader:
                                from ..multi_column_data import coordinate_data_resharding
                                data_reshard_success = coordinate_data_resharding(
                                    self.train_dataloader, failed_ranks
                                )
                                if data_reshard_success:
                                    print(f"📊 Data resharding completed for failed ranks")
                                else:
                                    print(f"⚠️  Data resharding failed - using existing distribution")
                        else:
                            print(f"❌ Failed to coordinate recovery - training may be unstable")
                elif failure_status["status"] == "gather_failed":
                    if failure_status["gather_time"] > 10.0:  # Very slow response
                        print(f"⚠️  Very slow rank communication: {failure_status['gather_time']:.1f}s")

            # Periodic data distribution monitoring (every 200 steps to avoid overhead)
            if batch_idx % 200 == 0 and hasattr(self, 'train_dataloader') and self.train_dataloader:
                from ..multi_column_data import get_data_distribution_stats
                data_stats = get_data_distribution_stats(self.train_dataloader)

                if data_stats:
                    load_ratio = data_stats.get('load_ratio', 1.0)
                    samples_assigned = data_stats.get('samples_assigned', 0)

                    if abs(load_ratio - 1.0) > 0.1:  # More than 10% imbalance
                        print(f"⚖️  Data load imbalance detected:")
                        print(f"   Load ratio: {load_ratio:.2f} (1.0 = perfectly balanced)")
                        print(f"   Samples assigned: {samples_assigned}")

                        if 'resharded' in data_stats:
                            print(f"   Resharded for failed ranks: {data_stats.get('failed_ranks', [])}")

        # Handle critical memory situations BEFORE forward pass
        if memory_health['status'] == 'emergency':
            print(f"    EMERGENCY: Memory at {memory_health['gpu_utilization']:.1%}, "
                  f"cleaning up aggressively")
            cleanup_stats = self.memory_monitor.cleanup_memory(aggressive=True)
            print(f"    Freed {cleanup_stats['freed_gb']:.2f}GB")

            # Check if we should emergency stop
            if self.memory_monitor.should_emergency_stop():
                raise RuntimeError(
                    f"Emergency stop: Consistently high memory usage "
                    f"({memory_health['gpu_utilization']:.1%}). "
                    f"Reduce batch size or model size."
                )

        elif memory_health['status'] in ['critical', 'warning']:
            print(f"    Memory {memory_health['status'].upper()}: "
                  f"{memory_health['gpu_utilization']:.1%} used, "
                  f"recommend batch_size={memory_health['recommended_batch_size']}")

            # Perform standard cleanup
            if memory_health['status'] == 'critical':
                cleanup_stats = self.memory_monitor.cleanup_memory(aggressive=False)
                if cleanup_stats['freed_gb'] > 0.1:
                    print(f"    Freed {cleanup_stats['freed_gb']:.2f}GB GPU memory")

        # Validate device placement before forward pass
        model_device = next(self.model.parameters()).device
        if input_ids.device != model_device:
            raise RuntimeError(
                f"Device mismatch: input on {input_ids.device}, model on {model_device}. "
                f"This usually indicates batch was moved to wrong device."
            )

        # Forward pass with timing
        start_time = time.time()

        with torch.amp.autocast('cuda', enabled=True):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        forward_time = time.time() - start_time

        # Calculate losses
        main_loss = outputs.get('loss')

        # Validate main loss before continuing
        if main_loss is None:
            raise RuntimeError("Model did not return a loss value")
        if not main_loss.requires_grad:
            raise RuntimeError("Loss does not require gradients - check model configuration")

        # Check loss health BEFORE adding auxiliary losses
        loss_health_result = self.loss_health.check_loss_health(
            main_loss.item(),
            self.step_count
        )

        if not loss_health_result['is_valid']:
            print(f"     CRITICAL: {loss_health_result['reason']}")
            # TODO: Implement checkpoint restore here
            raise RuntimeError(f"Invalid loss detected: {loss_health_result['reason']}")

        # Handle loss spikes - skip auxiliary losses if main loss is high
        skip_auxiliary_losses = main_loss.item() > 5.0 or loss_health_result['is_spike']
        if skip_auxiliary_losses and batch_idx % 100 == 0:
            print(f"     Skipping auxiliary losses due to high main loss: {main_loss.item():.4f}")

        total_loss = main_loss

        # Apply composite loss if available and not skipping
        aux_losses = {}
        if self.composite_loss and not skip_auxiliary_losses:
            aux_losses = self.composite_loss(
                inputs=input_ids,
                targets=labels,
                logits=outputs.logits if hasattr(outputs, 'logits') else None,
                model=self.model,
                router_logits=getattr(outputs, 'router_logits', None),
                expert_outputs=getattr(outputs, 'expert_outputs', None)
            )

            # Track auxiliary loss moving averages for auto-scaling
            if not hasattr(self, 'aux_loss_emas'):
                self.aux_loss_emas = {}

            # Validate each auxiliary loss BEFORE adding
            valid_aux_losses = {}
            for name, loss_value in aux_losses.items():
                if name == 'total':
                    continue

                # Check if loss is a valid tensor with gradients
                if not isinstance(loss_value, torch.Tensor):
                    if batch_idx % 100 == 0:
                        print(f"     Skipping non-tensor {name} loss")
                    continue

                if not loss_value.requires_grad:
                    if batch_idx % 100 == 0:
                        print(f"     Warning: {name} loss doesn't require grad")
                    continue

                if torch.isnan(loss_value) or torch.isinf(loss_value):
                    if batch_idx % 100 == 0:
                        print(f"     Skipping invalid {name} loss")
                    continue

                # Update EMA for this loss component
                current_loss_val = loss_value.item()
                if name not in self.aux_loss_emas:
                    self.aux_loss_emas[name] = current_loss_val
                else:
                    self.aux_loss_emas[name] = 0.95 * self.aux_loss_emas[name] + 0.05 * current_loss_val

                # Scale auxiliary loss relative to main loss (much more conservative)
                aux_ema = self.aux_loss_emas[name]
                main_loss_val = main_loss.item()

                # Clamp to at most 0.1x main loss (not 10x!)
                max_aux_loss = main_loss_val * 0.1
                loss_value = torch.clamp(loss_value, max=max_aux_loss)

                # Additional scaling by EMA ratio to normalize contribution
                if aux_ema > 0 and main_loss_val > 0:
                    scaling_factor = min(1.0, main_loss_val / (aux_ema * 10))  # Conservative scaling
                    loss_value = loss_value * scaling_factor

                valid_aux_losses[name] = loss_value
                total_loss = total_loss + loss_value

            # Log individual loss components for monitoring
            if batch_idx % 1000 == 0 and valid_aux_losses:
                print(f"    Main loss: {main_loss.item():.6f}")
                for name, loss_value in valid_aux_losses.items():
                    ema_val = self.aux_loss_emas.get(name, 0.0)
                    print(f"    {name} loss: {loss_value.item():.6f} (EMA: {ema_val:.6f})")
                print(f"    Total loss: {total_loss.item():.6f}")

            # Check for loss validity after adding auxiliary losses
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"     Invalid total loss detected! Main: {main_loss.item():.6f}")
                for name, loss_value in valid_aux_losses.items():
                    print(f"      {name}: {loss_value.item():.6f}")
                print(f"      Total: {total_loss.item():.6f}")

                # Reset auxiliary loss EMAs if they might be corrupted
                self.aux_loss_emas.clear()
                raise RuntimeError("NaN or Inf loss detected after adding auxiliary losses")

        # Backward pass with timing
        backward_start = time.time()

        # Handle DeepSpeed vs standard training
        if self.deepspeed_engine:
            # Check gradient health before DeepSpeed backward (we can still monitor)
            try:
                # Get gradient norm before DeepSpeed processes it
                with torch.no_grad():
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    pre_deepspeed_grad_norm = total_norm ** 0.5 if total_norm > 0 else 0.0
            except Exception:
                pre_deepspeed_grad_norm = 0.0

            # DeepSpeed training
            self.deepspeed_engine.backward(total_loss)
            self.deepspeed_engine.step()

            # Extract gradient norm from DeepSpeed with comprehensive fallbacks
            grad_norm = None
            grad_norm_pre_clip = pre_deepspeed_grad_norm

            try:
                # Method 1: Try DeepSpeed's built-in gradient norm tracking
                if hasattr(self.deepspeed_engine, 'get_global_grad_norm'):
                    grad_norm = self.deepspeed_engine.get_global_grad_norm()
                    if grad_norm is not None and torch.is_tensor(grad_norm):
                        grad_norm = grad_norm.item()

                # Method 2: Try optimizer-specific gradient norm
                elif hasattr(self.deepspeed_engine, 'optimizer') and hasattr(self.deepspeed_engine.optimizer, 'get_global_grad_norm'):
                    grad_norm = self.deepspeed_engine.optimizer.get_global_grad_norm()
                    if grad_norm is not None and torch.is_tensor(grad_norm):
                        grad_norm = grad_norm.item()

                # Method 3: Check if optimizer has gradient clipping info
                elif hasattr(self.deepspeed_engine, 'optimizer'):
                    optimizer = self.deepspeed_engine.optimizer

                    # FP16 optimizer might have gradient scale info
                    if hasattr(optimizer, 'cur_scale'):
                        # We can't get exact grad norm but can detect scaling issues
                        if hasattr(optimizer, 'last_overflow_time') and optimizer.last_overflow_time == optimizer.step_count:
                            grad_norm = float('inf')  # Overflow detected
                        else:
                            grad_norm = pre_deepspeed_grad_norm  # Use our pre-calculation
                    else:
                        grad_norm = pre_deepspeed_grad_norm

                # Method 4: DeepSpeed internal tracking
                elif hasattr(self.deepspeed_engine, '_global_grad_norm'):
                    grad_norm = self.deepspeed_engine._global_grad_norm
                    if torch.is_tensor(grad_norm):
                        grad_norm = grad_norm.item()

                # Fallback: use our pre-calculation
                if grad_norm is None:
                    grad_norm = pre_deepspeed_grad_norm

                # Ensure grad_norm is a float
                if torch.is_tensor(grad_norm):
                    grad_norm = grad_norm.item()
                elif grad_norm is None:
                    grad_norm = 0.0

                # Update our gradient health monitor with DeepSpeed results
                # We'll track the pre-DeepSpeed norm since DeepSpeed handles clipping internally
                if hasattr(self, 'gradient_health'):
                    # Manually update the history since DeepSpeed handled clipping
                    self.gradient_health.grad_norm_history.append(grad_norm)
                    self.gradient_health.grad_norm_pre_clip_history.append(grad_norm_pre_clip)
                    self.gradient_health.total_steps += 1

                    # Check for explosions in pre-DeepSpeed norm
                    if grad_norm_pre_clip > self.gradient_health.explosion_threshold:
                        self.gradient_health.recent_explosions.append(self.step_count)
                        self.gradient_health.total_explosions += 1
                        print(f"    DeepSpeed gradient explosion detected: {grad_norm_pre_clip:.2f}")

                # Log gradient information periodically
                if self.step_count % 1000 == 0 or grad_norm_pre_clip > 5.0:
                    print(f"    DeepSpeed gradients: pre={grad_norm_pre_clip:.3f}, post={grad_norm:.3f}")

            except Exception as e:
                # Comprehensive fallback
                print(f"    Warning: DeepSpeed gradient norm extraction failed: {e}")
                grad_norm = pre_deepspeed_grad_norm
                grad_norm_pre_clip = pre_deepspeed_grad_norm
        else:
            # Standard training with mixed precision support
            if self.gradient_surgeon and self.config.multi_task:
                # Apply gradient surgery with scaler support
                task_losses = {'main': total_loss}  # Could have multiple tasks
                self._apply_gradient_surgery(task_losses, optimizer)
            else:
                # Standard backward pass with mixed precision
                optimizer.zero_grad()
                if self.scaler is not None:
                    # Scale loss and backward for mixed precision
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

            # Unscale gradients before computing norm for accurate monitoring
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)

            # Check gradient health BEFORE clipping
            grad_health_result = self.gradient_health.check_gradient_health(
                self.model,
                self.step_count,
                compute_histogram=(self.step_count % 1000 == 0)  # Detailed analysis every 1000 steps
            )

            # Handle critical gradient explosion - skip step
            if grad_health_result['should_skip']:
                print(f"    CRITICAL: Gradient explosion detected: {grad_health_result['grad_norm_pre_clip']:.2f}")
                print(f"    Skipping optimizer step to prevent NaN corruption")
                optimizer.zero_grad()  # Clear corrupted gradients
                if self.scaler is not None:
                    self.scaler.update()  # Update scaler even when skipping
                return {
                    'loss': total_loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'gradient_norm': grad_health_result['grad_norm_pre_clip'],
                    'gradient_norm_pre_clip': grad_health_result['grad_norm_pre_clip'],
                    'skipped': True,
                    'skip_reason': 'gradient_explosion',
                    **grad_health_result
                }

            # Check if we should reduce learning rate due to repeated explosions
            if grad_health_result['should_reduce_lr']:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = current_lr * 0.5  # Reduce by half
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"    Reducing learning rate due to gradient instability: {current_lr:.2e} -> {new_lr:.2e}")
                # Reset explosion counter after LR reduction
                self.gradient_health.reset_explosion_counter()

            # Apply adaptive gradient clipping
            grad_norm = self.gradient_health.clip_gradients(
                self.model,
                self.step_count
            )

            grad_norm_pre_clip = grad_health_result['grad_norm_pre_clip']

            # Log gradient health information
            if grad_health_result['is_explosion'] or self.step_count % 1000 == 0:
                print(f"    Gradient health: norm={grad_norm_pre_clip:.3f}, "
                      f"clip_value={grad_health_result['clip_value']:.1f}, "
                      f"explosions={grad_health_result['recent_explosions']}")

            # Emergency stop check
            if self.gradient_health.should_emergency_stop():
                raise RuntimeError(
                    f"Emergency stop: Too many gradient explosions "
                    f"({grad_health_result['recent_explosions']} recent). "
                    f"Training is unstable."
                )

            # Optimizer step with scaler support and health monitoring
            if self.scaler is not None:
                # Monitor scaler health before step
                scaler_scale = self.scaler.get_scale()
                scaler_state = {
                    'scale': scaler_scale,
                    'growth_factor': self.scaler.get_growth_factor(),
                    'backoff_factor': self.scaler.get_backoff_factor(),
                    'growth_interval': self.scaler.get_growth_interval()
                }

                # Check for scaler issues
                if scaler_scale < 1.0 or scaler_scale > 2**16:
                    print(f"    Warning: Scaler scale unusual: {scaler_scale}")

                self.scaler.step(optimizer)
                self.scaler.update()

                # Periodic scaler reset to prevent error accumulation
                if (self.step_count - self.scaler_last_reset) >= self.scaler_reset_interval:
                    print(f"    Resetting mixed precision scaler (step {self.step_count})")
                    # Save current scale for continuity
                    current_scale = self.scaler.get_scale()
                    self.scaler = torch.cuda.amp.GradScaler(
                        init_scale=min(current_scale, 2**15),  # Cap at reasonable value
                        growth_factor=2.0,
                        backoff_factor=0.5,
                        growth_interval=2000
                    )
                    self.scaler_last_reset = self.step_count

                # Log scaler state periodically
                if self.step_count % 1000 == 0:
                    print(f"    Scaler state: scale={scaler_scale:.1f}, "
                          f"growth_factor={scaler_state['growth_factor']}")
            else:
                optimizer.step()

        backward_time = time.time() - backward_start
        opt_time = backward_time  # Combined for DeepSpeed

        # Intelligent learning rate management
        if self.deepspeed_engine:
            current_lr = self.deepspeed_engine.get_lr()[0] if self.deepspeed_engine.get_lr() else 0.0
            warmup_info = {"warmup_step": self.step_count, "deepspeed_managed": True}
            lr_info = {"lr_step": self.step_count, "deepspeed_managed": True}
        else:
            # Use intelligent LR manager for gradient accumulation aware scheduling
            if hasattr(self, 'lr_manager') and self.lr_manager is not None:
                # Get gradient accumulation steps from LR manager configuration
                gradient_accumulation_steps = self.lr_manager.config.gradient_accumulation_steps

                # Check if this is an actual optimizer step (not just gradient accumulation)
                # Only step LR scheduler after accumulating the required number of gradients
                is_optimizer_step = (self.step_count % gradient_accumulation_steps) == 0

                if is_optimizer_step:
                    # Calculate the actual optimizer step number (for LR schedule calculation)
                    optimizer_step = self.step_count // gradient_accumulation_steps

                    # Pass validation loss if available for adaptive LR
                    validation_loss = getattr(self, '_last_validation_loss', None)
                    lr_step_info = self.lr_manager.step(validation_loss)

                    warmup_info = {
                        "warmup_step": self.step_count,
                        "optimizer_step": optimizer_step,
                        "phase": lr_step_info.get('phase', 'unknown'),
                        "plateau_patience": lr_step_info.get('plateau_patience', 0),
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "gradient_accumulation_aware": True
                    }

                    lr_info = {
                        "lr_step": self.step_count,
                        "optimizer_step": optimizer_step,
                        "intelligent_lr": True,
                        "lr_reduced": lr_step_info.get('lr_reduced', False),
                        "plateau_detected": lr_step_info.get('plateau_detected', False),
                        "gradient_accumulation_steps": gradient_accumulation_steps
                    }

                    # Log significant LR events
                    if lr_step_info.get('lr_reduced', False):
                        print(f"    🔽 LR reduced due to plateau at optimizer step {optimizer_step} (micro-step {self.step_count})")
                        print(f"        New LR: {lr_step_info['lr']:.2e}")

                    # Log LR schedule progress periodically
                    if optimizer_step % 100 == 0:
                        phase = lr_step_info.get('phase', 'unknown')
                        print(f"    📈 LR Schedule: optimizer_step={optimizer_step}, phase={phase}, "
                              f"lr={lr_step_info['lr']:.2e}, gradient_accum={gradient_accumulation_steps}")
                else:
                    # Not an optimizer step, just accumulating gradients
                    accumulation_step = (self.step_count % gradient_accumulation_steps) + 1
                    warmup_info = {
                        "warmup_step": self.step_count,
                        "accumulation_step": accumulation_step,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "accumulating": True
                    }
                    lr_info = {
                        "lr_step": self.step_count,
                        "accumulation_step": accumulation_step,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "accumulating": True
                    }
            else:
                # Fallback for cases without LR manager
                warmup_info = {"warmup_step": self.step_count, "fallback": True}
                lr_info = {"lr_step": self.step_count, "fallback": True}

            current_lr = optimizer.param_groups[0]['lr']

        # Collect gradient metrics
        grad_metrics = {}
        if self.metrics_collector:
            grad_metrics = self.metrics_collector.collect_gradient_metrics(self.model)

        # End metrics collection
        step_metrics = {}
        if self.metrics_collector:
            step_info = self.metrics_collector.end_step(
                loss=total_loss.item(),
                learning_rate=current_lr,
                grad_norm=grad_norm.item() if grad_norm is not None else None,
                forward_time=forward_time,
                backward_time=backward_time,
                optimizer_time=opt_time,
                **grad_metrics
            )
            step_metrics = {
                'batch_time': step_info.batch_time,
                'memory_allocated': step_info.memory_allocated,
                'memory_cached': step_info.memory_cached
            }

        # Log metrics asynchronously
        if self.async_logger and self.performance_manager.should_log_step(self.step_count):
            # Calculate iterations per second
            it_per_sec = 1.0 / step_metrics.get('batch_time', 1.0) if step_metrics.get('batch_time', 0) > 0 else 0.0

            metrics = {
                'train/loss': total_loss.item(),
                'train/main_loss': main_loss.item(),
                'train/learning_rate': current_lr,
                'train/grad_norm': grad_norm.item() if grad_norm is not None else 0.0,
                'train/grad_norm_pre_clip': grad_norm_pre_clip if grad_norm_pre_clip is not None else 0.0,
                'train/it_per_sec': it_per_sec,
                # Add batch time and memory metrics with train/ prefix for consistent grouping
                'train/batch_time': step_metrics.get('batch_time', 0.0),
                'train/memory_allocated_gb': step_metrics.get('memory_allocated', 0.0),
                'train/memory_cached_gb': step_metrics.get('memory_cached', 0.0),
            }

            # Add gradient metrics with train/ prefix
            for key, value in grad_metrics.items():
                metrics[f'train/{key}'] = value

            # Add individual auxiliary loss components
            for name, loss_value in aux_losses.items():
                if name != 'total':
                    metrics[f'train/aux_{name}'] = loss_value.item()

            self.async_logger.log_metrics(metrics, self.step_count)

        # Update observability with training step information (Phase 7)
        if self.observability:
            try:
                # Get memory and GPU metrics
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
                gpu_utilization = memory_health.get('gpu_utilization', 0.0)

                # Calculate throughput metrics
                total_time = forward_time + backward_time + opt_time
                samples_per_second = current_batch_size / total_time if total_time > 0 else 0.0

                # Update observability with comprehensive metrics
                self.observability.update_training_step(
                    step=self.step_count,
                    epoch=epoch,
                    loss=total_loss.item(),
                    learning_rate=current_lr,
                    batch_size=current_batch_size,
                    sequence_length=input_ids.size(1) if torch.is_tensor(input_ids) else 0,
                    # Performance metrics
                    forward_time=forward_time,
                    backward_time=backward_time,
                    optimizer_time=opt_time,
                    total_time=total_time,
                    samples_per_second=samples_per_second,
                    # Memory metrics
                    gpu_memory_used=gpu_memory_used,
                    gpu_memory_reserved=gpu_memory_reserved,
                    gpu_utilization=gpu_utilization,
                    memory_status=memory_health.get('status', 'unknown'),
                    memory_oom_risk=memory_health.get('oom_risk', 0.0),
                    # Gradient metrics
                    gradient_norm=grad_norm.item() if grad_norm is not None and hasattr(grad_norm, 'item') else (grad_norm if grad_norm is not None else 0.0),
                    gradient_norm_pre_clip=grad_norm_pre_clip.item() if grad_norm_pre_clip is not None and hasattr(grad_norm_pre_clip, 'item') else (grad_norm_pre_clip if grad_norm_pre_clip is not None else 0.0),
                    # Auxiliary loss information
                    main_loss=main_loss.item(),
                    aux_losses={name: loss.item() for name, loss in aux_losses.items() if isinstance(loss, torch.Tensor)},
                    # Training mode information
                    deepspeed_enabled=self.deepspeed_engine is not None,
                    distributed_enabled=self.is_distributed,
                    mixed_precision=self.scaler is not None
                )
            except Exception as e:
                # Don't let observability errors break training
                print(f"    Warning: Observability update failed: {e}")

        # Update training state
        self.step_count += 1
        if total_loss.item() < self.best_loss:
            self.best_loss = total_loss.item()

        # Intelligent memory management - replace basic cleanup
        memory_cleanup_needed = False

        # Check if we need cleanup based on step interval or memory status
        # Reduced frequency from every 10 steps to every 500 steps for performance
        if self.step_count % 500 == 0:  # Regular cleanup interval (low frequency)
            memory_cleanup_needed = True
        elif memory_health.get('status') in ['critical', 'emergency']:  # Only critical cases
            memory_cleanup_needed = True
        elif memory_health.get('oom_risk', 0.0) > 0.7:  # Very high OOM risk
            memory_cleanup_needed = True

        if memory_cleanup_needed and torch.cuda.is_available():
            cleanup_aggressive = memory_health.get('status') == 'emergency'
            cleanup_stats = self.memory_monitor.cleanup_memory(aggressive=cleanup_aggressive)

            # Log significant cleanup
            if cleanup_stats['freed_gb'] > 0.1:
                print(f"    Periodic cleanup freed {cleanup_stats['freed_gb']:.2f}GB")

        # Update memory monitor with current stats
        self.memory_monitor.update_memory_history(current_batch_size)

        # Record training metrics for distributed health monitoring
        if self.health_checker:
            self.health_checker.record_training_metrics(
                step=self.step_count,
                loss=total_loss.item(),
                learning_rate=current_lr,
                gradient_norm=grad_norm.item() if grad_norm is not None else 0.0
            )

        # Return step results with comprehensive monitoring info
        return {
            'loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'learning_rate': current_lr,
            'grad_norm': grad_norm.item() if grad_norm is not None and hasattr(grad_norm, 'item') else (grad_norm if grad_norm is not None else 0.0),
            'grad_norm_pre_clip': grad_norm_pre_clip.item() if grad_norm_pre_clip is not None and hasattr(grad_norm_pre_clip, 'item') else (grad_norm_pre_clip if grad_norm_pre_clip is not None else 0.0),
            'warmup_info': warmup_info,
            'lr_info': lr_info,
            'step_metrics': step_metrics,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'optimizer_time': opt_time,
            # Memory health information
            'memory_status': memory_health['status'],
            'memory_utilization': memory_health['gpu_utilization'],
            'memory_available_gb': memory_health['available_gb'],
            'memory_oom_risk': memory_health['oom_risk'],
            'recommended_batch_size': memory_health['recommended_batch_size'],
            # Loss health information
            'loss_health': loss_health_result
        }

    def set_validation_loss(self, validation_loss: float):
        """
        Set the validation loss for adaptive LR plateau detection.

        Args:
            validation_loss: Current validation loss
        """
        self._last_validation_loss = validation_loss

        # Log validation loss setting for debugging
        if hasattr(self, 'lr_manager') and self.lr_manager is not None:
            if self.lr_manager.config.enable_adaptive:
                lr_stats = self.lr_manager.get_statistics()
                print(f"    📊 Validation loss set: {validation_loss:.4f} "
                      f"(adaptive LR enabled, patience: {lr_stats.get('plateau_patience', 'N/A')})")

    def _apply_gradient_surgery(self, task_losses: Dict[str, torch.Tensor], optimizer):
        """Apply gradient surgery for multi-task learning."""
        if self.gradient_surgeon:
            try:
                # Apply gradient surgery using the configured method
                optimizer.zero_grad()

                # Compute gradients for each task
                task_gradients = {}
                task_list = list(task_losses.items())
                for idx, (task_name, loss) in enumerate(task_list):
                    # Only retain graph for non-final tasks to avoid memory leak
                    is_final_task = (idx == len(task_list) - 1)
                    loss.backward(retain_graph=not is_final_task)

                    task_gradients[task_name] = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            task_gradients[task_name].append(param.grad.clone())
                        else:
                            task_gradients[task_name].append(torch.zeros_like(param))

                    # Clear gradients after cloning (except for final task)
                    if not is_final_task:
                        optimizer.zero_grad()

                # Apply gradient surgery - returns a LIST of gradients
                modified_gradients = self.gradient_surgeon.apply_surgery(task_gradients)

                # Clear any remaining gradients before setting modified ones
                optimizer.zero_grad()

                # Update model parameters with modified gradients
                # modified_gradients is a list, not a dict
                for param, grad in zip(self.model.parameters(), modified_gradients):
                    param.grad = grad.clone() if grad is not None else None

            except Exception as e:
                print(f" Gradient surgery failed: {e}, falling back to standard training")
                # Fallback to standard training
                optimizer.zero_grad()
                task_losses['main'].backward()
        else:
            # Standard backward pass
            optimizer.zero_grad()
            task_losses['main'].backward()

    def save_checkpoint(self, checkpoint_dir: str, tag: str = None) -> str:
        """
        Save checkpoint with DeepSpeed support and distributed synchronization.

        Args:
            checkpoint_dir: Directory to save checkpoint
            tag: Optional tag for the checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Coordinate synchronized checkpointing if distributed
        if self.distributed_manager and self.distributed_manager.is_initialized():
            print(f"🔄 Coordinating synchronized checkpoint across {self.distributed_manager.world_size} ranks...")

            # Use fault-tolerant checkpointing
            sync_success = self.distributed_manager.checkpoint_with_fault_tolerance(
                checkpoint_dir,
                self.step_count
            )

            if not sync_success:
                print(f"❌ Failed to coordinate distributed checkpoint - proceeding with local checkpoint")
            else:
                print(f"✅ Distributed checkpoint coordination successful")
        if self.deepspeed_engine:
            # DeepSpeed checkpoint saving
            checkpoint_path = self.deepspeed_engine.save_checkpoint(checkpoint_dir, tag)
            print(f" DeepSpeed checkpoint saved: {checkpoint_path}")
            return checkpoint_path
        else:
            # Standard PyTorch checkpoint saving
            import torch
            from pathlib import Path

            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_file = checkpoint_dir / f"checkpoint{'_' + tag if tag else ''}.pt"

            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'step_count': self.step_count,
                'epoch_count': self.epoch_count,
                'best_loss': self.best_loss,
                'scaler_last_reset': getattr(self, 'scaler_last_reset', 0),
            }

            if hasattr(self, 'optimizer'):
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

            # Save scaler state for mixed precision training
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

            # Save monitor states for continuity
            if hasattr(self, 'gradient_health'):
                checkpoint['gradient_health_state'] = {
                    'total_steps': self.gradient_health.total_steps,
                    'total_explosions': self.gradient_health.total_explosions,
                    'stats': self.gradient_health.stats
                }

            if hasattr(self, 'loss_health'):
                checkpoint['loss_health_state'] = {
                    'best_loss': self.loss_health.best_loss,
                    'steps_since_improvement': self.loss_health.steps_since_improvement,
                    'spike_count': self.loss_health.spike_count,
                    'divergence_count': self.loss_health.divergence_count
                }

            # Save observability state (Phase 7)
            if self.observability:
                try:
                    checkpoint['observability_state'] = self.observability.create_checkpoint_data()
                except Exception as e:
                    print(f"    Warning: Failed to save observability state: {e}")
                    checkpoint['observability_state'] = {'error': str(e)}

            torch.save(checkpoint, checkpoint_file)
            print(f" Standard checkpoint saved: {checkpoint_file}")
            return str(checkpoint_file)

    def load_checkpoint(self, checkpoint_path: str, tag: str = None) -> Dict[str, Any]:
        """
        Load checkpoint with DeepSpeed support.

        Args:
            checkpoint_path: Path to checkpoint
            tag: Optional tag for the checkpoint

        Returns:
            Checkpoint metadata
        """
        if self.deepspeed_engine:
            # DeepSpeed checkpoint loading
            _, client_state = self.deepspeed_engine.load_checkpoint(checkpoint_path, tag)
            print(f"DeepSpeed checkpoint loaded: {checkpoint_path}")
            return client_state or {}
        else:
            # Standard PyTorch checkpoint loading with device-agnostic handling
            import torch
            from pathlib import Path

            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

            print(f"🔄 Loading checkpoint: {checkpoint_file}")
            print(f"   Target device: {self.device}")

            # Device-agnostic loading - always load to CPU first, then move to target device
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                print("   ✓ Checkpoint loaded to CPU")
            except Exception as e:
                # Fallback to target device if CPU loading fails
                print(f"   ⚠️  CPU loading failed: {e}")
                print(f"   Trying direct loading to {self.device}")
                checkpoint = torch.load(checkpoint_file, map_location=self.device)

            # Load model state with device handling
            if 'model_state_dict' in checkpoint:
                try:
                    # Move model state to target device if needed
                    model_state = checkpoint['model_state_dict']
                    if self.device != torch.device('cpu'):
                        # Move tensors to target device
                        model_state = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                     for k, v in model_state.items()}
                    self.model.load_state_dict(model_state)
                    print("   ✓ Model state restored")
                except Exception as e:
                    print(f"   ❌ Failed to restore model state: {e}")
                    raise

            # Load optimizer state if available
            if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer_state = checkpoint['optimizer_state_dict']
                    if self.device != torch.device('cpu'):
                        # Handle optimizer state device placement
                        for state in optimizer_state['state'].values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(self.device)
                    self.optimizer.load_state_dict(optimizer_state)
                    print("   ✓ Optimizer state restored")
                except Exception as e:
                    print(f"   ⚠️  Failed to restore optimizer state: {e}")
                    print("   Continuing with fresh optimizer state")

            # Load basic training state
            self.step_count = checkpoint.get('step_count', 0)
            self.epoch_count = checkpoint.get('epoch_count', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.scaler_last_reset = checkpoint.get('scaler_last_reset', 0)
            print(f"   ✓ Training state: step={self.step_count}, epoch={self.epoch_count}, best_loss={self.best_loss:.4f}")

            # Restore LR manager state
            restored_states = {'model': True, 'optimizer': hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint}

            if hasattr(self, 'lr_manager') and self.lr_manager is not None and 'lr_manager_state' in checkpoint:
                try:
                    lr_state = checkpoint['lr_manager_state']
                    # LR manager state is mostly statistics, no device handling needed
                    print(f"   ✓ LR manager state found: {lr_state.get('current_phase', 'unknown')} phase")
                    restored_states['lr_manager'] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore LR manager state: {e}")
                    restored_states['lr_manager'] = False
            else:
                restored_states['lr_manager'] = False

            # Restore mixed precision scaler state
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                try:
                    scaler_state = checkpoint['scaler_state_dict']
                    # Move scaler state to appropriate device if needed
                    if self.device != torch.device('cpu') and '_scale' in scaler_state:
                        if isinstance(scaler_state['_scale'], torch.Tensor):
                            scaler_state['_scale'] = scaler_state['_scale'].to(self.device)
                    self.scaler.load_state_dict(scaler_state)
                    print("   ✓ Mixed precision scaler state restored")
                    restored_states['scaler'] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore scaler state: {e}")
                    print("   Creating new scaler")
                    restored_states['scaler'] = False
            else:
                restored_states['scaler'] = False

            # Restore gradient health monitor state
            if hasattr(self, 'gradient_health') and 'gradient_health_state' in checkpoint:
                try:
                    gh_state = checkpoint['gradient_health_state']
                    self.gradient_health.explosion_threshold = gh_state.get('explosion_threshold', 5.0)
                    self.gradient_health.current_clip_value = gh_state.get('clip_value', 1.0)
                    self.gradient_health.total_explosions = gh_state.get('total_explosions', 0)
                    self.gradient_health.total_steps = gh_state.get('total_steps', 0)

                    # Restore history (limited to prevent memory issues)
                    if 'recent_explosions' in gh_state:
                        self.gradient_health.recent_explosions.extend(gh_state['recent_explosions'][-50:])
                    if 'grad_norm_history' in gh_state:
                        self.gradient_health.grad_norm_history.extend(gh_state['grad_norm_history'][-100:])
                    if 'grad_norm_pre_clip_history' in gh_state:
                        self.gradient_health.grad_norm_pre_clip_history.extend(gh_state['grad_norm_pre_clip_history'][-100:])

                    print(f"   ✓ Gradient health restored: {self.gradient_health.total_explosions} explosions, {self.gradient_health.total_steps} steps")
                    restored_states['gradient_health'] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore gradient health state: {e}")
                    restored_states['gradient_health'] = False
            else:
                restored_states['gradient_health'] = False

            # Restore memory monitor state
            if hasattr(self, 'memory_monitor') and 'memory_monitor_state' in checkpoint:
                try:
                    mem_state = checkpoint['memory_monitor_state']
                    # Memory monitor state doesn't need device handling
                    print(f"   ✓ Memory monitor state found: {mem_state.get('emergency_count', 0)} emergencies")
                    restored_states['memory_monitor'] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore memory monitor state: {e}")
                    restored_states['memory_monitor'] = False
            else:
                restored_states['memory_monitor'] = False

            # Restore loss health monitor state
            if hasattr(self, 'loss_health') and 'loss_health_state' in checkpoint:
                try:
                    lh_state = checkpoint['loss_health_state']
                    if 'loss_history' in lh_state:
                        self.loss_health.loss_history.extend(lh_state['loss_history'][-100:])
                    self.loss_health.spike_threshold = lh_state.get('spike_threshold', 5.0)
                    self.loss_health.nan_count = lh_state.get('nan_count', 0)
                    self.loss_health.inf_count = lh_state.get('inf_count', 0)
                    self.loss_health.spike_count = lh_state.get('spike_count', 0)

                    print(f"   ✓ Loss health restored: {self.loss_health.nan_count} NaN, {self.loss_health.inf_count} Inf, {self.loss_health.spike_count} spikes")
                    restored_states['loss_health'] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore loss health state: {e}")
                    restored_states['loss_health'] = False
            else:
                restored_states['loss_health'] = False

            # Restore random states for reproducibility
            if 'random_states' in checkpoint:
                try:
                    import random
                    import numpy as np
                    rand_states = checkpoint['random_states']

                    if 'python_random' in rand_states:
                        random.setstate(rand_states['python_random'])
                    if 'numpy_random' in rand_states:
                        np.random.set_state(rand_states['numpy_random'])
                    if 'torch_random' in rand_states:
                        torch.set_rng_state(rand_states['torch_random'])
                    if 'torch_cuda_random' in rand_states and rand_states['torch_cuda_random'] is not None:
                        if torch.cuda.is_available():
                            torch.cuda.set_rng_state(rand_states['torch_cuda_random'])

                    print("   ✓ Random states restored for reproducibility")
                    restored_states['random_states'] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore random states: {e}")
                    restored_states['random_states'] = False
            else:
                restored_states['random_states'] = False

            # Check for early stopping and training progress states (for informational purposes)
            early_stopping_info = {}
            if 'early_stopping_state' in checkpoint:
                early_stopping_info = checkpoint['early_stopping_state']
                print(f"   ℹ️  Early stopping state: enabled={early_stopping_info.get('enabled', False)}, "
                      f"patience={early_stopping_info.get('patience', 'N/A')}")

            training_progress_info = {}
            if 'training_progress' in checkpoint:
                training_progress_info = checkpoint['training_progress']
                print(f"   ℹ️  Training progress: epoch {training_progress_info.get('current_epoch', 'N/A')}/{training_progress_info.get('total_epochs', 'N/A')}, "
                      f"complete={training_progress_info.get('training_complete', False)}")

            print(f"✅ Checkpoint loaded: {checkpoint_file}")
            print(f"   States restored: {sum(restored_states.values())}/{len(restored_states)}")

            return {
                'step_count': self.step_count,
                'epoch_count': self.epoch_count,
                'best_loss': self.best_loss,
                'restored_states': restored_states,
                'early_stopping_info': early_stopping_info,
                'training_progress_info': training_progress_info,
                'checkpoint_file': str(checkpoint_file)
            }

    def cleanup(self):
        """Clean up all components."""
        # Stop observability first to export final data
        if self.observability:
            print("🔍 Cleaning up observability integration...")
            try:
                self.observability.stop_training_observation()
                self.observability.export_all_data()
                self.observability.shutdown()
            except Exception as e:
                print(f" Warning: Observability cleanup failed: {e}")

        if self.async_logger:
            self.async_logger.stop()

        if self.gpu_manager:
            self.gpu_manager.cleanup_gpu_memory(aggressive=True)

        # Health checker cleanup (must come before error handler cleanup)
        if self.health_checker:
            print(" Cleaning up health checker...")
            self.health_checker.cleanup()

        # Error handler cleanup (must come before distributed cleanup)
        if self.error_handler:
            print(" Cleaning up error handler...")
            self.error_handler.cleanup()

        # Distributed training cleanup (must come before DeepSpeed cleanup)
        if self.distributed_manager:
            print(" Cleaning up distributed training...")
            self.distributed_manager.cleanup()

        # DeepSpeed cleanup
        if self.deepspeed_engine:
            print(" Cleaning up DeepSpeed engine")
            # DeepSpeed handles its own cleanup automatically

        print(" Enhanced trainer cleanup completed")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'best_loss': self.best_loss
        }

        if self.metrics_collector:
            stats['metrics'] = self.metrics_collector.get_performance_summary()

        if self.adaptive_lr_manager:
            stats['adaptive_lr'] = self.adaptive_lr_manager.get_lr_statistics()
        else:
            stats['adaptive_lr'] = {'status': 'disabled'}

        if self.warmup_scheduler:
            stats['warmup'] = self.warmup_scheduler._get_warmup_info()

        if self.async_logger:
            stats['logging'] = self.async_logger.get_logging_statistics()

        if self.gpu_manager:
            stats['memory'] = self.gpu_manager.get_memory_stats()

        # Add health monitoring statistics
        if hasattr(self, 'gradient_health'):
            stats['gradient_health'] = self.gradient_health.get_health_summary()

        if hasattr(self, 'loss_health'):
            stats['loss_health'] = self.loss_health.get_health_summary()

        if hasattr(self, 'memory_monitor'):
            stats['memory_monitor'] = self.memory_monitor.get_memory_summary()

        # Add scaler information
        if self.scaler is not None:
            stats['mixed_precision'] = {
                'scaler_scale': self.scaler.get_scale(),
                'last_reset_step': getattr(self, 'scaler_last_reset', 0),
                'steps_since_reset': self.step_count - getattr(self, 'scaler_last_reset', 0)
            }

        # Add observability statistics (Phase 7)
        if self.observability:
            try:
                stats['observability'] = self.observability.get_observability_summary()
            except Exception as e:
                stats['observability'] = {'error': f'Failed to get summary: {e}', 'enabled': True}
        else:
            stats['observability'] = {'enabled': False}

        return stats