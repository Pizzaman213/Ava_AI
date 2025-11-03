"""
Enhanced Trainer using Modular Components

This module provides the core Enhanced Trainer that integrates all the
modular training components for maximum flexibility and maintainability.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from torch.amp import GradScaler, autocast  # type: ignore[import]

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import deepspeed  # type: ignore[import]

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None  # type: ignore[assignment]

from ...config.training_config import EnhancedTrainingConfig
from ...evaluation.comprehensive_eval import ComprehensiveEvaluator

# Import loss components from unified losses module
from ...losses import (
    AdaptiveLossScaling,
    CompositeLoss,
    DeepSeekLoss,
    NGramRepetitionPenalty,
    SequenceRepetitionDetector,
)

# Import Phase 4 episodic memory components - DISABLED (modules removed)
# from ..memory.episodic_memory import (
#     AdaptiveMemoryManager,
#     EpisodicMemoryBank,
#     ExperienceReplay,
# )
# Stub classes to prevent errors
class AdaptiveMemoryManager:
    def __init__(self, *args, **kwargs):
        pass
    def update_performance(self, *args, **kwargs):
        pass
    def get_importance_threshold(self, *args, **kwargs):
        return 0.5

class EpisodicMemoryBank:
    def __init__(self, *args, **kwargs):
        pass
    def add_experience(self, *args, **kwargs):
        pass
    def sample(self, *args, **kwargs):
        return []
    def __len__(self):
        return 0

class ExperienceReplay:
    def __init__(self, *args, **kwargs):
        pass
    def replay_batch(self, *args, **kwargs):
        return None

# Import Phase 7 observability components - DISABLED (modules removed)
# from ..observability.training_integration import (
#     ObservabilityConfig,
#     ObservabilityIntegration,
#     create_lightweight_observability,
#     create_observability_integration,
# )
# Stub classes to prevent errors
class ObservabilityConfig:
    def __init__(self, *args, **kwargs):
        """Accept any arguments to prevent initialization errors."""
        pass
class ObservabilityIntegration:
    def __init__(self, *args, **kwargs):
        pass
    def initialize(self, *args, **kwargs):
        pass
    def log_metrics(self, *args, **kwargs):
        pass
    def start_training_observation(self, *args, **kwargs):
        pass
    def handle_out_of_memory(self, *args, **kwargs):
        pass
    def handle_training_error(self, *args, **kwargs):
        pass
    def update_training_step(self, *args, **kwargs):
        pass
    def create_checkpoint_data(self, *args, **kwargs):
        return {}
    def stop_training_observation(self, *args, **kwargs):
        pass
    def export_all_data(self, *args, **kwargs):
        pass
    def shutdown(self, *args, **kwargs):
        pass
    def get_observability_summary(self, *args, **kwargs):
        return {}
def create_lightweight_observability(*args, **kwargs):
    return ObservabilityIntegration()
def create_observability_integration(*args, **kwargs):
    return ObservabilityIntegration()
from ...optimization.precision.quantization import ModelQuantizer
# from ...retrieval.rag_system import KnowledgeBase, RAGSystem  # DISABLED (module removed)
# Stub classes to prevent errors
class KnowledgeBase:
    pass
class RAGSystem:
    def __init__(self, *args, **kwargs):
        pass
from ...logging.async_logging import AsyncLogger, AsyncLoggingConfig

# Import all the new modular components
from ...utils.gpu_memory import GPUMemoryManager
from ...optimization.learning_rate import AdvancedWarmupScheduler, WarmupConfig, IntelligentLRManager, LRConfig
from ...distributed.distributed_health_checker import (
    DistributedHealthChecker,
    get_health_checker,
    record_training_metrics,
)
from ...distributed.distributed_manager import (
    DistributedConfig,
    DistributedManager,
    get_distributed_manager,
)
from ...optimization.gradients import GradientHealthMonitor, LossHealthMonitor, AdaptiveGradientSurgeon, GradientSurgeon
from ...distributed.memory_monitor import MemoryMonitor
from ..monitoring.metrics import MetricConfig, TrainingMetricsCollector
from ..monitoring.performance_modes import PerformanceModeConfig, PerformanceModeManager
from ...distributed.rank_aware_error_handler import (
    ErrorSeverity,
    ErrorType,
    RankAwareErrorHandler,
    get_error_handler,
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
        run_manager=None,
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

        # Adaptive learning rate manager (optional)
        self.adaptive_lr_manager = None

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
        # FIXED: Separate micro-steps (every forward/backward) from optimizer steps
        self.micro_step_count = 0  # Incremented every forward/backward pass
        self.optimizer_step_count = 0  # Incremented only when optimizer.step() is called
        self.step_count = 0  # Legacy support - points to micro_step_count
        self.epoch_count = 0
        self.best_loss = float("inf")

        # Running loss tracker for accurate checkpoint reporting
        self.running_loss_avg = 0.0
        self.running_loss_count = 0
        self.running_loss_window_size = 100  # EMA over last 100 batches

        # Enable gradient checkpointing for memory savings if configured
        self.gradient_checkpointing_enabled = getattr(
            config.training, "gradient_checkpointing", False
        )
        if self.gradient_checkpointing_enabled:
            self._enable_gradient_checkpointing()
            print("✓ Gradient checkpointing enabled for memory optimization")

        # Mixed precision gradient scaler with health monitoring
        self.scaler = (
            GradScaler('cuda') if torch.cuda.is_available() else None
        )
        self.scaler_reset_interval = (
            1000  # Reset scaler every N steps to prevent error accumulation
        )
        self.scaler_last_reset = 0

        # Initialize gradient and loss health monitors with config settings
        gh_config = config.training.gradient_health if hasattr(config.training, 'gradient_health') else {}  # type: ignore[attr-defined]

        # Handle both dict and object config formats
        def get_config_value(cfg, key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        # Check if gradient health is enabled in config (default: False for speed)
        self.gradient_health_enabled = get_config_value(gh_config, 'enabled', False)

        if self.gradient_health_enabled:
            self.gradient_health = GradientHealthMonitor(
                initial_clip_value=get_config_value(gh_config, 'initial_clip_value', 1.0),
                final_clip_value=get_config_value(gh_config, 'final_clip_value', 5.0),
                warmup_steps=get_config_value(gh_config, 'warmup_steps', 2000),
                history_size=100,
                explosion_threshold=get_config_value(gh_config, 'explosion_threshold', 1000.0),
            )
        else:
            self.gradient_health = None  # type: ignore[assignment]
        self.loss_health = LossHealthMonitor(
            history_size=100, spike_threshold_sigma=3.0, divergence_threshold=2.0
        )

        # Initialize memory monitor for proactive OOM prevention
        # Load memory config from config (supports both dict and object access)
        memory_silent = False
        target_util = 0.85
        warning_thresh = 0.985  # SPEED FIX: Raised to 98.5% to reduce false alarms
        critical_thresh = 0.992  # SPEED FIX: Raised to 99.2% to reduce cleanup frequency
        emergency_thresh = 0.998  # SPEED FIX: Raised to 99.8% - only cleanup on true emergencies

        if hasattr(config, 'memory') and config.memory:
            if hasattr(config.memory, 'silent_mode'):
                memory_silent = config.memory.silent_mode
            elif isinstance(config.memory, dict) and 'silent_mode' in config.memory:
                memory_silent = config.memory['silent_mode']

            # Load thresholds from config if available
            if hasattr(config.memory, 'target_utilization'):
                target_util = config.memory.target_utilization  # type: ignore[attr-defined]
            elif isinstance(config.memory, dict) and 'target_utilization' in config.memory:
                target_util = config.memory['target_utilization']

            if hasattr(config.memory, 'warning_threshold'):
                warning_thresh = config.memory.warning_threshold  # type: ignore[attr-defined]
            elif isinstance(config.memory, dict) and 'warning_threshold' in config.memory:
                warning_thresh = config.memory['warning_threshold']

            if hasattr(config.memory, 'critical_threshold'):
                critical_thresh = config.memory.critical_threshold  # type: ignore[attr-defined]
            elif isinstance(config.memory, dict) and 'critical_threshold' in config.memory:
                critical_thresh = config.memory['critical_threshold']

            if hasattr(config.memory, 'emergency_threshold'):
                emergency_thresh = config.memory.emergency_threshold  # type: ignore[attr-defined]
            elif isinstance(config.memory, dict) and 'emergency_threshold' in config.memory:
                emergency_thresh = config.memory['emergency_threshold']

        # DEBUG: Print actual threshold values being used
        print(f"🔧 Memory Monitor Configuration:")
        print(f"   Target utilization: {target_util:.1%}")
        print(f"   Warning threshold:  {warning_thresh:.1%}")
        print(f"   Critical threshold: {critical_thresh:.1%}")
        print(f"   Emergency threshold: {emergency_thresh:.1%}")

        self.memory_monitor = MemoryMonitor(
            target_utilization=target_util,
            warning_threshold=warning_thresh,
            critical_threshold=critical_thresh,
            emergency_threshold=emergency_thresh,
            history_size=100,
            memory_headroom_gb=1.0,
            silent_mode=memory_silent,
        )
        print("Gradient, loss, and memory monitors initialized")

        # Initialize dynamic batch sizer if enabled
        self.dynamic_batch_sizer = None
        print("🔍 DEBUG: Initializing dynamic batch sizer...")
        print(f"🔍 DEBUG: config type = {type(config)}")
        print(f"🔍 DEBUG: config.training type = {type(config.training)}")
        if hasattr(config.training, '__dict__'):
            print(f"🔍 DEBUG: config.training.__dict__ keys = {list(config.training.__dict__.keys())[:20]}")  # First 20 keys
        if hasattr(config.training, 'keys') and callable(getattr(config.training, 'keys')):
            print(f"🔍 DEBUG: config.training.keys() = {list(config.training.keys())[:20]}")  # If it's a dict  # type: ignore[attr-defined]
        try:
            # Check if dynamic_batching is configured
            dynamic_batching = None
            if hasattr(config.training, "dynamic_batching"):
                dynamic_batching = config.training.dynamic_batching
                print(f"🔍 DEBUG: Found dynamic_batching in config.training")
            elif hasattr(config, "dynamic_batching"):
                dynamic_batching = config.dynamic_batching  # type: ignore[attr-defined]
                print(f"🔍 DEBUG: Found dynamic_batching in config")
            else:
                print(f"🔍 DEBUG: dynamic_batching not found in config")
                # Try direct dict access
                if isinstance(config.training, dict) and 'dynamic_batching' in config.training:
                    dynamic_batching = config.training['dynamic_batching']
                    print(f"🔍 DEBUG: Found dynamic_batching via dict access!")
                elif hasattr(config, '__dict__') and 'training' in config.__dict__:
                    training_dict = config.__dict__['training']
                    if isinstance(training_dict, dict) and 'dynamic_batching' in training_dict:
                        dynamic_batching = training_dict['dynamic_batching']
                        print(f"🔍 DEBUG: Found dynamic_batching via __dict__ access!")

            if dynamic_batching:
                enabled = getattr(dynamic_batching, "enabled", False)
                print(f"🔍 DEBUG: dynamic_batching.enabled = {enabled}")

                if enabled:
                    try:
                        from ..strategies.progressive_training import DynamicBatchSizer  # type: ignore[import-not-found]
                    except ImportError:
                        print("⚠️  dynamic_batch_sampler module not found, skipping dynamic batching")
                        DynamicBatchSizer = None  # type: ignore

                    if DynamicBatchSizer is not None:
                        self.dynamic_batch_sizer = DynamicBatchSizer(
                            initial_batch_size=getattr(config.training, "batch_size", 8),
                            min_batch_size=getattr(dynamic_batching, "min_batch_size", 1),
                            max_batch_size=getattr(dynamic_batching, "max_batch_size", 64),
                            target_memory_utilization=getattr(dynamic_batching, "target_memory_utilization", 0.85),
                            adjustment_frequency=getattr(dynamic_batching, "adjustment_frequency", 100),
                            adjustment_factor=getattr(dynamic_batching, "adjustment_factor", 1.25),
                            warmup_steps=getattr(dynamic_batching, "warmup_steps", 500),
                            smooth_transitions=getattr(dynamic_batching, "smooth_transitions", True),
                            memory_monitor=self.memory_monitor
                        )
                        print(f"✓ Dynamic batch sizing enabled: {self.dynamic_batch_sizer.min_batch_size}-{self.dynamic_batch_sizer.max_batch_size}")
                else:
                    print("ℹ️ Dynamic batching configured but disabled")
            else:
                print("ℹ️ Dynamic batching not configured")
        except Exception as e:
            import traceback
            print(f"⚠️  Could not initialize dynamic batch sizing: {e}")
            print(f"⚠️  Traceback: {traceback.format_exc()}")
            self.dynamic_batch_sizer = None

        # Checkpoint restore settings
        self.last_valid_checkpoint_path = None
        self.checkpoint_restore_attempts = 0
        self.max_restore_attempts = 3

    @property
    def _get_base_model(self):
        """Get the base model, handling DeepSpeed wrapping."""
        if self.deepspeed_engine is not None:
            return self.deepspeed_engine.module
        return self.model

    def _init_gpu_memory_manager(self):
        """Initialize GPU memory manager."""
        self.gpu_manager = GPUMemoryManager(auto_cleanup=True)
        print("GPU memory manager initialized")

    def _init_performance_manager(self):
        """Initialize performance mode manager."""
        # Convert PerformanceConfig to PerformanceModeConfig
        from ..monitoring.performance_modes import PerformanceMode, PerformanceModeConfig

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
            async_logging=not self.config.performance.ultra_fast_mode,
        )

        self.performance_manager = PerformanceModeManager(perf_mode_config)
        summary = self.performance_manager.get_performance_summary()
        print(f"Performance mode: {summary['mode']}")
        print(f"Optimizations: {', '.join(summary['active_optimizations'])}")

    def _init_distributed_manager(self):
        """Initialize distributed training manager."""
        from ...distributed.distributed_manager import (
            DistributedConfig,
            get_distributed_manager,
            is_distributed,
        )

        # Check if we're in a distributed environment
        if not is_distributed():
            print("Single-node training mode")
            return

        print("🚀 Initializing distributed training manager...")

        # Create distributed configuration from training config
        distributed_config = DistributedConfig(
            backend=getattr(self.config.deepspeed, "backend", "nccl"),
            timeout_seconds=getattr(self.config.training, "distributed_timeout", 1800),
            enable_barriers=getattr(self.config.training, "enable_barriers", True),
            enable_heartbeat=getattr(self.config.training, "enable_heartbeat", True),
            enable_rank_failure_recovery=getattr(
                self.config.training, "enable_rank_failure_recovery", True
            ),
            max_failed_ranks=getattr(self.config.training, "max_failed_ranks", 1),
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
            self.distributed_manager.register_cleanup_handler(
                self._distributed_cleanup_handler
            )
        else:
            print("❌ Failed to initialize distributed training")
            self.distributed_manager = None

    def _distributed_cleanup_handler(self):
        """Cleanup handler for distributed training."""
        print("🧹 Distributed trainer cleanup handler called")
        # Add any trainer-specific distributed cleanup here

    def _init_error_handler(self):
        """Initialize rank-aware error handler."""
        from ...distributed.distributed_manager import is_distributed

        if not is_distributed():
            print("Single-node training: error handler disabled")
            return

        print("🛡️ Initializing rank-aware error handler...")

        try:
            # Get error handler with configuration
            self.error_handler = get_error_handler(
                max_retries=getattr(self.config.training, "max_error_retries", 3),
                heartbeat_interval=getattr(
                    self.config.training, "error_heartbeat_interval", 10.0
                ),
                failure_timeout=getattr(
                    self.config.training, "rank_failure_timeout", 60.0
                ),
                enable_recovery=getattr(
                    self.config.training, "enable_error_recovery", True
                ),
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
        if hasattr(self, "async_logger") and self.async_logger:
            self.async_logger.log_metrics(
                {
                    f"rank_failure_{failed_rank}": 1,
                    "error_type": error_info.error_type.value,
                    "severity": error_info.severity.value,
                }
            )

    def _init_health_checker(self):
        """Initialize distributed health checker."""
        from ...distributed.distributed_manager import is_distributed

        if not is_distributed():
            print("Single-node training: health checker disabled")
            return

        print("🩺 Initializing distributed health checker...")

        try:
            # Get health checker with configuration
            self.health_checker = get_health_checker(
                check_interval=getattr(
                    self.config.training, "health_check_interval", 30.0
                ),
                loss_history_size=getattr(
                    self.config.training, "health_history_size", 100
                ),
                anomaly_threshold=getattr(
                    self.config.training, "health_anomaly_threshold", 2.5
                ),
                enable_gradient_sync=getattr(
                    self.config.training, "enable_gradient_health_sync", True
                ),
                enable_performance_sync=getattr(
                    self.config.training, "enable_performance_health_sync", True
                ),
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
            "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
        ) or ("LOCAL_RANK" in os.environ)

        if not self.is_distributed:
            print(" DeepSpeed enabled for single-GPU training (ZeRO optimizations)")
            # Set complete distributed environment for single GPU
            import os
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('LOCAL_RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            os.environ.setdefault('MASTER_ADDR', 'localhost')
            os.environ.setdefault('MASTER_PORT', '29500')  # CRITICAL: Must set port
            self.is_distributed = True  # Enable for single GPU too
            print(f"   Set distributed env: RANK=0, WORLD_SIZE=1, MASTER_PORT=29500")

        # Load or generate DeepSpeed configuration
        self.deepspeed_config = self._create_deepspeed_config()

        print(f"✓ DeepSpeed configuration prepared (ZeRO Stage {self.config.deepspeed.zero_stage})")

    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration dictionary."""
        ds_config = self.config.deepspeed

        # FIXED: Gradient accumulation should come from training config, not deepspeed config
        # DeepSpeed will handle gradient accumulation internally when enabled
        gradient_accum_steps = getattr(self.config.training, "gradient_accumulation_steps", 1)

        config = {
            "train_batch_size": ds_config.train_batch_size or self.config.training.batch_size,
            "train_micro_batch_size_per_gpu": ds_config.micro_batch_size or self.config.training.batch_size,
            "gradient_accumulation_steps": gradient_accum_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "weight_decay": "auto",
                    "beta1": "auto",
                    "beta2": "auto",
                    "eps": "auto",
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": self.config.training.learning_rate,
                    "warmup_num_steps": getattr(self.config.training, 'warmup_steps', 1000),
                },
            },
            "zero_optimization": {
                "stage": ds_config.zero_stage,
                "allgather_partitions": ds_config.allgather_partitions,
                "allgather_bucket_size": ds_config.zero_allgather_bucket_size,
                "overlap_comm": ds_config.overlap_comm,
                "reduce_scatter": ds_config.zero_reduce_scatter,
                "reduce_bucket_size": ds_config.zero_reduce_bucket_size,
                "contiguous_gradients": ds_config.zero_contiguous_gradients,
            },
            "gradient_clipping": ds_config.gradient_clipping or 1.0,
            "wall_clock_breakdown": False,  # FIXED: Disable timers to avoid engine_timers error
            "data_types": {"grad_accum_dtype": "fp32", "params_dtype": "fp32"},
            "steps_per_print": 100000,  # Reduce DeepSpeed logging overhead
            "tensorboard": {
                "enabled": False  # Disable DeepSpeed's tensorboard to avoid overhead
            },
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
                    "min_loss_scale": 1,
                }
            elif ds_config.precision_type == "bf16":
                config["bf16"] = {"enabled": True}

        # Add ZeRO stage-specific configurations
        if ds_config.zero_stage == 3:
            config["zero_optimization"].update(
                {
                    "stage3_prefetch_bucket_size": ds_config.zero_stage3_prefetch_bucket_size,
                    "stage3_param_persistence_threshold": ds_config.zero_stage3_param_persistence_threshold,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_gather_16bit_weights_on_model_save": True,
                }
            )

        # Add CPU offloading
        if ds_config.cpu_offload:
            # CRITICAL: Allow custom optimizer with ZeRO-Offload
            config["zero_force_ds_cpu_optimizer"] = False

            if ds_config.zero_stage == 2:
                config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            elif ds_config.zero_stage == 3:
                config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
                config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

        # Add NVMe offloading
        if ds_config.nvme_offload and ds_config.cpu_offload:
            config["zero_optimization"]["offload_optimizer"][
                "nvme_path"
            ] = "/local_nvme"
            if ds_config.zero_stage == 3:
                config["zero_optimization"]["offload_param"][
                    "nvme_path"
                ] = "/local_nvme"

        # Add activation checkpointing
        if ds_config.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": ds_config.partition_activations,
                "cpu_checkpointing": ds_config.cpu_checkpointing,
                "contiguous_memory_optimization": ds_config.contiguous_memory_optimization,
                "synchronize_checkpoint_boundary": ds_config.synchronize_dp_processes,
            }

        return config

    def _init_async_logger(self):
        """Initialize async logging system."""
        if self.performance_manager.should_use_async_logging():
            logging_config = AsyncLoggingConfig(
                enable_system_metrics=True,
                wandb_cache_size=self.config.wandb.wandb_cache_size,
                wandb_flush_interval=self.config.wandb.wandb_cache_flush_interval,
            )
            self.async_logger = AsyncLogger(
                logging_config,
                wandb_available=not self.config.wandb.disable_wandb,
                wandb_offline=self.config.wandb.wandb_offline,
                disable_wandb=self.config.wandb.disable_wandb,
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
                collect_system=self.performance_manager.config.mode.value
                != "minimal_progress",
                gradient_freq=100,
                memory_freq=50,
            )
            self.metrics_collector = TrainingMetricsCollector(metrics_config)
            print("Training metrics collector initialized")
        else:
            self.metrics_collector = None

    def _init_loss_functions(self):
        """Initialize advanced loss functions."""
        # Always initialize DeepSeek-style loss if multi-token prediction is enabled
        use_multi_token_prediction = getattr(self.config.losses, "use_multi_token_prediction", False)

        if use_multi_token_prediction:
            # Initialize DeepSeek-style loss
            vocab_size = getattr(self.config.model, "vocab_size", 32000)
            hidden_size = getattr(self.config.model, "hidden_size", 4096)
            num_experts = getattr(self.config.model, "num_experts", None)

            # Get EOS token ID from tokenizer
            eos_token_id = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                eos_token_id = self.tokenizer.eos_token_id

            self.deepseek_loss = DeepSeekLoss(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_experts=num_experts,
                # Multi-token prediction settings
                use_mtp=getattr(self.config.losses, "use_multi_token_prediction", True),
                num_future_tokens=getattr(self.config.losses, "num_future_tokens", 3),
                mtp_weight=getattr(self.config.losses, "mtp_weight", 0.1),
                # Temperature scaling settings
                initial_temperature=getattr(self.config.losses, "initial_temperature", 1.0),
                adaptive_temperature=getattr(self.config.losses, "adaptive_temperature", True),
                label_smoothing=getattr(self.config.losses, "label_smoothing", 0.1),
                # MoE balancing settings
                use_moe_balancing=getattr(self.config.losses, "use_moe_balancing", True),
                gradient_balance_weight=getattr(self.config.losses, "gradient_balance_weight", 0.1),
                # EOS penalty settings (CRITICAL FIX for early termination)
                eos_token_id=eos_token_id,
                min_sequence_length=getattr(self.config.training, "min_sequence_length", 20),
                eos_penalty_weight=getattr(self.config.training, "eos_penalty_weight", 5.0)
            )

            # CRITICAL: Move to device and convert to BF16 if using BF16 training
            if getattr(self.config.training, "mixed_precision", "fp16") == "bf16":
                self.deepseek_loss = self.deepseek_loss.to(device=self.device, dtype=torch.bfloat16)
            else:
                self.deepseek_loss = self.deepseek_loss.to(self.device)
            self.composite_loss = None
            print("✓ DeepSeek-style loss initialized with MTP and auxiliary-free balancing")
        else:
            # Original composite loss configuration
            self.deepseek_loss = None
            loss_config = {}
            if any(
                [
                    self.config.losses.use_focal_loss,
                    self.config.losses.use_contrastive_loss,
                    self.config.losses.use_diversity_loss,
                ]
            ):
                if self.config.losses.use_focal_loss:
                    focal_alpha = getattr(self.config.losses, "focal_alpha", 1.0)
                    focal_gamma = getattr(self.config.losses, "focal_gamma", 2.0)
                    focal_weight = getattr(self.config.losses, "focal_loss_weight", 0.1)
                    loss_config["focal"] = {
                        "type": "focal",
                        "alpha": focal_alpha,
                        "gamma": focal_gamma,
                        "weight": focal_weight,
                    }
                if self.config.losses.use_contrastive_loss:
                    contrastive_weight = getattr(self.config.losses, "contrastive_loss_weight", 0.05)
                    contrastive_temp = getattr(self.config.losses, "contrastive_temperature", 0.07)
                    loss_config["contrastive"] = {
                        "type": "contrastive",
                        "temperature": contrastive_temp,
                        "weight": contrastive_weight,
                    }
                if self.config.losses.use_diversity_loss:
                    diversity_weight = getattr(self.config.losses, "diversity_loss_weight", 0.05)
                    loss_config["diversity"] = {"type": "diversity", "weight": diversity_weight}

                # Add auxiliary loss only if enabled (for MoE stability)
                if getattr(self.config.losses, "use_auxiliary_loss", True):
                    aux_weight = getattr(self.config.losses, "auxiliary_loss_weight", 0.001)
                    aux_load_balance = getattr(self.config.losses, "auxiliary_load_balancing_weight", 0.0001)
                    aux_router_z = getattr(self.config.losses, "auxiliary_router_z_weight", 0.00001)
                    loss_config["auxiliary"] = {
                        "type": "auxiliary",
                        "load_balancing_weight": aux_load_balance,
                        "router_z_weight": aux_router_z,
                        "weight": aux_weight,
                    }

                self.composite_loss = CompositeLoss(loss_config)
                print(f" Advanced loss functions: {list(loss_config.keys())}")
            else:
                self.composite_loss = None

        # Adaptive loss scaling
        if self.config.losses.adaptive_loss_scaling:
            if self.deepseek_loss is not None:
                # For DeepSeek loss, we have main + mtp + moe components
                num_losses = 3
            elif self.composite_loss is not None:
                # loss_config is guaranteed to exist when composite_loss is not None
                num_losses = len(loss_config) if loss_config else 1  # type: ignore[possibly-unbound]
            else:
                num_losses = 1
            self.adaptive_scaler = AdaptiveLossScaling(num_losses=num_losses)
        else:
            self.adaptive_scaler = None

        # Initialize repetition penalties (DEFAULT: ENABLED to prevent mode collapse)
        # CRITICAL FIX: Support both config.enhanced_features.losses and config.losses paths
        # Most configs have enhanced_features.losses, but allow config.losses as fallback
        losses_config = None
        if hasattr(self.config, 'enhanced_features') and hasattr(self.config.enhanced_features, 'losses'):  # type: ignore[attr-defined]
            losses_config = self.config.enhanced_features.losses  # type: ignore[attr-defined]
            print("📊 Using enhanced_features.losses config path")
        elif hasattr(self.config, 'losses'):
            losses_config = self.config.losses
            print("📊 Using losses config path")
        else:
            print("⚠️  No losses config found, using defaults")

        use_ngram_penalty = getattr(losses_config, "use_ngram_penalty", True) if losses_config else True
        use_repetition_detector = getattr(losses_config, "use_immediate_repetition_detector", True) if losses_config else True

        if use_ngram_penalty:
            vocab_size = getattr(self.config.model, "vocab_size", 50257)
            ngram_size = getattr(losses_config, "ngram_size", 3) if losses_config else 3
            ngram_weight = getattr(losses_config, "ngram_penalty_weight", 2.0) if losses_config else 2.0

            self.ngram_penalty = NGramRepetitionPenalty(
                ngram_size=ngram_size,
                penalty_weight=ngram_weight,
                vocab_size=vocab_size
            ).to(self.device)
            print(f"✓ N-gram repetition penalty initialized (n={ngram_size}, weight={ngram_weight})")
        else:
            self.ngram_penalty = None
            print("⚠️  N-gram repetition penalty DISABLED")

        if use_repetition_detector:
            immediate_weight = getattr(losses_config, "immediate_repetition_weight", 3.0) if losses_config else 3.0

            self.repetition_detector = SequenceRepetitionDetector(
                penalty_weight=immediate_weight
            ).to(self.device)
            print(f"✓ Immediate repetition detector initialized (weight={immediate_weight})")
        else:
            self.repetition_detector = None
            print("⚠️  Immediate repetition detector DISABLED")

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
                    fusion_type=self.config.rag.rag_fusion_type,
                )

                if self.config.rag.knowledge_base_path:
                    kb_path = Path(self.config.rag.knowledge_base_path)
                    if kb_path.exists():
                        self.knowledge_base = KnowledgeBase(embedding_dim=256)  # type: ignore[call-arg]
                        self.knowledge_base.load(kb_path)  # type: ignore[attr-defined]
                        self.rag_system.set_knowledge_base(self.knowledge_base)  # type: ignore[attr-defined]
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
                "tokenizer_name": "gpt2",  # Default
                "bleu_max_n": 4,
                "rouge_types": ["rouge-1", "rouge-2", "rouge-l"],
            }
            self.evaluator = ComprehensiveEvaluator(config=eval_config)

            if self.config.evaluation.eval_metrics:
                self.eval_metrics = self.config.evaluation.eval_metrics.split(",")
            else:
                self.eval_metrics = ["perplexity"]

            print(f" Evaluator: {', '.join(self.eval_metrics)}")
        else:
            self.evaluator = None

    def _init_quantization(self):
        """Initialize quantization."""
        if (
            self.config.quantization.quantization_aware
            or self.config.quantization.use_nvfp4
        ):
            if self.config.quantization.use_nvfp4:
                quant_config = QuantizationConfig(
                    bit_width=4,
                    use_nvfp4=True,
                    nvfp4_block_size=self.config.quantization.nvfp4_block_size,
                    stochastic_rounding=self.config.quantization.stochastic_rounding,
                    use_hadamard_transform=self.config.quantization.use_hadamard_transform,
                    symmetric=True,
                    per_channel=True,
                )
                print(" NVFP4 4-bit quantization enabled")
            else:
                quant_config = QuantizationConfig(
                    bit_width=self.config.quantization.bit_width,
                    symmetric=True,
                    per_channel=True,
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
                    importance_threshold=self.config.memory.memory_importance_threshold,
                )

                self.memory_manager = AdaptiveMemoryManager(
                    memory_bank=self.memory_bank,
                    adaptation_rate=self.config.memory.memory_adaptation_rate,
                    performance_window=self.config.memory.memory_performance_window,
                )

                self.experience_replay = ExperienceReplay(
                    memory_bank=self.memory_bank,
                    replay_ratio=self.config.memory.memory_replay_ratio,
                    replay_strategy=self.config.memory.memory_replay_strategy,
                )

                print(
                    f" Episodic memory: {self.config.memory.memory_capacity} capacity"
                )
            except Exception as e:
                print(f" Memory initialization failed: {e}")
                self.memory_bank = None
                self.memory_manager = None
                self.experience_replay = None
        else:
            self.memory_bank = None
            self.memory_manager = None
            self.experience_replay = None

    def _handle_memory_pressure(self, memory_health: Dict[str, Any]):
        """Handle memory pressure with dynamic adjustments for LLM training."""
        if not hasattr(self, "_memory_pressure_state"):
            self._memory_pressure_state = {
                "consecutive_warnings": 0,
                "last_action_step": 0,
                "original_grad_accumulation": getattr(
                    self.config.training, "gradient_accumulation_steps", 1
                ),
                "memory_adjustments_made": 0,
            }

        state = self._memory_pressure_state
        current_status = memory_health["status"]

        # React to warning, critical, or emergency status
        if current_status in ["warning", "critical", "emergency"]:
            state["consecutive_warnings"] += 1

            # Determine action thresholds based on severity
            if current_status == "emergency":
                required_warnings = 1  # Immediate action
                min_steps_between_actions = 10
            elif current_status == "critical":
                required_warnings = 2  # Quick action
                min_steps_between_actions = 25
            else:  # warning
                required_warnings = 3  # Conservative action
                min_steps_between_actions = 50

            # Take action if we've had enough consecutive warnings and waited enough steps
            if (
                state["consecutive_warnings"] >= required_warnings
                and self.step_count > state["last_action_step"] + min_steps_between_actions
            ):
                # First try increasing gradient accumulation
                current_grad_accum = getattr(
                    self,
                    "_current_grad_accumulation",
                    state["original_grad_accumulation"],
                )

                # Determine appropriate accumulation increase
                if current_status == "emergency":
                    multiplier = 4  # Aggressive
                elif current_status == "critical":
                    multiplier = 2  # Moderate
                else:
                    multiplier = 1.5  # Conservative

                new_grad_accum = min(int(current_grad_accum * multiplier), 32)  # Cap at 32

                if new_grad_accum > current_grad_accum:
                    self._current_grad_accumulation = new_grad_accum
                    state["last_action_step"] = self.step_count
                    state["consecutive_warnings"] = 0
                    state["memory_adjustments_made"] += 1

                    print(
                        f"    🔄 Memory pressure relief: increased gradient accumulation {current_grad_accum} → {new_grad_accum}"
                    )
                    print(
                        f"       Status: {current_status}, GPU: {memory_health['gpu_utilization']:.1%}"
                    )

                    # Force garbage collection after adjustment
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                else:
                    # If we can't increase grad accumulation further, enable more aggressive measures
                    if state["memory_adjustments_made"] < 3:
                        print(f"    ⚠️  Memory pressure persists at {memory_health['gpu_utilization']:.1%}, enabling aggressive cleanup")
                        if hasattr(self, 'memory_monitor'):
                            cleanup_stats = self.memory_monitor.cleanup_memory(aggressive=True)
                            print(f"    🧹 Aggressive cleanup freed {cleanup_stats.get('freed_gb', 0):.2f}GB")
                        state["last_action_step"] = self.step_count
                        state["memory_adjustments_made"] += 1

        else:
            # Reset warnings if memory pressure reduced to healthy levels
            if current_status == "healthy":
                state["consecutive_warnings"] = 0
            else:
                state["consecutive_warnings"] = max(0, state["consecutive_warnings"] - 1)

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory optimization."""
        try:
            # For HuggingFace models
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
                print("    ✓ HuggingFace gradient checkpointing enabled")

            # For custom models with explicit layer checkpointing
            elif hasattr(self.model, "enable_gradient_checkpointing"):
                self.model.enable_gradient_checkpointing()  # type: ignore[attr-defined]
                print("    ✓ Custom gradient checkpointing enabled")

            # For models with transformer layers, enable layer-wise checkpointing
            elif hasattr(self.model, "transformer") and hasattr(
                self.model.transformer, "h"  # type: ignore[attr-defined]
            ):
                # Enable checkpointing on transformer layers
                for layer in self.model.transformer.h:  # type: ignore[attr-defined]
                    if hasattr(layer, "gradient_checkpointing"):
                        layer.gradient_checkpointing = True
                print("    ✓ Layer-wise gradient checkpointing enabled")

            else:
                print(
                    "    ⚠️  Gradient checkpointing not supported by this model architecture"
                )
                self.gradient_checkpointing_enabled = False

        except Exception as e:
            print(f"    ⚠️  Failed to enable gradient checkpointing: {e}")
            self.gradient_checkpointing_enabled = False

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
                    log_level=(
                        "DEBUG"
                        if self.performance_manager.config.mode.value == "standard"
                        else "INFO"
                    ),
                    dashboard_update_interval=(
                        5.0
                        if self.performance_manager.config.mode.value == "standard"
                        else 10.0
                    ),
                    metric_sampling_rate=(
                        0.2
                        if self.performance_manager.config.mode.value == "standard"
                        else 0.1
                    ),
                    enable_async_metrics=self.performance_manager.config.async_logging,
                    # Integration settings
                    export_interval=100,
                    checkpoint_observability=True,
                )

                self.observability = ObservabilityIntegration(obs_config)
                print("🔍 Full observability integration initialized")

            # Prepare training context for initialization
            training_context = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "device": self.device,
                "config": self.config,
                "distributed": self.is_distributed,
                "deepspeed": self.deepspeed_engine is not None,
                # Note: train_loader and optimizer will be added later when available
                # Pre-flight validation will run again in train.py with full context
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

        setup_info.update(
            {
                "performance_mode": self.performance_manager.config.mode.value,
                "async_logging": self.async_logger is not None,
                "metrics_collection": self.metrics_collector is not None,
                "deepspeed_enabled": self.deepspeed_engine is not None,
                "observability_enabled": self.observability is not None,
            }
        )

        # Start observability observation if enabled
        if self.observability:
            # Enhanced training context with optimizer information
            enhanced_context = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "device": self.device,
                "config": self.config,
                "optimizer": optimizer,
                "distributed": self.is_distributed,
                "deepspeed": self.deepspeed_engine is not None,
                "performance_mode": self.performance_manager.config.mode.value,
            }
            self.observability.start_training_observation(enhanced_context)

        print(" Training setup completed")
        return setup_info

    def _setup_deepspeed_training(
        self, optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """Set up DeepSpeed training with robust initialization."""
        try:
            # Validate distributed training environment
            import torch.distributed as dist  # type: ignore[import]

            if not dist.is_available():
                raise RuntimeError("PyTorch distributed not available for DeepSpeed")

            # More robust distributed initialization check
            initialization_success = False

            # Check if distributed is already initialized
            if dist.is_initialized():
                print(
                    f"Distributed already initialized: rank {dist.get_rank()}/{dist.get_world_size()}"
                )
                initialization_success = True
            else:
                # Try to initialize with comprehensive environment variable checking
                required_vars = ["RANK", "WORLD_SIZE"]
                optional_vars = ["MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]

                missing_vars = [var for var in required_vars if var not in os.environ]
                if missing_vars:
                    raise RuntimeError(
                        f"DeepSpeed requires distributed training environment variables. "
                        f"Missing: {missing_vars}. "
                        f"Please set RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT."
                    )

                # Log environment variables for debugging
                env_info = {
                    var: os.environ.get(var, "Not set")
                    for var in required_vars + optional_vars
                }
                print(f"DeepSpeed environment: {env_info}")

                # Try initialization with error handling
                try:
                    print("Initializing torch.distributed for DeepSpeed...")
                    backend = "nccl" if torch.cuda.is_available() else "gloo"
                    dist.init_process_group(backend=backend)
                    print(
                        f"Distributed initialized successfully: rank {dist.get_rank()}/{dist.get_world_size()}"
                    )
                    initialization_success = True
                except Exception as init_error:
                    raise RuntimeError(
                        f"Failed to initialize distributed training: {init_error}"
                    )

            if not initialization_success:
                raise RuntimeError(
                    "Could not initialize distributed training for DeepSpeed"
                )

            # Validate world size
            world_size = dist.get_world_size()
            if world_size < 1:
                raise RuntimeError(f"Invalid world size: {world_size}")

            # Load and merge DeepSpeed configs more carefully
            final_config = (
                self.deepspeed_config.copy() if self.deepspeed_config is not None else {}
            )  # Start with programmatic config

            if self.config.deepspeed.config_file:
                import json

                config_file_path = self.config.deepspeed.config_file
                print(f"Loading DeepSpeed config from: {config_file_path}")

                try:
                    with open(config_file_path, "r") as f:
                        file_config = json.load(f)

                    # Smart merge: keep our feature flags, but allow file to override training settings
                    protected_keys = [
                        "train_batch_size",
                        "train_micro_batch_size_per_gpu",
                        "gradient_accumulation_steps",
                    ]
                    for key, value in file_config.items():
                        if key in protected_keys:
                            print(
                                f"  File config overriding {key}: {final_config.get(key)} -> {value}"
                            )
                        final_config[key] = value

                except Exception as config_error:
                    print(
                        f"Warning: Failed to load DeepSpeed config file: {config_error}"
                    )
                    print("  Using programmatic config only")

            # Validate config before initialization
            required_config_keys = ["train_batch_size", "zero_optimization"]
            missing_config = [
                key for key in required_config_keys if key not in final_config
            ]
            if missing_config:
                raise RuntimeError(
                    f"DeepSpeed config missing required keys: {missing_config}"
                )

            print(f"Final DeepSpeed config keys: {list(final_config.keys())}")

            # Wrap model for DeepSpeed compatibility
            try:
                from ..models.deepspeed_wrapper import DeepSpeedModelWrapper  # type: ignore[import-not-found]
                wrapped_model = DeepSpeedModelWrapper(self.model)
                print("  ✓ Model wrapped for DeepSpeed compatibility")
            except ImportError:
                print("⚠️  deepspeed_wrapper not found, using unwrapped model")
                wrapped_model = self.model

            # Initialize DeepSpeed engine with better error handling
            try:
                self.deepspeed_engine, optimizer, _, lr_scheduler = (
                    deepspeed.initialize(  # type: ignore[union-attr]
                        model=wrapped_model,  # Use wrapped model
                        optimizer=optimizer,
                        config=final_config,
                        lr_scheduler=None,  # We'll handle LR scheduling manually
                    )
                )
            except Exception as ds_init_error:
                print(f"DeepSpeed initialize() failed: {ds_init_error}")
                print("  This might be due to:")
                print("  - Incompatible config settings")
                print("  - Insufficient GPU memory")
                print("  - Model architecture incompatibility")
                # Ensure clean state on failure
                self.deepspeed_engine = None
                self.config.deepspeed.use_deepspeed = False
                raise

            # Workaround: Initialize engine_timers if missing (DeepSpeed compatibility fix)
            if not hasattr(self.deepspeed_engine, 'engine_timers'):
                print("⚠️  Warning: DeepSpeed engine_timers not initialized, disabling timers")
                # Create a dummy timer object to prevent AttributeError
                from types import SimpleNamespace
                self.deepspeed_engine.engine_timers = SimpleNamespace(
                    forward_timers=None,
                    backward_timers=None,
                    step_timers=None
                )
                # Also disable the timer start/stop methods
                self.deepspeed_engine._start_timers = lambda *args, **kwargs: None
                self.deepspeed_engine._stop_timers = lambda *args, **kwargs: None

            # Store references safely
            # IMPORTANT: Do NOT overwrite self.model - keep both references
            # self.model stays as the original model
            # self.deepspeed_engine is the DeepSpeed wrapper
            self.optimizer = optimizer  # DeepSpeed-managed optimizer
            self.lr_scheduler = lr_scheduler  # May be None

            # Validate the engine was created properly
            if not hasattr(self.deepspeed_engine, "backward"):
                raise RuntimeError("DeepSpeed engine missing required methods")

            print(f" DeepSpeed initialized successfully:")
            print(f"   ZeRO stage: {self.config.deepspeed.zero_stage}")
            print(f"   World size: {world_size}")
            print(f"   Rank: {dist.get_rank()}")
            print(f"   LR scheduler: {'Yes' if lr_scheduler else 'No'}")

            return {
                "deepspeed_engine": True,
                "zero_stage": self.config.deepspeed.zero_stage,
                "world_size": world_size,
                "rank": dist.get_rank(),
                "warmup_enabled": lr_scheduler
                is not None,  # If DeepSpeed provides scheduler
                "adaptive_lr_enabled": lr_scheduler is not None,
            }

        except Exception as e:
            print(f" DeepSpeed initialization failed: {e}")
            print("   Falling back to standard training")

            # Cleanup any partial initialization
            if hasattr(self, "deepspeed_engine"):
                self.deepspeed_engine = None

            return self._setup_standard_training(optimizer)

    def _setup_standard_training(
        self, optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """Set up standard (non-DeepSpeed) training with intelligent LR management."""
        # CRITICAL FIX: Wrap model in DDP for non-DeepSpeed distributed training
        if self.distributed_manager and self.distributed_manager.is_initialized():
            if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                print(f"🔄 Wrapping model in DistributedDataParallel...")
                print(f"   Rank: {self.distributed_manager.rank}, Device: {self.device}")

                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.device.index] if self.device.type == "cuda" else None,
                    output_device=self.device.index if self.device.type == "cuda" else None,
                    find_unused_parameters=False,  # Set to True only if needed
                    broadcast_buffers=True,
                    gradient_as_bucket_view=True,  # Memory optimization
                )
                print(f"✅ Model wrapped in DDP successfully")

        # Get training parameters
        num_epochs = self.config.training.epochs if self.config.training.epochs else 3
        batch_size = getattr(self.config.training, "batch_size", 8) or 8
        gradient_accumulation_steps = (
            getattr(self.config.training, "gradient_accumulation_steps", 1) or 1
        )

        # Create intelligent LR configuration
        # Calculate min_lr_ratio from config's lr_end and learning_rate
        learning_rate = getattr(self.config.training, "learning_rate", 1e-4)
        lr_end = getattr(self.config.training, "lr_end", learning_rate * 0.1)
        min_lr_ratio = lr_end / learning_rate if learning_rate > 0 else 0.1

        # Calculate warmup_ratio from warmup_steps if provided, else use warmup_ratio
        warmup_steps_config = getattr(self.config.training, "warmup_steps", None)
        warmup_ratio_config = getattr(self.config.training, "warmup_ratio", 0.1)

        lr_config = LRConfig(
            warmup_ratio=warmup_ratio_config,  # Will be overridden if warmup_steps is set
            warmup_min_ratio=0.01,
            warmup_schedule="linear",
            main_schedule="cosine",
            min_lr_ratio=min_lr_ratio,  # Respect config's lr_end
            enable_adaptive=getattr(self.config.training, "enable_adaptive_lr", False),  # Disable by default
            plateau_patience=getattr(self.config.training, "plateau_patience", 10),
            plateau_threshold=0.01,
            plateau_factor=0.5,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_lr_recovery=True,
        )

        # DISABLED: IntelligentLRManager (using AdaptiveLRManager from train.py instead)
        # This gets overridden in train.py with adaptive_lr_manager
        self.lr_manager = None

        # Store for later dataset size calculation
        self.training_params = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }

        # Legacy schedulers set to None (replaced by lr_manager)
        self.warmup_scheduler = None
        self.lr_scheduler = None

        print(f"✓ Intelligent LR management enabled:")
        print(f"   Warmup ratio: {lr_config.warmup_ratio:.1%}")
        print(
            f"   Adaptive LR: {'Enabled' if lr_config.enable_adaptive else 'Disabled'}"
        )
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")

        return {
            "deepspeed_engine": False,
            "warmup_enabled": True,
            "adaptive_lr_enabled": lr_config.enable_adaptive,
            "intelligent_lr_enabled": True,
            "gradient_accumulation_aware": True,
        }

    def setup_dataset_aware_lr(self, train_loader, dataset_size: Optional[int] = None):
        """
        Configure LR manager with actual dataset information.

        Args:
            train_loader: Training dataloader
            dataset_size: Optional dataset size override
        """
        if not hasattr(self, "lr_manager") or self.lr_manager is None:
            return  # No LR manager to configure

        # Estimate dataset size if not provided
        if dataset_size is None:
            try:
                if hasattr(train_loader.dataset, "__len__"):
                    dataset_size = len(train_loader.dataset)
                else:
                    # For streaming datasets, estimate based on first few batches
                    print("⏳ Estimating dataset size for streaming dataset...")
                    sample_batches = 10
                    total_samples = 0
                    i = 0

                    temp_iter = iter(train_loader)
                    for i in range(min(sample_batches, 50)):  # Don't sample too many
                        try:
                            batch = next(temp_iter)
                            if isinstance(batch, dict) and "input_ids" in batch:
                                total_samples += batch["input_ids"].size(0)
                        except StopIteration:
                            break

                    if total_samples > 0:
                        avg_batch_size = total_samples / min(sample_batches, i + 1)
                        # Estimate total dataset size (rough approximation)
                        dataset_size = int(
                            avg_batch_size * 1000
                        )  # Assume 1000 batches minimum
                        print(f"📊 Estimated dataset size: ~{dataset_size:,} samples")
                    else:
                        dataset_size = 100000  # Fallback
                        print(
                            f"⚠️  Could not estimate dataset size, using fallback: {dataset_size:,}"
                        )

            except Exception as e:
                dataset_size = 100000  # Fallback
                print(
                    f"⚠️  Error estimating dataset size: {e}, using fallback: {dataset_size:,}"
                )

        # Calculate total steps with actual dataset information
        total_steps = self.lr_manager.calculate_total_steps(
            num_epochs=self.training_params["num_epochs"],
            dataset_size=dataset_size,
            batch_size=self.training_params["batch_size"],
            gradient_accumulation_steps=self.training_params[
                "gradient_accumulation_steps"
            ],
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
        batch_idx: int,
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
                return self._train_step_impl(
                    input_ids, attention_mask, labels, optimizer, epoch, batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM errors with collective coordination if distributed
                    oom_info = {
                        "rank": (
                            getattr(self.distributed_manager, "rank", 0)
                            if self.distributed_manager
                            else 0
                        ),
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "batch_size": input_ids.size(0),
                        "memory_allocated": (
                            torch.cuda.memory_allocated()
                            if torch.cuda.is_available()
                            else 0
                        ),
                        "memory_reserved": (
                            torch.cuda.memory_reserved()
                            if torch.cuda.is_available()
                            else 0
                        ),
                        "error_message": str(e),
                    }

                    # Coordinate OOM handling across all ranks if distributed
                    if (
                        self.distributed_manager
                        and self.distributed_manager.is_initialized()
                    ):
                        print(
                            f"🔥 OOM detected on rank {oom_info['rank']} - coordinating with other ranks..."
                        )

                        # Broadcast OOM signal to all ranks
                        broadcast_success = (
                            self.distributed_manager.broadcast_oom_signal(oom_info)
                        )

                        if broadcast_success:
                            # Coordinate recovery action across ranks
                            recovery_success = (
                                self.distributed_manager.coordinate_oom_recovery(
                                    "reduce_batch_size"
                                )
                            )

                            if recovery_success:
                                print(
                                    f"✅ Collective OOM recovery coordinated across all ranks"
                                )
                            else:
                                print(
                                    f"❌ Failed to coordinate OOM recovery - falling back to local handling"
                                )

                    # Handle OOM with error handler and observability
                    handled = self.error_handler.handle_error(
                        e,
                        ErrorType.MEMORY,
                        ErrorSeverity.CRITICAL,
                        context=oom_info,
                        recoverable=True,
                    )

                    # Notify observability of OOM error
                    if self.observability:
                        self.observability.handle_out_of_memory(
                            self.step_count, oom_info
                        )

                    if not handled:
                        raise
                    # Return dummy results to continue training
                    return {
                        "loss": float("inf"),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "skipped": True,
                        "skip_reason": "memory_error",
                    }
                else:
                    # Handle other runtime errors
                    error_context = {"epoch": epoch, "batch_idx": batch_idx}
                    handled = self.error_handler.handle_error(
                        e,
                        ErrorType.COMPUTE,
                        ErrorSeverity.ERROR,
                        context=error_context,
                        recoverable=True,
                    )

                    # Notify observability of runtime error
                    if self.observability:
                        self.observability.handle_training_error(
                            e, self.step_count, error_context
                        )

                    if not handled:
                        raise
                    # Return dummy results
                    return {
                        "loss": float("inf"),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "skipped": True,
                        "skip_reason": "compute_error",
                    }
            except Exception as e:
                # Handle unexpected errors
                error_context = {"epoch": epoch, "batch_idx": batch_idx}
                handled = self.error_handler.handle_error(
                    e,
                    ErrorType.UNKNOWN,
                    ErrorSeverity.ERROR,
                    context=error_context,
                    recoverable=False,
                )

                # Notify observability of unexpected error
                if self.observability:
                    self.observability.handle_training_error(
                        e, self.step_count, error_context
                    )

                if not handled:
                    raise
                return {
                    "loss": float("inf"),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "skipped": True,
                    "skip_reason": "unknown_error",
                }
        else:
            # No error handler, proceed normally
            return self._train_step_impl(
                input_ids, attention_mask, labels, optimizer, epoch, batch_idx
            )

    def _train_step_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        batch_idx: int,
    ) -> Dict[str, Any]:
        """
        Internal implementation of training step.
        """
        # Track step time for it/s calculation
        step_start_time = time.time()

        # FIXED: Use micro_step_count for gradient accumulation calculations
        # optimizer_step_count will be incremented only when optimizer actually steps
        current_micro_step = self.micro_step_count

        # Start metrics collection
        if self.metrics_collector:
            self.metrics_collector.start_step(self.step_count, epoch, batch_idx)

        # Check memory health at start of step - estimate batch size from input
        current_batch_size = input_ids.size(0) if torch.is_tensor(input_ids) else 8

        # SPEED OPTIMIZATION: Only check memory health periodically (every 500 steps)
        # Checking every step causes massive overhead with synchronization and cleanup
        # CRITICAL FIX: Changed from 100 to 500 steps, removed early-step checks
        # Early steps have unstable memory and trigger false alarms
        should_check_memory = (
            self.optimizer_step_count % 500 == 0  # Check every 500 OPTIMIZER steps (not micro-steps)
        )

        if should_check_memory:
            memory_health = self.memory_monitor.check_memory_health(current_batch_size)
        else:
            # Use cached/lightweight check - just get current stats without full analysis
            memory_health = {
                'status': 'healthy',
                'action_needed': False,
                'recommended_batch_size': current_batch_size,
                'current_batch_size': current_batch_size,
                'gpu_utilization': 0.85,  # Assume reasonable utilization
                'available_gb': 2.0,  # Assume some available memory
                'allocated_gb': 9.0,  # Assume typical allocation
                'cached_gb': 10.0,  # Assume typical cached
                'oom_risk': 0.1,  # Low risk when not checking
            }

        # REMOVED: Dynamic gradient accumulation adjustment (fundamentally flawed)
        # The DataLoader batch size is fixed, so changing gradient_accumulation_steps
        # doesn't increase effective batch size - it just processes more batches before
        # stepping the optimizer, reducing training throughput.
        # Dynamic batch size adjustment should only be done by recreating the DataLoader,
        # which is complex and risky during training.

        # We keep dynamic_batch_sizer for monitoring purposes only
        if self.dynamic_batch_sizer is not None:
            # Monitor memory and track batch size history, but don't adjust
            _, _, _ = self.dynamic_batch_sizer.adjust_batch_size(
                self.step_count,
                memory_health
            )

        # Check collective memory health across all ranks if distributed
        collective_memory_health = None
        if self.distributed_manager and self.distributed_manager.is_initialized():
            # Periodic collective memory health checks (every 2000 steps to avoid overhead)
            # SPEED OPTIMIZATION: Reduced frequency from 200 to 2000 for additional speedup
            if batch_idx % 2000 == 0:
                collective_memory_health = (
                    self.distributed_manager.check_collective_memory_health()
                )

                if collective_memory_health["status"] in ["warning", "critical"]:
                    print(
                        f"⚠️  Collective memory health: {collective_memory_health['status']}"
                    )
                    print(
                        f"   Max utilization: {collective_memory_health['max_utilization']:.1%}"
                    )
                    print(
                        f"   Min available: {collective_memory_health['min_available_gb']:.1f}GB"
                    )
                    print(
                        f"   Problematic ranks: {len(collective_memory_health['problematic_ranks'])}/{collective_memory_health['total_ranks']}"
                    )

                    # If collective memory is critical, coordinate preventive action
                    if collective_memory_health["status"] == "critical":
                        print(
                            f"🚨 Critical collective memory situation - coordinating preventive action..."
                        )
                        self.distributed_manager.coordinate_oom_recovery(
                            "reduce_batch_size"
                        )

            # Periodic rank failure detection (every 5000 steps to avoid overhead)
            # SPEED OPTIMIZATION: Reduced frequency from 500 to 5000 for additional speedup
            if batch_idx % 5000 == 0:
                failure_status = self.distributed_manager.detect_rank_failures()

                if failure_status["status"] == "success":
                    if not failure_status["all_ranks_healthy"]:
                        failed_ranks = failure_status["failed_ranks"]
                        print(f"💥 Detected failed ranks: {failed_ranks}")
                        print(
                            f"   Healthy ranks: {len(failure_status['healthy_ranks'])}/{failure_status['total_ranks']}"
                        )

                        # Coordinate recovery for failed ranks
                        print(f"🔄 Coordinating recovery for failed ranks...")
                        recovery_success = (
                            self.distributed_manager.coordinate_rank_replacement(
                                failed_ranks, ""
                            )
                        )

                        if recovery_success:
                            print(
                                f"✅ Successfully coordinated recovery for failed ranks"
                            )

                            # Coordinate data resharding if dataloader supports it
                            if (
                                hasattr(self, "train_dataloader")
                                and self.train_dataloader  # type: ignore[attr-defined]
                            ):
                                from ...data.multi_column_data import (
                                    coordinate_data_resharding,
                                )

                                data_reshard_success = coordinate_data_resharding(
                                    self.train_dataloader, failed_ranks  # type: ignore[attr-defined]
                                )
                                if data_reshard_success:
                                    print(
                                        f"📊 Data resharding completed for failed ranks"
                                    )
                                else:
                                    print(
                                        f"⚠️  Data resharding failed - using existing distribution"
                                    )
                        else:
                            print(
                                f"❌ Failed to coordinate recovery - training may be unstable"
                            )
                elif failure_status["status"] == "gather_failed":
                    if failure_status["gather_time"] > 10.0:  # Very slow response
                        print(
                            f"⚠️  Very slow rank communication: {failure_status['gather_time']:.1f}s"
                        )

            # Periodic data distribution monitoring (every 200 steps to avoid overhead)
            if (
                batch_idx % 200 == 0
                and hasattr(self, "train_dataloader")
                and self.train_dataloader  # type: ignore[attr-defined]
            ):
                from ...data.multi_column_data import get_data_distribution_stats

                data_stats = get_data_distribution_stats(self.train_dataloader)  # type: ignore[attr-defined]

                if data_stats:
                    load_ratio = data_stats.get("load_ratio", 1.0)
                    samples_assigned = data_stats.get("samples_assigned", 0)

                    if abs(load_ratio - 1.0) > 0.1:  # More than 10% imbalance
                        print(f"⚖️  Data load imbalance detected:")
                        print(
                            f"   Load ratio: {load_ratio:.2f} (1.0 = perfectly balanced)"
                        )
                        print(f"   Samples assigned: {samples_assigned}")

                        if "resharded" in data_stats:
                            print(
                                f"   Resharded for failed ranks: {data_stats.get('failed_ranks', [])}"
                            )

        # Handle critical memory situations BEFORE forward pass
        if memory_health["status"] == "emergency":
            print(
                f"    EMERGENCY: Memory at {memory_health['gpu_utilization']:.1%}, "
                f"cleaning up aggressively"
            )
            cleanup_stats = self.memory_monitor.cleanup_memory(aggressive=True)
            print(f"    Freed {cleanup_stats['freed_gb']:.2f}GB")

            # Check if we should emergency stop
            if self.memory_monitor.should_emergency_stop():
                raise RuntimeError(
                    f"Emergency stop: Consistently high memory usage "
                    f"({memory_health['gpu_utilization']:.1%}). "
                    f"Reduce batch size or model size."
                )

        elif memory_health["status"] in ["critical", "warning"]:
            # Only print if not in silent mode
            if not self.memory_monitor.silent_mode:
                print(
                    f"    Memory {memory_health['status'].upper()}: "
                    f"{memory_health['gpu_utilization']:.1%} used, "
                    f"recommend batch_size={memory_health['recommended_batch_size']}"
                )

            # Perform standard cleanup
            if memory_health["status"] == "critical":
                cleanup_stats = self.memory_monitor.cleanup_memory(aggressive=False)
                if cleanup_stats["freed_gb"] > 0.1 and not self.memory_monitor.silent_mode:
                    print(f"    Freed {cleanup_stats['freed_gb']:.2f}GB GPU memory")

            # Dynamic batch size adjustment for LLM training
            # Check config to see if auto grad accumulation is enabled
            enable_auto_grad_accum = getattr(self.config.memory, 'enable_auto_grad_accumulation', False)
            if enable_auto_grad_accum:
                self._handle_memory_pressure(memory_health)

        # Validate device placement before forward pass
        base_model = self._get_base_model
        model_device = next(base_model.parameters()).device
        if input_ids.device != model_device:
            raise RuntimeError(
                f"Device mismatch: input on {input_ids.device}, model on {model_device}. "
                f"This usually indicates batch was moved to wrong device."
            )

        # Forward pass with timing
        start_time = time.time()

        # CRITICAL FIX: Use correct autocast dtype from config (BF16 vs FP16)
        autocast_dtype = torch.bfloat16 if getattr(self.config.training, "mixed_precision", "fp16") == "bf16" else torch.float16

        with autocast(device_type='cuda', enabled=True, dtype=autocast_dtype):
            # Use DeepSpeed engine for forward pass if available, otherwise use model directly
            if self.deepspeed_engine is not None:
                # Runtime check: Ensure engine_timers exists before forward pass
                if not hasattr(self.deepspeed_engine, 'engine_timers'):
                    from types import SimpleNamespace
                    self.deepspeed_engine.engine_timers = SimpleNamespace(
                        forward_timers=None,
                        backward_timers=None,
                        step_timers=None
                    )
                    self.deepspeed_engine._start_timers = lambda *args, **kwargs: None
                    self.deepspeed_engine._stop_timers = lambda *args, **kwargs: None

                # DeepSpeed wrapper returns (loss, outputs_dict) tuple
                result = self.deepspeed_engine(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                # Unpack tuple if using wrapper
                if isinstance(result, tuple) and len(result) == 2:
                    _, outputs = result  # Discard loss, we'll compute it ourselves
                else:
                    outputs = result
            else:
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

        # Ensure outputs is always a dict for consistent attribute access
        if not isinstance(outputs, dict):
            outputs = {"logits": outputs[0]} if isinstance(outputs, tuple) else {}

        forward_time = time.time() - start_time

        # Calculate losses - use DeepSeek loss if configured
        if self.deepseek_loss is not None:
            # CRITICAL FIX: When using DeepSeek loss, compute it from logits
            # The model's built-in loss is ignored to avoid double calculation
            # Extract hidden states and logits from model outputs
            hidden_states = outputs.get("hidden_states", None)
            if hidden_states is not None and isinstance(hidden_states, (list, tuple)):
                # Use the last layer's hidden states
                hidden_states = hidden_states[-1]

            logits = outputs.get("logits", None)
            if logits is None:
                raise RuntimeError("Model did not return logits - cannot compute DeepSeek loss")

            # Get MoE routing info if available
            # CRITICAL FIX: Model returns "gate_logits", not "router_logits"
            gate_logits = outputs.get("gate_logits", outputs.get("router_logits", None))
            expert_indices = outputs.get("expert_indices", None)

            # Compute DeepSeek-style loss (ignores model's built-in loss)
            loss_dict = self.deepseek_loss(
                logits=logits,
                targets=labels,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                gate_logits=gate_logits,
                expert_indices=expert_indices
            )

            # DeepSeek loss returns total_loss = main_loss + MTP + MoE balancing
            # This is the complete loss and should be used for backpropagation
            total_loss = loss_dict["total_loss"]
            main_loss = loss_dict.get("main_loss", total_loss)

            # Extract auxiliary losses for logging (already included in total_loss)
            aux_losses = {k: v for k, v in loss_dict.items() if k != "total_loss" and k != "main_loss"}

            # ENHANCED: Extract MTP-specific metrics for detailed monitoring
            # This provides visibility into whether MTP is contributing to training
            mtp_metrics = {}
            if 'mtp_mtp_loss' in loss_dict:
                mtp_metrics['mtp_loss'] = loss_dict['mtp_mtp_loss']
                mtp_metrics['mtp_weight'] = loss_dict.get('mtp_mtp_weight', 0.0)
                if 'mtp_per_token_losses' in loss_dict:
                    per_token = loss_dict['mtp_per_token_losses']
                    if isinstance(per_token, list) and len(per_token) > 0:
                        mtp_metrics['mtp_token1_loss'] = per_token[0] if len(per_token) > 0 else 0.0
                        mtp_metrics['mtp_token2_loss'] = per_token[1] if len(per_token) > 1 else 0.0
                        mtp_metrics['mtp_token3_loss'] = per_token[2] if len(per_token) > 2 else 0.0

            # Store MTP metrics separately for easy access in logging
            self.mtp_metrics = mtp_metrics

            # Store for logging
            self.valid_aux_losses = aux_losses
        else:
            # Use model's built-in loss calculation
            main_loss = outputs.get("loss")

            # Validate main loss before continuing
            if main_loss is None:
                raise RuntimeError("Model did not return a loss value")
            if not main_loss.requires_grad:
                raise RuntimeError(
                    "Loss does not require gradients - check model configuration"
                )

            total_loss = main_loss
            aux_losses = {}
            # Initialize for logging
            self.valid_aux_losses = {}
            self.mtp_metrics = {}  # No MTP metrics without DeepSeek loss

        # Check loss health BEFORE proceeding
        loss_health_result = self.loss_health.check_loss_health(
            main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss,
            self.step_count
        )

        if not loss_health_result["is_valid"]:
            print(f"     CRITICAL: {loss_health_result['reason']}")
            print(f"     Skipping optimizer step due to invalid loss")
            # Return early to skip this step instead of crashing
            return {
                "loss": float('inf'),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": 0.0,
                "grad_norm_pre_clip": 0.0,
                "clipped": False,
                "skipped": True,
                "skip_reason": loss_health_result['reason']
            }

        # Handle loss spikes - skip auxiliary losses if main loss is extremely high
        # FIXED: Raised threshold from 5.0 to 10.0 to allow auxiliary losses during early training
        # Early in training, main loss is naturally high (6-8), but auxiliary signals are crucial
        # IMPORTANT: When using DeepSeek loss, MTP is ALREADY INCLUDED in total_loss above
        # This flag only affects the additional composite losses (repetition penalties, etc.)
        skip_auxiliary_losses = main_loss.item() > 10.0 or loss_health_result["is_spike"]
        if skip_auxiliary_losses and batch_idx % 500 == 0:  # Reduced logging frequency (was 100)
            logger.debug(
                f"Skipping auxiliary losses due to high main loss: {main_loss.item():.4f}"
            )

        # Apply composite loss if available and not using DeepSeek loss
        if self.composite_loss and not skip_auxiliary_losses and self.deepseek_loss is None:
            # Extract logits from outputs (could be dict or object)
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
                router_logits = outputs.get("router_logits")
                expert_outputs = outputs.get("expert_outputs")
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else None
                router_logits = getattr(outputs, "router_logits", None)
                expert_outputs = getattr(outputs, "expert_outputs", None)

            aux_losses = self.composite_loss(
                inputs=input_ids,
                targets=labels,
                logits=logits,
                model=self.model,
                router_logits=router_logits,
                expert_outputs=expert_outputs,
            )

            # Track auxiliary loss moving averages for auto-scaling
            if not hasattr(self, "aux_loss_emas"):
                self.aux_loss_emas = {}

            # Validate each auxiliary loss BEFORE adding
            # Store as instance variable for logging later
            valid_aux_losses = {}
            for name, loss_value in aux_losses.items():
                if name == "total":
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
                    self.aux_loss_emas[name] = (
                        0.95 * self.aux_loss_emas[name] + 0.05 * current_loss_val
                    )

                # CRITICAL: Different handling for router/load-balance losses
                # Router loss is ESSENTIAL for MoE training to prevent expert collapse
                if "router" in name.lower() or "load_balance" in name.lower():
                    # CLARIFICATION: The base model (moe_model.py) does NOT compute router loss
                    # DeepSeek loss computes it via AuxiliaryFreeMoEBalancer using gradient balancing
                    # This gradient-based balancing is already included in DeepSeek's total_loss
                    # So we only track router losses for logging, not adding to total_loss again
                    valid_aux_losses[name] = loss_value
                    # DO NOT: total_loss = total_loss + loss_value  # Would double-count DeepSeek's balancing!
                else:
                    # Other auxiliary losses: apply conservative scaling
                    aux_ema = self.aux_loss_emas[name]
                    main_loss_val = main_loss.item()

                    # FIXED: Increased clamp from 0.1x to 0.5x main loss for stronger auxiliary signals
                    # Auxiliary losses need sufficient magnitude to influence training effectively
                    max_aux_loss = main_loss_val * 0.5
                    loss_value = torch.clamp(loss_value, max=max_aux_loss)

                    # Additional scaling by EMA ratio to normalize contribution
                    if aux_ema > 0 and main_loss_val > 0:
                        scaling_factor = min(
                            1.0, main_loss_val / (aux_ema * 10)
                        )  # Conservative scaling
                        loss_value = loss_value * scaling_factor

                    valid_aux_losses[name] = loss_value
                    total_loss = total_loss + loss_value

            # Add repetition penalties if enabled
            # CRITICAL FIX: Use logits WITH GRADIENTS for diversity penalty
            # The NGramRepetitionPenalty has two components:
            # 1. N-gram penalty: counts repetitions in token IDs (no gradients needed)
            # 2. Diversity penalty: operates on logits WITH GRADIENTS (this provides backprop signal!)
            if not skip_auxiliary_losses:
                if hasattr(self, 'ngram_penalty') and self.ngram_penalty is not None:
                    # NGramRepetitionPenalty.forward(logits, targets, mask)
                    # - logits: for diversity penalty (WITH GRADIENTS) ✓
                    # - targets: for n-gram counting (uses input_ids for context)
                    # The diversity penalty on logits provides the gradient signal!
                    ngram_penalties = self.ngram_penalty(logits, input_ids, attention_mask)

                    if 'total_repetition_penalty' in ngram_penalties:
                        repetition_penalty = ngram_penalties['total_repetition_penalty']
                        if not torch.isnan(repetition_penalty) and not torch.isinf(repetition_penalty):
                            # Scale penalty by config weight if available
                            penalty_weight = getattr(self.config.training, 'repetition_penalty_weight', 0.5)
                            scaled_penalty = repetition_penalty * penalty_weight
                            valid_aux_losses['ngram_repetition'] = scaled_penalty
                            total_loss = total_loss + scaled_penalty

                if hasattr(self, 'repetition_detector') and self.repetition_detector is not None:
                    # SequenceRepetitionDetector operates on token IDs (no gradients needed for counting)
                    # Use input_ids to detect repetition patterns in the context
                    immediate_penalties = self.repetition_detector(input_ids, attention_mask)

                    if 'immediate_repetition_penalty' in immediate_penalties:
                        immediate_penalty = immediate_penalties['immediate_repetition_penalty']
                        if not torch.isnan(immediate_penalty) and not torch.isinf(immediate_penalty):
                            # Scale penalty by config weight if available
                            immediate_weight = getattr(self.config.training, 'immediate_repetition_weight', 1.0)
                            scaled_immediate = immediate_penalty * immediate_weight
                            valid_aux_losses['immediate_repetition'] = scaled_immediate
                            total_loss = total_loss + scaled_immediate

            # Store valid_aux_losses for logging
            self.valid_aux_losses = valid_aux_losses

            # Log individual loss components for monitoring
            if batch_idx % 1000 == 0 and valid_aux_losses:
                print(f"    Main loss: {main_loss.item():.6f}")
                for name, loss_value in valid_aux_losses.items():
                    ema_val = self.aux_loss_emas.get(name, 0.0)
                    print(
                        f"    {name} loss: {loss_value.item():.6f} (EMA: {ema_val:.6f})"
                    )
                print(f"    Total loss: {total_loss.item():.6f}")

            # FIXED: Check for loss validity after adding auxiliary losses - skip step instead of crash
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"     Invalid total loss detected! Main: {main_loss.item():.6f}")
                for name, loss_value in valid_aux_losses.items():
                    print(f"      {name}: {loss_value.item():.6f}")
                print(f"      Total: {total_loss.item():.6f}")
                print(f"     Skipping optimizer step due to invalid loss")

                # Reset auxiliary loss EMAs if they might be corrupted
                self.aux_loss_emas.clear()

                # Return early to skip this step instead of crashing
                return {
                    "loss": float('inf'),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": 0.0,
                    "grad_norm_pre_clip": 0.0,
                    "clipped": False,
                    "skipped": True,
                    "skip_reason": "invalid_loss"
                }

        # Backward pass with timing
        backward_start = time.time()

        # CRITICAL FIX: Define gradient_accumulation_steps and is_accumulation_complete early
        # so they're available in all code paths (DeepSpeed and standard training)
        gradient_accumulation_steps: int = getattr(
            self.config.training, "gradient_accumulation_steps", 1
        )
        is_accumulation_complete: bool = ((current_micro_step + 1) % gradient_accumulation_steps) == 0

        # Handle DeepSpeed vs standard training
        if self.deepspeed_engine:
            # FIXED: DeepSpeed handles everything internally including gradient accumulation
            # DeepSpeed will automatically accumulate gradients over multiple backward() calls
            # and only step() the optimizer when gradient_accumulation_steps is reached.
            # This is configured in the DeepSpeed config (see _create_deepspeed_config).

            # DeepSpeed training - handles gradient accumulation internally
            self.deepspeed_engine.backward(total_loss)
            self.deepspeed_engine.step()

            # Extract gradient norm from DeepSpeed with comprehensive fallbacks
            grad_norm = None
            grad_norm_pre_clip = None  # Not available for DeepSpeed

            try:
                # Method 1: Try DeepSpeed's built-in gradient norm tracking
                if hasattr(self.deepspeed_engine, "get_global_grad_norm"):
                    grad_norm = self.deepspeed_engine.get_global_grad_norm()
                    if grad_norm is not None and torch.is_tensor(grad_norm):
                        grad_norm = grad_norm.item()  # type: ignore[union-attr]

                # Method 2: Try optimizer-specific gradient norm
                elif hasattr(self.deepspeed_engine, "optimizer") and hasattr(
                    self.deepspeed_engine.optimizer, "get_global_grad_norm"
                ):
                    grad_norm = self.deepspeed_engine.optimizer.get_global_grad_norm()
                    if grad_norm is not None and torch.is_tensor(grad_norm):
                        grad_norm = grad_norm.item()  # type: ignore[union-attr]

                # Method 3: Check if optimizer has gradient clipping info
                elif hasattr(self.deepspeed_engine, "optimizer"):
                    optimizer = self.deepspeed_engine.optimizer

                    # FP16 optimizer might have gradient scale info
                    if hasattr(optimizer, "cur_scale"):
                        # We can't get exact grad norm but can detect scaling issues
                        if (
                            hasattr(optimizer, "last_overflow_time")
                            and optimizer.last_overflow_time == optimizer.step_count  # type: ignore[attr-defined]
                        ):
                            grad_norm = float("inf")  # Overflow detected

                # Method 4: DeepSpeed internal tracking
                elif hasattr(self.deepspeed_engine, "_global_grad_norm"):
                    grad_norm = self.deepspeed_engine._global_grad_norm
                    if torch.is_tensor(grad_norm):
                        grad_norm = grad_norm.item()  # type: ignore[union-attr]

                # Fallback: Default to 0 if no method worked
                if grad_norm is None:
                    grad_norm = 0.0

                # Ensure grad_norm is a float
                if torch.is_tensor(grad_norm):
                    grad_norm = grad_norm.item()  # type: ignore[union-attr]
                elif grad_norm is None:
                    grad_norm = 0.0

                # Update our gradient health monitor with DeepSpeed results
                # Note: We can't get pre-clip norms with DeepSpeed, so we track post-clip only
                if hasattr(self, "gradient_health") and self.gradient_health is not None and grad_norm is not None:
                    # Manually update the history since DeepSpeed handled clipping
                    self.gradient_health.grad_norm_history.append(grad_norm)
                    # For DeepSpeed, pre_clip == post_clip (we can't separate them)
                    self.gradient_health.grad_norm_pre_clip_history.append(grad_norm)
                    self.gradient_health.total_steps += 1

                    # Check for explosions in gradient norm
                    if grad_norm > self.gradient_health.explosion_threshold:
                        self.gradient_health.recent_explosions.append(self.step_count)
                        self.gradient_health.total_explosions += 1
                        print(
                            f"    DeepSpeed gradient explosion detected: {grad_norm:.2f}"
                        )

                # Log gradient information periodically
                # SPEED OPTIMIZATION: Reduced frequency from 1000 to 2000
                if self.step_count % 2000 == 0 and grad_norm is not None:
                    print(
                        f"    DeepSpeed gradients: norm={grad_norm:.3f}"
                    )

            except Exception as e:
                # Comprehensive fallback
                print(f"    Warning: DeepSpeed gradient norm extraction failed: {e}")
                grad_norm = 0.0
                grad_norm_pre_clip = None
        else:
            # Standard training with mixed precision support
            # NOTE: gradient_accumulation_steps and is_accumulation_complete already defined above

            # FIXED: Proper gradient accumulation logic
            # Accumulate gradients over multiple steps, then step optimizer
            # Zero gradients AFTER optimizer step (not before!)
            # Optimizer steps at END of accumulation cycle (step N-1, 2N-1, 3N-1, ...)

            # NOTE: We zero gradients AFTER optimizer.step() below, not here!
            # This fixes the critical bug where gradients were cleared before stepping.

            # FIXED: Add gradient sync control for distributed training
            # Only synchronize gradients on the last accumulation step
            should_sync_grads = is_accumulation_complete

            # Check if model is wrapped in DDP
            is_ddp = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)

            # FIXED: Use no_sync context manager for gradient accumulation in DDP
            # This prevents all_reduce on every backward, only syncing when accumulation completes
            from contextlib import nullcontext

            # Safety check: In DDP, all ranks must agree on sync timing
            # This is automatically handled by is_accumulation_complete since all ranks
            # process the same number of batches per epoch (drop_last=True in DataLoader)
            sync_context = nullcontext() if (not is_ddp or should_sync_grads) else self.model.no_sync()  # type: ignore[attr-defined]

            with sync_context:
                # CRITICAL FIX: Scale loss by gradient accumulation steps
                # This ensures accumulated gradients have correct magnitude
                scaled_loss = total_loss / gradient_accumulation_steps

                if self.gradient_surgeon and self.config.multi_task:
                    # Apply gradient surgery with scaler support
                    task_losses = {"main": scaled_loss}  # Could have multiple tasks
                    self._apply_gradient_surgery(task_losses, optimizer)
                else:
                    # Standard backward pass with mixed precision
                    if self.scaler is not None:
                        # Scale loss and backward for mixed precision
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

            # Only unscale gradients and check health on the last accumulation step (before optimizer.step())
            # With gradient accumulation, we accumulate multiple backward() calls, then step once
            if is_accumulation_complete:
                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)

                # ULTRA-OPTIMIZED: Only check gradient health if enabled
                if self.gradient_health_enabled and self.gradient_health is not None:
                    ultra_fast = getattr(self.config, 'performance', None) and getattr(self.config.performance, 'ultra_fast_mode', False)
                    should_check = not ultra_fast or self.step_count % 10 == 0 or self.step_count < 100

                    if should_check:
                        base_model = self._get_base_model
                        grad_health_result = self.gradient_health.check_gradient_health(
                            base_model,
                            self.step_count,
                            compute_histogram=(self.step_count % 5000 == 0),
                        )
                    else:
                        # Skip check - use minimal result
                        grad_health_result = {
                            "should_skip": False,
                            "should_reduce_lr": False,
                            "grad_norm": 0.0,
                            "grad_norm_pre_clip": 0.0,
                            "is_explosion": False,
                            "clip_value": self.gradient_health.get_clip_value(self.step_count),
                            "recent_explosions": 0,
                        }
                else:
                    # Gradient health disabled - use standard clipping only
                    grad_health_result = {
                        "should_skip": False,
                        "should_reduce_lr": False,
                        "grad_norm": 0.0,
                        "grad_norm_pre_clip": 0.0,
                        "is_explosion": False,
                        "clip_value": self.config.training.max_gradient_norm if hasattr(self.config.training, 'max_gradient_norm') else 1.0,
                        "recent_explosions": 0,
                    }
            else:
                # Not the last accumulation step - skip gradient checking
                # Just continue accumulating gradients
                grad_health_result = {
                    "should_skip": False,
                    "should_reduce_lr": False,
                    "grad_norm": 0.0,
                    "grad_norm_pre_clip": 0.0,
                    "is_explosion": False,
                    "clip_value": 1.0,
                    "recent_explosions": 0,
                }

            # Handle critical gradient explosion - skip step
            if grad_health_result["should_skip"]:
                print(
                    f"    CRITICAL: Gradient explosion detected: {grad_health_result['grad_norm_pre_clip']:.2f}"
                )
                print(f"    Skipping optimizer step to prevent NaN corruption")
                optimizer.zero_grad(set_to_none=True)  # Clear corrupted gradients
                if self.scaler is not None:
                    self.scaler.update()  # Update scaler even when skipping
                return {
                    "loss": total_loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_health_result["grad_norm_pre_clip"],
                    "gradient_norm_pre_clip": grad_health_result["grad_norm_pre_clip"],
                    "skipped": True,
                    "skip_reason": "gradient_explosion",
                    **grad_health_result,
                }

            # Check if we should reduce learning rate due to repeated explosions
            if grad_health_result["should_reduce_lr"]:
                current_lr = optimizer.param_groups[0]["lr"]

                # Be less aggressive with LR reduction to prevent learning from stalling
                reduction_factor = 0.8  # Was 0.5, now more conservative
                min_lr = 1e-7  # Prevent LR from going too low

                new_lr = max(current_lr * reduction_factor, min_lr)

                # Only reduce if we're not already at minimum
                if current_lr > min_lr:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                    print(
                        f"    Reducing learning rate due to gradient instability: {current_lr:.2e} -> {new_lr:.2e}"
                    )
                    # Reset explosion counter after LR reduction (if enabled)
                    if self.gradient_health_enabled and self.gradient_health is not None:
                        self.gradient_health.reset_explosion_counter()
                else:
                    print(f"    Learning rate already at minimum ({min_lr:.2e}), trying checkpoint restore instead")
                    # Try checkpoint restore if we have persistent issues at minimum LR
                    if hasattr(self, '_attempt_checkpoint_restore'):
                        self._attempt_checkpoint_restore("Persistent gradient explosions at minimum LR")
                        return {"loss": total_loss.item(), "learning_rate": current_lr, "restored_checkpoint": True}

            # Apply gradient clipping (adaptive if gradient_health enabled, otherwise standard)
            base_model = self._get_base_model
            if self.gradient_health_enabled and self.gradient_health is not None:
                grad_norm_pre_clip_clipped, grad_norm = self.gradient_health.clip_gradients(base_model, self.step_count)
                grad_norm_pre_clip = grad_health_result["grad_norm_pre_clip"]
            else:
                # Standard gradient clipping when gradient_health is disabled
                max_grad_norm = grad_health_result["clip_value"]
                grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_grad_norm)
                grad_norm_pre_clip = grad_norm
                grad_norm_pre_clip_clipped = grad_norm

            # Log gradient health information
            # SPEED OPTIMIZATION: Reduced frequency from 1000 to 2000
            if grad_health_result["is_explosion"] or self.step_count % 2000 == 0:
                print(
                    f"    Gradient health: norm={grad_norm_pre_clip:.3f}, "
                    f"clip_value={grad_health_result['clip_value']:.1f}, "
                    f"explosions={grad_health_result['recent_explosions']}"
                )

            # Emergency stop check (only if gradient_health enabled)
            if self.gradient_health_enabled and self.gradient_health is not None and self.gradient_health.should_emergency_stop():
                raise RuntimeError(
                    f"Emergency stop: Too many gradient explosions "
                    f"({grad_health_result['recent_explosions']} recent). "
                    f"Training is unstable."
                )

            # Optimizer step with scaler support and health monitoring
            # CRITICAL FIX: Only step optimizer when gradient accumulation is complete
            # (is_accumulation_complete was already calculated above, reuse it)

            if is_accumulation_complete:
                if self.scaler is not None:
                    # Monitor scaler health before step
                    scaler_scale = self.scaler.get_scale()
                    scaler_state = {
                        "scale": scaler_scale,
                        "growth_factor": self.scaler.get_growth_factor(),
                        "backoff_factor": self.scaler.get_backoff_factor(),
                        "growth_interval": self.scaler.get_growth_interval(),
                    }

                    # Check for scaler issues
                    if scaler_scale < 1.0 or scaler_scale > 2**16:
                        print(f"    Warning: Scaler scale unusual: {scaler_scale}")

                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # Periodic scaler reset to prevent error accumulation
                    if (
                        self.step_count - self.scaler_last_reset
                    ) >= self.scaler_reset_interval:
                        print(
                            f"    Resetting mixed precision scaler (step {self.step_count})"
                        )
                        # Save current scale for continuity
                        current_scale = self.scaler.get_scale()
                        self.scaler = GradScaler(
                            init_scale=min(current_scale, 2**15),  # Cap at reasonable value
                            growth_factor=2.0,
                            backoff_factor=0.5,
                            growth_interval=2000,
                        )
                        self.scaler_last_reset = self.step_count

                    # Log scaler state periodically
                    if self.step_count % 1000 == 0:
                        print(
                            f"    Scaler state: scale={scaler_scale:.1f}, "
                            f"growth_factor={scaler_state['growth_factor']}"
                        )
                else:
                    optimizer.step()

                # CRITICAL FIX: Zero gradients AFTER optimizer step
                # This ensures gradients from accumulation cycle are used before clearing
                optimizer.zero_grad(set_to_none=True)
            else:
                # Accumulating gradients, skip optimizer step
                grad_norm = 0.0
                grad_norm_pre_clip = 0.0

        backward_time = time.time() - backward_start
        opt_time = backward_time  # Combined for DeepSpeed

        # FIXED: Intelligent learning rate management - only step on optimizer updates
        if self.deepspeed_engine:
            current_lr = (
                self.deepspeed_engine.get_lr()[0]
                if self.deepspeed_engine.get_lr()
                else 0.0
            )
            warmup_info = {"warmup_step": self.optimizer_step_count, "deepspeed_managed": True}
            lr_info = {"lr_step": self.optimizer_step_count, "deepspeed_managed": True}
        else:
            # Only update LR when we actually step the optimizer
            if is_accumulation_complete:
                # Use adaptive LR manager if available, otherwise fall back to intelligent LR manager
                if hasattr(self, "adaptive_lr_manager") and self.adaptive_lr_manager is not None:
                    # FIXED: Adaptive LR manager - use optimizer_step_count (will be incremented after this)
                    lr_step_info = self.adaptive_lr_manager.step(total_loss.item(), self.optimizer_step_count)

                elif hasattr(self, "lr_manager") and self.lr_manager is not None:
                    # FIXED: Call lr_manager.step() only on optimizer steps with optimizer_step_count
                    # This ensures the LR schedule progresses at the correct rate
                    lr_step_info = self.lr_manager.step(
                        None, training_step=self.optimizer_step_count
                    )
                else:
                    # No LR manager
                    lr_step_info = {}

                # This is an optimizer step - create detailed info
                warmup_info = {
                    "warmup_step": self.micro_step_count,
                    "optimizer_step": self.optimizer_step_count,
                    "phase": lr_step_info.get("phase", "unknown"),
                    "plateau_patience": lr_step_info.get("plateau_patience", 0),
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "gradient_accumulation_aware": True,
                }

                lr_info = {
                    "lr_step": self.micro_step_count,
                    "optimizer_step": self.optimizer_step_count,
                    "intelligent_lr": True,
                    "lr_reduced": lr_step_info.get("lr_reduced", False),
                    "plateau_detected": lr_step_info.get("plateau_detected", False),
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                }

                # Log significant LR events (but skip warmup spam)
                adjustment_type = lr_step_info.get("adjustment_type", "")
                if lr_step_info.get("lr_reduced", False) or lr_step_info.get("lr_adjusted", False):
                    # Only log if NOT warmup (warmup spams every step)
                    if adjustment_type != "warmup":
                        new_lr_val = lr_step_info.get('new_lr') or lr_step_info.get('lr') or lr_step_info.get('current_lr') or 0.0
                        print(
                            f"    🔽 LR adjusted at optimizer step {self.optimizer_step_count} (micro-step {self.micro_step_count})"
                        )
                        print(f"        New LR: {new_lr_val:.2e}")

                # Log LR schedule progress periodically
                if self.optimizer_step_count % 100 == 0:
                    phase = lr_step_info.get("phase", "unknown")
                    current_lr_val = lr_step_info.get('lr') or lr_step_info.get('current_lr') or lr_step_info.get('new_lr') or 0.0
                    print(
                        f"    📈 LR Schedule: optimizer_step={self.optimizer_step_count}, phase={phase}, "
                        f"lr={current_lr_val:.2e}, gradient_accum={gradient_accumulation_steps}"
                    )
            else:
                # Not an optimizer step, just accumulating gradients
                accumulation_step = (
                    self.micro_step_count % gradient_accumulation_steps
                ) + 1
                warmup_info = {
                    "warmup_step": self.micro_step_count,
                    "optimizer_step": self.optimizer_step_count,
                    "accumulation_step": accumulation_step,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "accumulating": True,
                }
                lr_info = {
                    "lr_step": self.micro_step_count,
                    "optimizer_step": self.optimizer_step_count,
                    "accumulation_step": accumulation_step,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "accumulating": True,
                }

            current_lr = optimizer.param_groups[0]["lr"]

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
                grad_norm=grad_norm.item() if (grad_norm is not None and torch.is_tensor(grad_norm)) else (grad_norm if grad_norm is not None else None),  # type: ignore[union-attr]
                forward_time=forward_time,
                backward_time=backward_time,
                optimizer_time=opt_time,
                **grad_metrics,
            )
            step_metrics = {
                "batch_time": step_info.batch_time,
                "memory_allocated": step_info.memory_allocated,
                "memory_cached": step_info.memory_cached,
            }

        # Log metrics asynchronously - reduced frequency for performance
        # Only log detailed metrics every 100 steps instead of every step
        if self.async_logger and self.step_count % 100 == 0:
            # Calculate iterations per second
            batch_time = step_metrics.get("batch_time", 1.0)
            it_per_sec = (
                1.0 / batch_time
                if batch_time is not None and batch_time > 0
                else 0.0
            )

            metrics = {
                "train/loss": total_loss.item(),
                "train/main_loss": main_loss.item(),
                "train/learning_rate": current_lr,
                "train/grad_norm": (grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm) if grad_norm is not None else 0.0,
                "train/grad_norm_pre_clip": (
                    grad_norm_pre_clip.item() if torch.is_tensor(grad_norm_pre_clip) else grad_norm_pre_clip if grad_norm_pre_clip is not None else 0.0  # type: ignore[union-attr]
                ),
                "train/it_per_sec": it_per_sec,
                # Add batch time and memory metrics with train/ prefix for consistent grouping
                "train/batch_time": step_metrics.get("batch_time", 0.0),
                "train/memory_allocated_gb": step_metrics.get("memory_allocated", 0.0),
                "train/memory_cached_gb": step_metrics.get("memory_cached", 0.0),
            }

            # ENHANCED: Add comprehensive loss component breakdown for debugging
            # This helps identify which loss components are contributing and by how much
            total_loss_val = total_loss.item()
            main_loss_val = main_loss.item()

            # Calculate percentage of total loss from main vs auxiliary
            if total_loss_val > 0:
                metrics["train/loss_components/main_loss_pct"] = (main_loss_val / total_loss_val) * 100.0
                aux_total = total_loss_val - main_loss_val
                if aux_total > 0:
                    metrics["train/loss_components/aux_loss_total_pct"] = (aux_total / total_loss_val) * 100.0

            # Add all auxiliary loss components with detailed breakdown
            if hasattr(self, 'valid_aux_losses') and self.valid_aux_losses:
                for name, loss_value in self.valid_aux_losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        loss_val = loss_value.item()
                        metrics[f"train/loss_components/{name}"] = loss_val
                        if total_loss_val > 0:
                            metrics[f"train/loss_components/{name}_pct"] = (loss_val / total_loss_val) * 100.0

            # Add DeepSeek loss components if available (MTP, MoE balancing, etc.)
            # These are stored in valid_aux_losses when using DeepSeek loss
            if (
                hasattr(self, 'deepseek_loss') and self.deepseek_loss is not None
                and hasattr(self, 'valid_aux_losses') and self.valid_aux_losses
            ):
                for name, loss_value in self.valid_aux_losses.items():
                    if isinstance(loss_value, torch.Tensor) and not torch.isnan(loss_value):
                        loss_val = loss_value.item()
                        metrics[f"train/deepseek/{name}"] = loss_val
                        if total_loss_val > 0:
                            metrics[f"train/deepseek/{name}_pct"] = (loss_val / total_loss_val) * 100.0

            # ENHANCED: Add MTP-specific metrics for detailed monitoring
            # This provides visibility into Multi-Token Prediction performance
            if hasattr(self, 'mtp_metrics') and self.mtp_metrics:
                for name, value in self.mtp_metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()

                    # Add MTP metrics under train/mtp/ namespace
                    metrics[f"train/mtp/{name}"] = value

                    # Add percentage contribution for loss values
                    if 'loss' in name and total_loss_val > 0 and isinstance(value, (int, float)):
                        metrics[f"train/mtp/{name}_pct"] = (value / total_loss_val) * 100.0

            # Add gradient health statistics with custom step metric
            if self.gradient_health_enabled and hasattr(self, "gradient_health") and self.gradient_health is not None:
                grad_health_stats = self.gradient_health.get_health_summary()
                metrics.update({
                    "gradient_step": self.step_count,
                    "gradient/explosion_rate": grad_health_stats.get("explosion_rate", 0.0),
                    "gradient/total_explosions": grad_health_stats.get("total_explosions", 0),
                    "gradient/mean_norm": grad_health_stats.get("mean_grad_norm", 0.0),
                    "gradient/std_norm": grad_health_stats.get("std_grad_norm", 0.0),
                    "gradient/max_norm": grad_health_stats.get("max_grad_norm", 0.0),
                    "gradient/min_norm": grad_health_stats.get("min_grad_norm", 0.0),
                    "gradient/clip_value": self.gradient_health.get_clip_value(self.step_count),
                })

            # Add MoE-specific metrics if available - only log expensive per-expert stats every 500 steps
            # Handle both dict and object outputs
            if isinstance(outputs, dict):
                router_logits = outputs.get('router_logits')
            else:
                router_logits = outputs.router_logits if hasattr(outputs, 'router_logits') else None

            if router_logits is not None:
                if isinstance(router_logits, (list, tuple)):
                    router_logits = router_logits[0] if len(router_logits) > 0 else None

                if router_logits is not None and torch.is_tensor(router_logits):
                    # Calculate expert selection statistics
                    expert_probs = torch.softmax(router_logits, dim=-1)
                    expert_selections = torch.argmax(expert_probs, dim=-1)

                    # Expert usage distribution
                    num_experts = expert_probs.size(-1)
                    expert_counts = torch.zeros(num_experts, device=router_logits.device)
                    for i in range(num_experts):
                        expert_counts[i] = (expert_selections == i).sum().float()

                    total_selections = expert_counts.sum()
                    if total_selections > 0:
                        expert_usage = expert_counts / total_selections

                        # FIXED: Only log per-expert usage every 2000 steps (expensive operation)
                        # Reduced from 500 to 2000 to minimize training overhead
                        if self.step_count % 2000 == 0:
                            # Log per-expert usage (all experts, not capped)
                            for i in range(num_experts):
                                metrics[f"train/moe/expert_{i}_usage"] = expert_usage[i].item()

                            # Also log as percentage for better readability
                            for i in range(num_experts):
                                metrics[f"train/moe/expert_{i}_usage_pct"] = expert_usage[i].item() * 100.0

                        # Log aggregate MoE metrics (lightweight)
                        metrics.update({
                            "train/moe/expert_entropy": -(expert_usage * torch.log(expert_usage + 1e-10)).sum().item(),
                            "train/moe/expert_max_usage": expert_usage.max().item(),
                            "train/moe/expert_min_usage": expert_usage.min().item(),
                            "train/moe/expert_usage_std": expert_usage.std().item(),
                            "train/moe/expert_balance": 1.0 - (expert_usage.max() - expert_usage.min()).item(),
                            "train/moe/num_experts": float(num_experts),
                        })

                    # Router confidence metrics
                    max_probs = expert_probs.max(dim=-1)[0]
                    metrics.update({
                        "train/moe/router_confidence_mean": max_probs.mean().item(),
                        "train/moe/router_confidence_std": max_probs.std().item(),
                        "train/moe/router_confidence_min": max_probs.min().item(),
                        "train/moe/router_confidence_max": max_probs.max().item(),
                    })

            # Add expert indices statistics if available
            # Handle both dict and object outputs
            if isinstance(outputs, dict):
                expert_indices = outputs.get('expert_indices')
            else:
                expert_indices = outputs.expert_indices if hasattr(outputs, 'expert_indices') else None

            if expert_indices is not None:
                if torch.is_tensor(expert_indices):
                    unique_experts = torch.unique(expert_indices).numel()
                    metrics["train/moe/active_experts"] = float(unique_experts)

            # Add dynamic batching metrics if enabled
            if self.dynamic_batch_sizer is not None:
                batch_stats = self.dynamic_batch_sizer.get_statistics()
                metrics.update({
                    "train/batch_size": batch_stats['current_batch_size'],
                    "train/dynamic_batch/avg_batch_size": batch_stats['avg_batch_size'],
                    "train/dynamic_batch/min_batch_size": batch_stats['min_batch_size'],
                    "train/dynamic_batch/max_batch_size": batch_stats['max_batch_size'],
                    "train/dynamic_batch/total_adjustments": batch_stats['total_adjustments'],
                    "train/dynamic_batch/increases": batch_stats['increases'],
                    "train/dynamic_batch/decreases": batch_stats['decreases'],
                    "train/dynamic_batch/adjustment_rate": batch_stats['adjustment_rate'],
                })

                # Add GPU memory utilization for context
                if memory_health:
                    metrics["train/dynamic_batch/gpu_utilization"] = memory_health.get("gpu_utilization", 0.0)

            # Add gradient metrics with train/ prefix
            for key, value in grad_metrics.items():
                metrics[f"train/{key}"] = value

            # Add individual auxiliary loss components
            for name, loss_value in aux_losses.items():
                if name != "total":
                    # Handle both tensor and scalar values
                    if isinstance(loss_value, torch.Tensor):
                        metrics[f"train/aux_{name}"] = loss_value.item()
                    elif isinstance(loss_value, (int, float)):
                        metrics[f"train/aux_{name}"] = float(loss_value)
                    elif isinstance(loss_value, list):
                        # For per-token losses from MTP
                        for i, val in enumerate(loss_value):
                            metrics[f"train/aux_{name}_token{i+1}"] = float(val)

            self.async_logger.log_metrics(metrics, self.step_count)

        # Update observability with training step information (Phase 7)
        if self.observability:
            try:
                # Get memory and GPU metrics
                gpu_memory_used = (
                    torch.cuda.memory_allocated() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                gpu_memory_reserved = (
                    torch.cuda.memory_reserved() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                gpu_utilization = memory_health.get("gpu_utilization", 0.0)

                # Calculate throughput metrics
                total_time = forward_time + backward_time + opt_time
                samples_per_second = (
                    current_batch_size / total_time if total_time > 0 else 0.0
                )

                # Update observability with comprehensive metrics
                self.observability.update_training_step(
                    step=self.step_count,
                    epoch=epoch,
                    loss=total_loss.item(),
                    learning_rate=current_lr,
                    batch_size=current_batch_size,
                    sequence_length=(
                        input_ids.size(1) if torch.is_tensor(input_ids) else 0
                    ),
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
                    memory_status=memory_health.get("status", "unknown"),
                    memory_oom_risk=memory_health.get("oom_risk", 0.0),
                    # Gradient metrics
                    gradient_norm=(
                        grad_norm.item()  # type: ignore[union-attr]
                        if grad_norm is not None and hasattr(grad_norm, "item")
                        else (grad_norm if grad_norm is not None else 0.0)
                    ),
                    gradient_norm_pre_clip=(
                        grad_norm_pre_clip.item()  # type: ignore[union-attr]
                        if grad_norm_pre_clip is not None
                        and hasattr(grad_norm_pre_clip, "item")
                        else (
                            grad_norm_pre_clip
                            if grad_norm_pre_clip is not None
                            else 0.0
                        )
                    ),
                    # Auxiliary loss information
                    main_loss=main_loss.item(),
                    aux_losses={
                        name: loss.item()
                        for name, loss in aux_losses.items()
                        if isinstance(loss, torch.Tensor)
                    },
                    # Training mode information
                    deepspeed_enabled=self.deepspeed_engine is not None,
                    distributed_enabled=self.is_distributed,
                    mixed_precision=self.scaler is not None,
                )
            except Exception as e:
                # Don't let observability errors break training
                print(f"    Warning: Observability update failed: {e}")

        # Update training state (step_count already incremented at beginning)
        loss_val = total_loss.item()
        if loss_val < self.best_loss:
            self.best_loss = loss_val

        # Intelligent memory management - replace basic cleanup
        memory_cleanup_needed = False

        # SPEED OPTIMIZATION: Only cleanup on true emergencies or very rare periodic checks
        # Use config.clear_cache_frequency if available, otherwise default to 10000
        clear_cache_freq = getattr(self.config.memory, 'clear_cache_frequency', 10000)

        if self.step_count % clear_cache_freq == 0:  # Regular cleanup interval (from config)
            memory_cleanup_needed = True
        # CRITICAL FIX: Only check emergency/OOM risk when we actually checked memory health
        # Otherwise cached values can trigger false alarms
        elif should_check_memory:
            if memory_health.get("status") == "emergency":  # ONLY emergency (99.5%+)
                memory_cleanup_needed = True
            elif memory_health.get("oom_risk", 0.0) > 0.95:  # Only extreme OOM risk (was 0.9)
                memory_cleanup_needed = True

        if memory_cleanup_needed and torch.cuda.is_available():
            cleanup_aggressive = memory_health.get("status") == "emergency"
            cleanup_stats = self.memory_monitor.cleanup_memory(
                aggressive=cleanup_aggressive
            )

            # Log significant cleanup
            if cleanup_stats["freed_gb"] > 0.1:
                print(f"    Periodic cleanup freed {cleanup_stats['freed_gb']:.2f}GB")

        # SPEED OPTIMIZATION: Only update memory history on actual checks (not every step)
        # This was being called EVERY step even when we skip the health check
        if should_check_memory:
            self.memory_monitor.update_memory_history(current_batch_size)

        # Record training metrics for distributed health monitoring
        if self.health_checker:
            self.health_checker.record_training_metrics(
                loss=total_loss.item(),
                gradient_norm=(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm) if grad_norm is not None else 0.0,  # type: ignore[union-attr]
                learning_rate=current_lr
            )

        # FIXED: Increment step counts at END of step
        # Always increment micro_step_count (every forward/backward pass)
        self.micro_step_count += 1
        self.step_count = self.micro_step_count  # Legacy support

        # Only increment optimizer_step_count when optimizer actually stepped
        if is_accumulation_complete:
            self.optimizer_step_count += 1

        # Calculate step time and iterations per second
        step_time = time.time() - step_start_time
        iterations_per_sec = 1.0 / step_time if step_time > 0 else 0.0

        # Return step results with comprehensive monitoring info
        return {
            "loss": total_loss.item(),
            "main_loss": main_loss.item(),
            "learning_rate": current_lr,
            "step_time": step_time,
            "iterations_per_sec": iterations_per_sec,
            "grad_norm": (
                grad_norm.item()  # type: ignore[union-attr]
                if grad_norm is not None and hasattr(grad_norm, "item")
                else (grad_norm if grad_norm is not None else 0.0)
            ),
            "grad_norm_pre_clip": (
                grad_norm_pre_clip.item()  # type: ignore[union-attr]
                if grad_norm_pre_clip is not None
                and hasattr(grad_norm_pre_clip, "item")
                else (grad_norm_pre_clip if grad_norm_pre_clip is not None else 0.0)
            ),
            "warmup_info": warmup_info,
            "lr_info": lr_info,
            "step_metrics": step_metrics,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optimizer_time": opt_time,
            # Memory health information
            "memory_status": memory_health["status"],
            "memory_utilization": memory_health["gpu_utilization"],
            "memory_available_gb": memory_health["available_gb"],
            "memory_oom_risk": memory_health["oom_risk"],
            "recommended_batch_size": memory_health["recommended_batch_size"],
            # Loss health information
            "loss_health": loss_health_result,
        }

    def set_validation_loss(self, validation_loss: float):
        """
        Set the validation loss for adaptive LR plateau detection.

        Args:
            validation_loss: Current validation loss
        """
        self._last_validation_loss = validation_loss

        if hasattr(self, "lr_manager") and self.lr_manager is not None:
            # Pass the new validation loss to the LR manager for plateau detection
            # Don't pass training_step here to avoid incrementing step counter
            lr_step_info = self.lr_manager.step(validation_loss)
            if self.lr_manager.config.enable_adaptive:
                lr_stats = self.lr_manager.get_statistics()
                print(
                    f"    📊 Validation loss set: {validation_loss:.4f} "
                    f"(adaptive LR enabled, patience: {lr_stats.get('plateau_patience', 'N/A')})"
                )

    def update_running_loss(self, loss: float):
        """
        Update exponential moving average of training loss.
        This provides an accurate running loss for checkpoint reporting.

        Args:
            loss: Current batch loss value
        """
        # Exponential moving average with configurable window
        alpha = 2.0 / (self.running_loss_window_size + 1)

        if self.running_loss_count == 0:
            # First loss - initialize
            self.running_loss_avg = loss
        else:
            # Update EMA
            self.running_loss_avg = alpha * loss + (1 - alpha) * self.running_loss_avg

        self.running_loss_count += 1

    def _apply_gradient_surgery(self, task_losses: Dict[str, torch.Tensor], optimizer):
        """Apply gradient surgery for multi-task learning."""
        if self.gradient_surgeon:
            try:
                # Apply gradient surgery using the configured method
                optimizer.zero_grad(set_to_none=True)

                # Compute gradients for each task
                task_gradients = {}
                task_list = list(task_losses.items())
                for idx, (task_name, loss) in enumerate(task_list):
                    # Only retain graph for non-final tasks to avoid memory leak
                    is_final_task = idx == len(task_list) - 1
                    loss.backward(retain_graph=not is_final_task)

                    task_gradients[task_name] = []
                    base_model = self._get_base_model
                    for param in base_model.parameters():
                        if param.grad is not None:
                            task_gradients[task_name].append(param.grad.clone())
                        else:
                            task_gradients[task_name].append(torch.zeros_like(param))

                    # Clear gradients after cloning (except for final task)
                    if not is_final_task:
                        optimizer.zero_grad(set_to_none=True)

                # Apply gradient surgery - returns a LIST of gradients
                modified_gradients = self.gradient_surgeon.apply_surgery(task_gradients)

                # Clear any remaining gradients before setting modified ones
                optimizer.zero_grad(set_to_none=True)

                # Update model parameters with modified gradients
                # modified_gradients is a list, not a dict
                # MEMORY OPTIMIZATION: No need to clone - gradients already processed
                base_model = self._get_base_model
                for param, grad in zip(base_model.parameters(), modified_gradients):
                    param.grad = grad if grad is not None else None

                # MEMORY OPTIMIZATION: Explicitly delete task gradients to free memory
                del task_gradients
                del modified_gradients

            except Exception as e:
                print(
                    f" Gradient surgery failed: {e}, falling back to standard training"
                )
                # Fallback to standard training
                optimizer.zero_grad(set_to_none=True)
                task_losses["main"].backward()
        else:
            # Standard backward pass
            optimizer.zero_grad(set_to_none=True)
            task_losses["main"].backward()

    def save_checkpoint(self, checkpoint_dir: str, tag: Optional[str] = None) -> str:
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
            print(
                f"🔄 Coordinating synchronized checkpoint across {self.distributed_manager.world_size} ranks..."
            )

            # Use fault-tolerant checkpointing
            sync_success = self.distributed_manager.checkpoint_with_fault_tolerance(
                checkpoint_dir, self.step_count
            )

            if not sync_success:
                print(
                    f"❌ Failed to coordinate distributed checkpoint - proceeding with local checkpoint"
                )
            else:
                print(f"✅ Distributed checkpoint coordination successful")
        if self.deepspeed_engine:
            # DeepSpeed checkpoint saving with client_state
            from pathlib import Path

            # FIXED: Build client_state dict with training metadata
            # Store both micro_step_count and optimizer_step_count for proper resumption
            client_state = {
                "step": self.optimizer_step_count,  # Primary step count (optimizer steps)
                "step_count": self.step_count,  # Legacy micro-step count
                "micro_step_count": self.micro_step_count,
                "optimizer_step_count": self.optimizer_step_count,
                "epoch": self.epoch_count,
                "epoch_count": self.epoch_count,
                "loss": self.best_loss,
                "best_loss": self.best_loss,
            }

            # Save with client_state
            self.deepspeed_engine.save_checkpoint(checkpoint_dir, tag, client_state=client_state)
            # DeepSpeed returns None, construct path manually
            checkpoint_path = Path(checkpoint_dir) / (tag or f"step_{self.step_count}")
            print(f" DeepSpeed checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
        else:
            # Standard PyTorch checkpoint saving
            from pathlib import Path

            import torch  # type: ignore[import]

            checkpoint_dir_path = Path(checkpoint_dir)
            checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

            checkpoint_file = (
                checkpoint_dir_path / f"checkpoint{'_' + tag if tag else ''}.pt"
            )

            base_model = self._get_base_model
            checkpoint = {
                "model_state_dict": base_model.state_dict(),
                "step_count": self.step_count,  # Legacy micro-step count
                "micro_step_count": self.micro_step_count,
                "optimizer_step_count": self.optimizer_step_count,
                "epoch_count": self.epoch_count,
                "best_loss": self.best_loss,
                "scaler_last_reset": getattr(self, "scaler_last_reset", 0),
            }

            if hasattr(self, "optimizer"):
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

            # Save scaler state for mixed precision training
            if self.scaler is not None:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()

            # Save monitor states for continuity
            if hasattr(self, "gradient_health"):
                if self.gradient_health_enabled and self.gradient_health is not None:
                    checkpoint["gradient_health_state"] = {  # type: ignore
                        "total_steps": self.gradient_health.total_steps,
                        "total_explosions": self.gradient_health.total_explosions,
                        "stats": self.gradient_health.stats,
                    }

            if hasattr(self, "loss_health"):
                checkpoint["loss_health_state"] = {
                    "best_loss": self.loss_health.best_loss,
                    "steps_since_improvement": self.loss_health.steps_since_improvement,
                    "spike_count": self.loss_health.spike_count,
                    "divergence_count": self.loss_health.divergence_count,
                }

            # CRITICAL FIX: Save learning rate manager state
            if hasattr(self, "lr_manager") and self.lr_manager is not None:
                checkpoint["lr_manager_state"] = {
                    "current_step": self.lr_manager.current_step,
                    "warmup_steps": self.lr_manager.warmup_steps,
                    "total_steps": self.lr_manager.total_steps,
                    "initial_lrs": self.lr_manager.initial_lrs,
                    "in_recovery": self.lr_manager.in_recovery,
                    "recovery_start_step": self.lr_manager.recovery_start_step,
                    "pre_recovery_lr": self.lr_manager.pre_recovery_lr,
                    "reduction_count": self.lr_manager.reduction_count,
                    "recovery_count": self.lr_manager.recovery_count,
                }
                # Save plateau detector state if adaptive LR is enabled
                if self.lr_manager.plateau_detector is not None:
                    checkpoint["lr_manager_state"]["plateau_detector"] = {  # type: ignore[index]
                        "best_loss": self.lr_manager.plateau_detector.best_loss,  # type: ignore[attr-defined]
                        "patience_counter": self.lr_manager.plateau_detector.patience_counter,  # type: ignore[attr-defined]
                        "num_reductions": self.lr_manager.plateau_detector.num_reductions,  # type: ignore[attr-defined]
                        "checks_since_last_reduction": self.lr_manager.plateau_detector.checks_since_last_reduction,  # type: ignore[attr-defined]
                    }

            # Save observability state (Phase 7)
            if self.observability:
                try:
                    checkpoint["observability_state"] = (
                        self.observability.create_checkpoint_data()
                    )
                except Exception as e:
                    print(f"    Warning: Failed to save observability state: {e}")
                    checkpoint["observability_state"] = {"error": str(e)}

            torch.save(checkpoint, checkpoint_file)
            print(f" Standard checkpoint saved: {checkpoint_file}")

            # Update last valid checkpoint path for potential restore
            self.last_valid_checkpoint_path = str(checkpoint_file)

            return str(checkpoint_file)

    def load_checkpoint(self, checkpoint_path: str, tag: Optional[str] = None) -> Dict[str, Any]:
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
            _, client_state = self.deepspeed_engine.load_checkpoint(
                checkpoint_path, tag
            )
            print(f"DeepSpeed checkpoint loaded: {checkpoint_path}")

            # FIXED: Restore training state from client_state with proper step counts
            if client_state:
                # Restore both step counters with proper fallback chain
                # Priority: optimizer_step_count > step (for optimizer step count)
                # Priority: micro_step_count > step_count > step (for micro step count)
                self.optimizer_step_count = client_state.get("optimizer_step_count", client_state.get("step", 0))
                self.micro_step_count = client_state.get("micro_step_count", client_state.get("step_count", client_state.get("step", 0)))
                self.step_count = self.micro_step_count  # Legacy support
                self.epoch_count = client_state.get("epoch", client_state.get("epoch_count", 0))
                self.best_loss = client_state.get("loss", client_state.get("best_loss", float("inf")))

                print(
                    f"   ✓ Training state: optimizer_step={self.optimizer_step_count}, "
                    f"micro_step={self.micro_step_count}, epoch={self.epoch_count}, best_loss={self.best_loss:.4f}"
                )

                # Mark states as restored
                restored_states = {
                    "model": True,  # DeepSpeed handles model state
                    "optimizer": True,  # DeepSpeed handles optimizer state
                }

                # Return properly structured result
                return {
                    "step_count": self.step_count,
                    "epoch_count": self.epoch_count,
                    "best_loss": self.best_loss,
                    "restored_states": restored_states,
                    "checkpoint_file": str(checkpoint_path),
                }

            return client_state or {}
        else:
            # Standard PyTorch checkpoint loading with device-agnostic handling
            from pathlib import Path

            import torch  # type: ignore[import]

            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

            print(f"🔄 Loading checkpoint: {checkpoint_file}")
            print(f"   Target device: {self.device}")

            # Device-agnostic loading - always load to CPU first, then move to target device
            try:
                checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                print("   ✓ Checkpoint loaded to CPU")
            except Exception as e:
                # Fallback to target device if CPU loading fails
                print(f"   ⚠️  CPU loading failed: {e}")
                print(f"   Trying direct loading to {self.device}")
                checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)

            # CRITICAL FIX: Load model state first, THEN optimizer state
            # Optimizer state must be loaded AFTER model parameters are in place
            if "model_state_dict" in checkpoint:
                try:
                    # Move model state to target device if needed
                    model_state = checkpoint["model_state_dict"]
                    if self.device != torch.device("cpu"):
                        # Move tensors to target device
                        model_state = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in model_state.items()
                        }
                    base_model = self._get_base_model
                    base_model.load_state_dict(model_state)
                    print("   ✓ Model state restored")
                except Exception as e:
                    print(f"   ❌ Failed to restore model state: {e}")
                    raise

            # IMPORTANT: Optimizer state must be loaded AFTER model state is fully loaded
            # This ensures optimizer momentum/state aligns with loaded model parameters
            optimizer_restored = False
            has_optimizer = hasattr(self, "optimizer") and self.optimizer is not None
            has_optimizer_state = "optimizer_state_dict" in checkpoint

            if has_optimizer and has_optimizer_state:
                try:
                    # Verify optimizer has correct parameter groups matching loaded model
                    num_param_groups = len(self.optimizer.param_groups)
                    base_model = self._get_base_model
                    num_model_params = sum(1 for _ in base_model.parameters())

                    optimizer_state = checkpoint["optimizer_state_dict"]

                    # Move optimizer state tensors to correct device
                    if self.device != torch.device("cpu"):
                        # Handle optimizer state device placement
                        for state in optimizer_state["state"].values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(self.device)

                    # Load optimizer state - this must happen AFTER model is loaded
                    self.optimizer.load_state_dict(optimizer_state)
                    optimizer_restored = True  # Mark as successfully restored
                    print(f"   ✓ Optimizer state restored ({num_param_groups} param groups, {num_model_params} params)")
                except Exception as e:
                    print(f"   ⚠️  Failed to restore optimizer state: {e}")
                    print(f"      Error type: {type(e).__name__}")
                    print("   Continuing with fresh optimizer state (momentum/adam state reset)")
            else:
                # Debug: Print why optimizer wasn't loaded
                if not has_optimizer:
                    print(f"   ⚠️  Optimizer not available for restoration (has_optimizer={has_optimizer})")
                if not has_optimizer_state:
                    print(f"   ⚠️  Optimizer state not in checkpoint (has_optimizer_state={has_optimizer_state})")

            # FIXED: Load basic training state with proper step counters
            # Check for new step counters, fall back to legacy for backwards compatibility
            # Priority: optimizer_step_count > step (for optimizer step count)
            # Priority: micro_step_count > step_count > step (for micro step count)
            self.optimizer_step_count = checkpoint.get("optimizer_step_count", checkpoint.get("step", 0))
            self.micro_step_count = checkpoint.get("micro_step_count", checkpoint.get("step_count", checkpoint.get("step", 0)))
            self.step_count = self.micro_step_count  # Legacy support
            self.epoch_count = checkpoint.get("epoch", checkpoint.get("epoch_count", 0))
            self.best_loss = checkpoint.get("loss", checkpoint.get("best_loss", float("inf")))
            self.scaler_last_reset = checkpoint.get("scaler_last_reset", 0)
            print(
                f"   ✓ Training state: optimizer_step={self.optimizer_step_count}, "
                f"micro_step={self.micro_step_count}, epoch={self.epoch_count}, best_loss={self.best_loss:.4f}"
            )

            # CRITICAL FIX: Track what was actually restored
            restored_states = {
                "model": True,
                "optimizer": optimizer_restored,  # Use the flag we set during actual loading
            }

            # FIXED: Restore adaptive_lr_manager or old lr_manager (check adaptive first)
            if (
                hasattr(self, "adaptive_lr_manager")
                and self.adaptive_lr_manager is not None
                and "adaptive_lr_manager_state" in checkpoint
            ):
                try:
                    self.adaptive_lr_manager.load_state_dict(checkpoint["adaptive_lr_manager_state"])  # type: ignore[attr-defined]
                    print(
                        f"   ✓ Adaptive LR manager state restored: step={self.adaptive_lr_manager.step_count}, "  # type: ignore[attr-defined]
                        f"best_loss={self.adaptive_lr_manager.best_loss:.4f}, reductions={self.adaptive_lr_manager.lr_reductions}"  # type: ignore[attr-defined]
                    )
                    restored_states["adaptive_lr_manager"] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore adaptive LR manager state: {e}")
                    restored_states["adaptive_lr_manager"] = False
            elif (
                hasattr(self, "lr_manager")
                and self.lr_manager is not None
                and "lr_manager_state" in checkpoint
            ):
                try:
                    lr_state = checkpoint["lr_manager_state"]
                    # Restore LR manager internal state
                    self.lr_manager.current_step = lr_state.get("current_step", 0)
                    self.lr_manager.warmup_steps = lr_state.get("warmup_steps", self.lr_manager.warmup_steps)
                    self.lr_manager.total_steps = lr_state.get("total_steps", self.lr_manager.total_steps)
                    self.lr_manager.initial_lrs = lr_state.get("initial_lrs", self.lr_manager.initial_lrs)
                    self.lr_manager.in_recovery = lr_state.get("in_recovery", False)
                    self.lr_manager.recovery_start_step = lr_state.get("recovery_start_step", 0)
                    self.lr_manager.pre_recovery_lr = lr_state.get("pre_recovery_lr", None)
                    self.lr_manager.reduction_count = lr_state.get("reduction_count", 0)
                    self.lr_manager.recovery_count = lr_state.get("recovery_count", 0)

                    # Restore plateau detector state if exists
                    if "plateau_detector" in lr_state and self.lr_manager.plateau_detector is not None:
                        pd_state = lr_state["plateau_detector"]
                        self.lr_manager.plateau_detector.best_loss = pd_state.get("best_loss", float('inf'))  # type: ignore[attr-defined]
                        self.lr_manager.plateau_detector.patience_counter = pd_state.get("patience_counter", 0)  # type: ignore[attr-defined]
                        self.lr_manager.plateau_detector.num_reductions = pd_state.get("num_reductions", 0)  # type: ignore[attr-defined]
                        self.lr_manager.plateau_detector.checks_since_last_reduction = pd_state.get("checks_since_last_reduction", 0)  # type: ignore[attr-defined]

                    phase = "recovery" if self.lr_manager.in_recovery else ("warmup" if self.lr_manager.current_step < self.lr_manager.warmup_steps else "main")
                    print(
                        f"   ✓ LR manager state restored: step={self.lr_manager.current_step}, phase={phase}, reductions={self.lr_manager.reduction_count}"
                    )
                    restored_states["lr_manager"] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore LR manager state: {e}")
                    print(f"      Error details: {type(e).__name__}")
                    restored_states["lr_manager"] = False
            else:
                restored_states["lr_manager"] = False

            # Restore mixed precision scaler state
            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                try:
                    scaler_state = checkpoint["scaler_state_dict"]
                    # Move scaler state to appropriate device if needed
                    if self.device != torch.device("cpu") and "_scale" in scaler_state:
                        if isinstance(scaler_state["_scale"], torch.Tensor):
                            scaler_state["_scale"] = scaler_state["_scale"].to(
                                self.device
                            )
                    self.scaler.load_state_dict(scaler_state)
                    print("   ✓ Mixed precision scaler state restored")
                    restored_states["scaler"] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore scaler state: {e}")
                    print("   Creating new scaler")
                    restored_states["scaler"] = False
            else:
                restored_states["scaler"] = False

            # Restore gradient health monitor state (only if enabled)
            if (
                self.gradient_health_enabled
                and hasattr(self, "gradient_health")
                and self.gradient_health is not None
                and "gradient_health_state" in checkpoint
            ):
                try:
                    gh_state = checkpoint["gradient_health_state"]
                    self.gradient_health.explosion_threshold = gh_state.get(
                        "explosion_threshold", 5.0
                    )
                    self.gradient_health.current_clip_value = gh_state.get(  # type: ignore[attr-defined]
                        "clip_value", 1.0
                    )
                    self.gradient_health.total_explosions = gh_state.get(
                        "total_explosions", 0
                    )
                    self.gradient_health.total_steps = gh_state.get("total_steps", 0)

                    # Restore history (limited to prevent memory issues)
                    if "recent_explosions" in gh_state:
                        self.gradient_health.recent_explosions.extend(
                            gh_state["recent_explosions"][-50:]
                        )
                    if "grad_norm_history" in gh_state:
                        self.gradient_health.grad_norm_history.extend(
                            gh_state["grad_norm_history"][-100:]
                        )
                    if "grad_norm_pre_clip_history" in gh_state:
                        self.gradient_health.grad_norm_pre_clip_history.extend(
                            gh_state["grad_norm_pre_clip_history"][-100:]
                        )

                    print(
                        f"   ✓ Gradient health restored: {self.gradient_health.total_explosions} explosions, {self.gradient_health.total_steps} steps"
                    )
                    restored_states["gradient_health"] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore gradient health state: {e}")
                    restored_states["gradient_health"] = False
            else:
                restored_states["gradient_health"] = False

            # Restore memory monitor state
            if hasattr(self, "memory_monitor") and "memory_monitor_state" in checkpoint:
                try:
                    mem_state = checkpoint["memory_monitor_state"]
                    # Memory monitor state doesn't need device handling
                    print(
                        f"   ✓ Memory monitor state found: {mem_state.get('emergency_count', 0)} emergencies"
                    )
                    restored_states["memory_monitor"] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore memory monitor state: {e}")
                    restored_states["memory_monitor"] = False
            else:
                restored_states["memory_monitor"] = False

            # Restore loss health monitor state
            if hasattr(self, "loss_health") and "loss_health_state" in checkpoint:
                try:
                    lh_state = checkpoint["loss_health_state"]
                    if "loss_history" in lh_state:
                        self.loss_health.loss_history.extend(
                            lh_state["loss_history"][-100:]
                        )
                    self.loss_health.spike_threshold = lh_state.get(  # type: ignore[attr-defined]
                        "spike_threshold", 5.0
                    )
                    self.loss_health.nan_count = lh_state.get("nan_count", 0)  # type: ignore[attr-defined]
                    self.loss_health.inf_count = lh_state.get("inf_count", 0)  # type: ignore[attr-defined]
                    self.loss_health.spike_count = lh_state.get("spike_count", 0)  # type: ignore[attr-defined]

                    print(
                        f"   ✓ Loss health restored: {self.loss_health.nan_count} NaN, {self.loss_health.inf_count} Inf, {self.loss_health.spike_count} spikes"  # type: ignore[attr-defined]
                    )
                    restored_states["loss_health"] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore loss health state: {e}")
                    restored_states["loss_health"] = False
            else:
                restored_states["loss_health"] = False

            # Restore random states for reproducibility
            if "random_states" in checkpoint:
                try:
                    import random

                    import numpy as np

                    rand_states = checkpoint["random_states"]

                    if "python_random" in rand_states:
                        random.setstate(rand_states["python_random"])
                    if "numpy_random" in rand_states:
                        np.random.set_state(rand_states["numpy_random"])
                    if "torch_random" in rand_states:
                        torch.set_rng_state(rand_states["torch_random"])
                    if (
                        "torch_cuda_random" in rand_states
                        and rand_states["torch_cuda_random"] is not None
                    ):
                        if torch.cuda.is_available():
                            torch.cuda.set_rng_state(rand_states["torch_cuda_random"])

                    print("   ✓ Random states restored for reproducibility")
                    restored_states["random_states"] = True
                except Exception as e:
                    print(f"   ⚠️  Failed to restore random states: {e}")
                    restored_states["random_states"] = False
            else:
                restored_states["random_states"] = False

            # Check for early stopping and training progress states (for informational purposes)
            early_stopping_info = {}
            if "early_stopping_state" in checkpoint:
                early_stopping_info = checkpoint["early_stopping_state"]
                print(
                    f"   ℹ️  Early stopping state: enabled={early_stopping_info.get('enabled', False)}, "
                    f"patience={early_stopping_info.get('patience', 'N/A')}"
                )

            training_progress_info = {}
            if "training_progress" in checkpoint:
                training_progress_info = checkpoint["training_progress"]
                print(
                    f"   ℹ️  Training progress: epoch {training_progress_info.get('current_epoch', 'N/A')}/{training_progress_info.get('total_epochs', 'N/A')}, "
                    f"complete={training_progress_info.get('training_complete', False)}"
                )

            print(f"✅ Checkpoint loaded: {checkpoint_file}")
            print(
                f"   States restored: {sum(restored_states.values())}/{len(restored_states)}"
            )

            return {
                "step_count": self.step_count,
                "epoch_count": self.epoch_count,
                "best_loss": self.best_loss,
                "restored_states": restored_states,
                "early_stopping_info": early_stopping_info,
                "training_progress_info": training_progress_info,
                "checkpoint_file": str(checkpoint_file),
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
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "best_loss": self.best_loss,
        }

        if self.metrics_collector:
            stats["metrics"] = self.metrics_collector.get_performance_summary()

        if hasattr(self, 'warmup_scheduler') and self.warmup_scheduler:
            stats["warmup"] = self.warmup_scheduler._get_warmup_info()

        if self.async_logger:
            stats["logging"] = self.async_logger.get_logging_statistics()

        if self.gpu_manager:
            stats["memory"] = self.gpu_manager.get_memory_stats()

        # Add health monitoring statistics
        if self.gradient_health_enabled and hasattr(self, "gradient_health") and self.gradient_health is not None:
            stats["gradient_health"] = self.gradient_health.get_health_summary()

        if hasattr(self, "loss_health"):
            stats["loss_health"] = self.loss_health.get_health_summary()

        if hasattr(self, "memory_monitor"):
            stats["memory_monitor"] = self.memory_monitor.get_memory_summary()

        # Add scaler information
        if self.scaler is not None:
            stats["mixed_precision"] = {
                "scaler_scale": self.scaler.get_scale(),
                "last_reset_step": getattr(self, "scaler_last_reset", 0),
                "steps_since_reset": self.step_count
                - getattr(self, "scaler_last_reset", 0),
            }

        # Add observability statistics (Phase 7)
        if self.observability:
            try:
                stats["observability"] = self.observability.get_observability_summary()
            except Exception as e:
                stats["observability"] = {
                    "error": f"Failed to get summary: {e}",
                    "enabled": True,
                }
        else:
            stats["observability"] = {"enabled": False}

        return stats

    def _attempt_checkpoint_restore(self, reason: str) -> None:
        """Attempt to restore from the last valid checkpoint when loss becomes invalid.

        This method is called when an invalid loss is detected to try recovery
        from a previous good state instead of crashing the training.

        Args:
            reason: Reason for the invalid loss detection
        """
        if self.checkpoint_restore_attempts >= self.max_restore_attempts:
            print(
                f"     ⚠️  Maximum restore attempts ({self.max_restore_attempts}) reached. "
                f"Will not attempt further restoration."
            )
            return

        # Find the best checkpoint to restore from
        checkpoint_path = self._find_best_checkpoint_for_restore()
        if not checkpoint_path:
            print("     ⚠️  No valid checkpoint found for restoration.")
            return

        try:
            self.checkpoint_restore_attempts += 1
            print(
                f"     🔄 Attempting checkpoint restore #{self.checkpoint_restore_attempts}: {checkpoint_path}"
            )
            print(f"     📋 Restore reason: {reason}")

            # Load the checkpoint
            checkpoint_data = self.load_checkpoint(checkpoint_path)

            # Reset loss health monitor to prevent immediate re-triggering
            if hasattr(self, "loss_health"):
                self.loss_health.reset_history()  # type: ignore[attr-defined]
                print("     ✓ Loss health monitor history reset")

            # Reset gradient health monitor
            if self.gradient_health_enabled and hasattr(self, "gradient_health") and self.gradient_health is not None:
                self.gradient_health.reset_history()  # type: ignore[attr-defined]
                print("     ✓ Gradient health monitor history reset")

            # Lower learning rate to reduce instability
            if hasattr(self, "optimizer") and self.optimizer:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group["lr"]
                    param_group["lr"] = old_lr * 0.5  # Reduce LR by half
                    print(
                        f"     📉 Reduced learning rate from {old_lr:.2e} to {param_group['lr']:.2e}"
                    )

            print("     ✅ Checkpoint restoration completed. Training will continue.")

        except Exception as e:
            print(f"     ❌ Checkpoint restoration failed: {e}")
            print("     Will proceed with original error handling.")

    def _find_best_checkpoint_for_restore(self) -> Optional[str]:
        """Find the best checkpoint for restoration.

        Returns:
            Path to the best checkpoint, or None if no valid checkpoint found
        """
        from pathlib import Path
        from typing import Optional

        # First, try the last valid checkpoint we tracked
        if (
            self.last_valid_checkpoint_path
            and Path(self.last_valid_checkpoint_path).exists()
        ):
            return self.last_valid_checkpoint_path

        # If we have a run manager, try to find the best checkpoint from it
        if self.run_manager and hasattr(self.run_manager, "get_checkpoint_path"):
            try:
                best_checkpoint = self.run_manager.get_checkpoint_path("best")
                if best_checkpoint and Path(best_checkpoint).exists():
                    return str(best_checkpoint)
            except Exception as e:
                print(f"     ⚠️  Could not get best checkpoint from run manager: {e}")

        # Try to find checkpoints in the current run directory
        if self.run_manager and hasattr(self.run_manager, "run_dir"):
            run_dir = Path(self.run_manager.run_dir)
            checkpoint_dir = run_dir / "checkpoints"
            if checkpoint_dir.exists():
                # Look for best_model.pt first
                best_model_path = checkpoint_dir / "best_model.pt"
                if best_model_path.exists():
                    return str(best_model_path)

                # Fall back to latest_model.pt
                latest_model_path = checkpoint_dir / "latest_model.pt"
                if latest_model_path.exists():
                    return str(latest_model_path)

                # Look for any .pt files
                pt_files = list(checkpoint_dir.glob("*.pt"))
                if pt_files:
                    # Sort by modification time and return the newest
                    newest_checkpoint = max(pt_files, key=lambda p: p.stat().st_mtime)
                    return str(newest_checkpoint)

        return None
