"""
Enhanced Trainer using Modular Components

This module provides the core Enhanced Trainer that integrates all the
modular training components for maximum flexibility and maintainability.
"""

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
from ..retrieval.rag_system import RAGSystem, KnowledgeBase
from ..evaluation.comprehensive_eval import ComprehensiveEvaluator
from ..optimization.quantization import ModelQuantizer, QuantizationConfig
from ..memory.episodic_memory import EpisodicMemoryBank, AdaptiveMemoryManager, ExperienceReplay


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

        # Initialize all modular components
        self._init_gpu_memory_manager()
        self._init_performance_manager()
        self._init_deepspeed()
        self._init_async_logger()
        self._init_metrics_collector()
        self._init_loss_functions()
        self._init_gradient_surgery()
        self._init_rag_system()
        self._init_evaluator()
        self._init_quantization()
        self._init_episodic_memory()

        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.best_loss = float('inf')

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
            print("⚠️ DeepSpeed enabled but not in distributed environment")
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

            # Always add auxiliary loss for MoE stability
            loss_config['auxiliary'] = {
                'type': 'auxiliary',
                'load_balancing_weight': 0.001,
                'router_z_weight': 0.0001,
                'weight': 0.1
            }

            self.composite_loss = CompositeLoss(loss_config)
            print(f"🎯 Advanced loss functions: {list(loss_config.keys())}")
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
                print("🔧 Adaptive gradient surgery initialized")
            else:
                self.gradient_surgeon = GradientSurgeon(
                    method=self.config.gradient.gradient_surgery_method
                )
                print("🔧 Gradient surgery initialized")
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
                        print(f"📚 RAG system with KB: {kb_path.name}")
                    else:
                        print(f"⚠️ Knowledge base not found: {kb_path}")
                        self.rag_system = None
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
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

            print(f"📊 Evaluator: {', '.join(self.eval_metrics)}")
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
                print("🚀 NVFP4 4-bit quantization enabled")
            else:
                quant_config = QuantizationConfig(
                    bit_width=self.config.quantization.bit_width,
                    symmetric=True,
                    per_channel=True
                )
                print(f"🔧 {self.config.quantization.bit_width}-bit quantization enabled")

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

                print(f"🧠 Episodic memory: {self.config.memory.memory_capacity} capacity")
            except Exception as e:
                print(f"⚠️ Memory initialization failed: {e}")
                self.memory_bank = None
                self.memory_manager = None
                self.experience_replay = None
        else:
            self.memory_bank = None
            self.memory_manager = None
            self.experience_replay = None

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
            'deepspeed_enabled': self.deepspeed_engine is not None
        })

        print("🚀 Training setup completed")
        return setup_info

    def _setup_deepspeed_training(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Set up DeepSpeed training."""
        try:
            # Load DeepSpeed config from file if provided
            if self.config.deepspeed.config_file:
                import json
                with open(self.config.deepspeed.config_file, 'r') as f:
                    file_config = json.load(f)
                # Merge with generated config (file config takes precedence)
                self.deepspeed_config.update(file_config)

            # Initialize DeepSpeed engine
            self.deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=self.model,
                optimizer=optimizer,
                config=self.deepspeed_config,
                lr_scheduler=None  # We'll handle LR scheduling manually
            )

            self.model = self.deepspeed_engine.module  # Get the wrapped model
            self.optimizer = optimizer  # DeepSpeed-managed optimizer
            self.lr_scheduler = lr_scheduler

            print(f"✅ DeepSpeed initialized with ZeRO stage {self.config.deepspeed.zero_stage}")

            return {
                'deepspeed_engine': True,
                'zero_stage': self.config.deepspeed.zero_stage,
                'warmup_enabled': False,  # DeepSpeed handles warmup
                'adaptive_lr_enabled': False,  # DeepSpeed handles LR scheduling
            }

        except Exception as e:
            print(f"❌ DeepSpeed initialization failed: {e}")
            print("   Falling back to standard training")
            return self._setup_standard_training(optimizer)

    def _setup_standard_training(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Set up standard (non-DeepSpeed) training."""
        # Create warmup scheduler
        warmup_config = WarmupConfig(
            warmup_steps=2000,  # Could be from config
            schedule="cosine",
            start_ratio=0.01,
            use_gradient_norm=True
        )
        self.warmup_scheduler = AdvancedWarmupScheduler(optimizer, warmup_config)

        # Create adaptive LR manager
        adaptive_lr_config = AdaptiveLRConfig(
            plateau_patience=500,
            divergence_threshold=1.5,
            stability_threshold=5,
            max_lr=1e-3
        )
        self.adaptive_lr_manager = AdaptiveLearningRateManager(optimizer, adaptive_lr_config)

        return {
            'deepspeed_engine': False,
            'warmup_enabled': True,
            'adaptive_lr_enabled': True,
        }

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
        # Start metrics collection
        if self.metrics_collector:
            self.metrics_collector.start_step(self.step_count, epoch, batch_idx)

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
        total_loss = main_loss

        # Apply composite loss if available
        if self.composite_loss:
            aux_losses = self.composite_loss(
                inputs=input_ids,
                targets=labels,
                logits=outputs.logits if hasattr(outputs, 'logits') else None,
                model=self.model,
                router_logits=getattr(outputs, 'router_logits', None),
                expert_outputs=getattr(outputs, 'expert_outputs', None)
            )
            for name, loss_value in aux_losses.items():
                if name != 'total':  # Skip total if present
                    total_loss = total_loss + loss_value

        # Backward pass with timing
        backward_start = time.time()

        # Handle DeepSpeed vs standard training
        if self.deepspeed_engine:
            # DeepSpeed training
            self.deepspeed_engine.backward(total_loss)
            self.deepspeed_engine.step()
            grad_norm = None  # DeepSpeed handles gradient clipping internally
        else:
            # Standard training
            if self.gradient_surgeon and self.config.multi_task:
                # Apply gradient surgery
                task_losses = {'main': total_loss}  # Could have multiple tasks
                self._apply_gradient_surgery(task_losses, optimizer)
            else:
                # Standard backward pass
                optimizer.zero_grad()
                total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()

        backward_time = time.time() - backward_start
        opt_time = backward_time  # Combined for DeepSpeed

        # Learning rate management
        if self.deepspeed_engine:
            current_lr = self.deepspeed_engine.get_lr()[0] if self.deepspeed_engine.get_lr() else 0.0
            warmup_info = {"warmup_step": self.step_count}  # DeepSpeed handles warmup
            lr_info = {"lr_step": self.step_count}  # DeepSpeed handles LR scheduling
        else:
            current_lr = optimizer.param_groups[0]['lr']

            # Warmup step
            warmup_info = self.warmup_scheduler.step(
                loss=total_loss.item(),
                model=self.model
            )

            # Adaptive LR step
            lr_info = self.adaptive_lr_manager.step(total_loss.item(), batch_idx)

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
            metrics = {
                'train/loss': total_loss.item(),
                'train/main_loss': main_loss.item(),
                'train/learning_rate': current_lr,
                'train/grad_norm': grad_norm.item() if grad_norm is not None else 0.0,
                **grad_metrics,
                **step_metrics
            }
            self.async_logger.log_metrics(metrics, self.step_count)

        # Update training state
        self.step_count += 1
        if total_loss.item() < self.best_loss:
            self.best_loss = total_loss.item()

        # Return step results
        return {
            'loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'learning_rate': current_lr,
            'grad_norm': grad_norm.item() if grad_norm is not None else 0.0,
            'warmup_info': warmup_info,
            'lr_info': lr_info,
            'step_metrics': step_metrics,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'optimizer_time': opt_time
        }

    def _apply_gradient_surgery(self, task_losses: Dict[str, torch.Tensor], optimizer):
        """Apply gradient surgery for multi-task learning."""
        if self.gradient_surgeon:
            try:
                # Apply gradient surgery using the configured method
                optimizer.zero_grad()

                # Compute gradients for each task
                task_gradients = {}
                for task_name, loss in task_losses.items():
                    loss.backward(retain_graph=True)
                    task_gradients[task_name] = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            task_gradients[task_name].append(param.grad.clone())
                        else:
                            task_gradients[task_name].append(torch.zeros_like(param))
                    optimizer.zero_grad()

                # Apply gradient surgery
                modified_gradients = self.gradient_surgeon.apply_surgery(task_gradients)

                # Update model parameters with modified gradients
                for param, grad in zip(self.model.parameters(), modified_gradients.get('main', [])):
                    if param.grad is not None:
                        param.grad = grad

            except Exception as e:
                print(f"⚠️ Gradient surgery failed: {e}, falling back to standard training")
                # Fallback to standard training
                optimizer.zero_grad()
                task_losses['main'].backward()
        else:
            # Standard backward pass
            optimizer.zero_grad()
            task_losses['main'].backward()

    def save_checkpoint(self, checkpoint_dir: str, tag: str = None) -> str:
        """
        Save checkpoint with DeepSpeed support.

        Args:
            checkpoint_dir: Directory to save checkpoint
            tag: Optional tag for the checkpoint

        Returns:
            Path to saved checkpoint
        """
        if self.deepspeed_engine:
            # DeepSpeed checkpoint saving
            checkpoint_path = self.deepspeed_engine.save_checkpoint(checkpoint_dir, tag)
            print(f"💾 DeepSpeed checkpoint saved: {checkpoint_path}")
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
            }

            if hasattr(self, 'optimizer'):
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

            torch.save(checkpoint, checkpoint_file)
            print(f"💾 Standard checkpoint saved: {checkpoint_file}")
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
            # Standard PyTorch checkpoint loading
            import torch
            from pathlib import Path

            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

            checkpoint = torch.load(checkpoint_file, map_location=self.device)

            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state if available
            if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load training state
            self.step_count = checkpoint.get('step_count', 0)
            self.epoch_count = checkpoint.get('epoch_count', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))

            print(f"Standard checkpoint loaded: {checkpoint_file}")
            return {
                'step_count': self.step_count,
                'epoch_count': self.epoch_count,
                'best_loss': self.best_loss
            }

    def cleanup(self):
        """Clean up all components."""
        if self.async_logger:
            self.async_logger.stop()

        if self.gpu_manager:
            self.gpu_manager.cleanup_gpu_memory(aggressive=True)

        # DeepSpeed cleanup
        if self.deepspeed_engine:
            print("🧹 Cleaning up DeepSpeed engine")
            # DeepSpeed handles its own cleanup automatically

        print("✅ Enhanced trainer cleanup completed")

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

        if self.warmup_scheduler:
            stats['warmup'] = self.warmup_scheduler._get_warmup_info()

        if self.async_logger:
            stats['logging'] = self.async_logger.get_logging_statistics()

        if self.gpu_manager:
            stats['memory'] = self.gpu_manager.get_memory_stats()

        return stats