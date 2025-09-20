#!/usr/bin/env python3
"""
A100-Optimized Training Script for Advanced LLM Framework

This training script integrates all NVIDIA A100-specific optimizations:
- TensorFloat-32 (TF32) automatic acceleration
- BFloat16/Float16 mixed precision training
- FlashAttention v3 with A100 optimizations
- CUDA graphs for reduced kernel launch overhead
- torch.compile with max-autotune mode
- Structured sparsity for Tensor Cores
- Memory pool optimization
- NVLink topology-aware communication
- Multi-Instance GPU (MIG) support

Usage:
    # Basic A100-optimized training
    python train_a100.py --config configs/gpu/small.yaml --enable-a100-optimizations

    # Full A100 optimization with all features
    python train_a100.py --config configs/gpu/large.yaml --enable-all-a100-features

    # Multi-GPU training with NVLink optimization
    torchrun --nproc_per_node=8 train_a100.py --config configs/gpu/large.yaml --nvlink-optimization

Examples:
    # Single A100 training with TF32 and BF16
    python train_a100.py --config configs/gpu/medium.yaml --tf32 --mixed-precision bf16

    # DGX A100 training (8 GPUs with NVSwitch)
    torchrun --nproc_per_node=8 train_a100.py --config configs/gpu/large.yaml --dgx-a100

    # MIG training with multiple instances
    python train_a100.py --config configs/gpu/small.yaml --mig-instances 4
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from collections import defaultdict

# Add project root to path
sys.path.append('/project/code')

# Core imports
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.data import create_dataloaders
from src.Ava.data_streaming import create_streaming_dataloaders
from transformers import AutoTokenizer

# A100 Optimization imports
from src.Ava.optimization.a100_optimizer import (
    A100Optimizer, StructuredSparsityOptimizer, MIGManager, benchmark_a100_optimizations
)
from src.Ava.optimization.flash_attention_v3 import (
    FlashAttentionV3, SequenceParallelFlashAttention, benchmark_flash_attention
)
from src.Ava.optimization.memory_optimizer import (
    A100MemoryOptimizer, GradientAccumulator, ActivationCheckpointing, profile_memory_usage
)
from src.Ava.optimization.nvlink_optimizer import (
    NVLinkOptimizer, GPUDirectOptimizer, TopologyAwareDataLoader, benchmark_nvlink_communication
)

# Enhanced feature imports (from original train.py)
from src.Ava.layers.mixture_of_heads import MixtureOfHeads
from src.Ava.layers.mixture_of_activations import MixtureOfActivations
from src.Ava.retrieval import RAGSystem
from src.Ava.losses import CompositeLoss
from src.Ava.training import GradientSurgeon
from src.Ava.evaluation.comprehensive_eval import ComprehensiveEvaluator
from src.Ava.memory import EpisodicMemoryBank

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class A100OptimizedTrainer:
    """
    Trainer with comprehensive A100 GPU optimizations.
    """

    def __init__(self, args, config, model, tokenizer, device):
        """
        Initialize A100-optimized trainer.

        Args:
            args: Command line arguments
            config: Model configuration
            model: PyTorch model
            tokenizer: Tokenizer
            device: Training device
        """
        self.args = args
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.start_time = time.time()

        # Initialize A100 optimizers
        self._initialize_a100_optimizers()

        # Apply optimizations to model
        self.model = self._optimize_model()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self._setup_optimizer()

        # Setup loss and metrics
        self.criterion = self._setup_loss()
        self.metrics = defaultdict(list)

        # Initialize evaluator if needed
        if args.eval_during_training:
            self.evaluator = ComprehensiveEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                metrics=args.eval_metrics.split(',') if args.eval_metrics else ['perplexity']
            )

        # Setup distributed training if available
        self._setup_distributed()

        logger.info("A100-optimized trainer initialized successfully")

    def _initialize_a100_optimizers(self):
        """Initialize all A100-specific optimizers."""
        # Main A100 optimizer
        self.a100_optimizer = A100Optimizer(
            enable_tf32=self.args.enable_tf32,
            mixed_precision=self.args.mixed_precision,
            use_cuda_graphs=self.args.use_cuda_graphs,
            memory_pool_size=self.args.memory_pool_size,
            enable_torch_compile=self.args.enable_torch_compile,
            compile_mode=self.args.compile_mode,
            enable_gradient_checkpointing=self.args.gradient_checkpointing,
            enable_memory_efficient_attention=self.args.memory_efficient_attention,
            profile_mode=self.args.profile_mode
        )

        # Memory optimizer
        self.memory_optimizer = A100MemoryOptimizer(
            enable_memory_pool=True,
            pool_size_gb=self.args.memory_pool_size / 1024 if self.args.memory_pool_size else None,
            enable_gradient_checkpointing=self.args.gradient_checkpointing,
            checkpoint_policy=self.args.checkpoint_policy,
            memory_threshold_gb=self.args.memory_threshold,
            enable_cpu_offload=self.args.cpu_offload,
            profile_memory=self.args.profile_memory
        )

        # Structured sparsity optimizer
        if self.args.use_structured_sparsity:
            self.sparsity_optimizer = StructuredSparsityOptimizer(sparsity_level=0.5)

        # NVLink optimizer for multi-GPU
        if torch.cuda.device_count() > 1 and self.args.nvlink_optimization:
            self.nvlink_optimizer = NVLinkOptimizer(
                enable_nvlink_optimization=True,
                enable_gpu_direct=self.args.gpu_direct,
                hierarchical_allreduce=self.args.hierarchical_allreduce,
                fusion_buffer_size_mb=self.args.fusion_buffer_size,
                profile_communication=self.args.profile_communication
            )

        # MIG manager if requested
        if self.args.mig_instances > 0:
            self.mig_manager = MIGManager()
            if self.mig_manager.mig_enabled:
                profiles = self.mig_manager.get_mig_profiles()
                logger.info(f"Available MIG profiles: {profiles}")

        # Gradient accumulator
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=self.args.gradient_accumulation_steps,
            use_gradient_scaling=self.args.mixed_precision != "none",
            max_grad_norm=self.args.max_grad_norm
        )

    def _optimize_model(self):
        """Apply A100 optimizations to the model."""
        logger.info("Applying A100 optimizations to model...")

        # Replace attention layers with FlashAttention v3
        if self.args.use_flash_attention:
            self._replace_attention_with_flash()

        # Apply memory optimizations
        self.model = self.memory_optimizer.optimize_model_memory(self.model)

        # Apply A100-specific optimizations (torch.compile, etc.)
        self.model = self.a100_optimizer.optimize_model(self.model, compile=self.args.enable_torch_compile)

        # Apply structured sparsity if requested
        if self.args.use_structured_sparsity and hasattr(self, 'sparsity_optimizer'):
            sparsity_info = self.sparsity_optimizer.sparsify_model(
                self.model,
                target_layers=self.args.sparsity_target_layers.split(',') if self.args.sparsity_target_layers else None
            )
            logger.info(f"Applied structured sparsity to {len(sparsity_info)} layers")

        return self.model

    def _replace_attention_with_flash(self):
        """Replace standard attention with FlashAttention v3."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Get attention parameters
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                dropout = module.dropout.p if hasattr(module, 'dropout') else 0.0

                # Create FlashAttention replacement
                flash_attn = FlashAttentionV3(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    causal=self.args.causal_attention,
                    block_size=self.args.flash_block_size,
                    enable_sequence_parallel=self.args.sequence_parallel,
                    use_tensor_cores=True,
                    attention_type=self.args.attention_type,
                    window_size=self.args.attention_window_size,
                    profile_mode=self.args.profile_mode
                )

                # Replace module
                parent_module = self.model
                module_names = name.split('.')
                for n in module_names[:-1]:
                    parent_module = getattr(parent_module, n)
                setattr(parent_module, module_names[-1], flash_attn)

                logger.info(f"Replaced attention layer {name} with FlashAttention v3")

    def _setup_data_loaders(self):
        """Setup optimized data loaders."""
        # Create base data loaders
        if self.args.streaming_data:
            train_loader, val_loader = create_streaming_dataloaders(
                dataset_path=self.args.dataset,
                tokenizer=self.tokenizer,
                batch_size=self.args.batch_size,
                max_seq_length=self.args.max_seq_length,
                num_workers=self.args.num_workers
            )
        else:
            train_loader, val_loader = create_dataloaders(
                dataset_path=self.args.dataset,
                tokenizer=self.tokenizer,
                batch_size=self.args.batch_size,
                max_seq_length=self.args.max_seq_length,
                num_workers=self.args.num_workers
            )

        # Optimize for A100 memory bandwidth
        train_loader = self.a100_optimizer.optimize_dataloader(
            train_loader,
            prefetch_factor=self.args.prefetch_factor,
            pin_memory=True
        )
        val_loader = self.a100_optimizer.optimize_dataloader(
            val_loader,
            prefetch_factor=self.args.prefetch_factor,
            pin_memory=True
        )

        # Apply NVLink optimization if available
        if hasattr(self, 'nvlink_optimizer'):
            topology_loader = TopologyAwareDataLoader(train_loader, self.nvlink_optimizer)
            train_loader = topology_loader.optimize_for_nvlink()

        return train_loader, val_loader

    def _setup_optimizer(self):
        """Setup optimizer with A100-aware settings."""
        # Filter parameters for optimization
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Choose optimizer
        if self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=self.args.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=self.args.weight_decay,
                eps=1e-8
            )
        elif self.args.optimizer == 'lion':
            # Lion optimizer (if available)
            try:
                from lion_pytorch import Lion
                optimizer = Lion(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            except ImportError:
                logger.warning("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate)
        else:
            optimizer = torch.optim.SGD(
                params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )

        # Setup scheduler
        total_steps = len(self.train_loader) * self.args.num_epochs // self.args.gradient_accumulation_steps

        if self.args.scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=self.args.min_learning_rate)
        elif self.args.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = None

        return optimizer, scheduler

    def _setup_loss(self):
        """Setup loss function."""
        if self.args.use_composite_loss:
            # Composite loss with multiple objectives
            loss_config = {
                'cross_entropy': {'type': 'cross_entropy', 'weight': 1.0},
            }

            if self.args.use_focal_loss:
                loss_config['focal'] = {'type': 'focal', 'alpha': 1.0, 'gamma': 2.0, 'weight': 0.1}

            if self.args.use_contrastive_loss:
                loss_config['contrastive'] = {'type': 'contrastive', 'temperature': 0.07, 'weight': 0.1}

            criterion = CompositeLoss(loss_config)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        return criterion

    def _setup_distributed(self):
        """Setup distributed training with NVLink optimization."""
        if not dist.is_initialized() and torch.cuda.device_count() > 1:
            # Initialize distributed
            dist.init_process_group(backend='nccl')

            # Optimize model for distributed training
            if hasattr(self, 'nvlink_optimizer'):
                self.model = self.nvlink_optimizer.optimize_ddp_model(
                    self.model,
                    device_ids=[self.args.local_rank] if self.args.local_rank >= 0 else None,
                    broadcast_buffers=True
                )
            else:
                self.model = DDP(self.model, device_ids=[self.args.local_rank] if self.args.local_rank >= 0 else None)

            # Configure GPU Direct if available
            if self.args.gpu_direct:
                gpu_direct_optimizer = GPUDirectOptimizer()
                gpu_direct_optimizer.configure_gpu_direct()

    def train(self):
        """Main training loop with A100 optimizations."""
        logger.info(f"Starting A100-optimized training for {self.args.num_epochs} epochs")
        logger.info(f"Batch size: {self.args.batch_size}, Gradient accumulation: {self.args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")

        best_val_loss = float('inf')
        training_history = []

        for epoch in range(self.args.num_epochs):
            epoch_start = time.time()

            # Training phase
            train_metrics = self._train_epoch(epoch)

            # Validation phase
            val_metrics = self._validate_epoch(epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Log epoch metrics
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Time: {epoch_time:.1f}s")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, val_metrics['loss'], is_best=True)

            # Regular checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self._save_checkpoint(epoch, val_metrics['loss'], is_best=False)

            # Profile and log A100 metrics
            if self.args.profile_mode and (epoch + 1) % self.args.profile_every == 0:
                self._profile_and_log_metrics()

            # Update training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                **train_metrics,
                **val_metrics
            })

        # Final profiling and benchmarking
        if self.args.run_final_benchmark:
            self._run_final_benchmarks()

        # Save training history
        self._save_training_history(training_history)

        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")

        return training_history

    def _train_epoch(self, epoch):
        """Train for one epoch with A100 optimizations."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", disable=not self.args.show_progress)

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass with mixed precision
            with self.a100_optimizer.amp_context():
                with self.memory_optimizer.memory_efficient_forward(clear_cache=(batch_idx % 100 == 0)):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels')
                    )

                    loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        batch['labels'].view(-1)
                    )

            # Gradient accumulation
            metrics = self.gradient_accumulator.accumulate_gradients(
                loss=loss,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )

            total_loss += metrics['loss']
            num_batches += 1

            # Update progress bar
            if self.gradient_accumulator.should_update_weights():
                progress_bar.set_postfix({
                    'loss': metrics['loss'],
                    'grad_norm': metrics.get('grad_norm', 0.0),
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # Periodic memory profiling
            if self.args.profile_memory and batch_idx % self.args.profile_every == 0:
                mem_stats = self.memory_optimizer.get_memory_summary()
                logger.info(f"Memory - Allocated: {mem_stats['allocated_gb']:.2f}GB, "
                           f"Reserved: {mem_stats['reserved_gb']:.2f}GB, "
                           f"Free: {mem_stats['free_gb']:.2f}GB")

        return {
            'loss': total_loss / num_batches,
            'num_batches': num_batches
        }

    def _validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} Validation", disable=not self.args.show_progress):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Forward pass
                with self.a100_optimizer.amp_context():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels')
                    )

                    loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        batch['labels'].view(-1)
                    )

                total_loss += loss.item()
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'perplexity': np.exp(total_loss / num_batches)
        }

    def _profile_and_log_metrics(self):
        """Profile and log A100-specific metrics."""
        logger.info("=" * 50)
        logger.info("A100 Performance Metrics:")

        # A100 optimizer metrics
        self.a100_optimizer.log_metrics()

        # Memory metrics
        mem_summary = self.memory_optimizer.get_memory_summary()
        logger.info(f"Memory Usage - Peak: {mem_summary['peak_allocated_gb']:.2f}GB, "
                   f"Current: {mem_summary['allocated_gb']:.2f}GB")

        # NVLink metrics if available
        if hasattr(self, 'nvlink_optimizer'):
            nvlink_stats = self.nvlink_optimizer.get_communication_stats()
            logger.info(f"NVLink - Bandwidth: {nvlink_stats['nvlink_bandwidth_gbps']:.1f} Gbps, "
                       f"Efficiency: {nvlink_stats['communication_efficiency']:.1%}")

        logger.info("=" * 50)

    def _run_final_benchmarks(self):
        """Run comprehensive benchmarks at the end of training."""
        logger.info("Running final A100 benchmarks...")

        # Benchmark model optimizations
        input_shape = (self.args.batch_size, self.args.max_seq_length, self.config.hidden_size)
        optimization_results = benchmark_a100_optimizations(self.model, input_shape, num_iterations=50)

        logger.info("Optimization Benchmarks:")
        for name, results in optimization_results.items():
            if isinstance(results, dict):
                logger.info(f"  {name}: {results['speedup']:.2f}x speedup, "
                           f"{results['throughput']:.1f} iter/s")

        # Benchmark FlashAttention
        if self.args.use_flash_attention:
            flash_speedup = benchmark_flash_attention(
                seq_length=self.args.max_seq_length,
                embed_dim=self.config.hidden_size,
                num_heads=self.config.num_attention_heads
            )
            logger.info(f"FlashAttention v3 Speedup: {flash_speedup:.2f}x")

        # Benchmark NVLink communication
        if hasattr(self, 'nvlink_optimizer') and dist.is_initialized():
            nvlink_results = benchmark_nvlink_communication([1, 10, 100, 500])
            logger.info(f"NVLink Average Bandwidth: {nvlink_results['summary']['avg_bandwidth_gbps']:.1f} Gbps")

    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'args': vars(self.args),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }

        # Save checkpoint
        checkpoint_dir = Path(self.args.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            checkpoint_path = checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _save_training_history(self, history):
        """Save training history to JSON."""
        history_path = Path(self.args.output_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='A100-Optimized Training Script')

    # Basic arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, default='wikitext', help='Dataset name or path')
    parser.add_argument('--output-dir', type=str, default='outputs/a100_model', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--max-seq-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing')

    # A100-specific optimizations
    parser.add_argument('--enable-a100-optimizations', action='store_true', help='Enable all A100 optimizations')
    parser.add_argument('--enable-all-a100-features', action='store_true', help='Enable all A100 features')
    parser.add_argument('--enable-tf32', action='store_true', help='Enable TensorFloat-32')
    parser.add_argument('--mixed-precision', type=str, default='none', choices=['none', 'fp16', 'bf16'],
                       help='Mixed precision training')
    parser.add_argument('--use-cuda-graphs', action='store_true', help='Use CUDA graphs')
    parser.add_argument('--enable-torch-compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--compile-mode', type=str, default='max-autotune',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode')
    parser.add_argument('--memory-pool-size', type=int, default=None, help='Memory pool size in MB')

    # FlashAttention arguments
    parser.add_argument('--use-flash-attention', action='store_true', help='Use FlashAttention v3')
    parser.add_argument('--flash-block-size', type=int, default=128, help='FlashAttention block size')
    parser.add_argument('--causal-attention', action='store_true', help='Use causal attention')
    parser.add_argument('--attention-type', type=str, default='standard',
                       choices=['standard', 'block_sparse', 'sliding_window'],
                       help='Attention type')
    parser.add_argument('--attention-window-size', type=int, default=256, help='Sliding window size')
    parser.add_argument('--sequence-parallel', action='store_true', help='Enable sequence parallelism')

    # Memory optimization arguments
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--checkpoint-policy', type=str, default='selective',
                       choices=['none', 'all', 'selective', 'adaptive'],
                       help='Gradient checkpointing policy')
    parser.add_argument('--memory-threshold', type=float, default=30.0, help='Memory threshold in GB')
    parser.add_argument('--cpu-offload', action='store_true', help='Enable CPU offloading')
    parser.add_argument('--memory-efficient-attention', action='store_true', help='Use memory-efficient attention')

    # Structured sparsity arguments
    parser.add_argument('--use-structured-sparsity', action='store_true', help='Use 2:4 structured sparsity')
    parser.add_argument('--sparsity-target-layers', type=str, default=None,
                       help='Comma-separated list of layers to sparsify')

    # NVLink and distributed arguments
    parser.add_argument('--nvlink-optimization', action='store_true', help='Enable NVLink optimization')
    parser.add_argument('--dgx-a100', action='store_true', help='Use DGX A100 optimizations')
    parser.add_argument('--gpu-direct', action='store_true', help='Enable GPU Direct RDMA')
    parser.add_argument('--hierarchical-allreduce', action='store_true', help='Use hierarchical AllReduce')
    parser.add_argument('--fusion-buffer-size', type=int, default=64, help='Fusion buffer size in MB')
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')

    # MIG arguments
    parser.add_argument('--mig-instances', type=int, default=0, help='Number of MIG instances')

    # Other optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'lion'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'linear', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--min-learning-rate', type=float, default=1e-6, help='Minimum learning rate')

    # Data loading arguments
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor for data loading')
    parser.add_argument('--streaming-data', action='store_true', help='Use streaming data loading')

    # Evaluation arguments
    parser.add_argument('--eval-during-training', action='store_true', help='Evaluate during training')
    parser.add_argument('--eval-metrics', type=str, default='perplexity', help='Comma-separated evaluation metrics')

    # Loss function arguments
    parser.add_argument('--use-composite-loss', action='store_true', help='Use composite loss')
    parser.add_argument('--use-focal-loss', action='store_true', help='Use focal loss')
    parser.add_argument('--use-contrastive-loss', action='store_true', help='Use contrastive loss')

    # Profiling and debugging
    parser.add_argument('--profile-mode', action='store_true', help='Enable profiling mode')
    parser.add_argument('--profile-memory', action='store_true', help='Profile memory usage')
    parser.add_argument('--profile-communication', action='store_true', help='Profile communication')
    parser.add_argument('--profile-every', type=int, default=100, help='Profile every N steps')
    parser.add_argument('--show-progress', action='store_true', help='Show progress bars')
    parser.add_argument('--run-final-benchmark', action='store_true', help='Run final benchmarks')

    # Checkpointing
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Handle convenience flags
    if args.enable_a100_optimizations:
        args.enable_tf32 = True
        args.mixed_precision = 'bf16'
        args.use_cuda_graphs = True
        args.memory_efficient_attention = True

    if args.enable_all_a100_features:
        args.enable_tf32 = True
        args.mixed_precision = 'bf16'
        args.use_cuda_graphs = True
        args.enable_torch_compile = True
        args.use_flash_attention = True
        args.gradient_checkpointing = True
        args.memory_efficient_attention = True
        args.use_structured_sparsity = True
        if torch.cuda.device_count() > 1:
            args.nvlink_optimization = True

    if args.dgx_a100:
        args.nvlink_optimization = True
        args.gpu_direct = True
        args.hierarchical_allreduce = True
        args.fusion_buffer_size = 128

    return args


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_properties(0).name}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create model configuration
    config = EnhancedMoEConfig(**config_dict)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config_dict.get('tokenizer_name', 'gpt2'),
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    logger.info("Creating model...")
    model = EnhancedMoEModel(config)
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = A100OptimizedTrainer(args, config, model, tokenizer, device)

    # Start training
    try:
        training_history = trainer.train()
        logger.info("Training completed successfully!")

        # Print final summary
        logger.info("=" * 50)
        logger.info("Training Summary:")
        logger.info(f"Best validation loss: {min(h['val_loss'] for h in training_history):.4f}")
        logger.info(f"Final learning rate: {training_history[-1]['learning_rate']:.2e}")
        logger.info(f"Total training time: {sum(h['epoch_time'] for h in training_history) / 3600:.2f} hours")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()