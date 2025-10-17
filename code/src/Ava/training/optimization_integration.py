"""
Optimization Integration Module

Provides a unified interface to integrate all training optimizations
into the main training script. This ensures all optimizations are
automatically applied when training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class OptimizedTrainingSetup:
    """
    Unified setup for all training optimizations.

    This class orchestrates all optimization components to ensure
    they're properly initialized and integrated.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        enable_all: bool = True,
        verbose: bool = True
    ):
        """
        Initialize optimized training setup.

        Args:
            config: Training configuration dictionary
            enable_all: Enable all optimizations (recommended)
            verbose: Print optimization status
        """
        self.config = config
        self.enable_all = enable_all
        self.verbose = verbose

        # Components (initialized on-demand)
        self.hw_optimizer = None
        self.mp_manager = None
        self.throughput_tracker = None
        self.memory_profiler = None
        self.training_monitor = None

        if verbose:
            logger.info("=" * 70)
            logger.info("🚀 OPTIMIZED TRAINING SETUP")
            logger.info("=" * 70)

    def setup_hardware(self):
        """Setup hardware-specific optimizations."""
        try:
            from Ava.optimization.hardware_optimizations import auto_optimize_hardware  # type: ignore[import-not-found]
            self.hw_optimizer = auto_optimize_hardware()
        except ImportError:
            logger.warning("hardware_optimizations module not found, skipping hardware setup")
            self.hw_optimizer = None
            return

        if self.verbose:
            logger.info("\n[1/7] Setting up hardware optimizations...")

        recommended = self.hw_optimizer.get_recommended_settings()

        # Update config with recommendations if not already set
        if 'dtype' not in self.config:
            self.config['dtype'] = recommended.get('recommended_dtype', 'bfloat16')

        if 'batch_size' not in self.config:
            # Parse recommended batch size (e.g., "medium (16-32)")
            batch_rec = recommended.get('recommended_batch_size', '32')
            if isinstance(batch_rec, str) and '(' in batch_rec:
                # Extract middle value from range
                import re
                match = re.search(r'\((\d+)-(\d+)\)', batch_rec)
                if match:
                    low, high = int(match.group(1)), int(match.group(2))
                    self.config['batch_size'] = (low + high) // 2

        if self.verbose:
            logger.info("✓ Hardware optimizations applied")

        return self.hw_optimizer

    def setup_mixed_precision(self):
        """Setup mixed precision training."""
        try:
            from Ava.optimization.gradient_optimizations import MixedPrecisionManager  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("gradient_optimizations module not found, skipping mixed precision")
            return None

        if self.verbose:
            logger.info("\n[2/7] Setting up mixed precision training...")

        # Auto-detect dtype
        dtype = None
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

        self.mp_manager = MixedPrecisionManager(
            enabled=self.config.get('mixed_precision', True),
            dtype=dtype,
            dynamic_loss_scale=True
        )

        if self.verbose:
            logger.info(f"✓ Mixed precision enabled: {self.mp_manager.dtype}")

        return self.mp_manager

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply model optimizations."""
        try:
            from Ava.optimization.compilation_optimizations import optimize_for_training  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("compilation_optimizations module not found, skipping model optimization")
            return model

        if self.verbose:
            logger.info("\n[3/7] Optimizing model...")

        # Compile model with torch.compile
        if self.config.get('compile_model', True):
            try:
                model = optimize_for_training(
                    model,
                    example_inputs=None,
                    compile_mode=self.config.get('compile_mode', 'reduce-overhead')
                )
                if self.verbose:
                    logger.info("✓ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}, continuing without compilation")

        return model

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimized optimizer."""
        try:
            from Ava.optimization.fused_optimizers import create_optimizer  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("fused_optimizers module not found, using default AdamW")
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.get('learning_rate', 3e-4),
                weight_decay=self.config.get('weight_decay', 0.01)
            )

        if self.verbose:
            logger.info("\n[4/7] Creating optimized optimizer...")

        optimizer_type = self.config.get('optimizer_type', 'fused_adam')
        lr = self.config.get('learning_rate', 3e-4)
        weight_decay = self.config.get('weight_decay', 0.01)

        optimizer = create_optimizer(
            model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay
        )

        if self.verbose:
            logger.info(f"✓ Optimizer created: {optimizer_type}")

        return optimizer

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> DataLoader:
        """Create optimized dataloader."""
        try:
            from Ava.data.optimized_dataloader import create_production_dataloader  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("optimized_dataloader module not found, using default DataLoader")
            batch_size = batch_size or self.config.get('batch_size', 32)
            num_workers = num_workers or self.config.get('num_workers', 4)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers or 4,  # type: ignore[arg-type]
                shuffle=kwargs.get('shuffle', True),
                pin_memory=kwargs.get('pin_memory', True)
            )

        if self.verbose:
            logger.info("\n[5/7] Creating optimized dataloader...")

        batch_size = batch_size or self.config.get('batch_size', 32)
        num_workers = num_workers or self.config.get('num_workers', 4)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataloader = create_production_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            use_packing=self.config.get('use_sequence_packing', True),
            pack_block_size=self.config.get('max_seq_length', 2048),
            use_dynamic_batching=self.config.get('use_dynamic_batching', False),
            **kwargs
        )

        if self.verbose:
            logger.info(f"✓ Dataloader created: batch_size={batch_size}, workers={num_workers}")

        return dataloader

    def setup_monitoring(self, model: Optional[nn.Module] = None):
        """Setup training monitoring."""
        try:
            from Ava.training.profiling_tools import TrainingMonitor, ThroughputTracker, MemoryProfiler  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("profiling_tools module not found, skipping monitoring setup")
            self.throughput_tracker = None
            self.memory_profiler = None
            self.training_monitor = None
            return

        if self.verbose:
            logger.info("\n[6/7] Setting up monitoring and profiling...")

        # Throughput tracker
        self.throughput_tracker = ThroughputTracker(
            window_size=100,
            log_interval=self.config.get('log_interval', 10)
        )

        # Memory profiler
        self.memory_profiler = MemoryProfiler(
            check_interval=100
        )

        # Comprehensive training monitor
        if model is not None:
            self.training_monitor = TrainingMonitor(
                model=model,
                log_interval=self.config.get('log_interval', 10),
                output_dir=self.config.get('output_dir', './monitoring')
            )

        if self.verbose:
            logger.info("✓ Monitoring and profiling configured")

        return self.training_monitor

    def setup_gradient_optimization(self):
        """Setup gradient optimization components."""
        try:
            from Ava.optimization.gradient_optimizations import (  # type: ignore[import-not-found]
                AdaptiveGradientClipper,
                GradientNoiseInjector
            )
        except ImportError:
            logger.warning("gradient_optimizations module not found, skipping gradient optimization")
            return None

        if self.verbose:
            logger.info("\n[7/7] Setting up gradient optimizations...")

        # Adaptive gradient clipper
        grad_clipper = AdaptiveGradientClipper(
            clip_type=self.config.get('grad_clip_type', 'adaptive'),
            base_clip_value=self.config.get('max_grad_norm', 1.0)
        )

        # Gradient noise (optional, for generalization)
        grad_noise = None
        if self.config.get('gradient_noise', False):
            grad_noise = GradientNoiseInjector(
                initial_noise_scale=self.config.get('gradient_noise_scale', 1e-3)
            )

        if self.verbose:
            logger.info("✓ Gradient optimizations configured")
            logger.info("=" * 70)
            logger.info("🎯 All optimizations ready!\n")

        return grad_clipper, grad_noise

    def create_complete_setup(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """
        Create complete optimized training setup.

        Returns a dictionary with all components ready to use.

        Args:
            model: The model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset

        Returns:
            Dictionary containing:
                - model: Optimized model
                - optimizer: Optimized optimizer
                - train_loader: Optimized training dataloader
                - val_loader: Optional validation dataloader
                - mp_manager: Mixed precision manager
                - grad_clipper: Gradient clipper
                - monitor: Training monitor
                - config: Updated configuration
        """
        # 1. Hardware
        self.setup_hardware()

        # 2. Mixed precision
        mp_manager = self.setup_mixed_precision()

        # 3. Optimize model
        model = self.optimize_model(model)

        # 4. Create optimizer
        optimizer = self.create_optimizer(model)

        # 5. Create dataloaders
        train_loader = self.create_dataloader(train_dataset)
        val_loader = None
        if val_dataset is not None:
            val_loader = self.create_dataloader(
                val_dataset,
                shuffle=False
            )

        # 6. Setup monitoring
        monitor = self.setup_monitoring(model)

        # 7. Gradient optimization
        grad_result = self.setup_gradient_optimization()
        grad_clipper, grad_noise = grad_result if grad_result else (None, None)

        return {
            'model': model,
            'optimizer': optimizer,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'mp_manager': mp_manager,
            'grad_clipper': grad_clipper,
            'grad_noise': grad_noise,
            'monitor': monitor,
            'throughput_tracker': self.throughput_tracker,
            'memory_profiler': self.memory_profiler,
            'config': self.config
        }


def create_optimized_training_loop(
    setup_dict: Dict[str, Any],
    num_epochs: int,
    checkpoint_dir: str = './checkpoints',
    log_interval: int = 10
):
    """
    Create an optimized training loop with all optimizations integrated.

    Args:
        setup_dict: Dictionary from create_complete_setup()
        num_epochs: Number of training epochs
        checkpoint_dir: Directory for checkpoints
        log_interval: Logging interval

    Returns:
        Training loop function
    """
    model = setup_dict['model']
    optimizer = setup_dict['optimizer']
    train_loader = setup_dict['train_loader']
    mp_manager = setup_dict['mp_manager']
    grad_clipper = setup_dict['grad_clipper']
    monitor = setup_dict['monitor']

    def training_step(batch, step):
        """Single optimized training step."""
        # Start monitoring
        if monitor:
            monitor.start_step()

        # Forward pass with mixed precision
        with mp_manager.autocast():
            outputs = model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs

        # Backward pass
        loss_scaled = mp_manager.scale_loss(loss)
        loss_scaled.backward()

        # Gradient clipping
        clip_stats = grad_clipper.clip_gradients(model.named_parameters(), named=True)

        # Optimizer step
        opt_metrics = mp_manager.step_optimizer(optimizer, max_grad_norm=None)  # Already clipped

        # Zero gradients
        optimizer.zero_grad()

        # End monitoring
        if monitor:
            # Safely get batch dimensions
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids')
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                seq_len = input_ids.shape[1] if input_ids is not None else 512
            else:
                batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1
                seq_len = 512

            stats = monitor.end_step(
                batch_size=batch_size,
                seq_len=seq_len,
                loss=loss.item(),
                grad_norm=clip_stats.get('grad_norm', 0.0)
            )

            # Log periodically
            if step % log_interval == 0:
                logger.info(
                    f"Step {step}: loss={loss.item():.4f}, "
                    f"grad_norm={clip_stats.get('grad_norm', 0):.3f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        return loss.item(), clip_stats

    def training_loop():
        """Main training loop."""
        model.train()
        global_step = 0

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*70}")

            for batch_idx, batch in enumerate(train_loader):
                loss, stats = training_step(batch, global_step)
                global_step += 1

            # Epoch complete
            if monitor:
                summary = monitor.get_summary()
                logger.info(f"\nEpoch {epoch + 1} Summary:")
                logger.info(f"  Throughput: {summary['throughput'].get('tokens_per_sec', 0):.0f} tokens/s")
                logger.info(f"  MFU: {summary['throughput'].get('mfu', 0):.2%}")
                logger.info(f"  Memory: {summary['memory'].get('current_mb', 0):.1f} MB")

    return training_loop


# Convenience function for quick setup
def quick_optimize(
    model: nn.Module,
    train_dataset: Dataset,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick optimization setup with sensible defaults.

    Usage:
        setup = quick_optimize(model, dataset)
        training_loop = create_optimized_training_loop(setup, num_epochs=3)
        training_loop()

    Args:
        model: Model to optimize
        train_dataset: Training dataset
        config: Optional configuration dictionary
        **kwargs: Additional config overrides

    Returns:
        Complete setup dictionary
    """
    if config is None:
        config = {}

    # Apply kwargs overrides
    config.update(kwargs)

    # Create optimizer
    optimizer_setup = OptimizedTrainingSetup(config, enable_all=True, verbose=True)

    # Create complete setup
    return optimizer_setup.create_complete_setup(model, train_dataset)
