"""
Optimization Integration Patch for train.py

This module provides functions to integrate all new optimizations
into the existing train.py without breaking existing functionality.

Add this import at the top of train.py:
    from train_optimized_patch import apply_all_optimizations, OptimizedTrainingWrapper

Then wrap your training setup with:
    training_components = apply_all_optimizations(config, model, dataset)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def apply_all_optimizations(
    config: Dict[str, Any],
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None
) -> Dict[str, Any]:
    """
    Apply all training optimizations to existing training setup.

    This function is designed to be inserted into existing train.py
    with minimal code changes.

    Args:
        config: Training configuration dictionary
        model: Model to optimize
        train_dataset: Training dataset
        val_dataset: Optional validation dataset

    Returns:
        Dictionary with optimized components
    """
    logger.info("\n" + "=" * 80)
    logger.info("🚀 APPLYING ALL TRAINING OPTIMIZATIONS")
    logger.info("=" * 80 + "\n")

    try:
        from Ava.training.optimization_integration import OptimizedTrainingSetup

        # Create optimization setup
        opt_setup = OptimizedTrainingSetup(
            config=config,
            enable_all=True,
            verbose=True
        )

        # Get complete setup
        components = opt_setup.create_complete_setup(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

        logger.info("\n✅ All optimizations applied successfully!\n")

        return components

    except Exception as e:
        logger.error(f"Failed to apply optimizations: {e}")
        logger.warning("Falling back to standard training setup...")

        # Fallback to basic setup
        return {
            'model': model,
            'optimizer': None,  # Will need to create
            'train_loader': None,  # Will need to create
            'val_loader': None,
            'mp_manager': None,
            'grad_clipper': None,
            'monitor': None,
            'config': config
        }


class OptimizedTrainingWrapper:
    """
    Wrapper class that can be used to enhance existing training loops.

    Usage in train.py:
        wrapper = OptimizedTrainingWrapper(config)
        wrapper.setup(model, dataset)

        # In training loop:
        with wrapper.training_step(batch, step) as ctx:
            outputs = model(**batch)
            loss = outputs['loss']
            ctx.backward(loss)
            ctx.optimizer_step(optimizer)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize wrapper with configuration."""
        self.config = config
        self.components = {}

        # Initialize components lazily
        self._mp_manager = None
        self._grad_clipper = None
        self._monitor = None
        self._hw_optimizer = None

    def setup(self, model: nn.Module, dataset: Dataset):
        """Setup all optimization components."""
        logger.info("Setting up optimizations...")

        # Hardware optimizations
        self._setup_hardware()

        # Mixed precision
        self._setup_mixed_precision()

        # Model optimization
        model = self._optimize_model(model)

        # Monitoring
        self._setup_monitoring(model)

        # Gradient optimization
        self._setup_gradient_optimization()

        self.components['model'] = model

        logger.info("✓ Optimization setup complete")

        return model

    def _setup_hardware(self):
        """Setup hardware optimizations."""
        try:
            from Ava.optimization.hardware_optimizations import auto_optimize_hardware
            self._hw_optimizer = auto_optimize_hardware()
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")

    def _setup_mixed_precision(self):
        """Setup mixed precision."""
        try:
            from Ava.optimization.gradient_optimizations import MixedPrecisionManager

            dtype = None
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16

            self._mp_manager = MixedPrecisionManager(
                enabled=self.config.get('mixed_precision', True),
                dtype=dtype
            )
        except Exception as e:
            logger.warning(f"Mixed precision setup failed: {e}")

    def _optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model with compilation."""
        try:
            if self.config.get('compile_model', True):
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("✓ Model compiled")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")

        return model

    def _setup_monitoring(self, model: nn.Module):
        """Setup monitoring."""
        try:
            from Ava.training.profiling_tools import TrainingMonitor

            self._monitor = TrainingMonitor(
                model=model,
                log_interval=self.config.get('log_interval', 10)
            )
        except Exception as e:
            logger.warning(f"Monitoring setup failed: {e}")

    def _setup_gradient_optimization(self):
        """Setup gradient optimization."""
        try:
            from Ava.optimization.gradient_optimizations import AdaptiveGradientClipper

            self._grad_clipper = AdaptiveGradientClipper(
                clip_type='adaptive',
                base_clip_value=self.config.get('max_grad_norm', 1.0)
            )
        except Exception as e:
            logger.warning(f"Gradient optimization setup failed: {e}")

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimized optimizer."""
        try:
            from Ava.optimization.fused_optimizers import create_optimizer

            return create_optimizer(
                model,
                optimizer_type=self.config.get('optimizer_type', 'fused_adam'),
                lr=self.config.get('learning_rate', 3e-4),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
        except Exception as e:
            logger.warning(f"Fused optimizer creation failed: {e}, using standard Adam")
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.get('learning_rate', 3e-4),
                weight_decay=self.config.get('weight_decay', 0.01)
            )

    def create_dataloader(self, dataset: Dataset, **kwargs) -> torch.utils.data.DataLoader:
        """Create optimized dataloader."""
        try:
            from Ava.data.optimized_dataloader import create_production_dataloader

            return create_production_dataloader(
                dataset=dataset,
                batch_size=self.config.get('batch_size', 32),
                num_workers=self.config.get('num_workers', 4),
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Optimized dataloader creation failed: {e}, using standard DataLoader")
            from torch.utils.data import DataLoader

            return DataLoader(
                dataset,
                batch_size=self.config.get('batch_size', 32),
                num_workers=self.config.get('num_workers', 4),
                pin_memory=torch.cuda.is_available(),
                **kwargs
            )

    def forward_pass(self, model: nn.Module, batch: Dict[str, torch.Tensor]):
        """Optimized forward pass with mixed precision."""
        if self._mp_manager is not None:
            with self._mp_manager.autocast():
                return model(**batch)
        else:
            return model(**batch)

    def backward_pass(self, loss: torch.Tensor) -> torch.Tensor:
        """Optimized backward pass with loss scaling."""
        if self._mp_manager is not None:
            scaled_loss = self._mp_manager.scale_loss(loss)
            scaled_loss.backward()
            return scaled_loss
        else:
            loss.backward()
            return loss

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module
    ) -> Dict[str, float]:
        """Optimized optimizer step with gradient clipping."""
        metrics = {}

        # Gradient clipping
        if self._grad_clipper is not None:
            clip_stats = self._grad_clipper.clip_gradients(
                model.named_parameters(),
                named=True
            )
            metrics.update(clip_stats)

        # Optimizer step with mixed precision
        if self._mp_manager is not None:
            opt_metrics = self._mp_manager.step_optimizer(optimizer)
            metrics.update(opt_metrics)
        else:
            optimizer.step()

        return metrics

    def log_step(self, step: int, loss: float, metrics: Dict[str, float]):
        """Log training step."""
        if self._monitor is not None and step % self.config.get('log_interval', 10) == 0:
            # Monitoring is handled internally
            pass


def enhance_existing_training_loop(
    original_train_fn,
    config: Dict[str, Any],
    model: nn.Module,
    dataset: Dataset
):
    """
    Decorator to enhance existing training function with optimizations.

    Usage:
        @enhance_existing_training_loop(config, model, dataset)
        def train_epoch(model, dataloader, optimizer):
            # existing training code
            pass
    """

    def wrapper(*args, **kwargs):
        # Setup optimizations
        opt_wrapper = OptimizedTrainingWrapper(config)
        optimized_model = opt_wrapper.setup(model, dataset)

        # Replace model with optimized version
        if len(args) > 0 and isinstance(args[0], nn.Module):
            args = (optimized_model,) + args[1:]

        # Call original function
        return original_train_fn(*args, **kwargs)

    return wrapper


# Quick integration helpers

def quick_setup_optimizations(config: Dict[str, Any]) -> OptimizedTrainingWrapper:
    """
    Quick setup for existing train.py.

    Add to train.py after config is loaded:
        from train_optimized_patch import quick_setup_optimizations
        opt_wrapper = quick_setup_optimizations(config)
        model = opt_wrapper.setup(model, dataset)
        optimizer = opt_wrapper.create_optimizer(model)
        dataloader = opt_wrapper.create_dataloader(dataset)
    """
    return OptimizedTrainingWrapper(config)


def inject_optimizations_into_step(
    wrapper: OptimizedTrainingWrapper,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    step: int
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Inject optimizations into a single training step.

    Add to existing training loop:
        from train_optimized_patch import inject_optimizations_into_step

        # In training loop:
        loss, metrics = inject_optimizations_into_step(
            opt_wrapper, model, optimizer, batch, step
        )
    """
    # Forward
    outputs = wrapper.forward_pass(model, batch)
    loss = outputs['loss'] if isinstance(outputs, dict) else outputs

    # Backward
    wrapper.backward_pass(loss)

    # Optimizer step with clipping
    metrics = wrapper.optimizer_step(optimizer, model)

    # Zero gradients
    optimizer.zero_grad()

    # Log
    wrapper.log_step(step, loss.item(), metrics)

    return loss, metrics


# === EXAMPLE INTEGRATION INTO EXISTING TRAIN.PY ===

def example_integration():
    """
    Example showing how to integrate into existing train.py.

    BEFORE:
    -------
    def main():
        config = load_config()
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters())
        dataloader = DataLoader(dataset)

        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    AFTER (Minimal changes):
    ------------------------
    def main():
        config = load_config()
        model = create_model()

        # ADD: Setup optimizations
        from train_optimized_patch import quick_setup_optimizations
        opt_wrapper = quick_setup_optimizations(config)

        # REPLACE: Use optimized components
        model = opt_wrapper.setup(model, dataset)
        optimizer = opt_wrapper.create_optimizer(model)
        dataloader = opt_wrapper.create_dataloader(dataset)

        for batch in dataloader:
            # REPLACE: Use optimized step
            outputs = opt_wrapper.forward_pass(model, batch)
            loss = outputs['loss']
            opt_wrapper.backward_pass(loss)
            opt_wrapper.optimizer_step(optimizer, model)
            optimizer.zero_grad()

    Or use the injection helper:
    -----------------------------
    from train_optimized_patch import inject_optimizations_into_step

    for step, batch in enumerate(dataloader):
        loss, metrics = inject_optimizations_into_step(
            opt_wrapper, model, optimizer, batch, step
        )
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLES")
    print("=" * 80)
    print(example_integration.__doc__)
