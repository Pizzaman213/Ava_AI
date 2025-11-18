"""
Example: Using the Modular Training Framework

This file demonstrates how to use the SimplifiedEnhancedTrainer
and its components in a realistic training scenario.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from Ava.training.train import (
    SimplifiedEnhancedTrainer,
    TrainingContext,
    LossComputationManager,
)
from Ava.config.training_config import EnhancedTrainingConfig


# ============================================================================
# Example 1: Basic Training Loop
# ============================================================================


def example_basic_training():
    """Simplest training example."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Training Loop")
    print("=" * 70)

    # Setup
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")
    config = EnhancedTrainingConfig()

    # Create trainer
    trainer = SimplifiedEnhancedTrainer(
        model=model,
        config=config,
        device=device,
    )

    # Initialize
    trainer.initialize(optimizer)
    print("✓ Trainer initialized")

    try:
        # Create dummy data
        num_epochs = 2
        batch_size = 32

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training loop
            for step in range(5):  # 5 steps for demo
                # Create dummy batch
                batch = {
                    "input_ids": torch.randn(batch_size, 100, device=device),
                    "labels": torch.randint(0, 10, (batch_size,), device=device),
                }

                # Training step
                metrics = trainer.train_step(batch)
                if step % 2 == 0:
                    print(f"  Step {step}: loss={metrics.get('loss', 0):.4f}")

    finally:
        trainer.cleanup()
        print("\n✓ Training completed and cleaned up")


# ============================================================================
# Example 2: Using Individual Managers
# ============================================================================


def example_individual_managers():
    """Using managers independently for testing."""
    print("\n" + "=" * 70)
    print("Example 2: Using Individual Managers")
    print("=" * 70)

    model = nn.Linear(100, 10)
    device = torch.device("cpu")
    config = EnhancedTrainingConfig()

    # Create context
    context = TrainingContext(
        model=model,
        device=device,
        config=config,
    )

    # ---- Test LossComputationManager ----
    print("\n--- LossComputationManager ---")
    loss_manager = LossComputationManager(context)
    loss_manager.initialize()

    # Register loss functions
    loss_manager.register_loss_function(
        "ce_loss",
        nn.CrossEntropyLoss(),
        weight=1.0,
    )
    loss_manager.register_loss_function(
        "mse_loss",
        nn.MSELoss(),
        weight=0.1,
    )
    print("✓ Loss functions registered")

    # Compute loss
    outputs = {
        "logits": torch.randn(8, 10),
    }
    targets = torch.randint(0, 10, (8,))

    try:
        loss, breakdown = loss_manager.compute_loss(outputs, targets)
        print(f"✓ Loss computed: {loss.item():.4f}")
        print(f"  Breakdown: {breakdown}")
    except Exception as e:
        print(f"✗ Loss computation error: {e}")

    # Get status
    status = loss_manager.get_status()
    print(f"✓ Status: {status}")

    loss_manager.cleanup()

    # ---- Test Distributed Manager ----
    print("\n--- DistributedTrainingManager ---")
    from Ava.training.train import DistributedTrainingManager

    dist_manager = DistributedTrainingManager(context)
    dist_manager.initialize()

    status = dist_manager.get_status()
    print(f"✓ Distributed status: {status}")
    print(f"  Is main rank: {dist_manager.is_main_rank()}")
    print(f"  Should log: {dist_manager.should_log()}")

    dist_manager.cleanup()

    # ---- Test Checkpoint Manager ----
    print("\n--- CheckpointManager ---")
    from Ava.training.train import CheckpointManager

    checkpoint_dir = Path("/tmp/test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Mock run manager
    class MockRunManager:
        checkpoint_dir = str(checkpoint_dir)

    checkpoint_manager = CheckpointManager(context)
    checkpoint_manager.context.run_manager = MockRunManager()
    checkpoint_manager.initialize()

    # Save checkpoint
    ckpt_path = checkpoint_manager.save_checkpoint(
        epoch=0,
        step=100,
        metrics={"test_metric": 0.5},
    )
    print(f"✓ Checkpoint saved: {ckpt_path}")

    # Load checkpoint
    metadata = checkpoint_manager.load_checkpoint(ckpt_path)
    print(f"✓ Checkpoint loaded: epoch={metadata['epoch']}, step={metadata['step']}")

    checkpoint_manager.cleanup()


# ============================================================================
# Example 3: Custom Loss Functions
# ============================================================================


def example_custom_losses():
    """Using custom loss functions with the framework."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Loss Functions")
    print("=" * 70)

    # Define custom loss
    class CustomLoss(nn.Module):
        def forward(self, outputs, targets):
            logits = outputs["logits"]
            return torch.nn.functional.cross_entropy(logits, targets)

    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cpu")
    config = EnhancedTrainingConfig()

    # Create trainer
    trainer = SimplifiedEnhancedTrainer(model, config, device)
    trainer.initialize(optimizer)

    try:
        # Register custom losses
        trainer.loss_manager.register_loss_function(
            "custom_ce",
            CustomLoss(),
            weight=1.0,
        )
        trainer.loss_manager.register_loss_function(
            "l1_penalty",
            nn.L1Loss(),
            weight=0.01,
        )
        print("✓ Custom losses registered")

        # Training step with custom losses
        batch = {
            "input_ids": torch.randn(16, 100, device=device),
            "labels": torch.randint(0, 10, (16,), device=device),
        }

        metrics = trainer.train_step(batch)
        print(f"✓ Training step completed")
        print(f"  Loss breakdown: {metrics}")

    finally:
        trainer.cleanup()


# ============================================================================
# Example 4: Training with Evaluation
# ============================================================================


def example_training_with_evaluation():
    """Full training loop with periodic evaluation."""
    print("\n" + "=" * 70)
    print("Example 4: Training with Evaluation")
    print("=" * 70)

    # Setup
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cpu")
    config = EnhancedTrainingConfig()

    trainer = SimplifiedEnhancedTrainer(model, config, device)
    trainer.initialize(optimizer)

    # Register loss
    trainer.loss_manager.register_loss_function(
        "ce",
        nn.CrossEntropyLoss(),
    )

    try:
        # Create dummy data loaders
        def create_dummy_loader(num_batches=3, batch_size=16):
            for _ in range(num_batches):
                yield {
                    "input_ids": torch.randn(batch_size, 100, device=device),
                    "labels": torch.randint(0, 10, (batch_size,), device=device),
                }

        train_loader = create_dummy_loader(num_batches=5)
        val_loader = create_dummy_loader(num_batches=2)

        # Training
        print("\nTraining with evaluation...")
        epoch_metrics = trainer.train_epoch(
            train_loader,
            eval_loader=val_loader,
            steps_per_epoch=None,
        )

        print(f"✓ Epoch completed")
        print(f"  Training loss: {epoch_metrics.get('avg_loss', 0):.4f}")
        print(f"  Evaluation loss: {epoch_metrics.get('eval_loss', 0):.4f}")

        # Save checkpoint
        trainer.save_checkpoint(save_best=True)
        print("✓ Checkpoint saved")

    finally:
        trainer.cleanup()


# ============================================================================
# Example 5: Monitoring and Metrics
# ============================================================================


def example_monitoring():
    """Using the monitoring manager."""
    print("\n" + "=" * 70)
    print("Example 5: Monitoring and Metrics")
    print("=" * 70)

    model = nn.Linear(100, 10)
    device = torch.device("cpu")
    config = EnhancedTrainingConfig()

    context = TrainingContext(model, device, config)

    from Ava.training.train import MonitoringManager

    monitor = MonitoringManager(context)
    monitor.initialize()

    # Log metrics
    print("\nLogging metrics...")
    for step in range(10):
        metrics = {
            "loss": 0.5 - step * 0.01,  # Decreasing loss
            "accuracy": 0.8 + step * 0.01,  # Increasing accuracy
            "learning_rate": 0.001 * (0.95 ** step),  # Decreasing LR
        }

        monitor.log_metrics(metrics, step=step)

    print("✓ Metrics logged")

    # Training step logging
    print("\nLogging training step...")
    monitor.log_training_step(
        step=10,
        epoch=1,
        loss=0.45,
        learning_rate=0.0005,
        grad_norm=0.1,
    )
    print("✓ Training step logged")

    # Get status
    status = monitor.get_status()
    print(f"✓ Monitoring status:")
    print(f"  Current loss: {status['metrics'].get('loss', 'N/A')}")
    print(f"  Average step time: {status['avg_step_time_ms']:.2f}ms")

    monitor.cleanup()


# ============================================================================
# Example 6: Status and Debugging
# ============================================================================


def example_status_and_debugging():
    """Getting status for debugging."""
    print("\n" + "=" * 70)
    print("Example 6: Status and Debugging")
    print("=" * 70)

    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cpu")
    config = EnhancedTrainingConfig()

    trainer = SimplifiedEnhancedTrainer(model, config, device)
    trainer.initialize(optimizer)

    try:
        # Get full trainer status
        print("\nFull Trainer Status:")
        status = trainer.get_status()

        print(f"\nTraining Context:")
        for key, value in status["training_context"].items():
            print(f"  {key}: {value}")

        print(f"\nDistributed:")
        for key, value in status["distributed"].items():
            print(f"  {key}: {value}")

        print(f"\nCheckpoint:")
        for key, value in status["checkpoint"].items():
            print(f"  {key}: {value}")

        print(f"\nMonitoring:")
        for key, value in status["monitoring"].items():
            print(f"  {key}: {value}")

    finally:
        trainer.cleanup()


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Modular Training Framework Examples")
    print("=" * 70)

    # Run examples
    example_basic_training()
    example_individual_managers()
    example_custom_losses()
    example_training_with_evaluation()
    example_monitoring()
    example_status_and_debugging()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
