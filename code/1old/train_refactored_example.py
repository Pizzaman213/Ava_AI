#!/usr/bin/env python3
"""
Ava Training Pipeline - Refactored Orchestration Script (Example)

This shows how to use the modular training framework with the new managers.
The actual train.py should be updated to follow this pattern.

Key improvements over monolithic train.py:
- ~90% smaller main script (from 5,022 to ~400-500 lines)
- Clear separation of concerns with specialized managers
- Easy to test, debug, and extend
- Backward compatible with existing configurations
"""

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch
import yaml
from pathlib import Path
from typing import Optional

# Setup path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.Ava.config import EnhancedTrainingConfig, TrainingConfigManager
from src.Ava.utils.logging import setup_training_logger, get_logger
from src.Ava.training.train.data_loader_manager import DataLoaderManager
from src.Ava.training.train.model_manager import ModelManager
from src.Ava.training.train.optimizer_manager import OptimizerManager
from src.Ava.training.train.evaluation_manager import EvaluationManager
from src.Ava.training.train.base import TrainingContext


def load_config(config_path: str) -> tuple:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Tuple of (raw_config_dict, enhanced_training_config)
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    training_config = TrainingConfigManager.from_dict(config_dict)
    return config_dict, training_config


def setup_training_context(
    config_dict: dict,
    training_config: EnhancedTrainingConfig
) -> TrainingContext:
    """Create training context with all necessary information.

    Args:
        config_dict: Raw configuration dictionary
        training_config: Enhanced training configuration

    Returns:
        Training context instance
    """
    context = TrainingContext(
        config=training_config,
        raw_config_dict=config_dict,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
    )
    return context


def main(
    config_path: str,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    **kwargs
):
    """Main training orchestration function (slim version).

    Args:
        config_path: Path to training configuration YAML
        batch_size: Optional override for batch size
        learning_rate: Optional override for learning rate
        **kwargs: Additional configuration overrides
    """
    # Setup logging
    setup_training_logger()
    logger = get_logger()

    logger.info("="*80)
    logger.info("🚀 STARTING REFINED AVA TRAINING PIPELINE")
    logger.info("="*80)

    # Load configuration
    logger.info(f"\n📋 Loading configuration from: {config_path}")
    config_dict, training_config = load_config(config_path)

    # Apply command-line overrides
    if batch_size is not None:
        training_config.training.batch_size = batch_size
        logger.info(f"   Batch size override: {batch_size}")
    if learning_rate is not None:
        training_config.training.learning_rate = learning_rate
        logger.info(f"   Learning rate override: {learning_rate:.2e}")

    # Create training context
    context = setup_training_context(config_dict, training_config)
    logger.info(f"\n🖥️  Device: {context.device}")
    logger.info(f"   Dtype: {context.dtype}")
    if context.world_size > 1:
        logger.info(f"   Distributed: Rank {context.rank}/{context.world_size}")

    # Initialize managers
    logger.info("\n⚙️  Initializing training managers...")

    data_loader_manager = DataLoaderManager(context)
    model_manager = ModelManager(context)
    optimizer_manager = OptimizerManager(context)
    evaluation_manager = EvaluationManager(context)

    # Create model and tokenizer
    logger.info("\n🤖 Creating model and tokenizer...")
    model, tokenizer = model_manager.create_model_and_tokenizer(
        config_dict, training_config
    )

    # Create dataloaders
    logger.info("\n📊 Creating dataloaders...")
    train_loader, val_loader = data_loader_manager.create_dataloaders(
        training_config, tokenizer, config_dict, batch_size
    )

    # Calculate total steps for warmup scheduling
    samples_per_epoch = getattr(
        training_config.data, 'max_samples',
        training_config.data.buffer_size
    ) or 10000
    steps_per_epoch = samples_per_epoch // (
        training_config.training.batch_size *
        getattr(training_config.training, 'gradient_accumulation_steps', 1)
    )
    total_steps = steps_per_epoch * training_config.training.num_epochs

    # Setup optimizer and learning rate management
    logger.info("\n⚡ Setting up optimizer and learning rate management...")
    optimizer, adaptive_lr_manager = optimizer_manager.setup_optimizer_and_lr(
        model, config_dict, training_config, total_steps
    )

    # Quick generation quality test
    logger.info("\n✨ Testing generation quality...")
    test_prompts = [
        "The meaning of life is",
        "Artificial intelligence is",
        "The future of technology is",
    ]
    gen_results = evaluation_manager.test_generation_quality(
        model, tokenizer, context.device, test_prompts, max_length=50
    )
    logger.info(f"   Generated {len(gen_results['generated_texts'])} samples")
    logger.info(f"   Average length: {gen_results['avg_length']:.1f} tokens")

    # Quick validation
    logger.info("\n📈 Running validation...")
    val_loss, val_perplexity = evaluation_manager.evaluate_model(
        model, val_loader, context.device, use_bf16=True, max_batches=10
    )
    if val_loss is not None:
        logger.info(f"   Validation loss: {val_loss:.4f}")
        if val_perplexity is not None:
            logger.info(f"   Validation perplexity: {val_perplexity:.4f}")

    # Smoke test for checkpoint resume
    logger.info("\n✓ Running resume smoke test...")
    smoke_test_passed = evaluation_manager.resume_smoke_test(
        model, train_loader, context.device, num_steps=3
    )

    logger.info("\n" + "="*80)
    if smoke_test_passed:
        logger.info("✅ All initialization tests passed - ready for training!")
    else:
        logger.info("⚠️  Smoke test failed - check configuration")
    logger.info("="*80)

    # Return initialized components for actual training
    return {
        "context": context,
        "model": model,
        "tokenizer": tokenizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "adaptive_lr_manager": adaptive_lr_manager,
        "managers": {
            "data_loader": data_loader_manager,
            "model": model_manager,
            "optimizer": optimizer_manager,
            "evaluation": evaluation_manager,
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Ava language model with modular framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )

    args = parser.parse_args()

    # Run initialization
    training_components = main(
        config_path=args.config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("\n" + "="*80)
    print("Training components initialized successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Use training_components['model'] for forward passes")
    print("2. Use training_components['optimizer'] to update weights")
    print("3. Use managers to load data, evaluate, and manage checkpoints")
    print("4. Integrate with SimplifiedEnhancedTrainer for actual training loop")
