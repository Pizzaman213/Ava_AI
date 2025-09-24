#!/usr/bin/env python3
"""
Refactored Enhanced Training Script - Modular Architecture

This is the completely refactored training script that uses all the new modular
components extracted from the original train.py. It's much cleaner and more
maintainable while preserving all the advanced functionality.

Usage:
    # Basic enhanced training
    python train.py --config ../configs/gpu/small.yaml --enable-all-features

    # RAG-enabled training
    python train.py --config ../configs/gpu/small.yaml --use-rag --knowledge-base-path data/kb/

    # Multi-task training with gradient surgery
    python train.py --config ../configs/gpu/small.yaml --multi-task --gradient-surgery

    # Ultra-fast training mode
    python train.py --config ../configs/gpu/small.yaml --ultra-fast-mode
"""

import sys
import torch
import yaml
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Union, Dict

# Add project root to path
sys.path.append('/project/code')

# Import new modular components
from src.Ava.config import TrainingConfigManager, EnhancedTrainingConfig
from src.Ava.training.enhanced_trainer import EnhancedModularTrainer
from src.Ava.utils import register_cleanup_handlers
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.data_streaming import create_streaming_dataloaders
from src.Ava.multi_column_data import create_multi_column_dataloader
from src.Ava.training.run_manager import RunManager
from transformers import AutoTokenizer

# Optional imports with fallbacks
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Weights & Biases not installed. Install with: pip install wandb")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("⚠️ DeepSpeed not installed. Install with: pip install deepspeed")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path_obj = Path(config_path)

    # If path doesn't exist, try different relative paths
    if not config_path_obj.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        alt_path = script_dir / config_path
        if alt_path.exists():
            config_path_obj = alt_path
        else:
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / config_path
            if alt_path.exists():
                config_path_obj = alt_path
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path_obj, 'r') as f:
        return yaml.safe_load(f)


def create_model_and_tokenizer(config_dict: dict, training_config: EnhancedTrainingConfig) -> tuple:
    """Create model and tokenizer from configuration."""
    model_config_dict = config_dict.get('model', {})

    # Create enhanced model config with feature flags
    # Override YAML config with training_config values
    enhanced_model_config = model_config_dict.copy()
    enhanced_model_config.update({
        'use_moh': training_config.architecture.use_moh,
        'use_moa': training_config.architecture.use_moa,
        'use_cross_attention': training_config.architecture.use_cross_attention,
        'use_alibi': training_config.architecture.use_alibi,
        'router_type': training_config.architecture.expert_routing_type
    })

    model_config = EnhancedMoEConfig(**enhanced_model_config)

    # Initialize model
    model = EnhancedMoEModel(model_config)

    # Initialize tokenizer
    tokenizer_name = config_dict.get('tokenizer', {}).get('name', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_dataloaders(training_config: EnhancedTrainingConfig, tokenizer, config_dict: dict) -> tuple:
    """Create training and validation dataloaders."""

    if training_config.multi_column_data.use_multi_column:
        # Use multi-column data loader
        print("📊 Using multi-column data loader")

        # Load dataset config if it's a file path
        dataset_config = training_config.multi_column_data.dataset_config
        if isinstance(dataset_config, str) and dataset_config.strip():
            import yaml
            with open(dataset_config, 'r') as f:
                dataset_config = yaml.safe_load(f)
        elif not dataset_config or dataset_config == '':
            # Default config if none provided
            dataset_config = {
                'columns': [{'name': 'text', 'type': 'text'}],
                'combine_strategy': 'concatenate'
            }

        from typing import cast
        from src.Ava.multi_column_data import DatasetConfig

        train_loader = create_multi_column_dataloader(
            config=cast(Union[DatasetConfig, Dict], dataset_config),
            tokenizer=tokenizer,
            batch_size=config_dict.get('training', {}).get('batch_size', 8),
            split='train'
        )

        val_loader = create_multi_column_dataloader(
            config=cast(Union[DatasetConfig, Dict], dataset_config),
            tokenizer=tokenizer,
            batch_size=config_dict.get('training', {}).get('batch_size', 8),
            split='validation'
        )

    elif training_config.data.streaming:
        # Use streaming data loader
        print("🌊 Using streaming data loader")

        train_loader, val_loader = create_streaming_dataloaders(
            tokenizer=tokenizer,
            batch_size=config_dict.get('training', {}).get('batch_size', 8),
            max_length=training_config.data.max_length,
            data_dir=training_config.data.data_dir,
            buffer_size=training_config.data.buffer_size,
            max_samples=training_config.data.max_samples
        )

    else:
        # Fallback to basic data loading (would need implementation)
        raise NotImplementedError("Basic data loading not implemented in this refactored version")

    return train_loader, val_loader


def setup_optimizer(model: torch.nn.Module, config_dict: dict, training_config: EnhancedTrainingConfig) -> torch.optim.Optimizer:
    """Set up optimizer from configuration."""
    training_cfg = config_dict.get('training', {})

    # Override with command line args if provided
    lr = training_config.training.learning_rate or training_cfg.get('learning_rate', 5e-5)
    weight_decay = training_cfg.get('weight_decay', 0.01)

    # Ensure values are numeric
    lr = float(lr)
    weight_decay = float(weight_decay)

    # Create optimizer
    optimizer_type = training_cfg.get('optimizer', 'adamw').lower()

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    return optimizer


def setup_wandb(training_config: EnhancedTrainingConfig, config_dict: dict, model_config: dict, run_manager=None):
    """Initialize Weights & Biases if enabled."""
    if not training_config.wandb.use_wandb or not WANDB_AVAILABLE:
        return None

    try:
        import wandb
        # Prepare wandb config
        wandb_config = {
            # Model configuration
            'model_type': 'EnhancedMoE',
            **model_config,

            # Training configuration
            **config_dict.get('training', {}),

            # Enhanced features
            'use_moh': training_config.architecture.use_moh,
            'use_moa': training_config.architecture.use_moa,
            'use_rag': training_config.rag.use_rag,
            'gradient_surgery': training_config.gradient.gradient_surgery,
            'quantization_aware': training_config.quantization.quantization_aware,
            'performance_mode': training_config.performance.ultra_fast_mode
        }

        # Initialize wandb run
        run_name = run_manager.run_id if run_manager else f"moe_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb_run = wandb.init(
            project=training_config.wandb.wandb_project,
            name=training_config.wandb.wandb_name or run_name,
            config=wandb_config,
            tags=training_config.wandb.wandb_tags,
            resume="allow",
            dir=str(run_manager.run_dir) if run_manager else './wandb',
            save_code=True
        )

        print(f"WandB initialized: {wandb_run.name}")
        return wandb_run

    except Exception as e:
        print(f"⚠️ WandB initialization failed: {e}")
        return None


def train_epoch(trainer: EnhancedModularTrainer, dataloader, optimizer, epoch: int, total_epochs: int) -> dict:
    """Train for one epoch using the enhanced modular trainer."""
    trainer.model.train()
    epoch_stats = {
        'total_loss': 0.0,
        'num_batches': 0,
        'start_time': time.time()
    }

    # Create progress bar if not in ultra-fast mode
    from src.Ava.training.performance_modes import PerformanceMode
    show_progress = trainer.performance_manager.config.mode != PerformanceMode.ULTRA_FAST
    progress_bar = tqdm(
        dataloader,
        desc=f'Epoch {epoch}/{total_epochs}',
        disable=not show_progress,
        dynamic_ncols=True
    )

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(trainer.device)
        attention_mask = batch['attention_mask'].to(trainer.device)
        labels = batch.get('labels', input_ids).to(trainer.device)

        # Perform training step using the modular trainer
        step_results = trainer.train_step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            optimizer=optimizer,
            epoch=epoch,
            batch_idx=batch_idx
        )

        # Update epoch statistics
        epoch_stats['total_loss'] += step_results['loss']
        epoch_stats['num_batches'] += 1

        # Update progress bar
        if show_progress and trainer.performance_manager.should_update_progress(batch_idx):
            current_loss = epoch_stats['total_loss'] / epoch_stats['num_batches']
            progress_bar.set_postfix({
                'Loss': f"{current_loss:.4f}",
                'LR': f"{step_results['learning_rate']:.2e}",
                'GradNorm': f"{step_results['grad_norm']:.3f}"
            })

        # Memory management
        if batch_idx % 100 == 0:
            trainer.gpu_manager.cleanup_gpu_memory()

    # Calculate final epoch statistics
    epoch_time = time.time() - epoch_stats['start_time']
    avg_loss = epoch_stats['total_loss'] / max(epoch_stats['num_batches'], 1)

    return {
        'avg_loss': avg_loss,
        'total_loss': epoch_stats['total_loss'],
        'num_batches': epoch_stats['num_batches'],
        'epoch_time': epoch_time,
        'tokens_per_second': epoch_stats['num_batches'] * 8 * 512 / epoch_time  # Approximate
    }


def evaluate_model(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    """Evaluate model and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels', input_ids).to(device)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            loss = outputs.get('loss', torch.tensor(0.0))
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    """Main training function using modular components."""

    # Register GPU cleanup handlers
    register_cleanup_handlers()

    # 1. Parse arguments and create configuration
    print("📋 Setting up configuration...")
    config_manager = TrainingConfigManager()
    parser = config_manager.create_argument_parser()
    args = parser.parse_args()

    # Parse to structured configuration
    training_config = config_manager.parse_args_to_config(args)

    # Validate configuration
    validation_messages = config_manager.validate_config(training_config)
    for message in validation_messages:
        print(f"WARNING: {message}")

    # Load base config file
    config_dict = load_config(args.config)

    # Get feature summary
    feature_summary = config_manager.get_feature_summary(training_config)
    print(f"Enhanced Features ({feature_summary['total_features']}): {', '.join(feature_summary['enabled_features'])}")
    print(f"Performance Mode: {feature_summary['performance_mode']}")
    print(f"Expert Routing: {feature_summary['expert_routing']}")

    # 2. Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 3. Initialize run manager (optional)
    run_manager = None
    if not training_config.run_management.disable_run_manager:
        run_manager = RunManager(
            base_output_dir=str(Path(training_config.output.output_dir)),
            run_name=training_config.run_management.run_name
        )
        print(f"Run Manager: {run_manager.run_id}")

    # 4. Create model and tokenizer
    print("Initializing model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(config_dict, training_config)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M parameters")

    # 5. Create dataloaders
    print("Setting up data loaders...")
    train_loader, val_loader = create_dataloaders(training_config, tokenizer, config_dict)
    print(f"Data loaders ready")

    # 6. Initialize enhanced modular trainer
    print("Initializing Enhanced Modular Trainer...")
    trainer = EnhancedModularTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=training_config,
        run_manager=run_manager
    )

    # 7. Set up optimizer and training components
    print("Setting up optimizer and training components...")
    optimizer = setup_optimizer(model, config_dict, training_config)
    setup_info = trainer.setup_training(optimizer)

    print("Training setup:")
    for key, value in setup_info.items():
        print(f"  - {key}: {value}")

    # 8. Initialize WandB
    wandb_run = setup_wandb(training_config, config_dict, config_dict.get('model', {}), run_manager)
    if wandb_run and trainer.async_logger:
        trainer.async_logger.set_wandb_run(wandb_run)

    # 9. Training loop
    print("\nStarting Training")
    print("=" * 60)

    num_epochs = training_config.training.epochs or config_dict.get('training', {}).get('num_epochs', 3)
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        try:
            # Train
            train_results = train_epoch(trainer, train_loader, optimizer, epoch, num_epochs)
            print(f"  Train: {train_results['avg_loss']:.4f} ({train_results['epoch_time']:.1f}s)")

            # Evaluate
            val_loss = evaluate_model(model, val_loader, device)
            print(f"  Val: {val_loss:.4f}")

            # Log to WandB
            if wandb_run:
                try:
                    import wandb
                    wandb.log({
                        'epoch': epoch,
                        'train/epoch_loss': train_results['avg_loss'],
                        'train/tokens_per_second': train_results.get('tokens_per_second', 0),
                        'val/loss': val_loss,
                        'train/epoch_time': train_results['epoch_time']
                    })
                except Exception as e:
                    print(f"⚠️ WandB logging failed: {e}")

            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                if run_manager:
                    run_manager.save_checkpoint(
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch,
                        step=trainer.step_count,
                        loss=val_loss,
                        is_best=True,
                        additional_data={
                            'config': config_dict,
                            'training_config': training_config.__dict__ if hasattr(training_config, '__dict__') else str(training_config)
                        }
                    )
                    print(f"  Best model saved: {val_loss:.4f}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  WARNING: GPU OOM, cleaning up and continuing...")
                trainer.gpu_manager.cleanup_gpu_memory(aggressive=True)
                continue
            else:
                raise

    # 10. Training completion
    print("\n" + "=" * 60)
    print("✅ Training Complete!")

    # Get final statistics
    final_stats = trainer.get_training_statistics()
    print(f"📊 Final Statistics:")
    print(f"  - Total Steps: {final_stats['step_count']}")
    print(f"  - Best Validation Loss: {best_val_loss:.4f}")

    if 'memory' in final_stats:
        memory_stats = final_stats['memory']
        if 'allocated_gb' in memory_stats:
            print(f"  - GPU Memory: {memory_stats['allocated_gb']:.2f}GB")

    # Save final model
    if run_manager:
        run_manager.save_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=num_epochs,
            step=trainer.step_count,
            loss=best_val_loss,
            additional_data={
                'config': config_dict,
                'training_config': training_config.__dict__ if hasattr(training_config, '__dict__') else str(training_config),
                'final_model': True
            }
        )

        run_manager.finish_run('completed', {
            'best_val_loss': best_val_loss,
            'total_epochs': num_epochs,
            'total_steps': trainer.step_count
        })

        final_path = run_manager.get_checkpoint_path('best')
        print(f"💾 Model saved: {final_path}")
        print(f"🆔 Run ID: {run_manager.run_id}")

        print(f"\n🎯 To generate text:")
        print(f"python /project/code/scripts/generation/generate.py --model-path {final_path} --prompt 'Your text here'")

    # Cleanup
    trainer.cleanup()

    # Distributed training cleanup
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    if wandb_run:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass

    print("\nAll done! The modular architecture makes training much cleaner!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # Cleanup will be handled by registered handlers
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise