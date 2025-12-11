#!/usr/bin/env python3
"""
Ava Pipeline Training Script

This is the new pipeline-based training script that uses the modular
Ava component architecture. It replaces the monolithic train_100m_full.py
with a cleaner, more maintainable structure.

Usage:
    python train_pipeline.py --config code/configs/moe/large.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 train_pipeline.py --config code/configs/moe/4x_a6000_max_speed.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root / 'code' / 'src'))

# Ava pipeline imports
from Ava.training.train import (
    TrainingContext,
    DataLoaderManager,
    ModelBuilder,
    OptimizerManager,
    TrainingLoopManager,
    TrainingLoopConfig,
    ValidationManager,
    GenerationManager,
    MetricsManager,
)
from Ava.training.orchestration import RunManager, TrainingPipeline
from Ava.training.utils import setup_distributed, cleanup_distributed
from Ava.config.yaml_loader import load_yaml_with_path_resolution
from Ava.config.training_config import DynamicConfig
from Ava.kernels import KernelConfig, set_kernel_config, TRITON_AVAILABLE
from Ava.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def log_optimization_status(config: dict, rank: int = 0) -> None:
    """
    Log the status of all optimization features at training start.

    Provides visibility into which optimizations are enabled/disabled
    to help diagnose performance issues.
    """
    if rank != 0:
        return

    model_config = config.get('model', {})
    training_config = config.get('training', {})
    batching_config = training_config.get('batching', {})
    dynamic_batching = batching_config.get('dynamic_batching', {})
    data_config = config.get('data', {})
    perf_config = config.get('performance', {})
    hybrid_config = config.get('hybrid_caching', {})
    overlapped_config = config.get('overlapped_checkpointing', {})
    double_config = config.get('double_checkpointing', {})
    fp8_config = config.get('fp8', {})

    print("\n" + "=" * 60)
    print("ACTIVE OPTIMIZATIONS STATUS")
    print("=" * 60)

    # Model optimizations
    print("\n[Model Optimizations]")
    print(f"  use_flash_attention:     {'YES' if model_config.get('use_flash_attention', False) else 'NO'}")
    print(f"  gradient_checkpointing:  {'YES' if model_config.get('gradient_checkpointing', False) else 'NO'}")
    print(f"  use_grouped_gemm:        {'YES' if model_config.get('use_grouped_gemm', False) else 'NO'}")
    print(f"  use_triton_kernels:      {'YES' if model_config.get('use_triton_kernels', False) else 'NO'}")
    print(f"  use_torch_compile:       {'YES' if model_config.get('use_torch_compile', False) else 'NO'}")
    print(f"  use_optimized_moe:       {'YES' if model_config.get('use_optimized_moe', False) else 'NO'}")

    # Dynamic batching
    print("\n[Dynamic Batching]")
    db_enabled = dynamic_batching.get('enabled', False)
    print(f"  enabled:                 {'YES' if db_enabled else 'NO'}")
    if db_enabled:
        token_budget = dynamic_batching.get('token_budget', {})
        print(f"  token_budget:            {'YES' if token_budget.get('enabled', False) else 'NO'}")
        print(f"  predictive_memory:       {'YES' if dynamic_batching.get('predictive_memory_estimation', False) else 'NO'}")
        print(f"  trend_detection:         {'YES' if dynamic_batching.get('trend_detection', False) else 'NO'}")

    # Data loading
    print("\n[Data Loading]")
    print(f"  use_pretokenized:        {'YES' if data_config.get('use_pretokenized', False) else 'NO'}")
    print(f"  use_sequence_packing:    {'YES' if data_config.get('use_sequence_packing', False) else 'NO'}")
    print(f"  lazy_file_discovery:     {'YES' if data_config.get('lazy_file_discovery', False) else 'NO'}")

    # Advanced optimizations
    print("\n[Advanced Optimizations]")
    print(f"  hybrid_caching:          {'YES' if hybrid_config.get('enabled', False) else 'NO'}")
    print(f"  overlapped_checkpointing: {'YES' if overlapped_config.get('enabled', False) else 'NO'}")
    print(f"  double_checkpointing:    {'YES' if double_config.get('enabled', False) else 'NO'}")
    print(f"  fp8_training:            {'YES' if fp8_config.get('enabled', False) else 'NO'}")

    # Performance settings
    print("\n[Performance Settings]")
    print(f"  enable_torch_compile:    {'YES' if perf_config.get('enable_torch_compile', False) else 'NO'}")
    print(f"  enable_tf32:             {'YES' if perf_config.get('enable_tf32', False) else 'NO'}")
    print(f"  enable_cudnn_benchmark:  {'YES' if perf_config.get('enable_cudnn_benchmark', False) else 'NO'}")

    print("=" * 60 + "\n")


def setup_logging(log_dir: Path, rank: int = 0) -> logging.Logger:
    """Setup logging for training."""
    logger = logging.getLogger('train_pipeline')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    if rank == 0:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', '%H:%M:%S')
        )
        logger.addHandler(console_handler)

        # File handler
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        )
        logger.addHandler(file_handler)

    return logger


def main(args):
    """Main training function using the Ava pipeline architecture."""

    # =========================================================================
    # Phase 1: Distributed Setup
    # =========================================================================
    rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    # Phase 2: Load Configuration
    # =========================================================================
    config = load_yaml_with_path_resolution(args.config)

    # Extract key config sections
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    kernel_opt_config = config.get('kernel_optimization', {})

    # =========================================================================
    # Phase 2.1: Configure Kernel Optimizations
    # =========================================================================
    if kernel_opt_config and TRITON_AVAILABLE:
        kernel_config = KernelConfig(
            use_bitonic_topk=kernel_opt_config.get('use_fused_softmax_topk', True),
            use_heap_topk=kernel_opt_config.get('use_fused_softmax_topk', True),
            router_block_size=kernel_opt_config.get('router_block_size', 4),
            use_fused_softmax_topk=kernel_opt_config.get('use_fused_softmax_topk', True),
        )
        set_kernel_config(kernel_config)
        if rank == 0:
            print(f"[Kernel Optimization] Triton kernels enabled:")
            print(f"  - Fused softmax+topk: {kernel_config.use_fused_softmax_topk}")
            print(f"  - Router block size: {kernel_config.router_block_size}")
            print(f"  - Fused activations: {kernel_opt_config.get('use_fused_activations', True)}")
            print(f"  - Vectorized capacity: {kernel_opt_config.get('use_vectorized_capacity', True)}")
    elif rank == 0:
        if not TRITON_AVAILABLE:
            print("[Kernel Optimization] Triton not available, using PyTorch fallbacks")
        else:
            print("[Kernel Optimization] No kernel_optimization config found, using defaults")

    # =========================================================================
    # Phase 2.2: Log Optimization Status (visibility into what's enabled)
    # =========================================================================
    log_optimization_status(config, rank)

    # Override with CLI args
    num_epochs = args.epochs or training_config.get('num_epochs', 3)
    batch_size = args.batch_size or training_config.get('batch_size', 8)
    learning_rate = args.learning_rate or training_config.get('learning_rate', 5e-5)
    log_interval = args.log_interval or training_config.get('log_interval', 10)
    val_interval = args.val_interval

    # Get output directory from config or CLI args
    output_config = config.get('output', {})
    output_dir = output_config.get('output_dir', args.save_dir)

    # =========================================================================
    # Phase 3: Create RunManager (Ava's output organizer)
    # =========================================================================
    run_manager = RunManager(
        base_output_dir=str(output_dir),
        run_name=config.get('experiment_name', 'ava_training'),
        description=f"Training with config: {args.config}"
    )

    # Save the actual configuration files to the run directory
    if rank == 0:
        # Save full config
        run_manager.save_config(config, 'full')
        # Save individual sections for easy reference
        if model_config:
            run_manager.save_config(model_config, 'model')
        if training_config:
            run_manager.save_config(training_config, 'training')
        if data_config:
            run_manager.save_config(data_config, 'data')
        # Save command line args
        run_manager.save_args(args)

    # Setup logging
    log_dir = run_manager.run_dir / 'logs' if hasattr(run_manager, 'run_dir') else Path(args.log_dir)
    train_logger = setup_logging(log_dir, rank)

    # =========================================================================
    # Phase 3.1: Create CheckpointManager
    # =========================================================================
    checkpoint_dir = run_manager.run_dir / 'checkpoints' if hasattr(run_manager, 'run_dir') else Path(args.save_dir) / 'checkpoints'
    checkpoint_manager = CheckpointManager(
        save_dir=checkpoint_dir,
        max_keep=training_config.get('max_checkpoints', 3),
        config=config,
        async_save=True
    )

    # Resume state tracking
    resume_epoch = 0
    resume_step = 0

    # =========================================================================
    # Phase 3.2: Load Checkpoint if Resuming
    # =========================================================================
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            if rank == 0:
                train_logger.info(f"Will resume from checkpoint: {resume_path}")
            # We'll load the checkpoint after model/optimizer are created
        else:
            if rank == 0:
                train_logger.warning(f"Resume checkpoint not found: {resume_path}, starting fresh")
            args.resume = None

    max_steps = getattr(args, 'max_steps', None)

    if rank == 0:
        train_logger.info("=" * 60)
        train_logger.info("Ava Pipeline Training")
        train_logger.info("=" * 60)
        train_logger.info(f"Config: {args.config}")
        train_logger.info(f"Device: {device}")
        train_logger.info(f"World size: {world_size}")
        train_logger.info(f"Epochs: {num_epochs}")
        train_logger.info(f"Batch size: {batch_size}")
        train_logger.info(f"Learning rate: {learning_rate:.2e}")
        if max_steps:
            train_logger.info(f"Max steps: {max_steps}")

    # =========================================================================
    # Phase 4: Create TrainingContext (Ava's shared state hub)
    # =========================================================================
    context = TrainingContext(
        model=None,  # Will be set by ModelBuilder
        device=device,
        config=config,
        run_manager=run_manager,
        rank=rank,
        world_size=world_size,
        is_main_process=(rank == 0),
    )
    context.update_from_config(config)

    # =========================================================================
    # Phase 5: Create TrainingPipeline and Register Components
    # =========================================================================
    pipeline = TrainingPipeline(context)

    # Register all components
    pipeline.register('model', ModelBuilder(context))
    pipeline.register('optimizer', OptimizerManager(context))
    pipeline.register('data', DataLoaderManager(context))
    pipeline.register('training', TrainingLoopManager(context))
    pipeline.register('validation', ValidationManager(context))
    pipeline.register('generation', GenerationManager(context))
    pipeline.register('metrics', MetricsManager(context))

    try:
        # =====================================================================
        # Phase 6: Build Model
        # =====================================================================
        model_builder = pipeline.get('model')
        model_builder.initialize()

        model = model_builder.build_model(config, device)

        # Apply quantization if enabled
        quant_config = config.get('quantization', {})
        if quant_config.get('enabled', False):
            model = model_builder.apply_quantization(model, quant_config)

        # Move to device
        model = model_builder.move_to_device(model, device)

        # Apply optimizations (FP8, hybrid caching, torch.compile, etc.)
        model = model_builder.apply_optimizations(model, config)

        # Wrap in DDP for distributed
        model = model_builder.wrap_distributed(model, rank, world_size)

        context.model = model

        # =====================================================================
        # Phase 6.1: Initialize BatchSizeController for Dynamic Batching
        # =====================================================================
        dynamic_batching_config = config.get('training', {}).get('batching', {}).get('dynamic_batching', {})
        if not dynamic_batching_config:
            dynamic_batching_config = config.get('dynamic_batching', {})

        batch_controller = None
        if dynamic_batching_config.get('enabled', False):
            try:
                from Ava.training.optimizations.batch_size_controller import (
                    BatchSizeController,
                    create_batch_size_controller,
                )

                batch_controller = create_batch_size_controller(dynamic_batching_config)

                if batch_controller is not None:
                    # Run startup calibration if enabled
                    run_calibration = dynamic_batching_config.get('run_startup_calibration', True)

                    if run_calibration and rank == 0:
                        train_logger.info("Running batch size calibration...")

                        # Create sample batch function for calibration
                        def sample_batch_fn(bs: int):
                            """Create a sample batch for calibration."""
                            seq_len = dynamic_batching_config.get('base_sequence_length', 512)
                            vocab_size = config.get('model', {}).get('vocab_size', 50000)
                            return {
                                'input_ids': torch.randint(0, vocab_size, (bs, seq_len), device=device),
                                'attention_mask': torch.ones(bs, seq_len, device=device),
                            }

                        optimal_bs = batch_controller.startup_calibration(
                            model=model,
                            sample_batch_fn=sample_batch_fn,
                            max_time_seconds=dynamic_batching_config.get('calibration_timeout_sec', 30.0),
                        )
                        train_logger.info(f"Calibration complete: optimal batch size = {optimal_bs}")

                    context.batch_controller = batch_controller
                    if rank == 0:
                        train_logger.info(f"BatchSizeController initialized: {batch_controller.get_state()}")

            except ImportError as e:
                if rank == 0:
                    train_logger.warning(f"Could not import BatchSizeController: {e}")
            except Exception as e:
                if rank == 0:
                    train_logger.warning(f"BatchSizeController initialization failed: {e}")

        # =====================================================================
        # Phase 7: Create Optimizer (scheduler created after dataloader)
        # =====================================================================
        optimizer_mgr = pipeline.get('optimizer')
        optimizer_mgr.initialize()

        optimizer = optimizer_mgr.create_optimizer(
            model, config, learning_rate,
            weight_decay=training_config.get('weight_decay', 0.01)
        )

        context.optimizer = optimizer

        # =====================================================================
        # Phase 7.1: Load Checkpoint if Resuming
        # =====================================================================
        if args.resume:
            resume_path = Path(args.resume)
            if rank == 0:
                train_logger.info(f"Loading checkpoint from: {resume_path}")

            resume_epoch, resume_step = checkpoint_manager.load(
                model=model,
                optimizer=optimizer,
                checkpoint_path=resume_path
            )

            if rank == 0:
                train_logger.info(f"Resumed from epoch {resume_epoch}, step {resume_step}")

        # =====================================================================
        # Phase 8: Create DataLoaders
        # =====================================================================
        data_mgr = pipeline.get('data')
        data_mgr.initialize()

        # Load tokenizer
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            # Try multiple possible config locations for tokenizer path
            tokenizer_path = (
                data_config.get('tokenizer_path') or
                data_config.get('tokenizer_name') or
                config.get('data', {}).get('tokenizer_name') or
                '/project/code/data/Ava_Ai/tokenizer'
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            context.tokenizer = tokenizer
            if rank == 0:
                train_logger.info(f"Loaded tokenizer: {tokenizer_path}")
        except Exception as e:
            if rank == 0:
                train_logger.warning(f"Could not load tokenizer: {e}")

        # Disable generation if no tokenizer available
        generation_config = config.get('generation', {})
        if generation_config and tokenizer is None:
            if rank == 0:
                train_logger.warning("Generation disabled: no tokenizer available")
            generation_config = {}

        # Convert config dict to DynamicConfig for DataLoaderManager
        config_obj = DynamicConfig(config) if isinstance(config, dict) else config

        train_loader, val_loader = data_mgr.create_dataloaders(
            training_config=config_obj,
            tokenizer=tokenizer,
            config_dict=config,
            batch_size=batch_size
        )

        # Update total steps now that we have the data loader
        # Note: IterableDatasets (like InfiniteUltraFastDataset) don't have __len__
        # Try to get length safely, falling back to config value or default
        try:
            loader_len = len(train_loader)
        except TypeError:
            # Infinite/streaming datasets don't have length
            loader_len = config.get('training', {}).get('steps_per_epoch', 1000)
        total_steps = num_epochs * loader_len // context.gradient_accumulation_steps

        if rank == 0:
            train_logger.info(f"Total training steps: {total_steps}")

        # =====================================================================
        # Phase 8.1: Create Scheduler (now that we know total_steps)
        # =====================================================================
        warmup_steps = training_config.get('warmup_steps', 1000)
        scheduler = optimizer_mgr.create_scheduler(
            optimizer, warmup_steps, total_steps,
            min_lr=training_config.get('min_lr', 0.0)
        )
        context.scheduler = scheduler

        if rank == 0:
            train_logger.info(f"Scheduler: warmup={warmup_steps}, total={total_steps}")

        # =====================================================================
        # Phase 9: Setup Metrics
        # =====================================================================
        metrics_mgr = pipeline.get('metrics')
        metrics_mgr.initialize()

        wandb_config = config.get('wandb', {})
        # Set wandb directory inside the run folder
        wandb_dir = run_manager.run_dir / 'wandb'
        metrics_mgr.setup(
            log_dir=log_dir,
            wandb_config=wandb_config if wandb_config.get('enabled', False) else None,
            use_wandb=wandb_config.get('enabled', False),
            wandb_dir=wandb_dir,
        )

        # =====================================================================
        # Phase 10: Initialize Remaining Components
        # =====================================================================
        validation_mgr = pipeline.get('validation')
        validation_mgr.initialize()

        generation_mgr = pipeline.get('generation')
        generation_mgr.initialize()
        generation_mgr.set_log_dir(log_dir)

        training_mgr = pipeline.get('training')
        training_mgr.initialize()
        training_mgr.set_components(
            metrics_manager=metrics_mgr,
            generation_manager=generation_mgr,
            checkpoint_manager=checkpoint_manager
        )

        # Set global step if resuming (so TrainingLoopManager tracks correctly)
        if resume_step > 0:
            training_mgr._global_step = resume_step

        # Determine profile directory - use run folder if not explicitly specified
        profile_dir = getattr(args, 'profile_dir', None)
        if profile_dir is None or profile_dir == './profiles':
            # Default to run folder/profiles
            profile_dir = str(run_manager.run_dir / 'profiles') if hasattr(run_manager, 'run_dir') else './profiles'

        # Create training loop config
        loop_config = TrainingLoopConfig(
            gradient_accumulation_steps=context.gradient_accumulation_steps,
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            use_amp=context.use_amp,
            amp_dtype=context.amp_dtype,
            log_interval=log_interval,
            generate_every_n_steps=config.get('generation', {}).get('generate_every_n_steps', 500),
            save_steps=training_config.get('save_steps', 0),
            max_steps=getattr(args, 'max_steps', None),
            # Profiling options (disabled by default)
            enable_profiling=getattr(args, 'enable_profiling', False),
            profile_start_step=getattr(args, 'profile_start_step', 0),
            profile_end_step=getattr(args, 'profile_end_step', 999999),
            profile_dir=profile_dir,
        )

        # Setup profiler if enabled
        if loop_config.enable_profiling and rank == 0:
            training_mgr.setup_profiler(loop_config)

        # =====================================================================
        # Phase 11: Training Loop
        # =====================================================================
        if rank == 0:
            train_logger.info("\n" + "=" * 60)
            train_logger.info("Starting Training")
            train_logger.info("=" * 60)

        best_val_loss = float('inf')

        for epoch in range(resume_epoch, num_epochs):
            # Notify components of epoch start
            pipeline.on_epoch_start(epoch)

            # Train one epoch
            train_loss = training_mgr.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=loop_config,
                vocab_size=model_config.get('vocab_size', 50680),
                tokenizer=tokenizer,
                generation_config=generation_config,
                coherence_config=config.get('coherence', {}),
            )

            if rank == 0:
                train_logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

            # Check if max_steps reached
            if training_mgr.reached_max_steps(loop_config):
                if rank == 0:
                    train_logger.info(f"Reached max_steps ({loop_config.max_steps}), stopping training")
                break

            # Validation
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                val_loss = validation_mgr.validate(
                    model=model,
                    val_loader=val_loader,
                    use_amp=context.use_amp,
                    amp_dtype=context.amp_dtype,
                )

                metrics_mgr.log_validation(
                    step=training_mgr.get_global_step(),
                    epoch=epoch,
                    val_loss=val_loss
                )

                if validation_mgr.is_best(val_loss):
                    best_val_loss = val_loss
                    if rank == 0:
                        train_logger.info(f"New best validation loss: {val_loss:.4f}")
                        # Save best checkpoint
                        run_manager.save_checkpoint(
                            model_state=model.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            epoch=epoch,
                            step=training_mgr.get_global_step(),
                            loss=val_loss,
                            is_best=True,
                            additional_data={'train_loss': train_loss}
                        )

            # Notify components of epoch end
            pipeline.on_epoch_end(epoch)

        # =====================================================================
        # Phase 12: Finalize
        # =====================================================================
        if rank == 0:
            train_logger.info("\n" + "=" * 60)
            train_logger.info("Training Complete")
            train_logger.info("=" * 60)
            train_logger.info(f"Best validation loss: {best_val_loss:.4f}")

            # Save final metrics
            metrics_mgr.save_summary(log_dir / 'metrics_summary.json')

            # Log final generations table
            generations = generation_mgr.get_generation_history()
            if generations:
                metrics_mgr.log_generation_table(generations)

            # Stop profiler and log output location
            profile_dir = training_mgr.stop_profiler()
            if profile_dir:
                train_logger.info(f"Profile data saved to: {profile_dir}")

        # Finish run
        run_manager.finish_run(status='completed')

        # Cleanup all components
        pipeline.cleanup_all()

        # Shutdown checkpoint manager (waits for pending async saves)
        checkpoint_manager.shutdown()

    except Exception as e:
        train_logger.error(f"Training failed: {e}", exc_info=True)
        pipeline.on_error(e)
        run_manager.finish_run(status='failed', final_metrics={'error': str(e)})
        # Still shutdown checkpoint manager on error
        checkpoint_manager.shutdown()
        raise

    finally:
        cleanup_distributed(rank, world_size)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ava Pipeline Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file'
    )

    # Training hyperparameters (override config)
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps (overrides epochs)')

    # Directories - default to project outputs directory
    default_output_dir = str(Path(__file__).resolve().parent.parent.parent / 'outputs')
    parser.add_argument('--save-dir', type=str, default=default_output_dir, help='Base output directory (runs saved to {save-dir}/pretraining/)')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')

    # Logging
    parser.add_argument('--log-interval', type=int, default=None, help='Steps between log messages')
    parser.add_argument('--val-interval', type=int, default=1, help='Epochs between validation')

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')

    # Profiling options for Nsight Systems/Compute
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable Nsight-compatible GPU profiling')
    parser.add_argument('--profile-dir', type=str, default='./profiles',
                       help='Directory to save profiling outputs')
    parser.add_argument('--profile-start-step', type=int, default=0,
                       help='Step to start profiling (default: 0, profiles all)')
    parser.add_argument('--profile-end-step', type=int, default=999999,
                       help='Step to end profiling (default: 999999, profiles all)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
