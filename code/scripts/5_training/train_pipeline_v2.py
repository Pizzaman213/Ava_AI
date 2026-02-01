#!/usr/bin/env python3
"""
Ava Pipeline Training Script v2 - Phase-based architecture.

This script provides a clean entry point for training using the phase-based
architecture. All training logic has been decomposed into modular phases
in ava.training.phases.

Phase Architecture:
    p00: Dependencies   - Check required packages
    p01: Distributed    - Setup DDP/single GPU
    p02: Config         - Load YAML config
    p03: RunManager     - Setup output directories
    p04: Context        - Create TrainingContext
    p05: Pipeline       - Register components
    p06: Model          - Build model + calibration
    p07: Optimizer      - Create optimizer + load checkpoint
    p08: Data           - Create dataloaders + scheduler
    p09: Metrics        - Setup WandB, logging
    p10: Components     - Validation, generation, diagnostics
    p11: Training       - Main training loop
    p12: Finalize       - Cleanup and summary

Usage:
    # Single GPU
    python train_pipeline_v2.py --config ../../configs/moe/large.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 train_pipeline_v2.py \\
        --config ../../configs/moe/large.yaml

    # Resume from checkpoint
    python train_pipeline_v2.py --config ../../configs/moe/large.yaml \\
        --resume /path/to/checkpoint.pt

Note:
    This is the Gen 2 refactored pipeline. The original monolithic script
    is available as train_pipeline.py for backward compatibility.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Silence HuggingFace tokenizers parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parents[1] / 'src'
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import phase infrastructure
from ava.training.phases import PhaseExecutor, PhaseContext
from ava.logging.console.colored import ColoredFormatter


def setup_main_logger(rank: int = 0) -> logging.Logger:
    """Setup logger for main script."""
    logger = logging.getLogger('train_pipeline_v2')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()

    if rank == 0:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(ColoredFormatter(show_level=False, show_icons=True))
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def main(args: argparse.Namespace) -> None:
    """
    Main training function using phase-based architecture.

    This function creates a PhaseContext with the CLI arguments and
    delegates all training logic to the PhaseExecutor, which runs
    each phase in sequence.

    Args:
        args: Parsed command line arguments
    """
    # Create initial logger (will be replaced with rank-aware logger in phases)
    logger = setup_main_logger()

    # Initialize phase context with args
    ctx = PhaseContext(args=args, config={})

    # Create executor and run all phases
    executor = PhaseExecutor(logger)

    try:
        ctx = executor.run_all(ctx)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        executor.handle_error(ctx, KeyboardInterrupt("User interrupted"))
    except Exception as e:
        executor.handle_error(ctx, e)
        raise
    finally:
        executor.finalize(ctx)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ava Pipeline Training Script v2 (Phase-based)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to YAML config file'
    )

    # Training hyperparameters (override config)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum training steps (overrides epochs)')

    # Directories
    default_output_dir = str(Path(__file__).resolve().parent.parent.parent / 'outputs')
    parser.add_argument('--save-dir', type=str, default=default_output_dir,
                        help='Base output directory (runs saved to {save-dir}/pretraining/)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Log directory')

    # Logging
    parser.add_argument('--log-interval', type=int, default=None,
                        help='Steps between log messages')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epochs between validation')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--require-resume', action='store_true',
                        help='Fail if resume checkpoint is missing (instead of starting fresh)')

    # Profiling
    parser.add_argument('--enable-profiling', action='store_true',
                        help='Enable Nsight-compatible GPU profiling')
    parser.add_argument('--profile-dir', type=str, default='./profiles',
                        help='Directory to save profiling outputs')
    parser.add_argument('--profile-start-step', type=int, default=0,
                        help='Step to start profiling (default: 0, profiles all)')
    parser.add_argument('--profile-end-step', type=int, default=999999,
                        help='Step to end profiling (default: 999999, profiles all)')

    # Dependency checking
    parser.add_argument('--skip-dep-check', action='store_true',
                        help='Skip dependency check at startup')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
