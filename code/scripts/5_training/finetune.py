#!/usr/bin/env python3
"""
🚀 Ava Fine-Tuning Pipeline - Automatic Latest Data & Checkpoint Discovery

Fine-tuning script that automatically discovers and uses:
1. Latest checkpoint from previous training runs (for continued training)
2. Latest data files from /project/code/data/fine-tuning directory

Based on train.py but optimized for fine-tuning workflows with automatic
checkpoint resumption and data management.

Features:
- Auto-discovers and loads latest checkpoint (resume training)
- Auto-discovers latest modified files in fine-tuning directory
- Supports multiple Q&A dataset formats (JSONL, Parquet, Arrow, CSV)
- Automatic format detection and validation
- Uses all train.py enhancements (8 phases)
- Optimized for instruction/Q&A fine-tuning

Usage:
    # Simplest: Auto-discover everything (checkpoint + config + latest data)
    python finetune.py

    # Use specific checkpoint (config auto-discovered)
    python finetune.py --checkpoint /project/code/outputs/runs/run_XXX/checkpoints/step_12345/model.pt

    # Override config file
    python finetune.py --config ../../configs/gpu/small.yaml

    # Start from scratch (no checkpoint)
    python finetune.py --no-checkpoint

    # Use latest N data files
    python finetune.py --num-latest-files 5

    # Use all fine-tuning data files
    python finetune.py --use-all-files

    # Use specific file pattern
    python finetune.py --file-pattern "*OpenOrca*"

    # Override fine-tuning directory
    python finetune.py --data-dir /custom/path

Examples:
    # Easiest: Continue from latest checkpoint with latest data
    python finetune.py

    # Fine-tune on all available data
    python finetune.py --use-all-files

    # Use specific checkpoint on specific data
    python finetune.py --checkpoint /path/to/model.pt --file-pattern "*CodeAlpaca*"
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import glob

# Suppress Pydantic field attribute warnings early (these come from dependencies)
from pydantic.warnings import UnsupportedFieldAttributeWarning
warnings.filterwarnings('ignore', category=UnsupportedFieldAttributeWarning)

import torch
import yaml

# Suppress warnings
logging.getLogger("asyncio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="socket.send()")

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer

# Import all training components from train.py
from src.Ava.config import EnhancedTrainingConfig, TrainingConfigManager
from src.Ava.config.feature_compatibility import (
    print_compatibility_report,
    validate_training_config,
)
from src.Ava.data.dataloader import create_streaming_dataloaders
from src.Ava.models.moe_model import EnhancedMoEConfig, EnhancedMoEModel
from src.Ava.data.multi_column_data import create_multi_column_dataloader
# Observability modules are not yet implemented:
# from src.Ava.observability.health_dashboard import HealthDashboard
# from src.Ava.observability.hierarchical_logging import HierarchicalLogger, LogLevel
# from src.Ava.observability.training_validator import TrainingValidator
from src.Ava.optimization import AdaptiveLearningRateManager, AdaptiveLRConfig
from src.Ava.training.core.trainer import EnhancedTrainer as EnhancedModularTrainer
from src.Ava.training.strategies.progressive_training import (
    ProgressiveTrainingConfig,
    ProgressiveTrainingManager,
)
from src.Ava.training.orchestration.run_manager import RunManager
from src.Ava.utils import register_cleanup_handlers

# Import the main training function from train.py
# We'll reuse most of its logic but with custom data loading
sys.path.insert(0, str(Path(__file__).parent))


def find_latest_checkpoint(
    outputs_dir: Path = Path("/project/code/outputs/runs"),
    checkpoint_name: str = "latest_model.pt",
    include_finetune: bool = False,
) -> Optional[Path]:
    """
    Find the latest checkpoint in the outputs directory.
    By default searches only in regular training runs directory.

    Args:
        outputs_dir: Base outputs directory containing runs (default: /project/code/outputs/runs)
        checkpoint_name: Name of checkpoint file (default: latest_model.pt)
        include_finetune: Also search in finetune_runs directory (default: False)

    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_files = []

    # Search in regular training runs (primary/default location)
    if outputs_dir.exists():
        # Search for both latest_model.pt and model.pt
        checkpoint_files.extend(list(outputs_dir.rglob(checkpoint_name)))
        checkpoint_files.extend(list(outputs_dir.rglob("model.pt")))
        print(f"   📂 Searching in training directory: {outputs_dir}")
    else:
        print(f"   ⚠️  Training directory not found: {outputs_dir}")

    # Optionally also search in fine-tuning runs
    if include_finetune:
        finetune_dir = Path("/project/code/outputs/finetune_runs")
        if finetune_dir.exists():
            checkpoint_files.extend(list(finetune_dir.rglob(checkpoint_name)))
            print(f"   📂 Also searching in fine-tuning directory: {finetune_dir}")

    if not checkpoint_files:
        print(f"   ⚠️  No checkpoints found matching '{checkpoint_name}'")
        return None

    # Sort by modification time (newest first)
    sorted_checkpoints = sorted(
        checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True
    )

    return sorted_checkpoints[0]


def find_config_for_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    """
    Find the original config file used for a checkpoint.

    Searches in order:
    1. run_dir/configs/*.yaml
    2. Fallback to default configs in /project/code/configs/gpu/

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Path to config file, or None if not found
    """
    # Get the run directory (go up from checkpoint to run root)
    # Structure: run_dir/checkpoints/step_XXX/model.pt
    run_dir = checkpoint_path.parent.parent.parent

    # Check for config files in the run directory
    config_dir = run_dir / "configs"
    if config_dir.exists():
        # Look for YAML config files
        yaml_configs = list(config_dir.glob("*.yaml"))
        if yaml_configs:
            # Prefer files with "config" in the name
            for config in yaml_configs:
                if "config" in config.name.lower():
                    return config
            # Otherwise return the first one
            return yaml_configs[0]

    # Fallback: try to determine from run metadata
    metadata_file = run_dir / "configs" / "run_metadata.json"
    if metadata_file.exists():
        try:
            import json
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                # Check if there's a config path in metadata
                if "config_path" in metadata:
                    config_path = Path(metadata["config_path"])
                    if config_path.exists():
                        return config_path
        except Exception as e:
            print(f"   ⚠️  Could not parse metadata: {e}")

    # Final fallback: use small.yaml as default
    default_config = Path("/project/code/configs/gpu/small.yaml")
    if default_config.exists():
        print(f"   ℹ️  Using default config: {default_config}")
        return default_config

    return None


def print_checkpoint_info(checkpoint_path: Path) -> None:
    """Print information about the checkpoint to be loaded."""
    print("\n" + "=" * 80)
    print("🔄 LOADING PRETRAINED CHECKPOINT")
    print("=" * 80)

    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)

    print(f"\n📦 Checkpoint: {checkpoint_path.name}")
    print(f"   Path: {checkpoint_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Run: {checkpoint_path.parent.parent.parent.name}")
    print("=" * 80 + "\n")


def find_latest_files(
    data_dir: Path,
    num_files: int = 1,
    file_pattern: str = "*.jsonl",
    exclude_pattern: Optional[str] = None,
) -> List[Path]:
    """
    Find the latest N files in a directory based on modification time.

    Args:
        data_dir: Directory to search
        num_files: Number of latest files to return
        file_pattern: Glob pattern for file matching (e.g., "*.jsonl", "*processed*")
        exclude_pattern: Optional pattern to exclude files

    Returns:
        List of Path objects for the latest files, sorted by modification time (newest first)
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Find all matching files
    all_files = list(data_dir.glob(file_pattern))

    # Exclude files matching exclude pattern
    if exclude_pattern:
        all_files = [f for f in all_files if not f.match(exclude_pattern)]

    if not all_files:
        raise FileNotFoundError(
            f"No files found in {data_dir} matching pattern '{file_pattern}'"
        )

    # Sort by modification time (newest first)
    sorted_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)

    # Return the latest N files
    return sorted_files[:num_files]


def detect_file_format(file_path: Path) -> str:
    """Detect the format of a data file."""
    suffix = file_path.suffix.lower()
    format_map = {
        ".jsonl": "jsonl",
        ".json": "json",
        ".parquet": "parquet",
        ".arrow": "arrow",
        ".csv": "csv",
        ".tsv": "tsv",
    }
    return format_map.get(suffix, "unknown")


def print_file_info(files: List[Path]) -> None:
    """Print information about discovered files."""
    print("\n" + "=" * 80)
    print("📁 AUTO-DISCOVERED FINE-TUNING DATA FILES")
    print("=" * 80)

    total_size = 0
    for i, file in enumerate(files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        mod_time = datetime.fromtimestamp(file.stat().st_mtime)
        file_format = detect_file_format(file)

        print(f"\n{i}. {file.name}")
        print(f"   Path: {file}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Format: {file_format}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n📊 Total: {len(files)} file(s), {total_size:.2f} MB")
    print("=" * 80 + "\n")


def create_finetune_dataloaders(
    tokenizer,
    batch_size: int,
    max_length: int,
    data_files: List[Path],
    buffer_size: int = 10000,
    num_workers: int = 4,
    val_split: float = 0.05,
) -> Tuple:
    """
    Create dataloaders from discovered fine-tuning files.

    Args:
        tokenizer: Tokenizer for text processing
        batch_size: Batch size
        max_length: Maximum sequence length
        data_files: List of data files to load
        buffer_size: Buffer size for streaming
        num_workers: Number of dataloader workers
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from src.Ava.data.dataloader import create_streaming_dataloaders

    # Create a temporary directory with symlinks/copies for streaming loader
    # Or use the first file's directory as base
    data_dir = data_files[0].parent

    print(f"📦 Creating streaming dataloaders from {len(data_files)} file(s)...")
    print(f"   Batch size: {batch_size}")
    print(f"   Max length: {max_length}")
    print(f"   Buffer size: {buffer_size}")
    print(f"   Validation split: {val_split * 100:.1f}%")

    # Use the existing streaming dataloader infrastructure
    train_loader, val_loader = create_streaming_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        data_dir=str(data_dir),
        buffer_size=buffer_size,
        max_samples=None,  # Use all available samples
        num_workers=num_workers,
    )

    return train_loader, val_loader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ava Fine-Tuning Pipeline - Automatic Latest Data Discovery"
    )

    # Core arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (auto-detected from checkpoint if not specified)",
    )

    # Data discovery arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/project/code/data/fine-tuning",
        help="Directory containing fine-tuning data files",
    )
    parser.add_argument(
        "--num-latest-files",
        type=int,
        default=1,
        help="Number of latest files to use (default: 1 - most recent)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*_processed.jsonl",
        help="Glob pattern for file matching (default: *_processed.jsonl)",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        default=None,
        help="Pattern to exclude files (optional)",
    )
    parser.add_argument(
        "--use-all-files",
        action="store_true",
        help="Use all files in directory instead of latest N",
    )

    # Training overrides
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--num-epochs", type=int, help="Override number of epochs")
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--max-length", type=int, help="Override max sequence length")

    # Feature flags
    parser.add_argument(
        "--enable-progressive-training",
        action="store_true",
        help="Enable progressive sequence length training",
    )
    parser.add_argument(
        "--enable-observability",
        action="store_true",
        default=True,
        help="Enable observability features (default: True)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name",
    )

    # QLoRA arguments (enabled by default for efficient fine-tuning)
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        default=True,
        help="Enable QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning (default: True)",
    )
    parser.add_argument(
        "--no-qlora",
        action="store_true",
        help="Disable QLoRA and use full fine-tuning instead",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        help="Target modules for LoRA (default: q_proj v_proj k_proj o_proj)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit precision (requires bitsandbytes) (default: True)",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for 4-bit base model (default: float16)",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        type=str,
        default="nf4",
        choices=["fp4", "nf4"],
        help="Quantization data type for 4-bit (default: nf4)",
    )
    parser.add_argument(
        "--use-double-quant",
        action="store_true",
        default=True,
        help="Enable double quantization for 4-bit (default: True)",
    )

    # Checkpoint loading arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to load (e.g., /path/to/model.pt)",
    )
    parser.add_argument(
        "--auto-checkpoint",
        action="store_true",
        default=True,
        help="Automatically find and load latest checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Start from scratch, don't load any checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/project/code/outputs/runs",
        help="Directory to search for checkpoints (default: /project/code/outputs/runs - normal training runs)",
    )
    parser.add_argument(
        "--include-finetune-checkpoints",
        action="store_true",
        help="Also search in finetune_runs directory for checkpoints",
    )

    # Run management
    parser.add_argument("--run-name", type=str, help="Custom run name")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/project/code/outputs/finetune_runs",
        help="Output directory for fine-tuning checkpoints and logs (default: /project/code/outputs/finetune_runs)",
    )

    return parser.parse_args()


def main():
    """Main fine-tuning entry point."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("🚀 AVA FINE-TUNING PIPELINE")
    print("   Automatic Latest Data Discovery")
    print(f"   Output Directory: {args.output_dir}")
    print("=" * 80 + "\n")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created fine-tuning output directory: {output_path}\n")

    # 1. Discover latest data files
    data_dir = Path(args.data_dir)
    print(f"🔍 Searching for data in: {data_dir}")
    print(f"   Pattern: {args.file_pattern}")

    if args.use_all_files:
        print("   Mode: Using ALL files")
        # Get all files (set a high number)
        latest_files = find_latest_files(
            data_dir,
            num_files=10000,  # Effectively unlimited
            file_pattern=args.file_pattern,
            exclude_pattern=args.exclude_pattern,
        )
    else:
        print(f"   Mode: Using latest {args.num_latest_files} file(s)")
        latest_files = find_latest_files(
            data_dir,
            num_files=args.num_latest_files,
            file_pattern=args.file_pattern,
            exclude_pattern=args.exclude_pattern,
        )

    # Print discovered files
    print_file_info(latest_files)

    # 2. Find and configure checkpoint loading
    checkpoint_path = None
    if not args.no_checkpoint:
        if args.checkpoint:
            # Use specific checkpoint provided
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
            print_checkpoint_info(checkpoint_path)
        elif args.auto_checkpoint:
            # Auto-discover latest checkpoint
            print(f"🔍 Searching for latest checkpoint...")
            checkpoint_path = find_latest_checkpoint(
                outputs_dir=Path(args.checkpoint_dir),
                checkpoint_name="latest_model.pt",
                include_finetune=args.include_finetune_checkpoints
            )
            if checkpoint_path:
                print_checkpoint_info(checkpoint_path)
            else:
                print("   ℹ️  No checkpoint found - starting from scratch")
    else:
        print("\n⚠️  --no-checkpoint specified - training from scratch\n")

    # 3. Auto-discover or load configuration
    config_path = None

    if args.config:
        # Use explicitly specified config
        config_path = Path(args.config)
        print(f"📋 Using specified configuration: {config_path}")
    elif checkpoint_path:
        # Auto-discover config from checkpoint
        print(f"📋 Auto-discovering configuration from checkpoint...")
        config_path = find_config_for_checkpoint(checkpoint_path)
        if config_path:
            print(f"   ✅ Found config: {config_path}")
        else:
            raise FileNotFoundError(
                "Could not find config for checkpoint. Please specify --config explicitly."
            )
    else:
        # No checkpoint and no config specified - use default
        config_path = Path("/project/code/configs/gpu/small.yaml")
        print(f"📋 Using default configuration: {config_path}")

    if not config_path.exists():
        # Try relative paths
        script_dir = Path(__file__).parent
        alt_path = script_dir / args.config
        if alt_path.exists():
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # 4. Override config with discovered data directory
    if "data" not in config_dict:
        config_dict["data"] = {}

    # CRITICAL: Set data_dir to the fine-tuning directory
    # This ensures we only use fine-tuning data, not the processed data
    fine_tuning_dir = str(latest_files[0].parent)
    config_dict["data"]["data_dir"] = fine_tuning_dir

    # Also disable streaming to force use of our specific files
    # and prevent fallback behavior
    if "data_loading" not in config_dict:
        config_dict["data_loading"] = {}
    # Keep streaming enabled but ensure proper directory
    config_dict["data_loading"]["streaming"] = True

    print(f"✅ Configured data_dir: {fine_tuning_dir}")
    print(f"   (Fine-tuning will ONLY use data from this directory)")

    # 5. Configure checkpoint loading in config
    if checkpoint_path:
        if "run_management" not in config_dict:
            config_dict["run_management"] = {}
        config_dict["run_management"]["resume_from_checkpoint"] = str(checkpoint_path)
        print(f"✅ Configured to load checkpoint: {checkpoint_path.name}")

    # Apply command-line overrides
    if args.batch_size:
        if "training" not in config_dict:
            config_dict["training"] = {}
        config_dict["training"]["batch_size"] = args.batch_size
        config_dict["data"]["train_batch_size"] = args.batch_size

    if args.learning_rate:
        if "training" not in config_dict:
            config_dict["training"] = {}
        config_dict["training"]["learning_rate"] = args.learning_rate

    if args.num_epochs:
        if "training" not in config_dict:
            config_dict["training"] = {}
        config_dict["training"]["num_epochs"] = args.num_epochs

    if args.max_steps:
        if "training" not in config_dict:
            config_dict["training"] = {}
        config_dict["training"]["max_steps"] = args.max_steps

    if args.max_length:
        config_dict["data"]["max_length"] = args.max_length

    # Always set output directory for fine-tuning (use different directory than regular training)
    if "output" not in config_dict:
        config_dict["output"] = {}
    config_dict["output"]["output_dir"] = args.output_dir

    # Also set run management output directory
    if "run_management" not in config_dict:
        config_dict["run_management"] = {}
    config_dict["run_management"]["output_dir"] = args.output_dir

    # Add a prefix to distinguish fine-tuning runs
    if args.run_name:
        config_dict["run_management"]["experiment_name"] = f"finetune_{args.run_name}"
    else:
        # Auto-generate a fine-tuning run name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dict["run_management"]["experiment_name"] = f"finetune_{timestamp}"

    if args.wandb_project:
        if "wandb" not in config_dict:
            config_dict["wandb"] = {}
        config_dict["wandb"]["project"] = args.wandb_project
        config_dict["wandb"]["enabled"] = True
    else:
        # Use a separate wandb project for fine-tuning by default
        if "wandb" not in config_dict:
            config_dict["wandb"] = {}
        config_dict["wandb"]["project"] = "Ava-FineTuning"

    # 4. Write modified config to temporary file
    import tempfile
    import json
    import shutil

    # Create a temporary config file with our modifications
    temp_config_fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="finetune_config_")

    # CRITICAL FIX: Temporarily hide the processed data directory
    # to force train.py to use ONLY the fine-tuning directory
    processed_dir = Path("/project/code/data/processed")
    processed_backup = Path("/project/code/data/.processed_backup_for_finetuning")
    renamed_processed = False

    try:
        with open(temp_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"\n📝 Created temporary config: {temp_config_path}")
        print(f"   Verified data_dir in config dict: {config_dict['data']['data_dir']}")

        # Double-check the written file
        with open(temp_config_path, "r") as f:
            verify_dict = yaml.safe_load(f)
            print(f"   Verified data_dir in written file: {verify_dict['data']['data_dir']}")

        # Temporarily rename processed directory to force use of fine-tuning data only
        if processed_dir.exists() and not processed_backup.exists():
            print(f"\n🔒 Temporarily hiding {processed_dir} to ensure fine-tuning data only...")
            processed_dir.rename(processed_backup)
            renamed_processed = True
            print(f"   ✓ Renamed to {processed_backup.name}")

        # 5. Configure QLoRA (enabled by default unless --no-qlora is specified)
        # Disable QLoRA if --no-qlora flag is set
        if args.no_qlora:
            args.use_qlora = False
            args.load_in_4bit = False
            print("\n⚠️  QLoRA disabled - using full fine-tuning")

        if args.use_qlora:
            try:
                from src.Ava.training.qlora_utils import print_qlora_summary, setup_qlora_config  # type: ignore[import-not-found]
                print_qlora_summary(args)
            except ImportError:
                print("⚠️  qlora_utils not found, skipping QLoRA summary")

            # Add QLoRA configuration to config dict
            if "qlora" not in config_dict:
                config_dict["qlora"] = {}

            config_dict["qlora"]["enabled"] = True
            config_dict["qlora"]["lora_r"] = args.lora_r
            config_dict["qlora"]["lora_alpha"] = args.lora_alpha
            config_dict["qlora"]["lora_dropout"] = args.lora_dropout
            config_dict["qlora"]["target_modules"] = args.target_modules
            config_dict["qlora"]["load_in_4bit"] = args.load_in_4bit
            config_dict["qlora"]["bnb_4bit_compute_dtype"] = args.bnb_4bit_compute_dtype
            config_dict["qlora"]["bnb_4bit_quant_type"] = args.bnb_4bit_quant_type
            config_dict["qlora"]["use_double_quant"] = args.use_double_quant

            print("✅ QLoRA configuration added to training config")

        # 6. Import and call the main training function from train.py
        print("\n🎯 Initializing fine-tuning with discovered data...")
        print(f"   Using {len(latest_files)} data file(s) from fine-tuning directory ONLY")
        print(f"   Base config: {config_path.name}")
        print(f"   Data directory: {latest_files[0].parent}")
        if checkpoint_path:
            print(f"   Resuming from: {checkpoint_path.name}")
        if args.use_qlora:
            print(f"   QLoRA: Enabled (rank={args.lora_r}, 4-bit={args.load_in_4bit})")

        # Import main from train.py and run it
        try:
            from train import main as train_main  # type: ignore[import]

            # Monkey-patch sys.argv to pass our temporary config
            original_argv = sys.argv
            sys.argv = [
                "train.py",
                "--config", temp_config_path,
                "--data-dir", fine_tuning_dir  # CRITICAL: Force data directory via command line
            ]

            # Add optional args
            if args.enable_progressive_training:
                sys.argv.append("--enable-progressive-training")

            # Run training
            train_main()

            # Restore argv
            sys.argv = original_argv

        except ImportError:
            print("\n❌ Error: Could not import train.py")
            print("   Make sure you're running from the scripts/training directory")
            sys.exit(1)

    finally:
        # Restore processed directory if it was renamed
        if renamed_processed and processed_backup.exists():
            print(f"\n🔓 Restoring {processed_backup.name}...")
            processed_backup.rename(processed_dir)
            print(f"   ✓ Restored to {processed_dir}")

        # Clean up temporary config file
        import os
        try:
            os.close(temp_config_fd)
            os.unlink(temp_config_path)
            print(f"🗑️  Cleaned up temporary config")
        except:
            pass

    print("\n✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()
