#!/usr/bin/env python3
"""
 Ava Fine-Tuning Pipeline - Automatic Latest Data & Checkpoint Discovery

Fine-tuning script that automatically discovers and uses:
1. Latest checkpoint from previous training runs (for continued training)
2. Latest data files from /root/Ava_AI/code/data/fine-tuning directory

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
    python finetune.py --checkpoint /root/Ava_AI/code/outputs/runs/run_XXX/checkpoints/step_12345/model.pt

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

# Auto-install requirements if needed (must be before other imports)
import subprocess
import sys
from pathlib import Path

def auto_install_requirements():
    """Auto-install requirements if imports fail"""
    project_root = Path(__file__).resolve().parents[2]
    requirements_file = project_root / "requirements.txt"

    print("🔧 Installing Python requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file), "--upgrade", "-q"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

# Try imports with auto-install
try:
    import argparse
    import logging
    import os
    import warnings
    from datetime import datetime
    from typing import List, Optional, Tuple
    import glob
except (ImportError, ModuleNotFoundError) as e:
    print(f"❌ Import error: {e}")
    print("🔧 Attempting to install requirements...")
    if auto_install_requirements():
        print("🔄 Retrying imports...")
        import argparse
        import logging
        import os
        import warnings
        from datetime import datetime
        from typing import List, Optional, Tuple
        import glob
    else:
        print("❌ Failed to install requirements. Please run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

# Suppress Pydantic field attribute warnings early (these come from dependencies)
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
    warnings.filterwarnings('ignore', category=UnsupportedFieldAttributeWarning)
except ImportError:
    # Pydantic v1 or older version without this warning class
    pass

import torch
import yaml

# Suppress socket warnings
warnings.filterwarnings("ignore", message="socket.send()")

# Ensure stdout/stderr are not buffered
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Configure unified logging for Ava (colored output, tqdm-compatible)
from ava.core.logging import setup_ava_logging
setup_ava_logging(level=logging.INFO)

# Suppress only specific noisy loggers
logging.getLogger("asyncio").setLevel(logging.ERROR)

from transformers import AutoTokenizer

# Import all training components from train.py
from ava.config import EnhancedTrainingConfig, TrainingConfigManager
# Feature compatibility module was removed
# from ava.config.feature_compatibility import (
#     print_compatibility_report,
#     validate_training_config,
# )
from ava.data.pretokenized import create_ultra_fast_dataloaders
from ava.models.moe import EnhancedMoEConfig, EnhancedMoEModel
from ava.data.multi_column import create_multi_column_dataloader
# Observability modules are not yet implemented:
# from ava.observability.health_dashboard import HealthDashboard
# from ava.observability.hierarchical_logging import HierarchicalLogger, LogLevel
# from ava.observability.training_validator import TrainingValidator
from ava.optimizations import AdaptiveLearningRateManager, AdaptiveLRConfig
from ava.training.progressive import (
    ProgressiveStrategyConfig,
    ProgressiveTrainingManager,
)
from ava.training.run_manager import RunManager
from ava.utils import register_cleanup_handlers
from ava.core.paths import get_project_root, get_data_dir

# Import the main training function from train.py
# We'll reuse most of its logic but with custom data loading
sys.path.insert(0, str(Path(__file__).parent))
project_root = get_project_root()


def find_latest_checkpoint(
    outputs_dir: Optional[Path] = None,
    checkpoint_name: str = "latest_model.pt",
    include_finetune: bool = False,
) -> Optional[Path]:
    """
    Find the latest checkpoint in the outputs directory.
    By default searches only in regular training runs directory.

    Args:
        outputs_dir: Base outputs directory containing runs (default: auto-detected)
        checkpoint_name: Name of checkpoint file (default: latest_model.pt)
        include_finetune: Also search in finetune_runs directory (default: False)

    Returns:
        Path to latest checkpoint, or None if not found
    """
    if outputs_dir is None:
        outputs_dir = project_root / "code" / "outputs" / "runs"

    checkpoint_files = []

    # Search in regular training runs (primary/default location)
    if outputs_dir.exists():
        # Search for both latest_model.pt and model.pt
        checkpoint_files.extend(list(outputs_dir.rglob(checkpoint_name)))
        checkpoint_files.extend(list(outputs_dir.rglob("model.pt")))
        print(f"    Searching in training directory: {outputs_dir}", flush=True)
    else:
        print(f"     Training directory not found: {outputs_dir}", flush=True)

    # Optionally also search in fine-tuning runs
    if include_finetune:
        finetune_dir = project_root / "code" / "outputs" / "finetune_runs"
        if finetune_dir.exists():
            checkpoint_files.extend(list(finetune_dir.rglob(checkpoint_name)))
            print(f"    Also searching in fine-tuning directory: {finetune_dir}", flush=True)

    if not checkpoint_files:
        print(f"     No checkpoints found matching '{checkpoint_name}'", flush=True)
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
    2. Fallback to default configs in project configs directory

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
            print(f"     Could not parse metadata: {e}", flush=True)

    # Final fallback: use small.yaml as default
    default_config = project_root / "code" / "configs" / "moe" / "minimal_working.yaml"
    if default_config.exists():
        print(f"   ℹ  Using default config: {default_config}", flush=True)
        return default_config

    return None


def print_checkpoint_info(checkpoint_path: Path) -> None:
    """Print information about the checkpoint to be loaded."""
    print("\n" + "=" * 80, flush=True)
    print(" LOADING PRETRAINED CHECKPOINT", flush=True)
    print("=" * 80, flush=True)

    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)

    print(f"\n Checkpoint: {checkpoint_path.name}", flush=True)
    print(f"   Path: {checkpoint_path}", flush=True)
    print(f"   Size: {size_mb:.2f} MB", flush=True)
    print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"   Run: {checkpoint_path.parent.parent.parent.name}", flush=True)
    print("=" * 80 + "\n", flush=True)


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
    print("\n" + "=" * 80, flush=True)
    print(" AUTO-DISCOVERED FINE-TUNING DATA FILES", flush=True)
    print("=" * 80, flush=True)

    total_size = 0
    for i, file in enumerate(files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        mod_time = datetime.fromtimestamp(file.stat().st_mtime)
        file_format = detect_file_format(file)

        print(f"\n{i}. {file.name}", flush=True)
        print(f"   Path: {file}", flush=True)
        print(f"   Size: {size_mb:.2f} MB", flush=True)
        print(f"   Format: {file_format}", flush=True)
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    print(f"\n Total: {len(files)} file(s), {total_size:.2f} MB", flush=True)
    print("=" * 80 + "\n", flush=True)


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

    Note: Fine-tuning data must be pre-tokenized Arrow/Parquet files.

    Args:
        tokenizer: Tokenizer for special token IDs (optional, can be None)
        batch_size: Batch size
        max_length: Maximum sequence length
        data_files: List of data files to load (must be Arrow/Parquet)
        buffer_size: Buffer size for loading
        num_workers: Number of dataloader workers
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from ava.data.pretokenized import create_ultra_fast_dataloaders

    # Use the first file's directory as base
    data_dir = data_files[0].parent

    print(f" Creating pretokenized dataloaders from {len(data_files)} file(s)...", flush=True)
    print(f"   Batch size: {batch_size}", flush=True)
    print(f"   Max length: {max_length}", flush=True)
    print(f"   Buffer size: {buffer_size}", flush=True)
    print(f"   Validation split: {val_split * 100:.1f}%", flush=True)

    # Get special token IDs from tokenizer or use defaults
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0) if tokenizer else 0
    bos_token_id = getattr(tokenizer, 'bos_token_id', 2) if tokenizer else 2
    eos_token_id = getattr(tokenizer, 'eos_token_id', 1) if tokenizer else 1

    # Use pretokenized Arrow loader
    train_loader, val_loader = create_ultra_fast_dataloaders(
        batch_size=batch_size,
        max_length=max_length,
        data_dir=str(data_dir),
        buffer_size=buffer_size,
        num_workers=num_workers,
        val_split_ratio=val_split,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
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
        default=None,  # Will be set to project_root/code/data/fine-tuning in main()
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
        default=None,  # Will be set to project_root/code/outputs/runs in main()
        help="Directory to search for checkpoints (default: auto-detected)",
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
        default=None,  # Will be set to project_root/code/outputs/finetune_runs in main()
        help="Output directory for fine-tuning checkpoints and logs (default: auto-detected)",
    )

    # Dependency checking
    parser.add_argument(
        "--skip-dep-check",
        action="store_true",
        help="Skip dependency check at startup",
    )

    return parser.parse_args()


def main():
    """Main fine-tuning entry point."""
    args = parse_args()

    # Dependency check
    if not args.skip_dep_check:
        try:
            import importlib.util
            dep_check_path = Path(__file__).parent.parent / "check_dependencies.py"
            if dep_check_path.exists():
                spec = importlib.util.spec_from_file_location("check_dependencies", dep_check_path)
                dep_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dep_module)
                dep_module.quick_check(warn_only=True)
        except Exception:
            pass  # Don't fail if dependency checker unavailable

    # Set default paths if not provided
    if args.data_dir is None:
        args.data_dir = str(project_root / "code" / "data" / "fine-tuning")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(project_root / "code" / "outputs" / "runs")
    if args.output_dir is None:
        args.output_dir = str(project_root / "code" / "outputs" / "finetune_runs")

    # Create timestamped run directory for this finetuning session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_folder_name = f"finetune_{args.run_name}_{timestamp}"
    else:
        run_folder_name = f"finetune_run_{timestamp}"

    # Create individual run directory under the base output directory
    base_output_dir = Path(args.output_dir)
    run_output_dir = base_output_dir / run_folder_name

    # Update args.output_dir to point to this run's directory
    args.output_dir = str(run_output_dir)

    # Force unbuffered output
    print("\n" + "=" * 80, flush=True)
    print(" AVA FINE-TUNING PIPELINE", flush=True)
    print("   Automatic Latest Data Discovery", flush=True)
    print(f"   Run Directory: {args.output_dir}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Create run directory structure
    run_output_dir.mkdir(parents=True, exist_ok=True)
    (run_output_dir / "checkpoints").mkdir(exist_ok=True)
    (run_output_dir / "logs").mkdir(exist_ok=True)
    (run_output_dir / "configs").mkdir(exist_ok=True)
    print(f" Created fine-tuning run directory: {run_output_dir}\n", flush=True)

    # 1. Discover latest data files
    data_dir = Path(args.data_dir)
    print(f" Searching for data in: {data_dir}", flush=True)
    print(f"   Pattern: {args.file_pattern}", flush=True)

    if args.use_all_files:
        print("   Mode: Using ALL files", flush=True)
        # Get all files (set a high number)
        latest_files = find_latest_files(
            data_dir,
            num_files=10000,  # Effectively unlimited
            file_pattern=args.file_pattern,
            exclude_pattern=args.exclude_pattern,
        )
    else:
        print(f"   Mode: Using latest {args.num_latest_files} file(s)", flush=True)
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
            print(f" Searching for latest checkpoint...", flush=True)
            checkpoint_path = find_latest_checkpoint(
                outputs_dir=Path(args.checkpoint_dir),
                checkpoint_name="latest_model.pt",
                include_finetune=args.include_finetune_checkpoints
            )
            if checkpoint_path:
                print_checkpoint_info(checkpoint_path)
            else:
                print("   ℹ  No checkpoint found - starting from scratch", flush=True)
    else:
        print("\n  --no-checkpoint specified - training from scratch\n", flush=True)

    # 3. Auto-discover or load configuration
    config_path = None

    if args.config:
        # Use explicitly specified config
        config_path = Path(args.config)
        print(f" Using specified configuration: {config_path}", flush=True)
    elif checkpoint_path:
        # Auto-discover config from checkpoint
        print(f" Auto-discovering configuration from checkpoint...", flush=True)
        config_path = find_config_for_checkpoint(checkpoint_path)
        if config_path:
            print(f"    Found config: {config_path}", flush=True)
        else:
            raise FileNotFoundError(
                "Could not find config for checkpoint. Please specify --config explicitly."
            )
    else:
        # No checkpoint and no config specified - use default
        config_path = project_root / "code" / "configs" / "moe" / "minimal_working.yaml"
        print(f" Using default configuration: {config_path}", flush=True)

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

    print(f" Configured data_dir: {fine_tuning_dir}", flush=True)
    print(f"   (Fine-tuning will ONLY use data from this directory)", flush=True)

    # 5. Configure checkpoint loading in config
    if checkpoint_path:
        if "run_management" not in config_dict:
            config_dict["run_management"] = {}
        config_dict["run_management"]["resume_from_checkpoint"] = str(checkpoint_path)
        print(f" Configured to load checkpoint: {checkpoint_path.name}", flush=True)

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
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dict["run_management"]["experiment_name"] = f"finetune_{timestamp_str}"

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
    import os

    # Create a temporary config file with our modifications
    # FIX: Close the file descriptor immediately - we use open() to write, not the fd
    temp_config_fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="finetune_config_")
    os.close(temp_config_fd)  # Close fd immediately to prevent leak
    temp_config_fd = None  # Mark as closed

    # CRITICAL FIX: Temporarily hide the processed data directory
    # to force train.py to use ONLY the fine-tuning directory
    processed_dir = project_root / "code" / "data" / "processed"
    processed_backup = project_root / "code" / "data" / ".processed_backup_for_finetuning"
    renamed_processed = False

    try:
        with open(temp_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"\n Created temporary config: {temp_config_path}", flush=True)
        print(f"   Verified data_dir in config dict: {config_dict['data']['data_dir']}", flush=True)

        # Double-check the written file
        with open(temp_config_path, "r") as f:
            verify_dict = yaml.safe_load(f)
            print(f"   Verified data_dir in written file: {verify_dict['data']['data_dir']}", flush=True)

        # Save a copy of the config to the run's configs directory for reference
        run_config_path = Path(args.output_dir) / "configs" / "training_config.yaml"
        with open(run_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"   Saved config copy to: {run_config_path}", flush=True)

        # Temporarily rename processed directory to force use of fine-tuning data only
        if processed_dir.exists() and not processed_backup.exists():
            print(f"\n Temporarily hiding {processed_dir} to ensure fine-tuning data only...", flush=True)
            processed_dir.rename(processed_backup)
            renamed_processed = True
            print(f"    Renamed to {processed_backup.name}", flush=True)

        # 5. Import and call the main training function from train.py
        print("\n Initializing fine-tuning with discovered data...", flush=True)
        print(f"   Using {len(latest_files)} data file(s) from fine-tuning directory ONLY", flush=True)
        print(f"   Base config: {config_path.name}", flush=True)
        print(f"   Data directory: {latest_files[0].parent}", flush=True)
        if checkpoint_path:
            print(f"   Resuming from: {checkpoint_path.name}", flush=True)

        # Import main from train_pipeline.py and run it
        try:
            from train_pipeline import main as train_main  # type: ignore[import]

            # Monkey-patch sys.argv to pass our temporary config
            original_argv = sys.argv
            sys.argv = [
                "train_pipeline.py",
                "--config", temp_config_path,
                "--data-dir", fine_tuning_dir  # CRITICAL: Force data directory via command line
            ]

            # Add optional args
            if args.enable_progressive_training:
                sys.argv.append("--enable-progressive-training")

            # Parse args for train_100m_full.py
            # We need to import the argument parser from train_100m_full
            # Since it's at module level, we'll need to create a compatible args object
            from types import SimpleNamespace

            training_args = SimpleNamespace(
                config=temp_config_path,
                data_dir=fine_tuning_dir,
                epochs=args.num_epochs if args.num_epochs else 10,
                batch_size=args.batch_size if args.batch_size else None,
                learning_rate=args.learning_rate if args.learning_rate else None,
                max_steps=args.max_steps if args.max_steps else None,
                resume=str(checkpoint_path) if checkpoint_path else None,
                use_turn_aware_loader=True,
                disable_turn_aware_loader=False,
                val_interval=1,
                save_dir=f"{args.output_dir}/checkpoints",
                log_dir=f"{args.output_dir}/logs",
                log_interval=10,
            )

            # Run training with the args object
            train_main(training_args)

            # Restore argv
            sys.argv = original_argv

        except ImportError as e:
            print(f"\n Error: Could not import train_100m_full.py: {e}", flush=True)
            print("   Make sure you're running from the scripts/5_training directory", flush=True)
            sys.exit(1)

    finally:
        # Restore processed directory if it was renamed
        if renamed_processed and processed_backup.exists():
            print(f"\n Restoring {processed_backup.name}...", flush=True)
            processed_backup.rename(processed_dir)
            print(f"    Restored to {processed_dir}", flush=True)

        # Clean up temporary config file
        import os
        try:
            # FIX: fd was already closed above, just unlink the file
            os.unlink(temp_config_path)
            print(f"  Cleaned up temporary config", flush=True)
        except OSError as e:
            # File may already be deleted - expected during cleanup
            logging.debug(f"Temp config cleanup: {e}")

    print("\n Fine-tuning complete!", flush=True)


if __name__ == "__main__":
    main()
