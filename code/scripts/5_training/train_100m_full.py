#!/usr/bin/env python3
"""
Full-Featured Training Script for 100M Parameter Model
=========================================================

This script demonstrates a production-ready training pipeline with:
 Model initialization and management
 Data loading and preprocessing
 Distributed training support
 Learning rate scheduling
 Checkpointing and resumption
 Metrics tracking and logging
 Mixed precision training
 Gradient accumulation
 Validation and evaluation
 Weights & Biases integration (optional)
 DeepSpeed integration (optional)

Usage:
    # Basic training
    python train_100m_full.py --config configs/moe/tiny_moe.yaml

    # With custom parameters
    python train_100m_full.py \
        --config configs/moe/tiny_moe.yaml \
        --epochs 10 \
        --batch-size 32 \
        --learning-rate 5e-5 \
        --save-dir ./checkpoints \
        --log-interval 100

    # Distributed training (multi-GPU)
    torchrun --nproc_per_node=4 train_100m_full.py \
        --config configs/moe/tiny_moe.yaml

    # Resume from checkpoint
    python train_100m_full.py \
        --config configs/moe/tiny_moe.yaml \
        --resume ./checkpoints/model_epoch_5.pt

    # Train with turn-aware conversation loading (ENABLED BY DEFAULT)
    python train_100m_full.py \
        --config configs/moe/tiny_moe.yaml
    # ^ Uses turn-aware loading automatically for conversation datasets

    # Train WITHOUT turn-aware loading (use standard loader)
    python train_100m_full.py \
        --config configs/moe/tiny_moe.yaml \
        --disable-turn-aware-loader
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
    import os
    import torch
    import json
    import logging
    import gc
    from datetime import datetime
    from typing import Optional, Dict, Tuple, Any
    import argparse
except (ImportError, ModuleNotFoundError) as e:
    print(f"❌ Import error: {e}")
    print("🔧 Attempting to install requirements...")
    if auto_install_requirements():
        print("🔄 Retrying imports...")
        import os
        import torch
        import json
        import logging
        from datetime import datetime
        from typing import Optional, Dict, Tuple, Any
        import argparse
    else:
        print("❌ Failed to install requirements. Please run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

# Configure PyTorch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import MoE model and utilities
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.config.yaml_loader import load_yaml_with_path_resolution
from src.Ava.training.train.data_loader_manager import DataLoaderManager
from src.Ava.training.train.base import TrainingContext

# Import optimization utilities
try:
    from src.Ava.training.optimizations import DynamicBatchScheduler, AdaptiveGradientAccumulation
    DYNAMIC_BATCHING_AVAILABLE = True
except ImportError:
    DYNAMIC_BATCHING_AVAILABLE = False

# Import turn-aware loader (optional, for improved conversation coherence)
try:
    from src.Ava.data.conversation_turn_loader import (
        TurnAwareConversationDataLoader
    )
    TURN_AWARE_LOADER_AVAILABLE = True
except ImportError:
    TURN_AWARE_LOADER_AVAILABLE = False

try:
    from torch.distributed import init_process_group, destroy_process_group
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# INT8 quantization with bitsandbytes
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

# NVFP4/NF4 quantization with torchao
try:
    import torchao
    from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Import colored logging utilities
try:
    from src.Ava.utils.colored_logging import (
        ColoredFormatter,
        CleanFormatter,
        configure_root_logger,
        supports_color,
    )
    COLORED_LOGGING_AVAILABLE = True
except ImportError:
    COLORED_LOGGING_AVAILABLE = False


def setup_logging(log_dir: Path, rank: int = 0) -> logging.Logger:
    """Setup logging for training with colored console output."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('train_100m')

    # Clear existing handlers to avoid duplicates on re-initialization
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # CRITICAL: Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Only log on rank 0 in distributed training
    if rank == 0:
        # Console handler with colors (if available)
        console_handler = logging.StreamHandler()
        if COLORED_LOGGING_AVAILABLE:
            console_handler.setFormatter(ColoredFormatter(show_level=False))
        else:
            formatter = logging.Formatter(
                '[%(asctime)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (plain format with full timestamps)
        file_handler = logging.FileHandler(
            log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        if COLORED_LOGGING_AVAILABLE:
            file_handler.setFormatter(CleanFormatter(include_date=True))
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        logger.addHandler(file_handler)

    # Configure root logger to reduce noise from other modules
    if COLORED_LOGGING_AVAILABLE:
        configure_root_logger(level=logging.WARNING)

    return logger


# ============================================================================
# DATA LOADING
# ============================================================================

class DummyDataset(Dataset):
    """Dummy dataset for demonstration. Replace with real data."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones(self.seq_length)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def create_dataloaders(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_train_samples: int = None,  # None = use all available data
    num_val_samples: int = None,    # None = use all available data
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    drop_last: bool = True,
    data_dir: str = None,
    use_turn_aware_loader: bool = False,  # Use turn-aware conversation loading
    tokenizer = None,  # Tokenizer for turn-aware loader
    use_all_data: bool = True,  # Use all available parquet files
    dynamic_batching_config: dict = None,  # Dynamic batching configuration
) -> Tuple[Any, Any]:  # Returns DataLoader or DynamicBatchIterator
    """Create training and validation dataloaders.

    Supports two modes:
    1. Standard Arrow/Parquet loading (default)
    2. Turn-aware conversation loading (improved coherence)

    If num_train_samples is None, generates unlimited samples.
    """

    # ==== TURN-AWARE CONVERSATION LOADING (NEW FEATURE) ====
    if use_turn_aware_loader and TURN_AWARE_LOADER_AVAILABLE and tokenizer:
        """Use turn-aware conversation loading for improved dialogue coherence.

        This mode:
        - Preserves conversation structure (no splitting across batches)
        - Adds speaker markers (<user>, <assistant>)
        - Enables quality-weighted training
        - Tracks conversation metadata
        """
        from pathlib import Path as PathlibPath

        # Find conversation JSONL files
        data_path = PathlibPath(data_dir) if data_dir else PathlibPath("code/data/processed")

        # Look for JSONL conversation files
        jsonl_files = list(data_path.glob("*_processed.jsonl"))

        if jsonl_files:
            # Use first available JSONL file for training
            train_jsonl = str(jsonl_files[0])

            try:
                train_loader = TurnAwareConversationDataLoader.create_dataloader(
                    data_path=train_jsonl,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    max_length=seq_length,
                    num_workers=num_workers,
                    shuffle=(rank == 0),
                    min_turns=1,
                    quality_threshold=0.0,
                )

                # Create validation loader (same file, but with smaller batches)
                val_loader = TurnAwareConversationDataLoader.create_dataloader(
                    data_path=train_jsonl,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    max_length=seq_length,
                    num_workers=0,
                    shuffle=False,
                    min_turns=1,
                    quality_threshold=0.0,
                )

                print(f"\n Turn-Aware Conversation Loading ENABLED")
                print(f"  Data: {train_jsonl}")
                print(f"  Benefits: Improved dialogue coherence, speaker awareness, quality tracking")

                return train_loader, val_loader
            except Exception as e:
                print(f"\n Turn-aware loading failed ({e}), falling back to standard loading")
        else:
            print(f"\n No conversation JSONL files found in {data_dir}, using standard loading")

    # ==== STANDARD ARROW/PARQUET LOADING (DEFAULT) ====
    # Create a simple wrapper to handle tokenization
    class ArrowDataset:
        def __init__(self, dataset, seq_length, vocab_size):
            self.dataset = dataset
            self.seq_length = seq_length
            self.vocab_size = vocab_size

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Assume the Arrow file has 'input_ids' and 'attention_mask'
            input_ids = torch.tensor(item.get('input_ids', []), dtype=torch.long)
            attention_mask = torch.tensor(item.get('attention_mask', []), dtype=torch.long)

            # Pad or truncate to seq_length
            if len(input_ids) < self.seq_length:
                pad_len = self.seq_length - len(input_ids)
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len))
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len))
            else:
                input_ids = input_ids[:self.seq_length]
                attention_mask = attention_mask[:self.seq_length]

            return {
                'input_ids': input_ids,
                'labels': input_ids.clone(),
                'attention_mask': attention_mask,
            }

    # Initialize datasets
    train_dataset = None
    val_dataset = None

    # Fix double-prefixed paths (e.g., /project/code/code/... -> /project/code/...)
    if data_dir and '/code/code' in data_dir:
        data_dir = data_dir.replace('/code/code/', '/code/')
        print(f"  Fixed double-prefixed path to: {data_dir}")

    # Log data loading configuration
    if rank == 0:
        print(f" Data directory: {data_dir}")
        print(f" Using all data: {use_all_data}")
        print(f" Data dir exists: {Path(data_dir).exists()}")
        print(f" Datasets available: {DATASETS_AVAILABLE}")

    # Try to load real data first
    if data_dir and Path(data_dir).exists() and DATASETS_AVAILABLE:
        try:
            if rank == 0:
                print(" Starting data loading...")

            # Suppress verbose dataset loading messages for faster loading
            import datasets as ds_lib
            ds_lib.logging.set_verbosity_error()

            datasets_list = []

            # Check if premade train/val splits exist
            train_dir = Path(data_dir) / 'train'
            val_dir = Path(data_dir) / 'val'
            use_premade_splits = train_dir.exists() and val_dir.exists()

            if use_premade_splits and rank == 0:
                print(f" Using premade train/val splits from dataset")

            train_datasets_list = []
            val_datasets_list = []

            # Load Arrow files (.arrow directories and files) if use_all_data is True
            if use_all_data:
                if use_premade_splits:
                    # Load from train split
                    arrow_files = sorted(list(train_dir.glob('*.arrow'))) + \
                                 sorted(list(train_dir.glob('**/*.arrow')))
                else:
                    arrow_files = sorted(list(Path(data_dir).glob('*.arrow'))) + \
                                 sorted(list(Path(data_dir).glob('**/*.arrow')))

                if rank == 0 and arrow_files:
                    print(f" Found {len(arrow_files)} Arrow files in train split" if use_premade_splits else f" Found {len(arrow_files)} Arrow files")
                for arrow_file in arrow_files:
                    try:
                        file_size = arrow_file.stat().st_size
                        if file_size == 0:
                            continue
                        dataset = load_dataset('arrow', data_files=str(arrow_file))
                        split_name = list(dataset.keys())[0]
                        dataset = dataset[split_name]
                        train_datasets_list.append(dataset)
                    except Exception as e:
                        continue

            # Load Parquet files for training
            if use_all_data:
                if use_premade_splits:
                    train_parquet_files = sorted(list(train_dir.glob('*.parquet')))
                else:
                    train_parquet_files = sorted(list(Path(data_dir).glob('*.parquet')))
            else:
                # Load only limited parquet files if not using all data
                if use_premade_splits:
                    train_parquet_files = sorted(list(train_dir.glob('*.parquet')))[:5]
                else:
                    train_parquet_files = sorted(list(Path(data_dir).glob('*.parquet')))[:5]

            if rank == 0:
                print(f" Found {len(train_parquet_files)} Parquet files in train split" if use_premade_splits else f" Found {len(train_parquet_files)} Parquet files")

            # Fast batch loading: load all parquet files at once
            if train_parquet_files:
                try:
                    # Filter out empty files
                    valid_files = [str(f) for f in train_parquet_files if f.stat().st_size > 0]
                    if valid_files:
                        if rank == 0:
                            print(f"  Loading {len(valid_files)} train files in batch (faster)...")
                        # Load all files at once - much faster than one-by-one
                        dataset = load_dataset('parquet', data_files=valid_files, split='train')
                        train_datasets_list.append(dataset)
                        if rank == 0:
                            print(f"  Loaded {len(valid_files)} train parquet files")
                except Exception as e:
                    if rank == 0:
                        print(f"  Batch loading failed ({e}), falling back to sequential loading...")
                    # Fallback to sequential loading
                    for i, parquet_file in enumerate(train_parquet_files, 1):
                        try:
                            file_size = parquet_file.stat().st_size
                            if file_size == 0:
                                continue
                            dataset = load_dataset('parquet', data_files=str(parquet_file))
                            split_name = list(dataset.keys())[0]
                            dataset = dataset[split_name]
                            train_datasets_list.append(dataset)
                            if rank == 0 and i % 10 == 0:
                                print(f"  Loaded {i}/{len(train_parquet_files)} parquet files...")
                        except Exception as e:
                            if rank == 0:
                                print(f"  Failed to load {parquet_file.name}: {e}")
                            continue

            # Load Parquet files for validation
            if use_premade_splits:
                if use_all_data:
                    val_parquet_files = sorted(list(val_dir.glob('*.parquet')))
                else:
                    val_parquet_files = sorted(list(val_dir.glob('*.parquet')))[:5]

                if rank == 0:
                    print(f" Found {len(val_parquet_files)} Parquet files in val split")

                # Fast batch loading for validation files
                if val_parquet_files:
                    try:
                        # Filter out empty files
                        valid_files = [str(f) for f in val_parquet_files if f.stat().st_size > 0]
                        if valid_files:
                            if rank == 0:
                                print(f"  Loading {len(valid_files)} val files in batch (faster)...")
                            # Load all files at once - much faster than one-by-one
                            dataset = load_dataset('parquet', data_files=valid_files, split='train')
                            val_datasets_list.append(dataset)
                            if rank == 0:
                                print(f"  Loaded {len(valid_files)} val parquet files")
                    except Exception as e:
                        if rank == 0:
                            print(f"  Batch loading failed ({e}), falling back to sequential loading...")
                        # Fallback to sequential loading
                        for i, parquet_file in enumerate(val_parquet_files, 1):
                            try:
                                file_size = parquet_file.stat().st_size
                                if file_size == 0:
                                    continue
                                dataset = load_dataset('parquet', data_files=str(parquet_file))
                                split_name = list(dataset.keys())[0]
                                dataset = dataset[split_name]
                                val_datasets_list.append(dataset)
                                if rank == 0 and i % 10 == 0:
                                    print(f"  Loaded {i}/{len(val_parquet_files)} val parquet files...")
                            except Exception as e:
                                if rank == 0:
                                    print(f"  Failed to load {parquet_file.name}: {e}")
                                continue

            if train_datasets_list:
                # Use single dataset if only one, otherwise concatenate
                if len(train_datasets_list) == 1:
                    train_data = train_datasets_list[0]
                else:
                    from datasets import concatenate_datasets
                    if rank == 0:
                        print(f"  Concatenating {len(train_datasets_list)} train datasets...")
                    train_data = concatenate_datasets(train_datasets_list)

                total_train_examples = len(train_data)
                if rank == 0:
                    print(f" Loaded train data with {total_train_examples:,} total examples")
                    # Handle dynamic batching case where batch_size may need to come from config
                    effective_bs = batch_size
                    if effective_bs is None and dynamic_batching_config:
                        effective_bs = dynamic_batching_config.get('min_batch_size', 64)
                    if effective_bs is None:
                        effective_bs = 64  # Default fallback
                    num_batches = total_train_examples // effective_bs
                    print(f" Expected batches per epoch: {num_batches:,}")
                train_dataset = ArrowDataset(train_data, seq_length, vocab_size)

                # Use validation split if available, otherwise use train data
                if val_datasets_list:
                    # Use single dataset if only one, otherwise concatenate
                    if len(val_datasets_list) == 1:
                        val_data = val_datasets_list[0]
                    else:
                        if rank == 0:
                            print(f"  Concatenating {len(val_datasets_list)} val datasets...")
                        from datasets import concatenate_datasets
                        val_data = concatenate_datasets(val_datasets_list)

                    total_val_examples = len(val_data)
                    if rank == 0:
                        print(f" Loaded val data with {total_val_examples:,} total examples")
                    val_dataset = ArrowDataset(val_data, seq_length, vocab_size)
                else:
                    if rank == 0:
                        print(f" No validation split found, using train data for validation")
                    val_dataset = ArrowDataset(train_data, seq_length, vocab_size)
            else:
                if rank == 0:
                    print(" No datasets loaded from data directory")
        except Exception as e:
            # Failed to load data, will use dummy dataset
            if rank == 0:
                print(f" Warning: Failed to load real data: {type(e).__name__}: {e}")
                import traceback
                print(traceback.format_exc())
            pass

    # Fall back to dummy dataset if not loaded yet
    if train_dataset is None or val_dataset is None:
        # If num_train_samples is None, automatically detect and use ALL available data
        if num_train_samples is None:
            # Count total examples in all Arrow files
            if data_dir and Path(data_dir).exists():
                arrow_files = list(Path(data_dir).glob('*.arrow'))
                total_examples = 0
                for arrow_file in arrow_files:
                    try:
                        # Approximate: assume ~1 example per 1KB (rough heuristic)
                        file_size = arrow_file.stat().st_size
                        if file_size > 0:
                            total_examples += max(1000, file_size // 1024)  # At least 1000 per file
                    except:
                        pass
                num_train_samples = max(200000, total_examples)  # Use detected or default 200K
            else:
                num_train_samples = 200000  # Default fallback

        if num_val_samples is None:
            num_val_samples = num_train_samples // 10  # Validation = 10% of training

        train_dataset = DummyDataset(num_train_samples, seq_length, vocab_size)
        val_dataset = DummyDataset(num_val_samples, seq_length, vocab_size)

    # Check if dynamic batching is enabled
    use_dynamic_batching = (
        dynamic_batching_config is not None
        and dynamic_batching_config.get('enabled', False)
    )

    if use_dynamic_batching:
        # Import dynamic batch iterator
        from src.Ava.data.dynamic_batch_iterator import DynamicBatchIterator
        from src.Ava.training.optimizations.dynamic_batching import create_dynamic_batch_scheduler

        min_batch_size = dynamic_batching_config.get('min_batch_size', 64)
        max_batch_size = dynamic_batching_config.get('max_batch_size', 256)

        print(f"\n{'='*60}")
        print(f" DYNAMIC BATCHING ENABLED (fallback loader)")
        print(f"{'='*60}")
        print(f"   Min batch size: {min_batch_size}")
        print(f"   Max batch size: {max_batch_size}")
        print(f"   Memory thresholds: low={dynamic_batching_config.get('low_memory_threshold', 0.5):.0%}, "
              f"target={dynamic_batching_config.get('target_memory_threshold', 0.7):.0%}, "
              f"high={dynamic_batching_config.get('high_memory_threshold', 0.85):.0%}")
        print(f"   Adjustment frequency: every {dynamic_batching_config.get('adjustment_frequency', 10)} steps")
        print(f"   Warmup steps: {dynamic_batching_config.get('warmup_steps', 100)}")
        print(f"{'='*60}\n")

        # Create base DataLoader with min_batch_size
        actual_batch_size = min_batch_size
    else:
        actual_batch_size = batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=(rank == 0),  # Only shuffle on rank 0
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop last on validation
    )

    # Wrap with DynamicBatchIterator if enabled
    if use_dynamic_batching:
        # Create scheduler config dict for the factory function
        scheduler_config = {
            'dynamic_batching': dynamic_batching_config,
            'training': {'batch_size': min_batch_size}
        }

        # Create schedulers for train and val
        train_scheduler = create_dynamic_batch_scheduler(scheduler_config)
        val_scheduler = create_dynamic_batch_scheduler(scheduler_config)

        # Wrap with DynamicBatchIterator
        train_loader = DynamicBatchIterator(
            base_dataloader=train_loader,
            scheduler=train_scheduler,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )
        val_loader = DynamicBatchIterator(
            base_dataloader=val_loader,
            scheduler=val_scheduler,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )

    return train_loader, val_loader


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TransformerModel100M(nn.Module):
    """100M parameter Transformer model."""

    def __init__(self, vocab_size: int = 50680, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                dropout=0.1,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints with truly async saving using CUDA streams and pinned memory."""

    def __init__(self, save_dir: Path, max_keep: int = 3, config: Optional[Dict] = None, async_save: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints = []
        self.config = config
        self.async_save = async_save

        # Thread pool for async disk I/O
        self.save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AsyncCheckpoint") if async_save else None
        self.pending_saves = []  # Track multiple ongoing saves

        # CUDA stream for non-blocking GPU->CPU transfers
        self._checkpoint_stream = None
        if async_save and torch.cuda.is_available():
            self._checkpoint_stream = torch.cuda.Stream()

        # Double buffer for pinned memory (allows overlap)
        self._pinned_buffers = [{}, {}]
        self._current_buffer = 0
        self._buffer_in_use = [False, False]

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, step: int, metrics: Dict[str, float]):
        """Save checkpoint in format compatible with generate.py."""
        if self.async_save and self.save_executor is not None and torch.cuda.is_available():
            return self._save_truly_async(model, optimizer, epoch, step, metrics)
        elif self.async_save and self.save_executor is not None:
            return self._save_async_cpu(model, optimizer, epoch, step, metrics)
        else:
            return self._save_sync(model, optimizer, epoch, step, metrics)

    def _save_sync(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, step: int, metrics: Dict[str, float]):
        """Synchronous checkpoint save (original behavior)."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict() if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'
        torch.save(checkpoint, path)
        self.checkpoints.append(path)

        latest_path = self.save_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)

        if 'val_loss' in metrics:
            best_path = self.save_dir / 'best_model.pt'
            if not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path):
                torch.save(checkpoint, best_path)

        if len(self.checkpoints) > self.max_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

        return path

    def _save_truly_async(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                          epoch: int, step: int, metrics: Dict[str, float]):
        """
        Truly asynchronous checkpoint save using CUDA streams and pinned memory.

        This implementation:
        1. Uses a separate CUDA stream for GPU->CPU transfers (non-blocking)
        2. Uses pinned memory for fast async transfers
        3. Only blocks briefly to record the stream, not for the full transfer
        4. Disk I/O happens in a background thread after transfer completes
        """
        # Clean up completed saves
        self._cleanup_completed_saves()

        # Get the model state dict reference (this is fast, just gets references)
        actual_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        # Select buffer (double buffering to avoid blocking)
        buffer_idx = self._current_buffer
        self._current_buffer = 1 - self._current_buffer

        # Wait for buffer to be free (from previous save 2 saves ago)
        if self._buffer_in_use[buffer_idx]:
            # Wait for the save using this buffer to complete
            for future, buf_idx, _ in self.pending_saves:
                if buf_idx == buffer_idx and not future.done():
                    future.result()  # Wait for it

        self._buffer_in_use[buffer_idx] = True
        pinned_buffer = self._pinned_buffers[buffer_idx]

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'

        # Record CUDA event before transfer starts (for timing)
        transfer_start_event = torch.cuda.Event(enable_timing=True)
        transfer_end_event = torch.cuda.Event(enable_timing=True)

        # Use checkpoint stream for non-blocking transfers
        with torch.cuda.stream(self._checkpoint_stream):
            transfer_start_event.record()

            # Copy model state to pinned memory asynchronously
            model_state = {}
            for name, param in actual_model.state_dict().items():
                if param.is_cuda:
                    # Allocate pinned memory buffer if needed (reuse across saves)
                    if name not in pinned_buffer or pinned_buffer[name].shape != param.shape:
                        pinned_buffer[name] = torch.empty(
                            param.shape, dtype=param.dtype,
                            pin_memory=True, device='cpu'
                        )
                    # Non-blocking copy to pinned memory
                    pinned_buffer[name].copy_(param, non_blocking=True)
                    model_state[name] = pinned_buffer[name]
                else:
                    model_state[name] = param.cpu().clone()

            # Copy optimizer state similarly (handle nested structure)
            opt_state = self._copy_optimizer_state_async(optimizer, pinned_buffer)

            transfer_end_event.record()

        # Create checkpoint structure (references to pinned memory)
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state': opt_state,
            'metrics': metrics.copy(),
            'config': self.config,
        }

        def _async_save_worker():
            """Background worker: waits for GPU transfer, then saves to disk."""
            try:
                # Wait for GPU->CPU transfer to complete (in background thread)
                transfer_end_event.synchronize()

                # Now clone from pinned memory to regular memory for saving
                # (pinned memory will be reused, so we need to copy)
                final_checkpoint = {
                    'epoch': checkpoint_data['epoch'],
                    'step': checkpoint_data['step'],
                    'model_state_dict': {k: v.clone() for k, v in checkpoint_data['model_state_dict'].items()},
                    'optimizer_state': self._clone_optimizer_state(checkpoint_data['optimizer_state']),
                    'metrics': checkpoint_data['metrics'],
                    'config': checkpoint_data['config'],
                }

                # Mark buffer as free now that we've cloned
                self._buffer_in_use[buffer_idx] = False

                # Save to disk
                torch.save(final_checkpoint, path)

                # Save latest
                latest_path = self.save_dir / 'latest_model.pt'
                torch.save(final_checkpoint, latest_path)

                # Update best if needed
                if 'val_loss' in metrics:
                    best_path = self.save_dir / 'best_model.pt'
                    if not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path):
                        torch.save(final_checkpoint, best_path)

            except Exception as e:
                self._buffer_in_use[buffer_idx] = False
                print(f"[AsyncCheckpoint] Save failed: {e}")

        # Submit to background thread and return immediately
        future = self.save_executor.submit(_async_save_worker)
        self.pending_saves.append((future, buffer_idx, path))
        self.checkpoints.append(path)

        # Cleanup old checkpoints
        if len(self.checkpoints) > self.max_keep:
            old_path = self.checkpoints.pop(0)
            # Schedule deletion in background
            self.save_executor.submit(lambda p=old_path: p.unlink() if p.exists() else None)

        return path

    def _copy_optimizer_state_async(self, optimizer, pinned_buffer):
        """Copy optimizer state to pinned memory asynchronously."""
        opt_state_dict = optimizer.state_dict()
        result = {'state': {}, 'param_groups': opt_state_dict.get('param_groups', [])}

        for param_id, state in opt_state_dict.get('state', {}).items():
            result['state'][param_id] = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    buf_key = f"opt_{param_id}_{key}"
                    if buf_key not in pinned_buffer or pinned_buffer[buf_key].shape != value.shape:
                        pinned_buffer[buf_key] = torch.empty(
                            value.shape, dtype=value.dtype,
                            pin_memory=True, device='cpu'
                        )
                    pinned_buffer[buf_key].copy_(value, non_blocking=True)
                    result['state'][param_id][key] = pinned_buffer[buf_key]
                elif isinstance(value, torch.Tensor):
                    result['state'][param_id][key] = value.cpu().clone()
                else:
                    result['state'][param_id][key] = value

        return result

    def _clone_optimizer_state(self, opt_state):
        """Clone optimizer state from pinned memory."""
        result = {'state': {}, 'param_groups': opt_state.get('param_groups', [])}

        for param_id, state in opt_state.get('state', {}).items():
            result['state'][param_id] = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    result['state'][param_id][key] = value.clone()
                else:
                    result['state'][param_id][key] = value

        return result

    def _save_async_cpu(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, step: int, metrics: Dict[str, float]):
        """Async save for CPU-only training."""
        self._cleanup_completed_saves()

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': {k: v.clone() for k, v in (model.state_dict() if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module.state_dict()).items()},
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics.copy(),
            'config': self.config,
        }

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'

        def _save_worker():
            try:
                torch.save(checkpoint, path)
                latest_path = self.save_dir / 'latest_model.pt'
                torch.save(checkpoint, latest_path)
                if 'val_loss' in metrics:
                    best_path = self.save_dir / 'best_model.pt'
                    if not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path):
                        torch.save(checkpoint, best_path)
            except Exception as e:
                print(f"[AsyncCheckpoint] Save failed: {e}")

        future = self.save_executor.submit(_save_worker)
        self.pending_saves.append((future, -1, path))
        self.checkpoints.append(path)

        if len(self.checkpoints) > self.max_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

        return path

    def _cleanup_completed_saves(self):
        """Remove completed saves from pending list."""
        self.pending_saves = [(f, b, p) for f, b, p in self.pending_saves if not f.done()]

    def wait_for_pending_save(self):
        """Wait for all pending async saves to complete."""
        for future, buffer_idx, _ in self.pending_saves:
            if not future.done():
                future.result()
        self.pending_saves = []

    def shutdown(self):
        """Shutdown async save executor."""
        if self.save_executor is not None:
            self.wait_for_pending_save()
            self.save_executor.shutdown(wait=True)
        if self._checkpoint_stream is not None:
            self._checkpoint_stream.synchronize()

    def _get_best_loss(self, best_path: Path) -> float:
        """Get best loss from existing checkpoint."""
        try:
            checkpoint = torch.load(best_path, weights_only=False)
            return checkpoint.get('metrics', {}).get('val_loss', float('inf'))
        except:
            return float('inf')

    def load(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             checkpoint_path: Path) -> Tuple[int, int]:
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Handle both old and new checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        return checkpoint.get('epoch', 0), checkpoint.get('step', 0)


# ============================================================================
# METRICS TRACKING
# ============================================================================

class MetricsTracker:
    """Track training metrics."""

    def __init__(self, log_dir: Path, use_wandb: bool = False, wandb_config: Optional[Dict] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.metrics = {}
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if self.use_wandb:
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get('project', 'transformer-training'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=wandb_config.get('config', {}),
                tags=wandb_config.get('tags', []),
            )

    def update(self, step: int, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
            self.writer.add_scalar(f'train/{key}', value, step)

            if self.use_wandb:
                wandb.log({f'train/{key}': value}, step=step)

    def log_gradients(self, step: int, grad_stats: Dict[str, float]):
        """Log gradient statistics."""
        self.writer.add_scalar('gradients/avg_gradient',
                              grad_stats['total_norm'] / max(grad_stats['num_grads'], 1), step)
        self.writer.add_scalar('gradients/max_gradient', grad_stats['max_grad'], step)
        self.writer.add_scalar('gradients/min_gradient', grad_stats['min_grad'], step)
        self.writer.add_scalar('gradients/num_zero_grads', grad_stats['num_zero_grads'], step)

        if self.use_wandb:
            wandb.log({
                'gradients/avg_gradient': grad_stats['total_norm'] / max(grad_stats['num_grads'], 1),
                'gradients/max_gradient': grad_stats['max_grad'],
                'gradients/min_gradient': grad_stats['min_grad'],
                'gradients/num_zero_grads': grad_stats['num_zero_grads'],
            }, step=step)

    def log_moe_metrics(self, step: int, moe_metrics: Dict[str, float]):
        """Log MOE-specific metrics."""
        for key, value in moe_metrics.items():
            self.writer.add_scalar(f'moe/{key}', value, step)
            if self.use_wandb:
                wandb.log({f'moe/{key}': value}, step=step)

    def log_validation(self, step: int, **kwargs):
        """Log validation metrics."""
        for key, value in kwargs.items():
            self.writer.add_scalar(f'validation/{key}', value, step)
            if self.use_wandb:
                wandb.log({f'validation/{key}': value}, step=step)

    def save_summary(self, summary_path: Path):
        """Save metrics summary."""
        summary = {key: values for key, values in self.metrics.items()}
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def finish(self):
        """Finish logging and cleanup."""
        if self.use_wandb:
            wandb.finish()


# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

def setup_distributed():
    """Setup distributed training if available."""
    if not DISTRIBUTED_AVAILABLE:
        return 0, 1

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        init_process_group(backend='nccl')
        return rank, world_size

    return 0, 1


def cleanup_distributed(rank: int, world_size: int):
    """Cleanup distributed training."""
    if DISTRIBUTED_AVAILABLE and world_size > 1:
        destroy_process_group()


# ============================================================================
# GRADIENT CHECKING AND MONITORING
# ============================================================================

def check_gradients(model: nn.Module, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """Check gradient statistics and return metrics."""
    grad_stats = {
        'total_norm': 0.0,
        'max_grad': 0.0,
        'min_grad': float('inf'),
        'num_zero_grads': 0,
        'num_params': 0,
        'num_grads': 0,
    }

    for param in model.parameters():
        if param.grad is not None:
            grad_stats['num_params'] += 1
            grad_val = param.grad.data.abs()
            grad_stats['total_norm'] += grad_val.sum().item()
            grad_stats['max_grad'] = max(grad_stats['max_grad'], grad_val.max().item())
            grad_stats['min_grad'] = min(grad_stats['min_grad'], grad_val.min().item())
            grad_stats['num_grads'] += param.grad.numel()

            if (param.grad.abs() < 1e-10).all():
                grad_stats['num_zero_grads'] += 1

    if grad_stats['min_grad'] == float('inf'):
        grad_stats['min_grad'] = 0.0

    if logger is not None and grad_stats['num_params'] > 0:
        avg_grad = grad_stats['total_norm'] / grad_stats['num_grads']
        logger.info(
            f"Gradient stats: avg={avg_grad:.2e}, max={grad_stats['max_grad']:.2e}, "
            f"min={grad_stats['min_grad']:.2e}, zero_grads={grad_stats['num_zero_grads']}, "
            f"params_with_grads={grad_stats['num_params']}"
        )

    return grad_stats


# ============================================================================
# GENERATION CHECKS
# ============================================================================

def generate_sample(
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    max_length: int = 100,
    num_samples: int = 1,
    logger: Optional[logging.Logger] = None,
    tokenizer: Optional[Any] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    prompt: Optional[str] = None,
    prompt_ids: Optional[torch.Tensor] = None,
) -> str:
    """Generate sample text from model for quality checks.

    Args:
        model: The transformer model
        device: Device to run generation on
        vocab_size: Size of vocabulary
        max_length: Maximum length of generated sequence
        num_samples: Number of samples to generate
        logger: Logger for output
        tokenizer: Optional tokenizer to decode token IDs into text
        temperature: Sampling temperature (higher = more diverse, default 1.0)
        top_p: Nucleus sampling threshold (0-1, default 1.0)
        top_k: Keep top-k tokens for sampling (default 50)
        repetition_penalty: Penalize repeated tokens (>1.0, default 1.0)
        skip_special_tokens: Whether to skip special tokens in decoded output
        prompt: Optional text prompt to condition generation
        prompt_ids: Optional tensor of prompt token IDs

    Returns:
        Generated text as string representation
    """
    model.eval()

    def top_p_sampling(logits, top_p=0.9, temperature=1.0, top_k=50, repetition_penalty=1.0):
        """Apply temperature, top-k, top-p (nucleus) sampling, and repetition penalty."""
        # Apply temperature
        logits = logits / max(temperature, 1e-5)

        # Apply repetition penalty (penalize tokens that have been generated)
        if repetition_penalty > 1.0 and len(generated_ids) > 0:
            # Get recently generated tokens (last 50 tokens)
            recent_tokens = generated_ids[0, -50:] if generated_ids.shape[1] > 50 else generated_ids[0]
            logits[:, recent_tokens] /= repetition_penalty

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Apply top-k filtering
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            probs_filtered = torch.zeros_like(probs)
            probs_filtered.scatter_(-1, top_k_indices, top_k_probs)
            probs = probs_filtered

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Compute cumulative probabilities
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cutoff index for top-p
        sorted_indices_to_remove = cum_probs > top_p
        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False

        # Remove tokens below threshold
        sorted_probs[sorted_indices_to_remove] = 0.0

        # Renormalize probabilities
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample from the distribution
        next_token = torch.multinomial(sorted_probs, num_samples=1)

        # Map back to original indices
        next_token = sorted_indices.gather(-1, next_token)

        return next_token

    with torch.no_grad():
        # Start with prompt tokens or random tokens
        batch_size = num_samples

        if prompt_ids is not None:
            # Use provided prompt token IDs
            start_token = prompt_ids.to(device)
        elif prompt is not None and tokenizer is not None:
            # Encode prompt text to token IDs
            try:
                encoded = tokenizer.encode(prompt, return_tensors='pt')
                start_token = encoded.to(device)
            except Exception as e:
                if logger is not None:
                    logger.warning(f"Failed to encode prompt: {e}. Using random tokens instead.")
                start_token = torch.randint(0, vocab_size, (batch_size, 1)).to(device)
        else:
            # Use random tokens
            start_token = torch.randint(0, vocab_size, (batch_size, 1)).to(device)

        generated_ids = start_token.clone()

        # Get model's max position embeddings to avoid exceeding it
        max_pos_embeddings = getattr(model, 'max_position_embeddings', 256)
        effective_max_length = min(max_length, max_pos_embeddings)

        # Generate tokens one by one
        for _ in range(effective_max_length - 1):
            # Check if we've reached the position embedding limit
            if generated_ids.shape[1] >= max_pos_embeddings:
                if logger is not None:
                    logger.info(f"Stopping generation at {generated_ids.shape[1]} tokens (max_position_embeddings={max_pos_embeddings})")
                break

            # Forward pass
            outputs = model(generated_ids)
            logits = outputs['logits']

            # Get next token from last position
            next_token_logits = logits[:, -1, :]

            # Use temperature, top-k, top-p sampling with repetition penalty
            next_tokens = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature,
                                        top_k=top_k, repetition_penalty=repetition_penalty)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

            # Stop if all sequences ended (optional early stopping)
            if next_tokens.max().item() == 0:  # End token is 0
                break

        # Convert to text if tokenizer is available
        all_outputs = []

        for sample_idx in range(min(num_samples, generated_ids.shape[0])):
            generated_ids_cpu = generated_ids[sample_idx].cpu().tolist()

            if tokenizer is not None:
                try:
                    decoded_text = tokenizer.decode(generated_ids_cpu, skip_special_tokens=skip_special_tokens)

                    # Debug: Log token count and text length
                    if logger is not None:
                        logger.info(f"  Sample {sample_idx + 1}/{num_samples}: {len(generated_ids_cpu)} tokens → {len(decoded_text)} chars")

                    # Show prompt if available
                    if prompt is not None:
                        gen_str = f"Prompt: {prompt}\nGenerated text:\n{decoded_text}"
                    else:
                        gen_str = f"Generated text:\n{decoded_text}"
                except Exception as e:
                    # Fallback to token IDs if decoding fails
                    gen_str = f"Generated sequence (token IDs): {generated_ids_cpu[:50]}"
                    if logger is not None:
                        logger.warning(f"Tokenizer decode failed: {e}. Using token IDs instead.")
            else:
                # No tokenizer provided, show token IDs
                if sample_idx == 0 and logger is not None:
                    logger.warning("No tokenizer available - generation using token IDs only")
                if prompt is not None:
                    gen_str = f"Prompt: {prompt}\nGenerated sequence (token IDs): {generated_ids_cpu[:50]}"
                else:
                    gen_str = f"Generated sequence (token IDs): {generated_ids_cpu[:50]}"

            if logger is not None:
                logger.info(gen_str)
            else:
                # Even without logger, print to stdout for visibility
                print(gen_str)

            all_outputs.append(gen_str)

        return "\n---\n".join(all_outputs)


def async_generate_on_cpu(
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    executor: ThreadPoolExecutor,
    logger: Optional[logging.Logger] = None,
    global_step: int = 0,
    log_dir: Optional[Path] = None,
    **generation_kwargs
):
    """
    Launch generation asynchronously on CPU to avoid blocking training.

    Args:
        model: The transformer model
        device: Original device (will move model copy to CPU)
        vocab_size: Size of vocabulary
        executor: ThreadPoolExecutor for async execution
        logger: Logger for output (only for errors)
        global_step: Current training step (for output file naming)
        log_dir: Directory to save generation outputs
        **generation_kwargs: Additional kwargs for generate_sample

    Returns:
        Future object that can be checked for completion
    """
    def _run_generation_on_cpu():
        """Internal function to run generation on CPU - fully isolated from training"""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        # Create a separate logger for async generation that writes to file
        gen_logger = None
        output_file = None

        try:
            # Create output directory for generation logs
            if log_dir:
                gen_log_dir = log_dir / "async_generation"
                gen_log_dir.mkdir(exist_ok=True, parents=True)
                output_file = gen_log_dir / f"generation_step_{global_step}.txt"

            # Create a file-only logger (no console output)
            if output_file:
                gen_logger = logging.getLogger(f'async_gen_{global_step}')
                gen_logger.setLevel(logging.INFO)
                gen_logger.handlers.clear()  # Remove any existing handlers
                file_handler = logging.FileHandler(output_file)
                file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
                gen_logger.addHandler(file_handler)
                gen_logger.propagate = False  # Don't propagate to parent loggers

            # Capture all stdout/stderr to prevent any console output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Create a CPU copy of the model for generation
                cpu_device = torch.device('cpu')

                with torch.no_grad():
                    # Get model state dict and move to CPU
                    model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                    # Get model config if available
                    if hasattr(model, 'config'):
                        config = model.config
                    elif hasattr(model, 'module') and hasattr(model.module, 'config'):
                        config = model.module.config
                    else:
                        config = None

                    # Create a temporary model on CPU
                    if config is not None:
                        # Get the base model class (unwrap DDP if needed)
                        model_class = type(model.module) if hasattr(model, 'module') else type(model)
                        cpu_model = model_class(config)
                        cpu_model.load_state_dict(model_state)
                    else:
                        # Fallback: try to clone the entire model
                        import copy
                        cpu_model = copy.deepcopy(model)
                        cpu_model.cpu()

                    cpu_model.eval()

                    # Run generation on CPU with file-only logger
                    if gen_logger:
                        gen_logger.info(f"=== Async Generation at Step {global_step} ===")

                    generate_sample(
                        cpu_model,
                        cpu_device,
                        vocab_size,
                        logger=gen_logger,  # Use file-only logger
                        **generation_kwargs
                    )

                    if gen_logger:
                        gen_logger.info("=== Generation Complete ===")

                    # Clean up
                    del cpu_model
                    del model_state
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Write any captured output to file
            if output_file and gen_logger:
                if stdout_buffer.getvalue():
                    gen_logger.info(f"Captured stdout:\n{stdout_buffer.getvalue()}")
                if stderr_buffer.getvalue():
                    gen_logger.info(f"Captured stderr:\n{stderr_buffer.getvalue()}")

        except Exception as e:
            # Only log errors to file, not console
            error_msg = f"Async generation failed: {e}"
            if gen_logger:
                import traceback
                gen_logger.error(error_msg)
                gen_logger.error(f"Traceback: {traceback.format_exc()}")
            # Optionally log critical errors to main logger without disrupting training
            # (commented out to keep it fully silent)
            # if logger is not None:
            #     logger.debug(f"Async generation error (see {output_file})")

        finally:
            # Cleanup file logger
            if gen_logger:
                for handler in gen_logger.handlers[:]:
                    handler.close()
                    gen_logger.removeHandler(handler)

    # Submit to executor
    return executor.submit(_run_generation_on_cpu)


# ============================================================================
# QUANTIZATION FUNCTIONS
# ============================================================================

def quantize_model_int8(
    model: nn.Module,
    threshold: float = 6.0,
    skip_modules: Optional[list] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Quantize model Linear layers to INT8 using bitsandbytes.

    This replaces nn.Linear layers with bnb.nn.Linear8bitLt for memory-efficient
    INT8 training. Note: INT8 training requires careful handling of gradients.

    Args:
        model: The model to quantize
        threshold: Outlier threshold for mixed-precision decomposition (default: 6.0)
        skip_modules: List of module name patterns to skip (e.g., ['embed', 'lm_head'])
        logger: Optional logger for status messages

    Returns:
        Quantized model with INT8 linear layers
    """
    if not BITSANDBYTES_AVAILABLE:
        if logger:
            logger.warning("bitsandbytes not available, skipping INT8 quantization")
        return model

    if skip_modules is None:
        skip_modules = ['embed', 'lm_head']

    quantized_count = 0

    # Replace Linear layers with INT8 versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            # Skip specified modules (keep in higher precision)
            if any(skip in name.lower() for skip in skip_modules):
                continue

            # Create INT8 linear layer with fp16 weights for training
            # has_fp16_weights=True keeps a fp16 copy for gradient computation
            int8_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=True,  # Keep fp16 weights for training
                threshold=threshold,
            )

            # Copy weights - use fp16 for training compatibility
            int8_layer.weight = bnb.nn.Int8Params(
                module.weight.data.to(torch.float16),
                requires_grad=False,  # INT8 params don't support gradients directly
                has_fp16_weights=True
            )
            if module.bias is not None:
                int8_layer.bias = nn.Parameter(module.bias.data)

            # Replace the layer
            setattr(parent, attr_name, int8_layer)
            quantized_count += 1

    if logger:
        logger.info(f"   Quantized {quantized_count} Linear layers to INT8 (threshold={threshold})")

    return model


def quantize_model_nf4(
    model: nn.Module,
    skip_modules: Optional[list] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Quantize model to NF4 (Normalized Float 4-bit) using bitsandbytes.

    NF4 provides better accuracy than standard INT4 by using a normalized
    distribution that better matches neural network weight distributions.

    Args:
        model: The model to quantize
        skip_modules: List of module name patterns to skip
        logger: Optional logger for status messages

    Returns:
        Quantized model with NF4 linear layers
    """
    if not BITSANDBYTES_AVAILABLE:
        if logger:
            logger.warning("bitsandbytes not available, skipping NF4 quantization")
        return model

    if skip_modules is None:
        skip_modules = ['embed', 'lm_head']

    quantized_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            if any(skip in name.lower() for skip in skip_modules):
                continue

            # Create NF4 linear layer
            nf4_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type='nf4',  # Normalized Float 4
            )

            # Copy weights - requires_grad=False for quantized params
            nf4_layer.weight = bnb.nn.Params4bit(
                module.weight.data.to(torch.float16),
                requires_grad=False,  # Quantized params don't support gradients directly
                quant_type='nf4',
            )
            if module.bias is not None:
                nf4_layer.bias = nn.Parameter(module.bias.data)

            setattr(parent, attr_name, nf4_layer)
            quantized_count += 1

    if logger:
        logger.info(f"   Quantized {quantized_count} Linear layers to NF4 (4-bit)")

    return model


def quantize_model_nvfp4(
    model: nn.Module,
    block_size: int = 16,
    skip_modules: Optional[list] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Quantize model to NVFP4 (NVIDIA FP4) using torchao.

    NVFP4 is NVIDIA's 4-bit floating point format optimized for inference
    on Hopper (H100) and later GPUs. Provides ~8x memory reduction.

    Args:
        model: The model to quantize
        block_size: Block size for quantization (default: 16)
        skip_modules: List of module name patterns to skip
        logger: Optional logger for status messages

    Returns:
        Quantized model
    """
    if not TORCHAO_AVAILABLE:
        if logger:
            logger.warning("torchao not available, skipping NVFP4 quantization")
        return model

    if skip_modules is None:
        skip_modules = ['embed', 'lm_head']

    try:
        # Use torchao's int4_weight_only as approximation for FP4
        # Note: True NVFP4 requires Hopper+ GPUs and specific torchao version
        def skip_filter(mod, fqn):
            return any(skip in fqn.lower() for skip in skip_modules)

        quantize_(model, int4_weight_only(), filter_fn=lambda m, fqn: not skip_filter(m, fqn))

        if logger:
            logger.info(f"   Applied NVFP4/INT4 quantization (block_size={block_size})")

    except Exception as e:
        if logger:
            logger.warning(f"   NVFP4 quantization failed: {e}, falling back to unquantized")

    return model


def apply_quantization(
    model: nn.Module,
    quant_config: dict,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Apply quantization based on configuration.

    Args:
        model: The model to quantize
        quant_config: Quantization configuration dict with keys:
            - enabled: bool
            - type: 'none', 'int8', 'nf4', 'nvfp4'
            - int8_threshold: float (for INT8)
            - nvfp4_block_size: int (for NVFP4)
            - skip_modules: list of module patterns to skip
        logger: Optional logger

    Returns:
        Quantized model
    """
    if not quant_config.get('enabled', False):
        if logger:
            logger.info("   Quantization disabled")
        return model

    quant_type = quant_config.get('type', 'none').lower()
    skip_modules = quant_config.get('skip_modules', ['embed', 'lm_head'])

    if quant_type == 'int8':
        threshold = quant_config.get('int8_threshold', 6.0)
        return quantize_model_int8(model, threshold=threshold, skip_modules=skip_modules, logger=logger)

    elif quant_type == 'nf4':
        return quantize_model_nf4(model, skip_modules=skip_modules, logger=logger)

    elif quant_type == 'nvfp4' or quant_type == 'fp4':
        block_size = quant_config.get('nvfp4_block_size', 16)
        return quantize_model_nvfp4(model, block_size=block_size, skip_modules=skip_modules, logger=logger)

    elif quant_type == 'none':
        if logger:
            logger.info("   No quantization applied")
        return model

    else:
        if logger:
            logger.warning(f"   Unknown quantization type '{quant_type}', skipping")
        return model


# ============================================================================
# ASYNC BATCH PREFETCHER - Eliminates GPU starvation
# ============================================================================

class AsyncBatchPrefetcher:
    """
    Prefetches batches in a background thread for continuous GPU feeding.

    This class solves GPU starvation by completely decoupling data loading from
    GPU computation. The DataLoader runs in a separate thread, continuously
    loading batches while the main thread focuses solely on GPU operations.

    Key features:
    - Background thread for DataLoader iteration (never blocks main thread)
    - Dedicated CUDA stream for CPU→GPU transfers (overlaps with compute)
    - Queue-based buffering with configurable prefetch depth
    - CUDA event synchronization for correctness
    """

    def __init__(self, dataloader, device, prefetch_count: int = 3):
        """
        Initialize the async batch prefetcher.

        Args:
            dataloader: PyTorch DataLoader to iterate over
            device: Target device for batch transfer
            prefetch_count: Number of batches to keep ready (default: 3)
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_count = prefetch_count
        self.queue = Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.transfer_stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self._total_batches = 0

        # Start prefetch thread
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def _prefetch_loop(self):
        """Background thread that continuously loads batches and transfers to GPU."""
        try:
            for batch_idx, batch in enumerate(self.dataloader):
                if self.stop_event.is_set():
                    break

                self._total_batches = batch_idx + 1

                # Transfer to GPU using dedicated stream (overlaps with main thread's compute)
                if self.transfer_stream is not None:
                    with torch.cuda.stream(self.transfer_stream):
                        gpu_batch = {
                            'input_ids': batch['input_ids'].to(self.device, non_blocking=True),
                            'labels': batch['labels'].to(self.device, non_blocking=True),
                            'attention_mask': batch['attention_mask'].to(self.device, non_blocking=True),
                        }
                    # Record event so main thread knows when transfer is complete
                    event = torch.cuda.Event()
                    event.record(self.transfer_stream)
                else:
                    # CPU path - direct transfer
                    gpu_batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device),
                    }
                    event = None

                # Put batch in queue (blocks if queue is full - backpressure)
                self.queue.put((batch_idx, gpu_batch, event))

        except Exception as e:
            # Signal error to main thread
            self.queue.put(('ERROR', e, None))
        finally:
            # Signal end of iteration
            self.queue.put(None)

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self):
        """Get next batch from queue, waiting for GPU transfer if needed."""
        item = self.queue.get()

        if item is None:
            raise StopIteration

        if item[0] == 'ERROR':
            raise item[1]

        batch_idx, gpu_batch, event = item

        # Wait for GPU transfer to complete before using batch
        if event is not None:
            event.synchronize()

        return batch_idx, gpu_batch

    def __len__(self):
        """Return estimated length from underlying dataloader."""
        if hasattr(self.dataloader, '__len__'):
            return len(self.dataloader)
        if hasattr(self.dataloader, 'get_dynamic_total'):
            return self.dataloader.get_dynamic_total()
        return 0

    def get_dynamic_total(self):
        """Support dynamic batch size tracking."""
        if hasattr(self.dataloader, 'get_dynamic_total'):
            return self.dataloader.get_dynamic_total()
        return len(self.dataloader) if hasattr(self.dataloader, '__len__') else self._total_batches

    def stop(self):
        """Stop the prefetch thread and clean up."""
        self.stop_event.set()
        # Drain queue to unblock thread if it's waiting on queue.put()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                pass
        # Wait for thread to finish (with timeout)
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_interval: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    logger: Optional[logging.Logger] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
    vocab_size: int = 50680,
    generate_every_n_steps: int = 500,
    num_generations_per_step: int = 1,
    generation_max_length: int = 128,
    generation_temperature: float = 0.7,
    generation_top_p: float = 0.9,
    generation_top_k: int = 50,
    generation_repetition_penalty: float = 1.0,
    tokenizer: Optional[Any] = None,
    generation_skip_special_tokens: bool = True,
    generation_prompt: Optional[str] = None,
    checkpoint_manager: Optional['CheckpointManager'] = None,
    save_steps: int = 0,
    generation_executor: Optional[ThreadPoolExecutor] = None,
    log_dir: Optional[Path] = None,
) -> float:
    """Train for one epoch using async batch prefetching for constant GPU utilization."""

    model.train()
    total_loss = 0.0
    num_batches = 0
    # Use dynamic total if available for global_step calculation
    loader_len = train_loader.get_dynamic_total() if hasattr(train_loader, 'get_dynamic_total') else len(train_loader)
    global_step = epoch * loader_len

    # GradScaler is only needed for fp16, not bf16 (bf16 has same dynamic range as fp32)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    # Create async batch prefetcher - this runs DataLoader in background thread
    # with 3 batches prefetched ahead for continuous GPU feeding
    prefetcher = AsyncBatchPrefetcher(train_loader, device, prefetch_count=3)

    # Get initial total for progress bar
    initial_total = prefetcher.get_dynamic_total()
    pbar = tqdm(total=initial_total, desc=f"Epoch {epoch + 1}", disable=logger is None)

    try:
        for batch_idx, gpu_batch in prefetcher:
            # Batch is already on GPU (transferred in background thread)
            input_ids = gpu_batch['input_ids']
            labels = gpu_batch['labels']
            attention_mask = gpu_batch['attention_mask']

            try:
                # Forward pass with mixed precision
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        outputs = model(input_ids, attention_mask, labels)
                        loss = outputs['loss'] / gradient_accumulation_steps

                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss'] / gradient_accumulation_steps
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                # Track metrics
                loss_value = loss.item() * gradient_accumulation_steps
                total_loss += loss_value
                num_batches += 1
                global_step += 1

                if metrics_tracker is not None:
                    metrics_tracker.update(
                        global_step,
                        loss=loss_value,
                        lr=optimizer.param_groups[0]['lr']
                    )

                # Log progress and check gradients
                if batch_idx % log_interval == 0 and logger is not None:
                    avg_loss = total_loss / num_batches
                    # Get actual batch size from the batch
                    current_bs = input_ids.shape[0]
                    # Get dynamic total if available
                    total_batches = prefetcher.get_dynamic_total()
                    logger.info(
                        f"Epoch {epoch + 1} | Batch {batch_idx}/{total_batches} | "
                        f"BS: {current_bs} | Loss: {loss_value:.4f} | Avg Loss: {avg_loss:.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    # Check gradients after optimizer step
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        grad_stats = check_gradients(model, logger)
                        if metrics_tracker is not None:
                            metrics_tracker.log_gradients(global_step, grad_stats)

                # Generation testing during training (async on CPU)
                if generate_every_n_steps and generate_every_n_steps > 0 and global_step % generate_every_n_steps == 0 and logger is not None:
                    if generation_executor is not None:
                        # Launch fully async generation - no console output, writes to file
                        async_generate_on_cpu(
                            model, device, vocab_size,
                            executor=generation_executor,
                            logger=None,  # No logger to prevent console output
                            global_step=global_step,
                            log_dir=log_dir,
                            max_length=generation_max_length,
                            num_samples=num_generations_per_step,
                            tokenizer=tokenizer,
                            temperature=generation_temperature,
                            top_p=generation_top_p,
                            skip_special_tokens=generation_skip_special_tokens,
                            prompt=generation_prompt,
                            top_k=generation_top_k,
                            repetition_penalty=generation_repetition_penalty
                        )
                        # No logging to console - generation runs silently in background
                    else:
                        # Fallback to synchronous generation
                        logger.info(f"\n Testing generation at step {global_step}...")
                        generate_sample(
                            model, device, vocab_size,
                            max_length=generation_max_length,
                            num_samples=num_generations_per_step,
                            logger=logger,
                            tokenizer=tokenizer,
                            temperature=generation_temperature,
                            top_p=generation_top_p,
                            skip_special_tokens=generation_skip_special_tokens,
                            prompt=generation_prompt,
                            top_k=generation_top_k,
                            repetition_penalty=generation_repetition_penalty
                        )
                        logger.info(f" Generation test complete\n")

                # Step-based checkpoint saving (truly async - doesn't block training)
                if save_steps > 0 and global_step % save_steps == 0 and checkpoint_manager is not None:
                    if logger is not None:
                        logger.info(f" Queuing async checkpoint at step {global_step}...")
                    checkpoint_manager.save(model, optimizer, epoch, global_step, {'step_loss': loss_value})
                    # Note: save() returns immediately, GPU->CPU transfer and disk I/O happen in background

                # VRAM OPTIMIZATION: Periodic cache clearing every 500 steps
                # Prevents memory fragmentation and reduces VRAM usage by ~1-2GB
                if global_step % 500 == 0:
                    # Clear model caches (causal mask, RoPE)
                    if hasattr(model, 'clear_caches'):
                        model.clear_caches()
                    elif hasattr(model, 'module') and hasattr(model.module, 'clear_caches'):
                        # Handle DDP/FSDP wrapped models
                        model.module.clear_caches()

                    # Clear dataloader file cache if available
                    if hasattr(prefetcher, 'dataloader') and hasattr(prefetcher.dataloader, 'dataset'):
                        dataset = prefetcher.dataloader.dataset
                        if hasattr(dataset, 'clear_file_cache'):
                            dataset.clear_file_cache()

                    # Clear CUDA cache to reduce fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if logger is not None and global_step % 1000 == 0:
                        # Log memory stats every 1000 steps
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        logger.info(f"🧹 Cache cleared at step {global_step} | "
                                    f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

                # Update progress bar total if using dynamic batching
                new_total = prefetcher.get_dynamic_total()
                if pbar.total != new_total:
                    pbar.total = new_total
                    pbar.refresh()

                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                pbar.update(1)

            except Exception as e:
                if logger is not None:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                pbar.update(1)
                continue

    finally:
        # Always stop prefetcher to clean up background thread
        prefetcher.stop()
        pbar.close()

    avg_epoch_loss = total_loss / max(num_batches, 1)
    return avg_epoch_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Validate model."""

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", disable=logger is None)

        for batch in pbar:
            try:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        outputs = model(input_ids, attention_mask, labels)
                        loss = outputs['loss']
                else:
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss']

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                if logger is not None:
                    logger.error(f"Validation error: {e}")
                continue

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    """Main training function."""
    from transformers import AutoTokenizer

    # Setup distributed training
    rank, world_size = setup_distributed()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # Setup logging - create run directory if using framework structure
    log_dir = Path(args.log_dir)
    save_dir = Path(args.save_dir)

    # If save_dir is default and no explicit paths, use framework structure
    if args.save_dir == './checkpoints' and args.log_dir == './logs':
        # Create a run directory in outputs/runs
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        runs_dir = Path('/project/code/outputs/runs') / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)

        save_dir = runs_dir / 'checkpoints'
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir = runs_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        configs_dir = runs_dir / 'configs'
        configs_dir.mkdir(parents=True, exist_ok=True)

        # Update args to use new paths
        args.save_dir = str(save_dir)
        args.log_dir = str(log_dir)

    logger = setup_logging(log_dir, rank)

    if rank == 0:
        logger.info("="*80)
        logger.info(" STARTING 100M PARAMETER TRAINING")
        logger.info("="*80)
        logger.info(f"Device: {device}")
        logger.info(f"Distributed: Rank {rank}/{world_size}")

    # Load config if provided
    if args.config:
        config = load_yaml_with_path_resolution(args.config, project_root)
        if rank == 0:
            logger.info(f"Loaded config from {args.config}")
    else:
        config = {}

    # Model configuration with defaults
    model_config = config.get('model', {})
    vocab_size = model_config.get('vocab_size', 50680)
    hidden_size = model_config.get('hidden_size', 768)
    num_layers = model_config.get('num_layers', 12)
    num_heads = model_config.get('num_attention_heads', 12)
    max_position_embeddings = model_config.get('max_position_embeddings', 256)

    # Training configuration with defaults
    training_config = config.get('training', {})

    # Extract nested config sections (supports both flat and nested structure)
    batching_config = training_config.get('batching', {})
    optimizer_cfg = training_config.get('optimizer', {})
    schedule_config = training_config.get('schedule', {})
    logging_config = training_config.get('logging', {})
    validation_config = training_config.get('validation', {})
    generation_config = training_config.get('generation', {})

    # Handle batch_size with dynamic batching support
    # If batch_size is null/None in config, use dynamic_batching.min_batch_size or default 64
    batch_size = args.batch_size or batching_config.get('batch_size') or training_config.get('batch_size')
    if batch_size is None:
        # Check if dynamic batching is enabled (can be at top-level, under training, or under training.batching)
        db_config = (config.get('dynamic_batching', {}) or
                     training_config.get('dynamic_batching', {}) or
                     batching_config.get('dynamic_batching', {}))
        if db_config.get('enabled', False):
            batch_size = db_config.get('min_batch_size', 64)
        else:
            batch_size = 8  # Default fallback

    # Learning rate: check optimizer config first (nested), then flat training config
    learning_rate = args.learning_rate or optimizer_cfg.get('learning_rate') or training_config.get('learning_rate', 5e-5)

    # Schedule settings: check nested schedule config first, then flat training config
    num_epochs = args.epochs or schedule_config.get('num_epochs') or training_config.get('num_epochs', 3)
    max_steps = schedule_config.get('max_steps') or training_config.get('max_steps')
    warmup_steps = schedule_config.get('warmup_steps') or training_config.get('warmup_steps', 1000)

    # Batching settings
    gradient_accumulation_steps = batching_config.get('gradient_accumulation_steps') or training_config.get('gradient_accumulation_steps', 1)

    # Logging settings: check nested logging config first, then flat training config
    save_steps = logging_config.get('save_steps') or training_config.get('save_steps', 1000)

    # Generation testing configuration (check nested generation config first)
    generate_every_n_steps = generation_config.get('every_n_steps') or training_config.get('generate_every_n_steps', 500)
    num_generations_per_step = generation_config.get('num_per_step') or training_config.get('num_generations_per_step', 1)
    generation_max_length = generation_config.get('max_length') or training_config.get('generation_max_length', 128)
    generation_temperature = generation_config.get('temperature') or training_config.get('generation_temperature', 0.7)
    generation_top_p = generation_config.get('top_p') or training_config.get('generation_top_p', 0.9)
    generation_top_k = generation_config.get('top_k') or training_config.get('generation_top_k', 50)
    generation_repetition_penalty = generation_config.get('repetition_penalty') or training_config.get('generation_repetition_penalty', 1.0)
    generation_skip_special_tokens = generation_config.get('skip_special_tokens', training_config.get('generation_skip_special_tokens', True))
    generation_prompt = generation_config.get('prompt') or training_config.get('generation_prompt', None)

    # Data configuration
    data_config = config.get('data', {})
    seq_length = data_config.get('max_length', max_position_embeddings)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('dataloader_pin_memory', True)
    drop_last = data_config.get('dataloader_drop_last', True)
    data_dir = data_config.get('data_dir', '/project/code/data/pretokenized')  # Default to project data
    tokenizer_name = data_config.get('tokenizer_name', '/project/code/models/tokenizer/enhanced-50680')
    use_all_data = data_config.get('use_all_data', True)  # Use all available data by default

    # Load tokenizer for generation decoding
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        if rank == 0:
            logger.info(f"  Tokenizer name from config: {tokenizer_name}")
        if tokenizer_name:
            try:
                # Fix double-prefixed paths
                tokenizer_path = str(tokenizer_name)
                if '/code/code' in tokenizer_path:
                    tokenizer_path = tokenizer_path.replace('/code/code/', '/code/')
                    if rank == 0:
                        logger.info(f" Fixed tokenizer path: {tokenizer_path}")

                if rank == 0:
                    logger.info(f"  Loading tokenizer from: {tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                if rank == 0:
                    logger.info(f" Loaded tokenizer from {tokenizer_path} (vocab_size: {len(tokenizer)})")
            except Exception as e:
                if rank == 0:
                    logger.error(f" Failed to load tokenizer from {tokenizer_path}: {e}")
                    import traceback
                    logger.error(f"  Traceback: {traceback.format_exc()}")
        else:
            if rank == 0:
                logger.warning("  No tokenizer_name in config!")
    else:
        if rank == 0:
            logger.warning("Transformers not available. Generation will use token IDs.")

    # Performance configuration
    perf_config = config.get('performance', {})
    training_config = config.get('training', {})

    # Get precision config (new subcategory containing mixed_precision and quantization)
    precision_config = training_config.get('precision', {})

    # Get mixed precision - check both new (training.precision.mixed_precision) and old (training.mixed_precision) paths
    mixed_precision_setting = precision_config.get('mixed_precision', training_config.get('mixed_precision', 'bf16'))

    # Validate and determine dtype
    valid_precisions = ['bf16', 'fp16', 'fp32', 'no']
    if mixed_precision_setting not in valid_precisions:
        if rank == 0:
            logger.warning(f"   Invalid mixed_precision '{mixed_precision_setting}', valid options: {valid_precisions}. Defaulting to 'bf16'")
        mixed_precision_setting = 'bf16'

    # Determine dtype and whether to use AMP
    if mixed_precision_setting == 'bf16':
        amp_dtype = torch.bfloat16
        use_amp = True
    elif mixed_precision_setting == 'fp16':
        amp_dtype = torch.float16
        use_amp = True
    else:  # fp32 or 'no'
        amp_dtype = torch.float32
        use_amp = False

    # Get quantization config - check both new (training.precision.quantization) and old (training.quantization) paths
    quant_config = precision_config.get('quantization', training_config.get('quantization', {}))
    quant_enabled = quant_config.get('enabled', False)
    quant_type = quant_config.get('type', 'none').lower() if quant_enabled else 'none'

    # Validate quantization type
    valid_quant_types = ['none', 'int8', 'nf4', 'nvfp4', 'fp4']
    if quant_enabled and quant_type not in valid_quant_types:
        if rank == 0:
            logger.warning(f"   Invalid quantization type '{quant_type}', valid options: {valid_quant_types}. Disabling quantization.")
        quant_enabled = False
        quant_type = 'none'

    if rank == 0:
        logger.info(f"   Mixed precision: {mixed_precision_setting} (use_amp={use_amp})")
        if quant_enabled:
            logger.info(f"   Quantization: {quant_type}")

    # Logging configuration - check both 'logging' and 'wandb' sections
    logging_config = config.get('logging', {})
    wandb_config_section = config.get('wandb', {})

    # Prioritize 'wandb' section if it exists, otherwise use 'logging' section
    use_wandb = wandb_config_section.get('use_wandb', False) if wandb_config_section else logging_config.get('use_wandb', False)

    if wandb_config_section:
        wandb_config = {
            'project': wandb_config_section.get('project', 'transformer-training'),
            'entity': wandb_config_section.get('entity', None),
            'tags': wandb_config_section.get('tags', []),
            'name': wandb_config_section.get('name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
        }
    else:
        wandb_config = logging_config.get('wandb', {}) if use_wandb else {}

    if rank == 0:
        logger.info(f"\n Training Config:")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Learning rate: {learning_rate:.2e}")
        logger.info(f"   Gradient accumulation: {gradient_accumulation_steps}")

    # Create model - use MoE model from config if available
    if rank == 0:
        logger.info(f"\n Creating MoE model...")

    # Build MoE config from YAML config
    # Map config names to EnhancedMoEConfig field names
    moe_config = EnhancedMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=model_config.get('num_attention_heads', 8),
        intermediate_size=model_config.get('intermediate_size', hidden_size * 4),
        num_experts=model_config.get('num_experts', 2),
        num_experts_per_token=model_config.get('num_experts_per_token', 1),
        max_position_embeddings=max_position_embeddings,
        router_type=model_config.get('router_type', 'switch'),
        expert_capacity_factor=model_config.get('capacity_factor', 1.25),
        attention_dropout=model_config.get('attention_dropout', 0.1),
        dropout=model_config.get('dropout', 0.1),
        use_flash_attention=model_config.get('use_flash_attention', False),
        router_aux_loss_coef=model_config.get('router_z_loss_coef', 0.01),
        router_jitter_noise=model_config.get('router_jitter_noise', 0.01),
    )

    model = EnhancedMoEModel(moe_config)

    # Apply quantization if enabled (before moving to device)
    if quant_enabled:
        if rank == 0:
            logger.info(f"   Applying {quant_type} quantization...")
        model = apply_quantization(model, quant_config, logger if rank == 0 else None)

    model = model.to(device)

    # Apply hybrid caching if enabled
    hybrid_cache_config = config.get('hybrid_caching', {})
    if hybrid_cache_config.get('enabled', False):
        try:
            from code.src.Ava.training.optimizations.hybrid_cache import (
                apply_hybrid_caching,
                HybridCacheConfig,
            )
            cache_config = HybridCacheConfig(
                enabled=True,
                max_cache_size_gb=hybrid_cache_config.get('max_cache_size_gb', 0.5),
                kv_cache_ratio=hybrid_cache_config.get('kv_cache_ratio', 0.7),
                eviction_policy=hybrid_cache_config.get('eviction_policy', 'hybrid'),
                prefetch_enabled=hybrid_cache_config.get('prefetch_enabled', False),
                prefetch_lookahead=hybrid_cache_config.get('prefetch_lookahead', 2),
                min_score_threshold=hybrid_cache_config.get('min_score_threshold', 0.1),
            )
            if rank == 0:
                logger.info(f"\n Applying hybrid caching...")
                logger.info(f"   Cache size: {cache_config.max_cache_size_gb}GB")
                logger.info(f"   Policy: {cache_config.eviction_policy}")
            model, hybrid_cache = apply_hybrid_caching(model, cache_config)
            if rank == 0:
                logger.info(f"   Hybrid caching applied (20-30% throughput improvement)")
        except Exception as e:
            if rank == 0:
                logger.warning(f"   Hybrid caching failed: {e}")
                logger.warning(f"   Continuing without hybrid caching...")

    # Apply torch.compile if enabled (after moving to device, before DDP)
    perf_config = config.get('performance', {})
    if perf_config.get('enable_torch_compile', False):
        compile_mode = perf_config.get('torch_compile_mode', 'reduce-overhead')
        compile_dynamic = perf_config.get('torch_compile_dynamic', True)
        compile_fullgraph = perf_config.get('torch_compile_fullgraph', False)
        if rank == 0:
            logger.info(f"\n Applying torch.compile...")
            logger.info(f"   Mode: {compile_mode}")
            logger.info(f"   Dynamic: {compile_dynamic}")
            logger.info(f"   Fullgraph: {compile_fullgraph}")
        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                dynamic=compile_dynamic,
                fullgraph=compile_fullgraph,
            )
            if rank == 0:
                logger.info(f"   torch.compile applied successfully (15-25% speedup after warmup)")
        except Exception as e:
            if rank == 0:
                logger.warning(f"   torch.compile failed: {e}")
                logger.warning(f"   Continuing without compilation...")

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        if quant_enabled:
            # Estimate memory savings based on quantization type
            if quant_type == 'int8':
                bytes_per_param = 1
            elif quant_type in ['nf4', 'nvfp4', 'fp4']:
                bytes_per_param = 0.5
            else:
                bytes_per_param = 4
            estimated_memory_mb = total_params * bytes_per_param / (1024 * 1024)
            logger.info(f"   {quant_type.upper()} estimated weight memory: {estimated_memory_mb:.1f}MB")

    # Wrap in DDP if distributed
    if world_size > 1 and DISTRIBUTED_AVAILABLE:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )

    # Create optimizer and scheduler
    if rank == 0:
        logger.info(f"\n Setting up optimizer and scheduler...")

    # Get optimizer settings from config (supports both flat and nested structure)
    optimizer_config = training_config.get('optimizer', {})
    if isinstance(optimizer_config, dict):
        # New nested structure: training.optimizer.type, training.optimizer.weight_decay, etc.
        optimizer_type = optimizer_config.get('type', 'adamw').lower()
        weight_decay = optimizer_config.get('weight_decay', training_config.get('weight_decay', 0.01))
        lion_betas = optimizer_config.get('lion_betas', [0.9, 0.99])
    else:
        # Old flat structure: training.optimizer = 'lion'
        optimizer_type = str(optimizer_config).lower()
        weight_decay = training_config.get('weight_decay', 0.01)
        lion_betas = training_config.get('lion_betas', [0.9, 0.99])

    if optimizer_type == 'lion':
        # Import Lion optimizer
        try:
            from lion_pytorch import Lion
            optimizer = Lion(model.parameters(), lr=learning_rate, betas=tuple(lion_betas), weight_decay=weight_decay)
            if rank == 0:
                logger.info(f"   Optimizer: Lion (lr={learning_rate:.2e}, betas={lion_betas})")
        except ImportError:
            if rank == 0:
                logger.warning("Lion optimizer not available, falling back to AdamW")
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        # Default to AdamW
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if rank == 0:
            logger.info(f"   Optimizer: AdamW (lr={learning_rate:.2e})")

    # Create dataloaders
    if rank == 0:
        logger.info(f"\n Creating dataloaders...")

    # Prepare tokenizer for turn-aware loader (if using)
    tokenizer_for_loader = None
    if hasattr(args, 'use_turn_aware_loader') and args.use_turn_aware_loader:
        try:
            if TRANSFORMERS_AVAILABLE:
                from transformers import AutoTokenizer
                # Fix double-prefixed tokenizer paths
                tokenizer_path = tokenizer_name or "gpt2"
                if tokenizer_path and '/code/code' in str(tokenizer_path):
                    tokenizer_path = str(tokenizer_path).replace('/code/code/', '/code/')
                    if rank == 0:
                        logger.info(f" Fixed tokenizer path: {tokenizer_path}")

                tokenizer_for_loader = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True  # Allow loading local tokenizer.json
                )
        except Exception as e:
            if rank == 0:
                logger.warning(f"Failed to load tokenizer for turn-aware loader: {e}")

    # Use optimized DataLoaderManager for ultra-fast data loading
    if rank == 0:
        logger.info(" Creating dataloaders with DataLoaderManager...")

    try:
        # Import DynamicConfig to convert dict to config object
        from src.Ava.config.training_config import DynamicConfig

        # Convert config dict to DynamicConfig if it's a dict
        if isinstance(config, dict):
            config_obj = DynamicConfig(config)
        else:
            config_obj = config

        # Create training context for DataLoaderManager
        context = TrainingContext(model=model, config=config_obj, device=device)
        loader_manager = DataLoaderManager(context)

        # Create dataloaders using optimized manager
        train_loader, val_loader = loader_manager.create_dataloaders(
            training_config=config_obj,
            tokenizer=tokenizer_for_loader if tokenizer_for_loader else None,
            config_dict=config,
            batch_size=batch_size
        )

        if rank == 0:
            logger.info(" Dataloaders created with DataLoaderManager (optimized)")
    except Exception as e:
        if rank == 0:
            logger.warning(f"DataLoaderManager failed ({e}), falling back to create_dataloaders: {e}")

        # Extract dynamic_batching config for fallback loader (can be at top-level, under training, or under training.batching)
        dynamic_batching_config = None
        if isinstance(config, dict):
            # Check all possible locations for dynamic_batching config
            db_config = (config.get('dynamic_batching') or
                         config.get('training', {}).get('dynamic_batching', {}) or
                         config.get('training', {}).get('batching', {}).get('dynamic_batching', {}))
            if db_config and db_config.get('enabled', False):
                dynamic_batching_config = db_config
                if rank == 0:
                    logger.info(" Dynamic batching will be enabled in fallback loader")

        # Fallback to old method
        train_loader, val_loader = create_dataloaders(
            batch_size=batch_size,
            seq_length=seq_length,
            vocab_size=vocab_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            pin_memory=pin_memory,
            drop_last=drop_last,
            data_dir=data_dir,
            use_turn_aware_loader=getattr(args, 'use_turn_aware_loader', False),
            tokenizer=tokenizer_for_loader,
            use_all_data=use_all_data,
            dynamic_batching_config=dynamic_batching_config,
        )

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Prepare config dict for checkpoint saving
    # Ensure tokenizer_name is absolute path (avoid double-prefixing)
    tokenizer_name_absolute = tokenizer_name
    if tokenizer_name and not tokenizer_name.startswith('/'):
        # Only prepend project_root if path is relative (doesn't start with /)
        tokenizer_name_absolute = str(project_root / tokenizer_name)

    config_for_checkpoint = {
        'model': {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': model_config.get('num_attention_heads', 8),
            'intermediate_size': model_config.get('intermediate_size', hidden_size * 4),
            'num_experts': model_config.get('num_experts', 2),
            'num_experts_per_token': model_config.get('num_experts_per_token', 1),
            'max_position_embeddings': max_position_embeddings,
            'router_type': model_config.get('router_type', 'switch'),
            'expert_capacity_factor': model_config.get('capacity_factor', 1.25),
            'attention_dropout': model_config.get('attention_dropout', 0.1),
            'dropout': model_config.get('dropout', 0.1),
            'use_flash_attention': model_config.get('use_flash_attention', False),
            'router_aux_loss_coef': model_config.get('router_z_loss_coef', 0.01),
            'router_jitter_noise': model_config.get('router_jitter_noise', 0.01),
        },
        'data': {
            'tokenizer_name': tokenizer_name_absolute,
            'max_length': seq_length,
        }
    }

    # Checkpoint manager and metrics tracker
    checkpoint_manager = CheckpointManager(Path(args.save_dir), config=config_for_checkpoint)
    metrics_tracker = MetricsTracker(Path(args.log_dir), use_wandb=use_wandb, wandb_config=wandb_config)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if rank == 0:
            logger.info(f"\n Loading checkpoint from {args.resume}...")
        start_epoch, _ = checkpoint_manager.load(model, optimizer, Path(args.resume))

    # Training loop
    if rank == 0:
        logger.info(f"\n" + "="*80)
        logger.info("  STARTING TRAINING")
        logger.info("="*80)

    # Initialize ThreadPoolExecutor for async generation on CPU
    generation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AsyncGeneration")
    generation_future = None  # Track ongoing generation task

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        # Train epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch,
            log_interval=args.log_interval,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            logger=logger if rank == 0 else None,
            metrics_tracker=metrics_tracker if rank == 0 else None,
            vocab_size=vocab_size,
            generate_every_n_steps=generate_every_n_steps,
            num_generations_per_step=num_generations_per_step,
            generation_max_length=generation_max_length,
            generation_temperature=generation_temperature,
            generation_top_p=generation_top_p,
            generation_top_k=generation_top_k,
            generation_repetition_penalty=generation_repetition_penalty,
            tokenizer=tokenizer,
            generation_skip_special_tokens=generation_skip_special_tokens,
            generation_prompt=generation_prompt,
            checkpoint_manager=checkpoint_manager if rank == 0 else None,
            save_steps=save_steps,
            generation_executor=generation_executor if rank == 0 else None,
            log_dir=Path(args.log_dir) if rank == 0 else None,
        )

        if rank == 0:
            logger.info(f" Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(model, val_loader, device, use_amp=use_amp,
                              amp_dtype=amp_dtype, logger=logger if rank == 0 else None)

            if rank == 0:
                logger.info(f" Validation Loss: {val_loss:.4f}")
                metrics_tracker.log_validation(epoch, loss=val_loss)

                # Save checkpoint if best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_manager.save(model, optimizer, epoch, 0, {'val_loss': val_loss})
                    logger.info(f" Saved checkpoint (val_loss: {val_loss:.4f})")

    # Cleanup async operations
    if rank == 0:
        # Wait for any pending checkpoint saves
        if checkpoint_manager is not None:
            logger.info("\n Waiting for pending checkpoint saves...")
            checkpoint_manager.wait_for_pending_save()
            checkpoint_manager.shutdown()
            logger.info(" Checkpoint manager shut down successfully")

        # Shutdown async generation executor
        if generation_executor is not None:
            logger.info(" Shutting down async generation executor...")
            generation_executor.shutdown(wait=True)
            logger.info(" Async generation executor shut down successfully")

    # Final summary
    if rank == 0:
        logger.info(f"\n" + "="*80)
        logger.info(" TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f" Summary:")
        logger.info(f"   Total epochs: {num_epochs}")
        logger.info(f"   Best validation loss: {best_val_loss:.4f}")
        logger.info(f"   Model parameters: {total_params:,}")
        logger.info(f"   Save directory: {args.save_dir}")

        metrics_tracker.save_summary(Path(args.log_dir) / 'metrics_summary.json')
        metrics_tracker.finish()

    cleanup_distributed(rank, world_size)


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train 100M parameter transformer model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic training
  python train_100m_full.py

  # With config
  python train_100m_full.py --config configs/moe/tiny_moe.yaml

  # With custom parameters
  python train_100m_full.py --epochs 10 --batch-size 32 --learning-rate 5e-5

  # Resume from checkpoint
  python train_100m_full.py --resume checkpoints/checkpoint_epoch_5_step_0.pt

  # Distributed training (4 GPUs)
  torchrun --nproc_per_node=4 train_100m_full.py --config configs/moe/tiny_moe.yaml
        '''
    )

    # Model and data
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per GPU (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (default: 5e-5)')

    # Checkpointing and logging
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logs and metrics')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log interval in batches')
    parser.add_argument('--val-interval', type=int, default=1,
                       help='Validation interval in epochs')

    parser.add_argument('--use-turn-aware-loader', action='store_true', default=True,
                       help='Enable turn-aware conversation data loading for improved coherence (default: True for conversation datasets)')

    parser.add_argument('--disable-turn-aware-loader', action='store_true',
                       help='Disable turn-aware loading and use standard loader')

    args = parser.parse_args()

    # Handle turn-aware loader flags
    # Default: enabled for conversation datasets, disabled if explicitly requested
    if args.disable_turn_aware_loader:
        args.use_turn_aware_loader = False
    else:
        # Enable by default (or keep enabled if --use-turn-aware-loader was specified)
        args.use_turn_aware_loader = True

    main(args)