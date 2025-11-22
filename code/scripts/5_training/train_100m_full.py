#!/usr/bin/env python3
"""
Full-Featured Training Script for 100M Parameter Model
=========================================================

This script demonstrates a production-ready training pipeline with:
✓ Model initialization and management
✓ Data loading and preprocessing
✓ Distributed training support
✓ Learning rate scheduling
✓ Checkpointing and resumption
✓ Metrics tracking and logging
✓ Mixed precision training
✓ Gradient accumulation
✓ Validation and evaluation
✓ Weights & Biases integration (optional)
✓ DeepSpeed integration (optional)

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

    # Distributed training (multi-GPU - AUTOMATIC)
    # Automatically detects and uses all available GPUs
    python train_100m_full.py --config configs/moe/tiny_moe.yaml

    # Or with explicit GPU count using torchrun
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

import os
import sys
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, Any
import argparse

# Configure PyTorch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import MoE model and utilities
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.config.yaml_loader import load_yaml_with_path_resolution
from src.Ava.training.train.data_loader_manager import DataLoaderManager
from src.Ava.training.train.base import TrainingContext

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

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: Path, rank: int = 0) -> logging.Logger:
    """Setup logging for training."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('train_100m')
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Only log on rank 0 in distributed training
        if rank == 0:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            file_handler = logging.FileHandler(
                log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

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
) -> Tuple[DataLoader, DataLoader]:
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

                print(f"\n✓ Turn-Aware Conversation Loading ENABLED")
                print(f"  Data: {train_jsonl}")
                print(f"  Benefits: Improved dialogue coherence, speaker awareness, quality tracking")

                return train_loader, val_loader
            except Exception as e:
                print(f"\n⚠ Turn-aware loading failed ({e}), falling back to standard loading")
        else:
            print(f"\n⚠ No conversation JSONL files found in {data_dir}, using standard loading")

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
        print(f"⚠️  Fixed double-prefixed path to: {data_dir}")

    # Log data loading configuration
    if rank == 0:
        print(f"📂 Data directory: {data_dir}")
        print(f"📦 Using all data: {use_all_data}")
        print(f"📊 Data dir exists: {Path(data_dir).exists()}")
        print(f"📚 Datasets available: {DATASETS_AVAILABLE}")

    # Try to load real data first
    if data_dir and Path(data_dir).exists() and DATASETS_AVAILABLE:
        try:
            if rank == 0:
                print("🔄 Starting data loading...")

            datasets_list = []

            # Load Arrow files (.arrow directories and files) if use_all_data is True
            if use_all_data:
                arrow_files = sorted(list(Path(data_dir).glob('*.arrow'))) + \
                             sorted(list(Path(data_dir).glob('**/*.arrow')))
                if rank == 0 and arrow_files:
                    print(f"🔍 Found {len(arrow_files)} Arrow files")
                for arrow_file in arrow_files:
                    try:
                        file_size = arrow_file.stat().st_size
                        if file_size == 0:
                            continue
                        dataset = load_dataset('arrow', data_files=str(arrow_file))
                        split_name = list(dataset.keys())[0]
                        dataset = dataset[split_name]
                        datasets_list.append(dataset)
                    except Exception as e:
                        continue

            # Load ALL Parquet files (.parquet) if use_all_data is True
            if use_all_data:
                parquet_files = sorted(list(Path(data_dir).glob('*.parquet')))
            else:
                # Load only limited parquet files if not using all data
                parquet_files = sorted(list(Path(data_dir).glob('*.parquet')))[:5]  # Default: first 5

            if rank == 0:
                print(f"🔍 Found {len(parquet_files)} Parquet files")

            for i, parquet_file in enumerate(parquet_files, 1):
                try:
                    file_size = parquet_file.stat().st_size
                    if file_size == 0:
                        continue
                    dataset = load_dataset('parquet', data_files=str(parquet_file))
                    split_name = list(dataset.keys())[0]
                    dataset = dataset[split_name]
                    datasets_list.append(dataset)
                    if rank == 0 and i % 10 == 0:
                        print(f"  Loaded {i}/{len(parquet_files)} parquet files...")
                except Exception as e:
                    if rank == 0:
                        print(f"  Failed to load {parquet_file.name}: {e}")
                    continue

            if datasets_list:
                # Concatenate all datasets (Arrow + Parquet combined)
                from datasets import concatenate_datasets
                dataset = concatenate_datasets(datasets_list)
                total_examples = len(dataset)
                if rank == 0:
                    print(f"✓ Loaded {len(datasets_list)} data files with {total_examples:,} total examples")
                    num_batches = total_examples // batch_size
                    print(f"✓ Expected batches per epoch: {num_batches:,}")
                train_dataset = ArrowDataset(dataset, seq_length, vocab_size)
                val_dataset = ArrowDataset(dataset, seq_length, vocab_size)
            else:
                if rank == 0:
                    print("⚠ No datasets loaded from data directory")
        except Exception as e:
            # Failed to load data, will use dummy dataset
            if rank == 0:
                print(f"⚠ Warning: Failed to load real data: {type(e).__name__}: {e}")
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(rank == 0),  # Only shuffle on rank 0
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop last on validation
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
    """Manage model checkpoints."""

    def __init__(self, save_dir: Path, max_keep: int = 3, config: Optional[Dict] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints = []
        self.config = config

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, step: int, metrics: Dict[str, float]):
        """Save checkpoint in format compatible with generate.py."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict() if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,  # Include config for generation script
        }

        path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'
        torch.save(checkpoint, path)
        self.checkpoints.append(path)

        # Also save as latest_model.pt for generate.py
        latest_path = self.save_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)

        # Update best model if this is a good checkpoint
        if 'val_loss' in metrics:
            best_path = self.save_dir / 'best_model.pt'
            if not best_path.exists() or metrics.get('val_loss', float('inf')) < self._get_best_loss(best_path):
                torch.save(checkpoint, best_path)

        # Remove old checkpoints
        if len(self.checkpoints) > self.max_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

        return path

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

def setup_distributed(auto_multi_gpu: bool = True):
    """Setup distributed training if available.

    Args:
        auto_multi_gpu: If True, automatically use all available GPUs
    """
    if not DISTRIBUTED_AVAILABLE:
        return 0, 1

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        init_process_group(backend='nccl')
        return rank, world_size

    # Auto-detect and setup multi-GPU training if available
    if auto_multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Set up multi-GPU training automatically
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = str(num_gpus)
            try:
                init_process_group(backend='nccl')
                return 0, num_gpus
            except Exception as e:
                # Fall back to single GPU if distributed setup fails
                print(f"Warning: Failed to setup multi-GPU training: {e}")
                return 0, 1

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

        # Generate tokens one by one
        for _ in range(max_length - 1):
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
) -> float:
    """Train for one epoch."""

    model.train()
    total_loss = 0.0
    num_batches = 0
    global_step = epoch * len(train_loader)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch + 1}", disable=logger is None)

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for batch_idx, batch in pbar:
        try:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass with mixed precision
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss'] / gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss'] / gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if use_amp:
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
                logger.info(
                    f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss_value:.4f} | Avg Loss: {avg_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                # Check gradients after optimizer step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    grad_stats = check_gradients(model, logger)
                    if metrics_tracker is not None:
                        metrics_tracker.log_gradients(global_step, grad_stats)

            # Generation testing during training
            if generate_every_n_steps > 0 and global_step % generate_every_n_steps == 0 and logger is not None:
                logger.info(f"\n🎯 Testing generation at step {global_step}...")
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
                    top_k=generation_top_k,  # NEW: Pass top_k parameter
                    repetition_penalty=generation_repetition_penalty  # NEW: Pass repetition penalty
                )
                logger.info(f"✓ Generation test complete\n")

            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        except Exception as e:
            if logger is not None:
                logger.error(f"Error in batch {batch_idx}: {e}")
            continue

    avg_epoch_loss = total_loss / max(num_batches, 1)
    return avg_epoch_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
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
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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

    # Auto-detect available GPUs and setup distributed training
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Setup distributed training (auto-detect multiple GPUs)
    rank, world_size = setup_distributed(auto_multi_gpu=True)

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
        logger.info("🚀 STARTING 100M PARAMETER TRAINING")
        logger.info("="*80)
        logger.info(f"🖥️  Device: {device}")
        logger.info(f"📊 Available GPUs: {num_gpus}")
        if world_size > 1:
            logger.info(f"⚙️  Distributed Training: ENABLED")
            logger.info(f"   Rank {rank}/{world_size} - Using {world_size} GPUs")
        else:
            logger.info(f"📌 Single GPU Mode")

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
    batch_size = args.batch_size or training_config.get('batch_size', 8)
    learning_rate = args.learning_rate or training_config.get('learning_rate', 5e-5)
    num_epochs = args.epochs or training_config.get('num_epochs', 3)
    max_steps = training_config.get('max_steps')
    gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
    warmup_steps = training_config.get('warmup_steps', 1000)

    # Generation testing configuration
    generate_every_n_steps = training_config.get('generate_every_n_steps', 500)
    num_generations_per_step = training_config.get('num_generations_per_step', 1)
    generation_max_length = training_config.get('generation_max_length', 128)
    generation_temperature = training_config.get('generation_temperature', 0.7)
    generation_top_p = training_config.get('generation_top_p', 0.9)
    generation_top_k = training_config.get('generation_top_k', 50)  # NEW: Top-k sampling
    generation_repetition_penalty = training_config.get('generation_repetition_penalty', 1.0)  # NEW: Repetition penalty
    generation_skip_special_tokens = training_config.get('generation_skip_special_tokens', True)
    generation_prompt = training_config.get('generation_prompt', None)

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
                        logger.info(f"✓ Fixed tokenizer path: {tokenizer_path}")

                if rank == 0:
                    logger.info(f"  Loading tokenizer from: {tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                if rank == 0:
                    logger.info(f"✓ Loaded tokenizer from {tokenizer_path} (vocab_size: {len(tokenizer)})")
            except Exception as e:
                if rank == 0:
                    logger.error(f"✗ Failed to load tokenizer from {tokenizer_path}: {e}")
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
    mixed_precision = perf_config.get('float32_matmul_precision', 'high') if 'float32_matmul_precision' in perf_config else 'high'
    use_amp = True  # Always use mixed precision with bfloat16

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
        logger.info(f"\n📊 Training Config:")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Learning rate: {learning_rate:.2e}")
        logger.info(f"   Gradient accumulation: {gradient_accumulation_steps}")

    # Create model - use MoE model from config if available
    if rank == 0:
        logger.info(f"\n🤖 Creating MoE model...")

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
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Wrap in DDP if distributed
    if world_size > 1 and DISTRIBUTED_AVAILABLE:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )

    # Create optimizer and scheduler
    if rank == 0:
        logger.info(f"\n⚡ Setting up optimizer and scheduler...")

    # Get optimizer type from config
    optimizer_type = training_config.get('optimizer', 'adamw').lower()
    weight_decay = training_config.get('weight_decay', 0.01)

    if optimizer_type == 'lion':
        # Import Lion optimizer
        try:
            from lion_pytorch import Lion
            lion_betas = training_config.get('lion_betas', [0.9, 0.99])
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
        logger.info(f"\n📊 Creating dataloaders...")

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
                        logger.info(f"✓ Fixed tokenizer path: {tokenizer_path}")

                tokenizer_for_loader = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True  # Allow loading local tokenizer.json
                )
        except Exception as e:
            if rank == 0:
                logger.warning(f"Failed to load tokenizer for turn-aware loader: {e}")

    # Use optimized DataLoaderManager for ultra-fast data loading
    if rank == 0:
        logger.info("📊 Creating dataloaders with DataLoaderManager...")

    try:
        # Create training context for DataLoaderManager
        context = TrainingContext(model=model, config=config, device=device)
        loader_manager = DataLoaderManager(context)

        # Create dataloaders using optimized manager
        train_loader, val_loader = loader_manager.create_dataloaders(
            training_config=config,
            tokenizer=tokenizer_for_loader if tokenizer_for_loader else None,
            config_dict=config,
            batch_size=batch_size
        )

        if rank == 0:
            logger.info("✓ Dataloaders created with DataLoaderManager (optimized)")
    except Exception as e:
        if rank == 0:
            logger.warning(f"DataLoaderManager failed ({e}), falling back to create_dataloaders: {e}")
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
            logger.info(f"\n📂 Loading checkpoint from {args.resume}...")
        start_epoch, _ = checkpoint_manager.load(model, optimizer, Path(args.resume))

    # Training loop
    if rank == 0:
        logger.info(f"\n" + "="*80)
        logger.info("🏋️  STARTING TRAINING")
        logger.info("="*80)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        # Train epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch,
            log_interval=args.log_interval,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=True,
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
        )

        if rank == 0:
            logger.info(f"✅ Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(model, val_loader, device, use_amp=True,
                              logger=logger if rank == 0 else None)

            if rank == 0:
                logger.info(f"📈 Validation Loss: {val_loss:.4f}")
                metrics_tracker.log_validation(epoch, loss=val_loss)

                # Save checkpoint if best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_manager.save(model, optimizer, epoch, 0, {'val_loss': val_loss})
                    logger.info(f"💾 Saved checkpoint (val_loss: {val_loss:.4f})")

    # Final summary
    if rank == 0:
        logger.info(f"\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"📊 Summary:")
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
        description='Train 100M parameter transformer model with automatic multi-GPU support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic training (auto-detects GPUs)
  python train_100m_full.py

  # With config (automatically uses all available GPUs)
  python train_100m_full.py --config configs/moe/tiny_moe.yaml

  # With custom parameters
  python train_100m_full.py --epochs 10 --batch-size 32 --learning-rate 5e-5

  # Resume from checkpoint
  python train_100m_full.py --resume checkpoints/checkpoint_epoch_5_step_0.pt

  # Distributed training with explicit GPU count (if needed)
  torchrun --nproc_per_node=4 train_100m_full.py --config configs/moe/tiny_moe.yaml

Note: Multi-GPU training is automatically enabled when multiple GPUs are detected.
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
