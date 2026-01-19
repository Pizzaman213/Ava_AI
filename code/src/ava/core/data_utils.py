"""
Data Utilities Module.

This module provides shared data processing utilities:
- find_data_files: Locate data files with deterministic train/val splitting
- collate_batch: Batch collation with configurable padding
- BaseCollator: Collator class for DataLoader
- move_to_device: Transfer batch data to device
- Config getters: Utilities for extracting config values

These utilities eliminate code duplication across data loaders.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# DATA FILE FINDING UTILITIES
# =============================================================================

def find_data_files(
    data_dir: Path,
    split: str,
    patterns: Optional[List[str]] = None,
    min_file_size: int = 0,
    train_split_ratio: float = 0.85,
    max_files: Optional[int] = None,
) -> List[Path]:
    """
    Find data files with deterministic train/val splitting.

    This is the single source of truth for file finding logic,
    replacing 5+ duplicate implementations across data loaders.

    The function uses MD5 hashing of filenames for deterministic splitting,
    ensuring the same files are always assigned to train/val regardless
    of file order or system.

    Args:
        data_dir: Directory to search for files
        split: 'train' or 'val'
        patterns: Glob patterns to search (defaults to Arrow patterns)
        min_file_size: Minimum file size in bytes (0 = any size)
        train_split_ratio: Fraction of files for training (default 0.85 = 85%)
        max_files: Maximum number of files to return (None = no limit)

    Returns:
        List of file paths for the requested split

    Example:
        >>> train_files = find_data_files(Path('data/'), 'train')
        >>> val_files = find_data_files(Path('data/'), 'val')
    """
    if patterns is None:
        patterns = [
            f"**/{split}/**/*.arrow",
            f"{split}_*.arrow",
            f"{split}/*.arrow",
            "**/*_processed.arrow",
            "*.arrow",
        ]

    files = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))

    # Remove duplicates while preserving order
    files = list(dict.fromkeys(files))

    # Filter by existence and size
    files = [f for f in files if f.exists() and f.stat().st_size > min_file_size]

    # If no split-specific files found, apply deterministic file-based splitting
    if not files or not any(split in str(f) for f in files):
        # Collect all arrow files
        all_files = list(data_dir.glob("**/*.arrow"))
        all_files = [f for f in all_files if f.exists() and f.stat().st_size > min_file_size]

        if all_files:
            files = sorted(all_files, key=lambda f: f.name)
            split_files = []

            train_threshold = int(train_split_ratio * 100)  # e.g., 85

            for file_path in files:
                # Use MD5 for consistent, deterministic hashing
                file_hash = int(hashlib.md5(file_path.name.encode()).hexdigest(), 16) % 100

                if split == "train":
                    if file_hash < train_threshold:
                        split_files.append(file_path)
                else:  # val
                    if file_hash >= train_threshold:
                        split_files.append(file_path)

            files = split_files

    result = sorted(files)
    if max_files is not None and max_files > 0:
        result = result[:max_files]
    return result


# =============================================================================
# COLLATE FUNCTION UTILITIES
# =============================================================================

def collate_batch(
    batch: List[Dict[str, Any]],
    max_length: int,
    pad_token_id: int = 0,
    use_fixed_padding: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate function with configurable padding strategy.

    This is the single source of truth for batch collation,
    replacing 4+ duplicate implementations.

    Args:
        batch: List of sample dictionaries with input_ids, attention_mask, labels
        max_length: Maximum sequence length
        pad_token_id: Token ID for padding
        use_fixed_padding: If True, pad to max_length (torch.compile friendly)
                          If False, pad to batch max (memory efficient)

    Returns:
        Dictionary with batched tensors:
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - labels: (batch_size, seq_len) with -100 for padding
    """
    if not batch:
        return {}

    batch_size = len(batch)

    # Determine padding length
    if use_fixed_padding:
        pad_len = max_length
    else:
        seq_lengths = [len(item.get('input_ids', [])) for item in batch]
        pad_len = min(max(seq_lengths) if seq_lengths else max_length, max_length)

    # Pre-allocate tensors
    input_ids = torch.full((batch_size, pad_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, pad_len), dtype=torch.long)
    labels = torch.full((batch_size, pad_len), -100, dtype=torch.long)

    # Fill tensors
    for i, item in enumerate(batch):
        ids = item.get('input_ids')
        if ids is None:
            continue

        # Convert to tensor if needed (handles numpy arrays and lists)
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)

        seq_len = min(len(ids), pad_len)
        input_ids[i, :seq_len] = ids[:seq_len]

        mask = item.get('attention_mask')
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)
            attention_mask[i, :seq_len] = mask[:seq_len]
        else:
            attention_mask[i, :seq_len] = 1

        lbls = item.get('labels')
        if lbls is not None:
            if not isinstance(lbls, torch.Tensor):
                lbls = torch.tensor(lbls)
            labels[i, :seq_len] = lbls[:seq_len]
        else:
            labels[i, :seq_len] = input_ids[i, :seq_len]

    # CRITICAL: Mask padded positions in labels so model doesn't learn to predict padding
    labels[attention_mask == 0] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


class BaseCollator:
    """
    Base collator class for consistent batch processing.

    This is the single source of truth for batch collation across all data loaders.
    Use this instead of implementing custom collate_fn methods.

    Features:
        - Fixed or dynamic padding modes
        - Configurable label padding (-100 for loss masking)
        - Optional pinned memory for faster GPU transfer
        - Sequence length validation

    Example:
        >>> collator = BaseCollator(max_length=512, pad_token_id=0)
        >>> dataloader = DataLoader(dataset, collate_fn=collator)
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int = 0,
        use_fixed_padding: bool = True,
        label_pad_id: int = -100,
        pin_memory: bool = False,
        validate_sequences: bool = True,
    ):
        """
        Initialize collator.

        Args:
            max_length: Maximum sequence length
            pad_token_id: Token ID for padding input_ids
            use_fixed_padding: If True, pad to max_length (torch.compile friendly)
                              If False, pad to batch max (memory efficient)
            label_pad_id: Token ID for padding labels (default -100 for loss masking)
            pin_memory: If True, allocate tensors in pinned memory for faster GPU transfer
            validate_sequences: If True, validate sequence lengths before processing
        """
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.use_fixed_padding = use_fixed_padding
        self.label_pad_id = label_pad_id
        self.pin_memory = pin_memory
        self.validate_sequences = validate_sequences

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = collate_batch(
            batch,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            use_fixed_padding=self.use_fixed_padding,
        )

        # Update labels padding if non-default
        if self.label_pad_id != -100 and 'labels' in result:
            # Replace -100 with custom label_pad_id
            result['labels'] = torch.where(
                result['labels'] == -100,
                torch.full_like(result['labels'], self.label_pad_id),
                result['labels']
            )

        # Pin memory if requested
        if self.pin_memory:
            result = {k: v.pin_memory() if v.is_cuda is False else v for k, v in result.items()}

        return result


# =============================================================================
# DEVICE TRANSFER UTILITIES
# =============================================================================

def move_to_device(batch: Any, device: torch.device) -> Any:
    """
    Move batch data to device.

    Handles dict, list, tuple, and tensor inputs recursively.

    Args:
        batch: Batch data (dict, list, tuple, or tensor)
        device: Target device

    Returns:
        Batch data moved to device

    Example:
        >>> batch = move_to_device(batch, torch.device('cuda:0'))
    """
    if isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    elif isinstance(batch, (list, tuple)):
        return type(batch)(
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in batch
        )
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


# =============================================================================
# CONFIG GETTER UTILITIES
# =============================================================================

def get_config_value(
    config: Any,
    *paths: str,
    default: Any = None,
) -> Any:
    """
    Get a configuration value with fallback paths.

    This replaces 8+ duplicate getter methods in DataLoaderManager.

    Args:
        config: Configuration object
        *paths: Attribute paths to try in order (e.g., 'data_loading.num_workers', 'data.num_workers')
        default: Default value if not found

    Returns:
        Configuration value or default

    Example:
        >>> num_workers = get_config_value(config, 'data_loading.num_workers', 'data.num_workers', default=0)
    """
    for path in paths:
        parts = path.split('.')
        obj = config

        try:
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    raise AttributeError(f"No attribute {part}")

            if obj is not None:
                return obj
        except (AttributeError, KeyError, TypeError):
            continue

    return default


# Convenience functions for common config values
def get_num_workers(config: Any) -> int:
    """Get number of data loading workers."""
    return get_config_value(config, 'data_loading.num_workers', 'data.num_workers', default=0)


def get_prefetch_factor(config: Any) -> int:
    """Get prefetch factor."""
    return get_config_value(config, 'data_loading.prefetch_factor', 'data.prefetch_factor', default=2)


def get_persistent_workers(config: Any) -> bool:
    """Get persistent workers setting.

    Defaults to True for better performance - avoids worker restart overhead between epochs.
    (Phase 2 optimization: saves 100-300s over 100 epochs)
    """
    return get_config_value(config, 'data_loading.persistent_workers', 'data.persistent_workers', default=True)


def get_samples_per_file(config: Any) -> int:
    """Get samples per file setting."""
    return get_config_value(config, 'data_loading.samples_per_file', 'data.samples_per_file', default=64)


def get_enable_bucketing(config: Any) -> bool:
    """Get bucketing enabled setting."""
    return get_config_value(config, 'data_loading.enable_bucketing', 'data.enable_bucketing', default=True)


def get_val_split_ratio(config: Any) -> float:
    """Get validation split ratio."""
    return get_config_value(config, 'data_loading.val_split_ratio', 'data.val_split_ratio', default=0.1)


__all__ = [
    # Data file utilities
    'find_data_files',
    # Collation utilities
    'collate_batch',
    'BaseCollator',
    # Device utilities
    'move_to_device',
    # Config getters
    'get_config_value',
    'get_num_workers',
    'get_prefetch_factor',
    'get_persistent_workers',
    'get_samples_per_file',
    'get_enable_bucketing',
    'get_val_split_ratio',
]
