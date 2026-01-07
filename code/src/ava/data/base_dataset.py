"""
Base dataset utilities and shared functionality.

This module provides:
- File discovery utilities for Arrow, Parquet, and JSONL files
- Common base classes for dataset implementations
- Shared caching and buffering patterns

Usage:
    from ava.data.base_dataset import discover_data_files, BaseArrowDataset

    # Discover data files
    files = discover_data_files('/path/to/data', split='train')

    # Or use base class
    class MyDataset(BaseArrowDataset):
        def __iter__(self):
            for file in self.files:
                yield from self._read_file(file)
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


# ============================================================================
# File Discovery Utilities
# ============================================================================

# Supported file extensions
SUPPORTED_EXTENSIONS = ('.arrow', '.parquet', '.jsonl', '.json')

# Default file patterns for split-aware discovery
def get_split_patterns(split: str) -> List[str]:
    """
    Get file patterns for a specific data split.

    Args:
        split: Data split name ('train', 'val', 'test', etc.)

    Returns:
        List of glob patterns to search
    """
    patterns = [
        # Split-specific patterns (highest priority)
        f"*/{split}/**/*.arrow",
        f"**/{split}/**/*.arrow",
        f"{split}_*.arrow",
        f"{split}/*.arrow",
        f"*/{split}/**/*.parquet",
        f"**/{split}/**/*.parquet",
        f"{split}_*.parquet",
        f"{split}/*.parquet",
        f"*/{split}/**/*.jsonl",
        f"**/{split}/**/*.jsonl",
        f"{split}_*.jsonl",
    ]
    return patterns


def get_fallback_patterns() -> List[str]:
    """
    Get fallback patterns when split-specific files aren't found.

    Returns:
        List of generic glob patterns
    """
    return [
        "processed*.jsonl",
        "*.jsonl",
        "*.arrow",
        "*.parquet",
        "**/*.arrow",
        "**/*.parquet",
        "**/*.jsonl",
    ]


def discover_data_files(
    data_dir: Union[str, Path],
    split: Optional[str] = None,
    extensions: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
    use_fallback: bool = True,
    recursive: bool = True,
) -> List[Path]:
    """
    Discover data files in a directory with split-aware patterns.

    This is the unified file discovery function that replaces duplicate
    implementations across streaming.py, pretokenized.py, indexed.py, etc.

    Args:
        data_dir: Directory to search for data files
        split: Optional data split ('train', 'val', 'test')
        extensions: File extensions to look for (default: arrow, parquet, jsonl)
        max_files: Maximum number of files to return (None = all)
        use_fallback: Whether to use fallback patterns if split-specific not found
        recursive: Whether to search subdirectories

    Returns:
        List of discovered file paths, sorted by name

    Example:
        >>> files = discover_data_files('/data', split='train')
        >>> files = discover_data_files('/data', extensions=['.arrow'])
        >>> files = discover_data_files('/data', max_files=10)
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return []

    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    found_files: List[Path] = []

    # Try split-specific patterns first
    if split:
        patterns = get_split_patterns(split)
        for pattern in patterns:
            if not recursive and '**' in pattern:
                continue
            for f in data_path.glob(pattern):
                if f.is_file() and f.suffix in extensions:
                    found_files.append(f)

        # Deduplicate while preserving order
        found_files = list(dict.fromkeys(found_files))

    # Fall back to generic patterns if no split-specific files found
    if not found_files and use_fallback:
        patterns = get_fallback_patterns()
        for pattern in patterns:
            if not recursive and '**' in pattern:
                continue
            for f in data_path.glob(pattern):
                if f.is_file() and f.suffix in extensions:
                    found_files.append(f)

        # Deduplicate
        found_files = list(dict.fromkeys(found_files))

    # Sort for deterministic ordering
    found_files = sorted(found_files, key=lambda p: p.name)

    # Apply max_files limit
    if max_files is not None and len(found_files) > max_files:
        found_files = found_files[:max_files]

    if found_files:
        logger.debug(f"Discovered {len(found_files)} files in {data_path}")
    else:
        logger.warning(f"No data files found in {data_path} (split={split})")

    return found_files


def get_file_format(file_path: Union[str, Path]) -> str:
    """
    Determine the format of a data file based on extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'arrow', 'parquet', 'jsonl', or 'unknown'
    """
    suffix = Path(file_path).suffix.lower()
    format_map = {
        '.arrow': 'arrow',
        '.parquet': 'parquet',
        '.jsonl': 'jsonl',
        '.json': 'jsonl',
    }
    return format_map.get(suffix, 'unknown')


def estimate_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    path = Path(file_path)
    if path.exists():
        return path.stat().st_size
    return 0


def estimate_total_samples(
    files: List[Path],
    samples_per_mb: float = 100.0,
) -> int:
    """
    Estimate total number of samples across files.

    This is a rough estimate based on file sizes. Actual count
    requires reading file metadata.

    Args:
        files: List of file paths
        samples_per_mb: Estimated samples per megabyte (varies by format)

    Returns:
        Estimated total sample count
    """
    total_bytes = sum(estimate_file_size(f) for f in files)
    total_mb = total_bytes / (1024 * 1024)
    return int(total_mb * samples_per_mb)


# ============================================================================
# Base Dataset Classes
# ============================================================================

class DatasetFileMixin:
    """
    Mixin providing file discovery and management for datasets.

    Inherit from this to get standardized file handling:

        class MyDataset(DatasetFileMixin, IterableDataset):
            def __init__(self, data_dir, split='train'):
                self.data_dir = Path(data_dir)
                self.split = split
                self._discover_files()
    """

    data_dir: Path
    split: Optional[str]
    files: List[Path]

    def _discover_files(
        self,
        extensions: Optional[Sequence[str]] = None,
        max_files: Optional[int] = None,
    ) -> None:
        """
        Discover data files and store in self.files.

        Args:
            extensions: File extensions to look for
            max_files: Maximum number of files
        """
        split = getattr(self, 'split', None)
        self.files = discover_data_files(
            self.data_dir,
            split=split,
            extensions=extensions,
            max_files=max_files,
        )

        if not self.files:
            logger.warning(
                f"No data files found in {self.data_dir} "
                f"(split={split}, extensions={extensions})"
            )

    def get_file_count(self) -> int:
        """Get number of discovered files."""
        return len(getattr(self, 'files', []))

    def get_total_size_mb(self) -> float:
        """Get total size of all files in megabytes."""
        files = getattr(self, 'files', [])
        total_bytes = sum(estimate_file_size(f) for f in files)
        return total_bytes / (1024 * 1024)


class ShuffleBufferMixin:
    """
    Mixin providing shuffle buffer functionality for streaming datasets.

    Provides a buffer that accumulates samples and yields them in
    shuffled order for better training randomness.
    """

    buffer_size: int
    _shuffle_buffer: List

    def _init_shuffle_buffer(self, buffer_size: int = 10000) -> None:
        """Initialize the shuffle buffer."""
        self.buffer_size = buffer_size
        self._shuffle_buffer = []

    def _add_to_buffer(self, sample) -> Optional[any]:
        """
        Add sample to buffer, returning a random sample if buffer is full.

        Args:
            sample: Sample to add

        Returns:
            Random sample from buffer if full, else None
        """
        import random

        self._shuffle_buffer.append(sample)

        if len(self._shuffle_buffer) >= self.buffer_size:
            # Pop random sample
            idx = random.randint(0, len(self._shuffle_buffer) - 1)
            result = self._shuffle_buffer[idx]
            self._shuffle_buffer[idx] = self._shuffle_buffer[-1]
            self._shuffle_buffer.pop()
            return result

        return None

    def _flush_buffer(self) -> Iterator:
        """
        Flush remaining samples from buffer in shuffled order.

        Yields:
            Remaining samples in random order
        """
        import random
        random.shuffle(self._shuffle_buffer)
        yield from self._shuffle_buffer
        self._shuffle_buffer.clear()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # File discovery
    'discover_data_files',
    'get_split_patterns',
    'get_fallback_patterns',
    'get_file_format',
    'estimate_file_size',
    'estimate_total_samples',
    'SUPPORTED_EXTENSIONS',
    # Mixins
    'DatasetFileMixin',
    'ShuffleBufferMixin',
]
