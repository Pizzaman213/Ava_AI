"""Ava data processing modules"""

# Note: The following files have been moved to _archived/data/:
# - data_profiler.py (moved from parent directory)
# - deduplication.py
# - optimized_dataloader.py
# These are data preparation/analysis tools not used in the core training loop

from .arrow_reader import ArrowReader, arrow_reader
from .encoding_detector import EncodingDetector

__all__ = ["ArrowReader", "arrow_reader", "EncodingDetector"]
