#!/usr/bin/env python3
"""
Advanced Streaming Data Preparation Pipeline with Detailed Monitoring
- Processes data without loading everything into memory
- Provides comprehensive progress tracking and statistics
- Supports multiple data formats and quality validation
- Memory-efficient with detailed resource monitoring
- Enhanced with NVIDIA RAPIDS for GPU acceleration
"""
import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Iterator, List, Dict, Optional, Tuple, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk
import gc
from tqdm import tqdm
import psutil
import hashlib
from datetime import datetime, timedelta
import pickle
import traceback
from collections import defaultdict
import numpy as np
import pandas as pd
import shutil
import threading
from queue import Queue

# NVIDIA RAPIDS imports
try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import StandardScaler
    from cuml.feature_extraction.text import TfidfVectorizer
    RAPIDS_AVAILABLE = True
    logging.info("NVIDIA RAPIDS detected - GPU acceleration enabled")
except ImportError:
    RAPIDS_AVAILABLE = False
    logging.warning("NVIDIA RAPIDS not available - falling back to CPU processing")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(level="INFO", log_file=None):
    """Setup enhanced logging configuration with file and console output"""
    handlers = [logging.StreamHandler()]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
        )
        handlers.append(file_handler)  # type: ignore
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Also setup a separate statistics logger
    stats_logger = logging.getLogger('stats')
    # Save stats log to the same directory as main log
    stats_log_path = Path("/project/code/outputs/logs") / 'processing_stats.log'
    stats_handler = logging.FileHandler(stats_log_path)
    stats_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    stats_logger.addHandler(stats_handler)
    stats_logger.setLevel(logging.INFO)


def parse_args():
    """Parse enhanced command line arguments"""
    parser = argparse.ArgumentParser(
        description="Advanced Streaming Data Preparation Pipeline with NVIDIA RAPIDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single file:
    %(prog)s --input-path data.jsonl --output-dir processed/
  
  Process with GPU acceleration:
    %(prog)s --input-path data/ --use-gpu --gpu-batch-size 10000
  
  Process with custom settings:
    %(prog)s --input-path data/ --tokenizer bert-base-uncased --max-length 512 --chunk-size 5000
  
  Resume interrupted processing:
    %(prog)s --input-path data/ --resume --checkpoint-dir checkpoints/
        """
    )
    
    # Required arguments
    parser.add_argument("--input-path", type=str, required=True,
                       help="Input file or directory path")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str,
                       default="/project/code/data/pretraining/processed",
                       help="Output directory for processed data")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Directory for saving checkpoints")
    
    # Tokenization settings
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="HuggingFace tokenizer name or path")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum text length to keep")
    
    # Processing configuration
    parser.add_argument("--chunk-size", type=int, default=5000,
                       help="Number of examples to process at once")
    parser.add_argument("--max-memory-mb", type=int, default=1000,
                       help="Maximum memory usage in MB")
    parser.add_argument("--num-workers", type=int, default=min(mp.cpu_count(), 8),
                       help="Number of parallel workers")
    parser.add_argument("--fast-mode", action="store_true",
                       help="Enable fast mode with optimized settings")
    
    # GPU/RAPIDS settings
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use NVIDIA GPU acceleration with RAPIDS")
    parser.add_argument("--gpu-batch-size", type=int, default=5000,
                       help="Batch size for GPU processing")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.8,
                       help="Fraction of GPU memory to use")
    
    # Data splits
    parser.add_argument("--train-split", type=float, default=0.95,
                       help="Fraction of data for training")
    parser.add_argument("--val-split", type=float, default=0.025,
                       help="Fraction of data for validation")
    parser.add_argument("--test-split", type=float, default=0.025,
                       help="Fraction of data for testing")
    
    # Quality control
    parser.add_argument("--validate-data", action="store_true",
                       help="Perform data quality validation")
    parser.add_argument("--remove-duplicates", action="store_true",
                       help="Remove duplicate examples")
    parser.add_argument("--quality-threshold", type=float, default=0.5,
                       help="Minimum quality score to keep example")
    
    # Resume and monitoring
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--monitor-interval", type=int, default=1000,
                       help="Examples between progress updates")
    parser.add_argument("--save-interval", type=int, default=10000,
                       help="Examples between checkpoint saves")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging verbosity level")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log file path")
    
    # Format detection
    parser.add_argument("--auto-detect-format", action="store_true", default=True,
                       help="Automatically detect input format")
    parser.add_argument("--format", type=str, choices=["jsonl", "json", "parquet", "csv", "text", "huggingface"],
                       help="Force specific input format")
    
    return parser.parse_args()


class HardwareOptimizer:
    """Automatically optimize data streaming to maintain optimal CPU/GPU utilization"""

    def __init__(self, target_cpu_percent=80, target_memory_percent=60, use_gpu=False):
        self.target_cpu = target_cpu_percent
        self.target_memory = target_memory_percent
        self.use_gpu = use_gpu
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = [] if use_gpu and RAPIDS_AVAILABLE else None
        self.chunk_size = 500  # Start small
        self.num_workers = min(mp.cpu_count() // 2, 4)  # Conservative workers
        self.monitoring = True
        self.monitor_thread = None
        self.adjustments_made = 0
        self.max_chunk_size = 5000  # Limit maximum chunk size

    def start_monitoring(self):
        """Start background monitoring of system resources"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                mem_percent = psutil.virtual_memory().percent

                self.cpu_history.append(cpu_percent)
                self.memory_history.append(mem_percent)

                # Monitor GPU if available
                if self.use_gpu and RAPIDS_AVAILABLE:
                    try:
                        if RAPIDS_AVAILABLE:
                            gpu_percent = cp.cuda.runtime.getDeviceProperties(0)['memory.used'] / \
                                         cp.cuda.runtime.getDeviceProperties(0)['memory.total'] * 100
                            self.gpu_history.append(gpu_percent)
                    except:
                        self.gpu_history.append(0)

                # Keep only last 10 measurements
                if len(self.cpu_history) > 10:
                    self.cpu_history.pop(0)
                if len(self.memory_history) > 10:
                    self.memory_history.pop(0)
                if self.gpu_history and len(self.gpu_history) > 10:
                    self.gpu_history.pop(0)

                # Auto-adjust if needed
                if len(self.cpu_history) >= 3:
                    self._auto_adjust()

                time.sleep(2)
            except:
                pass

    def _auto_adjust(self):
        """Automatically adjust parameters based on system load"""
        avg_cpu = np.mean(self.cpu_history[-3:])
        avg_memory = np.mean(self.memory_history[-3:])
        avg_gpu = np.mean(self.gpu_history[-3:]) if self.gpu_history else 0

        # Prioritize memory safety
        if avg_memory > self.target_memory:
            # Memory pressure - reduce chunk size immediately
            old_chunk = self.chunk_size
            self.chunk_size = max(int(self.chunk_size * 0.5), 100)
            if self.chunk_size != old_chunk:
                logging.info(f"💾 Memory at {avg_memory:.1f}%, reducing chunk size to {self.chunk_size}")
                self.adjustments_made += 1
                return  # Skip CPU adjustments when memory is high

        # GPU memory pressure handling
        if self.use_gpu and RAPIDS_AVAILABLE and avg_gpu > 85:
            old_chunk = self.chunk_size
            self.chunk_size = max(int(self.chunk_size * 0.7), 100)
            if self.chunk_size != old_chunk:
                logging.info(f"🎮 GPU memory at {avg_gpu:.1f}%, reducing chunk size to {self.chunk_size}")
                self.adjustments_made += 1
                return

        # Adjust chunk size based on CPU usage only if memory is safe
        if avg_memory < self.target_memory - 10:
            if avg_cpu < self.target_cpu - 20:
                # CPU underutilized, increase chunk size carefully
                old_chunk = self.chunk_size
                self.chunk_size = min(int(self.chunk_size * 1.2), self.max_chunk_size)
                if self.chunk_size != old_chunk:
                    logging.info(f"⚡ CPU at {avg_cpu:.1f}%, increasing chunk size to {self.chunk_size}")
                    self.adjustments_made += 1

            elif avg_cpu > self.target_cpu + 10:
                # CPU overutilized, decrease chunk size
                old_chunk = self.chunk_size
                self.chunk_size = max(int(self.chunk_size * 0.8), 100)
                if self.chunk_size != old_chunk:
                    logging.info(f"⚠️ CPU at {avg_cpu:.1f}%, decreasing chunk size to {self.chunk_size}")
                    self.adjustments_made += 1

        # Adjust workers based on memory
        if avg_memory > self.target_memory + 15:
            # Memory pressure, reduce workers
            old_workers = self.num_workers
            self.num_workers = max(1, self.num_workers - 1)
            if self.num_workers != old_workers:
                logging.info(f"💾 Memory at {avg_memory:.1f}%, reducing workers to {self.num_workers}")
                self.adjustments_made += 1

        elif avg_memory < self.target_memory - 20 and avg_cpu < self.target_cpu - 10:
            # Both CPU and memory have room, increase workers
            old_workers = self.num_workers
            self.num_workers = min(mp.cpu_count(), self.num_workers + 1)
            if self.num_workers != old_workers:
                logging.info(f"🚀 Resources available, increasing workers to {self.num_workers}")
                self.adjustments_made += 1

    def get_optimal_params(self):
        """Get current optimal parameters"""
        params = {
            'chunk_size': self.chunk_size,
            'num_workers': self.num_workers,
            'cpu_usage': np.mean(self.cpu_history) if self.cpu_history else 0,
            'memory_usage': np.mean(self.memory_history) if self.memory_history else 0,
            'adjustments': self.adjustments_made
        }
        if self.gpu_history:
            params['gpu_usage'] = np.mean(self.gpu_history) if self.gpu_history else 0
        return params

    def get_status(self):
        """Get current optimization status"""
        if not self.cpu_history:
            return "Initializing..."

        avg_cpu = np.mean(self.cpu_history)
        avg_mem = np.mean(self.memory_history)
        avg_gpu = np.mean(self.gpu_history) if self.gpu_history else 0

        if self.use_gpu and RAPIDS_AVAILABLE:
            return f"💻 CPU: {avg_cpu:.0f}%, 🎮 GPU: {avg_gpu:.0f}%, 💾 Mem: {avg_mem:.0f}%"
        elif avg_cpu < 40:
            return f"⚡ Underutilized (CPU: {avg_cpu:.0f}%)"
        elif avg_cpu > 90:
            return f"🔥 Overloaded (CPU: {avg_cpu:.0f}%)"
        else:
            return f"✅ Optimal (CPU: {avg_cpu:.0f}%, Mem: {avg_mem:.0f}%)"


class DataStatistics:
    """Track detailed statistics during processing"""
    def __init__(self):
        self.total_examples = 0
        self.valid_examples = 0
        self.invalid_examples = 0
        self.duplicates_removed = 0
        self.total_tokens = 0
        self.total_bytes = 0
        self.length_distribution = defaultdict(int)
        self.quality_scores = []
        self.processing_times = []
        self.memory_snapshots = []
        self.gpu_memory_snapshots = [] if RAPIDS_AVAILABLE else None
        self.errors = defaultdict(int)
        self.start_time = time.time()
        self.file_stats = defaultdict(lambda: {
            'examples': 0, 'tokens': 0, 'errors': 0, 'time': 0
        })
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (int, float)):
                    setattr(self, key, getattr(self, key) + value)
                elif isinstance(getattr(self, key), list):
                    getattr(self, key).append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        summary = {
            'total_examples': self.total_examples,
            'valid_examples': self.valid_examples,
            'invalid_examples': self.invalid_examples,
            'duplicates_removed': self.duplicates_removed,
            'total_tokens': self.total_tokens,
            'total_gb': self.total_bytes / 1e9,
            'avg_tokens_per_example': self.total_tokens / max(1, self.valid_examples),
            'processing_time': elapsed,
            'throughput': self.total_examples / max(1, elapsed),
            'avg_quality': np.mean(self.quality_scores) if self.quality_scores else 0,
            'memory_peak_mb': max(self.memory_snapshots) if self.memory_snapshots else 0,
            'errors': dict(self.errors),
            'file_stats': dict(self.file_stats)
        }
        if self.gpu_memory_snapshots and RAPIDS_AVAILABLE:
            summary['gpu_memory_peak_mb'] = max(self.gpu_memory_snapshots) if self.gpu_memory_snapshots else 0
        return summary


def monitor_memory() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    mem_info = process.memory_info()
    stats = {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
        'percent': process.memory_percent(),
        'available_gb': psutil.virtual_memory().available / 1e9
    }
    
    # Add GPU memory info if available
    if RAPIDS_AVAILABLE:
        try:
            if RAPIDS_AVAILABLE:
                gpu_mem_used = cp.cuda.runtime.memGetInfo()[1] / 1024 / 1024  # MB
                gpu_mem_total = cp.cuda.runtime.getDeviceProperties(0)['memory.total'] / 1024 / 1024  # MB
                stats['gpu_memory_mb'] = gpu_mem_used
            else:
                stats['gpu_memory_mb'] = 0
            stats['gpu_memory_percent'] = (gpu_mem_used / gpu_mem_total) * 100
        except:
            stats['gpu_memory_mb'] = 0
            stats['gpu_memory_percent'] = 0
    
    return stats


def calculate_text_quality(text: str) -> float:
    """Calculate quality score for text (0-1)"""
    if not text or len(text) < 10:
        return 0.0
    
    score = 1.0
    
    # Length penalty
    if len(text) < 50:
        score *= 0.5
    elif len(text) > 10000:
        score *= 0.8
    
    # Check for repetition
    words = text.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        score *= min(1.0, unique_ratio * 2)  # Penalize high repetition
    
    # Check for special characters
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if special_char_ratio > 0.3:
        score *= 0.7
    
    # Check for proper sentence structure (basic)
    if not any(text.strip().endswith(p) for p in ['.', '!', '?']):
        score *= 0.9
    
    return min(1.0, max(0.0, score))


def stream_jsonl_file(filepath: Path, chunk_size: int = 1000, 
                     min_length: int = 10, validate: bool = False,
                     stats: Optional[DataStatistics] = None, use_gpu: bool = False) -> Iterator[List[Tuple[str, Dict]]]:
    """
    Enhanced JSONL streaming with validation and statistics
    Returns tuples of (text, metadata)
    """
    chunk = []
    line_count = 0
    error_count = 0
    seen_hashes = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        with tqdm(desc=f"Reading {filepath.name}", unit=" lines") as pbar:
            for line_num, line in enumerate(f, 1):
                pbar.update(1)
                line_count += 1
                
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract text from various formats with priority
                    text = None
                    for field in ['text', 'content', 'output', 'response', 'document', 'passage']:
                        if field in data:
                            text = str(data[field])
                            break
                    
                    # Handle conversation format
                    if not text and 'messages' in data:
                        messages = data['messages']
                        if isinstance(messages, list):
                            text = "\n".join([msg.get('content', '') for msg in messages if 'content' in msg])
                    
                    # Handle instruction format
                    if not text and 'instruction' in data:
                        text = f"Instruction: {data['instruction']}\n"
                        if 'input' in data and data['input']:
                            text += f"Input: {data['input']}\n"
                        if 'output' in data:
                            text += f"Output: {data['output']}"
                    
                    if text and len(text) >= min_length:
                        # Duplicate detection
                        if validate:
                            text_hash = hashlib.md5(text.encode()).hexdigest()
                            if text_hash in seen_hashes:
                                if stats:
                                    stats.duplicates_removed += 1
                                continue
                            seen_hashes.add(text_hash)
                        
                        # Quality check
                        quality = calculate_text_quality(text) if validate else 1.0
                        
                        metadata = {
                            'line_num': line_num,
                            'quality': quality,
                            'length': len(text),
                            'source': filepath.name
                        }
                        
                        chunk.append((text, metadata))
                        
                        if stats:
                            stats.total_bytes += len(text.encode('utf-8'))
                        
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                            
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    error_count += 1
                    if stats:
                        stats.errors[type(e).__name__] += 1
                    
                    if error_count <= 10:  # Log first 10 errors
                        logging.debug(f"Error on line {line_num}: {e}")
                
                # Periodic statistics update
                if line_num % 10000 == 0 and stats:
                    mem = monitor_memory()
                    stats.memory_snapshots.append(mem['rss_mb'])
                    if 'gpu_memory_mb' in mem and stats.gpu_memory_snapshots is not None:
                        stats.gpu_memory_snapshots.append(mem['gpu_memory_mb'])
                    logging.info(f"  Line {line_num:,}: Memory {mem['rss_mb']:.1f}MB, "
                               f"GPU {mem.get('gpu_memory_mb', 0):.1f}MB, "
                               f"Chunks yielded: {line_num // chunk_size}")
    
    # Yield remaining chunk
    if chunk:
        yield chunk
    
    if stats:
        stats.file_stats[filepath.name]['examples'] = line_count
        stats.file_stats[filepath.name]['errors'] = error_count


def stream_parquet_file(filepath: Path, chunk_size: int = 1000, use_gpu: bool = False) -> Iterator[List[Dict]]:
    """
    Stream Parquet file in chunks using PyArrow or cuDF for GPU acceleration
    """
    if use_gpu and RAPIDS_AVAILABLE:
        # Use cuDF for GPU acceleration
        try:
            gdf = cudf.read_parquet(filepath)
            
            # Try common text columns
            text_col = None
            for col in ['text', 'content', 'output', 'response']:
                if col in gdf.columns:
                    text_col = col
                    break
            
            if text_col:
                # Convert to pandas for easier handling
                df = gdf.to_pandas()
                valid_texts = df[text_col].dropna().astype(str)
                texts = [t for t in valid_texts if len(t) > 10]
                
                # Yield in chunks
                for i in range(0, len(texts), chunk_size):
                    yield texts[i:i + chunk_size]
            else:
                logging.warning(f"No text column found in {filepath}")
        except Exception as e:
            logging.error(f"cuDF failed to read {filepath}: {e}")
            # Fallback to PyArrow
            parquet_file = pq.ParquetFile(filepath)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                texts = []
                df = batch.to_pandas()
                for col in ['text', 'content', 'output', 'response']:
                    if col in df.columns:
                        valid_texts = df[col].dropna().astype(str)
                        texts.extend([t for t in valid_texts if len(t) > 10])
                        break
                if texts:
                    yield texts
    else:
        # CPU-only processing with PyArrow
        parquet_file = pq.ParquetFile(filepath)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            texts = []
            df = batch.to_pandas()
            for col in ['text', 'content', 'output', 'response']:
                if col in df.columns:
                    valid_texts = df[col].dropna().astype(str)
                    texts.extend([t for t in valid_texts if len(t) > 10])
                    break
            if texts:
                yield texts


def process_with_rapids(texts: List[str], tokenizer, max_length: int, use_gpu: bool = False) -> Dict:
    """
    Process texts using NVIDIA RAPIDS for GPU acceleration
    """
    if not use_gpu or not RAPIDS_AVAILABLE:
        # Fallback to CPU processing
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='np'
        )
    
    try:
        # Convert texts to cuDF Series for GPU processing
        gdf = cudf.Series(texts)
        
        # Simple preprocessing on GPU
        # Note: For full tokenization on GPU, you would typically use libraries like cuML or custom CUDA kernels
        # This is a simplified example showing how to leverage GPU for preprocessing steps
        
        # Convert back to list for tokenizer (tokenizer still runs on CPU)
        # In practice, you might want to use a GPU-accelerated tokenizer like FasterTokenizer
        processed_texts = gdf.to_pandas().tolist()
        
        # Tokenize on CPU (most tokenizers don't have native GPU support yet)
        tokens = tokenizer(
            processed_texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='np'
        )
        
        return tokens
    except Exception as e:
        logging.warning(f"RAPIDS processing failed, falling back to CPU: {e}")
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='np'
        )


def process_file_streaming(filepath: Path, tokenizer, args: argparse.Namespace,
                          output_dir: Path, stats: DataStatistics,
                          chunk_size: int = 1000):
    """
    Process a single file in streaming mode with detailed monitoring
    """
    logger = logging.getLogger(__name__)
    stats_logger = logging.getLogger('stats')
    
    file_start_time = time.time()
    
    # Auto-detect format if needed
    if args.auto_detect_format and not args.format:
        # Check if it's a HuggingFace dataset directory
        if filepath.is_dir() and (filepath / "dataset_info.json").exists():
            format_type = 'huggingface'
        # Check for Arrow files (HuggingFace dataset format)
        elif filepath.suffix == '.arrow':
            # Check if parent directory has dataset_info.json
            parent = filepath.parent
            if (parent / "dataset_info.json").exists():
                format_type = 'huggingface'
            else:
                format_type = 'arrow'
        elif filepath.suffix == '.jsonl':
            format_type = 'jsonl'
        elif filepath.suffix == '.json':
            format_type = 'json'
        elif filepath.suffix == '.parquet':
            format_type = 'parquet'
        elif filepath.suffix in ['.txt', '.text']:
            format_type = 'text'
        else:
            # Try to detect from content
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{'):
                        format_type = 'jsonl' if filepath.stat().st_size > 1000000 else 'json'
                    else:
                        format_type = 'text'
            except:
                format_type = 'text'
    else:
        format_type = args.format or 'jsonl'
    
    logger.info(f"Detected format: {format_type}")
    logger.info(f"Using GPU acceleration: {args.use_gpu and RAPIDS_AVAILABLE}")
    
    # Get appropriate streamer
    if format_type == 'jsonl':
        streamer = stream_jsonl_file(filepath, args.chunk_size, args.min_length,
                                    args.validate_data, stats, args.use_gpu)
    elif format_type == 'json':
        streamer = stream_json_file(filepath, args.chunk_size)
    elif format_type == 'parquet':
        streamer = stream_parquet_file(filepath, args.chunk_size, args.use_gpu)
    elif format_type == 'huggingface':
        # For HuggingFace datasets, we'll handle them inline
        streamer = None  # Will be handled specially below
    else:
        logger.warning(f"Unsupported format: {format_type}")
        return 0
    
    # Setup output files with proper splits - using Arrow format
    train_dir = output_dir / f"train_{filepath.stem}"
    val_dir = output_dir / f"val_{filepath.stem}"
    test_dir = output_dir / f"test_{filepath.stem}" if args.test_split > 0 else None

    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    if test_dir:
        test_dir.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    total_examples = 0
    train_examples = 0
    val_examples = 0
    test_examples = 0
    quality_filtered = 0
    token_count = 0

    # File size and estimated time
    if filepath.is_file():
        file_size_gb = filepath.stat().st_size / 1e9
    else:
        file_size_gb = 0  # Directory size already calculated
    logger.info(f"Processing {filepath.name}")
    logger.info(f"  Size: {file_size_gb:.2f} GB")
    logger.info(f"  Format: {format_type}")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Quality validation: {args.validate_data}")
    logger.info(f"  GPU acceleration: {args.use_gpu and RAPIDS_AVAILABLE}")

    # Initialize Arrow writers
    train_batch = []
    val_batch = []
    test_batch = []
    train_file_idx = 0
    val_file_idx = 0
    test_file_idx = 0

    # Define Arrow schema
    pa_schema = pa.schema([
        ('input_ids', pa.list_(pa.int32())),
        ('attention_mask', pa.list_(pa.int32())),
        ('text', pa.string()),
    ])

    def save_arrow_batch(batch_data, output_dir, file_idx, split_name):
        """Save a batch of data to Arrow format"""
        if not batch_data:
            return file_idx

        # Convert to Arrow table
        table = pa.Table.from_pydict({
            'input_ids': [d['input_ids'] for d in batch_data],
            'attention_mask': [d['attention_mask'] for d in batch_data],
            'text': [d.get('text', '') for d in batch_data],
        })

        # Save as Arrow file
        arrow_file = output_dir / f"data-{file_idx:05d}-of-00100.arrow"
        with pa.OSFile(str(arrow_file), 'wb') as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        return file_idx + 1

    # Process chunks
    with tqdm(desc="Processing", unit=" chunks") as pbar:
        
        chunk_count = 0
        last_checkpoint = 0
        last_monitor = 0
        
        try:
            for chunk_data in streamer:
                chunk_count += 1
                chunk_start = time.time()

                # Check if data is already preprocessed
                if chunk_data and isinstance(chunk_data[0], tuple) and chunk_data[0][0] == 'PREPROCESSED':
                    # Data is already tokenized, just save it
                    print(f"  Using preprocessed data, skipping tokenization...", flush=True)

                    for _, example in chunk_data:
                        # Example already has input_ids and attention_mask
                        processed_example = {
                            'input_ids': example['input_ids'],
                            'attention_mask': example['attention_mask'],
                            'text': example.get('text', '')
                        }

                        # Count tokens
                        actual_tokens = sum(processed_example['attention_mask'])
                        token_count += actual_tokens
                        stats.total_tokens += actual_tokens

                        # Determine split and add to appropriate batch
                        random_val = np.random.random()
                        if random_val < args.train_split:
                            train_batch.append(processed_example)
                            train_examples += 1

                            if len(train_batch) >= min(args.chunk_size, 2000):
                                train_file_idx = save_arrow_batch(train_batch, train_dir, train_file_idx, 'train')
                                train_batch = []
                                gc.collect()

                        elif random_val < args.train_split + args.val_split:
                            val_batch.append(processed_example)
                            val_examples += 1

                            if len(val_batch) >= min(args.chunk_size, 2000):
                                val_file_idx = save_arrow_batch(val_batch, val_dir, val_file_idx, 'val')
                                val_batch = []
                                gc.collect()

                        elif test_dir:
                            test_batch.append(processed_example)
                            test_examples += 1

                            if len(test_batch) >= min(args.chunk_size, 2000):
                                test_file_idx = save_arrow_batch(test_batch, test_dir, test_file_idx, 'test')
                                test_batch = []
                                gc.collect()

                        total_examples += 1
                        stats.total_examples += 1

                    continue  # Skip normal processing

                # Extract texts and metadata for non-preprocessed data
                if isinstance(chunk_data[0], tuple):
                    texts = [item[0] for item in chunk_data]
                    metadata_list = [item[1] for item in chunk_data]
                else:
                    texts = chunk_data
                    metadata_list = [{}] * len(texts)
                
                # Quality filtering
                if args.validate_data:
                    filtered_data = []
                    for text, metadata in zip(texts, metadata_list):
                        if metadata.get('quality', 1.0) >= args.quality_threshold:
                            filtered_data.append((text, metadata))
                        else:
                            quality_filtered += 1
                            stats.invalid_examples += 1
                    
                    if filtered_data:
                        texts = [item[0] for item in filtered_data]
                        metadata_list = [item[1] for item in filtered_data]
                    else:
                        continue
                
                # Tokenize chunk with GPU acceleration if enabled
                try:
                    tokens = process_with_rapids(
                        texts,
                        tokenizer,
                        args.max_length,
                        args.use_gpu
                    )
                except Exception as e:
                    logger.error(f"Tokenization error: {e}")
                    stats.errors['TokenizationError'] += 1
                    continue
                
                # Process each example in chunk
                for i in range(len(texts)):
                    # Create example with text preserved
                    example = {
                        'input_ids': tokens['input_ids'][i].tolist(),
                        'attention_mask': tokens['attention_mask'][i].tolist(),
                        'text': texts[i] if i < len(texts) else ''
                    }

                    # Count tokens
                    actual_tokens = sum(example['attention_mask'])
                    token_count += actual_tokens
                    stats.total_tokens += actual_tokens

                    # Determine split and add to appropriate batch
                    random_val = np.random.random()
                    if random_val < args.train_split:
                        train_batch.append(example)
                        train_examples += 1

                        # Save batch more frequently to avoid memory buildup
                        if len(train_batch) >= min(args.chunk_size, 2000):
                            train_file_idx = save_arrow_batch(train_batch, train_dir, train_file_idx, 'train')
                            train_batch = []
                            gc.collect()  # Force garbage collection

                    elif random_val < args.train_split + args.val_split:
                        val_batch.append(example)
                        val_examples += 1

                        # Save batch more frequently to avoid memory buildup
                        if len(val_batch) >= min(args.chunk_size, 2000):
                            val_file_idx = save_arrow_batch(val_batch, val_dir, val_file_idx, 'val')
                            val_batch = []
                            gc.collect()  # Force garbage collection

                    elif test_dir:
                        test_batch.append(example)
                        test_examples += 1

                        # Save batch more frequently to avoid memory buildup
                        if len(test_batch) >= min(args.chunk_size, 2000):
                            test_file_idx = save_arrow_batch(test_batch, test_dir, test_file_idx, 'test')
                            test_batch = []
                            gc.collect()  # Force garbage collection
                    
                    total_examples += 1
                    stats.total_examples += 1
                    stats.valid_examples += 1
                    
                    # Update length distribution
                    length_bucket = (actual_tokens // 100) * 100
                    stats.length_distribution[length_bucket] += 1
                
                # Chunk processing time
                chunk_time = time.time() - chunk_start
                stats.processing_times.append(chunk_time)
                
                # Progress monitoring
                if total_examples - last_monitor >= args.monitor_interval:
                    mem = monitor_memory()
                    stats.memory_snapshots.append(mem['rss_mb'])
                    if 'gpu_memory_mb' in mem and stats.gpu_memory_snapshots is not None:
                        stats.gpu_memory_snapshots.append(mem['gpu_memory_mb'])
                    
                    elapsed = time.time() - file_start_time
                    rate = total_examples / elapsed if elapsed > 0 else 0
                    eta = (file_size_gb * 1e9 / filepath.stat().st_size * total_examples - total_examples) / rate if rate > 0 else 0
                    
                    logger.info(f"  Progress: {total_examples:,} examples")
                    logger.info(f"    Chunks: {chunk_count}, Tokens: {token_count:,}")
                    logger.info(f"    Train: {train_examples:,}, Val: {val_examples:,}, Test: {test_examples:,}")
                    logger.info(f"    Quality filtered: {quality_filtered:,}")
                    logger.info(f"    Rate: {rate:.0f} ex/s, ETA: {timedelta(seconds=int(eta))}")
                    logger.info(f"    Memory: {mem['rss_mb']:.1f}MB (available: {mem['available_gb']:.1f}GB)")
                    if 'gpu_memory_mb' in mem:
                        logger.info(f"    GPU Memory: {mem['gpu_memory_mb']:.1f}MB ({mem['gpu_memory_percent']:.1f}%)")
                    
                    stats_logger.info(json.dumps({
                        'file': filepath.name,
                        'examples': total_examples,
                        'tokens': token_count,
                        'memory_mb': mem['rss_mb'],
                        'gpu_memory_mb': mem.get('gpu_memory_mb', 0),
                        'rate': rate
                    }))
                    
                    last_monitor = total_examples
                
                # Checkpoint saving
                if args.checkpoint_dir and total_examples - last_checkpoint >= args.save_interval:
                    checkpoint_state = {
                        'filepath': str(filepath),
                        'total_examples': total_examples,
                        'train_examples': train_examples,
                        'val_examples': val_examples,
                        'test_examples': test_examples,
                        'chunk_count': chunk_count,
                        'stats': stats.get_summary()
                    }
                    save_checkpoint(Path(args.checkpoint_dir), checkpoint_state)
                    last_checkpoint = total_examples
                
                # Memory management
                if chunk_count % 50 == 0:
                    gc.collect()
                    
                    # Check memory limit
                    mem = monitor_memory()
                    if mem['rss_mb'] > args.max_memory_mb:
                        logger.warning(f"Memory limit exceeded ({mem['rss_mb']:.1f}MB > {args.max_memory_mb}MB)")
                        logger.info("Forcing garbage collection...")
                        gc.collect()
                        time.sleep(1)  # Give system time to free memory
        
        finally:
            # Save any remaining batches
            if train_batch:
                save_arrow_batch(train_batch, train_dir, train_file_idx, 'train')
            if val_batch:
                save_arrow_batch(val_batch, val_dir, val_file_idx, 'val')
            if test_batch and test_dir:
                save_arrow_batch(test_batch, test_dir, test_file_idx, 'test')

            # Save dataset info for HuggingFace compatibility
            for dir_path, split_name, example_count in [(train_dir, 'train', train_examples),
                                                         (val_dir, 'validation', val_examples),
                                                         (test_dir, 'test', test_examples)]:
                if dir_path and example_count > 0:
                    dataset_info = {
                        'dataset_name': filepath.stem,
                        'split': split_name,
                        'num_examples': example_count,
                        'features': {
                            'input_ids': {'dtype': 'int32', 'shape': [args.max_length]},
                            'attention_mask': {'dtype': 'int32', 'shape': [args.max_length]},
                            'text': {'dtype': 'string'}
                        }
                    }
                    with open(dir_path / 'dataset_info.json', 'w') as f:
                        json.dump(dataset_info, f, indent=2)
    
    # Final statistics
    file_time = time.time() - file_start_time
    stats.file_stats[filepath.name]['time'] = file_time
    stats.file_stats[filepath.name]['tokens'] = token_count
    
    logger.info(f"✅ Completed {filepath.name}")
    logger.info(f"   Total: {total_examples:,} examples in {timedelta(seconds=int(file_time))}")
    logger.info(f"   Train: {train_examples:,} ({train_examples/max(1,total_examples)*100:.1f}%)")
    logger.info(f"   Val: {val_examples:,} ({val_examples/max(1,total_examples)*100:.1f}%)")
    if test_examples > 0:
        logger.info(f"   Test: {test_examples:,} ({test_examples/max(1,total_examples)*100:.1f}%)")
    logger.info(f"   Tokens: {token_count:,} (avg: {token_count/max(1,total_examples):.1f})")
    logger.info(f"   Rate: {total_examples/max(1,file_time):.1f} examples/sec")
    if quality_filtered > 0:
        logger.info(f"   Quality filtered: {quality_filtered:,}")
    
    return total_examples


def main():
    args = parse_args()

    # Initialize hardware optimizer for auto-optimization
    hw_optimizer = HardwareOptimizer(
        target_cpu_percent=70, 
        target_memory_percent=60, 
        use_gpu=args.use_gpu
    )
    hw_optimizer.start_monitoring()

    # Apply fast mode optimizations
    if args.fast_mode:
        args.chunk_size = 2000  # Conservative for memory safety
        args.num_workers = min(mp.cpu_count() // 2, 4)  # Fewer workers to save memory
        args.validate_data = False
        args.remove_duplicates = False
        print("🚀 Fast mode enabled - using optimized settings")
        print(f"   Chunk size: {args.chunk_size}")
        print(f"   Workers: {args.num_workers}")
        print("   Validation: disabled")

    # Auto-optimize based on hardware
    print("🔧 Hardware auto-optimization enabled")
    print(f"   Target CPU utilization: 75%")
    print(f"   Target memory utilization: 70%")
    if args.use_gpu and RAPIDS_AVAILABLE:
        print("   GPU acceleration: ENABLED")
    time.sleep(2)  # Let optimizer gather initial stats

    # Setup logging - ensure logs directory exists
    log_dir = Path("/project/code/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or str(log_dir / f"prepare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(level=args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    stats_logger = logging.getLogger('stats')
    
    logger.info("\n" + "="*80)
    logger.info(" " * 15 + "ADVANCED DATA PREPARATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Version: 2.1 - Enhanced with NVIDIA RAPIDS")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    start_time = time.time()
    stats = DataStatistics()
    
    # Setup paths
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / ".checkpoints"
    
    # Load checkpoint if resuming
    checkpoint_state = None
    if args.resume:
        checkpoint_state = load_checkpoint(checkpoint_dir)
        if checkpoint_state:
            logger.info(f"Resuming from checkpoint with {checkpoint_state['total_examples']:,} examples processed")
    
    # Find files with multiple format support
    if input_path.is_dir():
        patterns = ["*.jsonl", "*.json", "*.parquet", "*.txt", "*.csv"]
        files = []
        for pattern in patterns:
            files.extend(sorted(input_path.glob(pattern)))

        # Also check for HuggingFace dataset directories
        for subdir in input_path.iterdir():
            if subdir.is_dir():
                # Check if main dataset directory
                if (subdir / "dataset_info.json").exists():
                    logger.info(f"Found HuggingFace dataset: {subdir.name}")
                    files.append(subdir)
                # Check for train/validation/test subdirectories
                else:
                    for split_name in ['train', 'train_sft', 'validation', 'test', 'test_sft']:
                        split_dir = subdir / split_name
                        if split_dir.exists() and (split_dir / "dataset_info.json").exists():
                            logger.info(f"Found HuggingFace dataset: {subdir.name}/{split_name}")
                            files.append(split_dir)
        
        logger.info(f"Found {len(files)} data files")
        if files:
            total_size_gb = 0
            for f in files:
                if f.is_dir():
                    # Calculate size of directory
                    total_size_gb += sum(file.stat().st_size for file in f.rglob('*') if file.is_file()) / 1e9
                else:
                    total_size_gb += f.stat().st_size / 1e9
            logger.info(f"Total size: {total_size_gb:.2f} GB")
            
            # Show file breakdown
            format_counts = defaultdict(int)
            for f in files:
                if f.is_dir() and (f / "dataset_info.json").exists():
                    format_counts['.huggingface'] += 1
                else:
                    format_counts[f.suffix] += 1
            logger.info("File formats: " + ", ".join([f"{ext}: {count}" for ext, count in format_counts.items()]))
    else:
        files = [input_path]
        total_size_gb = input_path.stat().st_size / 1e9
    
    # Initialize tokenizer with error handling
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer loaded: vocab size = {len(tokenizer)}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.info("Trying alternative tokenizer loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            logger.error("Failed to load tokenizer. Exiting.")
            return
    
    # Process each file in streaming mode
    logger.info(f"\nProcessing {len(files)} files")
    logger.info(f"Configuration:")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Quality validation: {args.validate_data}")
    logger.info(f"  Remove duplicates: {args.remove_duplicates}")
    logger.info(f"  GPU acceleration: {args.use_gpu and RAPIDS_AVAILABLE}")
    logger.info(f"  Train/Val/Test split: {args.train_split:.0%}/{args.val_split:.0%}/{args.test_split:.0%}")
    logger.info("="*80)
    
    total_examples = 0
    train_files = []
    val_files = []
    test_files = []
    
    # Skip already processed files if resuming
    skip_files = set()
    if checkpoint_state and 'processed_files' in checkpoint_state:
        skip_files = set(checkpoint_state['processed_files'])
        logger.info(f"Skipping {len(skip_files)} already processed files")
    
    for file_idx, filepath in enumerate(files, 1):
        if str(filepath) in skip_files:
            logger.info(f"\n[{file_idx}/{len(files)}] Skipping {filepath.name} (already processed)")
            continue

        # Auto-adjust parameters based on current system load
        if file_idx % 3 == 0:  # Check every 3 files
            optimal = hw_optimizer.get_optimal_params()
            if optimal['chunk_size'] != args.chunk_size:
                old_chunk = args.chunk_size
                args.chunk_size = optimal['chunk_size']
                logger.info(f"⚡ Auto-adjusted: chunk size {old_chunk} → {args.chunk_size}")

            # Show optimization status
            status = hw_optimizer.get_status()
            logger.info(f"💻 System: {status}")

        logger.info(f"\n[{file_idx}/{len(files)}] Processing {filepath.name}")
        logger.info("-" * 60)

        try:
            file_examples = process_file_streaming(
                filepath, tokenizer, args, output_dir, stats,
                args.chunk_size
            )
            
            total_examples += file_examples
            
            # Track output files
            train_files.append(output_dir / f"train_{filepath.stem}.jsonl")
            val_files.append(output_dir / f"val_{filepath.stem}.jsonl")
            if args.test_split > 0:
                test_files.append(output_dir / f"test_{filepath.stem}.jsonl")
            
            # Update checkpoint
            if args.checkpoint_dir:
                checkpoint_state = {
                    'processed_files': list(skip_files) + [str(f) for f in files[:file_idx]],
                    'total_examples': total_examples,
                    'stats': stats.get_summary()
                }
                save_checkpoint(checkpoint_dir, checkpoint_state)
            
        except Exception as e:
            logger.error(f"Failed to process {filepath.name}: {e}")
            logger.debug(traceback.format_exc())
            stats.errors['FileProcessingError'] += 1
        
        finally:
            # Clear memory after each file
            gc.collect()
            
            # Show memory status
            mem = monitor_memory()
            logger.info(f"Memory after file: {mem['rss_mb']:.1f}MB")
            if 'gpu_memory_mb' in mem:
                logger.info(f"GPU Memory: {mem['gpu_memory_mb']:.1f}MB ({mem['gpu_memory_percent']:.1f}%)")
    
    # Combine all temporary Arrow directories
    logger.info("\n" + "="*80)
    logger.info("FINALIZING ARROW DATASETS")
    logger.info("="*80)

    # The Arrow files are already in their directories, just need to organize them
    logger.info("Arrow datasets created:")
    for split_type in ['train', 'val', 'test']:
        split_dirs = [p for p in output_dir.glob(f"{split_type}_*") if p.is_dir()]
        if split_dirs:
            total_examples = 0
            total_files = 0
            for split_dir in split_dirs:
                arrow_files = list(split_dir.glob("*.arrow"))
                if arrow_files:
                    total_files += len(arrow_files)
                    # Try to get example count from dataset info
                    info_file = split_dir / "dataset_info.json"
                    if info_file.exists():
                        with open(info_file) as f:
                            info = json.load(f)
                            total_examples += info.get('num_examples', 0)

            if total_files > 0:
                logger.info(f"  {split_type}: {total_examples:,} examples in {total_files} Arrow files")
    
    # Convert to HuggingFace dataset format
    logger.info("\n" + "="*80)
    logger.info("CONVERTING TO DATASET FORMAT")
    logger.info("="*80)
    
    def save_dataset_split(jsonl_file: Path, output_path: Path, split_name: str, max_examples: Optional[int] = None):
        """Save a dataset split in HuggingFace format"""
        if not jsonl_file.exists():
            logger.warning(f"{split_name} file not found: {jsonl_file}")
            return 0
        
        data = []
        with open(jsonl_file, 'r') as f:
            for idx, line in enumerate(tqdm(f, desc=f"Loading {split_name}")):
                data.append(json.loads(line))
                
                # Save in batches to manage memory
                if len(data) >= 50000 or (max_examples and idx >= max_examples - 1):
                    # Create dataset from current batch
                    batch_dataset = Dataset.from_list(data)
                    
                    if output_path.exists():
                        # Load existing, concatenate, and save to temp location first
                        temp_path = output_path.parent / f"{output_path.name}_temp"
                        existing = load_from_disk(str(output_path))
                        
                        # Concatenate datasets
                        from datasets import concatenate_datasets
                        final = concatenate_datasets([existing, batch_dataset])
                        
                        # Save to temp location
                        final.save_to_disk(str(temp_path), num_proc=args.num_workers)
                        
                        # Remove old and rename temp
                        import shutil
                        shutil.rmtree(str(output_path))
                        temp_path.rename(output_path)
                    else:
                        batch_dataset.save_to_disk(
                            str(output_path), 
                            num_proc=args.num_workers
                        )
                    
                    logger.info(f"  Saved {len(data)} examples to {output_path}")
                    data = []
                    
                    if max_examples and idx >= max_examples - 1:
                        break
        
        # Save remaining data
        if data:
            batch_dataset = Dataset.from_list(data)
            
            if output_path.exists():
                # Load existing, concatenate, and save to temp location first
                temp_path = output_path.parent / f"{output_path.name}_temp"
                existing = load_from_disk(str(output_path))
                
                from datasets import concatenate_datasets
                final = concatenate_datasets([existing, batch_dataset])
                
                # Save to temp location
                final.save_to_disk(str(temp_path), num_proc=args.num_workers)
                
                # Remove old and rename temp
                import shutil
                shutil.rmtree(str(output_path))
                temp_path.rename(output_path)
            else:
                batch_dataset.save_to_disk(
                    str(output_path),
                    num_proc=args.num_workers
                )
            logger.info(f"  Saved final {len(data)} examples to {output_path}")
        
        # Get total count
        if output_path.exists():
            dataset = load_from_disk(str(output_path))
            return len(dataset)
        return 0
    
    # Count examples in each split (datasets already saved as Arrow format)
    train_count = 0
    val_count = 0
    test_count = 0

    train_path = output_dir / "train"
    val_path = output_dir / "validation"
    test_path = output_dir / "test"

    if train_path.exists():
        try:
            train_ds = load_from_disk(str(train_path))
            train_count = len(train_ds)
        except Exception:
            pass

    if val_path.exists():
        try:
            val_ds = load_from_disk(str(val_path))
            val_count = len(val_ds)
        except Exception:
            pass

    if test_path.exists():
        try:
            test_ds = load_from_disk(str(test_path))
            test_count = len(test_ds)
        except Exception:
            pass

    logger.info(f"\nDataset splits saved:")
    logger.info(f"  Train: {train_count:,} examples")
    logger.info(f"  Validation: {val_count:,} examples")
    if test_count > 0:
        logger.info(f"  Test: {test_count:,} examples")
    
    # Save comprehensive configuration and statistics
    elapsed = time.time() - start_time
    summary = stats.get_summary()
    
    config = {
        'pipeline_version': '2.1',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'tokenizer': args.tokenizer,
            'max_length': args.max_length,
            'min_length': args.min_length,
            'chunk_size': args.chunk_size,
            'num_workers': args.num_workers,
            'train_split': args.train_split,
            'val_split': args.val_split,
            'test_split': args.test_split,
            'quality_validation': args.validate_data,
            'remove_duplicates': args.remove_duplicates,
            'quality_threshold': args.quality_threshold,
            'use_gpu': args.use_gpu,
            'gpu_batch_size': args.gpu_batch_size if args.use_gpu else None
        },
        'results': {
            'total_examples': total_examples,
            'valid_examples': summary['valid_examples'],
            'invalid_examples': summary['invalid_examples'],
            'duplicates_removed': summary['duplicates_removed'],
            'total_tokens': summary['total_tokens'],
            'total_gb_processed': summary['total_gb'],
            'processing_time_seconds': elapsed,
            'processing_time_human': str(timedelta(seconds=int(elapsed))),
            'throughput_examples_per_sec': total_examples / elapsed if elapsed > 0 else 0,
            'throughput_mb_per_sec': (summary['total_gb'] * 1000) / elapsed if elapsed > 0 else 0,
            'avg_tokens_per_example': summary['avg_tokens_per_example'],
            'avg_quality_score': summary['avg_quality'],
            'memory_peak_mb': summary['memory_peak_mb'],
            'gpu_memory_peak_mb': summary.get('gpu_memory_peak_mb', 0)
        },
        'errors': summary['errors'],
        'file_statistics': summary['file_stats'],
        'token_length_distribution': dict(stats.length_distribution),
        'input_files': [str(f) for f in files],
        'output_directory': str(output_dir)
    }
    
    config_file = output_dir / 'processing_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"\nConfiguration saved to: {config_file}")
    
    # Generate detailed report
    generate_detailed_report(output_dir, stats, args)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info(" " * 20 + "PROCESSING COMPLETE")

    # Stop optimizer and show final stats
    hw_optimizer.stop_monitoring()
    final_stats = hw_optimizer.get_optimal_params()
    logger.info("\n" + "="*80)
    logger.info("🔧 Hardware Optimization Summary:")
    logger.info(f"  Average CPU usage: {final_stats['cpu_usage']:.1f}%")
    logger.info(f"  Average memory usage: {final_stats['memory_usage']:.1f}%")
    if 'gpu_usage' in final_stats:
        logger.info(f"  Average GPU usage: {final_stats['gpu_usage']:.1f}%")
    logger.info(f"  Auto-adjustments made: {final_stats['adjustments']}")
    logger.info(f"  Final chunk size: {final_stats['chunk_size']}")
    logger.info(f"  Final worker count: {final_stats['num_workers']}")
    logger.info("="*80)
    logger.info("="*80)
    logger.info("\n📊 SUMMARY STATISTICS:")
    logger.info(f"  ✅ Total examples processed: {total_examples:,}")
    logger.info(f"  ✅ Valid examples: {summary['valid_examples']:,}")
    logger.info(f"  ❌ Invalid/filtered: {summary['invalid_examples']:,}")
    if summary['duplicates_removed'] > 0:
        logger.info(f"  🔁 Duplicates removed: {summary['duplicates_removed']:,}")
    
    logger.info("\n⚡ PERFORMANCE METRICS:")
    logger.info(f"  ⏱️  Total time: {timedelta(seconds=int(elapsed))}")
    logger.info(f"  🚀 Throughput: {total_examples/elapsed:.1f} examples/sec")
    logger.info(f"  📦 Data processed: {summary['total_gb']:.2f} GB")
    logger.info(f"  🧮 Total tokens: {summary['total_tokens']:,}")
    logger.info(f"  📏 Avg tokens/example: {summary['avg_tokens_per_example']:.1f}")
    
    logger.info("\n💾 RESOURCE USAGE:")
    logger.info(f"  Peak memory: {summary['memory_peak_mb']:.1f} MB")
    logger.info(f"  Current memory: {monitor_memory()['rss_mb']:.1f} MB")
    if 'gpu_memory_peak_mb' in summary and summary['gpu_memory_peak_mb'] > 0:
        logger.info(f"  Peak GPU memory: {summary['gpu_memory_peak_mb']:.1f} MB")
    logger.info(f"  Memory efficient: ✅ (streaming mode)")
    
    if summary['errors']:
        logger.info("\n⚠️  ERRORS ENCOUNTERED:")
        for error_type, count in summary['errors'].items():
            logger.info(f"  {error_type}: {count:,}")
    
    logger.info("\n📁 OUTPUT FILES:")
    logger.info(f"  Directory: {output_dir}")
    logger.info(f"  Config: {output_dir}/processing_config.json")
    logger.info(f"  Report: {output_dir}/processing_report.md")
    logger.info(f"  Train dataset: {output_dir}/train/")
    logger.info(f"  Val dataset: {output_dir}/validation/")
    if args.test_split > 0:
        logger.info(f"  Test dataset: {output_dir}/test/")
    
    logger.info("\n" + "="*80)
    logger.info(" " * 25 + "SUCCESS! 🎉")
    logger.info("="*80)


if __name__ == "__main__":
    main()
