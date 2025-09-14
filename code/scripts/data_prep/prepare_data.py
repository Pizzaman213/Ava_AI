#!/usr/bin/env python3
"""
Advanced Streaming Data Preparation Pipeline with Detailed Monitoring
- Processes data without loading everything into memory
- Provides comprehensive progress tracking and statistics
- Supports multiple data formats and quality validation
- Memory-efficient with detailed resource monitoring
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
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Also setup a separate statistics logger
    stats_logger = logging.getLogger('stats')
    stats_handler = logging.FileHandler('processing_stats.log')
    stats_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    stats_logger.addHandler(stats_handler)
    stats_logger.setLevel(logging.INFO)


def parse_args():
    """Parse enhanced command line arguments"""
    parser = argparse.ArgumentParser(
        description="Advanced Streaming Data Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single file:
    %(prog)s --input-path data.jsonl --output-dir processed/
  
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
                       default="/Users/connorsecrist/Documents/LLM/LLM/data/pretraining/processed",
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
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Number of examples to process at once")
    parser.add_argument("--max-memory-mb", type=int, default=1000,
                       help="Maximum memory usage in MB")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(),
                       help="Number of parallel workers")
    
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
        return {
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


def monitor_memory() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
        'percent': process.memory_percent(),
        'available_gb': psutil.virtual_memory().available / 1e9
    }


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
                     stats: Optional[DataStatistics] = None) -> Iterator[List[Tuple[str, Dict]]]:
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
                    logging.info(f"  Line {line_num:,}: Memory {mem['rss_mb']:.1f}MB, "
                               f"Chunks yielded: {line_num // chunk_size}")
    
    # Yield remaining chunk
    if chunk:
        yield chunk
    
    if stats:
        stats.file_stats[filepath.name]['examples'] = line_count
        stats.file_stats[filepath.name]['errors'] = error_count


def stream_json_file(filepath: Path, chunk_size: int = 1000) -> Iterator[List[Dict]]:
    """
    Stream JSON array file in chunks
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        if isinstance(data, list):
            chunk = []
            for item in data:
                text = (item.get('text') or 
                       item.get('content') or 
                       item.get('output', ''))
                
                if text and len(text) > 10:
                    chunk.append(text)
                    
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            
            if chunk:
                yield chunk
        else:
            # Single record
            text = (data.get('text') or 
                   data.get('content') or 
                   data.get('output', ''))
            if text and len(text) > 10:
                yield [text]


def stream_parquet_file(filepath: Path, chunk_size: int = 1000) -> Iterator[List[Dict]]:
    """
    Stream Parquet file in chunks using PyArrow
    """
    parquet_file = pq.ParquetFile(filepath)
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        texts = []
        df = batch.to_pandas()
        
        # Try common text columns
        for col in ['text', 'content', 'output', 'response']:
            if col in df.columns:
                valid_texts = df[col].dropna().astype(str)
                texts.extend([t for t in valid_texts if len(t) > 10])
                break
        
        if texts:
            yield texts


def save_checkpoint(checkpoint_dir: Path, state: Dict[str, Any]):
    """Save processing checkpoint for resume capability"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(state, f)
    
    # Keep only last 3 checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
    for old_checkpoint in checkpoints[:-3]:
        old_checkpoint.unlink()
    
    logging.info(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Load most recent checkpoint if available"""
    if not checkpoint_dir or not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None
    
    latest = checkpoints[-1]
    try:
        with open(latest, 'rb') as f:
            state = pickle.load(f)
        logging.info(f"Loaded checkpoint: {latest}")
        return state
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return None


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
        if filepath.suffix == '.jsonl':
            format_type = 'jsonl'
        elif filepath.suffix == '.json':
            format_type = 'json'
        elif filepath.suffix == '.parquet':
            format_type = 'parquet'
        elif filepath.suffix in ['.txt', '.text']:
            format_type = 'text'
        else:
            # Try to detect from content
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line.startswith('{'):
                    format_type = 'jsonl' if filepath.stat().st_size > 1000000 else 'json'
                else:
                    format_type = 'text'
    else:
        format_type = args.format or 'jsonl'
    
    logger.info(f"Detected format: {format_type}")
    
    # Get appropriate streamer
    if format_type == 'jsonl':
        streamer = stream_jsonl_file(filepath, args.chunk_size, args.min_length, 
                                    args.validate_data, stats)
    elif format_type == 'json':
        streamer = stream_json_file(filepath, args.chunk_size)
    elif format_type == 'parquet':
        streamer = stream_parquet_file(filepath, args.chunk_size)
    else:
        logger.warning(f"Unsupported format: {format_type}")
        return 0
    
    # Setup output files with proper splits
    train_file = output_dir / f"train_{filepath.stem}.jsonl"
    val_file = output_dir / f"val_{filepath.stem}.jsonl"
    test_file = output_dir / f"test_{filepath.stem}.jsonl" if args.test_split > 0 else None
    
    # Statistics tracking
    total_examples = 0
    train_examples = 0
    val_examples = 0
    test_examples = 0
    quality_filtered = 0
    token_count = 0
    
    # File size and estimated time
    file_size_gb = filepath.stat().st_size / 1e9
    logger.info(f"Processing {filepath.name}")
    logger.info(f"  Size: {file_size_gb:.2f} GB")
    logger.info(f"  Format: {format_type}")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Quality validation: {args.validate_data}")
    
    # Open all output files
    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        test_f = open(test_file, 'w') if test_file else None
        
        chunk_count = 0
        last_checkpoint = 0
        last_monitor = 0
        
        try:
            for chunk_data in streamer:
                chunk_count += 1
                chunk_start = time.time()
                
                # Extract texts and metadata
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
                
                # Tokenize chunk with error handling
                try:
                    tokens = tokenizer(
                        texts,
                        truncation=True,
                        max_length=args.max_length,
                        padding='max_length',
                        return_tensors='np'
                    )
                except Exception as e:
                    logger.error(f"Tokenization error: {e}")
                    stats.errors['TokenizationError'] += 1
                    continue
                
                # Process each example in chunk
                for i in range(len(texts)):
                    # Create example with metadata
                    example = {
                        'input_ids': tokens['input_ids'][i].tolist(),
                        'attention_mask': tokens['attention_mask'][i].tolist(),
                        'metadata': metadata_list[i]
                    }
                    
                    # Count tokens
                    actual_tokens = sum(example['attention_mask'])
                    token_count += actual_tokens
                    stats.total_tokens += actual_tokens
                    
                    # Determine split
                    random_val = np.random.random()
                    if random_val < args.train_split:
                        train_f.write(json.dumps(example) + '\n')
                        train_examples += 1
                    elif random_val < args.train_split + args.val_split:
                        val_f.write(json.dumps(example) + '\n')
                        val_examples += 1
                    elif test_f:
                        test_f.write(json.dumps(example) + '\n')
                        test_examples += 1
                    
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
                    
                    elapsed = time.time() - file_start_time
                    rate = total_examples / elapsed if elapsed > 0 else 0
                    eta = (file_size_gb * 1e9 / filepath.stat().st_size * total_examples - total_examples) / rate if rate > 0 else 0
                    
                    logger.info(f"  Progress: {total_examples:,} examples")
                    logger.info(f"    Chunks: {chunk_count}, Tokens: {token_count:,}")
                    logger.info(f"    Train: {train_examples:,}, Val: {val_examples:,}, Test: {test_examples:,}")
                    logger.info(f"    Quality filtered: {quality_filtered:,}")
                    logger.info(f"    Rate: {rate:.0f} ex/s, ETA: {timedelta(seconds=int(eta))}")
                    logger.info(f"    Memory: {mem['rss_mb']:.1f}MB (available: {mem['available_gb']:.1f}GB)")
                    
                    stats_logger.info(json.dumps({
                        'file': filepath.name,
                        'examples': total_examples,
                        'tokens': token_count,
                        'memory_mb': mem['rss_mb'],
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
            if test_f:
                test_f.close()
    
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


def combine_jsonl_files(files: List[Path], output_file: Path):
    """
    Combine multiple JSONL files into one
    """
    with open(output_file, 'w') as out_f:
        for filepath in files:
            if filepath.exists():
                with open(filepath, 'r') as in_f:
                    for line in in_f:
                        out_f.write(line)
                # Remove temporary file
                filepath.unlink()


def generate_detailed_report(output_dir: Path, stats: DataStatistics, args: argparse.Namespace):
    """Generate comprehensive processing report"""
    report_file = output_dir / "processing_report.md"
    summary = stats.get_summary()
    
    with open(report_file, 'w') as f:
        f.write("# Data Processing Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Tokenizer: {args.tokenizer}\n")
        f.write(f"- Max Length: {args.max_length}\n")
        f.write(f"- Chunk Size: {args.chunk_size}\n")
        f.write(f"- Quality Validation: {args.validate_data}\n")
        f.write(f"- Remove Duplicates: {args.remove_duplicates}\n\n")
        
        f.write("## Summary Statistics\n")
        f.write(f"- Total Examples: {summary['total_examples']:,}\n")
        f.write(f"- Valid Examples: {summary['valid_examples']:,}\n")
        f.write(f"- Invalid Examples: {summary['invalid_examples']:,}\n")
        f.write(f"- Duplicates Removed: {summary['duplicates_removed']:,}\n")
        f.write(f"- Total Tokens: {summary['total_tokens']:,}\n")
        f.write(f"- Data Size: {summary['total_gb']:.2f} GB\n")
        f.write(f"- Processing Time: {timedelta(seconds=int(summary['processing_time']))}\n")
        f.write(f"- Throughput: {summary['throughput']:.1f} examples/sec\n")
        f.write(f"- Avg Tokens/Example: {summary['avg_tokens_per_example']:.1f}\n")
        f.write(f"- Avg Quality Score: {summary['avg_quality']:.3f}\n")
        f.write(f"- Peak Memory: {summary['memory_peak_mb']:.1f} MB\n\n")
        
        if summary['errors']:
            f.write("## Errors\n")
            for error_type, count in summary['errors'].items():
                f.write(f"- {error_type}: {count:,}\n")
            f.write("\n")
        
        f.write("## File Statistics\n")
        f.write("| File | Examples | Tokens | Errors | Time |\n")
        f.write("|------|----------|--------|--------|------|\n")
        for filename, file_stats in summary['file_stats'].items():
            f.write(f"| {filename} | {file_stats['examples']:,} | "
                   f"{file_stats['tokens']:,} | {file_stats['errors']:,} | "
                   f"{timedelta(seconds=int(file_stats['time']))} |\n")
        
        f.write("\n## Token Length Distribution\n")
        if stats.length_distribution:
            f.write("| Length Range | Count | Percentage |\n")
            f.write("|--------------|-------|------------|\n")
            total = sum(stats.length_distribution.values())
            for length, count in sorted(stats.length_distribution.items()):
                f.write(f"| {length}-{length+99} | {count:,} | {count/total*100:.1f}% |\n")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Report saved to: {report_file}")


def main():
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file or f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(level=args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    stats_logger = logging.getLogger('stats')
    
    logger.info("\n" + "="*80)
    logger.info(" " * 15 + "ADVANCED DATA PREPARATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Version: 2.0 - Enhanced with detailed monitoring")
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
        
        # Also check for HuggingFace dataset format
        if (input_path / "dataset_info.json").exists():
            logger.info("Detected HuggingFace dataset format")
            # Add handling for HF datasets
        
        logger.info(f"Found {len(files)} data files")
        if files:
            total_size_gb = sum(f.stat().st_size for f in files) / 1e9
            logger.info(f"Total size: {total_size_gb:.2f} GB")
            
            # Show file breakdown
            format_counts = defaultdict(int)
            for f in files:
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
    
    # Combine all temporary files
    logger.info("\n" + "="*80)
    logger.info("COMBINING OUTPUT FILES")
    logger.info("="*80)
    
    final_train = output_dir / "train.jsonl"
    final_val = output_dir / "validation.jsonl"
    final_test = output_dir / "test.jsonl" if args.test_split > 0 else None
    
    logger.info("Combining training files...")
    combine_jsonl_files(train_files, final_train)
    
    logger.info("Combining validation files...")
    combine_jsonl_files(val_files, final_val)
    
    if test_files and final_test:
        logger.info("Combining test files...")
        combine_jsonl_files(test_files, final_test)
    
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
    
    # Save each split
    train_count = save_dataset_split(final_train, output_dir / "train", "train")
    val_count = save_dataset_split(final_val, output_dir / "validation", "validation")
    test_count = 0
    if final_test:
        test_count = save_dataset_split(final_test, output_dir / "test", "test")
    
    logger.info(f"\nDataset splits saved:")
    logger.info(f"  Train: {train_count:,} examples")
    logger.info(f"  Validation: {val_count:,} examples")
    if test_count > 0:
        logger.info(f"  Test: {test_count:,} examples")
    
    # Clean up temporary JSONL files
    logger.info("\nCleaning up temporary files...")
    for f in [final_train, final_val, final_test]:
        if f and f.exists():
            f.unlink()
            logger.debug(f"  Removed: {f}")
    
    # Save comprehensive configuration and statistics
    elapsed = time.time() - start_time
    summary = stats.get_summary()
    
    config = {
        'pipeline_version': '2.0',
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
            'quality_threshold': args.quality_threshold
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
            'memory_peak_mb': summary['memory_peak_mb']
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