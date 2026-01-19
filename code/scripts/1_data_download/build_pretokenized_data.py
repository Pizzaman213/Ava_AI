#!/usr/bin/env python3
"""
Build complete pre-tokenized data repository.

Creates a clean `/pretokenized_data/` folder with:
- Tokenizer/ (configurable tokenizers)
- Data/ (all tokenized datasets)

Features:
- Train new tokenizers from raw data
- Re-tokenize all datasets
- Organize everything cleanly
- Summary and statistics
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import shutil
import json

def auto_install_requirements():
    """Auto-install requirements if needed"""
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

try:
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    import torch
except ImportError:
    print("Installing dependencies...")
    auto_install_requirements()
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    import torch

import argparse
import multiprocessing


# Get CPU count for parallel processing
NUM_CPUS = multiprocessing.cpu_count()



class TokenizerBuilder:
    """Creates and manages tokenizers"""

    @staticmethod
    def copy_tokenizer(source_path: Path, dest_path: Path, tokenizer_name: str = "16k"):
        """Copy existing tokenizer to pre-tokenized folder"""
        source = Path(source_path)
        dest = dest_path / tokenizer_name

        if not source.exists():
            print(f"✗ Tokenizer not found: {source}")
            return False

        print(f"Copying tokenizer: {tokenizer_name}")
        print(f"  From: {source}")
        print(f"  To:   {dest}")

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(source, dest)
        print(f"✓ Tokenizer copied\n")
        return True

    @staticmethod
    def list_available_tokenizers(tokenizer_dir: Path):
        """List all available tokenizers"""
        if not tokenizer_dir.exists():
            return []

        tokenizers = [d.name for d in tokenizer_dir.iterdir() if d.is_dir()]
        return sorted(tokenizers)


class DataTokenizer:
    """Tokenizes datasets with multiprocessing support and Rust tokenizer"""

    def __init__(self, tokenizer_path: str, num_proc: int = None, max_length: int = 512, sample_ratio: float = 1.0):
        """Initialize with tokenizer

        Args:
            tokenizer_path: Path to tokenizer
            num_proc: Number of processes for parallel tokenization (default: CPU count)
            max_length: Fixed sequence length for padding/truncation (enables zero-copy loading)
            sample_ratio: Ratio of data to sample (0.0-1.0), e.g., 0.05 for 5%
        """
        self.fast_tokenizer = None
        self.use_fast = False
        self.max_length = max_length
        self.sample_ratio = sample_ratio
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.num_proc = num_proc if num_proc else NUM_CPUS  # Use all cores

    def _load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer - prefer fast Rust tokenizer for 5-10x speedup"""
        from tokenizers import Tokenizer as RustTokenizer

        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer_path = Path(tokenizer_path)
        tokenizer_json = tokenizer_path / "tokenizer.json"

        # Try to load fast Rust tokenizer first (5-10x faster)
        if tokenizer_json.exists():
            try:
                self.fast_tokenizer = RustTokenizer.from_file(str(tokenizer_json))
                self.use_fast = True
                print(f"✓ Loaded FAST Rust tokenizer (vocab: {self.fast_tokenizer.get_vocab_size()})")
            except Exception as e:
                print(f"⚠ Could not load Rust tokenizer: {e}")
                self.fast_tokenizer = None
                self.use_fast = False

        # If we have the fast tokenizer, that's all we need for tokenization
        if self.use_fast and self.fast_tokenizer is not None:
            print(f"  Using Rust tokenizer (HuggingFace fallback not required)\n")
            # Return a minimal object that stores the path for reference
            class MinimalTokenizer:
                def __init__(self, path):
                    self.name_or_path = str(path)
            return MinimalTokenizer(tokenizer_path)

        # Fallback: load HuggingFace tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            print(f"✓ Tokenizer loaded (HuggingFace). Vocab size: {len(tokenizer)}\n")
            return tokenizer
        except Exception as e:
            pass

        # Try transformers subdirectory
        transformers_path = tokenizer_path / "transformers"
        if transformers_path.exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(transformers_path))
                print(f"✓ Tokenizer loaded (HuggingFace). Vocab size: {len(tokenizer)}\n")
                return tokenizer
            except:
                pass

        raise ValueError(f"Could not load tokenizer from {tokenizer_path}")

    def _is_image_column(self, col_name: str) -> bool:
        """Check if column name suggests it contains images"""
        image_keywords = ['image', 'img', 'picture', 'photo', 'visual', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp']
        return any(keyword in col_name.lower() for keyword in image_keywords)

    def _is_image_data(self, value) -> bool:
        """Check if value is image data (bytes, PIL Image, etc)"""
        try:
            # Check for PIL Image
            from PIL import Image
            if isinstance(value, Image.Image):
                return True
        except:
            pass

        # Check for bytes/bytearray that look like images
        if isinstance(value, (bytes, bytearray)):
            # Check magic numbers for common image formats
            magic_numbers = [
                b'\x89PNG',      # PNG
                b'\xff\xd8\xff', # JPEG
                b'GIF8',         # GIF
                b'BM',           # BMP
                b'RIFF',         # WEBP/WAV
            ]
            for magic in magic_numbers:
                if value.startswith(magic):
                    return True
            return False

        # Check for dict-like structures (HuggingFace Image feature)
        if isinstance(value, dict):
            if 'bytes' in value or 'path' in value:
                return True

        return False

    def _detect_text_columns(self, dataset) -> List[str]:
        """Detect which columns contain text, filtering out image columns"""
        if not hasattr(dataset, 'column_names'):
            return ["text"]

        text_columns = []
        image_columns_found = []

        for col in dataset.column_names:
            # Skip if column name suggests images
            if self._is_image_column(col):
                image_columns_found.append(col)
                continue

            # Check for known text column names
            if col in ["text", "content", "data", "input", "instruction", "output", "completion"]:
                text_columns.append(col)
                continue

            # Check actual data type
            try:
                sample_value = dataset[0][col]

                # Skip if it's image data
                if self._is_image_data(sample_value):
                    image_columns_found.append(col)
                    continue

                # Include if it's text
                if isinstance(sample_value, str):
                    text_columns.append(col)
            except:
                pass

        # Log filtered columns
        if image_columns_found:
            print(f"  ⊘ Filtered out image columns: {', '.join(image_columns_found)}")

        return text_columns if text_columns else ["text"]

    def _tokenize_batch(self, examples: Dict, text_columns: List[str]) -> Dict:
        """Tokenize a batch of examples - uses fast Rust tokenizer when available"""
        texts = []

        for col in text_columns:
            if col in examples:
                col_texts = examples[col]
                if isinstance(col_texts, str):
                    col_texts = [col_texts]
                texts.extend(col_texts)

        if not texts:
            return {"input_ids": [], "attention_mask": []}

        # Use fast Rust tokenizer if available (5-10x faster)
        if self.use_fast and self.fast_tokenizer is not None:
            encodings = self.fast_tokenizer.encode_batch(texts)
            return {
                "input_ids": [e.ids for e in encodings],
                "attention_mask": [e.attention_mask for e in encodings],
            }

        # Fallback to HuggingFace tokenizer
        encodings = self.tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=True
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings.get("attention_mask", []),
        }

    def _tokenize_and_save_streaming(self, dataset, output_path: Path, text_columns: List[str], shard_size: int = 100000, source_path: Path = None):
        """True streaming tokenization with FIXED-LENGTH padding for zero-copy loading

        Uses multiprocessing (not threading) to bypass Python's GIL and achieve true parallelism.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import numpy as np
        import time
        from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
        import threading
        from queue import Queue

        start = time.time()
        output_path.mkdir(parents=True, exist_ok=True)

        # Load fast tokenizer for this worker
        from tokenizers import Tokenizer as RustTokenizer
        tokenizer_path = Path(self.tokenizer.name_or_path) / "tokenizer.json"
        fast_tok = RustTokenizer.from_file(str(tokenizer_path))

        # Get pad token ID (default to 0 if not found)
        pad_token_id = 0
        try:
            pad_token_id = fast_tok.token_to_id("[PAD]") or 0
        except:
            pass

        # Enable parallelism in the Rust tokenizer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Prefetch settings for faster I/O
        prefetch_batches = 100  # Large prefetch buffer
        batch_size = 5000  # Smaller batches for better IPC throughput
        num_tokenizer_processes = self.num_proc  # Use all cores for tokenization (multiprocessing!)

        print(f"  HIGH-SPEED MULTIPROCESSING MODE (bypasses GIL)")
        print(f"  Shard size: {shard_size:,} sequences per file")
        print(f"  FIXED-LENGTH PADDING: {self.max_length} tokens (enables zero-copy loading)")
        print(f"  Pad token ID: {pad_token_id}")
        print(f"  Batch size: {batch_size:,} | Prefetch: {prefetch_batches} | Tokenizer processes: {num_tokenizer_processes}")

        # Schema for Arrow output - USE FIXED SIZE LIST for zero-copy loading!
        schema = pa.schema([
            ('input_ids', pa.list_(pa.int64(), self.max_length)),
            ('attention_mask', pa.list_(pa.int64(), self.max_length)),
        ])

        # Streaming state - use pre-allocated numpy arrays for speed
        shard_idx = 0
        total_sequences = 0
        skipped_empty = 0

        # Pre-allocate buffer as numpy array (much faster than list.extend())
        buffer_capacity = shard_size + batch_size * 10  # Extra space to avoid reallocation
        buffer_ids = np.zeros((buffer_capacity, self.max_length), dtype=np.int64)
        buffer_masks = np.zeros((buffer_capacity, self.max_length), dtype=np.int64)
        buffer_pos = 0  # Current position in buffer

        def flush_shard():
            """Write current buffer to disk as Arrow shard with fixed-size arrays"""
            nonlocal shard_idx, buffer_pos, total_sequences
            if buffer_pos == 0:
                return

            shard_file = output_path / f"data-{shard_idx:05d}.arrow"

            # Slice the pre-allocated arrays (no copy, just a view)
            ids_to_write = buffer_ids[:buffer_pos]
            masks_to_write = buffer_masks[:buffer_pos]

            # Create Arrow arrays directly from numpy using FixedSizeListArray
            # This avoids the slow tolist() conversion
            num_rows = buffer_pos
            flat_ids = pa.array(ids_to_write.ravel(), type=pa.int64())
            flat_masks = pa.array(masks_to_write.ravel(), type=pa.int64())

            ids_list_arr = pa.FixedSizeListArray.from_arrays(flat_ids, self.max_length)
            mask_list_arr = pa.FixedSizeListArray.from_arrays(flat_masks, self.max_length)

            table = pa.table({
                'input_ids': ids_list_arr,
                'attention_mask': mask_list_arr
            })

            with open(str(shard_file), 'wb') as f:
                writer = ipc.new_stream(f, table.schema)
                writer.write_table(table)
                writer.close()

            total_sequences += buffer_pos
            shard_idx += 1
            buffer_pos = 0  # Reset position (reuse buffer)

        # Queues for pipeline: readers -> tokenizers -> writer
        # Use multiprocessing queues for true parallelism (bypasses GIL)
        text_queue = MPQueue(maxsize=prefetch_batches)  # Raw text batches (multiprocessing)
        token_queue = MPQueue(maxsize=prefetch_batches)  # Tokenized batches (multiprocessing)
        done_reading = MPEvent()  # Multiprocessing event

        # Scan for parquet/arrow/json files directly in the source directory
        # This bypasses HuggingFace's slow streaming
        if source_path is None:
            source_path = output_path

        parquet_files = list(Path(source_path).rglob("*.parquet")) if source_path else []
        arrow_files = list(Path(source_path).rglob("*.arrow")) if source_path else []
        json_files = (list(Path(source_path).rglob("*.json")) + list(Path(source_path).rglob("*.jsonl"))) if source_path else []
        txt_files = list(Path(source_path).rglob("*.txt")) if source_path else []

        all_data_files = parquet_files + arrow_files + json_files
        print(f"  Source: {source_path}")
        print(f"  Found: {len(parquet_files)} parquet, {len(arrow_files)} arrow, {len(json_files)} json, {len(txt_files)} txt")

        # Apply sampling if requested
        if self.sample_ratio < 1.0 and all_data_files:
            import random
            random.seed(42)  # Reproducible sampling
            num_files_to_keep = max(1, int(len(all_data_files) * self.sample_ratio))
            all_data_files = sorted(random.sample(all_data_files, num_files_to_keep))
            print(f"  SAMPLING: {self.sample_ratio*100:.0f}% = {num_files_to_keep} files (from {len(parquet_files) + len(arrow_files) + len(json_files)} total)")

        # If we found parquet/arrow files, use the COMBINED worker approach (no IPC overhead)
        use_combined_workers = len(all_data_files) > 0

        if use_combined_workers:
            # COMBINED WORKER MODE: Each process reads, tokenizes, and writes independently
            # This eliminates all IPC bottlenecks
            from multiprocessing import Value, Lock
            import pyarrow.parquet as pq

            num_workers = 6  # Use 6 cores
            print(f"  ZERO-IPC MODE: {num_workers} combined worker processes")
            print(f"  Each worker: reads -> tokenizes -> writes (no queue bottleneck)")

            # Shared counters for coordination
            shard_counter = Value('i', 0)
            seq_counter = Value('i', 0)
            shard_lock = Lock()

            # Create work items by splitting files into chunks
            # This allows 6 workers even with fewer files
            work_items = []
            chunk_size = 500000  # Rows per chunk

            for f in all_data_files:
                file_path = str(f)
                try:
                    if file_path.endswith('.parquet'):
                        # Get row count from parquet metadata (fast, no loading)
                        pf = pq.ParquetFile(file_path)
                        num_rows = pf.metadata.num_rows
                    elif file_path.endswith('.json.gz') or file_path.endswith('.jsonl.gz'):
                        # Estimate line count for compressed JSON (quick heuristic)
                        # C4 files are ~380MB compressed with ~384k lines
                        import os
                        file_size = os.path.getsize(file_path)
                        num_rows = int(file_size / 1000)  # ~1KB per line compressed
                    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
                        # Count lines for uncompressed JSON
                        with open(file_path, 'r') as f_count:
                            num_rows = sum(1 for _ in f_count)
                    else:
                        # For arrow files, estimate or load
                        num_rows = chunk_size  # Default to one chunk
                except:
                    num_rows = chunk_size

                # Split into chunks
                for start in range(0, num_rows, chunk_size):
                    end = min(start + chunk_size, num_rows)
                    work_items.append((file_path, start, end))

            print(f"  Split {len(all_data_files)} files into {len(work_items)} work chunks")

            # Work queue for distribution
            file_queue_mp = MPQueue()
            for item in work_items:
                file_queue_mp.put(item)
            # Add sentinel values
            for _ in range(num_workers):
                file_queue_mp.put(None)

            # Start combined worker processes
            workers = []
            for _ in range(num_workers):
                p = Process(
                    target=_combined_worker_process,
                    args=(file_queue_mp, str(output_path), text_columns, str(tokenizer_path), self.max_length, shard_counter, shard_lock, seq_counter),
                    daemon=True
                )
                p.start()
                workers.append(p)

            # Monitor progress
            last_print_time = time.time()
            while any(p.is_alive() for p in workers):
                time.sleep(0.5)
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    elapsed = current_time - start
                    with shard_lock:
                        total_sequences = seq_counter.value
                        shard_idx = shard_counter.value
                    rate = total_sequences / max(1, elapsed)
                    active = sum(1 for p in workers if p.is_alive())
                    print(f"  {total_sequences:,} seq | {shard_idx} shards | {rate:,.0f}/s | workers:{active}/{num_workers}", end='\r')
                    last_print_time = current_time

            # Wait for all workers
            for p in workers:
                p.join(timeout=10.0)
                if p.is_alive():
                    p.terminate()

            # Get final counts
            with shard_lock:
                total_sequences = seq_counter.value
                shard_idx = shard_counter.value

        else:
            # FALLBACK: streaming mode for non-parquet sources (uses queues)
            print(f"  STREAMING MODE (fallback for non-parquet sources)")

            def streaming_reader_fallback():
                text_buffer = []
                for row in dataset:
                    for col in text_columns:
                        if col in row and row[col]:
                            text = str(row[col])
                            if text.strip():
                                text_buffer.append(text)
                                if len(text_buffer) >= batch_size:
                                    text_queue.put(text_buffer)
                                    text_buffer = []
                if text_buffer:
                    text_queue.put(text_buffer)

            worker_params = {'tokenizer_path': str(tokenizer_path), 'max_length': self.max_length}

            # Single reader thread
            reader_thread = threading.Thread(target=streaming_reader_fallback, daemon=True)
            reader_thread.start()

            # Tokenizer processes
            tokenizer_processes = []
            for _ in range(self.num_proc):
                p = Process(
                    target=_tokenizer_worker_process,
                    args=(text_queue, token_queue, done_reading, worker_params['tokenizer_path'], worker_params['max_length']),
                    daemon=True
                )
                p.start()
                tokenizer_processes.append(p)

            def reader_monitor():
                reader_thread.join()
                done_reading.set()

            monitor_thread = threading.Thread(target=reader_monitor, daemon=True)
            monitor_thread.start()

            # Main loop
            last_print_time = time.time()
            while True:
                all_done = all(not p.is_alive() for p in tokenizer_processes)
                if all_done and token_queue.empty():
                    break
                try:
                    batch_ids, batch_masks = token_queue.get(timeout=0.1)
                except:
                    continue

                batch_len = len(batch_ids)
                if buffer_pos + batch_len > buffer_capacity:
                    flush_shard()

                batch_ids_arr = np.array(batch_ids, dtype=np.int64)
                batch_masks_arr = np.array(batch_masks, dtype=np.int64)
                buffer_ids[buffer_pos:buffer_pos + batch_len] = batch_ids_arr
                buffer_masks[buffer_pos:buffer_pos + batch_len] = batch_masks_arr
                buffer_pos += batch_len

                if buffer_pos >= shard_size:
                    flush_shard()

                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    elapsed = current_time - start
                    rate = total_sequences / max(1, elapsed)
                    print(f"  {total_sequences:,} seq | {shard_idx} shards | {rate:,.0f}/s", end='\r')
                    last_print_time = current_time

            monitor_thread.join(timeout=10.0)
            for p in tokenizer_processes:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
            flush_shard()

        elapsed = time.time() - start
        rate = total_sequences / max(1, elapsed)
        print(f"\n  ✓ COMPLETE: {total_sequences:,} sequences in {elapsed:.1f}s ({rate:,.0f} seq/s)")
        print(f"  Written {shard_idx} shards to {output_path}")
        print(f"  Fixed sequence length: {self.max_length} (zero-copy compatible)")
        if skipped_empty > 0:
            print(f"  Skipped {skipped_empty:,} empty sequences")

        return shard_idx, total_sequences


# Module-level worker functions (required for multiprocessing)
def _combined_worker_process(work_queue, output_path, text_columns, tokenizer_path, max_length, shard_counter, shard_lock, seq_counter):
    """Combined reader+tokenizer+writer process - eliminates IPC bottleneck

    Work items are tuples: (file_path, start_row, end_row) for chunked processing

    For instruction-tuning data (system_prompt, question/instruction, response/output),
    concatenates into: [SYSTEM] {system} [USER] {question} [ASSISTANT] {response}
    """
    import pyarrow.parquet as pq
    import pyarrow.ipc as ipc
    import pyarrow as pa
    import numpy as np
    from tokenizers import Tokenizer as RustTokenizer
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load tokenizer
    tok = RustTokenizer.from_file(tokenizer_path)
    pad_id = tok.token_to_id("[PAD]") or 0

    # Local buffer
    shard_size = 100000
    buffer_capacity = shard_size + 10000
    buffer_ids = np.zeros((buffer_capacity, max_length), dtype=np.int64)
    buffer_masks = np.zeros((buffer_capacity, max_length), dtype=np.int64)
    buffer_pos = 0
    local_total = 0

    def pad_or_truncate(ids, pad_value=0):
        if len(ids) >= max_length:
            return ids[:max_length]
        else:
            return ids + [pad_value] * (max_length - len(ids))

    def flush_local_shard():
        nonlocal buffer_pos, local_total
        if buffer_pos == 0:
            return

        # Get unique shard index atomically
        with shard_lock:
            shard_idx = shard_counter.value
            shard_counter.value += 1

        shard_file = Path(output_path) / f"data-{shard_idx:05d}.arrow"

        # Create Arrow arrays directly from numpy
        flat_ids = pa.array(buffer_ids[:buffer_pos].ravel(), type=pa.int64())
        flat_masks = pa.array(buffer_masks[:buffer_pos].ravel(), type=pa.int64())
        ids_list_arr = pa.FixedSizeListArray.from_arrays(flat_ids, max_length)
        mask_list_arr = pa.FixedSizeListArray.from_arrays(flat_masks, max_length)

        table = pa.table({'input_ids': ids_list_arr, 'attention_mask': mask_list_arr})

        with open(str(shard_file), 'wb') as f:
            writer = ipc.new_stream(f, table.schema)
            writer.write_table(table)
            writer.close()

        local_total += buffer_pos
        with shard_lock:
            seq_counter.value += buffer_pos
        buffer_pos = 0

    # Cache for loaded tables (in case multiple chunks from same file)
    table_cache = {}

    # Define instruction-tuning column mappings
    SYSTEM_COLS = {'system_prompt', 'system', 'context'}
    USER_COLS = {'question', 'instruction', 'input', 'prompt', 'user'}
    ASSISTANT_COLS = {'response', 'output', 'answer', 'completion', 'assistant'}
    SKIP_COLS = {'id', 'idx', 'index', 'row_id', 'uuid'}  # Never tokenize these

    while True:
        try:
            work_item = work_queue.get(timeout=1.0)
        except:
            continue

        if work_item is None:
            break

        file_path, start_row, end_row = work_item

        try:
            # Load file (or use cache)
            if file_path not in table_cache:
                if file_path.endswith('.parquet'):
                    table_cache[file_path] = pq.read_table(file_path)
                elif file_path.endswith('.arrow'):
                    with open(file_path, 'rb') as f:
                        reader = ipc.open_stream(f)
                        table_cache[file_path] = reader.read_all()
                elif file_path.endswith('.json.gz') or file_path.endswith('.jsonl.gz'):
                    # Handle compressed JSON (like C4)
                    import gzip
                    import json
                    texts = []
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                                if 'text' in obj:
                                    texts.append(obj['text'])
                                elif 'content' in obj:
                                    texts.append(obj['content'])
                            except:
                                continue
                    # Store as simple list, not Arrow table
                    table_cache[file_path] = {'_texts': texts, '_is_json': True}
                elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
                    import json
                    texts = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                                if 'text' in obj:
                                    texts.append(obj['text'])
                                elif 'content' in obj:
                                    texts.append(obj['content'])
                            except:
                                continue
                    table_cache[file_path] = {'_texts': texts, '_is_json': True}
                else:
                    continue

            cached = table_cache[file_path]

            # Handle JSON files differently (already have texts list)
            if isinstance(cached, dict) and cached.get('_is_json'):
                all_texts = cached['_texts']
                # Slice to our chunk
                texts = all_texts[start_row:end_row]

                # Batch tokenize
                batch_size = 5000
                for i in range(0, len(texts), batch_size):
                    batch_texts = [t.strip() for t in texts[i:i+batch_size] if t and t.strip()]
                    if not batch_texts:
                        continue
                    encodings = tok.encode_batch(batch_texts)

                    for enc in encodings:
                        if len(enc.ids) == 0:
                            continue

                        if buffer_pos >= shard_size:
                            flush_local_shard()

                        padded = pad_or_truncate(enc.ids, pad_id)
                        real_len = min(len(enc.ids), max_length)
                        mask = [1] * real_len + [0] * (max_length - real_len)

                        buffer_ids[buffer_pos] = padded
                        buffer_masks[buffer_pos] = mask
                        buffer_pos += 1
                continue  # Skip the Arrow table processing below

            table = cached
            cols = set(c.lower() for c in table.column_names)
            col_map = {c.lower(): c for c in table.column_names}  # lowercase -> actual name

            # Slice to our chunk
            chunk_table = table.slice(start_row, end_row - start_row)
            num_rows = chunk_table.num_rows

            # Detect if this is instruction-tuning format
            has_system = bool(cols & SYSTEM_COLS)
            has_user = bool(cols & USER_COLS)
            has_assistant = bool(cols & ASSISTANT_COLS)
            is_instruction_format = has_user and has_assistant

            if is_instruction_format:
                # INSTRUCTION FORMAT: Concatenate columns into single sequence per row
                system_col = col_map.get(next((c for c in SYSTEM_COLS if c in cols), None))
                user_col = col_map.get(next((c for c in USER_COLS if c in cols), None))
                assistant_col = col_map.get(next((c for c in ASSISTANT_COLS if c in cols), None))

                # Get column data
                system_data = chunk_table.column(system_col).to_pylist() if system_col else [None] * num_rows
                user_data = chunk_table.column(user_col).to_pylist() if user_col else [None] * num_rows
                assistant_data = chunk_table.column(assistant_col).to_pylist() if assistant_col else [None] * num_rows

                # Build concatenated texts
                texts = []
                for i in range(num_rows):
                    parts = []
                    if system_data[i]:
                        parts.append(f"[SYSTEM] {system_data[i].strip()}")
                    if user_data[i]:
                        parts.append(f"[USER] {user_data[i].strip()}")
                    if assistant_data[i]:
                        parts.append(f"[ASSISTANT] {assistant_data[i].strip()}")
                    if parts:
                        texts.append(" ".join(parts))

                # Batch tokenize
                batch_size = 5000
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    encodings = tok.encode_batch(batch_texts)

                    for enc in encodings:
                        if len(enc.ids) == 0:
                            continue

                        if buffer_pos >= shard_size:
                            flush_local_shard()

                        padded = pad_or_truncate(enc.ids, pad_id)
                        real_len = min(len(enc.ids), max_length)
                        mask = [1] * real_len + [0] * (max_length - real_len)

                        buffer_ids[buffer_pos] = padded
                        buffer_masks[buffer_pos] = mask
                        buffer_pos += 1

            else:
                # PLAIN TEXT FORMAT: Process each text column separately (excluding skip cols)
                for col in text_columns:
                    if col.lower() in SKIP_COLS:
                        continue
                    if col not in chunk_table.column_names:
                        continue

                    column = chunk_table.column(col)

                    try:
                        values = column.to_pylist()
                    except:
                        continue

                    texts = [v.strip() for v in values if v and isinstance(v, str) and v.strip()]
                    if not texts:
                        continue

                    batch_size = 5000
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        encodings = tok.encode_batch(batch_texts)

                        for enc in encodings:
                            if len(enc.ids) == 0:
                                continue

                            if buffer_pos >= shard_size:
                                flush_local_shard()

                            padded = pad_or_truncate(enc.ids, pad_id)
                            real_len = min(len(enc.ids), max_length)
                            mask = [1] * real_len + [0] * (max_length - real_len)

                            buffer_ids[buffer_pos] = padded
                            buffer_masks[buffer_pos] = mask
                            buffer_pos += 1

        except Exception as e:
            print(f"\n  Warning: Failed to process {file_path}[{start_row}:{end_row}]: {e}")

    # Flush remaining
    flush_local_shard()


def _reader_worker_process(file_queue, text_queue, text_columns, batch_size):
    """Worker PROCESS to read files - runs in separate process to bypass GIL (LEGACY)"""
    import pyarrow.parquet as pq
    import pyarrow.ipc as ipc

    while True:
        try:
            file_path = file_queue.get(timeout=1.0)
        except:
            continue

        # Sentinel value means no more files
        if file_path is None:
            break

        try:
            # Load table based on file type
            if file_path.endswith('.parquet'):
                table = pq.read_table(file_path)
            elif file_path.endswith('.arrow'):
                with open(file_path, 'rb') as f:
                    reader = ipc.open_stream(f)
                    table = reader.read_all()
            else:
                continue

            text_buffer = []

            for col in text_columns:
                if col in table.column_names:
                    # Use chunked reading for large columns
                    column = table.column(col)
                    for chunk in column.chunks:
                        # Use numpy for faster extraction when possible
                        try:
                            values = chunk.to_pylist()
                        except:
                            continue

                        for val in values:
                            if val and isinstance(val, str) and val.strip():
                                text_buffer.append(val.strip())

                                if len(text_buffer) >= batch_size:
                                    text_queue.put(text_buffer)
                                    text_buffer = []

            if text_buffer:
                text_queue.put(text_buffer)

        except Exception as e:
            print(f"\n  Warning: Failed to read {file_path}: {e}")


def _tokenizer_worker_process(text_queue, token_queue, done_reading, tokenizer_path, max_length):
    """Worker PROCESS to tokenize batches - runs in separate process to bypass GIL"""
    from tokenizers import Tokenizer as RustTokenizer
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable internal parallelism in subprocess

    local_tok = RustTokenizer.from_file(tokenizer_path)
    local_pad_id = local_tok.token_to_id("[PAD]") or 0

    def pad_or_truncate(ids, pad_value=0):
        if len(ids) >= max_length:
            return ids[:max_length]
        else:
            return ids + [pad_value] * (max_length - len(ids))

    processed = 0
    while True:
        # Check termination condition
        if done_reading.is_set():
            try:
                # Non-blocking check if queue is empty
                if text_queue.empty():
                    break
            except:
                break

        try:
            text_batch = text_queue.get(timeout=0.5)
        except:
            if done_reading.is_set():
                break
            continue

        encodings = local_tok.encode_batch(text_batch)
        batch_ids = []
        batch_masks = []

        for enc in encodings:
            if len(enc.ids) == 0:
                continue
            padded_ids = pad_or_truncate(enc.ids, local_pad_id)
            real_len = min(len(enc.ids), max_length)
            padded_mask = [1] * real_len + [0] * (max_length - real_len)
            batch_ids.append(padded_ids)
            batch_masks.append(padded_mask)

        if batch_ids:
            token_queue.put((batch_ids, batch_masks))
            processed += len(batch_ids)


def _tokenize_chunk_worker(args):
    """Worker: tokenize a chunk of texts"""
    texts, tok_path = args
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    encodings = tok.encode_batch(texts)
    return [(e.ids, e.attention_mask) for e in encodings]


def _write_shard_worker(args):
    """Worker: write a single Arrow shard"""
    import pyarrow as pa
    from pathlib import Path

    idx, ids_chunk, mask_chunk, out_path_str = args
    out_path = Path(out_path_str)
    out_path.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([('input_ids', pa.list_(pa.int32())),
                        ('attention_mask', pa.list_(pa.int32()))])
    table = pa.Table.from_pydict({'input_ids': ids_chunk, 'attention_mask': mask_chunk}, schema=schema)

    shard_file = out_path / f"data-{idx:05d}.arrow"
    with open(str(shard_file), 'wb') as f:
        writer = pa.ipc.new_stream(f, schema)
        writer.write_table(table)
        writer.close()


# Add methods back to DataTokenizer class
DataTokenizer._load_dataset = lambda self, dataset_path: _load_dataset_impl(self, dataset_path)
DataTokenizer._load_text_files = lambda self: None


def _load_dataset_impl(tokenizer_instance, dataset_path: Path) -> Optional[Dataset]:
    """Load dataset in STREAMING mode - never loads full dataset into RAM"""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        return None

    # Skip image files when loading
    skip_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg"}

    # Try streaming mode first for memory efficiency
    formats = [
        ("parquet", "Parquet", lambda files: load_dataset("parquet", data_files=files, streaming=True)),
        ("json.gz", "JSON.GZ", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("jsonl.gz", "JSONL.GZ", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("json", "JSON", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("jsonl", "JSONL", lambda files: load_dataset("json", data_files=files, streaming=True)),
        ("csv", "CSV", lambda files: load_dataset("csv", data_files=files, streaming=True)),
        ("arrow", "Arrow", lambda files: load_dataset("arrow", data_files=files, streaming=True)),
    ]

    for ext, name, loader in formats:
        try:
            # Find files for this format
            if ext == "json.gz":
                files = [str(f) for f in dataset_path.glob("**/*.json.gz") if f.suffix.lower() not in skip_exts]
            elif ext == "jsonl.gz":
                files = [str(f) for f in dataset_path.glob("**/*.jsonl.gz") if f.suffix.lower() not in skip_exts]
            elif ext == "json":
                files = [str(f) for f in dataset_path.glob("**/*.json") if f.suffix.lower() not in skip_exts and not f.name.endswith('.gz')]
            elif ext == "jsonl":
                files = [str(f) for f in dataset_path.glob("**/*.jsonl") if f.suffix.lower() not in skip_exts and not f.name.endswith('.gz')]
            elif ext == "parquet":
                files = [str(f) for f in dataset_path.glob("**/*.parquet") if f.suffix.lower() not in skip_exts]
            elif ext == "csv":
                files = [str(f) for f in dataset_path.glob("**/*.csv") if f.suffix.lower() not in skip_exts]
            elif ext == "arrow":
                files = [str(f) for f in dataset_path.glob("**/*.arrow") if f.suffix.lower() not in skip_exts]
            else:
                files = [str(f) for f in dataset_path.glob(f"**/*.{ext}")]

            if files:
                print(f"  Format: {name} ({len(files)} files) [STREAMING]")
                dataset = loader(files)
                # Handle DatasetDict vs IterableDataset
                if hasattr(dataset, 'keys') and 'train' in dataset:
                    return dataset['train']
                return dataset
        except Exception as e:
            # If streaming fails, try non-streaming as fallback
            try:
                if files:
                    print(f"  Streaming failed, trying memory-mapped mode...")
                    dataset = load_dataset(ext if ext != "jsonl" else "json", data_files=files)
                    if hasattr(dataset, 'keys') and 'train' in dataset:
                        return dataset['train']
                    return dataset
            except:
                pass
            continue

    return None


def _tokenize_dataset_impl(self, dataset_path: Path, output_path: Path, dataset_name: str = "dataset", skip_images: bool = True, preloaded_dataset=None) -> Tuple[bool, Optional[int]]:
    """Load, tokenize, and save dataset using true streaming"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" TOKENIZING: {dataset_name}")
    print(f"{'='*70}")
    print(f"Input:  {dataset_path}")
    print(f"Output: {output_path}")
    if skip_images:
        print(f"Image filtering: ENABLED\n")
    else:
        print(f"Image filtering: DISABLED\n")

    try:
        if preloaded_dataset is not None:
            print(f"Using prefetched dataset...")
            dataset = preloaded_dataset
        else:
            print(f"Loading dataset (streaming mode)...")
            dataset = self._load_dataset(dataset_path)

        if dataset is None:
            print(f"Could not load dataset from {dataset_path}")
            return False, None

        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
            print(f"Using 'train' split")

        # Handle both streaming (IterableDataset) and regular datasets
        is_streaming = hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__')

        if is_streaming:
            print(f"Dataset: STREAMING MODE (memory efficient)")
            # For streaming, peek at first item to get columns
            first_item = next(iter(dataset.take(1)))
            column_names = list(first_item.keys())
            print(f"  Columns: {', '.join(column_names)}")
        else:
            print(f"Dataset ready")
            column_names = dataset.column_names
            print(f"  Total columns: {len(column_names)}")
            print(f"  Column names: {', '.join(column_names)}")
            try:
                print(f"  Examples: {len(dataset):,}\n")
            except:
                print(f"  Examples: (streaming)\n")

        # Detect text columns
        if skip_images:
            image_keywords = ['image', 'img', 'picture', 'photo', 'visual', 'png', 'jpg']
            text_columns = [c for c in column_names if not any(kw in c.lower() for kw in image_keywords)]
            # Prefer known text columns
            preferred = ['text', 'content', 'data', 'input', 'instruction', 'output', 'completion']
            text_columns = [c for c in text_columns if c in preferred] or text_columns
        else:
            text_columns = column_names

        if not text_columns:
            text_columns = column_names[:1]  # Fallback to first column

        print(f"  Text columns: {', '.join(text_columns)}")
        print(f"  Mode: TRUE STREAMING\n")

        print(f"Tokenizing and saving...")
        num_shards, num_examples = self._tokenize_and_save_streaming(
            dataset, output_path, text_columns, shard_size=100000, source_path=dataset_path
        )

        print(f"\nTokenization complete!")
        print(f"  Sequences: {num_examples:,}")
        print(f"  Shards: {num_shards}")
        print(f"  Location: {output_path}\n")

        return True, num_examples

    except Exception as e:
        print(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False, None


# Attach methods to DataTokenizer class
DataTokenizer.tokenize_dataset = _tokenize_dataset_impl


def find_source_datasets(source_dir: Path) -> List[Tuple[Path, str]]:
    """Find ALL datasets in source directory and subdirectories"""
    datasets = []
    skip_dirs = {"Ava_Ai", "models", ".cache", "__pycache__", ".git", ".gitattributes"}
    skip_files = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg"}
    processed_paths = set()

    source_dir = Path(source_dir)

    # Find all directories with data files
    for path in source_dir.rglob("*"):
        if not path.is_dir():
            continue

        # Skip hidden directories and known skip dirs
        if path.name.startswith(".") or path.name in skip_dirs or path.name.endswith("-tokenized"):
            continue

        # Check if this directory has data files
        has_data = False
        for ext in ["*.parquet", "*.json", "*.jsonl", "*.csv", "*.arrow", "*.txt", "*.gz"]:
            files = list(path.glob(ext))
            # Filter out non-data files (images, etc.)
            data_files = [f for f in files if f.suffix.lower() not in skip_files]
            if data_files:
                has_data = True
                break

        if has_data and path not in processed_paths:
            # Create a meaningful name from the directory structure
            rel_path = path.relative_to(source_dir)
            dataset_name = str(rel_path).replace("/", "_").replace("\\", "_")

            datasets.append((path, dataset_name))
            processed_paths.add(path)

    return sorted(datasets, key=lambda x: x[1])


def build_pretokenized_repo(source_dir: Path, output_dir: Path, tokenizer_path: Optional[Path] = None, vocab_size: int = 50680, skip_images: bool = True, num_proc: int = None, max_length: int = 512, sample_ratio: float = 1.0):
    """Build complete pre-tokenized data repository

    Args:
        source_dir: Directory containing raw data
        output_dir: Output directory for pre-tokenized data
        tokenizer_path: Path to tokenizer (auto-created if not exists)
        vocab_size: Vocabulary size for tokenizer training
        skip_images: Whether to skip image columns during tokenization
        num_proc: Number of processes for parallel tokenization
        max_length: Fixed sequence length for padding/truncation (enables zero-copy loading)
        sample_ratio: Ratio of data to sample (0.0-1.0), e.g., 0.05 for 5%
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data subdirectory
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Use provided tokenizer path or default
    if tokenizer_path is None:
        tokenizer_name = f"vocab_{vocab_size}" if vocab_size != 50680 else "vocab"
        tokenizer_path = output_dir / "tokenizers" / tokenizer_name

    print(f"\n{'='*70}")
    print(" BUILDING PRE-TOKENIZED DATA REPOSITORY")
    print(f"{'='*70}")
    print(f"Source:      {source_dir}")
    print(f"Output:      {output_dir}")
    print(f"Tokenizer:   {tokenizer_path}")
    print(f"Data:        {data_dir}\n")

    # Setup tokenizer
    print(f"{'='*70}")
    print(" STEP 1: Training custom tokenizer")
    print(f"{'='*70}\n")

    tokenizer_path = Path(tokenizer_path)

    if tokenizer_path.exists() and (tokenizer_path / "tokenizer.json").exists():
        print(f"✓ Using existing tokenizer at: {tokenizer_path}\n")
    else:
        # Train custom tokenizer using the dedicated script
        print(f"Training custom tokenizer (vocab_size={vocab_size})...")

        # Import and call the tokenizer training function
        from train_custom_tokenizer import train_custom_bpe_tokenizer

        success = train_custom_bpe_tokenizer(
            source_dir=source_dir,
            output_path=tokenizer_path,
            vocab_size=vocab_size,
            add_eos_token=True,
            eos_token="[EOS]"
        )

        if not success:
            print(f"✗ Failed to train tokenizer\n")
            return False

    # Step 2: Find datasets
    print(f"{'='*70}")
    print(" STEP 2: Discovering datasets")
    print(f"{'='*70}\n")

    datasets = find_source_datasets(source_dir)

    if not datasets:
        print(f"No datasets found in {source_dir}\n")
        return False

    print(f"Found {len(datasets)} dataset(s):\n")
    for path, name in datasets:
        print(f"  • {name}")
    print()

    # Step 3: Tokenize datasets (true streaming - no prefetch needed)
    print(f"{'='*70}")
    print(" STEP 3: Tokenizing datasets (true streaming)")
    print(f"{'='*70}")

    # Load tokenizer once
    tokenizer = DataTokenizer(str(tokenizer_path), num_proc=num_proc, max_length=max_length, sample_ratio=sample_ratio)
    results = {}

    # Process each dataset with streaming
    for dataset_path, dataset_name in datasets:
        output_dataset_path = data_dir / dataset_name
        success, num_examples = tokenizer.tokenize_dataset(
            dataset_path,
            output_dataset_path,
            dataset_name,
            skip_images=skip_images
        )
        results[dataset_name] = (success, num_examples)

    # Step 4: Summary
    print(f"{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}\n")

    success_count = sum(1 for s, _ in results.values() if s)
    total_examples = sum(n for _, n in results.values() if n is not None)

    print(f"Tokenizers:")
    print(f"  Location: {tokenizer_path}")
    tokenizers_dir = tokenizer_path.parent
    available = TokenizerBuilder.list_available_tokenizers(tokenizers_dir)
    for tok in available:
        print(f"  ✓ {tok}")
    print()

    print(f"Datasets:")
    for dataset_name, (success, num_examples) in results.items():
        status = "✓" if success else "✗"
        examples_str = f"{num_examples:,}" if num_examples else "N/A"
        print(f"  {status} {dataset_name}: {examples_str} examples")

    print(f"\nSuccess: {success_count}/{len(results)}")
    print(f"Total examples: {total_examples:,}")
    print(f"Repository location: {output_dir}\n")

    # Save metadata with vocab size info
    metadata = {
        "created": str(Path.cwd()),
        "vocab_size": vocab_size,
        "max_length": max_length,
        "skip_images": skip_images,
        "tokenizers": available,
        "datasets": {name: num for name, (success, num) in results.items() if success},
        "total_examples": total_examples,
        "ready_for_training": success_count == len(results),
        "zero_copy_compatible": True,  # Fixed-length padding enables zero-copy loading
        "timestamp": str(__import__('datetime').datetime.now().isoformat())
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save a training-ready indicator
    if success_count == len(results):
        with open(output_dir / ".training_ready", "w") as f:
            f.write(f"vocab_size={vocab_size}\n")
            f.write(f"total_examples={total_examples}\n")
            f.write(f"datasets={success_count}\n")

    return success_count == len(results)


def find_latest_moe_config(config_dir: Path) -> Optional[Path]:
    """Find the latest MOE config file"""
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return None

    moe_configs = sorted(config_dir.glob("*.yaml"), key=lambda x: x.stat().st_mtime, reverse=True)
    return moe_configs[0] if moe_configs else None


def load_vocab_size_from_config(config_path: Path) -> Optional[int]:
    """Load vocab_size from MOE config YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        vocab_size = config.get('model', {}).get('vocab_size')
        if vocab_size:
            print(f"✓ Loaded vocab_size from config: {vocab_size}")
            return vocab_size
    except Exception as e:
        print(f"⚠ Could not load vocab_size from config: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build pre-tokenized data repository with configurable tokenizer size"
    )
    parser.add_argument("--source-dir", default="/root/Ava_AI/code/data",
                       help="Source data directory (default: /root/Ava_AI/code/data)")
    parser.add_argument("--output-dir", default="/root/Ava_AI/pretokenized_data",
                       help="Output directory (default: /root/Ava_AI/pretokenized_data)")
    parser.add_argument("--tokenizer-path",
                       default=None,
                       help="Path to tokenizer (auto-discovered if not provided)")
    parser.add_argument("--vocab-size", type=int, default=None,
                       help="Tokenizer vocabulary size (reads from config if not provided)")
    parser.add_argument("--config", type=str, default=None,
                       help="MOE config file to extract vocab_size from (e.g., code/configs/moe/large.yaml)")
    parser.add_argument("--tokenizer-name", default="vocab",
                       help="Name for tokenizer directory (default: vocab)")
    parser.add_argument("--auto-discover-config", action="store_true", default=True,
                       help="Auto-discover latest config from code/configs/moe/ (default: True)")
    parser.add_argument("--no-auto-discover-config", action="store_false", dest="auto_discover_config",
                       help="Disable auto-discovery of config")
    parser.add_argument("--skip-images", action="store_true", default=True,
                       help="Skip image columns during tokenization (default: True)")
    parser.add_argument("--include-images", action="store_true",
                       help="Include image columns (overrides --skip-images)")
    parser.add_argument("--num-proc", type=int, default=None,
                       help=f"Number of processes for parallel tokenization (default: {NUM_CPUS} - all cores)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Pad/truncate all sequences to this fixed length (default: 512). Required for fast zero-copy loading.")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                       help="Sample ratio of data to process (0.0-1.0). E.g., 0.05 for 5%% of data.")

    args = parser.parse_args()

    # Determine vocab size with priority order
    vocab_size = args.vocab_size
    config_source = None

    # Priority 1: Explicit --vocab-size argument
    if vocab_size is None:
        # Priority 2: Load from --config file
        if args.config:
            config_path = Path(args.config)
            vocab_size = load_vocab_size_from_config(config_path)
            config_source = str(config_path)

        # Priority 3: Auto-discover latest config
        if vocab_size is None and args.auto_discover_config:
            config_path = find_latest_moe_config(Path("/root/Ava_AI/code/configs/moe"))
            if config_path:
                print(f"Auto-discovered config: {config_path}")
                vocab_size = load_vocab_size_from_config(config_path)
                config_source = str(config_path)

        # Priority 4: Default value
        if vocab_size is None:
            vocab_size = 50680  # Default from large.yaml
            print(f"⚠ Using default vocab_size: {vocab_size}")

    # Setup tokenizer name and path
    tokenizer_name = f"{args.tokenizer_name}_{vocab_size}" if vocab_size != 50680 else args.tokenizer_name

    if args.tokenizer_path is None:
        args.tokenizer_path = f"/root/Ava_AI/pretokenized_data/tokenizers/{tokenizer_name}"

    # Determine image filtering
    skip_images = True
    if args.include_images:
        skip_images = False
    elif hasattr(args, 'skip_images') and not args.skip_images:
        skip_images = False

    print(f"\n{'='*70}")
    print(" CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"Source directory:    {args.source_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Tokenizer path:      {args.tokenizer_path}")
    print(f"Vocabulary size:     {vocab_size}")
    print(f"Image filtering:     {'ENABLED ⊘' if skip_images else 'DISABLED'}")
    print(f"Num processes:       {args.num_proc if args.num_proc else NUM_CPUS} (all cores)")
    print(f"Max sequence length: {args.max_length} (fixed-length padding for zero-copy loading)")
    if args.sample_ratio < 1.0:
        print(f"Sample ratio:        {args.sample_ratio*100:.0f}% of data")
    if config_source:
        print(f"Config source:       {config_source}")
    print(f"{'='*70}\n")

    try:
        success = build_pretokenized_repo(
            Path(args.source_dir),
            Path(args.output_dir),
            tokenizer_path=Path(args.tokenizer_path),
            vocab_size=vocab_size,
            skip_images=skip_images,
            num_proc=args.num_proc,
            max_length=args.max_length,
            sample_ratio=args.sample_ratio
        )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠ Build cancelled by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
