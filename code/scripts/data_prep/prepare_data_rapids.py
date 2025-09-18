#!/usr/bin/env python3
"""
Local Data Preparation Script
High-performance data processing for LLM training - works offline with minimal dependencies
"""

import json
import time
import argparse
import re
import os
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from concurrent.futures import ProcessPoolExecutor
import glob

# Try to import optional acceleration libraries
PANDAS_AVAILABLE = False
NUMPY_AVAILABLE = False
DATASETS_AVAILABLE = False
TQDM_AVAILABLE = False
RAPIDS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    print("✓ Pandas available - enhanced CPU processing enabled")
except ImportError:
    print("⚠️ Pandas not available - using basic processing")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
    print("✓ HuggingFace datasets library available")
except ImportError:
    print("⚠️ HuggingFace datasets not available - using file-based processing")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Simple progress bar fallback
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', 'Processing')
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:
                print(f"{desc}: {i}/{total} ({i/total*100:.1f}%)")
            yield item

# Try RAPIDS only if GPU is available and not disabled
try:
    # Check if RAPIDS should be disabled
    if (os.environ.get('CUDA_VISIBLE_DEVICES') == '' or
        os.environ.get('DISABLE_RAPIDS') == '1' or
        not os.path.exists('/usr/local/cuda')):
        print("⚠️ RAPIDS disabled - using CPU processing")
        raise ImportError("RAPIDS disabled")

    import cudf
    import cupy as cp
    import rmm
    RAPIDS_AVAILABLE = True
    print("✓ RAPIDS libraries available - GPU acceleration enabled")

    # Memory pool initialization for RAPIDS
    try:
        rmm.reinitialize(
            managed_memory=False,
            devices=0,
            pool_allocator=True,
            initial_pool_size=2**29,  # 512MB initial pool
        )
    except Exception as e:
        print(f"⚠️ RAPIDS memory pool init failed: {e}")
        RAPIDS_AVAILABLE = False

except ImportError:
    RAPIDS_AVAILABLE = False

class LocalDataProcessor:
    """Local data processor - works offline with minimal dependencies"""

    def __init__(self, output_dir: str = "/project/data/pretraining/processed",
                 use_gpu: bool = False, use_multiprocessing: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.use_multiprocessing = use_multiprocessing
        self.cpu_count = mp.cpu_count()

        if self.use_gpu and RAPIDS_AVAILABLE:
            try:
                gpu_count = cp.cuda.runtime.getDeviceCount()
                print(f"🚀 GPU acceleration enabled with {gpu_count} GPU(s)")
            except:
                print("🚀 GPU acceleration enabled")
        elif PANDAS_AVAILABLE:
            print(f"💻 Enhanced CPU processing mode with {self.cpu_count} cores")
        else:
            print(f"💻 Basic CPU processing mode with {self.cpu_count} cores")

    def discover_datasets(self, raw_data_dir: str = "/project/data/pretraining/raw") -> List[str]:
        """Discover all available datasets (supports multiple formats)"""
        raw_path = Path(raw_data_dir)
        datasets = []

        for item in raw_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check for HuggingFace format
                if any(item.glob("*/data-*.arrow")) or any(item.glob("data-*.arrow")):
                    datasets.append(str(item))
                # Check for JSON/JSONL files
                elif any(item.glob("*.json")) or any(item.glob("*.jsonl")):
                    datasets.append(str(item))
                # Check for text files
                elif any(item.glob("*.txt")):
                    datasets.append(str(item))

        return sorted(datasets)

    def load_json_file(self, file_path: str) -> List[Dict]:
        """Load data from JSON/JSONL files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    # JSONL format - one JSON object per line
                    data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                    return data
                else:
                    # Regular JSON format
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return []

    def load_text_file(self, file_path: str) -> List[str]:
        """Load data from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return [line.strip() for line in lines if line.strip()]
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return []

    def extract_text_efficiently(self, samples: List[Dict]) -> List[str]:
        """Extract text from samples with GPU acceleration"""
        if not samples:
            return []

        if self.use_gpu:
            # Convert to cuDF for GPU processing
            try:
                df = cudf.DataFrame(samples)

                # Try common text fields
                text_fields = ['text', 'content', 'output', 'response', 'instruction',
                              'input', 'question', 'answer', 'dialogue', 'conversation']

                extracted_texts = []
                for field in text_fields:
                    if field in df.columns:
                        # GPU-accelerated string operations
                        valid_texts = df[field].dropna()
                        if len(valid_texts) > 0:
                            extracted_texts.extend(valid_texts.to_pandas().tolist())
                            break

                # Handle instruction-response pairs on GPU
                if not extracted_texts and 'instruction' in df.columns:
                    instructions = df['instruction'].fillna('')
                    inputs = df.get('input', cudf.Series([''] * len(df), dtype='str'))
                    outputs = df.get('output', cudf.Series([''] * len(df), dtype='str'))

                    # GPU string concatenation
                    combined = instructions + " " + inputs + " " + outputs
                    extracted_texts = combined.to_pandas().tolist()

                return [str(text).strip() for text in extracted_texts if str(text).strip()]

            except Exception as e:
                print(f"GPU processing failed, falling back to CPU: {e}")
                # Fall through to CPU processing

        # CPU fallback
        extracted_texts = []
        for sample in samples:
            text_fields = ['text', 'content', 'output', 'response', 'instruction',
                          'input', 'question', 'answer', 'dialogue', 'conversation']

            for field in text_fields:
                if field in sample and sample[field]:
                    extracted_texts.append(str(sample[field]).strip())
                    break
            else:
                # Handle instruction-response format
                if 'instruction' in sample:
                    instruction = str(sample.get('instruction', ''))
                    input_text = str(sample.get('input', ''))
                    output_text = str(sample.get('output', ''))
                    combined = f"{instruction} {input_text} {output_text}".strip()
                    if combined:
                        extracted_texts.append(combined)

        return [text for text in extracted_texts if text]

    def process_dataset_batch(self, dataset_path: str, batch_size: int = 10000) -> Iterator[List[str]]:
        """Process dataset in batches - supports multiple formats"""
        dataset_path = Path(dataset_path)

        try:
            # Check if this is a HuggingFace datasets directory
            if DATASETS_AVAILABLE and any(dataset_path.glob("*/data-*.arrow")):
                # HuggingFace format processing
                for split_dir in dataset_path.iterdir():
                    if split_dir.is_dir() and not split_dir.name.startswith('.'):
                        try:
                            print(f"    Processing HF split: {split_dir.name}")
                            split_dataset = load_from_disk(str(split_dir))

                            # Check if dataset has select method (handle different dataset types)
                            if hasattr(split_dataset, 'select') and hasattr(split_dataset, '__len__'):
                                # Process in batches for memory efficiency
                                for i in range(0, len(split_dataset), batch_size):
                                    batch = split_dataset.select(range(i, min(i + batch_size, len(split_dataset))))
                                    samples = [dict(item) for item in batch]

                                    # Text extraction
                                    texts = self.extract_text_efficiently(samples)
                                    if texts:
                                        yield texts
                            else:
                                # Handle iterable datasets
                                batch_samples = []
                                for idx, sample in enumerate(split_dataset):
                                    batch_samples.append(dict(sample))
                                    if len(batch_samples) >= batch_size:
                                        texts = self.extract_text_efficiently(batch_samples)
                                        if texts:
                                            yield texts
                                        batch_samples = []

                                # Process remaining samples
                                if batch_samples:
                                    texts = self.extract_text_efficiently(batch_samples)
                                    if texts:
                                        yield texts

                        except Exception as e:
                            print(f"    Failed to process HF split {split_dir}: {e}")
                            continue

            else:
                # File-based processing (JSON, JSONL, TXT)
                print(f"    Processing files in: {dataset_path.name}")

                # Find all data files
                json_files = list(dataset_path.glob("*.json"))
                jsonl_files = list(dataset_path.glob("*.jsonl"))
                txt_files = list(dataset_path.glob("*.txt"))
                parquet_files = list(dataset_path.glob("*.parquet"))

                all_files = json_files + jsonl_files + txt_files + parquet_files

                for file_path in all_files:
                    # Skip metadata and config files
                    if file_path.name in ['.checksums.json', 'config.json', 'metadata.json']:
                        print(f"      Skipping metadata file: {file_path.name}")
                        continue

                    print(f"      Processing file: {file_path.name}")

                    if file_path.suffix in ['.json', '.jsonl']:
                        # Load JSON data
                        data = self.load_json_file(str(file_path))

                        # Process in batches
                        for i in range(0, len(data), batch_size):
                            batch = data[i:i + batch_size]
                            texts = self.extract_text_efficiently(batch)
                            if texts:
                                yield texts

                    elif file_path.suffix == '.parquet':
                        # Load parquet data
                        if PANDAS_AVAILABLE:
                            try:
                                import pandas as pd
                                df = pd.read_parquet(str(file_path))
                                data = df.to_dict('records')

                                # Process in batches
                                for i in range(0, len(data), batch_size):
                                    batch = data[i:i + batch_size]
                                    texts = self.extract_text_efficiently(batch)
                                    if texts:
                                        yield texts
                            except Exception as e:
                                print(f"    Failed to process parquet file {file_path.name}: {e}")
                                continue
                        else:
                            print(f"    Skipping parquet file {file_path.name} - pandas not available")
                            continue

                    elif file_path.suffix == '.txt':
                        # Load text data
                        lines = self.load_text_file(str(file_path))

                        # Convert to dict format and process
                        for i in range(0, len(lines), batch_size):
                            batch = [{'text': line} for line in lines[i:i + batch_size]]
                            texts = self.extract_text_efficiently(batch)
                            if texts:
                                yield texts

        except Exception as e:
            print(f"Failed to process dataset {dataset_path}: {e}")

    def clean_text_gpu(self, texts: List[str]) -> List[str]:
        """GPU-accelerated text cleaning"""
        if not texts:
            return []

        if self.use_gpu and len(texts) > 1000:  # Use GPU for large batches
            try:
                # Convert to cuDF Series for GPU string operations
                text_series = cudf.Series(texts)

                # GPU-accelerated cleaning operations
                # Remove excessive whitespace
                cleaned = text_series.str.replace(r'\s+', ' ', regex=True)

                # Remove very short texts
                cleaned = cleaned[cleaned.str.len() >= 10]

                # Remove very long texts (potential outliers)
                cleaned = cleaned[cleaned.str.len() <= 10000]

                # Remove duplicate texts (GPU-accelerated)
                cleaned = cleaned.drop_duplicates()

                return cleaned.to_pandas().tolist()

            except Exception as e:
                print(f"GPU text cleaning failed, using CPU: {e}")

        # CPU fallback using pandas for efficiency
        try:
            df = pd.DataFrame({'text': texts})

            # Remove excessive whitespace
            df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

            # Remove very short/long texts
            df = df[(df['text'].str.len() >= 10) & (df['text'].str.len() <= 10000)]

            # Remove duplicates
            df = df.drop_duplicates(subset=['text'])

            return df['text'].tolist()

        except Exception:
            # Basic fallback
            cleaned = []
            seen = set()

            for text in texts:
                # Basic cleaning
                text = re.sub(r'\s+', ' ', text.strip())

                # Filter by length
                if 10 <= len(text) <= 10000 and text not in seen:
                    cleaned.append(text)
                    seen.add(text)

            return cleaned

    def compute_text_stats_gpu(self, texts: List[str]) -> Dict:
        """Compute statistics using GPU acceleration"""
        if not texts:
            return {}

        if self.use_gpu:
            try:
                text_series = cudf.Series(texts)

                # GPU-accelerated statistics
                stats = {
                    'total_texts': len(text_series),
                    'avg_length': float(text_series.str.len().mean()),
                    'min_length': int(text_series.str.len().min()),
                    'max_length': int(text_series.str.len().max()),
                    'total_characters': int(text_series.str.len().sum()),
                }

                # Word count estimation (GPU)
                word_counts = text_series.str.split().str.len()
                stats.update({
                    'avg_words': float(word_counts.mean()),
                    'total_words': int(word_counts.sum())
                })

                return stats

            except Exception as e:
                print(f"GPU stats computation failed, using CPU: {e}")

        # CPU fallback using pandas for efficiency
        try:
            df = pd.DataFrame({'text': texts})
            lengths = df['text'].str.len()
            word_counts = df['text'].str.split().str.len()

            return {
                'total_texts': len(df),
                'avg_length': float(lengths.mean()),
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'total_characters': int(lengths.sum()),
                'avg_words': float(word_counts.mean()),
                'total_words': int(word_counts.sum())
            }

        except Exception:
            # Basic CPU fallback
            lengths = [len(text) for text in texts]
            word_counts = [len(text.split()) for text in texts]

            return {
                'total_texts': len(texts),
                'avg_length': sum(lengths) / len(lengths) if lengths else 0,
                'min_length': min(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0,
                'total_characters': sum(lengths),
                'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
                'total_words': sum(word_counts)
            }

    def process_all_datasets(self, raw_data_dir: str = "/project/data/pretraining/raw",
                           max_samples_per_dataset: Optional[int] = None,
                           max_total_tokens: Optional[int] = None):
        """Process all datasets with GPU acceleration"""

        datasets = self.discover_datasets(raw_data_dir)
        print(f"Found {len(datasets)} datasets to process")

        all_texts = []
        dataset_stats = {}
        total_tokens_so_far = 0
        processing_stats = {
            'datasets_processed': 0,
            'total_samples': 0,
            'total_tokens': 0,
            'failed_datasets': []
        }

        # Process each dataset
        for dataset_path in tqdm(datasets, desc="Processing datasets"):
            dataset_name = Path(dataset_path).name
            print(f"\n🔄 Processing: {dataset_name}")

            try:
                dataset_texts = []
                sample_count = 0

                # Process in GPU-accelerated batches
                for text_batch in self.process_dataset_batch(dataset_path, batch_size=10000):
                    # GPU text cleaning
                    cleaned_batch = self.clean_text_gpu(text_batch)
                    dataset_texts.extend(cleaned_batch)
                    sample_count += len(cleaned_batch)

                    if max_samples_per_dataset and sample_count >= max_samples_per_dataset:
                        dataset_texts = dataset_texts[:max_samples_per_dataset]
                        break

                if dataset_texts:
                    # Compute GPU-accelerated statistics
                    stats = self.compute_text_stats_gpu(dataset_texts)
                    dataset_stats[dataset_name] = stats

                    # Estimate tokens (rough: 4 chars per token)
                    dataset_tokens = stats['total_characters'] // 4

                    # Check if adding this dataset would exceed token limit
                    if max_total_tokens and (total_tokens_so_far + dataset_tokens) > max_total_tokens:
                        # Calculate how many samples we can include
                        remaining_tokens = max_total_tokens - total_tokens_so_far
                        if remaining_tokens > 0:
                            # Take samples proportionally to stay under limit
                            ratio = remaining_tokens / dataset_tokens
                            samples_to_take = max(1, int(len(dataset_texts) * ratio))
                            dataset_texts = dataset_texts[:samples_to_take]
                            # Recalculate stats for the reduced dataset
                            stats = self.compute_text_stats_gpu(dataset_texts)
                            dataset_stats[dataset_name] = stats
                            dataset_tokens = stats['total_characters'] // 4
                            print(f"  🎯 Limited to {samples_to_take} samples to stay under {max_total_tokens:,} token limit")
                        else:
                            print(f"  🛑 Stopping processing - reached {max_total_tokens:,} token limit")
                            break

                    # Save dataset texts
                    output_file = self.output_dir / f"{dataset_name}_processed.jsonl"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for text in dataset_texts:
                            f.write(json.dumps({'text': text}) + '\n')

                    # Add to combined dataset
                    all_texts.extend(dataset_texts)
                    total_tokens_so_far += dataset_tokens

                    print(f"  ✓ Processed {len(dataset_texts)} samples (~{dataset_tokens:,} tokens)")
                    print(f"    Total tokens so far: {total_tokens_so_far:,}")
                    processing_stats['datasets_processed'] += 1
                    processing_stats['total_samples'] += len(dataset_texts)
                    processing_stats['total_tokens'] = total_tokens_so_far

                    # Stop if we've reached the token limit
                    if max_total_tokens and total_tokens_so_far >= max_total_tokens:
                        print(f"  🎯 Reached target of {max_total_tokens:,} tokens - stopping processing")
                        break

                else:
                    print(f"  ⚠️ No valid texts found in {dataset_name}")

            except Exception as e:
                print(f"  ✗ Failed to process {dataset_name}: {e}")
                processing_stats['failed_datasets'].append(dataset_name)
                continue

        # Save combined dataset
        print(f"\n💾 Saving combined dataset...")
        combined_file = self.output_dir / "combined_processed.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for text in tqdm(all_texts, desc="Writing combined dataset"):
                f.write(json.dumps({'text': text}) + '\n')

        # Compute overall statistics
        overall_stats = self.compute_text_stats_gpu(all_texts)

        # Save comprehensive statistics
        stats_summary = {
            'processing_timestamp': time.time(),
            'processing_stats': processing_stats,
            'dataset_stats': dataset_stats,
            'overall_stats': overall_stats,
            'gpu_accelerated': self.use_gpu,
            'output_files': {
                'combined': str(combined_file),
                'individual_datasets': [str(self.output_dir / f"{name}_processed.jsonl")
                                      for name in dataset_stats.keys()]
            }
        }

        with open(self.output_dir / "processing_stats.json", 'w') as f:
            json.dump(stats_summary, f, indent=2)

        # Print summary
        print(f"\n🎉 Processing Complete!")
        print(f"  📊 Datasets processed: {processing_stats['datasets_processed']}")
        print(f"  📝 Total samples: {processing_stats['total_samples']:,}")
        print(f"  📚 Total texts: {overall_stats.get('total_texts', 0):,}")
        print(f"  📏 Avg text length: {overall_stats.get('avg_length', 0):.1f} chars")
        print(f"  💬 Total words: {overall_stats.get('total_words', 0):,}")
        print(f"  🚀 GPU acceleration: {'✓' if self.use_gpu else '✗'}")
        print(f"  📂 Output directory: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Local Data Preparation - Works Offline")
    parser.add_argument("--raw-data-dir", default="/project/data/pretraining/raw",
                       help="Directory containing raw datasets")
    parser.add_argument("--output-dir", default="/project/data/pretraining/processed",
                       help="Output directory for processed data")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum total tokens to process (e.g., 1000000000 for 1B)")
    parser.add_argument("--gpu", action="store_true",
                       help="Try to use GPU acceleration (requires RAPIDS)")
    parser.add_argument("--no-multiprocessing", action="store_true",
                       help="Disable multiprocessing")
    parser.add_argument("--batch-size", type=int, default=5000,
                       help="Batch size for processing")

    args = parser.parse_args()

    print("🔄 Starting Local Data Preparation")
    print("="*50)

    # Initialize processor
    processor = LocalDataProcessor(
        output_dir=args.output_dir,
        use_gpu=args.gpu,
        use_multiprocessing=not args.no_multiprocessing
    )

    # Process all datasets
    processor.process_all_datasets(
        raw_data_dir=args.raw_data_dir,
        max_samples_per_dataset=args.max_samples,
        max_total_tokens=args.max_tokens
    )

if __name__ == "__main__":
    main()