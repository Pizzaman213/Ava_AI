#!/usr/bin/env python3
"""
Enhanced Multi-Column Data Preparation Script
High-performance data processing with multi-column format understanding for LLM training
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
    # Create dummy cudf and cp modules for type checking
    cudf = None
    cp = None

class LocalDataProcessor:
    """Enhanced data processor with multi-column format understanding"""

    def __init__(self, output_dir: str = "/project/code/processed",
                 use_gpu: bool = False, use_multiprocessing: bool = True,
                 format_strategy: str = "auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.use_multiprocessing = use_multiprocessing
        self.cpu_count = mp.cpu_count()
        self.format_strategy = format_strategy

        # Multi-column format templates
        self.format_templates = {
            "instruction_response": "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}",
            "qa_context": "Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}",
            "conversation": "Human: {input}\n\nAssistant: {output}",
            "code_instruction": "# Problem: {instruction}\n# Language: {language}\n\n```{language}\n{code}\n```",
            "preference": "Prompt: {prompt}\n\nChosen: {chosen}\n\nRejected: {rejected}"
        }

        if self.use_gpu and RAPIDS_AVAILABLE:
            try:
                if cp is not None:
                    gpu_count = cp.cuda.runtime.getDeviceCount()
                    print(f"🚀 GPU acceleration enabled with {gpu_count} GPU(s)")
                else:
                    print("🚀 GPU acceleration enabled")
            except:
                print("🚀 GPU acceleration enabled")
        elif PANDAS_AVAILABLE:
            print(f"💻 Enhanced CPU processing mode with {self.cpu_count} cores")
        else:
            print(f"💻 Basic CPU processing mode with {self.cpu_count} cores")

        print(f"📝 Format strategy: {format_strategy}")

    def discover_datasets(self, raw_data_dir: str = "/project/code/data") -> List[str]:
        """Discover all available datasets (supports multiple formats)"""
        raw_path = Path(raw_data_dir)
        datasets = []

        for item in raw_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Skip the processed output directory
                if item.name == 'processed':
                    continue

                # Skip download summary file
                if item.name == 'download_summary.json':
                    continue

                # Check for HuggingFace format (arrow files)
                if any(item.glob("*/data-*.arrow")) or any(item.glob("data-*.arrow")):
                    datasets.append(str(item))
                # Check for JSON/JSONL files (including in subdirectories)
                elif (any(item.glob("*.json")) or any(item.glob("*.jsonl")) or
                      any(item.glob("*/*.json")) or any(item.glob("*/*.jsonl"))):
                    datasets.append(str(item))
                # Check for text files
                elif any(item.glob("*.txt")) or any(item.glob("*/*.txt")):
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

    def detect_dataset_format(self, samples: List[Dict]) -> str:
        """Detect the format of a dataset based on its structure"""
        if not samples:
            return "unknown"

        sample = samples[0]

        # Check for instruction-response format (Alpaca/Dolly style)
        if 'instruction' in sample and 'response' in sample:
            return "instruction_response"

        # Check for QA with context format
        if 'question' in sample and 'answer' in sample and 'context' in sample:
            return "qa_context"

        # Check for conversation format (OpenAssistant style)
        if 'messages' in sample or ('input' in sample and 'output' in sample):
            return "conversation"

        # Check for preference learning format
        if 'chosen' in sample and 'rejected' in sample:
            return "preference"

        # Check for code instruction format
        if any(key in sample for key in ['code', 'solution', 'programming_language']):
            return "code_instruction"

        # Default to simple text
        return "simple_text"

    def format_multi_column_sample(self, sample: Dict, format_type: str = None) -> str:
        """Format a sample using multi-column templates"""
        if format_type is None:
            format_type = self.detect_dataset_format([sample])

        if format_type == "instruction_response":
            return self.format_templates["instruction_response"].format(
                instruction=sample.get('instruction', ''),
                context=sample.get('context', ''),
                response=sample.get('response', '')
            )

        elif format_type == "qa_context":
            return self.format_templates["qa_context"].format(
                context=sample.get('context', ''),
                question=sample.get('question', ''),
                answer=sample.get('answer', '')
            )

        elif format_type == "conversation":
            if 'messages' in sample:
                # Handle OpenAssistant-style messages
                messages = sample.get('messages', [])
                formatted = ""
                for msg in messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if role == 'human':
                        formatted += f"Human: {content}\n\n"
                    elif role == 'assistant':
                        formatted += f"Assistant: {content}\n\n"
                return formatted.strip()
            else:
                # Simple input/output format
                return self.format_templates["conversation"].format(
                    input=sample.get('input', ''),
                    output=sample.get('output', '')
                )

        elif format_type == "preference":
            return self.format_templates["preference"].format(
                prompt=sample.get('prompt', ''),
                chosen=sample.get('chosen', ''),
                rejected=sample.get('rejected', '')
            )

        elif format_type == "code_instruction":
            language = sample.get('programming_language', sample.get('language', 'python'))
            return self.format_templates["code_instruction"].format(
                instruction=sample.get('instruction', sample.get('problem', '')),
                language=language,
                code=sample.get('code', sample.get('solution', ''))
            )

        else:
            # Fallback to simple text extraction
            return self.extract_simple_text(sample)

    def extract_simple_text(self, sample: Dict) -> str:
        """Extract text from a sample using simple heuristics"""
        # Priority text fields
        text_fields = ['text', 'content', 'output', 'response', 'completion',
                      'answer', 'dialogue', 'conversation', 'message']

        for field in text_fields:
            if field in sample and sample[field]:
                return str(sample[field]).strip()

        # Fallback: concatenate all string values
        text_parts = []
        for key, value in sample.items():
            if isinstance(value, str) and value.strip() and len(value) > 10:
                text_parts.append(value.strip())

        return " ".join(text_parts) if text_parts else ""

    def extract_text_efficiently(self, samples: List[Dict]) -> List[str]:
        """Extract and format text from samples with multi-column understanding"""
        if not samples:
            return []

        # Detect the dataset format
        format_type = self.detect_dataset_format(samples) if self.format_strategy == "auto" else self.format_strategy
        print(f"    📋 Detected format: {format_type}")

        if self.use_gpu and len(samples) > 1000:
            # Use GPU for large batches
            try:
                return self._extract_text_gpu(samples, format_type)
            except Exception as e:
                print(f"    GPU processing failed, falling back to CPU: {e}")

        # CPU processing with multi-column formatting
        extracted_texts = []
        for sample in samples:
            try:
                if format_type != "simple_text":
                    formatted_text = self.format_multi_column_sample(sample, format_type)
                else:
                    formatted_text = self.extract_simple_text(sample)

                if formatted_text and len(formatted_text.strip()) > 10:
                    extracted_texts.append(formatted_text.strip())
            except Exception as e:
                # Fallback to simple extraction on error
                simple_text = self.extract_simple_text(sample)
                if simple_text and len(simple_text.strip()) > 10:
                    extracted_texts.append(simple_text.strip())

        return extracted_texts

    def _extract_text_gpu(self, samples: List[Dict], format_type: str) -> List[str]:
        """GPU-accelerated text extraction and formatting"""
        if not self.use_gpu or cudf is None:
            return []

        try:
            df = cudf.DataFrame(samples)
            extracted_texts = []

            if format_type == "instruction_response":
                # GPU string operations for instruction-response format
                instructions = df.get('instruction', cudf.Series([''] * len(df))).fillna('')
                contexts = df.get('context', cudf.Series([''] * len(df))).fillna('')
                responses = df.get('response', cudf.Series([''] * len(df))).fillna('')

                # Format using template
                template = self.format_templates["instruction_response"]
                # Note: GPU template formatting would require custom kernels
                # For now, convert to pandas for template formatting
                for i in range(len(df)):
                    formatted = template.format(
                        instruction=instructions.iloc[i],
                        context=contexts.iloc[i],
                        response=responses.iloc[i]
                    )
                    extracted_texts.append(formatted)

            elif format_type == "qa_context":
                questions = df.get('question', cudf.Series([''] * len(df))).fillna('')
                answers = df.get('answer', cudf.Series([''] * len(df))).fillna('')
                contexts = df.get('context', cudf.Series([''] * len(df))).fillna('')

                template = self.format_templates["qa_context"]
                for i in range(len(df)):
                    formatted = template.format(
                        question=questions.iloc[i],
                        answer=answers.iloc[i],
                        context=contexts.iloc[i]
                    )
                    extracted_texts.append(formatted)

            else:
                # Fallback to simple GPU text extraction
                text_fields = ['text', 'content', 'output', 'response']
                for field in text_fields:
                    if field in df.columns:
                        valid_texts = df[field].dropna()
                        if len(valid_texts) > 0:
                            extracted_texts.extend(valid_texts.to_pandas().tolist())
                            break

            return [str(text).strip() for text in extracted_texts if str(text).strip()]

        except Exception as e:
            print(f"    GPU text extraction failed: {e}")
            return []

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

                # Find all data files (including in subdirectories)
                json_files = list(dataset_path.glob("*.json")) + list(dataset_path.glob("*/*.json"))
                jsonl_files = list(dataset_path.glob("*.jsonl")) + list(dataset_path.glob("*/*.jsonl"))
                txt_files = list(dataset_path.glob("*.txt")) + list(dataset_path.glob("*/*.txt"))
                parquet_files = list(dataset_path.glob("*.parquet")) + list(dataset_path.glob("*/*.parquet"))

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

        if self.use_gpu and len(texts) > 1000 and cudf is not None:  # Use GPU for large batches
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

        if self.use_gpu and cudf is not None:
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

    def process_all_datasets(self, raw_data_dir: str = "/project/code/data",
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

    def process_specific_datasets(self, dataset_paths: List[str],
                                max_samples_per_dataset: Optional[int] = None,
                                max_total_tokens: Optional[int] = None):
        """Process specific datasets with GPU acceleration"""

        print(f"Found {len(dataset_paths)} datasets to process")

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
        for dataset_path in tqdm(dataset_paths, desc="Processing datasets"):
            dataset_name = Path(dataset_path).name
            print(f"\n🔄 Processing: {dataset_name}")

            try:
                dataset_texts = []
                sample_count = 0

                # Process in GPU-accelerated batches
                for text_batch in self.process_dataset_batch(dataset_path, batch_size=10000):
                    if not text_batch:
                        continue

                    # Apply per-dataset sample limit
                    if max_samples_per_dataset and sample_count >= max_samples_per_dataset:
                        break

                    # Clean and process texts
                    processed_texts = self.clean_text_gpu(text_batch)
                    valid_texts = [t for t in processed_texts if t and len(t.strip()) > 50]

                    dataset_texts.extend(valid_texts)
                    sample_count += len(valid_texts)

                    # Apply global token limit
                    if max_total_tokens:
                        estimated_tokens = sum(len(t.split()) for t in valid_texts)
                        total_tokens_so_far += estimated_tokens
                        if total_tokens_so_far >= max_total_tokens:
                            print(f"  ⚡ Reached token limit ({max_total_tokens:,}), stopping")
                            break

                if dataset_texts:
                    # Compute dataset statistics
                    stats = self.compute_text_stats_gpu(dataset_texts)
                    dataset_stats[dataset_name] = stats

                    # Save individual dataset
                    output_file = self.output_dir / f"{dataset_name}_processed.jsonl"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for text in dataset_texts:
                            f.write(json.dumps({'text': text}) + '\n')

                    all_texts.extend(dataset_texts)
                    processing_stats['datasets_processed'] += 1
                    processing_stats['total_samples'] += len(dataset_texts)
                    processing_stats['total_tokens'] += stats.get('total_words', 0)

                    print(f"  ✓ Processed {len(dataset_texts):,} texts")
                    print(f"  📊 Avg length: {stats.get('avg_length', 0):.1f} chars")
                    print(f"  💬 Total words: {stats.get('total_words', 0):,}")

                    if max_total_tokens and total_tokens_so_far >= max_total_tokens:
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
        if all_texts:
            overall_stats = self.compute_text_stats_gpu(all_texts)
        else:
            overall_stats = {}

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
    parser.add_argument("--raw-data-dir", default="/project/code/data",
                       help="Directory containing raw datasets")
    parser.add_argument("--output-dir", default="/project/code/processed",
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
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="Specific datasets to process (default: all datasets)")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    parser.add_argument("--format-strategy",
                       choices=["auto", "instruction_response", "qa_context", "conversation", "preference", "code_instruction", "simple_text"],
                       default="auto",
                       help="Data formatting strategy (default: auto-detect)")

    args = parser.parse_args()

    print("🔄 Starting Local Data Preparation")
    print("="*50)

    # Initialize processor to discover datasets
    temp_processor = LocalDataProcessor()
    available_datasets = temp_processor.discover_datasets(args.raw_data_dir)

    if args.list_datasets:
        print(f"\n📋 Available datasets in {args.raw_data_dir}:")
        if available_datasets:
            for i, dataset in enumerate(available_datasets, 1):
                dataset_name = Path(dataset).name
                print(f"  {i}. {dataset_name}")
        else:
            print("  No datasets found!")
        return

    if args.datasets:
        # Process specific datasets
        selected_datasets = []
        for dataset_name in args.datasets:
            # Find matching dataset path
            matching = [d for d in available_datasets if Path(d).name == dataset_name or d.endswith(dataset_name)]
            if matching:
                selected_datasets.extend(matching)
            else:
                print(f"⚠️  Dataset '{dataset_name}' not found")

        if not selected_datasets:
            print("❌ No valid datasets selected")
            return

        print(f"\n📋 Processing {len(selected_datasets)} selected datasets:")
        for dataset in selected_datasets:
            print(f"  - {Path(dataset).name}")
    else:
        # Process all datasets (default behavior)
        selected_datasets = None
        print(f"\n📋 Processing ALL {len(available_datasets)} available datasets:")
        for dataset in available_datasets:
            print(f"  - {Path(dataset).name}")
        print("\n💡 Use --datasets <name1> <name2> to process specific datasets only")
        print("💡 Use --list-datasets to see all available datasets")

    # Initialize processor
    processor = LocalDataProcessor(
        output_dir=args.output_dir,
        use_gpu=args.gpu,
        use_multiprocessing=not args.no_multiprocessing,
        format_strategy=args.format_strategy
    )

    # Process datasets
    if args.datasets and 'selected_datasets' in locals() and selected_datasets:
        # Process specific datasets
        processor.process_specific_datasets(
            selected_datasets,
            max_samples_per_dataset=args.max_samples,
            max_total_tokens=args.max_tokens
        )
    else:
        # Process all datasets (default)
        processor.process_all_datasets(
            raw_data_dir=args.raw_data_dir,
            max_samples_per_dataset=args.max_samples,
            max_total_tokens=args.max_tokens
        )

if __name__ == "__main__":
    main()