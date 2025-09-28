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
from typing import List, Dict, Optional, Iterator, Set, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import glob
import sys
import logging
import hashlib
import unicodedata
import statistics
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for importing our modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.Ava.data.deduplication import global_deduplicator

# Try to import optional acceleration libraries
PANDAS_AVAILABLE = False
NUMPY_AVAILABLE = False
DATASETS_AVAILABLE = False
TQDM_AVAILABLE = False
RAPIDS_AVAILABLE = False
LANGDETECT_AVAILABLE = False
CHARDET_AVAILABLE = False
FTFY_AVAILABLE = False

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

# Additional quality enhancement libraries
try:
    import langdetect
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
    print("✓ Language detection available")
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
    print("✓ Character encoding detection available")
except ImportError:
    CHARDET_AVAILABLE = False

try:
    import ftfy
    FTFY_AVAILABLE = True
    print("✓ Text fixing library available")
except ImportError:
    FTFY_AVAILABLE = False


class DatasetQualityAnalyzer:
    """Advanced dataset quality analysis and fixing capabilities."""

    def __init__(self):
        self.quality_metrics = {
            'encoding_errors': 0,
            'malformed_json': 0,
            'empty_content': 0,
            'non_text_content': 0,
            'language_mismatches': 0,
            'low_quality_content': 0,
            'duplicate_content': 0,
            'total_samples': 0
        }

        # Content quality filters
        self.min_text_length = 10
        self.max_text_length = 100000
        self.min_word_count = 3
        self.max_repetition_ratio = 0.7
        self.target_languages = {'en', 'english'}

        # Harmful content patterns (basic safety check)
        self.harmful_patterns = [
            r'\b(?:kill|murder|suicide|bomb|terrorist)\b',
            r'\b(?:hate|racist|nazi|fascist)\b',
            r'\b(?:drug|cocaine|heroin|meth)\b',
        ]

    def detect_encoding(self, raw_bytes: bytes) -> str:
        """Detect character encoding of raw bytes."""
        if not CHARDET_AVAILABLE:
            return 'utf-8'

        try:
            result = chardet.detect(raw_bytes)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)

            # Only trust high-confidence detections
            if confidence > 0.8:
                return encoding
            else:
                return 'utf-8'
        except Exception:
            return 'utf-8'

    def fix_text_encoding(self, text: str) -> str:
        """Fix common text encoding issues."""
        if not isinstance(text, str):
            return str(text)

        try:
            # Use ftfy if available for advanced text fixing
            if FTFY_AVAILABLE:
                text = ftfy.fix_text(text)

            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)

            # Remove null bytes and other problematic characters
            text = text.replace('\x00', '').replace('\ufffd', '')

            # Fix common encoding mistakes
            replacements = {
                'â€™': "'",
                'â€œ': '"',
                'â€\x9d': '"',
                'â€"': '–',
                'â€"': '—',
                'Ã¡': 'á',
                'Ã©': 'é',
                'Ã­': 'í',
                'Ã³': 'ó',
                'Ãº': 'ú',
            }

            for wrong, right in replacements.items():
                text = text.replace(wrong, right)

            return text

        except Exception as e:
            logging.warning(f"Text encoding fix failed: {e}")
            return text

    def detect_language(self, text: str) -> str:
        """Detect the language of text content."""
        if not LANGDETECT_AVAILABLE or not text or len(text) < 20:
            return 'unknown'

        try:
            # Use only first 1000 chars for efficiency
            sample_text = text[:1000]
            detected_lang = detect(sample_text)
            return detected_lang
        except (LangDetectError, Exception):
            return 'unknown'

    def calculate_text_quality_score(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Calculate quality score for text content (0-1, higher is better)."""
        if not text or not isinstance(text, str):
            return 0.0, {"reason": "empty_or_invalid"}

        text = text.strip()
        if len(text) < self.min_text_length:
            return 0.0, {"reason": "too_short", "length": len(text)}

        if len(text) > self.max_text_length:
            return 0.0, {"reason": "too_long", "length": len(text)}

        words = text.split()
        if len(words) < self.min_word_count:
            return 0.0, {"reason": "too_few_words", "word_count": len(words)}

        # Calculate various quality metrics
        metrics = {}
        quality_score = 1.0

        # 1. Check repetition ratio
        word_counts = Counter(words)
        total_words = len(words)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
        repetition_ratio = most_common_count / total_words if total_words > 0 else 0
        metrics['repetition_ratio'] = repetition_ratio

        if repetition_ratio > self.max_repetition_ratio:
            quality_score *= 0.3  # Heavy penalty for repetitive content

        # 2. Check character variety
        unique_chars = len(set(text.lower()))
        char_variety = unique_chars / len(text) if len(text) > 0 else 0
        metrics['char_variety'] = char_variety

        if char_variety < 0.02:  # Very low character variety
            quality_score *= 0.5

        # 3. Check for excessive non-alphabetic characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text) if len(text) > 0 else 0
        metrics['alpha_ratio'] = alpha_ratio

        if alpha_ratio < 0.5:  # Less than 50% alphabetic characters
            quality_score *= 0.7

        # 4. Check for excessive whitespace
        whitespace_ratio = sum(1 for c in text if c.isspace()) / len(text) if len(text) > 0 else 0
        metrics['whitespace_ratio'] = whitespace_ratio

        if whitespace_ratio > 0.3:  # More than 30% whitespace
            quality_score *= 0.8

        # 5. Check for potentially harmful content
        harmful_detected = any(re.search(pattern, text.lower()) for pattern in self.harmful_patterns)
        metrics['harmful_content'] = harmful_detected

        if harmful_detected:
            quality_score *= 0.1  # Heavy penalty for harmful content

        # 6. Language detection
        if LANGDETECT_AVAILABLE:
            detected_lang = self.detect_language(text)
            metrics['detected_language'] = detected_lang

            if detected_lang not in self.target_languages and detected_lang != 'unknown':
                quality_score *= 0.6  # Penalty for non-target languages

        # 7. Check for common low-quality patterns
        low_quality_patterns = [
            r'^.{1,3}$',  # Very short content
            r'^\d+$',     # Only numbers
            r'^[^\w\s]+$', # Only special characters
            r'(.)\1{10,}', # Long character repetitions
        ]

        for pattern in low_quality_patterns:
            if re.search(pattern, text, re.MULTILINE):
                quality_score *= 0.4
                break

        metrics['overall_score'] = quality_score
        return quality_score, metrics

    def filter_and_fix_sample(self, sample: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Filter and fix a single data sample."""
        fixing_log = {
            'original_keys': list(sample.keys()) if isinstance(sample, dict) else [],
            'encoding_fixed': False,
            'text_cleaned': False,
            'quality_score': 0.0,
            'rejected_reason': None
        }

        try:
            if not isinstance(sample, dict):
                fixing_log['rejected_reason'] = 'not_dict'
                self.quality_metrics['malformed_json'] += 1
                return None, fixing_log

            # Extract text content from various possible fields
            text_fields = ['text', 'content', 'output', 'response', 'completion',
                          'answer', 'dialogue', 'conversation', 'message', 'instruction']

            extracted_text = ""
            for field in text_fields:
                if field in sample and sample[field]:
                    if isinstance(sample[field], str):
                        extracted_text = sample[field]
                        break
                    elif isinstance(sample[field], (list, dict)):
                        # Try to convert complex structures to text
                        extracted_text = str(sample[field])
                        break

            if not extracted_text:
                # Fallback: concatenate all string values
                text_parts = []
                for key, value in sample.items():
                    if isinstance(value, str) and len(value.strip()) > 5:
                        text_parts.append(value.strip())
                extracted_text = " ".join(text_parts)

            if not extracted_text or len(extracted_text.strip()) < self.min_text_length:
                fixing_log['rejected_reason'] = 'empty_content'
                self.quality_metrics['empty_content'] += 1
                return None, fixing_log

            # Fix text encoding issues
            original_text = extracted_text
            fixed_text = self.fix_text_encoding(extracted_text)
            if fixed_text != original_text:
                fixing_log['encoding_fixed'] = True
                self.quality_metrics['encoding_errors'] += 1

            # Calculate quality score
            quality_score, quality_metrics = self.calculate_text_quality_score(fixed_text)
            fixing_log.update(quality_metrics)
            fixing_log['quality_score'] = quality_score

            # Filter based on quality score
            if quality_score < 0.3:  # Threshold for acceptable quality
                fixing_log['rejected_reason'] = f"low_quality_score_{quality_score:.2f}"
                self.quality_metrics['low_quality_content'] += 1
                return None, fixing_log

            # Create cleaned sample
            cleaned_sample = {
                'text': fixed_text,
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'source_format': self._detect_source_format(sample)
            }

            # Preserve important metadata
            metadata_fields = ['source', 'dataset', 'id', 'timestamp', 'language', 'domain']
            for field in metadata_fields:
                if field in sample:
                    cleaned_sample[field] = sample[field]

            fixing_log['text_cleaned'] = True
            return cleaned_sample, fixing_log

        except Exception as e:
            fixing_log['rejected_reason'] = f'processing_error_{str(e)}'
            logging.warning(f"Sample processing failed: {e}")
            return None, fixing_log

    def _detect_source_format(self, sample: Dict[str, Any]) -> str:
        """Detect the likely source format of a data sample."""
        if 'instruction' in sample and 'response' in sample:
            return 'instruction_following'
        elif 'question' in sample and 'answer' in sample:
            return 'qa_pair'
        elif 'input' in sample and 'output' in sample:
            return 'input_output'
        elif 'prompt' in sample and 'completion' in sample:
            return 'prompt_completion'
        elif 'messages' in sample:
            return 'conversation'
        elif 'chosen' in sample and 'rejected' in sample:
            return 'preference'
        elif any(key in sample for key in ['code', 'solution', 'programming_language']):
            return 'code'
        else:
            return 'general_text'

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        total = self.quality_metrics['total_samples']
        if total == 0:
            return {"error": "No samples processed"}

        return {
            'total_samples_processed': total,
            'quality_issues': {
                'encoding_errors': {
                    'count': self.quality_metrics['encoding_errors'],
                    'percentage': self.quality_metrics['encoding_errors'] / total * 100
                },
                'malformed_json': {
                    'count': self.quality_metrics['malformed_json'],
                    'percentage': self.quality_metrics['malformed_json'] / total * 100
                },
                'empty_content': {
                    'count': self.quality_metrics['empty_content'],
                    'percentage': self.quality_metrics['empty_content'] / total * 100
                },
                'low_quality_content': {
                    'count': self.quality_metrics['low_quality_content'],
                    'percentage': self.quality_metrics['low_quality_content'] / total * 100
                }
            },
            'quality_filters': {
                'min_text_length': self.min_text_length,
                'max_text_length': self.max_text_length,
                'min_word_count': self.min_word_count,
                'max_repetition_ratio': self.max_repetition_ratio,
                'target_languages': list(self.target_languages)
            }
        }


class LocalDataProcessor:
    """Enhanced data processor with multi-column format understanding"""

    def __init__(self, output_dir: str = "/project/code/processed",
                 use_gpu: bool = False, use_multiprocessing: bool = True,
                 format_strategy: str = "auto", quality_threshold: float = 0.3,
                 enable_quality_filtering: bool = True, initial_batch_size: int = 50000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.use_multiprocessing = use_multiprocessing
        self.cpu_count = mp.cpu_count()
        self.format_strategy = format_strategy
        self.quality_threshold = quality_threshold
        self.enable_quality_filtering = enable_quality_filtering

        # Adaptive batch sizing for GPU
        self.current_batch_size = initial_batch_size
        self.min_batch_size = 10000
        self.max_batch_size = 500000
        self.batch_size_history = []
        self.gpu_utilization_target = 0.85  # Target 85% GPU utilization

        # Initialize quality analyzer
        self.quality_analyzer = DatasetQualityAnalyzer()
        if quality_threshold != 0.3:
            # Adjust quality filters if custom threshold provided
            self.quality_analyzer.min_text_length = max(10, int(quality_threshold * 30))

        # Enhanced processing statistics
        self.processing_stats = {
            'total_samples_processed': 0,
            'samples_accepted': 0,
            'samples_rejected': 0,
            'encoding_fixes': 0,
            'quality_improvements': 0,
            'rejection_reasons': defaultdict(int)
        }

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

        if self.use_gpu:
            print(f"⚡ Adaptive batch sizing enabled (target: {self.gpu_utilization_target:.0%} GPU utilization)")

    def get_gpu_utilization(self) -> float:
        """Get current GPU memory utilization (0.0 to 1.0)"""
        if not self.use_gpu or cp is None:
            return 0.0

        try:
            # Get GPU memory info
            mempool = cp.get_default_memory_pool()
            total_bytes = mempool.total_bytes()
            used_bytes = mempool.used_bytes()

            if total_bytes > 0:
                return used_bytes / total_bytes
            return 0.0
        except Exception:
            return 0.0

    def adjust_batch_size(self, processing_time: float, sample_count: int):
        """Dynamically adjust batch size based on GPU utilization and processing time"""
        if not self.use_gpu:
            return

        gpu_util = self.get_gpu_utilization()

        # Track batch performance
        self.batch_size_history.append({
            'batch_size': self.current_batch_size,
            'processing_time': processing_time,
            'samples': sample_count,
            'gpu_utilization': gpu_util,
            'throughput': sample_count / processing_time if processing_time > 0 else 0
        })

        # Only adjust after we have some data
        if len(self.batch_size_history) < 2:
            return

        # If GPU utilization is too low, increase batch size
        if gpu_util < self.gpu_utilization_target and self.current_batch_size < self.max_batch_size:
            new_batch_size = int(self.current_batch_size * 1.5)
            new_batch_size = min(new_batch_size, self.max_batch_size)

            if new_batch_size != self.current_batch_size:
                old_size = self.current_batch_size
                self.current_batch_size = new_batch_size
                print(f"    ⚡ GPU underutilized ({gpu_util:.1%}), increasing batch size: {old_size:,} → {new_batch_size:,}")

        # If GPU utilization is too high (risk of OOM), decrease batch size
        elif gpu_util > 0.95 and self.current_batch_size > self.min_batch_size:
            new_batch_size = int(self.current_batch_size * 0.7)
            new_batch_size = max(new_batch_size, self.min_batch_size)

            if new_batch_size != self.current_batch_size:
                old_size = self.current_batch_size
                self.current_batch_size = new_batch_size
                print(f"    ⚠️  GPU near capacity ({gpu_util:.1%}), reducing batch size: {old_size:,} → {new_batch_size:,}")

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
        """Load data from JSON/JSONL files with enhanced error handling"""
        data = []
        encoding_attempts = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']

        for encoding in encoding_attempts:
            try:
                # Try to detect encoding from raw bytes first
                if CHARDET_AVAILABLE:
                    with open(file_path, 'rb') as f:
                        raw_sample = f.read(10000)  # Read first 10KB for detection
                        detected_encoding = self.quality_analyzer.detect_encoding(raw_sample)
                        if detected_encoding and detected_encoding not in encoding_attempts:
                            encoding_attempts.insert(0, detected_encoding)

                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    if file_path.endswith('.jsonl'):
                        # JSONL format - one JSON object per line with robust parsing
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line or line.startswith('#'):  # Skip empty lines and comments
                                continue

                            try:
                                parsed_line = json.loads(line)
                                if isinstance(parsed_line, dict):
                                    data.append(parsed_line)
                                else:
                                    # Convert non-dict to dict
                                    data.append({'text': str(parsed_line)})
                            except json.JSONDecodeError as e:
                                # Try to fix common JSON issues
                                fixed_line = self._fix_malformed_json(line)
                                if fixed_line:
                                    try:
                                        parsed_line = json.loads(fixed_line)
                                        if isinstance(parsed_line, dict):
                                            data.append(parsed_line)
                                    except json.JSONDecodeError:
                                        print(f"    ⚠️ Skipping malformed line {line_num} in {Path(file_path).name}")
                                        self.processing_stats['rejection_reasons']['malformed_json'] += 1
                    else:
                        # Regular JSON format with error recovery
                        try:
                            content = f.read()
                            # Try to fix common issues
                            fixed_content = self._fix_malformed_json(content)
                            parsed_data = json.loads(fixed_content or content)

                            if isinstance(parsed_data, list):
                                data = parsed_data
                            elif isinstance(parsed_data, dict):
                                data = [parsed_data]
                            else:
                                data = [{'text': str(parsed_data)}]

                        except json.JSONDecodeError as e:
                            print(f"    ⚠️ Failed to parse JSON in {Path(file_path).name}: {e}")
                            self.processing_stats['rejection_reasons']['malformed_json'] += 1
                            return []

                print(f"      ✓ Loaded {len(data)} samples from {Path(file_path).name} (encoding: {encoding})")
                return data

            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                print(f"    ❌ Failed to load {Path(file_path).name} with {encoding}: {e}")
                continue

        print(f"    ❌ Failed to load {Path(file_path).name} with any encoding")
        return []

    def _fix_malformed_json(self, content: str) -> Optional[str]:
        """Attempt to fix common JSON formatting issues"""
        if not content or not isinstance(content, str):
            return None

        try:
            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]

            # Fix common issues
            fixes = [
                # Fix unescaped quotes
                (r'(?<!\\)"(?=[^,}\]]*[,}\]])', r'\\"'),
                # Fix trailing commas
                (r',(\s*[}\]])', r'\1'),
                # Fix missing quotes around keys
                (r'(\w+)(\s*:\s*)', r'"\1"\2'),
                # Fix single quotes
                (r"'([^']*)'(\s*:\s*)", r'"\1"\2'),
            ]

            for pattern, replacement in fixes:
                content = re.sub(pattern, replacement, content)

            # Validate by attempting to parse
            json.loads(content)
            return content

        except (json.JSONDecodeError, Exception):
            return None

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
        """Detect the format of a dataset based on multiple samples with confidence scoring"""
        if not samples:
            return "unknown"

        # Sample up to 10 items for format detection (not just the first one)
        sample_size = min(10, len(samples))
        test_samples = samples[:sample_size] if sample_size <= 10 else random.sample(samples, 10)

        # Score each format type based on how many samples match
        format_scores = {
            "instruction_response": 0,
            "qa_context": 0,
            "conversation": 0,
            "preference": 0,
            "code_instruction": 0,
            "simple_text": 0
        }

        for sample in test_samples:
            # Check for instruction-response format (Alpaca/Dolly style)
            if isinstance(sample, dict) and 'instruction' in sample and 'response' in sample:
                format_scores["instruction_response"] += 1

            # Check for QA with context format
            elif isinstance(sample, dict) and 'question' in sample and 'answer' in sample and 'context' in sample:
                format_scores["qa_context"] += 1

            # Check for conversation format (OpenAssistant style)
            elif isinstance(sample, dict) and ('messages' in sample or ('input' in sample and 'output' in sample)):
                format_scores["conversation"] += 1

            # Check for preference learning format
            elif isinstance(sample, dict) and 'chosen' in sample and 'rejected' in sample:
                format_scores["preference"] += 1

            # Check for code instruction format
            elif isinstance(sample, dict) and any(key in sample for key in ['code', 'solution', 'programming_language']):
                format_scores["code_instruction"] += 1

            # Default to simple text
            else:
                format_scores["simple_text"] += 1

        # Find the format with the highest score
        best_format = max(format_scores.items(), key=lambda x: x[1])
        best_format_name, best_score = best_format

        # Calculate confidence (percentage of samples that match)
        confidence = best_score / len(test_samples)

        # Only return the detected format if confidence is above threshold
        if confidence >= 0.5:  # At least 50% of samples match
            return best_format_name
        else:
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
        """Extract and format text from samples with enhanced quality filtering"""
        if not samples:
            return []

        # Detect the dataset format only once per file (cache it)
        if not hasattr(self, '_cached_format_type'):
            self._cached_format_type = {}

        # Use cached format or detect once
        cache_key = id(samples[0]) if samples else 'default'  # Simple cache based on first sample identity
        if cache_key not in self._cached_format_type:
            format_type = self.detect_dataset_format(samples[:10]) if self.format_strategy == "auto" else self.format_strategy
            self._cached_format_type[cache_key] = format_type
            print(f"    📋 Detected format: {format_type}")
        else:
            format_type = self._cached_format_type[cache_key]

        if self.use_gpu and len(samples) > 1000:
            # Use GPU for large batches
            try:
                return self._extract_text_gpu_enhanced(samples, format_type)
            except Exception as e:
                print(f"    GPU processing failed, falling back to CPU: {e}")

        # Enhanced CPU processing with quality filtering
        extracted_texts = []
        quality_logs = []

        for sample in samples:
            self.processing_stats['total_samples_processed'] += 1
            self.quality_analyzer.quality_metrics['total_samples'] += 1

            try:
                if self.enable_quality_filtering:
                    # Use quality analyzer for filtering and fixing
                    cleaned_sample, fixing_log = self.quality_analyzer.filter_and_fix_sample(sample)

                    if cleaned_sample is not None:
                        extracted_texts.append(cleaned_sample['text'])
                        self.processing_stats['samples_accepted'] += 1
                        if fixing_log.get('encoding_fixed'):
                            self.processing_stats['encoding_fixes'] += 1
                        quality_logs.append(fixing_log)
                    else:
                        self.processing_stats['samples_rejected'] += 1
                        rejection_reason = fixing_log.get('rejected_reason', 'unknown')
                        self.processing_stats['rejection_reasons'][rejection_reason] += 1

                else:
                    # Original processing without quality filtering
                    if format_type != "simple_text":
                        formatted_text = self.format_multi_column_sample(sample, format_type)
                    else:
                        formatted_text = self.extract_simple_text(sample)

                    if formatted_text and len(formatted_text.strip()) > 10:
                        extracted_texts.append(formatted_text.strip())
                        self.processing_stats['samples_accepted'] += 1

            except Exception as e:
                # Fallback to simple extraction on error
                try:
                    simple_text = self.extract_simple_text(sample)
                    if simple_text and len(simple_text.strip()) > 10:
                        extracted_texts.append(simple_text.strip())
                        self.processing_stats['samples_accepted'] += 1
                except Exception:
                    self.processing_stats['samples_rejected'] += 1
                    self.processing_stats['rejection_reasons']['extraction_error'] += 1

        # Log quality statistics if enabled (only for large batches to reduce spam)
        if self.enable_quality_filtering and len(quality_logs) > 0 and len(samples) >= 10000:
            avg_quality_score = statistics.mean([log.get('quality_score', 0) for log in quality_logs if log.get('quality_score', 0) > 0])
            print(f"    📊 Quality filtering: {len(extracted_texts)}/{len(samples)} samples accepted (avg score: {avg_quality_score:.2f})")

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

    def _extract_text_gpu_enhanced(self, samples: List[Dict], format_type: str) -> List[str]:
        """Enhanced GPU-accelerated text extraction with quality filtering"""
        if not self.use_gpu or cudf is None:
            return []

        try:
            # Use the original GPU extraction
            raw_texts = self._extract_text_gpu(samples, format_type)

            if not self.enable_quality_filtering:
                return raw_texts

            # Apply quality filtering to GPU-extracted texts
            filtered_texts = []
            for i, text in enumerate(raw_texts):
                self.processing_stats['total_samples_processed'] += 1
                self.quality_analyzer.quality_metrics['total_samples'] += 1

                # Create a dummy sample for quality analysis
                dummy_sample = {'text': text}
                cleaned_sample, fixing_log = self.quality_analyzer.filter_and_fix_sample(dummy_sample)

                if cleaned_sample is not None:
                    filtered_texts.append(cleaned_sample['text'])
                    self.processing_stats['samples_accepted'] += 1
                    if fixing_log.get('encoding_fixed'):
                        self.processing_stats['encoding_fixes'] += 1
                else:
                    self.processing_stats['samples_rejected'] += 1
                    rejection_reason = fixing_log.get('rejected_reason', 'unknown')
                    self.processing_stats['rejection_reasons'][rejection_reason] += 1

            # Only log for large batches to reduce spam
            if len(raw_texts) >= 10000:
                print(f"    🚀 GPU batch: {len(filtered_texts)}/{len(raw_texts)} samples accepted")
            return filtered_texts

        except Exception as e:
            print(f"    Enhanced GPU text extraction failed: {e}")
            return []

    def process_dataset_batch(self, dataset_path: str, batch_size: int = None) -> Iterator[List[str]]:
        """Process dataset in batches - supports multiple formats"""
        dataset_path = Path(dataset_path)

        # Use adaptive batch size if not specified
        if batch_size is None:
            batch_size = self.current_batch_size

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

                # Convert to list for global deduplication
                text_list = cleaned.to_pandas().tolist()

                # Apply global cross-batch deduplication using bloom filter
                unique_texts = global_deduplicator.process_batch(text_list)

                return unique_texts

            except Exception as e:
                print(f"GPU text cleaning failed, using CPU: {e}")

        # CPU fallback using pandas for efficiency
        try:
            df = pd.DataFrame({'text': texts})

            # Remove excessive whitespace
            df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

            # Remove very short/long texts
            df = df[(df['text'].str.len() >= 10) & (df['text'].str.len() <= 10000)]

            # Apply global cross-batch deduplication using bloom filter
            text_list = df['text'].tolist()
            unique_texts = global_deduplicator.process_batch(text_list)

            return unique_texts

        except Exception:
            # Basic fallback with global deduplication
            cleaned = []

            for text in texts:
                # Basic cleaning
                text = re.sub(r'\s+', ' ', text.strip())

                # Filter by length
                if 10 <= len(text) <= 10000:
                    cleaned.append(text)

            # Apply global deduplication even in fallback mode
            unique_texts = global_deduplicator.process_batch(cleaned)

            return unique_texts

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

                # Process in GPU-accelerated batches with adaptive sizing
                for text_batch in self.process_dataset_batch(dataset_path):
                    # Time the processing for adaptive batch sizing
                    import time
                    batch_start = time.time()

                    # GPU text cleaning
                    cleaned_batch = self.clean_text_gpu(text_batch)
                    dataset_texts.extend(cleaned_batch)
                    sample_count += len(cleaned_batch)

                    # Adjust batch size based on GPU utilization
                    batch_time = time.time() - batch_start
                    self.adjust_batch_size(batch_time, len(cleaned_batch))

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

        # Generate quality report
        quality_report = self.quality_analyzer.get_quality_report() if self.enable_quality_filtering else {}

        # Save comprehensive statistics with quality metrics
        stats_summary = {
            'processing_timestamp': time.time(),
            'processing_stats': processing_stats,
            'enhanced_processing_stats': dict(self.processing_stats),
            'dataset_stats': dataset_stats,
            'overall_stats': overall_stats,
            'quality_report': quality_report,
            'quality_filtering_enabled': self.enable_quality_filtering,
            'quality_threshold': self.quality_threshold,
            'gpu_accelerated': self.use_gpu,
            'libraries_available': {
                'pandas': PANDAS_AVAILABLE,
                'rapids': RAPIDS_AVAILABLE,
                'langdetect': LANGDETECT_AVAILABLE,
                'chardet': CHARDET_AVAILABLE,
                'ftfy': FTFY_AVAILABLE
            },
            'output_files': {
                'combined': str(combined_file),
                'individual_datasets': [str(self.output_dir / f"{name}_processed.jsonl")
                                      for name in dataset_stats.keys()]
            }
        }

        with open(self.output_dir / "processing_stats.json", 'w') as f:
            json.dump(stats_summary, f, indent=2)

        # Print summary with quality metrics
        print(f"\n🎉 Processing Complete!")
        print(f"  📊 Datasets processed: {processing_stats['datasets_processed']}")
        print(f"  📝 Total samples: {processing_stats['total_samples']:,}")
        print(f"  📚 Total texts: {overall_stats.get('total_texts', 0):,}")
        print(f"  📏 Avg text length: {overall_stats.get('avg_length', 0):.1f} chars")
        print(f"  💬 Total words: {overall_stats.get('total_words', 0):,}")
        print(f"  🚀 GPU acceleration: {'✓' if self.use_gpu else '✗'}")

        # Adaptive batch sizing summary
        if self.use_gpu and len(self.batch_size_history) > 0:
            print(f"\n⚡ Adaptive Batch Sizing Summary:")
            print(f"  Initial batch size: {self.batch_size_history[0]['batch_size']:,}")
            print(f"  Final batch size: {self.batch_size_history[-1]['batch_size']:,}")
            avg_gpu_util = statistics.mean([b['gpu_utilization'] for b in self.batch_size_history])
            print(f"  Average GPU utilization: {avg_gpu_util:.1%}")
            max_throughput = max([b['throughput'] for b in self.batch_size_history])
            print(f"  Peak throughput: {max_throughput:,.0f} samples/sec")

        # Quality filtering summary
        if self.enable_quality_filtering:
            print(f"\n📋 Quality Filtering Summary:")
            print(f"  ✅ Samples accepted: {self.processing_stats['samples_accepted']:,}")
            print(f"  ❌ Samples rejected: {self.processing_stats['samples_rejected']:,}")
            print(f"  🔧 Encoding fixes: {self.processing_stats['encoding_fixes']:,}")
            if self.processing_stats['samples_accepted'] + self.processing_stats['samples_rejected'] > 0:
                acceptance_rate = self.processing_stats['samples_accepted'] / (self.processing_stats['samples_accepted'] + self.processing_stats['samples_rejected'])
                print(f"  📈 Acceptance rate: {acceptance_rate:.1%}")

            # Top rejection reasons
            if self.processing_stats['rejection_reasons']:
                print(f"  📊 Top rejection reasons:")
                sorted_reasons = sorted(self.processing_stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_reasons[:5]:
                    print(f"    • {reason}: {count:,}")

        print(f"\n📂 Output directory: {self.output_dir}")

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

                # Process in GPU-accelerated batches with adaptive sizing
                for text_batch in self.process_dataset_batch(dataset_path):
                    if not text_batch:
                        continue

                    # Apply per-dataset sample limit
                    if max_samples_per_dataset and sample_count >= max_samples_per_dataset:
                        break

                    # Time the processing for adaptive batch sizing
                    import time
                    batch_start = time.time()

                    # Clean and process texts
                    processed_texts = self.clean_text_gpu(text_batch)
                    valid_texts = [t for t in processed_texts if t and len(t.strip()) > 50]

                    dataset_texts.extend(valid_texts)
                    sample_count += len(valid_texts)

                    # Adjust batch size based on GPU utilization
                    batch_time = time.time() - batch_start
                    self.adjust_batch_size(batch_time, len(valid_texts))

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

        # Generate quality report
        quality_report = self.quality_analyzer.get_quality_report() if self.enable_quality_filtering else {}

        # Save comprehensive statistics with quality metrics
        stats_summary = {
            'processing_timestamp': time.time(),
            'processing_stats': processing_stats,
            'enhanced_processing_stats': dict(self.processing_stats),
            'dataset_stats': dataset_stats,
            'overall_stats': overall_stats,
            'quality_report': quality_report,
            'quality_filtering_enabled': self.enable_quality_filtering,
            'quality_threshold': self.quality_threshold,
            'gpu_accelerated': self.use_gpu,
            'libraries_available': {
                'pandas': PANDAS_AVAILABLE,
                'rapids': RAPIDS_AVAILABLE,
                'langdetect': LANGDETECT_AVAILABLE,
                'chardet': CHARDET_AVAILABLE,
                'ftfy': FTFY_AVAILABLE
            },
            'output_files': {
                'combined': str(combined_file),
                'individual_datasets': [str(self.output_dir / f"{name}_processed.jsonl")
                                      for name in dataset_stats.keys()]
            }
        }

        with open(self.output_dir / "processing_stats.json", 'w') as f:
            json.dump(stats_summary, f, indent=2)

        # Print summary with quality metrics
        print(f"\n🎉 Processing Complete!")
        print(f"  📊 Datasets processed: {processing_stats['datasets_processed']}")
        print(f"  📝 Total samples: {processing_stats['total_samples']:,}")
        print(f"  📚 Total texts: {overall_stats.get('total_texts', 0):,}")
        print(f"  📏 Avg text length: {overall_stats.get('avg_length', 0):.1f} chars")
        print(f"  💬 Total words: {overall_stats.get('total_words', 0):,}")
        print(f"  🚀 GPU acceleration: {'✓' if self.use_gpu else '✗'}")

        # Adaptive batch sizing summary
        if self.use_gpu and len(self.batch_size_history) > 0:
            print(f"\n⚡ Adaptive Batch Sizing Summary:")
            print(f"  Initial batch size: {self.batch_size_history[0]['batch_size']:,}")
            print(f"  Final batch size: {self.batch_size_history[-1]['batch_size']:,}")
            avg_gpu_util = statistics.mean([b['gpu_utilization'] for b in self.batch_size_history])
            print(f"  Average GPU utilization: {avg_gpu_util:.1%}")
            max_throughput = max([b['throughput'] for b in self.batch_size_history])
            print(f"  Peak throughput: {max_throughput:,.0f} samples/sec")

        # Quality filtering summary
        if self.enable_quality_filtering:
            print(f"\n📋 Quality Filtering Summary:")
            print(f"  ✅ Samples accepted: {self.processing_stats['samples_accepted']:,}")
            print(f"  ❌ Samples rejected: {self.processing_stats['samples_rejected']:,}")
            print(f"  🔧 Encoding fixes: {self.processing_stats['encoding_fixes']:,}")
            if self.processing_stats['samples_accepted'] + self.processing_stats['samples_rejected'] > 0:
                acceptance_rate = self.processing_stats['samples_accepted'] / (self.processing_stats['samples_accepted'] + self.processing_stats['samples_rejected'])
                print(f"  📈 Acceptance rate: {acceptance_rate:.1%}")

            # Top rejection reasons
            if self.processing_stats['rejection_reasons']:
                print(f"  📊 Top rejection reasons:")
                sorted_reasons = sorted(self.processing_stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_reasons[:5]:
                    print(f"    • {reason}: {count:,}")

        print(f"\n📂 Output directory: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Local Data Preparation - Works Offline")
    parser.add_argument("--raw-data-dir", default="/project/code/data/raw",
                       help="Directory containing raw datasets")
    parser.add_argument("--output-dir", default="/project/code/data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum total tokens to process (e.g., 1000000000 for 1B)")
    parser.add_argument("--gpu", action="store_true", default=True,
                       help="Use GPU acceleration if available (default: True)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--no-multiprocessing", action="store_true",
                       help="Disable multiprocessing")
    parser.add_argument("--batch-size", type=int, default=50000,
                       help="Batch size for processing (default: 50000 for speed)")
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="Specific datasets to process (default: all datasets)")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    parser.add_argument("--format-strategy",
                       choices=["auto", "instruction_response", "qa_context", "conversation", "preference", "code_instruction", "simple_text"],
                       default="auto",
                       help="Data formatting strategy (default: auto-detect)")
    parser.add_argument("--quality-threshold", type=float, default=0.25,
                       help="Quality threshold for filtering (0.0-1.0, default=0.25, higher=stricter)")
    parser.add_argument("--disable-quality-filtering", action="store_true",
                       help="Disable quality filtering and encoding fixes")
    parser.add_argument("--target-languages", nargs="*", default=["en", "english"],
                       help="Target languages for content (default: English)")
    parser.add_argument("--save-quality-report", action="store_true",
                       help="Save detailed quality analysis report")
    parser.add_argument("--enable-harmful-filtering", action="store_true",
                       help="Enable harmful content filtering (disabled by default for RLHF/safety datasets)")
    parser.add_argument("--disable-harmful-filtering", action="store_true",
                       help="Explicitly disable harmful content filtering (default behavior)")

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

    # Initialize processor with enhanced quality options
    # Handle GPU flag (default True, but can be disabled with --no-gpu)
    use_gpu = args.gpu and not args.no_gpu

    processor = LocalDataProcessor(
        output_dir=args.output_dir,
        use_gpu=use_gpu,
        use_multiprocessing=not args.no_multiprocessing,
        format_strategy=args.format_strategy,
        quality_threshold=args.quality_threshold,
        enable_quality_filtering=not args.disable_quality_filtering,
        initial_batch_size=args.batch_size
    )

    # Configure quality analyzer if quality filtering is enabled
    if not args.disable_quality_filtering:
        processor.quality_analyzer.target_languages = set(args.target_languages)

        # Harmful content filtering is DISABLED by default (for RLHF/safety datasets)
        # Only enable if explicitly requested
        if args.enable_harmful_filtering and not args.disable_harmful_filtering:
            print(f"🔍 Quality filtering enabled (threshold: {args.quality_threshold})")
            print(f"🛡️  Harmful content filtering ENABLED")
            print(f"🌐 Target languages: {', '.join(args.target_languages)}")
        else:
            # Disable harmful patterns by default
            processor.quality_analyzer.harmful_patterns = []
            print(f"🔍 Quality filtering enabled (threshold: {args.quality_threshold})")
            print(f"⚡ Harmful content filtering disabled (default for RLHF/safety data)")
            print(f"🌐 Target languages: {', '.join(args.target_languages)}")
    else:
        print("⚠️  Quality filtering disabled - processing all content as-is")

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

    # Save detailed quality report if requested
    if args.save_quality_report and not args.disable_quality_filtering:
        quality_report_path = processor.output_dir / "quality_analysis_report.json"
        detailed_quality_report = {
            'analysis_timestamp': time.time(),
            'processing_parameters': {
                'quality_threshold': args.quality_threshold,
                'target_languages': args.target_languages,
                'libraries_used': {
                    'langdetect': LANGDETECT_AVAILABLE,
                    'chardet': CHARDET_AVAILABLE,
                    'ftfy': FTFY_AVAILABLE
                }
            },
            'quality_analyzer_report': processor.quality_analyzer.get_quality_report(),
            'processing_statistics': dict(processor.processing_stats),
            'recommendations': []
        }

        # Add quality recommendations
        if processor.processing_stats['samples_rejected'] > 0:
            rejection_rate = processor.processing_stats['samples_rejected'] / (
                processor.processing_stats['samples_accepted'] + processor.processing_stats['samples_rejected']
            )

            if rejection_rate > 0.5:
                detailed_quality_report['recommendations'].append(
                    "High rejection rate detected. Consider lowering quality threshold or reviewing data sources."
                )
            elif rejection_rate > 0.2:
                detailed_quality_report['recommendations'].append(
                    "Moderate rejection rate. Data quality could be improved at the source."
                )

            if processor.processing_stats['encoding_fixes'] > processor.processing_stats['samples_accepted'] * 0.1:
                detailed_quality_report['recommendations'].append(
                    "Many encoding issues detected. Consider standardizing data collection encoding."
                )

        with open(quality_report_path, 'w') as f:
            json.dump(detailed_quality_report, f, indent=2)

        print(f"\n📋 Detailed quality report saved: {quality_report_path}")

    # Print comprehensive deduplication statistics
    print("\n" + "="*60)
    print("🔍 GLOBAL DEDUPLICATION STATISTICS")
    print("="*60)

    dedup_stats = global_deduplicator.get_stats()

    print(f"📊 Total items processed: {dedup_stats['total_items_processed']:,}")
    print(f"✅ Unique items: {dedup_stats['unique_items']:,}")
    print(f"🔄 Duplicates found: {dedup_stats['duplicates_found']:,}")
    print(f"📈 Deduplication rate: {dedup_stats['deduplication_rate']:.1%}")

    print(f"\n🎯 Bloom Filter Performance:")
    print(f"   False positives: {dedup_stats['bloom_false_positives']:,}")
    print(f"   False positive rate: {dedup_stats['bloom_false_positive_rate']:.3%}")
    print(f"   Memory usage: {dedup_stats['bloom_filter']['memory_usage_mb']:.1f} MB")

    print(f"\n💾 Cache Statistics:")
    print(f"   Cache hits: {dedup_stats['cache_hits']:,}")
    print(f"   Cache evictions: {dedup_stats['cache_evictions']:,}")
    print(f"   Cache utilization: {dedup_stats['cache_utilization']:.1%}")

    if dedup_stats['deduplication_rate'] > 0.1:  # If >10% duplicates
        print(f"\n⚠️  HIGH DUPLICATE RATE DETECTED ({dedup_stats['deduplication_rate']:.1%})")
        print("   This suggests the dataset may have quality issues.")
        print("   Consider reviewing data sources for redundancy.")
    elif dedup_stats['deduplication_rate'] > 0.05:  # If >5% duplicates
        print(f"\n✅ Moderate deduplication ({dedup_stats['deduplication_rate']:.1%}) - normal for web data")
    else:
        print(f"\n🎉 Low duplication rate ({dedup_stats['deduplication_rate']:.1%}) - high quality dataset!")

    print(f"\n💡 Memory Efficiency:")
    data_processed_gb = dedup_stats['total_items_processed'] * 0.001  # Rough estimate
    memory_efficiency = data_processed_gb / max(dedup_stats['bloom_filter']['memory_usage_mb'] / 1024, 0.001)
    print(f"   Processed ~{data_processed_gb:.1f} GB of text data")
    print(f"   Using only {dedup_stats['bloom_filter']['memory_usage_mb']:.1f} MB memory")
    print(f"   Efficiency ratio: {memory_efficiency:.0f}x")

    print("="*60)

if __name__ == "__main__":
    main()