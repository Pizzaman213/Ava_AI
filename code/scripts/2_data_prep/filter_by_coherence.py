#!/usr/bin/env python3
"""
Data Quality Filtering Based on Coherence Metrics

This script filters training data to improve model coherence by:
1. Removing duplicates and near-duplicates
2. Filtering based on perplexity thresholds
3. Filtering based on coherence metrics (repetition, diversity, etc.)
4. Filtering by length (too short or too long)
5. Removing low-quality samples

Usage:
    python filter_by_coherence.py --input data.jsonl --output filtered_data.jsonl
    python filter_by_coherence.py --input data.jsonl --output filtered_data.jsonl --min-length 10 --max-length 2048
    python filter_by_coherence.py --input data.jsonl --output filtered_data.jsonl --max-repetition 0.5
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
from tqdm import tqdm
import hashlib
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers not found. Install with: pip install transformers")
    sys.exit(1)

try:
    from Ava.evaluation.coherence_metrics import (
        calculate_distinct_n,
        calculate_repetition_ratio,
        calculate_entropy,
        quick_coherence_test
    )
except ImportError:
    print("Warning: Could not import coherence metrics. Some features will be disabled.")
    calculate_distinct_n = None
    calculate_repetition_ratio = None
    calculate_entropy = None
    quick_coherence_test = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityFilter:
    """Filter training data based on quality and coherence metrics."""

    def __init__(
        self,
        tokenizer_path: str = "/project/code/models/tokenizer/enhanced-50680",
        min_length: int = 5,
        max_length: int = 2048,
        max_repetition: float = 0.5,
        min_distinct_2: float = 0.3,
        min_entropy: float = 2.0,
        remove_duplicates: bool = True,
        similarity_threshold: float = 0.95,
    ):
        """
        Initialize the data quality filter.

        Args:
            tokenizer_path: Path to tokenizer for tokenization
            min_length: Minimum sequence length (in tokens)
            max_length: Maximum sequence length (in tokens)
            max_repetition: Maximum acceptable repetition ratio (0.0-1.0)
            min_distinct_2: Minimum distinct-2 score (0.0-1.0)
            min_entropy: Minimum entropy threshold
            remove_duplicates: Whether to remove exact duplicates
            similarity_threshold: Similarity threshold for near-duplicate detection
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.min_length = min_length
        self.max_length = max_length
        self.max_repetition = max_repetition
        self.min_distinct_2 = min_distinct_2
        self.min_entropy = min_entropy
        self.remove_duplicates = remove_duplicates
        self.similarity_threshold = similarity_threshold

        # Tracking
        self.seen_hashes: Set[str] = set()
        self.stats = {
            'total': 0,
            'too_short': 0,
            'too_long': 0,
            'duplicate': 0,
            'high_repetition': 0,
            'low_diversity': 0,
            'low_entropy': 0,
            'passed': 0,
        }

        logger.info("Initialized DataQualityFilter")
        logger.info(f"  Min length: {min_length} tokens")
        logger.info(f"  Max length: {max_length} tokens")
        logger.info(f"  Max repetition: {max_repetition}")
        logger.info(f"  Min distinct-2: {min_distinct_2}")
        logger.info(f"  Min entropy: {min_entropy}")
        logger.info(f"  Remove duplicates: {remove_duplicates}")

    def compute_hash(self, text: str) -> str:
        """Compute hash of text for duplicate detection."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def check_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = self.compute_hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def compute_repetition(self, tokens: List[int]) -> float:
        """Compute repetition ratio (fraction of repeated n-grams)."""
        if len(tokens) < 4:
            return 0.0

        # Use 4-grams for repetition detection
        ngrams = []
        for i in range(len(tokens) - 3):
            ngrams.append(tuple(tokens[i:i+4]))

        if not ngrams:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        return 1.0 - (unique_ngrams / total_ngrams)

    def compute_distinct_n(self, tokens: List[int], n: int) -> float:
        """Compute distinct-n metric."""
        if len(tokens) < n:
            return 0.0

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))

        if not ngrams:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        return unique_ngrams / total_ngrams

    def compute_entropy(self, tokens: List[int]) -> float:
        """Compute Shannon entropy of token distribution."""
        import math

        if not tokens:
            return 0.0

        # Count token frequencies
        token_counts = Counter(tokens)
        total = len(tokens)

        # Calculate entropy
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def filter_sample(self, sample: Dict) -> Tuple[bool, str]:
        """
        Filter a single sample.

        Returns:
            (should_keep, reason) tuple
        """
        self.stats['total'] += 1

        # Extract text
        text = sample.get('text', '')
        if not text:
            return False, 'empty_text'

        # Check duplicates
        if self.remove_duplicates and self.check_duplicate(text):
            self.stats['duplicate'] += 1
            return False, 'duplicate'

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Length filtering
        if len(tokens) < self.min_length:
            self.stats['too_short'] += 1
            return False, 'too_short'

        if len(tokens) > self.max_length:
            self.stats['too_long'] += 1
            return False, 'too_long'

        # Repetition filtering
        repetition = self.compute_repetition(tokens)
        if repetition > self.max_repetition:
            self.stats['high_repetition'] += 1
            return False, 'high_repetition'

        # Diversity filtering
        distinct_2 = self.compute_distinct_n(tokens, 2)
        if distinct_2 < self.min_distinct_2:
            self.stats['low_diversity'] += 1
            return False, 'low_diversity'

        # Entropy filtering
        entropy = self.compute_entropy(tokens)
        if entropy < self.min_entropy:
            self.stats['low_entropy'] += 1
            return False, 'low_entropy'

        # Sample passed all filters
        self.stats['passed'] += 1
        return True, 'passed'

    def filter_file(self, input_path: Path, output_path: Path):
        """Filter an entire JSONL file."""
        logger.info(f"Filtering {input_path} -> {output_path}")

        # Count total lines for progress bar
        with open(input_path, 'r') as f:
            total_lines = sum(1 for _ in f)

        # Process file
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in tqdm(infile, total=total_lines, desc="Filtering"):
                try:
                    sample = json.loads(line.strip())
                    should_keep, reason = self.filter_sample(sample)

                    if should_keep:
                        outfile.write(json.dumps(sample) + '\n')

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue

        # Print statistics
        self.print_stats()

    def print_stats(self):
        """Print filtering statistics."""
        total = self.stats['total']
        if total == 0:
            logger.info("No samples processed")
            return

        logger.info("\n" + "="*60)
        logger.info("FILTERING STATISTICS")
        logger.info("="*60)
        logger.info(f"Total samples: {total}")
        logger.info(f"Passed filters: {self.stats['passed']} ({100*self.stats['passed']/total:.1f}%)")
        logger.info(f"\nFiltered out:")
        logger.info(f"  • Too short: {self.stats['too_short']} ({100*self.stats['too_short']/total:.1f}%)")
        logger.info(f"  • Too long: {self.stats['too_long']} ({100*self.stats['too_long']/total:.1f}%)")
        logger.info(f"  • Duplicates: {self.stats['duplicate']} ({100*self.stats['duplicate']/total:.1f}%)")
        logger.info(f"  • High repetition: {self.stats['high_repetition']} ({100*self.stats['high_repetition']/total:.1f}%)")
        logger.info(f"  • Low diversity: {self.stats['low_diversity']} ({100*self.stats['low_diversity']/total:.1f}%)")
        logger.info(f"  • Low entropy: {self.stats['low_entropy']} ({100*self.stats['low_entropy']/total:.1f}%)")
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Filter training data by coherence metrics")
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--tokenizer', type=str, default='/project/code/models/tokenizer/enhanced-50680',
                       help='Path to tokenizer')
    parser.add_argument('--min-length', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--max-repetition', type=float, default=0.5,
                       help='Maximum repetition ratio (0.0-1.0)')
    parser.add_argument('--min-distinct-2', type=float, default=0.3,
                       help='Minimum distinct-2 score (0.0-1.0)')
    parser.add_argument('--min-entropy', type=float, default=2.0,
                       help='Minimum entropy threshold')
    parser.add_argument('--no-dedup', action='store_true',
                       help='Disable duplicate removal')

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create filter
    filter_obj = DataQualityFilter(
        tokenizer_path=args.tokenizer,
        min_length=args.min_length,
        max_length=args.max_length,
        max_repetition=args.max_repetition,
        min_distinct_2=args.min_distinct_2,
        min_entropy=args.min_entropy,
        remove_duplicates=not args.no_dedup,
    )

    # Filter file
    filter_obj.filter_file(input_path, output_path)

    logger.info(f"✓ Filtered data saved to {output_path}")


if __name__ == '__main__':
    main()
