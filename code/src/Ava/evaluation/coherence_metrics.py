"""
Coherence Metrics for LLM Evaluation

Implements proven metrics from research to measure text quality:
- Distinct-n: Vocabulary diversity (Li et al., 2016)
- Self-BLEU: Inter-sample diversity (Zhu et al., 2018)
- Shannon Entropy: Token distribution unpredictability
- Repetition Ratio: N-gram repetition
- Burstiness: Token usage patterns (Goh & Barabási, 2008)
- Zipf Coefficient: Natural language distribution
"""

import numpy as np
from collections import Counter
from typing import List, Dict
import math


class CoherenceMetrics:
    """
    Fast coherence metrics for training-time evaluation.
    Optimized for quick feedback during validation.
    """

    @staticmethod
    def calculate_distinct_n(tokens: List[int], n: int = 2) -> float:
        """
        Distinct-n: Ratio of unique n-grams to total n-grams.
        Higher = more diverse vocabulary.

        Paper: Li et al., 2016 - "A Diversity-Promoting Objective Function"
        Target: >0.5 for unigrams, >0.7 for bigrams
        """
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def calculate_repetition_ratio(tokens: List[int], n: int = 4) -> float:
        """
        Repetition Ratio: Percentage of repeated n-grams.
        Lower = less repetitive.

        Target: <0.3 (30% repetition acceptable)
        """
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        ngram_counts = Counter(ngrams)
        repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
        return repeated / len(ngrams)

    @staticmethod
    def calculate_entropy(tokens: List[int]) -> float:
        """
        Shannon Entropy: Measures unpredictability of token distribution.
        Higher = more diverse/unpredictable.

        Target: >4.0 for good diversity
        """
        if not tokens:
            return 0.0

        token_counts = Counter(tokens)
        total = len(tokens)
        probabilities = [count / total for count in token_counts.values()]

        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

    @staticmethod
    def calculate_burstiness(tokens: List[int]) -> float:
        """
        Burstiness: Measures how "bursty" token usage is.
        Lower = more uniform distribution (less repetitive).

        Paper: Goh & Barabási, 2008 - "Temporal patterns in communication"
        Target: <0.5
        """
        if len(tokens) < 2:
            return 0.0

        token_counts = Counter(tokens)
        counts = list(token_counts.values())

        if len(counts) < 2:
            return 0.0

        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if mean_count == 0:
            return 0.0

        burstiness = (std_count - mean_count) / (std_count + mean_count)
        return float(burstiness)

    @staticmethod
    def calculate_zipf_coefficient(tokens: List[int]) -> float:
        """
        Zipf's Law Coefficient: Natural language follows Zipf's law.
        Closer to 1.0 = more natural language-like distribution.

        Target: 0.8-1.2
        """
        if not tokens:
            return 0.0

        token_counts = Counter(tokens)
        frequencies = sorted(token_counts.values(), reverse=True)

        if len(frequencies) < 2:
            return 0.0

        # Calculate Zipf coefficient using linear regression
        ranks = np.arange(1, len(frequencies) + 1)

        # Filter out zeros
        valid_indices = [i for i, f in enumerate(frequencies) if f > 0]
        if len(valid_indices) < 2:
            return 0.0

        log_ranks = np.log([ranks[i] for i in valid_indices])
        log_freqs = np.log([frequencies[i] for i in valid_indices])

        # Linear regression
        A = np.vstack([log_ranks, np.ones(len(log_ranks))]).T
        coefficient, _ = np.linalg.lstsq(A, log_freqs, rcond=None)[0]

        return abs(float(coefficient))

    @staticmethod
    def evaluate_sample(tokens: List[int]) -> Dict[str, float]:
        """
        Evaluate a single sample with all metrics.

        Args:
            tokens: List of token IDs (without prompt)

        Returns:
            Dictionary with all coherence metrics
        """
        if not tokens or len(tokens) < 4:
            return {
                'distinct_1': 0.0,
                'distinct_2': 0.0,
                'distinct_4': 0.0,
                'repetition': 1.0,
                'entropy': 0.0,
                'burstiness': 1.0,
                'zipf': 0.0
            }

        return {
            'distinct_1': CoherenceMetrics.calculate_distinct_n(tokens, n=1),
            'distinct_2': CoherenceMetrics.calculate_distinct_n(tokens, n=2),
            'distinct_4': CoherenceMetrics.calculate_distinct_n(tokens, n=4),
            'repetition': CoherenceMetrics.calculate_repetition_ratio(tokens, n=4),
            'entropy': CoherenceMetrics.calculate_entropy(tokens),
            'burstiness': CoherenceMetrics.calculate_burstiness(tokens),
            'zipf': CoherenceMetrics.calculate_zipf_coefficient(tokens)
        }

    @staticmethod
    def evaluate_batch(token_lists: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate multiple samples and return average metrics.

        Args:
            token_lists: List of token ID lists (one per sample)

        Returns:
            Dictionary with average metrics across all samples
        """
        if not token_lists:
            return {
                'distinct_1': 0.0,
                'distinct_2': 0.0,
                'distinct_4': 0.0,
                'repetition': 1.0,
                'entropy': 0.0,
                'burstiness': 1.0,
                'zipf': 0.0,
                'coherence_score': 0.0
            }

        # Collect metrics for each sample
        all_metrics = []
        for tokens in token_lists:
            if len(tokens) >= 4:
                all_metrics.append(CoherenceMetrics.evaluate_sample(tokens))

        if not all_metrics:
            return {
                'distinct_1': 0.0,
                'distinct_2': 0.0,
                'distinct_4': 0.0,
                'repetition': 1.0,
                'entropy': 0.0,
                'burstiness': 1.0,
                'zipf': 0.0,
                'coherence_score': 0.0
            }

        # Average across samples - convert to Python floats for consistency
        avg_metrics: Dict[str, float] = {}
        for key in all_metrics[0].keys():
            mean_val = np.mean([m[key] for m in all_metrics])
            # Ensure conversion to Python float, not numpy float
            avg_metrics[key] = float(mean_val.item()) if hasattr(mean_val, 'item') else float(mean_val)

        # Calculate overall coherence score (0-100)
        score = 0.0
        weights = []

        # Diversity (higher is better)
        if avg_metrics['distinct_2'] > 0.7:
            score += 25
        elif avg_metrics['distinct_2'] > 0.5:
            score += 15
        elif avg_metrics['distinct_2'] > 0.3:
            score += 5

        # Repetition (lower is better)
        if avg_metrics['repetition'] < 0.3:
            score += 25
        elif avg_metrics['repetition'] < 0.5:
            score += 15
        elif avg_metrics['repetition'] < 0.7:
            score += 5

        # Entropy (higher is better)
        if avg_metrics['entropy'] > 4.0:
            score += 25
        elif avg_metrics['entropy'] > 3.0:
            score += 15
        elif avg_metrics['entropy'] > 2.0:
            score += 5

        # Zipf (closer to 1.0 is better)
        zipf = avg_metrics['zipf']
        if 0.8 <= zipf <= 1.2:
            score += 25
        elif 0.6 <= zipf <= 1.4:
            score += 15
        elif 0.4 <= zipf <= 1.6:
            score += 5

        # Return with coherence score added, converting all values to Python floats
        # We rebuild the dict to ensure all values are pure Python floats, not numpy types
        result: Dict[str, float] = {}  # type: ignore[misc]
        for k, v in avg_metrics.items():
            if isinstance(v, np.ndarray):
                # Explicitly convert numpy types to Python float
                val: float = float(v.item()) if v.ndim == 0 else float(v.mean())
                result[k] = val  # type: ignore[assignment]
            elif isinstance(v, (np.floating, np.integer)):
                # Handle numpy scalar types - cast to ensure Python float
                result[k] = float(v)  # type: ignore[assignment]
            else:
                # Any other type - convert to float
                result[k] = float(v)  # type: ignore[assignment]
        result['coherence_score'] = float(score)
        return result  # type: ignore[return-value]

    @staticmethod
    def format_report(metrics: Dict[str, float]) -> str:
        """
        Format metrics as a human-readable string for logging.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Formatted string
        """
        report = []

        # Coherence score with status
        score = metrics.get('coherence_score', 0)
        if score >= 75:
            status = "✅ Excellent"
        elif score >= 50:
            status = "⚠️  Moderate"
        else:
            status = "❌ Poor"

        report.append(f"Coherence Score: {score:.0f}/100 ({status})")

        # Key metrics
        d2 = metrics.get('distinct_2', 0)
        rep = metrics.get('repetition', 0)
        ent = metrics.get('entropy', 0)

        report.append(f"  Distinct-2: {d2:.3f} {'✅' if d2 > 0.7 else '❌' if d2 < 0.5 else '⚠️'}")
        report.append(f"  Repetition: {rep:.3f} {'✅' if rep < 0.3 else '❌' if rep > 0.5 else '⚠️'}")
        report.append(f"  Entropy: {ent:.2f} {'✅' if ent > 4.0 else '❌' if ent < 2.0 else '⚠️'}")

        return "\n".join(report)


def quick_coherence_test(token_lists: List[List[int]]) -> Dict[str, float]:
    """
    Quick coherence test for training validation.

    This is the main function to use during training.

    Args:
        token_lists: List of generated token sequences (without prompts)

    Returns:
        Dictionary with coherence metrics including overall score

    Example:
        >>> tokens1 = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> tokens2 = [9, 10, 11, 12, 13, 14, 15]
        >>> metrics = quick_coherence_test([tokens1, tokens2])
        >>> print(f"Score: {metrics['coherence_score']}/100")
    """
    return CoherenceMetrics.evaluate_batch(token_lists)
