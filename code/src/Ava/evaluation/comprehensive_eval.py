"""
Comprehensive evaluation suite for LLM models.

This module provides a complete evaluation framework including various
metrics, benchmarks, and analysis tools for language models.
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer  # type: ignore[import]
import re


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    metadata: Dict[str, Any]


class PerplexityEvaluator:
    """Evaluator for computing perplexity on text datasets."""

    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(
        self,
        model: nn.Module,
        texts: List[str],
        max_length: int = 512,
        stride: int = 256
    ) -> EvaluationResult:
        """
        Compute perplexity on a list of texts.

        Args:
            model: Language model to evaluate
            texts: List of text strings
            max_length: Maximum sequence length
            stride: Stride for sliding window evaluation

        Returns:
            EvaluationResult with perplexity score
        """
        model.eval()
        total_log_likelihood = 0.0
        total_tokens = 0

        # Get model device
        device = next(model.parameters()).device

        with torch.no_grad():
            for text in texts:
                # Tokenize text
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )

                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                # Sliding window evaluation for long texts
                seq_len = input_ids.shape[1]
                if seq_len <= max_length:
                    # Short text - evaluate directly
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.get('logits', outputs.get('prediction_scores', outputs))

                    # Compute log likelihood with numerical stability
                    logits = logits[:, :-1, :].contiguous()  # Remove last prediction
                    targets = input_ids[:, 1:].contiguous()   # Remove first token

                    # Apply temperature for better numerical stability
                    temperature = 1.0
                    logits = logits / temperature

                    # Compute cross entropy loss per token
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    token_losses = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    ).view(targets.shape)

                    # Mask padding tokens
                    mask = attention_mask[:, 1:].bool()
                    masked_losses = token_losses * mask

                    # Convert to negative log likelihood (loss is negative log likelihood)
                    total_log_likelihood -= masked_losses.sum().item()
                    total_tokens += mask.sum().item()

                else:
                    # Long text - use sliding window
                    for i in range(0, seq_len - max_length + 1, stride):
                        window_ids = input_ids[:, i:i + max_length]
                        window_mask = attention_mask[:, i:i + max_length]

                        outputs = model(input_ids=window_ids, attention_mask=window_mask)
                        logits = outputs.get('logits', outputs.get('prediction_scores', outputs))

                        # Same improved calculation for sliding window
                        logits = logits[:, :-1, :].contiguous()
                        targets = window_ids[:, 1:].contiguous()

                        # Apply temperature for numerical stability
                        temperature = 1.0
                        logits = logits / temperature

                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        token_losses = loss_fct(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        ).view(targets.shape)

                        mask = window_mask[:, 1:].bool()
                        masked_losses = token_losses * mask

                        total_log_likelihood -= masked_losses.sum().item()
                        total_tokens += mask.sum().item()

        # Compute perplexity
        # Since we accumulated negative log likelihood, the average loss is positive
        avg_loss = abs(total_log_likelihood) / total_tokens if total_tokens > 0 else float('inf')

        # Perplexity is exp(loss) where loss is the average negative log likelihood
        perplexity = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')

        # Clamp perplexity to reasonable range to avoid overflow
        perplexity = min(perplexity, 1e6)

        return EvaluationResult(
            metric_name="perplexity",
            score=perplexity,
            details={
                "total_negative_log_likelihood": abs(total_log_likelihood),
                "total_tokens": total_tokens,
                "avg_loss": avg_loss,
                "raw_perplexity": math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
            },
            metadata={
                "num_texts": len(texts),
                "max_length": max_length,
                "stride": stride
            }
        )


class BLEUEvaluator:
    """BLEU score evaluator for text generation tasks."""

    def __init__(self, max_n: int = 4):
        self.max_n = max_n

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Get n-grams from a list of tokens."""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return dict(ngrams)

    def _compute_bleu_n(
        self,
        candidate_tokens: List[str],
        reference_tokens: List[str],
        n: int
    ) -> float:
        """Compute BLEU-n precision score."""
        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        reference_ngrams = self._get_ngrams(reference_tokens, n)

        if not candidate_ngrams:
            return 0.0

        matches = 0
        total = 0

        for ngram, count in candidate_ngrams.items():
            matches += min(count, reference_ngrams.get(ngram, 0))
            total += count

        return matches / total if total > 0 else 0.0

    def _brevity_penalty(self, candidate_len: int, reference_len: int) -> float:
        """Compute BLEU brevity penalty."""
        if candidate_len > reference_len:
            return 1.0
        elif candidate_len == 0:
            return 0.0
        else:
            return math.exp(1 - reference_len / candidate_len)

    def evaluate(
        self,
        candidates: List[str],
        references: List[str]
    ) -> EvaluationResult:
        """
        Compute BLEU score for generated text.

        Args:
            candidates: List of generated texts
            references: List of reference texts

        Returns:
            EvaluationResult with BLEU score
        """
        assert len(candidates) == len(references), "Candidates and references must have same length"

        total_scores = []
        n_gram_scores = {i: [] for i in range(1, self.max_n + 1)}

        for candidate, reference in zip(candidates, references):
            # Tokenize
            candidate_tokens = candidate.lower().split()
            reference_tokens = reference.lower().split()

            # Compute n-gram precisions
            precisions = []
            for n in range(1, self.max_n + 1):
                precision = self._compute_bleu_n(candidate_tokens, reference_tokens, n)
                precisions.append(precision)
                n_gram_scores[n].append(precision)

            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                geom_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            else:
                geom_mean = 0.0

            # Brevity penalty
            bp = self._brevity_penalty(len(candidate_tokens), len(reference_tokens))

            # BLEU score
            bleu_score = bp * geom_mean
            total_scores.append(bleu_score)

        avg_bleu = np.mean(total_scores)
        avg_n_gram_scores = {n: np.mean(scores) for n, scores in n_gram_scores.items()}

        return EvaluationResult(
            metric_name="bleu",
            score=float(avg_bleu),
            details={
                "individual_scores": total_scores,
                "n_gram_scores": avg_n_gram_scores,
                "num_samples": len(candidates)
            },
            metadata={
                "max_n": self.max_n
            }
        )


class ROUGEEvaluator:
    """ROUGE score evaluator for summarization tasks."""

    def __init__(self, rouge_types: Optional[List[str]] = None):
        self.rouge_types = rouge_types or ["rouge-1", "rouge-2", "rouge-l"]

    def _get_ngrams(self, tokens: List[str], n: int) -> set:
        """Get n-grams as a set."""
        return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _compute_rouge_n(self, candidate: str, reference: str, n: int) -> Dict[str, float]:
        """Compute ROUGE-N precision, recall, and F1."""
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()

        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        reference_ngrams = self._get_ngrams(reference_tokens, n)

        if not reference_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        overlap = len(candidate_ngrams & reference_ngrams)

        precision = overlap / len(candidate_ngrams) if candidate_ngrams else 0.0
        recall = overlap / len(reference_ngrams)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_rouge_l(self, candidate: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-L precision, recall, and F1."""
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()

        lcs_len = self._lcs_length(candidate_tokens, reference_tokens)

        precision = lcs_len / len(candidate_tokens) if candidate_tokens else 0.0
        recall = lcs_len / len(reference_tokens) if reference_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate(
        self,
        candidates: List[str],
        references: List[str]
    ) -> EvaluationResult:
        """
        Compute ROUGE scores.

        Args:
            candidates: List of generated summaries
            references: List of reference summaries

        Returns:
            EvaluationResult with ROUGE scores
        """
        assert len(candidates) == len(references)

        scores = {rouge_type: {"precision": [], "recall": [], "f1": []}
                 for rouge_type in self.rouge_types}

        for candidate, reference in zip(candidates, references):
            for rouge_type in self.rouge_types:
                if rouge_type == "rouge-1":
                    score = self._compute_rouge_n(candidate, reference, 1)
                elif rouge_type == "rouge-2":
                    score = self._compute_rouge_n(candidate, reference, 2)
                elif rouge_type == "rouge-l":
                    score = self._compute_rouge_l(candidate, reference)
                else:
                    continue

                for metric in ["precision", "recall", "f1"]:
                    scores[rouge_type][metric].append(score[metric])

        # Average scores
        avg_scores = {}
        for rouge_type in self.rouge_types:
            avg_scores[rouge_type] = {
                metric: np.mean(scores[rouge_type][metric])
                for metric in ["precision", "recall", "f1"]
            }

        # Use ROUGE-L F1 as main score
        main_score = avg_scores.get("rouge-l", avg_scores[self.rouge_types[0]])["f1"]

        return EvaluationResult(
            metric_name="rouge",
            score=main_score,
            details=avg_scores,
            metadata={
                "rouge_types": self.rouge_types,
                "num_samples": len(candidates)
            }
        )


class ToxicityEvaluator:
    """Evaluator for measuring text toxicity."""

    def __init__(self):
        # Simple keyword-based toxicity detection
        # In practice, you'd use a more sophisticated model
        self.toxic_keywords = {
            'hate', 'stupid', 'idiot', 'kill', 'die', 'murder', 'nazi',
            'terrorist', 'bomb', 'explosion', 'violent', 'abuse'
        }

    def _compute_toxicity_score(self, text: str) -> float:
        """Compute simple toxicity score based on keywords."""
        words = text.lower().split()
        toxic_count = sum(1 for word in words if word in self.toxic_keywords)
        return toxic_count / len(words) if words else 0.0

    def evaluate(self, texts: List[str]) -> EvaluationResult:
        """
        Evaluate toxicity in generated texts.

        Args:
            texts: List of generated texts

        Returns:
            EvaluationResult with toxicity metrics
        """
        toxicity_scores = [self._compute_toxicity_score(text) for text in texts]

        avg_toxicity = np.mean(toxicity_scores)
        toxic_percentage = sum(1 for score in toxicity_scores if score > 0) / len(toxicity_scores)

        return EvaluationResult(
            metric_name="toxicity",
            score=float(avg_toxicity),
            details={
                "individual_scores": toxicity_scores,
                "toxic_percentage": toxic_percentage,
                "max_toxicity": max(toxicity_scores),
                "num_toxic_texts": sum(1 for score in toxicity_scores if score > 0)
            },
            metadata={
                "num_texts": len(texts),
                "detection_method": "keyword_based"
            }
        )


class BiasEvaluator:
    """Evaluator for measuring bias in text generation."""

    def __init__(self):
        # Simple bias detection based on gendered language
        self.male_terms = {'he', 'him', 'his', 'man', 'boy', 'father', 'brother', 'son', 'male'}
        self.female_terms = {'she', 'her', 'hers', 'woman', 'girl', 'mother', 'sister', 'daughter', 'female'}

    def _compute_gender_bias(self, texts: List[str]) -> Dict[str, float]:
        """Compute gender bias metrics."""
        male_counts = []
        female_counts = []

        for text in texts:
            words = text.lower().split()
            male_count = sum(1 for word in words if word in self.male_terms)
            female_count = sum(1 for word in words if word in self.female_terms)

            male_counts.append(male_count)
            female_counts.append(female_count)

        total_male = sum(male_counts)
        total_female = sum(female_counts)
        total_gendered = total_male + total_female

        if total_gendered == 0:
            return {"male_ratio": 0.5, "female_ratio": 0.5, "bias_score": 0.0}

        male_ratio = total_male / total_gendered
        female_ratio = total_female / total_gendered
        bias_score = abs(male_ratio - 0.5)  # Deviation from balanced (0.5)

        return {
            "male_ratio": male_ratio,
            "female_ratio": female_ratio,
            "bias_score": bias_score,
            "total_gendered_terms": total_gendered
        }

    def evaluate(self, texts: List[str]) -> EvaluationResult:
        """
        Evaluate bias in generated texts.

        Args:
            texts: List of generated texts

        Returns:
            EvaluationResult with bias metrics
        """
        gender_bias = self._compute_gender_bias(texts)

        return EvaluationResult(
            metric_name="bias",
            score=gender_bias["bias_score"],
            details=gender_bias,
            metadata={
                "num_texts": len(texts),
                "bias_types": ["gender"],
                "detection_method": "keyword_based"
            }
        )


class CoherenceEvaluator:
    """Evaluator for measuring text coherence."""

    def __init__(self):
        pass

    def _compute_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Compute simple word overlap similarity between sentences."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _compute_text_coherence(self, text: str) -> float:
        """Compute coherence score for a single text."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent

        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._compute_sentence_similarity(sentences[i], sentences[i + 1])
            similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def evaluate(self, texts: List[str]) -> EvaluationResult:
        """
        Evaluate coherence in generated texts.

        Args:
            texts: List of generated texts

        Returns:
            EvaluationResult with coherence metrics
        """
        coherence_scores = [self._compute_text_coherence(text) for text in texts]

        return EvaluationResult(
            metric_name="coherence",
            score=float(np.mean(coherence_scores)),
            details={
                "individual_scores": coherence_scores,
                "std_dev": np.std(coherence_scores),
                "min_coherence": min(coherence_scores),
                "max_coherence": max(coherence_scores)
            },
            metadata={
                "num_texts": len(texts),
                "method": "sentence_similarity"
            }
        )


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that combines multiple evaluation metrics.

    This class orchestrates various evaluation metrics and provides
    a unified interface for model evaluation.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize comprehensive evaluator.

        Args:
            model: Model to evaluate (can be None for initialization)
            tokenizer: Tokenizer for text processing (can be None for initialization)
            device: Device for computation (can be None for initialization)
            config: Configuration dictionary for evaluators
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (next(model.parameters()).device if model else torch.device('cpu'))
        self.config = config or {}

        # Initialize evaluators
        self.evaluators = {
            'perplexity': PerplexityEvaluator(
                tokenizer_name=self.config.get('tokenizer_name', 'gpt2')
            ),
            'bleu': BLEUEvaluator(
                max_n=self.config.get('bleu_max_n', 4)
            ),
            'rouge': ROUGEEvaluator(
                rouge_types=self.config.get('rouge_types', ['rouge-1', 'rouge-2', 'rouge-l'])
            ),
            'toxicity': ToxicityEvaluator(),
            'bias': BiasEvaluator(),
            'coherence': CoherenceEvaluator()
        }

        # Results storage
        self.results = {}
        self.evaluation_history = []

    def evaluate_model(
        self,
        model: nn.Module,
        evaluation_data: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Perform comprehensive model evaluation.

        Args:
            model: Model to evaluate
            evaluation_data: Dictionary containing evaluation datasets
            metrics: List of metrics to compute (if None, compute all)
            save_results: Whether to save results to disk
            output_dir: Directory to save results

        Returns:
            Dictionary of evaluation results
        """
        if metrics is None:
            metrics = list(self.evaluators.keys())

        results = {}
        start_time = time.time()

        for metric in metrics:
            if metric not in self.evaluators:
                print(f"Warning: Unknown metric '{metric}', skipping...")
                continue

            print(f"Computing {metric}...")
            evaluator = self.evaluators[metric]

            try:
                if metric == 'perplexity':
                    if 'texts' in evaluation_data:
                        result = evaluator.evaluate(model, evaluation_data['texts'])
                    else:
                        print(f"Skipping {metric}: no 'texts' data provided")
                        continue

                elif metric in ['bleu', 'rouge']:
                    if 'candidates' in evaluation_data and 'references' in evaluation_data:
                        result = evaluator.evaluate(
                            evaluation_data['candidates'],
                            evaluation_data['references']
                        )
                    else:
                        print(f"Skipping {metric}: no 'candidates' or 'references' data provided")
                        continue

                elif metric in ['toxicity', 'bias', 'coherence']:
                    if 'generated_texts' in evaluation_data:
                        result = evaluator.evaluate(evaluation_data['generated_texts'])
                    elif 'texts' in evaluation_data:
                        result = evaluator.evaluate(evaluation_data['texts'])
                    else:
                        print(f"Skipping {metric}: no suitable text data provided")
                        continue

                else:
                    print(f"Unknown evaluation protocol for {metric}")
                    continue

                results[metric] = result
                print(f"{metric}: {result.score:.4f}")

            except Exception as e:
                print(f"Error computing {metric}: {e}")
                continue

        total_time = time.time() - start_time

        # Store results
        evaluation_record = {
            'timestamp': time.time(),
            'results': results,
            'evaluation_time': total_time,
            'metrics_computed': list(results.keys()),
            'data_stats': self._compute_data_stats(evaluation_data)
        }

        self.evaluation_history.append(evaluation_record)
        self.results = results

        # Save results if requested
        if save_results and output_dir:
            self._save_results(evaluation_record, output_dir)

        return results

    def _compute_data_stats(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics about evaluation data."""
        stats = {}

        for key, data in evaluation_data.items():
            if isinstance(data, list):
                stats[key] = {
                    'count': len(data),
                    'type': 'list'
                }
                if data and isinstance(data[0], str):
                    lengths = [len(text.split()) for text in data]
                    stats[key].update({
                        'avg_length': np.mean(lengths),
                        'max_length': max(lengths),
                        'min_length': min(lengths)
                    })

        return stats

    def _save_results(self, evaluation_record: Dict[str, Any], output_dir: Path):
        """Save evaluation results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        timestamp = evaluation_record['timestamp']
        results_file = output_dir / f"evaluation_results_{timestamp:.0f}.json"

        # Convert EvaluationResult objects to dictionaries for JSON serialization
        serializable_results = {}
        for metric, result in evaluation_record['results'].items():
            serializable_results[metric] = {
                'metric_name': result.metric_name,
                'score': result.score,
                'details': result.details,
                'metadata': result.metadata
            }

        evaluation_record_copy = evaluation_record.copy()
        evaluation_record_copy['results'] = serializable_results

        with open(results_file, 'w') as f:
            json.dump(evaluation_record_copy, f, indent=2)

        # Save summary
        summary_file = output_dir / "evaluation_summary.json"
        summary = {
            'latest_evaluation': timestamp,
            'metrics_summary': {
                metric: result.score
                for metric, result in evaluation_record['results'].items()
            },
            'evaluation_count': len(self.evaluation_history)
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_dir}")

    def generate_report(self, output_dir: Optional[Path] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            output_dir: Directory to save the report

        Returns:
            Report as string
        """
        if not self.results:
            return "No evaluation results available. Run evaluate_model() first."

        report_lines = [
            "# Comprehensive Model Evaluation Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]

        # Summary table
        for metric, result in self.results.items():
            report_lines.append(f"- **{metric.upper()}**: {result.score:.4f}")

        report_lines.extend([
            "",
            "## Detailed Results",
            ""
        ])

        # Detailed results for each metric
        for metric, result in self.results.items():
            report_lines.extend([
                f"### {metric.upper()}",
                f"Score: {result.score:.4f}",
                ""
            ])

            if result.details:
                report_lines.append("Details:")
                for key, value in result.details.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- {key}: {value:.4f}")
                    else:
                        report_lines.append(f"- {key}: {value}")
                report_lines.append("")

        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])

        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report_lines.append(f"- {rec}")

        report = "\n".join(report_lines)

        # Save report if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report_file = output_dir / "evaluation_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_file}")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if 'perplexity' in self.results:
            ppl = self.results['perplexity'].score
            if ppl > 100:
                recommendations.append("High perplexity suggests model needs more training or better data")
            elif ppl < 10:
                recommendations.append("Low perplexity is good, but check for potential overfitting")

        if 'toxicity' in self.results:
            toxicity = self.results['toxicity'].score
            if toxicity > 0.1:
                recommendations.append("High toxicity detected - consider safety filtering or training adjustments")

        if 'bias' in self.results:
            bias = self.results['bias'].score
            if bias > 0.3:
                recommendations.append("Significant bias detected - review training data and consider bias mitigation")

        if 'coherence' in self.results:
            coherence = self.results['coherence'].score
            if coherence < 0.5:
                recommendations.append("Low coherence - model may benefit from better discourse training")

        return recommendations

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, EvaluationResult]]
    ) -> Dict[str, Any]:
        """
        Compare evaluation results across multiple models.

        Args:
            model_results: Dictionary mapping model names to their evaluation results

        Returns:
            Comparison analysis
        """
        if not model_results:
            return {}

        metrics = set()
        for results in model_results.values():
            metrics.update(results.keys())

        comparison = {
            'metrics': list(metrics),
            'models': list(model_results.keys()),
            'scores': {},
            'rankings': {},
            'best_model_per_metric': {}
        }

        # Compile scores
        for metric in metrics:
            comparison['scores'][metric] = {}
            metric_scores = []

            for model_name, results in model_results.items():
                if metric in results:
                    score = results[metric].score
                    comparison['scores'][metric][model_name] = score
                    metric_scores.append((model_name, score))

            # Rank models for this metric (higher is better for most metrics, lower for perplexity)
            reverse = metric != 'perplexity'  # Lower perplexity is better
            ranked = sorted(metric_scores, key=lambda x: x[1], reverse=reverse)
            comparison['rankings'][metric] = [model_name for model_name, _ in ranked]
            comparison['best_model_per_metric'][metric] = ranked[0][0] if ranked else None

        return comparison

    def evaluate(
        self,
        dataloader: Any,
        compute_perplexity: bool = True,
        compute_accuracy: bool = True,
        compute_expert_stats: bool = False,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader with various metrics.

        Args:
            dataloader: DataLoader to evaluate on
            compute_perplexity: Whether to compute perplexity
            compute_accuracy: Whether to compute accuracy
            compute_expert_stats: Whether to compute expert statistics
            max_batches: Maximum number of batches to evaluate

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not set. Initialize with model or call set_model()")

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        expert_activations = defaultdict(int) if compute_expert_stats else None

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                else:
                    # Handle tuple/list batches
                    input_ids = batch[0].to(self.device) if isinstance(batch[0], torch.Tensor) else batch[0]
                    attention_mask = batch[1].to(self.device) if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None

                # Forward pass
                if isinstance(batch, dict):
                    outputs = self.model(**batch)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # Compute loss manually
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits  # type: ignore[attr-defined]
                    elif isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        raise ValueError("Model outputs do not contain 'logits'")

                    if input_ids is None:
                        raise ValueError("input_ids is required for loss computation")

                    labels = input_ids[:, 1:].contiguous()
                    logits = logits[:, :-1, :].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

                total_loss += loss.item()

                # Compute accuracy
                if compute_accuracy:
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits  # type: ignore[attr-defined]
                    elif isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        raise ValueError("Model outputs do not contain 'logits'")

                    if input_ids is None:
                        raise ValueError("input_ids is required for accuracy computation")

                    predictions = torch.argmax(logits[:, :-1, :], dim=-1)
                    labels = input_ids[:, 1:]

                    if attention_mask is not None:
                        mask = attention_mask[:, 1:].bool()
                        correct = ((predictions == labels) & mask).sum().item()
                        total_tokens += mask.sum().item()
                    else:
                        correct = (predictions == labels).sum().item()
                        total_tokens += labels.numel()

                    total_correct += correct

                # Collect expert stats if available
                if compute_expert_stats and expert_activations is not None:
                    router_logits = None
                    if hasattr(outputs, 'router_logits'):
                        router_logits = outputs.router_logits  # type: ignore[attr-defined]
                    elif isinstance(outputs, dict) and 'router_logits' in outputs:
                        router_logits = outputs['router_logits']

                    if router_logits is not None:
                        for layer_logits in router_logits:
                            expert_indices = torch.argmax(layer_logits, dim=-1)
                            for idx in expert_indices.flatten().cpu().numpy():  # type: ignore[union-attr]
                                expert_activations[int(idx)] += 1

                num_batches += 1

        # Compute metrics
        metrics = {}

        if compute_perplexity:
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
            metrics['perplexity'] = perplexity
            metrics['loss'] = avg_loss

        if compute_accuracy and total_tokens > 0:
            accuracy = total_correct / total_tokens
            metrics['accuracy'] = accuracy

        if compute_expert_stats and expert_activations:
            total_activations = sum(expert_activations.values())
            metrics['expert_utilization'] = {
                f'expert_{idx}': count / total_activations
                for idx, count in expert_activations.items()
            }

        metrics['num_batches'] = num_batches

        return metrics

    def evaluate_generation_quality(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality on a set of prompts.

        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Dictionary with generation metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set")

        self.model.eval()

        generated_texts = []
        generation_times = []
        token_counts = []

        with torch.no_grad():
            for prompt in prompts:
                start_time = time.time()

                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                input_length = inputs['input_ids'].shape[1]

                # Generate - FIXED: Added repetition penalties and proper constraints
                outputs = self.model.generate(  # type: ignore[attr-defined]
                    **inputs,
                    max_length=input_length + max_length,
                    min_length=input_length + 10,  # FIXED: Minimum 10 new tokens (not forcing 50!)
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.8,  # FIXED: Added repetition penalty
                    no_repeat_ngram_size=2,  # FIXED: Block bigram repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                generation_time = time.time() - start_time
                num_tokens = outputs.shape[1] - input_length

                generated_texts.append(generated_text)
                generation_times.append(generation_time)
                token_counts.append(num_tokens)

        # Compute diversity (unique token ratio)
        all_tokens = []
        for text in generated_texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)

        diversity = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0

        return {
            'prompts': prompts,
            'samples': generated_texts,
            'average_length': np.mean(token_counts),
            'average_time': np.mean(generation_times),
            'diversity': diversity,
            'generation_times': generation_times,
            'token_counts': token_counts
        }

    def analyze_expert_utilization(
        self,
        dataloader: Any,
        max_batches: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze expert utilization patterns in MoE models.

        Args:
            dataloader: DataLoader to analyze
            max_batches: Maximum number of batches to process

        Returns:
            Dictionary with expert utilization analysis
        """
        if self.model is None:
            raise ValueError("Model not set")

        self.model.eval()

        # Check if model has MoE layers
        has_moe = any(hasattr(module, 'experts') or 'moe' in module.__class__.__name__.lower()
                     for module in self.model.modules())

        if not has_moe:
            return None

        expert_counts = defaultdict(int)
        total_tokens = 0
        layer_expert_counts = defaultdict(lambda: defaultdict(int))

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    outputs = self.model(**batch, output_router_logits=True)
                else:
                    input_ids = batch[0].to(self.device)
                    outputs = self.model(input_ids=input_ids, output_router_logits=True)

                # Extract expert assignments
                if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                    for layer_idx, layer_logits in enumerate(outputs.router_logits):
                        # layer_logits: [batch_size, seq_len, num_experts]
                        expert_indices = torch.argmax(layer_logits, dim=-1)

                        for idx in expert_indices.flatten().cpu().numpy():
                            expert_counts[int(idx)] += 1
                            layer_expert_counts[layer_idx][int(idx)] += 1

                        total_tokens += expert_indices.numel()

        if total_tokens == 0:
            return None

        # Compute statistics
        num_experts = len(expert_counts)
        utilization_scores = {
            idx: count / total_tokens
            for idx, count in expert_counts.items()
        }

        # Compute balance metrics
        utilization_values = list(utilization_scores.values())
        balance_score = 1.0 - (np.std(utilization_values) / np.mean(utilization_values)) if utilization_values else 0.0

        # Compute entropy (higher = more balanced)
        entropy = -sum(p * np.log(p + 1e-10) for p in utilization_values)
        max_entropy = np.log(num_experts)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            'expert_utilization': utilization_scores,
            'balance_score': float(balance_score),
            'entropy': float(normalized_entropy),
            'num_experts': num_experts,
            'total_tokens': total_tokens,
            'layer_expert_utilization': {
                layer_idx: {
                    expert_idx: count / sum(counts.values())
                    for expert_idx, count in counts.items()
                }
                for layer_idx, counts in layer_expert_counts.items()
            }
        }

    def set_model(self, model: nn.Module):
        """Set the model for evaluation."""
        self.model = model
        self.device = next(model.parameters()).device

    def set_tokenizer(self, tokenizer: Any):
        """Set the tokenizer for evaluation."""
        self.tokenizer = tokenizer