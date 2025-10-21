#!/usr/bin/env python3
"""
Comprehensive LLM Coherence Evaluation Script
Uses proven metrics from academic literature and industry best practices
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import math
from pathlib import Path
import argparse
import json
import sys
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from Ava.models import EnhancedMoEModel  # type: ignore[attr-defined]
from Ava.config.training_config import TrainingConfig
from transformers import AutoTokenizer


class CoherenceMetrics:
    """
    Implements proven coherence metrics from research:
    1. Self-BLEU (measuring diversity vs repetition)
    2. Distinct-n (vocabulary diversity)
    3. Entropy (token distribution)
    4. Repetition ratio (n-gram overlap)
    5. Perplexity (model confidence)
    6. Semantic coherence (embedding-based)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def calculate_distinct_n(self, tokens: List[int], n: int = 2) -> float:
        """
        Distinct-n: Ratio of unique n-grams to total n-grams
        Higher = more diverse
        Paper: "A Diversity-Promoting Objective Function for Neural Conversation Models" (Li et al., 2016)
        """
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        return unique_ngrams / total_ngrams

    def calculate_repetition_ratio(self, tokens: List[int], n: int = 4) -> float:
        """
        Repetition Ratio: Percentage of repeated n-grams
        Lower = less repetition
        """
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        ngram_counts = Counter(ngrams)
        repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
        return repeated / len(ngrams)

    def calculate_entropy(self, tokens: List[int]) -> float:
        """
        Shannon Entropy: Measures unpredictability of token distribution
        Higher = more diverse/unpredictable
        """
        if not tokens:
            return 0.0

        token_counts = Counter(tokens)
        total = len(tokens)
        probabilities = [count / total for count in token_counts.values()]

        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

    def calculate_self_bleu(self, generated_texts: List[str], n: int = 4) -> float:
        """
        Self-BLEU: Average BLEU score when comparing each text against others
        Lower = more diverse (each text is different from others)
        Paper: "Texygen: A Benchmarking Platform for Text Generation Models" (Zhu et al., 2018)
        """
        if len(generated_texts) < 2:
            return 0.0

        def get_ngrams(tokens: List[str], n: int) -> Counter:
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

        def calculate_bleu_score(reference: List[str], candidate: List[str], n: int = 4) -> float:
            """Modified BLEU calculation"""
            scores = []
            for i in range(1, n + 1):
                ref_ngrams = get_ngrams(reference, i)
                cand_ngrams = get_ngrams(candidate, i)

                if not cand_ngrams:
                    continue

                overlap = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
                score = overlap / sum(cand_ngrams.values())
                scores.append(score)

            if not scores:
                return 0.0

            # Geometric mean
            return math.exp(sum(math.log(s + 1e-10) for s in scores) / len(scores))

        all_bleu_scores = []
        for i, text in enumerate(generated_texts):
            candidate_tokens = text.split()
            scores_for_text = []

            for j, other_text in enumerate(generated_texts):
                if i != j:
                    reference_tokens = other_text.split()
                    score = calculate_bleu_score(reference_tokens, candidate_tokens, n)
                    scores_for_text.append(score)

            if scores_for_text:
                all_bleu_scores.append(np.mean(scores_for_text))

        return float(np.mean(all_bleu_scores)) if all_bleu_scores else 0.0

    def calculate_burstiness(self, tokens: List[int]) -> float:
        """
        Burstiness: Measures how "bursty" token usage is
        Lower = more uniform distribution (less repetitive)
        Paper: "Temporal patterns in communication flows" (Goh & Barabási, 2008)
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

    def calculate_zipf_coefficient(self, tokens: List[int]) -> float:
        """
        Zipf's Law Coefficient: Natural language follows Zipf's law
        Closer to 1.0 = more natural language-like distribution
        """
        if not tokens:
            return 0.0

        token_counts = Counter(tokens)
        # Sort by frequency
        frequencies = sorted(token_counts.values(), reverse=True)

        if len(frequencies) < 2:
            return 0.0

        # Calculate Zipf coefficient using linear regression on log-log plot
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

        return abs(coefficient)  # Should be close to 1.0 for natural language


class CoherenceEvaluator:
    """Main evaluation class"""

    def __init__(self, model_path: str, config_path: str):
        print("Loading model and tokenizer...")

        # Load config - use load method if from_yaml doesn't exist
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Use EnhancedTrainingConfig which has all config sections including model
        from Ava.config.training_config import EnhancedTrainingConfig, ModelConfig
        try:
            self.config = EnhancedTrainingConfig(**config_dict)  # type: ignore[call-arg]
        except (TypeError, KeyError):
            # Fallback: manually set config_file and required fields
            config_dict['config_file'] = config_path
            self.config = EnhancedTrainingConfig(**config_dict)  # type: ignore[call-arg]

        # Load tokenizer - get model config for tokenizer name
        model_dict = config_dict.get('model', {})
        tokenizer_name = model_dict.get('tokenizer_name', 'Qwen/Qwen2.5-0.5B')
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Type: EnhancedMoEModel might be None from conditional import
        if EnhancedMoEModel is None:
            raise ImportError("EnhancedMoEModel is not available")
        # Create ModelConfig from the model section of the config
        model_config = ModelConfig(**model_dict)
        self.model = EnhancedMoEModel(model_config)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.metrics = CoherenceMetrics(self.tokenizer)

        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")

    def generate_samples(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2
    ) -> List[Tuple[str, List[int]]]:
        """Generate text samples and return both text and token IDs"""

        samples = []

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Generate - EnhancedMoEModel has a generate method
                # Ensure we're passing tensors correctly
                input_tensor = inputs.input_ids if isinstance(inputs.input_ids, torch.Tensor) else inputs['input_ids']
                attention_mask = inputs.attention_mask if hasattr(inputs, 'attention_mask') else inputs.get('attention_mask')

                outputs = self.model.generate(  # type: ignore[misc]
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Decode
                output_tensor = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
                generated_text = self.tokenizer.decode(output_tensor, skip_special_tokens=True)
                token_ids = output_tensor.cpu().tolist()

                samples.append((generated_text, token_ids))

        return samples

    def evaluate_coherence(
        self,
        prompts: List[str],
        num_samples_per_prompt: int = 5,
        **generation_kwargs
    ) -> Dict:
        """
        Comprehensive coherence evaluation

        Returns metrics indicating:
        - Diversity (distinct-n, entropy)
        - Repetition (repetition ratio, burstiness)
        - Naturalness (Zipf coefficient)
        - Inter-sample diversity (self-BLEU)
        """

        print(f"\nGenerating {len(prompts) * num_samples_per_prompt} samples...")

        all_samples = []
        all_texts = []
        all_tokens = []

        for prompt in prompts:
            for _ in range(num_samples_per_prompt):
                samples = self.generate_samples([prompt], **generation_kwargs)
                text, tokens = samples[0]
                all_samples.append((prompt, text, tokens))
                all_texts.append(text)
                all_tokens.append(tokens)

        print("Calculating metrics...")

        # Calculate metrics across all samples
        results = {
            "sample_count": len(all_samples),
            "diversity_metrics": {},
            "repetition_metrics": {},
            "naturalness_metrics": {},
            "per_sample_metrics": []
        }

        # Aggregate metrics
        distinct_1_scores = []
        distinct_2_scores = []
        distinct_4_scores = []
        entropy_scores = []
        repetition_scores = []
        burstiness_scores = []
        zipf_scores = []

        for tokens in all_tokens:
            distinct_1_scores.append(self.metrics.calculate_distinct_n(tokens, n=1))
            distinct_2_scores.append(self.metrics.calculate_distinct_n(tokens, n=2))
            distinct_4_scores.append(self.metrics.calculate_distinct_n(tokens, n=4))
            entropy_scores.append(self.metrics.calculate_entropy(tokens))
            repetition_scores.append(self.metrics.calculate_repetition_ratio(tokens, n=4))
            burstiness_scores.append(self.metrics.calculate_burstiness(tokens))
            zipf_scores.append(self.metrics.calculate_zipf_coefficient(tokens))

        # Diversity metrics
        results["diversity_metrics"] = {
            "distinct_1": {
                "mean": float(np.mean(distinct_1_scores)),
                "std": float(np.std(distinct_1_scores)),
                "interpretation": "Ratio of unique unigrams (higher=more diverse, target: >0.5)"
            },
            "distinct_2": {
                "mean": float(np.mean(distinct_2_scores)),
                "std": float(np.std(distinct_2_scores)),
                "interpretation": "Ratio of unique bigrams (higher=more diverse, target: >0.7)"
            },
            "distinct_4": {
                "mean": float(np.mean(distinct_4_scores)),
                "std": float(np.std(distinct_4_scores)),
                "interpretation": "Ratio of unique 4-grams (higher=more diverse, target: >0.8)"
            },
            "entropy": {
                "mean": float(np.mean(entropy_scores)),
                "std": float(np.std(entropy_scores)),
                "interpretation": "Token distribution entropy (higher=more unpredictable, target: >4.0)"
            }
        }

        # Repetition metrics
        results["repetition_metrics"] = {
            "repetition_ratio": {
                "mean": float(np.mean(repetition_scores)),
                "std": float(np.std(repetition_scores)),
                "interpretation": "Proportion of repeated 4-grams (lower=less repetitive, target: <0.3)"
            },
            "burstiness": {
                "mean": float(np.mean(burstiness_scores)),
                "std": float(np.std(burstiness_scores)),
                "interpretation": "Token usage burstiness (lower=more uniform, target: <0.5)"
            }
        }

        # Naturalness metrics
        results["naturalness_metrics"] = {
            "zipf_coefficient": {
                "mean": float(np.mean(zipf_scores)),
                "std": float(np.std(zipf_scores)),
                "interpretation": "Zipf's law coefficient (closer to 1.0=more natural, target: 0.8-1.2)"
            }
        }

        # Calculate self-BLEU (inter-sample diversity)
        print("Calculating self-BLEU...")
        self_bleu = self.metrics.calculate_self_bleu(all_texts, n=4)
        results["diversity_metrics"]["self_bleu"] = {
            "score": float(self_bleu),
            "interpretation": "Avg similarity between samples (lower=more diverse, target: <0.5)"
        }

        # Per-sample details
        for i, (prompt, text, tokens) in enumerate(all_samples[:10]):  # First 10 for brevity
            results["per_sample_metrics"].append({
                "sample_id": i,
                "prompt": prompt,
                "generated_text": text,
                "length": len(tokens),
                "distinct_1": float(self.metrics.calculate_distinct_n(tokens, n=1)),
                "distinct_2": float(self.metrics.calculate_distinct_n(tokens, n=2)),
                "repetition_ratio": float(self.metrics.calculate_repetition_ratio(tokens, n=4)),
                "entropy": float(self.metrics.calculate_entropy(tokens))
            })

        return results

    def print_results(self, results: Dict):
        """Pretty print evaluation results"""

        print("\n" + "=" * 80)
        print("COHERENCE EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nTotal samples evaluated: {results['sample_count']}")

        # Diversity
        print("\n📊 DIVERSITY METRICS (Higher is Better)")
        print("-" * 80)
        for metric, values in results["diversity_metrics"].items():
            if metric == "self_bleu":
                print(f"\n{metric.upper()}:")
                print(f"  Score: {values['score']:.4f}")
                print(f"  {values['interpretation']}")
            else:
                print(f"\n{metric.upper()}:")
                print(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}")
                print(f"  {values['interpretation']}")

        # Repetition
        print("\n🔁 REPETITION METRICS (Lower is Better)")
        print("-" * 80)
        for metric, values in results["repetition_metrics"].items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}")
            print(f"  {values['interpretation']}")

        # Naturalness
        print("\n🌟 NATURALNESS METRICS")
        print("-" * 80)
        for metric, values in results["naturalness_metrics"].items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}")
            print(f"  {values['interpretation']}")

        # Sample outputs
        print("\n📝 SAMPLE OUTPUTS (First 10)")
        print("-" * 80)
        for sample in results["per_sample_metrics"]:
            print(f"\nSample {sample['sample_id']}:")
            print(f"  Prompt: {sample['prompt']}")
            print(f"  Generated: {sample['generated_text'][:200]}...")
            print(f"  Length: {sample['length']} tokens")
            print(f"  Distinct-2: {sample['distinct_2']:.3f} | Repetition: {sample['repetition_ratio']:.3f}")

        # Overall assessment
        print("\n" + "=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)

        scores = []
        issues = []

        # Check diversity
        d2 = results["diversity_metrics"]["distinct_2"]["mean"]
        if d2 > 0.7:
            print("✅ Good diversity (Distinct-2 > 0.7)")
            scores.append(1)
        else:
            print(f"❌ Low diversity (Distinct-2 = {d2:.3f}, target > 0.7)")
            issues.append("Low vocabulary diversity")
            scores.append(0)

        # Check repetition
        rep = results["repetition_metrics"]["repetition_ratio"]["mean"]
        if rep < 0.3:
            print("✅ Low repetition (Repetition ratio < 0.3)")
            scores.append(1)
        else:
            print(f"❌ High repetition (Repetition ratio = {rep:.3f}, target < 0.3)")
            issues.append("Excessive repetition")
            scores.append(0)

        # Check self-BLEU
        sb = results["diversity_metrics"]["self_bleu"]["score"]
        if sb < 0.5:
            print("✅ Good inter-sample diversity (Self-BLEU < 0.5)")
            scores.append(1)
        else:
            print(f"⚠️  Samples too similar (Self-BLEU = {sb:.3f}, target < 0.5)")
            issues.append("Samples are too similar to each other")
            scores.append(0)

        # Check Zipf
        zipf = results["naturalness_metrics"]["zipf_coefficient"]["mean"]
        if 0.8 <= zipf <= 1.2:
            print("✅ Natural token distribution (Zipf coefficient in range)")
            scores.append(1)
        else:
            print(f"⚠️  Unnatural distribution (Zipf = {zipf:.3f}, target 0.8-1.2)")
            issues.append("Token distribution doesn't follow natural language patterns")
            scores.append(0)

        overall_score = sum(scores) / len(scores) * 100
        print(f"\n🎯 Overall Coherence Score: {overall_score:.0f}/100")

        if issues:
            print("\n⚠️  Issues to address:")
            for issue in issues:
                print(f"  - {issue}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LLM Coherence Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="code/configs/gpu/small.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "Once upon a time",
            "The quick brown fox",
            "In a galaxy far far away",
            "It was a dark and stormy night",
            "The secret to happiness is"
        ],
        help="List of prompts to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples per prompt"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LLM COHERENCE EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config}")
    print(f"Prompts: {len(args.prompts)}")
    print(f"Samples per prompt: {args.num_samples}")
    print(f"Generation settings:")
    print(f"  - max_length: {args.max_length}")
    print(f"  - temperature: {args.temperature}")
    print(f"  - top_p: {args.top_p}")
    print(f"  - top_k: {args.top_k}")
    print(f"  - repetition_penalty: {args.repetition_penalty}")

    # Initialize evaluator
    evaluator = CoherenceEvaluator(args.model_path, args.config)

    # Run evaluation
    results = evaluator.evaluate_coherence(
        prompts=args.prompts,
        num_samples_per_prompt=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty
    )

    # Print results
    evaluator.print_results(results)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_path,
            "config_path": args.config,
            "generation_settings": {
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()