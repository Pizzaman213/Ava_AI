"""
Evaluation utilities for assessing model performance.

This module provides tools for evaluating trained Qwen MoE++ models on various
metrics including perplexity, accuracy, and generation quality. It supports
evaluation on data from /project/code/data/pretraining/processed/.

Key Features:
- Perplexity calculation for language modeling
- Token accuracy metrics
- Expert utilization analysis
- Generation quality assessment
- Batch evaluation support
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


class ModelEvaluator:
    """
    Comprehensive evaluator for Qwen MoE++ models.

    This evaluator provides various metrics to assess model performance including
    perplexity, accuracy, and expert utilization statistics. It's designed to work
    with models trained on data from /project/code/data/pretraining/processed/.

    Args:
        model: Trained Qwen MoE++ model
        tokenizer: Tokenizer for text processing
        device (torch.device): Device to run evaluation on

    Example:
        >>> evaluator = ModelEvaluator(model, tokenizer)
        >>> metrics = evaluator.evaluate(val_loader)
        >>> print(f"Perplexity: {metrics['perplexity']:.2f}")
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        compute_perplexity: bool = True,
        compute_accuracy: bool = True,
        compute_expert_stats: bool = True,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader containing evaluation data
            compute_perplexity (bool): Whether to compute perplexity
            compute_accuracy (bool): Whether to compute token accuracy
            compute_expert_stats (bool): Whether to compute expert utilization stats
            max_batches (int, optional): Maximum number of batches to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0
        expert_counts = {}

        progress_bar = tqdm(dataloader, desc="Evaluating", total=max_batches)

        for batch_idx, batch in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break

            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Compute loss
            if compute_perplexity:
                loss = outputs['loss']
                total_loss += loss.item() * attention_mask.sum().item()
                total_tokens += attention_mask.sum().item()

            # Compute accuracy
            if compute_accuracy:
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)

                # Only compute accuracy on non-padding tokens
                mask = labels != -100
                correct = (predictions[mask] == labels[mask]).sum().item()
                total = mask.sum().item()

                correct_predictions += correct
                total_predictions += total

            # Collect expert statistics
            if compute_expert_stats and 'expert_stats' in outputs:
                for key, value in outputs['expert_stats'].items():
                    if key not in expert_counts:
                        expert_counts[key] = []
                    expert_counts[key].append(value.cpu().numpy())

            # Update progress bar
            current_metrics = {}
            if compute_perplexity and total_tokens > 0:
                current_metrics['perplexity'] = np.exp(total_loss / total_tokens)
            if compute_accuracy and total_predictions > 0:
                current_metrics['accuracy'] = correct_predictions / total_predictions
            progress_bar.set_postfix(current_metrics)

        # Compute final metrics
        metrics = {}

        if compute_perplexity:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            metrics['loss'] = avg_loss
            metrics['perplexity'] = np.exp(avg_loss)

        if compute_accuracy:
            metrics['accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0

        if compute_expert_stats and expert_counts:
            # Aggregate expert statistics
            for key, values in expert_counts.items():
                concatenated = np.concatenate(values)
                metrics[f'expert_{key}_mean'] = concatenated.mean()
                metrics[f'expert_{key}_std'] = concatenated.std()

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
            prompts (List[str]): List of prompts to generate from
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k filtering parameter
            top_p (float): Top-p filtering parameter

        Returns:
            Dictionary containing generation metrics and samples
        """
        from ..generation import TextGenerator

        generator = TextGenerator(self.model, self.tokenizer, self.device)

        generations = []
        generation_times = []

        for prompt in tqdm(prompts, desc="Generating"):
            import time
            start_time = time.time()

            generated = generator.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )

            generation_time = time.time() - start_time
            generations.append(generated)
            generation_times.append(generation_time)

        # Compute metrics
        avg_length = np.mean([len(self.tokenizer.encode(g)) for g in generations])
        avg_time = np.mean(generation_times)

        # Compute diversity metrics
        unique_tokens = set()
        total_tokens = 0
        for gen in generations:
            tokens = self.tokenizer.encode(gen)
            unique_tokens.update(tokens)
            total_tokens += len(tokens)

        diversity = len(unique_tokens) / total_tokens if total_tokens > 0 else 0

        return {
            'average_length': avg_length,
            'average_time': avg_time,
            'diversity': diversity,
            'samples': generations[:5],  # Return first 5 samples
            'prompts': prompts[:5]
        }

    def analyze_expert_utilization(
        self,
        dataloader,
        max_batches: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze how experts are being utilized across the dataset.

        Args:
            dataloader: DataLoader containing evaluation data
            max_batches (int): Number of batches to analyze

        Returns:
            Dictionary containing expert utilization statistics
        """
        self.model.eval()

        expert_activations = []
        expert_loads = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing experts", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_expert_stats=True
            )

            if 'expert_stats' in outputs:
                stats = outputs['expert_stats']
                if 'expert_activations' in stats:
                    expert_activations.append(stats['expert_activations'].cpu().numpy())
                if 'expert_loads' in stats:
                    expert_loads.append(stats['expert_loads'].cpu().numpy())

        analysis = {}

        if expert_activations:
            activations = np.concatenate(expert_activations)
            analysis['activation_mean'] = activations.mean()
            analysis['activation_std'] = activations.std()
            analysis['activation_min'] = activations.min()
            analysis['activation_max'] = activations.max()

        if expert_loads:
            loads = np.concatenate(expert_loads)
            analysis['load_balance'] = loads.std() / (loads.mean() + 1e-8)
            analysis['load_distribution'] = {
                'mean': loads.mean(),
                'std': loads.std(),
                'min': loads.min(),
                'max': loads.max()
            }

        return analysis


class PerplexityEvaluator:
    """
    Specialized evaluator for computing perplexity efficiently.

    This evaluator is optimized for perplexity calculation on large datasets
    from /project/code/data/pretraining/processed/ with memory-efficient
    batch processing.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for text processing
        device (torch.device): Device to run evaluation on

    Example:
        >>> evaluator = PerplexityEvaluator(model, tokenizer)
        >>> ppl = evaluator.compute_perplexity(val_loader)
        >>> print(f"Validation perplexity: {ppl:.2f}")
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    @torch.no_grad()
    def compute_perplexity(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        stride: int = 512
    ) -> float:
        """
        Compute perplexity on a dataset.

        Args:
            dataloader: DataLoader containing evaluation data
            max_batches (int, optional): Maximum number of batches
            stride (int): Stride for sliding window evaluation

        Returns:
            Perplexity score
        """
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        progress_bar = tqdm(dataloader, desc="Computing perplexity", total=max_batches)

        for batch_idx, batch in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Create labels (shift input_ids by 1)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last token

            # Mask padding tokens
            labels[attention_mask == 0] = -100

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Update progress bar
            if total_tokens > 0:
                current_ppl = np.exp(total_loss / total_tokens)
                progress_bar.set_postfix({'perplexity': f'{current_ppl:.2f}'})

        # Compute final perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)

        return perplexity

    def compute_sliding_window_perplexity(
        self,
        text: str,
        max_length: int = 512,
        stride: int = 256
    ) -> float:
        """
        Compute perplexity using sliding window approach for long texts.

        Args:
            text (str): Text to evaluate
            max_length (int): Maximum window length
            stride (int): Stride between windows

        Returns:
            Perplexity score
        """
        self.model.eval()

        # Tokenize full text
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)

        seq_len = input_ids.size(1)

        total_loss = 0
        total_tokens = 0

        # Slide window through text
        for begin_loc in range(0, seq_len - 1, stride):
            end_loc = min(begin_loc + max_length, seq_len)

            # Extract window
            window_ids = input_ids[:, begin_loc:end_loc]
            window_labels = window_ids.clone()

            # Shift labels
            window_labels[:, :-1] = window_ids[:, 1:]
            window_labels[:, -1] = -100

            # Forward pass
            outputs = self.model(input_ids=window_ids, labels=window_labels)
            loss = outputs['loss']

            # Weight by number of tokens
            num_tokens = end_loc - begin_loc - 1
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        # Compute perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)

        return perplexity