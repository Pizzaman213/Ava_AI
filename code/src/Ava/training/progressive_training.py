"""
Progressive Training Framework for Enhanced LLM Training

This module implements advanced progressive training strategies including:
- GrowLength: Progressive sequence length scaling
- Curriculum Learning: Difficulty-based training progression
- Dynamic Batch Sizing: Adaptive batch size optimization
- Progressive Model Scaling: Dynamic architecture expansion

These techniques can provide 30-50% faster convergence while maintaining
or improving final model performance.

References:
- Curriculum Learning: https://arxiv.org/abs/0904.0130
- Progressive Growing: https://arxiv.org/abs/1710.10196
- Dynamic Batch Sizing: https://arxiv.org/abs/1711.00489
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ProgressiveTrainingConfig:
    """Configuration for progressive training strategies."""

    # GrowLength configuration
    enable_grow_length: bool = True
    initial_seq_length: int = 128
    final_seq_length: int = 2048
    length_schedule: str = "linear"  # "linear", "exponential", "step"
    length_growth_steps: int = 10000

    # Curriculum learning configuration
    enable_curriculum: bool = True
    curriculum_metric: str = "loss"  # "loss", "attention_entropy", "perplexity"
    curriculum_schedule: str = "root_decay"  # "linear", "root_decay", "exponential"
    difficulty_percentile: float = 0.1  # Start with easiest 10%
    curriculum_steps: int = 50000

    # Dynamic batch sizing
    enable_dynamic_batch: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_size_adaptation_steps: int = 100
    target_gpu_utilization: float = 0.85

    # Progressive model scaling
    enable_progressive_model: bool = False
    initial_layers: int = 6
    final_layers: int = 12
    layer_growth_schedule: str = "step"
    layer_growth_steps: int = 20000

    # General settings
    warmup_steps: int = 1000
    eval_frequency: int = 1000
    save_frequency: int = 5000


class CurriculumLearning:
    """
    Curriculum Learning implementation for LLM training.

    Orders training data from easy to hard based on various difficulty metrics.
    """

    def __init__(
        self,
        config: ProgressiveTrainingConfig,
        dataset,
        tokenizer,
        device: str = "cuda"
    ):
        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # Cache for difficulty scores
        self.difficulty_scores = {}
        self.sorted_indices = None
        self.current_percentile = config.difficulty_percentile

        # Metrics for difficulty assessment
        self.difficulty_metrics = {
            'loss': self._compute_loss_difficulty,
            'attention_entropy': self._compute_attention_difficulty,
            'perplexity': self._compute_perplexity_difficulty,
            'length': self._compute_length_difficulty,
            'vocabulary_diversity': self._compute_vocab_difficulty
        }

    def compute_difficulty_scores(self, model: nn.Module, batch_size: int = 32):
        """Compute difficulty scores for the entire dataset."""
        logger.info("Computing curriculum difficulty scores...")

        model.eval()
        scores = []

        # Create temporary dataloader
        temp_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(temp_loader):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Compute difficulty based on selected metric
                if self.config.curriculum_metric in self.difficulty_metrics:
                    batch_scores = self.difficulty_metrics[self.config.curriculum_metric](
                        model, batch
                    )
                    scores.extend(batch_scores)

                if batch_idx % 100 == 0:
                    logger.info(f"Processed {batch_idx * batch_size} examples")

        # Store scores and sort indices
        self.difficulty_scores = {i: score for i, score in enumerate(scores)}
        self.sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i]
        )

        logger.info(f"Computed difficulty scores for {len(scores)} examples")

    def _compute_loss_difficulty(self, model: nn.Module, batch: Dict) -> List[float]:
        """Compute difficulty based on loss values."""
        outputs = model(**batch)
        losses = outputs.loss

        # Convert to per-example losses if using reduction='mean'
        if losses.dim() == 0:  # scalar loss
            # Approximate per-example loss (this is a simplification)
            batch_size = batch['input_ids'].size(0)
            return [losses.item()] * batch_size
        else:
            return losses.cpu().tolist()

    def _compute_attention_difficulty(self, model: nn.Module, batch: Dict) -> List[float]:
        """Compute difficulty based on attention entropy."""
        outputs = model(**batch, output_attentions=True)

        # Compute attention entropy across all heads and layers
        batch_size = batch['input_ids'].size(0)
        difficulties = []

        for sample_idx in range(batch_size):
            total_entropy = 0.0
            attention_count = 0

            for layer_attentions in outputs.attentions:
                # layer_attentions: [batch, heads, seq_len, seq_len]
                sample_attention = layer_attentions[sample_idx]  # [heads, seq_len, seq_len]

                # Compute entropy for each head
                for head_idx in range(sample_attention.size(0)):
                    head_attention = sample_attention[head_idx]  # [seq_len, seq_len]

                    # Compute entropy: -sum(p * log(p))
                    entropy = -torch.sum(
                        head_attention * torch.log(head_attention + 1e-8),
                        dim=-1
                    ).mean()

                    total_entropy += entropy.item()
                    attention_count += 1

            avg_entropy = total_entropy / attention_count if attention_count > 0 else 0.0
            difficulties.append(avg_entropy)

        return difficulties

    def _compute_perplexity_difficulty(self, model: nn.Module, batch: Dict) -> List[float]:
        """Compute difficulty based on perplexity."""
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch['labels']

        # Compute per-example perplexity
        batch_size = logits.size(0)
        difficulties = []

        for sample_idx in range(batch_size):
            sample_logits = logits[sample_idx]  # [seq_len, vocab_size]
            sample_labels = labels[sample_idx]   # [seq_len]

            # Mask padding tokens
            mask = sample_labels != -100
            if mask.sum() == 0:
                difficulties.append(0.0)
                continue

            # Compute cross-entropy loss
            sample_logits = sample_logits[mask]
            sample_labels = sample_labels[mask]

            loss = nn.functional.cross_entropy(
                sample_logits, sample_labels, reduction='mean'
            )

            # Convert to perplexity
            perplexity = torch.exp(loss).item()
            difficulties.append(perplexity)

        return difficulties

    def _compute_length_difficulty(self, model: nn.Module, batch: Dict) -> List[float]:
        """Compute difficulty based on sequence length."""
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)

        difficulties = []
        for sample_idx in range(batch_size):
            # Count non-padding tokens
            sample_length = (input_ids[sample_idx] != self.tokenizer.pad_token_id).sum().item()
            difficulties.append(float(sample_length))

        return difficulties

    def _compute_vocab_difficulty(self, model: nn.Module, batch: Dict) -> List[float]:
        """Compute difficulty based on vocabulary diversity."""
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)

        difficulties = []
        for sample_idx in range(batch_size):
            sample_tokens = input_ids[sample_idx]
            # Remove padding tokens
            sample_tokens = sample_tokens[sample_tokens != self.tokenizer.pad_token_id]

            # Compute vocabulary diversity (unique tokens / total tokens)
            unique_tokens = len(torch.unique(sample_tokens))
            total_tokens = len(sample_tokens)

            diversity = unique_tokens / total_tokens if total_tokens > 0 else 0.0
            # Higher diversity = higher difficulty
            difficulties.append(diversity)

        return difficulties

    def get_curriculum_subset(self, step: int, total_steps: int) -> List[int]:
        """Get indices for current curriculum subset."""
        if self.sorted_indices is None:
            logger.warning("Difficulty scores not computed. Using random order.")
            return list(range(len(self.dataset)))

        # Compute current percentile based on schedule
        progress = step / total_steps

        if self.config.curriculum_schedule == "linear":
            current_percentile = self.config.difficulty_percentile + progress * (1.0 - self.config.difficulty_percentile)
        elif self.config.curriculum_schedule == "root_decay":
            current_percentile = 1.0 - (1.0 - self.config.difficulty_percentile) * math.sqrt(1.0 - progress)
        elif self.config.curriculum_schedule == "exponential":
            current_percentile = 1.0 - (1.0 - self.config.difficulty_percentile) * math.exp(-3 * progress)
        else:
            current_percentile = 1.0  # Use all data

        # Get subset of indices
        subset_size = int(len(self.sorted_indices) * current_percentile)
        subset_indices = self.sorted_indices[:subset_size]

        logger.info(f"Step {step}: Using {current_percentile:.2%} of data ({subset_size} examples)")

        return subset_indices


class GrowLengthScheduler:
    """
    Progressive sequence length scaling (GrowLength).

    Gradually increases sequence length during training for faster convergence.
    """

    def __init__(self, config: ProgressiveTrainingConfig):
        self.config = config
        self.current_length = config.initial_seq_length

    def get_sequence_length(self, step: int) -> int:
        """Get current sequence length based on training step."""
        if not self.config.enable_grow_length:
            return self.config.final_seq_length

        if step >= self.config.length_growth_steps:
            return self.config.final_seq_length

        progress = step / self.config.length_growth_steps

        if self.config.length_schedule == "linear":
            length = self.config.initial_seq_length + progress * (
                self.config.final_seq_length - self.config.initial_seq_length
            )
        elif self.config.length_schedule == "exponential":
            # Exponential growth
            ratio = self.config.final_seq_length / self.config.initial_seq_length
            length = self.config.initial_seq_length * (ratio ** progress)
        elif self.config.length_schedule == "step":
            # Step-wise growth
            num_steps = 4  # Number of discrete steps
            step_size = (self.config.final_seq_length - self.config.initial_seq_length) / num_steps
            step_idx = int(progress * num_steps)
            length = self.config.initial_seq_length + step_idx * step_size
        else:
            length = self.config.final_seq_length

        # Round to nearest multiple of 8 for efficiency
        length = int(length)
        length = ((length + 7) // 8) * 8

        # Ensure within bounds
        length = max(self.config.initial_seq_length, min(length, self.config.final_seq_length))

        if length != self.current_length:
            logger.info(f"Step {step}: Sequence length updated to {length}")
            self.current_length = length

        return length


class DynamicBatchSizer:
    """
    Dynamic batch size optimization based on GPU utilization.

    Automatically adjusts batch size to maximize GPU utilization while
    avoiding out-of-memory errors.
    """

    def __init__(self, config: ProgressiveTrainingConfig):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.utilization_history = []
        self.oom_batch_sizes = set()
        self.successful_batch_sizes = {}

    def get_batch_size(self, step: int, current_seq_length: int) -> int:
        """Get optimal batch size for current step and sequence length."""
        if not self.config.enable_dynamic_batch:
            return self.config.max_batch_size

        # Adjust batch size every N steps
        if step % self.config.batch_size_adaptation_steps == 0 and step > 0:
            self._adapt_batch_size(current_seq_length)

        return self.current_batch_size

    def _adapt_batch_size(self, seq_length: int):
        """Adapt batch size based on recent performance."""
        # Get recent GPU utilization
        if len(self.utilization_history) < 5:
            return  # Not enough data

        avg_utilization = np.mean(self.utilization_history[-5:])

        # Create a key for current configuration
        config_key = (seq_length, self.current_batch_size)

        # Decision logic
        if avg_utilization < self.config.target_gpu_utilization - 0.1:
            # Low utilization - try to increase batch size
            new_batch_size = min(
                self.current_batch_size * 2,
                self.config.max_batch_size
            )

            # Check if this configuration has caused OOM before
            oom_key = (seq_length, new_batch_size)
            if oom_key not in self.oom_batch_sizes:
                self.current_batch_size = new_batch_size
                logger.info(f"Increased batch size to {self.current_batch_size} (utilization: {avg_utilization:.2%})")

        elif avg_utilization > self.config.target_gpu_utilization + 0.05:
            # High utilization - might want to decrease for stability
            new_batch_size = max(
                self.current_batch_size // 2,
                self.config.min_batch_size
            )
            self.current_batch_size = new_batch_size
            logger.info(f"Decreased batch size to {self.current_batch_size} (utilization: {avg_utilization:.2%})")

        # Record successful configuration
        self.successful_batch_sizes[config_key] = avg_utilization

    def report_oom(self, seq_length: int, batch_size: int):
        """Report an out-of-memory error for given configuration."""
        oom_key = (seq_length, batch_size)
        self.oom_batch_sizes.add(oom_key)

        # Reduce batch size immediately
        new_batch_size = max(batch_size // 2, self.config.min_batch_size)
        self.current_batch_size = new_batch_size

        logger.warning(f"OOM detected. Reduced batch size to {self.current_batch_size}")

    def update_utilization(self, utilization: float):
        """Update GPU utilization history."""
        self.utilization_history.append(utilization)

        # Keep only recent history
        if len(self.utilization_history) > 100:
            self.utilization_history = self.utilization_history[-50:]


class ProgressiveModelScaler:
    """
    Progressive model scaling during training.

    Gradually increases model capacity (layers, heads, etc.) during training.
    This is experimental and requires careful implementation.
    """

    def __init__(self, config: ProgressiveTrainingConfig, initial_model: nn.Module):
        self.config = config
        self.initial_model = initial_model
        self.current_layers = config.initial_layers

    def should_scale_model(self, step: int) -> bool:
        """Check if model should be scaled at current step."""
        if not self.config.enable_progressive_model:
            return False

        if self.current_layers >= self.config.final_layers:
            return False

        # Check if it's time to add a layer
        if self.config.layer_growth_schedule == "step":
            layers_to_add = self.config.final_layers - self.config.initial_layers
            steps_per_layer = self.config.layer_growth_steps // layers_to_add

            return step % steps_per_layer == 0 and step > 0

        return False

    def scale_model(self, model: nn.Module) -> nn.Module:
        """Add a new layer to the model."""
        # This is a simplified implementation
        # Real implementation would depend on specific model architecture
        logger.info(f"Scaling model from {self.current_layers} to {self.current_layers + 1} layers")

        # Add new layer (implementation specific)
        # This would require careful parameter initialization and optimizer state management

        self.current_layers += 1
        return model


class ProgressiveTrainer:
    """
    Main progressive training coordinator.

    Orchestrates all progressive training strategies including curriculum learning,
    sequence length growth, dynamic batch sizing, and optional model scaling.
    """

    def __init__(
        self,
        config: ProgressiveTrainingConfig,
        model: nn.Module,
        dataset,
        tokenizer,
        device: str = "cuda"
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # Initialize components
        self.curriculum = CurriculumLearning(config, dataset, tokenizer, device) if config.enable_curriculum else None
        self.length_scheduler = GrowLengthScheduler(config)
        self.batch_sizer = DynamicBatchSizer(config)
        self.model_scaler = ProgressiveModelScaler(config, model) if config.enable_progressive_model else None

        # Training state
        self.current_step = 0
        self.training_metrics = defaultdict(list)

    def setup_curriculum(self, batch_size: int = 32):
        """Setup curriculum learning by computing difficulty scores."""
        if self.curriculum:
            self.curriculum.compute_difficulty_scores(self.model, batch_size)

    def get_dataloader(self, step: int, total_steps: int) -> DataLoader:
        """Get dataloader with current progressive training settings."""
        # Get current sequence length
        seq_length = self.length_scheduler.get_sequence_length(step)

        # Get current batch size
        batch_size = self.batch_sizer.get_batch_size(step, seq_length)

        # Get curriculum subset if enabled
        if self.curriculum:
            indices = self.curriculum.get_curriculum_subset(step, total_steps)
            # Create subset dataset
            subset_dataset = torch.utils.data.Subset(self.dataset, indices)
        else:
            subset_dataset = self.dataset

        # Create dataloader with current settings
        dataloader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        return dataloader

    def preprocess_batch(self, batch: Dict, step: int) -> Dict:
        """Preprocess batch according to current training settings."""
        seq_length = self.length_scheduler.get_sequence_length(step)

        # Truncate or pad sequences to current length
        if 'input_ids' in batch:
            current_length = batch['input_ids'].size(1)

            if current_length > seq_length:
                # Truncate
                for key in ['input_ids', 'attention_mask', 'labels']:
                    if key in batch:
                        batch[key] = batch[key][:, :seq_length]
            elif current_length < seq_length:
                # Pad
                pad_length = seq_length - current_length
                for key in ['input_ids', 'attention_mask']:
                    if key in batch:
                        pad_value = self.tokenizer.pad_token_id if key == 'input_ids' else 0
                        padding = torch.full(
                            (batch[key].size(0), pad_length),
                            pad_value,
                            dtype=batch[key].dtype,
                            device=batch[key].device
                        )
                        batch[key] = torch.cat([batch[key], padding], dim=1)

                # Pad labels with -100 (ignore index)
                if 'labels' in batch:
                    padding = torch.full(
                        (batch['labels'].size(0), pad_length),
                        -100,
                        dtype=batch['labels'].dtype,
                        device=batch['labels'].device
                    )
                    batch['labels'] = torch.cat([batch['labels'], padding], dim=1)

        return batch

    def step(self, step: int, total_steps: int) -> Dict[str, Any]:
        """Perform a progressive training step."""
        self.current_step = step

        # Check if model should be scaled
        if self.model_scaler and self.model_scaler.should_scale_model(step):
            self.model = self.model_scaler.scale_model(self.model)

        # Get current training settings
        seq_length = self.length_scheduler.get_sequence_length(step)
        batch_size = self.batch_sizer.get_batch_size(step, seq_length)

        # Return current configuration
        config = {
            'sequence_length': seq_length,
            'batch_size': batch_size,
            'current_layers': getattr(self.model_scaler, 'current_layers', None),
            'curriculum_percentile': getattr(self.curriculum, 'current_percentile', 1.0)
        }

        return config

    def handle_oom(self, step: int):
        """Handle out-of-memory error."""
        seq_length = self.length_scheduler.get_sequence_length(step)
        batch_size = self.batch_sizer.current_batch_size

        self.batch_sizer.report_oom(seq_length, batch_size)

    def update_metrics(self, metrics: Dict[str, float]):
        """Update training metrics for adaptation."""
        for key, value in metrics.items():
            self.training_metrics[key].append(value)

        # Update GPU utilization if available
        if 'gpu_utilization' in metrics:
            self.batch_sizer.update_utilization(metrics['gpu_utilization'])

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of progressive training progress."""
        summary = {
            'current_step': self.current_step,
            'current_seq_length': self.length_scheduler.current_length,
            'current_batch_size': self.batch_sizer.current_batch_size,
            'oom_configurations': len(self.batch_sizer.oom_batch_sizes),
            'successful_configurations': len(self.batch_sizer.successful_batch_sizes),
        }

        if self.curriculum:
            summary['difficulty_scores_computed'] = len(self.curriculum.difficulty_scores) > 0
            summary['current_percentile'] = getattr(self.curriculum, 'current_percentile', 1.0)

        if self.model_scaler:
            summary['current_layers'] = self.model_scaler.current_layers

        return summary


def create_progressive_trainer(
    model: nn.Module,
    dataset,
    tokenizer,
    config_dict: Optional[Dict[str, Any]] = None,
    device: str = "cuda"
) -> ProgressiveTrainer:
    """
    Factory function to create a progressive trainer with sensible defaults.

    Args:
        model: PyTorch model to train
        dataset: Training dataset
        tokenizer: Tokenizer for the model
        config_dict: Optional configuration overrides
        device: Device to train on

    Returns:
        Configured ProgressiveTrainer instance
    """
    # Create config with defaults
    config = ProgressiveTrainingConfig()

    # Apply any overrides
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create and return trainer
    trainer = ProgressiveTrainer(config, model, dataset, tokenizer, device)

    return trainer