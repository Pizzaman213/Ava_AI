"""
Unified Quality Evaluation Module for Training.

This module consolidates quality evaluation into a single interface that
coordinates generation, coherence measurement, and quality scoring.

Components:
- GenerationManager: Text generation for quality monitoring
- CoherenceMeasurer: Coherence metrics (perplexity, repetition, flow)
- QualityScorer: Weighted quality score computation
- QualityEvaluator: Unified interface coordinating all components

Usage:
    from ava.training.quality_evaluator import QualityEvaluator

    evaluator = QualityEvaluator(context)
    evaluator.initialize()

    # Full evaluation
    metrics = evaluator.evaluate(model, batch)

    # Individual components
    generated = evaluator.generate_sample(model, tokenizer)
    coherence = evaluator.measure_coherence(model, batch)
    score = evaluator.compute_quality_score(val_loss=0.5, coherence=0.8)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .context import ManagerInterface, TrainingContext

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for all quality evaluation metrics."""

    # Generation metrics
    generated_text: Optional[str] = None
    generation_time_ms: float = 0.0

    # Coherence metrics
    perplexity: float = 0.0
    repetition_score: float = 0.0
    sentence_flow_score: float = 0.0
    topic_consistency: float = 0.0
    coherence_score: float = 0.0

    # Validation metrics
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None

    # Composite quality score
    quality_score: float = 0.0

    # Metadata
    step: int = 0
    epoch: int = 0
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            'quality/overall_score': self.quality_score,
            'quality/perplexity': self.perplexity,
            'quality/coherence_score': self.coherence_score,
            'quality/repetition_score': self.repetition_score,
            'quality/step': self.step,
            'quality/epoch': self.epoch,
        }
        if self.val_loss is not None:
            result['quality/val_loss'] = self.val_loss
        if self.generated_text is not None:
            result['quality/generated_sample'] = self.generated_text[:200]
        return result


class QualityEvaluator(ManagerInterface):
    """
    Unified interface for quality evaluation during training.

    Coordinates:
        - GenerationManager: Text generation for monitoring
        - CoherenceMeasurer: Coherence metric computation
        - QualityScorer: Weighted quality score calculation

    This class provides a single entry point for all quality-related
    operations, reducing code duplication and simplifying the training loop.

    Example:
        >>> context = TrainingContext(model=model, device=device)
        >>> evaluator = QualityEvaluator(context)
        >>> evaluator.initialize()

        >>> # Full evaluation
        >>> metrics = evaluator.evaluate(model, batch, tokenizer)

        >>> # Quick coherence check
        >>> coherence = evaluator.measure_coherence(model, batch)
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the quality evaluator.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)

        # Lazy-loaded components
        self._generation_manager = None
        self._coherence_measurer = None
        self._quality_scorer = None

        # Configuration
        self._evaluation_interval = 1000
        self._enable_generation = True
        self._enable_coherence = True

    def initialize(self) -> None:
        """Initialize all quality evaluation components."""
        self._initialized = True
        self.logger.debug("QualityEvaluator initialized")

    def cleanup(self) -> None:
        """Cleanup all components."""
        if self._generation_manager is not None:
            self._generation_manager.cleanup()

    @property
    def generation_manager(self):
        """Lazy-load generation manager."""
        if self._generation_manager is None:
            from .generation import GenerationManager
            self._generation_manager = GenerationManager(self.context)
            self._generation_manager.initialize()
        return self._generation_manager

    @property
    def coherence_measurer(self):
        """Lazy-load coherence measurer."""
        if self._coherence_measurer is None:
            from .coherence import FastCoherenceMeasurer
            self._coherence_measurer = FastCoherenceMeasurer()
        return self._coherence_measurer

    @property
    def quality_scorer(self):
        """Lazy-load quality scorer."""
        if self._quality_scorer is None:
            from .validation import QualityScorer
            from ..config.training_config import ModelSelectionConfig
            config = ModelSelectionConfig()
            self._quality_scorer = QualityScorer(config)
        return self._quality_scorer

    def evaluate(
        self,
        model: nn.Module,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        tokenizer: Optional[Any] = None,
        val_loss: Optional[float] = None,
        step: int = 0,
        epoch: int = 0,
        generate: bool = True,
        measure_coherence: bool = True,
    ) -> QualityMetrics:
        """
        Perform full quality evaluation.

        Args:
            model: Model to evaluate
            batch: Batch of data for coherence evaluation
            tokenizer: Tokenizer for generation
            val_loss: Validation loss (if already computed)
            step: Current training step
            epoch: Current epoch
            generate: Whether to generate sample text
            measure_coherence: Whether to measure coherence

        Returns:
            QualityMetrics with all evaluation results
        """
        self.assert_initialized()

        import time
        start_time = time.time()

        metrics = QualityMetrics(step=step, epoch=epoch)

        # Validation loss
        if val_loss is not None:
            metrics.val_loss = val_loss

        # Generate sample
        if generate and tokenizer is not None and self._enable_generation:
            try:
                generated = self.generate_sample(model, tokenizer)
                if generated:
                    metrics.generated_text = generated
            except Exception as e:
                self.logger.warning(f"Generation failed: {e}")

        # Measure coherence
        if measure_coherence and batch is not None and self._enable_coherence:
            try:
                coherence = self.measure_coherence(model, batch)
                metrics.perplexity = coherence.get('perplexity', 0.0)
                metrics.repetition_score = coherence.get('repetition_score', 0.0)
                metrics.sentence_flow_score = coherence.get('sentence_flow', 0.0)
                metrics.topic_consistency = coherence.get('topic_consistency', 0.0)
                metrics.coherence_score = coherence.get('coherence_score', 0.0)
            except Exception as e:
                self.logger.warning(f"Coherence measurement failed: {e}")

        # Compute quality score
        try:
            quality = self.compute_quality_score(
                val_loss=metrics.val_loss,
                coherence_score=metrics.coherence_score,
                perplexity=metrics.perplexity,
                step=step,
                epoch=epoch,
            )
            metrics.quality_score = quality.get('composite_score', 0.0)
        except Exception as e:
            self.logger.warning(f"Quality scoring failed: {e}")

        metrics.evaluation_time_ms = (time.time() - start_time) * 1000

        return metrics

    def generate_sample(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt: str = "Once upon a time",
        max_length: int = 100,
        temperature: float = 0.8,
    ) -> Optional[str]:
        """
        Generate a sample text for quality monitoring.

        Args:
            model: Model to use for generation
            tokenizer: Tokenizer for encoding/decoding
            prompt: Starting prompt
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated text or None on failure
        """
        return self.generation_manager.generate_sample(
            model=model,
            tokenizer=tokenizer,
            config={
                'prompts': [prompt],
                'max_length': max_length,
                'temperature': temperature,
            }
        )

    def measure_coherence(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Measure coherence metrics for a batch.

        Args:
            model: Model to evaluate
            batch: Batch with input_ids and labels

        Returns:
            Dictionary of coherence metrics
        """
        coherence_result = self.coherence_measurer.measure(
            model=model,
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels'),
        )

        if hasattr(coherence_result, 'to_dict'):
            return coherence_result.to_dict()
        return coherence_result if isinstance(coherence_result, dict) else {}

    def compute_quality_score(
        self,
        val_loss: Optional[float] = None,
        coherence_score: Optional[float] = None,
        perplexity: Optional[float] = None,
        step: int = 0,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Compute weighted quality score.

        Args:
            val_loss: Validation loss
            coherence_score: Coherence score (0-1)
            perplexity: Model perplexity
            step: Current step
            epoch: Current epoch

        Returns:
            Dictionary with normalized scores and composite
        """
        score = self.quality_scorer.compute(
            val_loss=val_loss,
            coherence_score=coherence_score,
            perplexity=perplexity,
            step=step,
            epoch=epoch,
        )

        return {
            'composite_score': score.composite_score if hasattr(score, 'composite_score') else 0.0,
            'val_loss_normalized': getattr(score, 'val_loss_normalized', 0.0),
            'coherence_normalized': getattr(score, 'coherence_normalized', 0.0),
            'perplexity_normalized': getattr(score, 'perplexity_normalized', 0.0),
        }

    def should_evaluate(self, step: int) -> bool:
        """Check if evaluation should run at this step."""
        return step > 0 and step % self._evaluation_interval == 0

    def set_evaluation_interval(self, interval: int) -> None:
        """Set the evaluation interval in steps."""
        self._evaluation_interval = max(1, interval)

    def enable_generation(self, enabled: bool = True) -> None:
        """Enable or disable sample generation."""
        self._enable_generation = enabled

    def enable_coherence(self, enabled: bool = True) -> None:
        """Enable or disable coherence measurement."""
        self._enable_coherence = enabled


# Convenience factory function
def create_quality_evaluator(context: TrainingContext) -> QualityEvaluator:
    """
    Create and initialize a quality evaluator.

    Args:
        context: Training context

    Returns:
        Initialized QualityEvaluator
    """
    evaluator = QualityEvaluator(context)
    evaluator.initialize()
    return evaluator


__all__ = [
    'QualityMetrics',
    'QualityEvaluator',
    'create_quality_evaluator',
]
