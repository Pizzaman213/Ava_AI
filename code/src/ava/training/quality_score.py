"""
Model Quality Score for multi-metric model selection.

Combines val_loss, coherence_score, and perplexity into a single
quality metric for determining the best model checkpoint.

Example:
    >>> from ava.config.training_config import ModelSelectionConfig
    >>> config = ModelSelectionConfig(val_loss_weight=0.5, coherence_score_weight=0.3)
    >>> scorer = QualityScorer(config)
    >>> score = scorer.compute(val_loss=2.5, coherence_score=0.75, perplexity=25.0)
    >>> print(f"Quality: {score.quality_score:.4f}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelQualityScore:
    """Container for model quality metrics and composite score.

    Stores both raw metrics and normalized values (0-1 scale, higher is better).
    The quality_score is a weighted combination of normalized metrics.

    Attributes:
        val_loss: Raw validation loss (lower is better)
        coherence_score: Raw coherence score 0-1 (higher is better)
        perplexity: Raw perplexity (lower is better)
        val_loss_normalized: Normalized val_loss 0-1 (higher is better)
        coherence_normalized: Normalized coherence 0-1 (higher is better)
        perplexity_normalized: Normalized perplexity 0-1 (higher is better)
        quality_score: Weighted composite score 0-1 (higher is better)
        step: Training step when this score was computed
        epoch: Training epoch when this score was computed
        timestamp: Unix timestamp when this score was computed
        metrics_present: Dict indicating which metrics were available
    """

    # Raw metrics
    val_loss: Optional[float] = None
    coherence_score: Optional[float] = None
    perplexity: Optional[float] = None

    # Normalized metrics (0-1 scale, higher is better)
    val_loss_normalized: float = 0.0
    coherence_normalized: float = 0.0
    perplexity_normalized: float = 0.0

    # Composite score (weighted combination)
    quality_score: float = 0.0

    # Metadata
    step: int = 0
    epoch: int = 0
    timestamp: float = field(default_factory=time.time)
    metrics_present: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and checkpointing.

        Returns:
            Dictionary with all score fields suitable for JSON serialization.
        """
        return {
            'quality_score': self.quality_score,
            'val_loss': self.val_loss,
            'coherence_score': self.coherence_score,
            'perplexity': self.perplexity,
            'val_loss_normalized': self.val_loss_normalized,
            'coherence_normalized': self.coherence_normalized,
            'perplexity_normalized': self.perplexity_normalized,
            'step': self.step,
            'epoch': self.epoch,
            'timestamp': self.timestamp,
            'metrics_present': self.metrics_present,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelQualityScore':
        """Create ModelQualityScore from dictionary.

        Args:
            data: Dictionary with score fields (e.g., from checkpoint)

        Returns:
            ModelQualityScore instance
        """
        return cls(
            val_loss=data.get('val_loss'),
            coherence_score=data.get('coherence_score'),
            perplexity=data.get('perplexity'),
            val_loss_normalized=data.get('val_loss_normalized', 0.0),
            coherence_normalized=data.get('coherence_normalized', 0.0),
            perplexity_normalized=data.get('perplexity_normalized', 0.0),
            quality_score=data.get('quality_score', 0.0),
            step=data.get('step', 0),
            epoch=data.get('epoch', 0),
            timestamp=data.get('timestamp', time.time()),
            metrics_present=data.get('metrics_present', {}),
        )

    def __gt__(self, other: 'ModelQualityScore') -> bool:
        """Compare quality scores (higher is better).

        Args:
            other: Another ModelQualityScore to compare against

        Returns:
            True if this score is higher (better) than other
        """
        if not isinstance(other, ModelQualityScore):
            return NotImplemented
        return self.quality_score > other.quality_score

    def __ge__(self, other: 'ModelQualityScore') -> bool:
        """Compare quality scores (higher is better)."""
        if not isinstance(other, ModelQualityScore):
            return NotImplemented
        return self.quality_score >= other.quality_score

    def __lt__(self, other: 'ModelQualityScore') -> bool:
        """Compare quality scores (higher is better)."""
        if not isinstance(other, ModelQualityScore):
            return NotImplemented
        return self.quality_score < other.quality_score

    def __le__(self, other: 'ModelQualityScore') -> bool:
        """Compare quality scores (higher is better)."""
        if not isinstance(other, ModelQualityScore):
            return NotImplemented
        return self.quality_score <= other.quality_score

    def __repr__(self) -> str:
        """String representation."""
        parts = [f"quality={self.quality_score:.4f}"]
        if self.val_loss is not None:
            parts.append(f"val_loss={self.val_loss:.4f}")
        if self.coherence_score is not None:
            parts.append(f"coherence={self.coherence_score:.4f}")
        if self.perplexity is not None:
            parts.append(f"ppl={self.perplexity:.2f}")
        return f"ModelQualityScore({', '.join(parts)})"


class QualityScorer:
    """
    Computes weighted quality scores from multiple metrics.

    Handles:
    - Normalization of different metric scales
    - Inversion of lower-is-better metrics (val_loss, perplexity)
    - Weighted combination
    - Missing metric handling with fallback options

    Example:
        >>> from ava.config.training_config import ModelSelectionConfig
        >>> config = ModelSelectionConfig()
        >>> scorer = QualityScorer(config)
        >>> score = scorer.compute(val_loss=2.5, coherence_score=0.8, perplexity=30.0)
        >>> is_better = scorer.is_improvement(score, previous_best_score)
    """

    def __init__(self, config: 'ModelSelectionConfig'):
        """Initialize the quality scorer.

        Args:
            config: ModelSelectionConfig with weights and caps
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration weights."""
        total_weight = (
            self.config.val_loss_weight +
            self.config.coherence_score_weight +
            self.config.perplexity_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Model selection weights sum to {total_weight:.2f}, not 1.0. "
                "Scores will be normalized by actual weights used."
            )

    def compute(
        self,
        val_loss: Optional[float] = None,
        coherence_score: Optional[float] = None,
        perplexity: Optional[float] = None,
        step: int = 0,
        epoch: int = 0,
    ) -> ModelQualityScore:
        """Compute quality score from available metrics.

        All metrics are optional. The score is computed from whatever
        metrics are provided, with weights normalized accordingly.

        Args:
            val_loss: Validation loss (lower is better)
            coherence_score: Coherence score 0-1 (higher is better)
            perplexity: Perplexity (lower is better)
            step: Current training step
            epoch: Current training epoch

        Returns:
            ModelQualityScore with normalized values and composite score
        """
        score = ModelQualityScore(
            val_loss=val_loss,
            coherence_score=coherence_score,
            perplexity=perplexity,
            step=step,
            epoch=epoch,
        )

        # Track which metrics are present
        score.metrics_present = {
            'val_loss': val_loss is not None,
            'coherence_score': coherence_score is not None,
            'perplexity': perplexity is not None,
        }

        # Normalize metrics to 0-1 scale (higher is better)
        if val_loss is not None:
            # Invert and cap: lower val_loss = higher score
            capped = min(max(val_loss, 0.0), self.config.val_loss_cap)
            score.val_loss_normalized = 1.0 - (capped / self.config.val_loss_cap)

        if coherence_score is not None:
            # Already 0-1, higher is better - just clamp
            score.coherence_normalized = max(0.0, min(1.0, coherence_score))

        if perplexity is not None:
            # Invert and cap: lower perplexity = higher score
            capped = min(max(perplexity, 1.0), self.config.perplexity_cap)
            # Use log scale for perplexity (more intuitive)
            import math
            log_capped = math.log(capped)
            log_cap = math.log(self.config.perplexity_cap)
            score.perplexity_normalized = 1.0 - (log_capped / log_cap)

        # Compute weighted combination
        total_weight = 0.0
        weighted_sum = 0.0

        if val_loss is not None:
            weighted_sum += self.config.val_loss_weight * score.val_loss_normalized
            total_weight += self.config.val_loss_weight

        if coherence_score is not None:
            weighted_sum += self.config.coherence_score_weight * score.coherence_normalized
            total_weight += self.config.coherence_score_weight

        if perplexity is not None:
            weighted_sum += self.config.perplexity_weight * score.perplexity_normalized
            total_weight += self.config.perplexity_weight

        # Normalize by actual weights used
        if total_weight > 0:
            score.quality_score = weighted_sum / total_weight
        else:
            score.quality_score = 0.0
            logger.warning("No metrics available for quality score computation")

        return score

    def is_improvement(
        self,
        current: ModelQualityScore,
        best: Optional[ModelQualityScore],
    ) -> bool:
        """Check if current score is an improvement over best.

        Args:
            current: Current quality score
            best: Previous best quality score (None if first evaluation)

        Returns:
            True if current is better than best
        """
        if best is None:
            return True

        if self.config.higher_is_better:
            return current.quality_score > best.quality_score
        else:
            return current.quality_score < best.quality_score

    def compute_from_coherence_metrics(
        self,
        val_loss: float,
        coherence_metrics: Optional[Dict[str, Any]],
        step: int = 0,
        epoch: int = 0,
    ) -> ModelQualityScore:
        """Compute quality score from validation loss and coherence metrics dict.

        This is a convenience method that extracts coherence_score and perplexity
        from a coherence metrics dictionary (as returned by CoherenceMeasurer).

        Args:
            val_loss: Validation loss
            coherence_metrics: Dict with 'coherence_score' and 'perplexity' keys
            step: Current training step
            epoch: Current training epoch

        Returns:
            ModelQualityScore with composite score
        """
        coherence_score = None
        perplexity = None

        if coherence_metrics:
            coherence_score = coherence_metrics.get('coherence_score')
            perplexity = coherence_metrics.get('perplexity')

        return self.compute(
            val_loss=val_loss,
            coherence_score=coherence_score,
            perplexity=perplexity,
            step=step,
            epoch=epoch,
        )
