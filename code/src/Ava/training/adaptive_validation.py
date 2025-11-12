"""
Adaptive Validation Scheduler for optimizing validation frequency based on training stability.

This module implements an adaptive validation scheduler that adjusts validation frequency
based on loss stability, reducing unnecessary validation overhead during stable training.
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AdaptiveValidationScheduler:
    """
    Adaptive validation scheduler that adjusts evaluation frequency based on training stability.

    This scheduler reduces validation overhead by:
    1. Running validation frequently during unstable training (high loss variance)
    2. Reducing validation frequency during stable training (low loss variance)
    3. Forcing validation at critical points (checkpoints, end of training)

    Expected savings: 5-15% of total training time
    """

    def __init__(
        self,
        min_eval_steps: int = 500,
        max_eval_steps: int = 5000,
        stability_window: int = 100,
        stability_threshold: float = 0.01,
        warmup_steps: int = 1000,
        force_eval_at_checkpoints: bool = True,
        adaptive_mode: str = "stability"  # "stability", "improvement", or "hybrid"
    ):
        """
        Initialize the adaptive validation scheduler.

        Args:
            min_eval_steps: Minimum steps between evaluations
            max_eval_steps: Maximum steps between evaluations
            stability_window: Number of recent losses to track for stability
            stability_threshold: Loss variance threshold for stability detection
            warmup_steps: Steps before adaptive scheduling starts
            force_eval_at_checkpoints: Always evaluate when saving checkpoints
            adaptive_mode: Mode for adaptation logic
        """
        self.min_eval_steps = min_eval_steps
        self.max_eval_steps = max_eval_steps
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.warmup_steps = warmup_steps
        self.force_eval_at_checkpoints = force_eval_at_checkpoints
        self.adaptive_mode = adaptive_mode

        # State tracking
        self.loss_history = deque(maxlen=stability_window)
        self.val_loss_history = deque(maxlen=10)  # Track validation losses
        self.next_eval_step = min_eval_steps
        self.last_eval_step = 0
        self.total_evals = 0
        self.skipped_evals = 0
        self.current_eval_interval = min_eval_steps

        # Metrics for monitoring
        self.stability_score = 1.0  # 1.0 = unstable, 0.0 = very stable
        self.improvement_rate = 0.0  # Rate of loss improvement

    def should_evaluate(
        self,
        step: int,
        current_loss: float,
        is_checkpoint: bool = False,
        is_final: bool = False
    ) -> bool:
        """
        Determine if evaluation should run at the current step.

        Args:
            step: Current training step
            current_loss: Current training loss
            is_checkpoint: Whether this is a checkpoint step
            is_final: Whether this is the final step

        Returns:
            bool: True if evaluation should run
        """
        # Always evaluate at final step
        if is_final:
            logger.info(f"Forcing evaluation at final step {step}")
            return True

        # Force evaluation at checkpoints if configured
        if is_checkpoint and self.force_eval_at_checkpoints:
            logger.info(f"Forcing evaluation at checkpoint step {step}")
            self.last_eval_step = step
            self.total_evals += 1
            return True

        # Update loss history
        self.loss_history.append(current_loss)

        # During warmup, use minimum interval
        if step < self.warmup_steps:
            should_eval = (step - self.last_eval_step) >= self.min_eval_steps
            if should_eval:
                self.last_eval_step = step
                self.total_evals += 1
                logger.debug(f"Warmup evaluation at step {step}")
            return should_eval

        # Check if we've reached the next scheduled evaluation
        if step < self.next_eval_step:
            self.skipped_evals += 1
            return False

        # Calculate adaptive interval based on mode
        if self.adaptive_mode == "stability":
            interval = self._calculate_stability_interval()
        elif self.adaptive_mode == "improvement":
            interval = self._calculate_improvement_interval()
        else:  # hybrid
            interval = self._calculate_hybrid_interval()

        # Update state
        self.current_eval_interval = interval
        self.next_eval_step = step + interval
        self.last_eval_step = step
        self.total_evals += 1

        logger.info(
            f"Adaptive validation at step {step}: "
            f"stability={self.stability_score:.3f}, "
            f"next_eval={self.next_eval_step}, "
            f"interval={interval}"
        )

        return True

    def _calculate_stability_interval(self) -> int:
        """
        Calculate evaluation interval based on loss stability.

        Returns:
            int: Steps until next evaluation
        """
        if len(self.loss_history) < self.stability_window:
            return self.min_eval_steps

        # Calculate stability metrics
        losses = np.array(self.loss_history)
        loss_std = np.std(losses)
        loss_mean = np.mean(losses)

        # Compute stability score (normalized variance)
        self.stability_score = loss_std / (loss_mean + 1e-8)

        # Map stability to evaluation interval
        if self.stability_score < self.stability_threshold:
            # Very stable - maximum interval
            interval = self.max_eval_steps
        elif self.stability_score < self.stability_threshold * 2:
            # Stable - scale linearly
            ratio = (self.stability_score - self.stability_threshold) / self.stability_threshold
            interval = int(self.max_eval_steps - ratio * (self.max_eval_steps - self.min_eval_steps))
        elif self.stability_score < self.stability_threshold * 5:
            # Somewhat unstable - medium interval
            interval = (self.min_eval_steps + self.max_eval_steps) // 2
        else:
            # Very unstable - minimum interval
            interval = self.min_eval_steps

        return interval

    def _calculate_improvement_interval(self) -> int:
        """
        Calculate evaluation interval based on rate of improvement.

        Returns:
            int: Steps until next evaluation
        """
        if len(self.loss_history) < 10:
            return self.min_eval_steps

        # Calculate improvement rate (negative = improving)
        recent_losses = list(self.loss_history)[-10:]
        older_losses = list(self.loss_history)[-20:-10] if len(self.loss_history) >= 20 else recent_losses

        recent_mean = np.mean(recent_losses)
        older_mean = np.mean(older_losses)

        self.improvement_rate = (recent_mean - older_mean) / (older_mean + 1e-8)

        # Map improvement to interval
        if self.improvement_rate < -0.01:
            # Improving rapidly - check frequently
            interval = self.min_eval_steps
        elif self.improvement_rate < 0:
            # Improving slowly - medium interval
            interval = (self.min_eval_steps + self.max_eval_steps) // 2
        else:
            # Not improving or degrading - check more often
            interval = self.min_eval_steps * 2

        return min(interval, self.max_eval_steps)

    def _calculate_hybrid_interval(self) -> int:
        """
        Calculate evaluation interval using both stability and improvement.

        Returns:
            int: Steps until next evaluation
        """
        stability_interval = self._calculate_stability_interval()
        improvement_interval = self._calculate_improvement_interval()

        # Weight towards stability when stable, towards improvement when unstable
        stability_weight = 1.0 - min(self.stability_score, 1.0)

        interval = int(
            stability_weight * stability_interval +
            (1 - stability_weight) * improvement_interval
        )

        return max(self.min_eval_steps, min(interval, self.max_eval_steps))

    def update_validation_loss(self, val_loss: float):
        """
        Update the scheduler with validation loss for better adaptation.

        Args:
            val_loss: Validation loss from recent evaluation
        """
        self.val_loss_history.append(val_loss)

        # Adjust intervals if validation is diverging from training
        if len(self.val_loss_history) >= 2:
            val_trend = self.val_loss_history[-1] - self.val_loss_history[-2]

            # If validation is getting worse, evaluate more frequently
            if val_trend > 0.01:
                self.current_eval_interval = max(
                    self.min_eval_steps,
                    int(self.current_eval_interval * 0.8)
                )
                logger.info(f"Validation degrading, reducing interval to {self.current_eval_interval}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            dict: Statistics about scheduler performance
        """
        total_possible_evals = self.total_evals + self.skipped_evals
        skip_rate = self.skipped_evals / max(total_possible_evals, 1)

        return {
            "total_evaluations": self.total_evals,
            "skipped_evaluations": self.skipped_evals,
            "skip_rate": skip_rate,
            "time_saved_percent": skip_rate * 100,
            "current_interval": self.current_eval_interval,
            "stability_score": self.stability_score,
            "improvement_rate": self.improvement_rate,
            "avg_loss": np.mean(self.loss_history) if self.loss_history else 0,
            "loss_std": np.std(self.loss_history) if len(self.loss_history) > 1 else 0,
        }

    def reset(self):
        """Reset the scheduler state."""
        self.loss_history.clear()
        self.val_loss_history.clear()
        self.next_eval_step = self.min_eval_steps
        self.last_eval_step = 0
        self.total_evals = 0
        self.skipped_evals = 0
        self.current_eval_interval = self.min_eval_steps
        self.stability_score = 1.0
        self.improvement_rate = 0.0


def create_adaptive_scheduler_from_config(config: Dict[str, Any]) -> Optional[AdaptiveValidationScheduler]:
    """
    Create an adaptive validation scheduler from configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        AdaptiveValidationScheduler or None if not configured
    """
    training_config = config.get("training", {})

    # Check if adaptive validation is enabled
    eval_steps = training_config.get("eval_steps", None)
    if eval_steps != "adaptive" and not isinstance(eval_steps, dict):
        return None

    # Extract adaptive validation parameters
    if isinstance(eval_steps, dict):
        params = eval_steps
    else:
        # Use defaults for "adaptive" string
        params = {}

    return AdaptiveValidationScheduler(
        min_eval_steps=params.get("min_eval_steps", 500),
        max_eval_steps=params.get("max_eval_steps", 5000),
        stability_window=params.get("stability_window", 100),
        stability_threshold=params.get("stability_threshold", 0.01),
        warmup_steps=params.get("warmup_steps", 1000),
        force_eval_at_checkpoints=params.get("force_eval_at_checkpoints", True),
        adaptive_mode=params.get("adaptive_mode", "hybrid")
    )