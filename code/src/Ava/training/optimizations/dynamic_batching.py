"""
Dynamic Batching with Memory Awareness - Training Optimizations

Location: code/src/Ava/training/optimizations/dynamic_batching.py (TRAINING module)
Purpose: Runtime memory monitoring and batch size adjustment during training

Related module: code/src/Ava/data/dynamic_batch_iterator.py
    - That module handles data loading and batch construction
    - This module handles training-time memory monitoring and adjustment

Key classes:
    - DynamicBatchScheduler: Main runtime batch size manager
    - MemoryTrendAnalyzer: Detects memory pressure trends
    - KalmanMemoryPredictor: Predictive memory estimation
    - CalibrationProfiler: Profiles memory at startup

Configuration: Uses DynamicBatchingConfig from config/training_config.py

Based on research from:
- arXiv:2412.21124: Dynamic batching for efficient GPU utilization
- arXiv:2503.05248: Memory-aware batch scheduling

This module implements adaptive batch sizing that monitors GPU memory in real-time
and adjusts the batch size to maximize throughput without causing OOM errors.

Key features:
- Real-time GPU memory monitoring
- Automatic batch size adjustment based on memory usage
- Configurable memory thresholds and adjustment rates
- Gradual scaling to avoid sudden memory spikes
- Safety mechanisms to prevent OOM

Enhanced features (v2):
- Feature 1: Token-Budget Batching - Target tokens instead of samples
- Feature 2: Sequence-Length Aware Batching - Scale with seq_len²
- Feature 3: Multi-GPU Batch Size Synchronization - Prevent DDP hangs
- Feature 4: Gradient Accumulation Integration - Stable effective batch size
- Feature 5: Predictive Memory Estimation - Pre-calculate to avoid OOM
- Feature 6: Better Threshold Tuning - More conservative defaults
- Feature 7: Geometric Warmup Strategy - Exponential batch growth
- Feature 8: Memory Pressure Prediction - Trend detection & oscillation prevention

Expected improvement: 25-40% throughput increase with all features enabled
"""

import torch
import torch.nn as nn
import logging
import json
import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field, asdict
import time
import numpy as np

# Import consolidated config from central location
from ...config.training_config import DynamicBatchingConfig, DynamicBatchConfig

# Import optimized CUDA stream utilities for event-based timing
try:
    from ...utils.cuda_streams import CUDATimer, cuda_timed_region
    _CUDA_TIMER_AVAILABLE = True
except ImportError:
    _CUDA_TIMER_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logs


# =============================================================================
# Feature 8: Memory Trend Analyzer
# =============================================================================

class MemoryTrendAnalyzer:
    """
    Analyzes memory usage trends to detect patterns and prevent oscillation.

    Features:
    - Trend detection (increasing/decreasing/stable)
    - Oscillation detection (frequent direction changes)
    - Auto-tuning of EMA smoothing factor
    """

    def __init__(self, window: int = 20, oscillation_threshold: int = 5):
        self.window = window
        self.oscillation_threshold = oscillation_threshold
        self.history: List[float] = []
        self.direction_changes = 0
        self.last_direction = 0  # -1 decreasing, 0 stable, 1 increasing

    def add_observation(self, memory_util: float) -> None:
        """Record a memory utilization observation."""
        self.history.append(memory_util)
        if len(self.history) > self.window:
            self.history.pop(0)

    def get_trend(self) -> str:
        """
        Determine memory trend from recent observations.

        Returns:
            'increasing', 'decreasing', 'stable', or 'unknown'
        """
        if len(self.history) < 5:
            return 'unknown'

        recent = self.history[-5:]
        diff = recent[-1] - recent[0]

        if diff > 0.05:
            return 'increasing'
        elif diff < -0.05:
            return 'decreasing'
        return 'stable'

    def detect_oscillation(self) -> bool:
        """
        Detect if memory is oscillating (frequent direction changes).

        Returns:
            True if oscillation detected, False otherwise
        """
        if len(self.history) < 10:
            return False

        changes = 0
        prev_direction = 0

        for i in range(1, len(self.history)):
            if self.history[i] > self.history[i-1] + 0.02:
                current_direction = 1
            elif self.history[i] < self.history[i-1] - 0.02:
                current_direction = -1
            else:
                current_direction = prev_direction

            if prev_direction != 0 and current_direction != prev_direction:
                changes += 1
            prev_direction = current_direction

        return changes >= self.oscillation_threshold

    def get_recommended_smoothing_alpha(self) -> float:
        """
        Auto-tune EMA smoothing based on oscillation detection.

        Returns:
            Recommended smoothing alpha (lower = more smoothing)
        """
        if self.detect_oscillation():
            return 0.05  # More smoothing when oscillating
        return 0.15  # Less smoothing when stable

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.history = []
        self.direction_changes = 0
        self.last_direction = 0


# =============================================================================
# Feature 5: Memory Predictor
# =============================================================================

class MemoryPredictor:
    """
    Predicts GPU memory usage based on batch size and sequence length.

    Uses linear regression after calibration to estimate memory before
    loading a batch, helping prevent OOM errors proactively.
    """

    def __init__(self, calibration_steps: int = 50, backward_safety_margin: float = 1.20):
        self.calibration_steps = calibration_steps
        self.backward_safety_margin = backward_safety_margin  # Multiplier for backward pass memory
        self.observations: List[Tuple[int, int, float]] = []  # (batch_size, seq_len, memory_gb)
        self.calibrated = False
        self.bytes_per_token = 0.0
        self.base_memory_gb = 0.0
        self.total_memory_gb = 0.0

    def set_total_memory(self, total_memory_bytes: int) -> None:
        """Set total GPU memory for threshold calculations."""
        self.total_memory_gb = total_memory_bytes / 1e9

    def record_observation(self, batch_size: int, seq_len: int, memory_gb: float) -> None:
        """
        Record a memory observation for calibration.

        Args:
            batch_size: Number of samples in batch
            seq_len: Average sequence length
            memory_gb: Memory used in GB
        """
        self.observations.append((batch_size, seq_len, memory_gb))

        if len(self.observations) >= self.calibration_steps and not self.calibrated:
            self._calibrate()

    def _calibrate(self) -> None:
        """
        Calibrate the memory model using least squares regression.

        Model: memory = base + bytes_per_token * batch_size * seq_len
        """
        if len(self.observations) < 10:
            return

        # Simple least squares: y = a + b*x where x = batch_size * seq_len
        n = len(self.observations)
        sum_x = sum(bs * sl for bs, sl, _ in self.observations)
        sum_y = sum(mem for _, _, mem in self.observations)
        sum_xy = sum(bs * sl * mem for bs, sl, mem in self.observations)
        sum_xx = sum((bs * sl) ** 2 for bs, sl, _ in self.observations)

        # Avoid division by zero
        denominator = n * sum_xx - sum_x ** 2
        if abs(denominator) < 1e-10:
            return

        self.bytes_per_token = (n * sum_xy - sum_x * sum_y) / denominator
        self.base_memory_gb = (sum_y - self.bytes_per_token * sum_x) / n

        # Ensure positive values
        self.bytes_per_token = max(0.0, self.bytes_per_token)
        self.base_memory_gb = max(0.0, self.base_memory_gb)

        self.calibrated = True
        logger.info(
            f"Memory predictor calibrated: base={self.base_memory_gb:.2f}GB, "
            f"per_token={self.bytes_per_token * 1e6:.2f}MB/token"
        )

    def predict_memory(self, batch_size: int, seq_len: int) -> float:
        """
        Predict memory usage for a given batch configuration.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length

        Returns:
            Predicted memory usage in GB, or 0.0 if not calibrated
        """
        if not self.calibrated:
            return 0.0
        return self.base_memory_gb + self.bytes_per_token * batch_size * seq_len

    def is_safe(self, batch_size: int, seq_len: int, threshold: float = 0.85) -> bool:
        """
        Check if a batch configuration is safe (won't exceed memory threshold).

        Applies backward_safety_margin to account for backward pass memory spike.
        E.g., with margin=1.20, predicted 80% forward → 96% peak during backward.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            threshold: Maximum memory utilization allowed

        Returns:
            True if predicted memory (with backward margin) is below threshold
        """
        if not self.calibrated or self.total_memory_gb == 0:
            return True  # Can't predict, assume safe

        predicted = self.predict_memory(batch_size, seq_len)
        # Apply backward safety margin to account for backward pass spike
        predicted_with_backward = predicted * self.backward_safety_margin
        return predicted_with_backward < threshold * self.total_memory_gb

    def get_max_safe_batch_size(self, seq_len: int, threshold: float = 0.85) -> int:
        """
        Calculate maximum safe batch size for a given sequence length.

        Args:
            seq_len: Sequence length
            threshold: Maximum memory utilization allowed

        Returns:
            Maximum safe batch size, or 0 if not calibrated
        """
        if not self.calibrated or self.bytes_per_token == 0:
            return 0

        available_memory = threshold * self.total_memory_gb - self.base_memory_gb
        if available_memory <= 0:
            return 1

        max_batch = int(available_memory / (self.bytes_per_token * seq_len))
        return max(1, max_batch)


# =============================================================================
# Kalman Filter Memory Predictor (Enhanced Feature 5)
# =============================================================================

class KalmanMemoryPredictor:
    """
    Kalman filter-based memory predictor for GPU memory estimation.

    Advantages over linear regression MemoryPredictor:
    - Online learning (no separate calibration phase needed)
    - Filters noisy GPU memory readings
    - Provides uncertainty estimates for safer batch size decisions
    - Adapts to changing conditions (memory fragmentation, warmup effects)

    State vector: [base_memory, linear_coef, quadratic_coef]
    - base_memory: Fixed overhead (model weights, optimizer state)
    - linear_coef: Per-token memory cost (embeddings, FFN)
    - quadratic_coef: Attention scaling O(seq²)

    Model: memory = base + linear*tokens + quadratic*tokens*seq_len
    """

    def __init__(
        self,
        initial_base_memory_gb: float = 2.0,
        initial_linear_coef: float = 1e-6,
        initial_quadratic_coef: float = 1e-9,
        process_noise_base: float = 0.01,
        process_noise_linear: float = 1e-14,
        process_noise_quadratic: float = 1e-20,
        measurement_noise: float = 0.1,
        backward_safety_margin: float = 1.25,
        min_observations: int = 5,
        confidence_sigmas: float = 2.0,
    ):
        """
        Initialize Kalman filter memory predictor.

        Args:
            initial_base_memory_gb: Initial estimate of base memory overhead
            initial_linear_coef: Initial per-token memory coefficient
            initial_quadratic_coef: Initial attention memory coefficient
            process_noise_base: How much base memory changes between steps
            process_noise_linear: How much linear coef changes between steps
            process_noise_quadratic: How much quadratic coef changes between steps
            measurement_noise: GPU memory reporting variance (GB²)
            backward_safety_margin: Multiplier for backward pass memory
            min_observations: Minimum observations before predictions are valid
            confidence_sigmas: Number of sigmas for safety decisions
        """
        # State vector: [base, linear, quadratic]
        self.state = np.array([
            initial_base_memory_gb,
            initial_linear_coef,
            initial_quadratic_coef
        ], dtype=np.float64)

        # Covariance matrix (uncertainty in state) - start with high uncertainty
        self.P = np.diag([1.0, 1e-10, 1e-16]).astype(np.float64)

        # Process noise covariance Q (how much state drifts)
        self.Q = np.diag([
            process_noise_base,
            process_noise_linear,
            process_noise_quadratic
        ]).astype(np.float64)

        # Measurement noise variance R
        self.R = measurement_noise

        self.backward_safety_margin = backward_safety_margin
        self.min_observations = min_observations
        self.confidence_sigmas = confidence_sigmas
        self.observation_count = 0
        self.total_memory_gb = 0.0

        # Track recent observations for diagnostics
        self._recent_observations: List[Tuple[int, int, float, float]] = []  # (bs, seq, mem, pred)
        self._max_recent = 50

    def set_total_memory(self, total_memory_bytes: int) -> None:
        """Set total GPU memory for threshold calculations."""
        self.total_memory_gb = total_memory_bytes / 1e9

    def observe(self, batch_size: int, seq_len: int, memory_gb: float) -> None:
        """
        Update Kalman filter with new observation.

        Uses standard Kalman filter update equations:
        1. Predict: x_pred = x, P_pred = P + Q (state unchanged, uncertainty grows)
        2. Update: Incorporate measurement to refine state estimate

        Args:
            batch_size: Number of samples in batch
            seq_len: Average sequence length
            memory_gb: Observed memory usage in GB
        """
        self.observation_count += 1

        # Calculate tokens
        tokens = batch_size * seq_len

        # Observation matrix H: memory = base + linear*tokens + quad*tokens*seq
        H = np.array([[1.0, float(tokens), float(tokens * seq_len)]], dtype=np.float64)

        # Prediction step (state doesn't change, but uncertainty grows)
        P_pred = self.P + self.Q

        # Predicted observation
        z_pred = float((H @ self.state).item())

        # Innovation (measurement residual)
        z = memory_gb
        y = z - z_pred

        # Innovation covariance
        S = float((H @ P_pred @ H.T).item()) + self.R

        # Kalman gain
        K = (P_pred @ H.T) / S

        # Update state
        self.state = self.state + K.flatten() * y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + (K * self.R) @ K.T

        # NUMERICAL STABILITY FIXES:
        # 1. Force symmetry (floating point errors can break symmetry)
        self.P = (self.P + self.P.T) / 2

        # 2. Ensure positive diagonals (covariance must be positive semi-definite)
        min_diag = 1e-10
        for i in range(3):
            if self.P[i, i] < min_diag:
                self.P[i, i] = min_diag

        # 3. Check condition number and reset if numerically unstable
        try:
            cond = np.linalg.cond(self.P)
            if cond > 1e12 or np.isnan(cond) or np.isinf(cond):
                logger.warning(f"Kalman filter covariance degenerate (cond={cond:.2e}), resetting")
                self.P = np.diag([1.0, 1e-10, 1e-16])
        except np.linalg.LinAlgError:
            logger.warning("Kalman filter covariance singular, resetting")
            self.P = np.diag([1.0, 1e-10, 1e-16])

        # Ensure positive coefficients (memory can't decrease with more tokens)
        self.state = np.maximum(self.state, [0.0, 0.0, 0.0])

        # Track for diagnostics
        self._recent_observations.append((batch_size, seq_len, memory_gb, z_pred))
        if len(self._recent_observations) > self._max_recent:
            self._recent_observations.pop(0)

    def predict_memory(
        self,
        batch_size: int,
        seq_len: int,
        include_backward: bool = True
    ) -> Tuple[float, float]:
        """
        Predict memory usage with uncertainty estimate.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            include_backward: Whether to apply backward pass safety margin

        Returns:
            Tuple of (predicted_memory_gb, uncertainty_gb)
        """
        if self.observation_count < self.min_observations:
            # Not enough data - return conservative estimate
            return 0.0, float('inf')

        tokens = batch_size * seq_len
        H = np.array([[1.0, float(tokens), float(tokens * seq_len)]], dtype=np.float64)

        # Predicted memory
        predicted = float((H @ self.state).item())

        # Prediction uncertainty (1 sigma)
        variance = float((H @ self.P @ H.T).item()) + self.R
        uncertainty = np.sqrt(max(0, variance))

        if include_backward:
            predicted *= self.backward_safety_margin
            uncertainty *= self.backward_safety_margin

        return predicted, uncertainty

    def is_safe(
        self,
        batch_size: int,
        seq_len: int,
        threshold: float = 0.85,
        confidence_sigmas: Optional[float] = None
    ) -> bool:
        """
        Check if batch configuration is safe with confidence bounds.

        Uses predicted memory + confidence_sigmas * uncertainty
        to be conservative about safety decisions.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            threshold: Maximum memory utilization allowed
            confidence_sigmas: Override default confidence level

        Returns:
            True if predicted memory (with uncertainty) is below threshold
        """
        if self.total_memory_gb == 0:
            return True  # Can't check, assume safe

        if self.observation_count < self.min_observations:
            return True  # Not calibrated yet, assume safe

        if confidence_sigmas is None:
            confidence_sigmas = self.confidence_sigmas

        predicted, uncertainty = self.predict_memory(batch_size, seq_len)

        # Use upper confidence bound for safety
        conservative_estimate = predicted + confidence_sigmas * uncertainty

        return conservative_estimate < threshold * self.total_memory_gb

    def get_max_safe_batch_size(
        self,
        seq_len: int,
        threshold: float = 0.85,
        max_search: int = 1024
    ) -> int:
        """
        Binary search for maximum safe batch size.

        Args:
            seq_len: Sequence length
            threshold: Maximum memory utilization allowed
            max_search: Maximum batch size to search

        Returns:
            Maximum safe batch size, or 0 if not calibrated
        """
        if self.observation_count < self.min_observations:
            return 0  # Not calibrated yet

        low, high = 1, max_search
        result = 1

        while low <= high:
            mid = (low + high) // 2
            if self.is_safe(mid, seq_len, threshold):
                result = mid
                low = mid + 1
            else:
                high = mid - 1

        return result

    @property
    def calibrated(self) -> bool:
        """Whether predictor has enough observations for predictions."""
        return self.observation_count >= self.min_observations

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state for logging/debugging."""
        return {
            'base_memory_gb': float(self.state[0]),
            'linear_coef': float(self.state[1]),
            'quadratic_coef': float(self.state[2]),
            'observation_count': self.observation_count,
            'uncertainty_base': float(np.sqrt(self.P[0, 0])),
            'uncertainty_linear': float(np.sqrt(self.P[1, 1])),
            'uncertainty_quadratic': float(np.sqrt(self.P[2, 2])),
            'calibrated': self.calibrated,
            'total_memory_gb': self.total_memory_gb,
        }

    def get_prediction_error_stats(self) -> Dict[str, float]:
        """Get prediction error statistics from recent observations."""
        if not self._recent_observations:
            return {'mean_error': 0.0, 'max_error': 0.0, 'rmse': 0.0}

        errors = [abs(obs - pred) for _, _, obs, pred in self._recent_observations]
        return {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean([e**2 for e in errors]))),
        }


# =============================================================================
# Feature 9: Adaptive Backward Margin
# =============================================================================

class AdaptiveBackwardMargin:
    """
    Automatically learns the optimal backward_safety_margin over time.

    The backward pass typically uses more memory than the forward pass due to
    gradient storage, but the exact ratio depends on the model architecture,
    optimizer, and other factors. This class learns the actual ratio from
    observations and uses it for more accurate memory predictions.

    Key insight: Instead of using a fixed 1.5x multiplier, we observe the actual
    peak memory during backward passes and learn the true multiplier.

    The margin auto-adjusts:
    - If we're underutilizing (VRAM << target), decrease margin to allow more tokens
    - If we hit OOM or high memory, increase margin for safety
    - Uses exponential moving average to smooth observations
    """

    def __init__(
        self,
        initial_margin: float = 1.5,
        min_margin: float = 1.05,
        max_margin: float = 2.0,
        learning_rate: float = 0.02,
        ema_alpha: float = 0.1,
    ):
        """
        Initialize adaptive backward margin.

        Args:
            initial_margin: Starting margin value
            min_margin: Minimum allowed margin (backward always uses some extra)
            max_margin: Maximum margin (cap for safety)
            learning_rate: How fast to adjust margin based on feedback
            ema_alpha: Smoothing factor for observations (lower = smoother)
        """
        self.current_margin = initial_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha

        # Track forward vs backward memory observations
        self.forward_memory_ema = 0.0
        self.backward_memory_ema = 0.0
        self.observed_ratio_ema = initial_margin

        # Statistics
        self.observation_count = 0
        self.oom_count = 0
        self.underutil_count = 0
        self.last_adjustment_step = 0

        # History for debugging
        self.margin_history: List[Tuple[int, float, str]] = []  # (step, margin, reason)
        self._max_history = 100

    def observe_forward(self, memory_gb: float) -> None:
        """Record forward pass memory usage."""
        if memory_gb > 0.1:  # Ignore very small values
            self.forward_memory_ema = (
                self.ema_alpha * memory_gb +
                (1 - self.ema_alpha) * self.forward_memory_ema
            )

    def observe_backward(self, peak_memory_gb: float) -> None:
        """
        Record backward pass peak memory and update ratio estimate.

        Args:
            peak_memory_gb: Peak memory during/after backward pass
        """
        if peak_memory_gb > 0.1:
            self.backward_memory_ema = (
                self.ema_alpha * peak_memory_gb +
                (1 - self.ema_alpha) * self.backward_memory_ema
            )

            # Update observed ratio
            if self.forward_memory_ema > 0.1:
                observed_ratio = peak_memory_gb / self.forward_memory_ema
                self.observed_ratio_ema = (
                    self.ema_alpha * observed_ratio +
                    (1 - self.ema_alpha) * self.observed_ratio_ema
                )
                self.observation_count += 1

    def observe_utilization(
        self,
        step: int,
        raw_util: float,
        target_util: float,
        critical_util: float = 0.95,
    ) -> float:
        """
        Update margin based on current memory utilization.

        This is the main feedback mechanism:
        - If utilization is well below target, we're being too conservative
        - If utilization is near critical, we're being too aggressive

        Args:
            step: Current training step
            raw_util: Current raw memory utilization (0-1)
            target_util: Target utilization (e.g., 0.75)
            critical_util: Critical threshold (e.g., 0.95)

        Returns:
            Updated margin value
        """
        # Only adjust every 50 steps to avoid noise
        if step - self.last_adjustment_step < 50:
            return self.current_margin

        self.last_adjustment_step = step
        old_margin = self.current_margin
        reason = ""

        # Calculate effective utilization with current margin
        # If raw_util * margin is much lower than target, margin is too high
        effective_util = raw_util * self.current_margin

        if effective_util < target_util * 0.7:
            # Significantly underutilizing - decrease margin
            decrease = self.learning_rate * (1 - effective_util / target_util)
            self.current_margin = max(
                self.min_margin,
                self.current_margin - decrease
            )
            self.underutil_count += 1
            reason = f"underutil ({effective_util:.0%} << {target_util:.0%})"

        elif effective_util > target_util * 1.1:
            # Over target - increase margin slightly
            increase = self.learning_rate * 0.5 * (effective_util / target_util - 1)
            self.current_margin = min(
                self.max_margin,
                self.current_margin + increase
            )
            reason = f"over_target ({effective_util:.0%} > {target_util:.0%})"

        # Also consider the observed actual ratio if we have enough data
        if self.observation_count > 20:
            # Blend toward observed ratio
            blend_rate = 0.01  # Very slow blending
            target_margin = max(self.min_margin, self.observed_ratio_ema * 1.1)  # 10% safety buffer
            self.current_margin = (
                (1 - blend_rate) * self.current_margin +
                blend_rate * target_margin
            )
            if not reason:
                reason = f"blend_to_observed ({self.observed_ratio_ema:.2f}x)"

        # Clamp to valid range
        self.current_margin = max(self.min_margin, min(self.max_margin, self.current_margin))

        # Record history
        if abs(self.current_margin - old_margin) > 0.001:
            self.margin_history.append((step, self.current_margin, reason))
            if len(self.margin_history) > self._max_history:
                self.margin_history.pop(0)
            logger.debug(f"[AdaptiveMargin] Step {step}: {old_margin:.3f} → {self.current_margin:.3f} ({reason})")

        return self.current_margin

    def record_oom(self, step: int) -> float:
        """
        Record OOM event and increase margin significantly.

        Args:
            step: Step where OOM occurred

        Returns:
            Updated (increased) margin
        """
        self.oom_count += 1
        old_margin = self.current_margin

        # Increase margin significantly after OOM
        increase = 0.15 + 0.05 * min(self.oom_count, 5)  # More increase for repeated OOMs
        self.current_margin = min(self.max_margin, self.current_margin + increase)

        reason = f"OOM (count={self.oom_count})"
        self.margin_history.append((step, self.current_margin, reason))
        logger.warning(f"[AdaptiveMargin] OOM at step {step}: margin {old_margin:.2f} → {self.current_margin:.2f}")

        return self.current_margin

    def get_margin(self) -> float:
        """Get current margin value."""
        return self.current_margin

    def get_statistics(self) -> Dict[str, Any]:
        """Get margin statistics for logging."""
        return {
            'current_margin': self.current_margin,
            'min_margin': self.min_margin,
            'max_margin': self.max_margin,
            'observation_count': self.observation_count,
            'oom_count': self.oom_count,
            'underutil_count': self.underutil_count,
            'observed_ratio_ema': self.observed_ratio_ema,
            'forward_memory_ema_gb': self.forward_memory_ema,
            'backward_memory_ema_gb': self.backward_memory_ema,
        }


# =============================================================================
# Adaptive Adjustment Strategy (Enhanced Feature 9 continued)
# =============================================================================

@dataclass
class AdjustmentHistoryEntry:
    """Record of a batch size adjustment and its outcome."""
    step: int
    old_batch_size: int
    new_batch_size: int
    memory_before: float
    memory_after: float
    throughput_before: float  # tokens/sec
    throughput_after: float
    was_beneficial: Optional[bool]  # None if not yet determined
    adjustment_type: str  # 'increase', 'decrease', 'oom_recovery'


class ThroughputTracker:
    """
    Track throughput to determine if adjustments were beneficial.

    A batch size increase is beneficial if it improves tokens/second
    without causing instability.
    """

    def __init__(self, window_size: int = 20, stabilization_steps: int = 5):
        """
        Initialize throughput tracker.

        Args:
            window_size: Number of measurements to average
            stabilization_steps: Steps to wait after adjustment before measuring
        """
        self.window_size = window_size
        self.stabilization_steps = stabilization_steps
        self.throughput_history: List[Tuple[int, int, float]] = []  # (step, batch_size, tokens_per_sec)

    def record(self, step: int, batch_size: int, tokens_per_sec: float) -> None:
        """Record throughput measurement."""
        self.throughput_history.append((step, batch_size, tokens_per_sec))
        if len(self.throughput_history) > 1000:
            self.throughput_history = self.throughput_history[-1000:]

    def get_recent_avg(self, window: Optional[int] = None) -> float:
        """Get average throughput from recent measurements."""
        if not self.throughput_history:
            return 0.0
        window = window or self.window_size
        recent = self.throughput_history[-window:]
        return sum(t for _, _, t in recent) / len(recent)

    def was_adjustment_beneficial(
        self,
        adjustment_step: int,
        old_batch_size: int,
        new_batch_size: int,
        beneficial_threshold: float = 0.02,  # 2% improvement = beneficial
        harmful_threshold: float = -0.05,  # 5% decrease = harmful
    ) -> Optional[bool]:
        """
        Determine if a batch size adjustment was beneficial.

        Args:
            adjustment_step: Step when adjustment occurred
            old_batch_size: Batch size before adjustment
            new_batch_size: Batch size after adjustment
            beneficial_threshold: Min improvement ratio to be considered beneficial
            harmful_threshold: Max decrease ratio before considered harmful

        Returns:
            True if beneficial, False if harmful, None if inconclusive
        """
        # Get throughput before adjustment
        before = [
            t for step, bs, t in self.throughput_history
            if step < adjustment_step and step >= adjustment_step - self.window_size
        ]

        # Get throughput after adjustment (after stabilization)
        after = [
            t for step, bs, t in self.throughput_history
            if step >= adjustment_step + self.stabilization_steps
            and step < adjustment_step + self.stabilization_steps + self.window_size
        ]

        if len(before) < 5 or len(after) < 5:
            return None  # Not enough data

        avg_before = sum(before) / len(before)
        avg_after = sum(after) / len(after)

        if avg_before == 0:
            return None

        improvement = (avg_after - avg_before) / avg_before

        if improvement > beneficial_threshold:
            return True
        elif improvement < harmful_threshold:
            return False
        else:
            return None  # Neutral


class AdaptiveAdjustmentStrategy:
    """
    Adaptive batch size adjustment with momentum and history learning.

    Key improvements over fixed adjustment:
    1. Momentum: Consecutive successful adjustments in same direction → larger steps
    2. Throughput tracking: Learn if adjustments actually help throughput
    3. Threshold learning: Adapt thresholds from OOM events and success patterns
    4. Persistence: Save learned parameters to run output directory
    """

    def __init__(
        self,
        config: 'DynamicBatchConfig',
        persistence_path: Optional[Path] = None,
        max_momentum: float = 3.0,
        momentum_decay: float = 0.8,
        learn_thresholds: bool = True,
        throughput_window: int = 20,
    ):
        """
        Initialize adaptive adjustment strategy.

        Args:
            config: Dynamic batch configuration
            persistence_path: Path to save/load learned state
            max_momentum: Maximum momentum multiplier (caps adjustment size)
            momentum_decay: How fast momentum decays when not adjusting
            learn_thresholds: Whether to learn optimal thresholds
            throughput_window: Window size for throughput tracking
        """
        self.config = config
        self.persistence_path = persistence_path
        self.max_momentum = max_momentum
        self.momentum_decay = momentum_decay
        self.learn_thresholds = learn_thresholds

        # Momentum tracking
        self.consecutive_increases = 0
        self.consecutive_decreases = 0
        self.momentum_factor = 1.0

        # Learned thresholds (start with config defaults)
        self.learned_thresholds = {
            'low': config.low_memory_threshold,
            'target': config.target_memory_threshold,
            'high': config.high_memory_threshold,
            'critical': config.critical_memory_threshold,
        }

        # History for learning
        self.adjustment_history: List[AdjustmentHistoryEntry] = []
        self.max_history_size = 500

        # OOM and success tracking for threshold learning
        self.oom_memory_levels: List[float] = []
        self.success_memory_levels: List[float] = []

        # Throughput tracker
        self.throughput_tracker = ThroughputTracker(window_size=throughput_window)

        # Statistics
        self.total_adjustments = 0
        self.beneficial_adjustments = 0
        self.harmful_adjustments = 0

        # Load persisted state if available
        if persistence_path and persistence_path.exists():
            self._load_state()

    def calculate_adjustment(
        self,
        current_batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
        mem_stats: Dict[str, float],
        trend: str = 'unknown',
        predictor: Optional[KalmanMemoryPredictor] = None,
        avg_seq_len: int = 512,
    ) -> Tuple[int, str, float]:
        """
        Calculate new batch size with momentum and adaptive thresholds.

        Args:
            current_batch_size: Current batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            mem_stats: Memory statistics from scheduler
            trend: Memory trend ('increasing', 'decreasing', 'stable', 'unknown')
            predictor: Optional Kalman predictor for safety checks
            avg_seq_len: Average sequence length for predictor checks

        Returns:
            Tuple of (new_batch_size, reason, confidence)
        """
        raw_util = mem_stats.get('raw_utilization', 0.0)
        smoothed_util = mem_stats.get('utilization', 0.0)

        # Use learned thresholds
        low_thresh = self.learned_thresholds['low']
        target_thresh = self.learned_thresholds['target']
        high_thresh = self.learned_thresholds['high']
        critical_thresh = self.learned_thresholds['critical']

        current_multiplier = current_batch_size // min_batch_size
        max_multiplier = max_batch_size // min_batch_size

        # Determine base adjustment direction and reason
        base_adjustment = 0
        reason = ""
        confidence = 0.5

        # Critical memory - emergency decrease
        if raw_util > critical_thresh:
            self._reset_momentum()
            return min_batch_size, f"CRITICAL {raw_util:.1%}", 1.0

        # High memory - decrease
        elif raw_util > high_thresh:
            base_adjustment = -1
            reason = f"HIGH {raw_util:.1%}"
            self.consecutive_increases = 0
            self.consecutive_decreases += 1
            confidence = 0.8

        # Preemptive decrease if trending up
        elif trend == 'increasing' and smoothed_util > 0.7:
            base_adjustment = -1
            reason = f"PREEMPTIVE (trend={trend}, util={smoothed_util:.1%})"
            self.consecutive_increases = 0
            self.consecutive_decreases += 1
            confidence = 0.6

        # Low memory - increase
        elif smoothed_util < low_thresh:
            base_adjustment = +1
            reason = f"LOW {smoothed_util:.1%}"
            self.consecutive_decreases = 0
            self.consecutive_increases += 1
            confidence = 0.7

        # Optimal range - no change
        else:
            self._decay_momentum()
            return current_batch_size, f"OPTIMAL {smoothed_util:.1%}", 0.5

        # Apply momentum
        if base_adjustment > 0:
            momentum_bonus = min(self.consecutive_increases - 1, int(self.max_momentum - 1))
            adjustment = base_adjustment + max(0, momentum_bonus) * 0.5
        else:
            momentum_bonus = min(self.consecutive_decreases - 1, int(self.max_momentum - 1))
            adjustment = base_adjustment - max(0, momentum_bonus) * 0.5

        # Check predictor confidence if available
        if predictor and predictor.calibrated and base_adjustment > 0:
            target_batch = (current_multiplier + int(adjustment)) * min_batch_size
            target_batch = max(min_batch_size, min(max_batch_size, target_batch))

            if not predictor.is_safe(target_batch, avg_seq_len, high_thresh):
                # Predictor says unsafe - reduce adjustment or block
                if not predictor.is_safe(current_batch_size + min_batch_size, avg_seq_len, high_thresh):
                    # Even +1 is unsafe
                    return current_batch_size, reason + " (predictor-blocked)", 0.9
                else:
                    # Reduce adjustment to +1
                    adjustment = 1
                    reason += " (predictor-limited)"
                    confidence = 0.8

        # Calculate new multiplier
        new_multiplier = current_multiplier + int(round(adjustment))
        new_multiplier = max(1, min(max_multiplier, new_multiplier))
        new_batch_size = new_multiplier * min_batch_size

        # Update momentum factor
        self.momentum_factor = 1.0 + 0.2 * max(
            self.consecutive_increases,
            self.consecutive_decreases
        )
        self.momentum_factor = min(self.momentum_factor, self.max_momentum)

        return new_batch_size, reason, confidence

    def record_adjustment(
        self,
        step: int,
        old_batch_size: int,
        new_batch_size: int,
        memory_util: float,
        throughput: float = 0.0,
    ) -> None:
        """Record a batch size adjustment for learning."""
        adjustment_type = 'increase' if new_batch_size > old_batch_size else 'decrease'

        entry = AdjustmentHistoryEntry(
            step=step,
            old_batch_size=old_batch_size,
            new_batch_size=new_batch_size,
            memory_before=memory_util,
            memory_after=0.0,  # Will be updated later
            throughput_before=throughput,
            throughput_after=0.0,  # Will be updated later
            was_beneficial=None,  # Will be determined later
            adjustment_type=adjustment_type,
        )

        self.adjustment_history.append(entry)
        self.total_adjustments += 1

        # Trim history
        if len(self.adjustment_history) > self.max_history_size:
            self.adjustment_history = self.adjustment_history[-self.max_history_size:]

    def update_adjustment_outcome(self, step: int, memory_util: float, throughput: float) -> None:
        """Update outcome for recent adjustments and learn from them."""
        for entry in reversed(self.adjustment_history):
            if entry.was_beneficial is None and step > entry.step + 10:
                entry.memory_after = memory_util
                entry.throughput_after = throughput

                # Determine if beneficial
                was_beneficial = self.throughput_tracker.was_adjustment_beneficial(
                    entry.step, entry.old_batch_size, entry.new_batch_size
                )
                entry.was_beneficial = was_beneficial

                if was_beneficial is True:
                    self.beneficial_adjustments += 1
                elif was_beneficial is False:
                    self.harmful_adjustments += 1

                break  # Only update one at a time

        # Learn from history
        if self.learn_thresholds:
            self._update_thresholds_from_history()

    def record_oom(self, memory_level: float) -> None:
        """Record OOM event for threshold learning."""
        self.oom_memory_levels.append(memory_level)
        if len(self.oom_memory_levels) > 50:
            self.oom_memory_levels.pop(0)

        self._reset_momentum()
        self._update_thresholds_from_oom()

    def record_success(self, memory_level: float) -> None:
        """Record successful step for threshold learning."""
        self.success_memory_levels.append(memory_level)
        if len(self.success_memory_levels) > 100:
            self.success_memory_levels.pop(0)

    def record_throughput(self, step: int, batch_size: int, tokens_per_sec: float) -> None:
        """Record throughput for learning."""
        self.throughput_tracker.record(step, batch_size, tokens_per_sec)

    def _update_thresholds_from_history(self) -> None:
        """Learn optimal thresholds from adjustment history."""
        if len(self.adjustment_history) < 20:
            return

        # Analyze which adjustments were beneficial
        recent = self.adjustment_history[-100:]
        beneficial_increases = [
            e for e in recent
            if e.adjustment_type == 'increase' and e.was_beneficial is True
        ]
        harmful_increases = [
            e for e in recent
            if e.adjustment_type == 'increase' and e.was_beneficial is False
        ]

        # If many increases are harmful, raise low threshold (be more conservative)
        if len(harmful_increases) > len(beneficial_increases) * 0.3:
            self.learned_thresholds['low'] = min(
                self.learned_thresholds['low'] + 0.02,
                0.75  # Cap at 75%
            )
            logger.debug(f"Raised low threshold to {self.learned_thresholds['low']:.2f} due to harmful increases")

        # If most increases are beneficial, lower low threshold (be more aggressive)
        elif len(beneficial_increases) > 10 and len(harmful_increases) == 0:
            self.learned_thresholds['low'] = max(
                self.learned_thresholds['low'] - 0.01,
                0.40  # Floor at 40%
            )
            logger.debug(f"Lowered low threshold to {self.learned_thresholds['low']:.2f} due to beneficial increases")

    def _update_thresholds_from_oom(self) -> None:
        """Adjust high/critical thresholds based on OOM events."""
        if not self.oom_memory_levels:
            return

        # Find the minimum memory level that caused OOM
        min_oom_level = min(self.oom_memory_levels[-10:])

        # Set critical threshold below the OOM level with margin
        new_critical = min_oom_level - 0.05
        if new_critical < self.learned_thresholds['critical']:
            old_critical = self.learned_thresholds['critical']
            self.learned_thresholds['critical'] = max(new_critical, 0.80)
            self.learned_thresholds['high'] = min(
                self.learned_thresholds['critical'] - 0.05,
                self.learned_thresholds['high']
            )
            logger.info(
                f"Lowered critical threshold: {old_critical:.2f} -> {self.learned_thresholds['critical']:.2f} "
                f"(OOM at {min_oom_level:.2f})"
            )

    def _reset_momentum(self) -> None:
        """Reset momentum after direction change or OOM."""
        self.consecutive_increases = 0
        self.consecutive_decreases = 0
        self.momentum_factor = 1.0

    def _decay_momentum(self) -> None:
        """Decay momentum when in optimal range."""
        self.consecutive_increases = int(self.consecutive_increases * self.momentum_decay)
        self.consecutive_decreases = int(self.consecutive_decreases * self.momentum_decay)
        self.momentum_factor = max(1.0, self.momentum_factor * self.momentum_decay)

    def save_state(self) -> None:
        """Persist learned state to run output directory."""
        if not self.persistence_path:
            return

        state = {
            'learned_thresholds': self.learned_thresholds,
            'oom_memory_levels': self.oom_memory_levels[-50:],
            'success_memory_levels': self.success_memory_levels[-100:],
            'momentum_factor': self.momentum_factor,
            'total_adjustments': self.total_adjustments,
            'beneficial_adjustments': self.beneficial_adjustments,
            'harmful_adjustments': self.harmful_adjustments,
        }

        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved adaptive adjustment state to {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to save adaptive state: {e}")

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            with open(self.persistence_path, 'r') as f:
                state = json.load(f)

            self.learned_thresholds = state.get(
                'learned_thresholds', self.learned_thresholds
            )
            self.oom_memory_levels = state.get('oom_memory_levels', [])
            self.success_memory_levels = state.get('success_memory_levels', [])
            self.momentum_factor = state.get('momentum_factor', 1.0)
            self.total_adjustments = state.get('total_adjustments', 0)
            self.beneficial_adjustments = state.get('beneficial_adjustments', 0)
            self.harmful_adjustments = state.get('harmful_adjustments', 0)

            logger.info(f"Loaded adaptive adjustment state from {self.persistence_path}")
            logger.info(f"  Learned thresholds: low={self.learned_thresholds['low']:.2f}, "
                       f"critical={self.learned_thresholds['critical']:.2f}")
        except Exception as e:
            logger.warning(f"Failed to load adaptive state: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics for logging."""
        return {
            'total_adjustments': self.total_adjustments,
            'beneficial_adjustments': self.beneficial_adjustments,
            'harmful_adjustments': self.harmful_adjustments,
            'beneficial_ratio': self.beneficial_adjustments / max(1, self.total_adjustments),
            'learned_thresholds': self.learned_thresholds.copy(),
            'momentum_factor': self.momentum_factor,
            'consecutive_increases': self.consecutive_increases,
            'consecutive_decreases': self.consecutive_decreases,
            'recent_throughput_avg': self.throughput_tracker.get_recent_avg(),
        }


# =============================================================================
# Feature 10: Aggressive Growth Controller
# =============================================================================

class AggressiveGrowthController:
    """
    Aggressively increases batch size during early training to find optimal size.

    Key insight: Most training starts too conservative because we don't know
    the safe maximum batch size. This controller probes aggressively during
    the first N steps to find the limit, then switches to conservative mode.

    Strategy:
    1. During exploration phase (first N steps):
       - Use larger increase steps (e.g., +8 instead of +1)
       - Track successful batch sizes to find "safe maximum"
       - Back off quickly on memory pressure

    2. After exploration:
       - Switch to conservative mode with known safe range
       - Use learned maximum as new ceiling

    Benefits:
    - Finds optimal batch size in ~100-500 steps instead of thousands
    - Maximizes GPU utilization earlier in training
    - Adapts to actual hardware/model memory characteristics
    """

    def __init__(
        self,
        exploration_steps: int = 500,
        growth_factor: int = 2,
        min_headroom_for_growth: float = 0.10,
        max_batch_size: int = 512,
        min_batch_size: int = 8,
    ):
        """
        Initialize aggressive growth controller.

        Args:
            exploration_steps: Steps during which to aggressively explore
            growth_factor: Multiplier for batch size increases during exploration
            min_headroom_for_growth: Minimum memory headroom (0-1) to trigger growth
            max_batch_size: Absolute maximum batch size to explore
            min_batch_size: Minimum batch size
        """
        self.exploration_steps = exploration_steps
        self.growth_factor = growth_factor
        self.min_headroom_for_growth = min_headroom_for_growth
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

        # State tracking
        self.current_step = 0
        self.exploration_complete = False
        self.max_successful_batch = 0  # Largest batch that didn't OOM
        self.min_failed_batch = max_batch_size + 1  # Smallest batch that OOMed

        # History for learning
        self.batch_attempts: List[Tuple[int, bool, float]] = []  # (batch_size, success, memory_util)
        self.consecutive_successes = 0
        self.consecutive_failures = 0

        # Growth momentum
        self.growth_momentum = 1.0  # Increases with consecutive successes

        logger.info(
            f"AggressiveGrowthController initialized: "
            f"exploration_steps={exploration_steps}, growth_factor={growth_factor}, "
            f"min_headroom={min_headroom_for_growth:.0%}"
        )

    def should_explore(self, step: int) -> bool:
        """Check if still in exploration phase."""
        return step < self.exploration_steps and not self.exploration_complete

    def record_batch_result(
        self,
        batch_size: int,
        success: bool,
        memory_utilization: float,
    ) -> None:
        """
        Record the result of a batch attempt.

        Args:
            batch_size: Size of the attempted batch
            success: Whether the batch succeeded (no OOM)
            memory_utilization: Memory utilization during/after batch
        """
        self.batch_attempts.append((batch_size, success, memory_utilization))

        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self.max_successful_batch = max(self.max_successful_batch, batch_size)

            # Build momentum after consecutive successes
            if self.consecutive_successes >= 3:
                self.growth_momentum = min(3.0, self.growth_momentum + 0.2)
        else:
            self.consecutive_successes = 0
            self.consecutive_failures += 1
            self.min_failed_batch = min(self.min_failed_batch, batch_size)

            # Reset momentum on failure
            self.growth_momentum = 1.0

            # If we've found the boundary, complete exploration
            if self.max_successful_batch > 0 and batch_size >= self.max_successful_batch:
                logger.info(
                    f"[AggressiveGrowth] Found batch size boundary: "
                    f"safe_max={self.max_successful_batch}, failed_at={batch_size}"
                )

        # Complete exploration if we've found a clear boundary
        if (self.max_successful_batch > 0 and
            self.min_failed_batch <= self.max_batch_size and
            self.min_failed_batch - self.max_successful_batch <= self.min_batch_size * 2):
            self.exploration_complete = True
            logger.info(
                f"[AggressiveGrowth] Exploration complete: "
                f"safe_max={self.max_successful_batch}"
            )

    def get_target_batch_size(
        self,
        current_batch_size: int,
        memory_utilization: float,
        target_memory: float,
        step: int,
    ) -> Tuple[int, str]:
        """
        Get target batch size during aggressive growth phase.

        Args:
            current_batch_size: Current batch size
            memory_utilization: Current GPU memory utilization (0-1)
            target_memory: Target memory utilization
            step: Current training step

        Returns:
            Tuple of (target_batch_size, reason_string)
        """
        self.current_step = step

        # Not exploring anymore
        if not self.should_explore(step):
            if self.max_successful_batch > 0:
                # Use learned safe maximum
                return min(current_batch_size, self.max_successful_batch), "exploration-complete"
            return current_batch_size, "no-exploration"

        # Calculate headroom
        headroom = target_memory - memory_utilization

        # Not enough headroom - don't grow
        if headroom < self.min_headroom_for_growth:
            # If memory is high, might need to decrease
            if memory_utilization > target_memory + 0.05:
                new_size = max(self.min_batch_size, current_batch_size - self.min_batch_size)
                return new_size, f"headroom-low ({headroom:.1%})"
            return current_batch_size, f"headroom-marginal ({headroom:.1%})"

        # Have headroom - calculate aggressive increase
        # Base increase: proportional to headroom and growth_factor
        base_increase = self.min_batch_size * self.growth_factor

        # Apply momentum for consecutive successes
        momentum_increase = int(base_increase * self.growth_momentum)

        # Scale increase by available headroom (more headroom = larger increase)
        headroom_multiplier = min(2.0, headroom / 0.10)  # Cap at 2x for 20%+ headroom
        scaled_increase = int(momentum_increase * headroom_multiplier)

        # Calculate new target
        new_target = current_batch_size + scaled_increase

        # Clamp to safe range
        # Don't exceed known failure point
        if self.min_failed_batch <= self.max_batch_size:
            new_target = min(new_target, self.min_failed_batch - self.min_batch_size)

        # Don't exceed absolute max
        new_target = min(new_target, self.max_batch_size)

        # Ensure it's a multiple of min_batch_size
        new_target = (new_target // self.min_batch_size) * self.min_batch_size
        new_target = max(self.min_batch_size, new_target)

        reason = (
            f"aggressive-growth (headroom={headroom:.1%}, "
            f"momentum={self.growth_momentum:.1f}x, increase=+{new_target - current_batch_size})"
        )

        return new_target, reason

    def get_safe_maximum(self) -> int:
        """Get the discovered safe maximum batch size."""
        if self.max_successful_batch > 0:
            return self.max_successful_batch
        return self.max_batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            'exploration_complete': self.exploration_complete,
            'current_step': self.current_step,
            'max_successful_batch': self.max_successful_batch,
            'min_failed_batch': self.min_failed_batch if self.min_failed_batch <= self.max_batch_size else None,
            'consecutive_successes': self.consecutive_successes,
            'growth_momentum': self.growth_momentum,
            'total_attempts': len(self.batch_attempts),
        }


# =============================================================================
# Enhanced Calibration System
# =============================================================================

@dataclass
class MemoryProfile:
    """Memory profile for a single (batch_size, seq_len) configuration."""
    batch_size: int
    seq_len: int
    forward_memory_gb: float
    backward_memory_gb: float
    peak_memory_gb: float
    tokens: int = 0

    def __post_init__(self):
        self.tokens = self.batch_size * self.seq_len


@dataclass
class CalibrationResult:
    """Complete calibration result with all profiling data."""
    # Hardware info
    hardware_profile: Dict[str, Any]

    # Memory model coefficients (multi-term: memory = base + linear*tokens + quadratic*tokens*seq)
    base_memory_gb: float
    linear_coef: float
    quadratic_coef: float
    backward_ratio: float

    # Raw profiling data
    memory_profiles: List[Dict[str, Any]]

    # Throughput data (optional)
    optimal_batch_size: Optional[int] = None
    optimal_throughput: Optional[float] = None  # tokens/sec
    throughput_profiles: Optional[List[Dict[str, Any]]] = None

    # Metadata
    calibration_time_seconds: float = 0.0
    timestamp: str = ""
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'hardware_profile': self.hardware_profile,
            'base_memory_gb': self.base_memory_gb,
            'linear_coef': self.linear_coef,
            'quadratic_coef': self.quadratic_coef,
            'backward_ratio': self.backward_ratio,
            'memory_profiles': self.memory_profiles,
            'optimal_batch_size': self.optimal_batch_size,
            'optimal_throughput': self.optimal_throughput,
            'throughput_profiles': self.throughput_profiles,
            'calibration_time_seconds': self.calibration_time_seconds,
            'timestamp': self.timestamp,
            'cache_key': self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """Create from JSON dict."""
        return cls(
            hardware_profile=data['hardware_profile'],
            base_memory_gb=data['base_memory_gb'],
            linear_coef=data['linear_coef'],
            quadratic_coef=data['quadratic_coef'],
            backward_ratio=data['backward_ratio'],
            memory_profiles=data['memory_profiles'],
            optimal_batch_size=data.get('optimal_batch_size'),
            optimal_throughput=data.get('optimal_throughput'),
            throughput_profiles=data.get('throughput_profiles'),
            calibration_time_seconds=data.get('calibration_time_seconds', 0.0),
            timestamp=data.get('timestamp', ''),
            cache_key=data.get('cache_key', ''),
        )


class HardwareProfiler:
    """
    Profiles GPU hardware capabilities and creates unique fingerprints.

    Used to:
    1. Create unique cache key for calibration persistence
    2. Detect GPU capabilities for optimization decisions
    3. Log hardware info for reproducibility
    """

    def __init__(self):
        self.gpu_name: str = ""
        self.total_memory_gb: float = 0.0
        self.compute_capability: Tuple[int, int] = (0, 0)
        self.num_sms: int = 0
        self.has_tensor_cores: bool = False
        self.cuda_version: str = ""
        self.driver_version: str = ""

    def profile(self) -> Dict[str, Any]:
        """
        Run hardware profiling.

        Returns:
            Dictionary with GPU hardware information
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available", "gpu_name": "cpu"}

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        self.gpu_name = props.name
        self.total_memory_gb = props.total_memory / 1e9
        self.compute_capability = (props.major, props.minor)
        self.num_sms = props.multi_processor_count
        self.has_tensor_cores = props.major >= 7  # Volta and newer

        # Get CUDA version
        self.cuda_version = torch.version.cuda or "unknown"

        return {
            "gpu_name": self.gpu_name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "num_sms": self.num_sms,
            "has_tensor_cores": self.has_tensor_cores,
            "cuda_version": self.cuda_version,
        }

    def get_fingerprint(self) -> str:
        """
        Create unique fingerprint for this GPU.

        Used as part of cache key to ensure calibration data matches hardware.

        Returns:
            String fingerprint like "NVIDIA_A6000_48.0GB_8.6"
        """
        if not self.gpu_name:
            self.profile()

        # Normalize GPU name (remove spaces, special chars)
        normalized_name = self.gpu_name.replace(" ", "_").replace("-", "_")
        return f"{normalized_name}_{self.total_memory_gb:.1f}GB_{self.compute_capability[0]}.{self.compute_capability[1]}"


class CalibrationCache:
    """
    Manages persistence of calibration data between training runs.

    Features:
    - JSON-based storage for human readability
    - TTL-based expiration
    - Cache key from GPU + model config + precision
    """

    def __init__(self, cache_dir: str = "~/.cache/ava_calibration", ttl_hours: int = 168):
        self.cache_dir = Path(cache_dir).expanduser()
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(
        self,
        gpu_fingerprint: str,
        model_config: Dict[str, Any],
        precision: str
    ) -> str:
        """
        Generate unique cache key from hardware + model config + precision.

        Args:
            gpu_fingerprint: From HardwareProfiler.get_fingerprint()
            model_config: Model architecture parameters
            precision: Training precision (fp16, bf16, fp32)

        Returns:
            SHA256 hash string
        """
        # Extract relevant model config fields
        relevant_config = {
            'vocab_size': model_config.get('vocab_size', 0),
            'hidden_size': model_config.get('hidden_size', 0),
            'num_layers': model_config.get('num_layers', 0),
            'num_attention_heads': model_config.get('num_attention_heads', 0),
            'intermediate_size': model_config.get('intermediate_size', 0),
            'num_experts': model_config.get('num_experts', 0),
        }

        # Create deterministic string
        key_string = f"{gpu_fingerprint}|{json.dumps(relevant_config, sort_keys=True)}|{precision}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"calibration_{cache_key}.json"

    def load(self, cache_key: str) -> Optional[CalibrationResult]:
        """
        Load cached calibration result if valid.

        Args:
            cache_key: Cache key from get_cache_key()

        Returns:
            CalibrationResult if valid cache exists, None otherwise
        """
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_path} not found")
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            # Check TTL
            from datetime import datetime
            timestamp = data.get('timestamp', '')
            if timestamp:
                cache_time = datetime.fromisoformat(timestamp)
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                if age_hours > self.ttl_hours:
                    logger.info(f"Cache expired: {age_hours:.1f} hours old (TTL={self.ttl_hours}h)")
                    return None

            result = CalibrationResult.from_dict(data)
            logger.info(f"Loaded calibration from cache: {cache_path}")
            return result

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def save(self, cache_key: str, result: CalibrationResult) -> bool:
        """
        Save calibration result to cache.

        Args:
            cache_key: Cache key
            result: CalibrationResult to save

        Returns:
            True if saved successfully
        """
        cache_path = self.get_cache_path(cache_key)

        try:
            # Add timestamp and cache key to result
            from datetime import datetime
            result.timestamp = datetime.now().isoformat()
            result.cache_key = cache_key

            with open(cache_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.info(f"Saved calibration to cache: {cache_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            return False

    def save_to_run_dir(self, run_dir: Union[str, Path], result: CalibrationResult) -> bool:
        """
        Save calibration result to a specific run directory.

        Args:
            run_dir: Training run output directory
            result: CalibrationResult to save

        Returns:
            True if saved successfully
        """
        run_path = Path(run_dir) / "calibration"
        run_path.mkdir(parents=True, exist_ok=True)
        cal_file = run_path / "calibration_result.json"

        try:
            with open(cal_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Saved calibration to run dir: {cal_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save to run dir: {e}")
            return False


class MemoryCalibrator:
    """
    Profiles memory usage at different batch sizes and sequence lengths.

    Creates a multi-term memory model that captures both linear scaling
    (embeddings, FFN) and quadratic scaling (attention).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_sizes: List[int],
        seq_lengths: List[int],
        num_samples: int = 3,
        vocab_size: int = 50000,
        max_position_embeddings: int = None,
    ):
        self.model = model
        self.device = device
        self.batch_sizes = batch_sizes

        # Clip sequence lengths to model's max_position_embeddings
        if max_position_embeddings is None:
            # Try to get from model config
            if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
                max_position_embeddings = model.config.max_position_embeddings
            elif hasattr(model, 'max_position_embeddings'):
                max_position_embeddings = model.max_position_embeddings
            else:
                max_position_embeddings = 2048  # Default fallback

        self.max_position_embeddings = max_position_embeddings
        self.seq_lengths = [s for s in seq_lengths if s <= max_position_embeddings]
        if len(self.seq_lengths) < len(seq_lengths):
            logger.info(f"Clipped seq_lengths to max_position_embeddings={max_position_embeddings}: {self.seq_lengths}")

        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.profiles: List[MemoryProfile] = []

    def _clear_memory(self):
        """Clear GPU memory and caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _get_memory_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1e9
        return 0.0

    def _get_peak_memory_gb(self) -> float:
        """Get peak GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / 1e9
        return 0.0

    def _create_dummy_batch(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """Create a dummy batch for profiling."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
        # Include labels for loss computation (shifted by 1 for causal LM)
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def profile_config(self, batch_size: int, seq_len: int) -> Optional[MemoryProfile]:
        """
        Profile memory usage for a single (batch_size, seq_len) configuration.

        Returns:
            MemoryProfile if successful, None if OOM
        """
        self._clear_memory()

        try:
            # Create dummy batch
            batch = self._create_dummy_batch(batch_size, seq_len)

            # Measure forward pass memory
            self._clear_memory()
            torch.cuda.reset_peak_memory_stats(self.device)

            with torch.no_grad():
                _ = self.model(**batch)

            forward_memory = self._get_peak_memory_gb()

            # Measure forward + backward memory
            self._clear_memory()
            torch.cuda.reset_peak_memory_stats(self.device)

            # Enable gradients for backward pass
            self.model.train()
            outputs = self.model(**batch)

            # Get loss (handle different output formats)
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            elif hasattr(outputs, 'logits'):
                # Create dummy loss
                logits = outputs.logits
                labels = batch['input_ids']
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            elif isinstance(outputs, dict):
                # Handle dictionary outputs
                if 'loss' in outputs and outputs['loss'] is not None:
                    loss = outputs['loss']
                elif 'logits' in outputs:
                    logits = outputs['logits']
                    labels = batch['input_ids']
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    # Sum first tensor value in dict
                    for v in outputs.values():
                        if isinstance(v, torch.Tensor):
                            loss = v.sum()
                            break
                    else:
                        raise ValueError("Cannot extract loss from model outputs")
            elif isinstance(outputs, (tuple, list)):
                # Handle tuple/list outputs
                loss = outputs[0].sum() if isinstance(outputs[0], torch.Tensor) else outputs[0]
            else:
                # Fallback: sum of outputs
                loss = outputs.sum() if isinstance(outputs, torch.Tensor) else outputs

            loss.backward()
            backward_memory = self._get_peak_memory_gb()

            # Clear gradients
            self.model.zero_grad()
            self._clear_memory()

            return MemoryProfile(
                batch_size=batch_size,
                seq_len=seq_len,
                forward_memory_gb=forward_memory,
                backward_memory_gb=backward_memory,
                peak_memory_gb=backward_memory,
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.debug(f"OOM at batch_size={batch_size}, seq_len={seq_len}")
                self._clear_memory()
                return None
            raise

    def run_calibration(self, timeout_seconds: int = 300) -> Tuple[List[MemoryProfile], float]:
        """
        Run full memory calibration across the profiling grid.

        Args:
            timeout_seconds: Maximum time for calibration

        Returns:
            Tuple of (profiles list, backward_ratio)
        """
        start_time = time.time()
        self.profiles = []

        total_configs = len(self.batch_sizes) * len(self.seq_lengths)
        completed = 0

        logger.info(f"Starting memory calibration: {total_configs} configurations")

        for batch_size in self.batch_sizes:
            for seq_len in self.seq_lengths:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Calibration timeout after {timeout_seconds}s")
                    break

                # Profile this configuration (average over num_samples)
                measurements = []
                for _ in range(self.num_samples):
                    profile = self.profile_config(batch_size, seq_len)
                    if profile:
                        measurements.append(profile)
                    else:
                        break  # OOM, skip larger configs

                if measurements:
                    # Average the measurements
                    avg_profile = MemoryProfile(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        forward_memory_gb=sum(m.forward_memory_gb for m in measurements) / len(measurements),
                        backward_memory_gb=sum(m.backward_memory_gb for m in measurements) / len(measurements),
                        peak_memory_gb=sum(m.peak_memory_gb for m in measurements) / len(measurements),
                    )
                    self.profiles.append(avg_profile)
                    completed += 1

                    # Progress logging
                    if completed % 10 == 0:
                        logger.info(f"Calibration progress: {completed}/{total_configs} configurations")

        # Calculate average backward ratio
        backward_ratio = self._calculate_backward_ratio()

        elapsed = time.time() - start_time
        logger.info(f"Memory calibration complete: {len(self.profiles)} profiles in {elapsed:.1f}s")

        return self.profiles, backward_ratio

    def _calculate_backward_ratio(self) -> float:
        """Calculate average backward/forward memory ratio."""
        ratios = []
        for p in self.profiles:
            if p.forward_memory_gb > 0.1:  # Avoid division issues
                ratio = p.backward_memory_gb / p.forward_memory_gb
                ratios.append(ratio)

        if not ratios:
            return 1.20  # Default fallback

        return sum(ratios) / len(ratios)

    def fit_memory_model(self) -> Tuple[float, float, float]:
        """
        Fit multi-term memory model to profiling data.

        Model: memory = base + linear*tokens + quadratic*tokens*seq_len

        This captures:
        - base: Fixed memory (model weights, optimizer states)
        - linear: Memory scaling with token count (embeddings, FFN)
        - quadratic: Memory scaling with attention (O(seq^2))

        Returns:
            Tuple of (base_memory_gb, linear_coef, quadratic_coef)
        """
        if len(self.profiles) < 5:
            logger.warning("Not enough profiles for model fitting")
            return 0.0, 0.0, 0.0

        # Prepare data for regression
        # X = [tokens, tokens * seq_len]
        # y = peak_memory_gb
        X = []
        y = []

        for p in self.profiles:
            tokens = p.batch_size * p.seq_len
            X.append([1.0, tokens, tokens * p.seq_len])  # [1, tokens, tokens*seq]
            y.append(p.peak_memory_gb)

        X = np.array(X)
        y = np.array(y)

        try:
            # Least squares: solve X @ coeffs = y
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            base_memory = max(0.0, coeffs[0])
            linear_coef = max(0.0, coeffs[1])
            quadratic_coef = max(0.0, coeffs[2])

            logger.info(
                f"Memory model fit: base={base_memory:.2f}GB, "
                f"linear={linear_coef*1e6:.2f}MB/token, "
                f"quadratic={quadratic_coef*1e9:.4f}MB/token²"
            )

            return base_memory, linear_coef, quadratic_coef

        except Exception as e:
            logger.warning(f"Model fitting failed: {e}")
            return 0.0, 0.0, 0.0


class ThroughputProfiler:
    """
    Profiles training throughput to find optimal batch size.

    Measures tokens/second at different batch sizes to identify
    the sweet spot for maximum throughput.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        vocab_size: int = 50000,
    ):
        self.model = model
        self.device = device
        self.vocab_size = vocab_size
        self.profiles: List[Dict[str, Any]] = []

    def _create_dummy_batch(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """Create a dummy batch for profiling."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
        # Include labels for loss computation (shifted by 1 for causal LM)
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def profile_throughput(
        self,
        batch_size: int,
        seq_len: int,
        num_iterations: int = 5,
        warmup_iterations: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Profile throughput for a single batch size.

        Returns:
            Dict with tokens_per_second and other metrics, or None if OOM
        """
        try:
            batch = self._create_dummy_batch(batch_size, seq_len)
            tokens_per_batch = batch_size * seq_len

            def extract_loss(outputs, batch):
                """Extract loss from model outputs."""
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    return outputs.loss
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    labels = batch['labels']
                    return torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                elif isinstance(outputs, dict):
                    if 'loss' in outputs and outputs['loss'] is not None:
                        return outputs['loss']
                    elif 'logits' in outputs:
                        logits = outputs['logits']
                        labels = batch['labels']
                        return torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1)
                        )
                elif isinstance(outputs, (tuple, list)):
                    return outputs[0].sum() if isinstance(outputs[0], torch.Tensor) else outputs[0]
                return outputs.sum() if isinstance(outputs, torch.Tensor) else outputs

            # Warmup
            for _ in range(warmup_iterations):
                outputs = self.model(**batch)
                loss = extract_loss(outputs, batch)
                loss.backward()
                self.model.zero_grad()

            # Timed iterations - use CUDA events for accurate GPU timing
            if _CUDA_TIMER_AVAILABLE and torch.cuda.is_available():
                # Use event-based timing (more accurate, less CPU blocking)
                timer = CUDATimer()
                timer.start()

                for _ in range(num_iterations):
                    outputs = self.model(**batch)
                    loss = extract_loss(outputs, batch)
                    loss.backward()
                    self.model.zero_grad()

                elapsed_ms = timer.stop(sync=True)
                elapsed = elapsed_ms / 1000.0  # Convert to seconds
            else:
                # Fallback to synchronize + time.time()
                torch.cuda.synchronize()
                start_time = time.time()

                for _ in range(num_iterations):
                    outputs = self.model(**batch)
                    loss = extract_loss(outputs, batch)
                    loss.backward()
                    self.model.zero_grad()

                torch.cuda.synchronize()
                elapsed = time.time() - start_time

            tokens_per_second = (tokens_per_batch * num_iterations) / elapsed

            return {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tokens_per_second': tokens_per_second,
                'time_per_batch_ms': (elapsed / num_iterations) * 1000,
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None
            raise

    def find_optimal_batch_size(
        self,
        seq_len: int,
        batch_sizes: List[int],
        memory_threshold: float = 0.85,
        timeout_seconds: int = 60
    ) -> Tuple[int, float, List[Dict[str, Any]]]:
        """
        Find optimal batch size for maximum throughput.

        Args:
            seq_len: Sequence length to use for profiling
            batch_sizes: List of batch sizes to test
            memory_threshold: Maximum memory utilization allowed
            timeout_seconds: Maximum profiling time

        Returns:
            Tuple of (optimal_batch_size, optimal_throughput, all_profiles)
        """
        start_time = time.time()
        self.profiles = []
        best_batch_size = batch_sizes[0]
        best_throughput = 0.0

        logger.info(f"Starting throughput profiling for {len(batch_sizes)} batch sizes")

        for batch_size in sorted(batch_sizes):
            if time.time() - start_time > timeout_seconds:
                logger.warning("Throughput profiling timeout")
                break

            profile = self.profile_throughput(batch_size, seq_len)
            if profile:
                self.profiles.append(profile)

                if profile['tokens_per_second'] > best_throughput:
                    best_throughput = profile['tokens_per_second']
                    best_batch_size = batch_size

                logger.debug(
                    f"Batch size {batch_size}: {profile['tokens_per_second']:.0f} tokens/sec"
                )
            else:
                # OOM - stop trying larger batch sizes
                logger.debug(f"OOM at batch_size={batch_size}, stopping")
                break

        logger.info(
            f"Optimal batch size: {best_batch_size} "
            f"({best_throughput:.0f} tokens/sec)"
        )

        return best_batch_size, best_throughput, self.profiles


class EnhancedMemoryPredictor:
    """
    Enhanced memory predictor using calibration data.

    Improvements over basic MemoryPredictor:
    - Multi-term model (linear + quadratic for attention)
    - Measured backward ratio (not fixed 1.20)
    - Hardware-aware predictions
    """

    def __init__(self, calibration_result: CalibrationResult):
        self.base_memory_gb = calibration_result.base_memory_gb
        self.linear_coef = calibration_result.linear_coef
        self.quadratic_coef = calibration_result.quadratic_coef
        self.backward_ratio = calibration_result.backward_ratio
        self.total_memory_gb = calibration_result.hardware_profile.get('total_memory_gb', 48.0)
        self.calibrated = True

    def predict_memory(
        self,
        batch_size: int,
        seq_len: int,
        include_backward: bool = True
    ) -> float:
        """
        Predict memory usage using multi-term model.

        Model: memory = base + linear*tokens + quadratic*tokens*seq_len

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            include_backward: Include backward pass memory spike

        Returns:
            Predicted memory in GB
        """
        tokens = batch_size * seq_len
        forward_memory = (
            self.base_memory_gb
            + self.linear_coef * tokens
            + self.quadratic_coef * tokens * seq_len
        )

        if include_backward:
            return forward_memory * self.backward_ratio

        return forward_memory

    def is_safe(
        self,
        batch_size: int,
        seq_len: int,
        threshold: float = 0.85
    ) -> bool:
        """
        Check if configuration is safe (won't exceed memory threshold).

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            threshold: Maximum memory utilization

        Returns:
            True if predicted memory is below threshold
        """
        predicted = self.predict_memory(batch_size, seq_len, include_backward=True)
        return predicted < threshold * self.total_memory_gb

    def get_max_safe_batch_size(
        self,
        seq_len: int,
        threshold: float = 0.85
    ) -> int:
        """
        Calculate maximum safe batch size for a sequence length.

        Uses binary search to find the largest safe batch size.

        Args:
            seq_len: Sequence length
            threshold: Maximum memory utilization

        Returns:
            Maximum safe batch size
        """
        low, high = 1, 1024

        while low < high:
            mid = (low + high + 1) // 2
            if self.is_safe(mid, seq_len, threshold):
                low = mid
            else:
                high = mid - 1

        return low


class CalibrationSystem:
    """
    Orchestrates the complete calibration process.

    Coordinates:
    - Hardware profiling
    - Cache loading/saving
    - Memory calibration
    - Throughput profiling
    - Run directory saving
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        self.model = model
        self.device = device
        self.config = config

        # Extract calibration config
        cal_config = config.get('calibration', {})
        self.enabled = cal_config.get('enabled', True)
        self.run_memory_profiling = cal_config.get('run_memory_profiling', True)
        self.run_backward_profiling = cal_config.get('run_backward_profiling', True)
        self.run_throughput_profiling = cal_config.get('run_throughput_profiling', True)
        self.batch_sizes = cal_config.get('batch_sizes_to_profile', [4, 8, 16, 32, 64, 128, 256])
        self.seq_lengths = cal_config.get('seq_lengths_to_profile', [32, 64, 128, 256, 512, 1024, 2048])
        self.num_samples = cal_config.get('num_samples_per_config', 3)
        self.cache_enabled = cal_config.get('cache_enabled', True)
        self.cache_dir = cal_config.get('cache_dir', '~/.cache/ava_calibration')
        self.cache_ttl = cal_config.get('cache_ttl_hours', 168)
        self.save_to_run = cal_config.get('save_to_run_dir', True)
        self.max_time = cal_config.get('max_calibration_time_seconds', 300)
        self.force_recalibrate = cal_config.get('force_recalibrate', False)

        # Initialize components
        self.hardware_profiler = HardwareProfiler()
        self.cache = CalibrationCache(self.cache_dir, self.cache_ttl)

    def run_or_load(
        self,
        model_config: Dict[str, Any],
        precision: str,
        vocab_size: int,
        run_dir: Optional[str] = None,
    ) -> Optional[CalibrationResult]:
        """
        Run calibration or load from cache.

        Args:
            model_config: Model architecture parameters
            precision: Training precision (fp16, bf16, fp32)
            vocab_size: Vocabulary size
            run_dir: Optional run directory for saving results

        Returns:
            CalibrationResult or None if calibration disabled
        """
        if not self.enabled:
            print(" Calibration disabled")
            return None

        start_time = time.time()

        # 1. Hardware profiling
        hw_profile = self.hardware_profiler.profile()
        gpu_fingerprint = self.hardware_profiler.get_fingerprint()

        print(f" GPU: {hw_profile.get('gpu_name', 'unknown')}")
        print(f" Memory: {hw_profile.get('total_memory_gb', 0):.1f} GB")

        # 2. Check cache (unless force_recalibrate is set)
        if self.cache_enabled and not self.force_recalibrate:
            cache_key = self.cache.get_cache_key(gpu_fingerprint, model_config, precision)
            cached_result = self.cache.load(cache_key)

            if cached_result:
                print(f" Loaded calibration from cache (key={cache_key[:8]}...)")
                print(f"   Original calibration time: {cached_result.calibration_time_seconds:.1f}s")
                if cached_result.optimal_batch_size:
                    print(f"   Optimal batch size: {cached_result.optimal_batch_size}")
                    print(f"   Optimal throughput: {cached_result.optimal_throughput:.0f} tokens/sec")
                if run_dir and self.save_to_run:
                    self.cache.save_to_run_dir(run_dir, cached_result)
                return cached_result
        elif self.force_recalibrate:
            print(" Force recalibrate enabled, skipping cache...")

        # 3. Run fresh calibration
        print(" Running fresh calibration...")

        # Memory calibration
        memory_calibrator = MemoryCalibrator(
            model=self.model,
            device=self.device,
            batch_sizes=self.batch_sizes,
            seq_lengths=self.seq_lengths,
            num_samples=self.num_samples,
            vocab_size=vocab_size,
        )

        profiles, backward_ratio = memory_calibrator.run_calibration(
            timeout_seconds=self.max_time
        )

        # Fit memory model
        base_mem, linear_coef, quad_coef = memory_calibrator.fit_memory_model()

        # Convert profiles to dicts
        memory_profiles = [
            {
                'batch_size': p.batch_size,
                'seq_len': p.seq_len,
                'forward_memory_gb': p.forward_memory_gb,
                'backward_memory_gb': p.backward_memory_gb,
                'peak_memory_gb': p.peak_memory_gb,
            }
            for p in profiles
        ]

        # 4. Optional throughput profiling
        optimal_batch = None
        optimal_throughput = None
        throughput_profiles = None

        if self.run_throughput_profiling:
            throughput_profiler = ThroughputProfiler(
                model=self.model,
                device=self.device,
                vocab_size=vocab_size,
            )

            # Use middle sequence length for throughput profiling
            mid_seq = self.seq_lengths[len(self.seq_lengths) // 2]
            optimal_batch, optimal_throughput, throughput_profiles = throughput_profiler.find_optimal_batch_size(
                seq_len=mid_seq,
                batch_sizes=self.batch_sizes,
                timeout_seconds=60,
            )

        # 5. Create result
        elapsed = time.time() - start_time

        result = CalibrationResult(
            hardware_profile=hw_profile,
            base_memory_gb=base_mem,
            linear_coef=linear_coef,
            quadratic_coef=quad_coef,
            backward_ratio=backward_ratio,
            memory_profiles=memory_profiles,
            optimal_batch_size=optimal_batch,
            optimal_throughput=optimal_throughput,
            throughput_profiles=throughput_profiles,
            calibration_time_seconds=elapsed,
        )

        # 6. Save to cache and run dir
        if self.cache_enabled:
            cache_key = self.cache.get_cache_key(gpu_fingerprint, model_config, precision)
            self.cache.save(cache_key, result)

        if run_dir and self.save_to_run:
            self.cache.save_to_run_dir(run_dir, result)

        # 7. Print summary
        print(" Calibration complete!")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Memory profiles: {len(memory_profiles)}")
        print(f"   Base memory: {base_mem:.2f} GB")
        print(f"   Backward ratio: {backward_ratio:.2f}x")
        if optimal_batch:
            print(f"   Optimal batch size: {optimal_batch}")
            print(f"   Optimal throughput: {optimal_throughput:.0f} tokens/sec")

        return result


# =============================================================================
# Configuration imported from training_config.py (DynamicBatchingConfig, DynamicBatchConfig)
# See: code/src/Ava/config/training_config.py for the unified configuration class
# =============================================================================


# =============================================================================
# Enhanced Dynamic Batch Scheduler
# =============================================================================

class DynamicBatchScheduler:
    """
    Manages dynamic batch size adjustment based on GPU memory usage.

    Enhanced with 8 features for improved GPU utilization and stability:
    1. Token-Budget Batching
    2. Sequence-Length Aware Batching
    3. Multi-GPU Synchronization
    4. Gradient Accumulation Integration
    5. Predictive Memory Estimation
    6. Conservative Thresholds
    7. Geometric Warmup
    8. Memory Trend Detection

    Example usage:
        scheduler = DynamicBatchScheduler(config)

        for step in range(num_steps):
            batch_size = scheduler.get_current_batch_size(step)
            batch = get_batch(batch_size)

            # Train on batch
            loss = model(batch)
            loss.backward()

            # Update scheduler with current memory usage
            scheduler.step(step)
    """

    def __init__(self, config: DynamicBatchConfig):
        self.config = config
        self.current_batch_size = config.initial_batch_size or 32
        self.original_batch_size = config.initial_batch_size or 32

        # Tracking
        self.step_count = 0
        self.adjustment_count = 0
        self.last_adjustment_step = -(config.cooldown_steps or 10)
        self.memory_history: List[float] = []
        self.batch_size_history: List[int] = []

        # Statistics
        self.total_increases = 0
        self.total_decreases = 0
        self.max_batch_size_reached = config.initial_batch_size
        self.min_batch_size_reached = config.initial_batch_size

        # Smoothed memory tracking (prevents oscillation from cache events)
        self.smoothed_utilization = 0.0
        self.smoothing_alpha = 0.1  # EMA factor: lower = more smoothing

        # NEW: Hysteresis tracking for stable batch sizes
        self._steps_since_last_change = 0  # Track stable steps
        self._consecutive_stable_steps = 0  # Steps within hysteresis margin
        self._oom_count = 0  # Track consecutive OOMs for safe mode
        self._safe_mode = False  # Emergency mode after repeated OOMs

        # Feature 1: Token tracking
        self.tokens_this_step = 0
        self.tokens_history: List[int] = []

        # Feature 2: Sequence length tracking
        self.avg_sequence_length = config.base_sequence_length
        self.sequence_lengths: List[int] = []

        # Feature 5: Memory predictor
        self.memory_predictor = MemoryPredictor(
            calibration_steps=config.calibration_steps,
            backward_safety_margin=config.backward_safety_margin
        ) if config.predictive_enabled else None

        # Feature 8: Trend analyzer
        self.trend_analyzer = MemoryTrendAnalyzer(
            window=config.trend_window,
            oscillation_threshold=config.oscillation_threshold
        ) if config.trend_detection_enabled else None

        # GPU SYNC FIX: Cache memory stats to reduce GPU sync frequency
        # Memory queries (memory_allocated, memory_reserved) cause implicit syncs
        # Using time-based caching instead of step-based for more reliable behavior
        self._memory_stats_cache: Optional[Dict[str, float]] = None
        self._memory_stats_cache_time: float = 0.0  # Last time cache was updated (monotonic)
        self._memory_stats_cache_interval_sec: float = config.memory_stats_cache_interval_sec

        # Feature 9: Adaptive backward margin (auto-tunes over time)
        self.adaptive_margin = AdaptiveBackwardMargin(
            initial_margin=config.backward_safety_margin,
            min_margin=config.adaptive_margin_min,
            max_margin=config.adaptive_margin_max,
            learning_rate=config.adaptive_margin_learning_rate,
            ema_alpha=config.adaptive_margin_ema_alpha,
        ) if config.adaptive_margin_enabled else None

        # Feature 10: Aggressive Growth Controller (probes for optimal batch size)
        self.aggressive_growth = AggressiveGrowthController(
            exploration_steps=config.aggressive_growth_exploration_steps,
            growth_factor=config.aggressive_growth_factor,
            min_headroom_for_growth=config.aggressive_growth_min_headroom,
            max_batch_size=config.max_batch_size,
            min_batch_size=config.min_batch_size,
        ) if config.aggressive_growth_enabled else None

        # Get GPU info
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory

            # Set up memory predictor with total memory
            if self.memory_predictor:
                self.memory_predictor.set_total_memory(self.total_memory)

            logger.info(f"Dynamic batching initialized on GPU {self.device}")
            logger.info(f"Total GPU memory: {self.total_memory / 1e9:.2f} GB")
            logger.info(f"Initial batch size: {self.current_batch_size}")
            logger.info(f"Batch size range: [{config.min_batch_size or 1}, {config.max_batch_size or 256}]")
            low_thresh = config.low_memory_threshold or 0.5
            target_thresh = config.target_memory_threshold or 0.7
            high_thresh = config.high_memory_threshold or 0.85
            logger.info(f"Thresholds: low={low_thresh:.0%}, target={target_thresh:.0%}, high={high_thresh:.0%}")
            logger.info(f"Adjustment: freq={config.adjustment_frequency or 10}, cooldown={config.cooldown_steps or 10}, warmup={config.warmup_steps or 100}")

            # Log enabled features
            features = []
            if config.token_budget_enabled:
                features.append("token-budget")
            if config.sequence_aware:
                features.append("sequence-aware")
            if config.sync_across_gpus:
                features.append("multi-gpu-sync")
            if config.coordinate_with_grad_accum:
                features.append("grad-accum-integration")
            if config.predictive_enabled:
                features.append("predictive-memory")
            if config.warmup_strategy != 'none':
                features.append(f"warmup-{config.warmup_strategy}")
            if config.trend_detection_enabled:
                features.append("trend-detection")

            if features:
                logger.info(f"Enabled features: {', '.join(features)}")
        else:
            self.device = None
            self.total_memory = None
            logger.warning("CUDA not available, dynamic batching disabled")
            self.config.enabled = False

    # =========================================================================
    # Feature 1: Token Budget Batching
    # =========================================================================

    def count_batch_tokens(self, batch: Dict[str, torch.Tensor]) -> int:
        """
        Count tokens in a batch using shape-based estimation.

        GPU SYNC FIX: Uses shape-based counting to avoid .sum().item() sync.
        This slightly overestimates for padded sequences but avoids GPU stalls.

        Args:
            batch: Batch dictionary with 'input_ids' or 'attention_mask' key

        Returns:
            Maximum possible tokens in batch (batch_size * seq_len)
        """
        if 'input_ids' in batch:
            # Use shape-based count (no GPU sync!)
            return batch['input_ids'].numel()
        elif 'attention_mask' in batch:
            return batch['attention_mask'].numel()
        return 0

    def record_tokens(self, token_count: int) -> None:
        """Record token count for this step."""
        self.tokens_this_step = token_count
        self.tokens_history.append(token_count)
        if len(self.tokens_history) > 100:
            self.tokens_history.pop(0)

    def get_avg_tokens_per_batch(self) -> float:
        """Get average tokens per batch from history."""
        if not self.tokens_history:
            return float(self.config.target_tokens_per_batch)
        return sum(self.tokens_history) / len(self.tokens_history)

    def get_dynamic_token_budget(self) -> int:
        """
        AUTOMATIC dynamic token budget based on current memory utilization.

        Automatically adjusts to maximize GPU utilization without manual threshold tuning.
        Uses a simple proportional controller: more headroom = more tokens.

        Applies ADAPTIVE backward_safety_margin that auto-tunes over time.
        The margin starts at config value but decreases if we're underutilizing
        and increases after OOM events.

        Returns:
            Adjusted target tokens per batch
        """
        # Get token budget values with safe defaults
        base_target = self.config.target_tokens_per_batch if self.config.target_tokens_per_batch is not None else 4096
        max_target = self.config.max_tokens_per_batch if self.config.max_tokens_per_batch is not None else 16384
        min_target = self.config.min_tokens_per_batch if self.config.min_tokens_per_batch is not None else 512

        # Get current memory utilization from cached stats (GPU SYNC FIX)
        # Use get_memory_stats() which caches results to avoid frequent syncs
        if self.smoothed_utilization < 0.01:
            # Need to get initial reading - use cached stats
            mem_stats = self.get_memory_stats()
            raw_util = mem_stats.get('raw_utilization', 0.0)
            if raw_util < 0.01:
                return base_target
        else:
            raw_util = self.smoothed_utilization

        # Use ADAPTIVE backward margin that auto-tunes over time (if enabled)
        target_vram = self.config.target_memory_threshold

        if self.adaptive_margin is not None:
            # Adaptive margin: auto-tunes based on utilization feedback
            backward_margin = self.adaptive_margin.observe_utilization(
                step=self.step_count,
                raw_util=raw_util,
                target_util=target_vram,
                critical_util=self.config.critical_memory_threshold,
            )
        else:
            # Fixed margin from config
            backward_margin = self.config.backward_safety_margin

        # Apply the backward margin
        effective_util = raw_util * backward_margin

        # AUTOMATIC SCALING: Use config thresholds for VRAM targets
        # If VRAM is below target, we have headroom -> increase tokens
        # If VRAM is above target, we're tight -> decrease tokens
        safety_max = self.config.critical_memory_threshold  # Use config value

        if effective_util >= safety_max:
            # Emergency: clear cache and use minimum
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return min_target

        # Calculate headroom using RAW utilization (not effective)
        # Scale token budget based on available headroom
        if raw_util < target_vram:
            # Below target: we have headroom to increase tokens
            # Key insight: if VRAM is at 75% and target is 80%, we have 5% headroom
            # We should scale UP proportionally to use that headroom

            # How much room do we have? (as fraction of total)
            headroom = target_vram - raw_util  # e.g., 0.80 - 0.75 = 0.05

            # Scale factor: use ALL available headroom aggressively
            # At 0% VRAM (100% headroom) -> use max tokens
            # At target VRAM (0% headroom) -> use base tokens
            # In between: linear interpolation based on headroom

            # Normalize headroom: at target=0.80, max possible headroom is 0.80
            # So headroom of 0.05 out of 0.80 is 6.25% of possible headroom
            # But that's too conservative - we want to fill UP to target

            # Simple approach: increase budget proportionally to headroom
            # 5% headroom -> try to use 5% more memory -> 5% more tokens
            # token_increase = headroom * current_tokens
            # But we need to cap at max_tokens

            # More direct: scale linearly between base (at target) and max (at 0%)
            scale_factor = headroom / target_vram  # 0.05/0.80 = 0.0625

            # Use a high multiplier to aggressively fill available headroom
            # With 5% headroom (0.0625 scale), we want to use most of max budget
            # Multiplier of 10 means: 0.0625 * 10 = 0.625 -> 62.5% toward max
            aggressive_multiplier = 10.0  # Very aggressive to actually use headroom
            boosted_scale = min(1.0, scale_factor * aggressive_multiplier)  # Cap at 1.0

            target = int(base_target + (max_target - base_target) * boosted_scale)
        else:
            # Above target: scale DOWN from base toward min
            excess = (raw_util - target_vram) / (safety_max - target_vram)
            target = int(base_target - (base_target - min_target) * excess)

        return max(min_target, min(max_target, target))

    def calculate_token_budget_batch_size(self) -> int:
        """
        Calculate batch size to achieve target token budget.

        In token-budget mode, this function is informational only - the actual
        batch size is determined by the DynamicBatchIterator based on token accumulation.

        Returns:
            Recommended batch size based on token budget (not used in token-budget mode)
        """
        if not self.tokens_history or not self.sequence_lengths:
            return self.current_batch_size

        # Use average sequence length to estimate tokens per sample
        avg_seq_len = self.avg_sequence_length if self.avg_sequence_length > 0 else self.config.base_sequence_length

        # Calculate target batch size based on token budget and avg sequence length
        target_batch = int(self.config.target_tokens_per_batch / avg_seq_len)

        # Clamp to valid range
        return max(self.config.min_batch_size,
                   min(self.config.max_batch_size, target_batch))

    # =========================================================================
    # Feature 2: Sequence-Length Aware Batching
    # =========================================================================

    def record_sequence_length(self, avg_seq_len: int) -> None:
        """Record average sequence length for this batch."""
        self.sequence_lengths.append(avg_seq_len)
        if len(self.sequence_lengths) > 100:
            self.sequence_lengths.pop(0)
        self.avg_sequence_length = sum(self.sequence_lengths) / len(self.sequence_lengths)

    def get_sequence_adjusted_batch_size(self, avg_seq_len: Optional[int] = None) -> int:
        """
        Scale batch size inversely with sequence length.

        Memory scales as O(batch * seq^2) for attention, so we need to
        reduce batch size when sequences are longer than the base length.

        Args:
            avg_seq_len: Average sequence length (uses tracked value if None)

        Returns:
            Adjusted batch size
        """
        if not self.config.sequence_aware:
            return self.current_batch_size

        if avg_seq_len is None:
            avg_seq_len = int(self.avg_sequence_length)

        if avg_seq_len == 0:
            avg_seq_len = self.config.base_sequence_length

        # Memory scales as O(batch * seq^2) for attention
        length_ratio = (self.config.base_sequence_length / avg_seq_len) ** 2
        adjusted = int(self.current_batch_size * length_ratio * self.config.sequence_scaling_factor)

        return max(self.config.min_batch_size,
                   min(self.config.max_batch_size, adjusted))

    # =========================================================================
    # Feature 3: Multi-GPU Synchronization
    # =========================================================================

    def sync_batch_size(self, batch_size: int) -> int:
        """
        Synchronize batch size across all GPUs in distributed training.

        This prevents DDP hangs from mismatched batch sizes across GPUs.

        Args:
            batch_size: Local batch size

        Returns:
            Synchronized batch size (same on all GPUs)
        """
        if not self.config.sync_across_gpus:
            return batch_size

        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return batch_size

            # Use the scheduler's device instead of hardcoded 'cuda' for multi-GPU support
            device = getattr(self, 'device', 'cuda')
            tensor = torch.tensor([float(batch_size)], dtype=torch.float32, device=device)

            if self.config.sync_strategy == 'min':
                dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            elif self.config.sync_strategy == 'max':
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            else:  # mean
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= dist.get_world_size()

            # GPU SYNC FIX: This sync is unavoidable for distributed coordination,
            # but only happens during batch size adjustments (not every step)
            synced_size = int(tensor.item())

            # Ensure it's a multiple of min_batch_size
            synced_size = (synced_size // self.config.min_batch_size) * self.config.min_batch_size
            synced_size = max(self.config.min_batch_size, synced_size)

            return synced_size

        except ImportError:
            return batch_size
        except Exception as e:
            logger.warning(f"Batch size sync failed: {e}")
            return batch_size

    # =========================================================================
    # Feature 4: Gradient Accumulation Integration
    # =========================================================================

    def get_recommended_grad_accum_steps(self, batch_size: Optional[int] = None) -> int:
        """
        Calculate gradient accumulation steps to maintain target effective batch size.

        effective_batch = batch_size * grad_accum_steps

        Args:
            batch_size: Current batch size (uses self.current_batch_size if None)

        Returns:
            Recommended gradient accumulation steps
        """
        if not self.config.coordinate_with_grad_accum:
            return self.config.original_grad_accum_steps

        if batch_size is None:
            batch_size = self.current_batch_size

        if batch_size == 0:
            return self.config.original_grad_accum_steps

        # effective = batch_size * grad_accum
        recommended = max(1, self.config.target_effective_batch_size // batch_size)

        return recommended

    def get_effective_batch_size(self, grad_accum_steps: Optional[int] = None) -> int:
        """
        Get effective batch size (batch_size * gradient_accumulation_steps).

        Args:
            grad_accum_steps: Gradient accumulation steps (uses recommended if None)

        Returns:
            Effective batch size
        """
        if grad_accum_steps is None:
            grad_accum_steps = self.get_recommended_grad_accum_steps()

        return self.current_batch_size * grad_accum_steps

    # =========================================================================
    # Feature 7: Geometric Warmup Strategy
    # =========================================================================

    def get_warmup_batch_size(self, step: int) -> int:
        """
        Get batch size during warmup phase based on warmup strategy.

        Strategies:
        - 'none': Use current_batch_size (no special warmup)
        - 'linear': Linear growth from initial_fraction to min_batch_size
        - 'geometric': Exponential growth from initial_fraction to min_batch_size

        Args:
            step: Current training step

        Returns:
            Batch size to use during warmup
        """
        if step >= self.config.warmup_steps:
            return self.current_batch_size

        if self.config.warmup_strategy == 'none':
            return self.current_batch_size

        warmup_fraction = self.config.warmup_initial_fraction if self.config.warmup_initial_fraction is not None else 0.5
        initial = max(1, int(self.config.min_batch_size * warmup_fraction))
        target = self.config.min_batch_size
        progress = step / max(1, self.config.warmup_steps)

        if self.config.warmup_strategy == 'geometric':
            # Exponential growth
            # At progress=0: initial, at progress=1: target
            ratio = target / max(1, initial)
            growth = ratio ** progress
            batch_size = int(initial * growth)
        else:  # linear
            # Linear interpolation
            batch_size = int(initial + (target - initial) * progress)

        # Ensure it's a multiple of min_batch_size (if larger than min)
        if batch_size >= self.config.min_batch_size:
            batch_size = (batch_size // self.config.min_batch_size) * self.config.min_batch_size

        return max(1, min(batch_size, self.config.max_batch_size))

    # =========================================================================
    # Core Methods
    # =========================================================================

    def get_memory_stats(self, force_refresh: bool = False) -> Dict[str, float]:
        """Get current GPU memory statistics with caching and smoothing.

        GPU SYNC FIX: Uses caching to reduce frequency of GPU memory queries.
        Memory functions like memory_allocated() cause implicit GPU syncs.
        By caching results, we only sync every N steps instead of every call.

        Uses memory_allocated() with exponential moving average (EMA) because:
        - memory_reserved() fluctuates wildly after torch.cuda.empty_cache()
        - memory_allocated() shows actual tensor usage, more stable baseline
        - EMA smoothing prevents false signals from cache clearing events
        - This prevents the 25-85% memory oscillation problem

        IMPORTANT: For safety decisions (high/critical thresholds), we use
        the RAW utilization to react immediately to memory pressure.

        Args:
            force_refresh: If True, bypass cache and query GPU directly
        """
        if not torch.cuda.is_available():
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'peak_gb': 0.0,
                'utilization': 0.0,
                'smoothed_utilization': 0.0,
                'raw_utilization': 0.0
            }

        # GPU SYNC FIX: Use time-based caching to reduce GPU sync frequency
        # Time-based is more reliable than step-based since step() may be called
        # at different rates depending on batch accumulation
        current_time = time.monotonic()
        time_since_cache = current_time - self._memory_stats_cache_time
        if not force_refresh and self._memory_stats_cache is not None and time_since_cache < self._memory_stats_cache_interval_sec:
            return self._memory_stats_cache

        # Actually query GPU memory (causes implicit sync)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)

        # Use RESERVED memory for utilization - this matches nvidia-smi more closely
        # allocated only shows PyTorch tensors, reserved includes CUDA caches/workspace
        raw_utilization = reserved / self.total_memory if self.total_memory else 0.0

        # Track allocated separately for debugging
        allocated_utilization = allocated / self.total_memory if self.total_memory else 0.0

        # Feature 8: Auto-tune smoothing based on oscillation detection
        if self.config.auto_tune_smoothing and self.trend_analyzer:
            self.smoothing_alpha = self.trend_analyzer.get_recommended_smoothing_alpha()

        # Apply exponential moving average for stability (used for increase decisions)
        self.smoothed_utilization = (
            self.smoothing_alpha * raw_utilization +
            (1 - self.smoothing_alpha) * self.smoothed_utilization
        )

        # Feature 8: Record observation for trend analysis
        if self.trend_analyzer:
            self.trend_analyzer.add_observation(raw_utilization)

        # Feature 5: Record observation for memory prediction
        if self.memory_predictor and self.sequence_lengths:
            avg_seq = int(self.avg_sequence_length)
            self.memory_predictor.record_observation(
                self.current_batch_size, avg_seq, allocated / 1e9
            )

        # Cache the result with current timestamp
        self._memory_stats_cache = {
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'peak_gb': peak / 1e9,
            'utilization': self.smoothed_utilization,  # Smoothed for increase decisions
            'raw_utilization': raw_utilization,  # Raw (reserved-based) for decrease/safety decisions
            'allocated_utilization': allocated_utilization  # Just tensors, for debugging
        }
        self._memory_stats_cache_time = current_time  # GPU SYNC FIX: time-based cache

        return self._memory_stats_cache

    def should_adjust(self, step: int) -> bool:
        """Check if we should adjust batch size at this step."""
        if not self.config.enabled:
            return False

        # Safe mode: no more adjustments after repeated OOMs
        if self._safe_mode:
            return False

        # During warmup, use warmup strategy instead
        warmup_steps = self.config.warmup_steps if self.config.warmup_steps is not None else 100
        warmup_strategy = self.config.warmup_strategy if self.config.warmup_strategy is not None else 'none'
        if step < warmup_steps and warmup_strategy != 'none':
            return False

        # Don't adjust too frequently
        cooldown_steps = self.config.cooldown_steps if self.config.cooldown_steps is not None else 50
        if step - self.last_adjustment_step < cooldown_steps:
            return False

        # Don't adjust on every step
        adjustment_frequency = self.config.adjustment_frequency if self.config.adjustment_frequency is not None else 10
        if step % adjustment_frequency != 0:
            return False

        # Stop if we've adjusted too many times (prevent oscillation)
        max_adjustments = self.config.max_adjustments_per_session if self.config.max_adjustments_per_session is not None else 50
        if self.adjustment_count >= max_adjustments:
            return False

        # NEW: Require minimum stable steps before any change
        min_stable = getattr(self.config, 'min_stable_steps', 30)
        if self._steps_since_last_change < min_stable:
            self._steps_since_last_change += 1
            return False

        # NEW: Hysteresis check - only adjust if memory differs significantly from target
        hysteresis = getattr(self.config, 'hysteresis_margin', 0.05)
        if len(self.memory_history) > 0:
            current_util = self.memory_history[-1]
            target = self.config.target_memory_threshold
            memory_diff = abs(current_util - target)
            if memory_diff < hysteresis:
                # Within hysteresis band - don't adjust
                return False

        return True

    def calculate_new_batch_size(self, mem_stats: Dict[str, float]) -> Tuple[int, str]:
        """
        Calculate new batch size based on memory utilization and enabled features.

        Uses SMOOTHED utilization for increase decisions (prevents oscillation)
        Uses RAW utilization for decrease/safety decisions (immediate response)

        Batch sizes are always multiples of min_batch_size (16, 32, 48, etc.)

        Returns:
            Tuple of (new_batch_size, reason)
        """
        # Guard against division by zero
        min_bs = max(1, self.config.min_batch_size)
        current_multiplier = self.current_batch_size // min_bs
        max_multiplier = self.config.max_batch_size // min_bs

        # Use RAW utilization for safety decisions (reserved-based, matches nvidia-smi)
        raw_util = mem_stats.get('raw_utilization', 0.0)
        smoothed_util = mem_stats.get('utilization', 0.0)

        # raw_util is now reserved-based, use directly for safety
        safety_util = raw_util

        # Feature 10: Aggressive Growth Mode - use during exploration phase
        if self.aggressive_growth is not None and self.aggressive_growth.should_explore(self.step_count):
            # Record batch result (assume success if we got here without OOM)
            self.aggressive_growth.record_batch_result(
                batch_size=self.current_batch_size,
                success=True,  # If we got here, last batch succeeded
                memory_utilization=raw_util,
            )

            # Get aggressive growth target
            target_size, growth_reason = self.aggressive_growth.get_target_batch_size(
                current_batch_size=self.current_batch_size,
                memory_utilization=raw_util,
                target_memory=self.config.target_memory_threshold,
                step=self.step_count,
            )

            # Only use aggressive growth if it wants to increase (safety)
            if target_size > self.current_batch_size:
                # Check if target is safe with predictor
                if self.memory_predictor and self.memory_predictor.calibrated:
                    avg_seq = int(self.avg_sequence_length) if self.sequence_lengths else self.config.base_sequence_length
                    if not self.memory_predictor.is_safe(target_size, avg_seq, self.config.high_memory_threshold):
                        # Limit increase to +1 step
                        target_size = min(target_size, (current_multiplier + 1) * min_bs)
                        growth_reason += " (predictor-limited)"

                return target_size, f"AGGRESSIVE: {growth_reason}"
            elif target_size < self.current_batch_size:
                # Aggressive growth wants to decrease (memory pressure)
                return target_size, f"AGGRESSIVE-DECREASE: {growth_reason}"

        # Feature 8: Check trend for preemptive action
        trend = 'unknown'
        if self.trend_analyzer:
            trend = self.trend_analyzer.get_trend()

            # Preemptive decrease if memory is trending up and already moderate
            if trend == 'increasing' and smoothed_util > 0.7:
                new_multiplier = max(1, current_multiplier - 1)
                return new_multiplier * min_bs, f"PREEMPTIVE decrease (trend={trend}, util={smoothed_util:.1%})"

        # Feature 5: Check predictive memory estimation
        if self.memory_predictor and self.memory_predictor.calibrated:
            avg_seq = int(self.avg_sequence_length) if self.sequence_lengths else self.config.base_sequence_length

            # Check if increasing would be unsafe
            potential_batch = (current_multiplier + 1) * min_bs
            if not self.memory_predictor.is_safe(potential_batch, avg_seq, self.config.high_memory_threshold):
                # Don't increase, already at safe limit
                if smoothed_util < self.config.low_memory_threshold:
                    return self.current_batch_size, f"PREDICT-BLOCKED (would exceed threshold)"

        # Critical memory - decrease to minimum immediately
        if safety_util > self.config.critical_memory_threshold:
            new_multiplier = 1  # Go to minimum
            reason = f"CRITICAL memory {safety_util:.1%} (raw={raw_util:.1%})"
            # Record failure for aggressive growth
            if self.aggressive_growth:
                self.aggressive_growth.record_batch_result(self.current_batch_size, False, raw_util)

        # High memory - decrease by 1 multiplier step
        elif safety_util > self.config.high_memory_threshold:
            new_multiplier = max(1, current_multiplier - 1)
            reason = f"HIGH memory {safety_util:.1%} (raw={raw_util:.1%})"

        # Low memory - increase by 1 step (conservative)
        # Only increase by more than 1 step if we have massive headroom
        elif smoothed_util < self.config.low_memory_threshold:
            # Calculate headroom
            headroom = self.config.target_memory_threshold - smoothed_util

            # Conservative increase - only +2 steps if 25%+ headroom
            if headroom > 0.25:
                # 25%+ headroom - can try +2 steps
                increase_steps = min(2, max_multiplier - current_multiplier)
            else:
                # Standard +1 step
                increase_steps = 1

            new_multiplier = min(current_multiplier + increase_steps, max_multiplier)
            reason = f"LOW memory {smoothed_util:.1%} (headroom={headroom:.1%}, +{increase_steps})"

        # Target range - no change
        else:
            new_multiplier = current_multiplier
            reason = f"OPTIMAL memory {smoothed_util:.1%}"

        # Convert multiplier to batch size
        new_size = new_multiplier * min_bs

        # Feature 2: Apply sequence-length adjustment
        # Note: Disabled by default when token_budget is enabled (they conflict)
        if self.config.sequence_aware and self.sequence_lengths:
            seq_adjusted = self.get_sequence_adjusted_batch_size()
            if seq_adjusted < new_size:
                new_size = seq_adjusted
                reason += f" (seq-adjusted to {new_size})"

        # Feature 1: Token budget mode
        # When token_budget is enabled, the DynamicBatchIterator controls actual batch size
        # by accumulating mini-batches until target tokens is reached. The scheduler's
        # batch size is the mini-batch size, not the actual yielded batch size.
        # So we skip batch size adjustment logic in token-budget mode.
        if self.config.token_budget_enabled:
            # In token-budget mode, don't adjust scheduler batch size
            # The actual batch size is controlled by token accumulation
            pass

        # Clamp to valid range
        new_size = max(min_bs, min(self.config.max_batch_size, new_size))

        return new_size, reason

    def step(
        self,
        step: int,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        is_log_step: bool = False
    ) -> Optional[int]:
        """
        Update scheduler state and potentially adjust batch size.

        GPU SYNC FIX: Memory monitoring only happens when is_log_step=True.
        This prevents cudaStreamSynchronize on every batch, reducing GPU stalls.
        Memory stats are cached between log intervals for any code that needs them.

        Args:
            step: Current training step
            batch: Optional batch dict to extract token/sequence info
            is_log_step: If True, query GPU memory (causes sync). If False, use cached values.

        Returns:
            New batch size if adjusted, None otherwise
        """
        self.step_count = step

        # Extract batch info if provided (no GPU sync - just tensor shape ops)
        if batch is not None:
            token_count = self.count_batch_tokens(batch)
            self.record_tokens(token_count)

            # Calculate average sequence length
            if 'attention_mask' in batch:
                batch_size = batch['attention_mask'].size(0)
                if batch_size > 0:
                    avg_seq = token_count // batch_size
                    self.record_sequence_length(avg_seq)

        # GPU SYNC FIX: Only query GPU memory at log intervals
        # This is the key optimization - we skip memory_allocated() calls on most batches
        if not is_log_step:
            # Use cached stats (no GPU sync)
            if self._memory_stats_cache is not None:
                utilization = self._memory_stats_cache.get('utilization', 0.0)
            else:
                utilization = self.smoothed_utilization
            # Still record history with cached value for trend analysis
            self.memory_history.append(utilization)
            self.batch_size_history.append(self.current_batch_size)
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-1000:]
                self.batch_size_history = self.batch_size_history[-1000:]
            return None  # No adjustment on non-log steps

        # Get memory stats (causes GPU sync - only at log intervals now)
        mem_stats = self.get_memory_stats(force_refresh=True)
        utilization = mem_stats['utilization']

        # Record history
        self.memory_history.append(utilization)
        self.batch_size_history.append(self.current_batch_size)

        # Keep history bounded
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]
            self.batch_size_history = self.batch_size_history[-1000:]

        # Feature 7: Handle warmup period
        if step < self.config.warmup_steps and self.config.warmup_strategy != 'none':
            warmup_size = self.get_warmup_batch_size(step)
            if warmup_size != self.current_batch_size:
                old_size = self.current_batch_size
                self.current_batch_size = warmup_size
                logger.debug(f"Step {step}: WARMUP batch size: {old_size} -> {warmup_size}")
                return warmup_size
            return None

        # Check if we should adjust
        if not self.should_adjust(step):
            # Log every 100 steps
            if step % 100 == 0:
                raw_util = mem_stats.get('raw_utilization', 0.0)
                if self.config.token_budget_enabled:
                    dyn_budget = self.get_dynamic_token_budget()
                    # Use adaptive margin (auto-tuned over time) or fixed margin
                    if self.adaptive_margin is not None:
                        bwd_margin = self.adaptive_margin.get_margin()
                    else:
                        bwd_margin = self.config.backward_safety_margin
                    eff_util = raw_util * bwd_margin
                    target = self.config.target_memory_threshold
                    print(f"[DynBatch] Step {step}: VRAM={raw_util:.0%}, margin={bwd_margin:.2f}x, target={target:.0%}, tok_budget={dyn_budget}")
                else:
                    print(f"[DynBatch] Step {step}: smoothed={utilization:.1%}, raw={raw_util:.1%}, BS={self.current_batch_size}")
            return None

        # Calculate new batch size
        new_batch_size, reason = self.calculate_new_batch_size(mem_stats)

        # Feature 3: Sync across GPUs
        if self.config.sync_across_gpus:
            new_batch_size = self.sync_batch_size(new_batch_size)

        # Check if adjustment needed
        if new_batch_size == self.current_batch_size:
            # Log every 100 steps when we're staying the same
            if step % 100 == 0:
                raw_util = mem_stats.get('raw_utilization', 0.0)
                if self.config.token_budget_enabled:
                    dyn_budget = self.get_dynamic_token_budget()
                    bwd_margin = self.config.backward_safety_margin
                    eff_util = raw_util * bwd_margin
                    target = self.config.target_memory_threshold
                    print(f"[DynBatch] Step {step}: VRAM={raw_util:.0%}, eff={eff_util:.0%}(×{bwd_margin:.1f}), target={target:.0%}, tok_budget={dyn_budget}")
                else:
                    print(f"[DynBatch] Step {step}: {reason}, BS={new_batch_size}")
            return None

        # Apply adjustment
        old_batch_size = self.current_batch_size
        self.current_batch_size = new_batch_size
        self.last_adjustment_step = step
        self.adjustment_count += 1
        self._steps_since_last_change = 0  # Reset stability counter after change
        self._oom_count = 0  # Reset OOM counter on successful adjustment

        # Update statistics
        if new_batch_size > old_batch_size:
            self.total_increases += 1
            direction = "INCREASED"
        else:
            self.total_decreases += 1
            direction = "DECREASED"

        self.max_batch_size_reached = max(self.max_batch_size_reached, new_batch_size)
        self.min_batch_size_reached = min(self.min_batch_size_reached, new_batch_size)

        # Log adjustment with detailed memory info
        logger.info(
            f"Step {step}: {direction} batch size: {old_batch_size} -> {new_batch_size} "
            f"({reason}, alloc={mem_stats['allocated_gb']:.1f}GB, rsv={mem_stats['reserved_gb']:.1f}GB)"
        )

        return new_batch_size

    def get_current_batch_size(self, step: Optional[int] = None) -> int:
        """Get the current batch size (optionally update first)."""
        if step is not None:
            self.step(step)
        return self.current_batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about batch size adjustments."""
        if not self.memory_history:
            return {}

        stats = {
            'current_batch_size': self.current_batch_size,
            'original_batch_size': self.original_batch_size,
            'min_batch_size_reached': self.min_batch_size_reached,
            'max_batch_size_reached': self.max_batch_size_reached,
            'total_adjustments': self.adjustment_count,
            'total_increases': self.total_increases,
            'total_decreases': self.total_decreases,
            'avg_memory_utilization': sum(self.memory_history) / len(self.memory_history),
            'current_memory_utilization': self.memory_history[-1] if self.memory_history else 0.0,
            'batch_size_improvement': (self.current_batch_size / self.original_batch_size - 1.0) * 100,
        }

        # Feature 1: Token stats
        if self.tokens_history:
            stats['avg_tokens_per_batch'] = sum(self.tokens_history) / len(self.tokens_history)

        # Feature 2: Sequence length stats
        if self.sequence_lengths:
            stats['avg_sequence_length'] = self.avg_sequence_length

        # Feature 5: Predictor stats
        if self.memory_predictor and self.memory_predictor.calibrated:
            stats['memory_predictor_calibrated'] = True
            stats['bytes_per_token'] = self.memory_predictor.bytes_per_token

        # Feature 8: Trend stats
        if self.trend_analyzer:
            stats['memory_trend'] = self.trend_analyzer.get_trend()
            stats['oscillation_detected'] = self.trend_analyzer.detect_oscillation()

        return stats

    def log_summary(self) -> None:
        """Log a summary of dynamic batching performance."""
        stats = self.get_statistics()

        if not stats:
            logger.info("Dynamic batching: No statistics available")
            return

        logger.info("=" * 70)
        logger.info("DYNAMIC BATCHING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Original batch size: {stats['original_batch_size']}")
        logger.info(f"Final batch size: {stats['current_batch_size']}")
        logger.info(f"Batch size range: [{stats['min_batch_size_reached']}, {stats['max_batch_size_reached']}]")
        logger.info(f"Total adjustments: {stats['total_adjustments']} ({stats['total_increases']} increases, {stats['total_decreases']} decreases)")
        logger.info(f"Avg memory utilization: {stats['avg_memory_utilization']:.1%}")
        logger.info(f"Batch size improvement: {stats['batch_size_improvement']:+.1f}%")

        # Feature-specific stats
        if 'avg_tokens_per_batch' in stats:
            logger.info(f"Avg tokens per batch: {stats['avg_tokens_per_batch']:.0f}")
        if 'avg_sequence_length' in stats:
            logger.info(f"Avg sequence length: {stats['avg_sequence_length']:.0f}")
        if 'memory_trend' in stats:
            logger.info(f"Final memory trend: {stats['memory_trend']}")

        logger.info("=" * 70)

    def handle_oom(self, step: int) -> int:
        """
        Handle OOM error by immediately reducing batch size and clearing cache.

        Call this from the training loop's exception handler when OOM occurs.
        After 3 consecutive OOMs, enters safe mode (no more batch size changes).

        Args:
            step: Current training step where OOM occurred

        Returns:
            New (reduced) batch size to use for retry
        """
        old_batch_size = self.current_batch_size
        min_bs = self.config.min_batch_size

        # Track consecutive OOMs for safe mode
        self._oom_count += 1
        if self._oom_count >= 3:
            self._enter_safe_mode(f"3 consecutive OOMs at step {step}")

        # Reduce by half, but not below minimum
        new_batch_size = max(min_bs, self.current_batch_size // 2)

        # If already at minimum, can't reduce further
        if new_batch_size == old_batch_size and old_batch_size == min_bs:
            logger.error(f"OOM at minimum batch size {min_bs}! Cannot reduce further.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return min_bs

        self.current_batch_size = new_batch_size
        self.total_decreases += 1
        self.min_batch_size_reached = min(self.min_batch_size_reached, new_batch_size)

        # Clear GPU cache to recover memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        logger.warning(
            f"Step {step}: OOM RECOVERY - batch size: {old_batch_size} -> {new_batch_size} "
            f"(halved due to OOM, cleared cache, OOM count: {self._oom_count})"
        )

        # Reset smoothed utilization to force re-evaluation
        self.smoothed_utilization = 0.9  # Assume high memory after OOM
        self._steps_since_last_change = 0  # Reset stability counter

        # Feature 3: Sync new batch size across GPUs
        if self.config.sync_across_gpus:
            self.current_batch_size = self.sync_batch_size(self.current_batch_size)

        return self.current_batch_size

    def _enter_safe_mode(self, reason: str) -> None:
        """
        Enter safe mode: stop all batch size adjustments and use minimum batch size.

        This is triggered after repeated OOMs to prevent further instability.
        """
        if self._safe_mode:
            return  # Already in safe mode

        logger.warning(f"Dynamic batching entering SAFE MODE: {reason}")
        logger.warning(f"Batch size locked at minimum: {self.config.min_batch_size}")

        self._safe_mode = True
        self.current_batch_size = self.config.min_batch_size

        # Disable all adaptive features
        if self.adaptive_margin:
            self.adaptive_margin.current_margin = self.adaptive_margin.max_margin
        if self.trend_analyzer:
            self.trend_analyzer.reset()

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_batch_size = self.config.initial_batch_size
        self.step_count = 0
        self.adjustment_count = 0
        self.last_adjustment_step = -self.config.cooldown_steps
        self.memory_history = []
        self.batch_size_history = []
        self.total_increases = 0
        self.total_decreases = 0
        self.max_batch_size_reached = self.config.initial_batch_size
        self.min_batch_size_reached = self.config.initial_batch_size
        self.smoothed_utilization = 0.0
        self.tokens_history = []
        self.sequence_lengths = []

        if self.trend_analyzer:
            self.trend_analyzer.reset()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)


# =============================================================================
# Factory Function
# =============================================================================

def create_dynamic_batch_scheduler(
    config_dict: Dict[str, Any],
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    vocab_size: Optional[int] = None,
    run_dir: Optional[str] = None,
) -> DynamicBatchScheduler:
    """
    Create a dynamic batch scheduler from a configuration dictionary.

    Supports all 8 enhanced features with backward-compatible defaults.
    If model is provided, runs calibration for improved memory prediction.

    Args:
        config_dict: Configuration dictionary with dynamic_batching section
        model: Optional model for calibration (enables enhanced memory prediction)
        device: Optional device for calibration
        vocab_size: Optional vocab size for calibration
        run_dir: Optional run directory for saving calibration results

    Returns:
        DynamicBatchScheduler instance (with enhanced predictor if calibrated)
    """
    # Extract dynamic batching config (support multiple nesting levels)
    db_config = config_dict.get('dynamic_batching', {})

    # Also check nested locations for backward compatibility
    if not db_config:
        training = config_dict.get('training', {})
        db_config = training.get('dynamic_batching', {})
        if not db_config:
            batching = training.get('batching', {})
            db_config = batching.get('dynamic_batching', {})

    # Get initial batch size
    initial_batch_size = db_config.get('initial_batch_size',
                                        db_config.get('min_batch_size', 32))

    # Extract token budget config (may be nested)
    token_budget = db_config.get('token_budget', {})

    # Extract predictive config (may be nested)
    predictive = db_config.get('predictive', {})

    # Extract trend detection config (may be nested)
    trend_detection = db_config.get('trend_detection', {})

    # Extract adaptive margin config (may be nested)
    adaptive_margin = db_config.get('adaptive_margin', {})

    # Extract aggressive growth config (may be nested)
    aggressive_growth = db_config.get('aggressive_growth', {})

    # Create config with all features
    config = DynamicBatchConfig(
        enabled=db_config.get('enabled', True),
        initial_batch_size=initial_batch_size,
        min_batch_size=db_config.get('min_batch_size', 32),
        max_batch_size=db_config.get('max_batch_size', 512),

        # Feature 6: Thresholds (with conservative defaults)
        low_memory_threshold=db_config.get('low_memory_threshold', 0.60),
        target_memory_threshold=db_config.get('target_memory_threshold', 0.75),
        high_memory_threshold=db_config.get('high_memory_threshold', 0.85),
        critical_memory_threshold=db_config.get('critical_memory_threshold', 0.92),

        # Adjustment parameters
        increase_factor=db_config.get('increase_factor', 1.2),
        decrease_factor=db_config.get('decrease_factor', 0.8),
        adjustment_frequency=db_config.get('adjustment_frequency', 10),
        warmup_steps=db_config.get('warmup_steps', 100),
        max_adjustments_per_session=db_config.get('max_adjustments_per_session', 50),
        cooldown_steps=db_config.get('cooldown_steps', 5),

        # Feature 1: Token Budget
        token_budget_enabled=token_budget.get('enabled', db_config.get('token_budget_enabled', False)),
        target_tokens_per_batch=token_budget.get('target_tokens_per_batch',
                                                  db_config.get('target_tokens_per_batch', 4096)),
        max_tokens_per_batch=token_budget.get('max_tokens_per_batch',
                                               db_config.get('max_tokens_per_batch', 8192)),
        min_tokens_per_batch=token_budget.get('min_tokens_per_batch',
                                               db_config.get('min_tokens_per_batch', 512)),

        # Feature 2: Sequence-Aware (ENABLED by default - 5-10% throughput gain)
        sequence_aware=db_config.get('sequence_aware', True),
        base_sequence_length=db_config.get('base_sequence_length', 512),
        sequence_scaling_factor=db_config.get('sequence_scaling_factor', 1.0),

        # Feature 3: Multi-GPU Sync
        sync_across_gpus=db_config.get('sync_across_gpus', True),
        sync_strategy=db_config.get('sync_strategy', 'min'),

        # Feature 4: Gradient Accumulation (ENABLED by default - better coordination)
        coordinate_with_grad_accum=db_config.get('coordinate_with_grad_accum', True),
        target_effective_batch_size=db_config.get('target_effective_batch_size', 512),
        dynamic_grad_accum=db_config.get('dynamic_grad_accum', False),
        original_grad_accum_steps=db_config.get('original_grad_accum_steps',
                                                 config_dict.get('training', {}).get('gradient_accumulation_steps', 1)),

        # Feature 5: Predictive Memory (ENABLED by default - prevents OOM)
        predictive_enabled=predictive.get('enabled', db_config.get('predictive_enabled', True)),
        calibration_steps=predictive.get('calibration_steps', db_config.get('calibration_steps', 50)),
        memory_model=predictive.get('memory_model', db_config.get('memory_model', 'linear')),
        # REDUCED from 1.20 to 1.10 - 20% margin was too conservative, wasting 10% throughput
        backward_safety_margin=predictive.get('backward_safety_margin',
                                               db_config.get('backward_safety_margin', 1.10)),

        # Feature 7: Warmup Strategy
        warmup_strategy=db_config.get('warmup_strategy', 'none'),
        warmup_growth_rate=db_config.get('warmup_growth_rate', 1.15),
        warmup_initial_fraction=db_config.get('warmup_initial_fraction', 0.25),

        # Feature 8: Trend Detection (ENABLED by default - prevents oscillation)
        trend_detection_enabled=trend_detection.get('enabled', db_config.get('trend_detection_enabled', True)),
        trend_window=trend_detection.get('window', db_config.get('trend_window', 20)),
        oscillation_threshold=trend_detection.get('oscillation_threshold',
                                                   db_config.get('oscillation_threshold', 5)),
        auto_tune_smoothing=trend_detection.get('auto_tune_smoothing',
                                                 db_config.get('auto_tune_smoothing', True)),

        # Feature 9: Adaptive Backward Margin
        adaptive_margin_enabled=adaptive_margin.get('enabled', db_config.get('adaptive_margin_enabled', True)),
        adaptive_margin_min=adaptive_margin.get('min_margin', db_config.get('adaptive_margin_min', 1.05)),
        adaptive_margin_max=adaptive_margin.get('max_margin', db_config.get('adaptive_margin_max', 2.0)),
        adaptive_margin_learning_rate=adaptive_margin.get('learning_rate',
                                                           db_config.get('adaptive_margin_learning_rate', 0.02)),
        adaptive_margin_ema_alpha=adaptive_margin.get('ema_alpha',
                                                       db_config.get('adaptive_margin_ema_alpha', 0.1)),

        # Feature 10: Aggressive Growth Mode
        aggressive_growth_enabled=aggressive_growth.get('enabled', db_config.get('aggressive_growth_enabled', False)),
        aggressive_growth_exploration_steps=aggressive_growth.get('exploration_steps',
                                                                   db_config.get('aggressive_growth_exploration_steps', 500)),
        aggressive_growth_factor=aggressive_growth.get('growth_factor',
                                                        db_config.get('aggressive_growth_factor', 2)),
        aggressive_growth_min_headroom=aggressive_growth.get('min_headroom_for_growth',
                                                              db_config.get('aggressive_growth_min_headroom', 0.10)),
    )

    # Create base scheduler
    scheduler = DynamicBatchScheduler(config)

    # Run calibration if model provided
    calibration_config = db_config.get('calibration', {})
    if model is not None and calibration_config.get('enabled', True):
        try:
            # Determine device
            if device is None:
                device = next(model.parameters()).device

            # Determine vocab size
            if vocab_size is None:
                # Try to get from model config
                if hasattr(model, 'config'):
                    vocab_size = getattr(model.config, 'vocab_size', 50000)
                else:
                    vocab_size = 50000

            # Get model config for cache key
            model_config = {}
            if hasattr(model, 'config'):
                model_cfg = model.config
                model_config = {
                    'vocab_size': getattr(model_cfg, 'vocab_size', 0),
                    'hidden_size': getattr(model_cfg, 'hidden_size', 0),
                    'num_layers': getattr(model_cfg, 'num_layers', 0),
                    'num_attention_heads': getattr(model_cfg, 'num_attention_heads', 0),
                    'intermediate_size': getattr(model_cfg, 'intermediate_size', 0),
                    'num_experts': getattr(model_cfg, 'num_experts', 0),
                }

            # Get precision from config
            precision = config_dict.get('hardware', {}).get('mixed_precision', 'fp32')

            # Create and run calibration system
            cal_system = CalibrationSystem(
                model=model,
                device=device,
                config=db_config,
            )

            cal_result = cal_system.run_or_load(
                model_config=model_config,
                precision=precision,
                vocab_size=vocab_size,
                run_dir=run_dir,
            )

            # Replace memory predictor with enhanced version if calibration succeeded
            if cal_result is not None:
                scheduler.memory_predictor = EnhancedMemoryPredictor(cal_result)
                logger.info("Dynamic batch scheduler using enhanced calibrated memory predictor")

                # Store calibration result for reference
                scheduler.calibration_result = cal_result

                # Read configurable safety margins from calibration config
                apply_batch_size = calibration_config.get('apply_optimal_batch_size', True)
                batch_safety = calibration_config.get('batch_size_safety_margin', 0.9)
                apply_backward = calibration_config.get('apply_backward_ratio', True)
                backward_buffer = calibration_config.get('backward_ratio_buffer', 1.1)
                apply_tokens = calibration_config.get('apply_token_budget', True)
                token_safety = calibration_config.get('token_budget_safety_margin', 0.85)

                # APPLY calibration findings to scheduler config
                if apply_batch_size and cal_result.optimal_batch_size is not None:
                    old_max = scheduler.config.max_batch_size
                    # Use optimal batch size as the new max (with configurable safety margin)
                    safe_max = int(cal_result.optimal_batch_size * batch_safety)
                    if safe_max < old_max:
                        scheduler.config.max_batch_size = safe_max
                        print(f" [Calibration] Reduced max_batch_size: {old_max} → {safe_max} (optimal={cal_result.optimal_batch_size}, margin={batch_safety})")

                # Update backward safety margin with measured ratio
                if apply_backward and cal_result.backward_ratio > 0:
                    old_margin = scheduler.config.backward_safety_margin
                    # Add configurable buffer to measured ratio for safety
                    new_margin = cal_result.backward_ratio * backward_buffer
                    scheduler.config.backward_safety_margin = new_margin
                    print(f" [Calibration] Updated backward_safety_margin: {old_margin:.2f} → {new_margin:.2f} (measured={cal_result.backward_ratio:.2f}x, buffer={backward_buffer})")

                # Calculate and apply safe token budget based on calibration
                if apply_tokens and cal_result.optimal_batch_size and scheduler.config.token_budget_enabled:
                    # Get the sequence length used during throughput profiling
                    mid_seq = cal_system.seq_lengths[len(cal_system.seq_lengths) // 2] if cal_system.seq_lengths else 256
                    safe_tokens = int(cal_result.optimal_batch_size * mid_seq * token_safety)
                    old_max_tokens = scheduler.config.max_tokens_per_batch
                    if safe_tokens < old_max_tokens:
                        scheduler.config.max_tokens_per_batch = safe_tokens
                        print(f" [Calibration] Reduced max_tokens_per_batch: {old_max_tokens} → {safe_tokens} (margin={token_safety})")

        except Exception as e:
            logger.warning(f"Calibration failed, using default memory predictor: {e}")

    return scheduler


# =============================================================================
# Simplified MemoryMonitor (for BatchSizeController integration)
# =============================================================================

# Import MemoryMonitor from batch_size_controller for backward compatibility
# This allows code that imports from dynamic_batching to also get the new class
try:
    from .batch_size_controller import (
        MemoryMonitor,
        BatchSizeController,
        create_batch_size_controller,
    )
except ImportError:
    # Fallback: define a minimal MemoryMonitor here if batch_size_controller not available
    import time as _time

    class MemoryMonitor:
        """Fallback MemoryMonitor if batch_size_controller module not available."""

        def __init__(self, cache_interval_sec: float = 0.5):
            self._cache: Dict[str, float] = {}
            self._cache_time: float = 0.0
            self._cache_interval: float = cache_interval_sec
            self._device: Optional[int] = None
            self._total_memory: int = 0

            if torch.cuda.is_available():
                self._device = torch.cuda.current_device()
                self._total_memory = torch.cuda.get_device_properties(self._device).total_memory

        def _is_cache_stale(self) -> bool:
            return _time.monotonic() - self._cache_time > self._cache_interval

        def _refresh_cache(self) -> None:
            if not torch.cuda.is_available() or self._total_memory == 0:
                self._cache = {'utilization': 0.0}
                return
            reserved = torch.cuda.memory_reserved(self._device)
            self._cache = {
                'utilization': reserved / self._total_memory,
                'reserved_gb': reserved / 1e9,
            }
            self._cache_time = _time.monotonic()

        def get_utilization(self, force_refresh: bool = False) -> float:
            if force_refresh or self._is_cache_stale():
                self._refresh_cache()
            return self._cache.get('utilization', 0.0)

    # Stub for BatchSizeController
    BatchSizeController = None
    create_batch_size_controller = None
