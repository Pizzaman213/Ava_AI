"""
Predictive OOM (Out-of-Memory) Monitoring System

This module provides proactive OOM detection and prevention by monitoring
memory growth patterns and adjusting batch size before OOM occurs.
"""

import torch
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PredictiveOOMMonitor:
    """
    Predictive OOM monitor that uses memory growth trends to prevent OOM.

    Instead of waiting for OOM to happen and then reacting, this monitor:
    1. Tracks memory usage over time
    2. Predicts future memory usage using linear regression
    3. Adjusts batch size preemptively when OOM is predicted
    4. Provides early warnings before critical memory levels

    Features:
    - Linear regression on memory growth for prediction
    - Configurable warning and critical thresholds
    - Automatic batch size adjustment recommendations
    - Memory spike detection
    - Peak memory tracking
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        history_size: int = 100,
        warning_threshold: float = 0.85,  # 85% memory usage warning
        critical_threshold: float = 0.92,  # 92% memory usage critical
        prediction_window: int = 10,  # Predict N steps ahead
        min_history_for_prediction: int = 10,
        enable_auto_adjustment: bool = True,
    ):
        """
        Initialize predictive OOM monitor.

        Args:
            device: PyTorch device to monitor (auto-detected if None)
            history_size: Number of memory measurements to keep
            warning_threshold: Memory fraction to trigger warnings (0-1)
            critical_threshold: Memory fraction to trigger critical alerts (0-1)
            prediction_window: How many steps ahead to predict
            min_history_for_prediction: Minimum history needed for predictions
            enable_auto_adjustment: Enable automatic batch size adjustments
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_size = history_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.prediction_window = prediction_window
        self.min_history_for_prediction = min_history_for_prediction
        self.enable_auto_adjustment = enable_auto_adjustment

        # Memory tracking
        self.memory_history = deque(maxlen=history_size)
        self.allocated_history = deque(maxlen=history_size)
        self.step_history = deque(maxlen=history_size)

        # Statistics
        self.peak_memory = 0.0
        self.total_memory = self._get_total_memory()
        self.warnings_issued = 0
        self.critical_alerts_issued = 0
        self.adjustments_made = 0
        self.last_adjustment_step = -1

        # State
        self.current_batch_size = None
        self.step_count = 0

        logger.info(f"PredictiveOOMMonitor initialized on {self.device}")
        logger.info(f"  Total memory: {self.total_memory / (1024**3):.2f} GB")
        logger.info(f"  Warning threshold: {self.warning_threshold * 100}%")
        logger.info(f"  Critical threshold: {self.critical_threshold * 100}%")

    def _get_total_memory(self) -> float:
        """Get total memory available on device in bytes."""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(self.device).total_memory
        else:
            # For CPU, use a reasonable default (16GB)
            return 16 * 1024**3

    def _get_current_memory(self) -> Tuple[float, float]:
        """
        Get current memory usage in bytes.

        Returns:
            Tuple of (allocated_memory, reserved_memory)
        """
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            return allocated, reserved
        else:
            # For CPU, return 0 (CPU memory is handled differently)
            return 0.0, 0.0

    def update(self, step: int, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Update memory statistics and check for potential OOM.

        Args:
            step: Current training step
            batch_size: Current batch size (for adjustment recommendations)

        Returns:
            Dictionary with monitoring results and recommendations
        """
        self.step_count = step
        if batch_size is not None:
            self.current_batch_size = batch_size

        # Get current memory usage
        allocated, reserved = self._get_current_memory()
        current_memory = reserved if reserved > 0 else allocated
        memory_fraction = current_memory / self.total_memory if self.total_memory > 0 else 0

        # Update history
        self.memory_history.append(memory_fraction)
        self.allocated_history.append(allocated)
        self.step_history.append(step)

        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

        # Initialize result
        result = {
            'status': 'ok',
            'current_memory_gb': current_memory / (1024**3),
            'memory_fraction': memory_fraction,
            'peak_memory_gb': self.peak_memory / (1024**3),
            'predicted_memory_fraction': None,
            'predicted_oom': False,
            'should_reduce_batch': False,
            'recommended_batch_size': self.current_batch_size,
            'warning_message': None,
        }

        # Check if we have enough history for prediction
        if len(self.memory_history) < self.min_history_for_prediction:
            return result

        # Predict future memory usage
        predicted_fraction = self._predict_memory()
        result['predicted_memory_fraction'] = predicted_fraction

        # Check for predicted OOM
        if predicted_fraction is not None:
            if predicted_fraction > self.critical_threshold:
                result['status'] = 'critical'
                result['predicted_oom'] = True
                self.critical_alerts_issued += 1

                # Calculate recommended batch size reduction
                if self.enable_auto_adjustment and self.current_batch_size:
                    reduction_factor = self._calculate_reduction_factor(
                        current_memory_fraction=memory_fraction,
                        predicted_memory_fraction=predicted_fraction
                    )
                    new_batch_size = max(1, int(self.current_batch_size * reduction_factor))
                    result['should_reduce_batch'] = True
                    result['recommended_batch_size'] = new_batch_size
                    result['warning_message'] = (
                        f"CRITICAL: OOM predicted in {self.prediction_window} steps "
                        f"(predicted: {predicted_fraction*100:.1f}% memory). "
                        f"Recommend reducing batch size: {self.current_batch_size} → {new_batch_size}"
                    )

                    self.adjustments_made += 1
                    self.last_adjustment_step = step
                else:
                    result['warning_message'] = (
                        f"CRITICAL: OOM predicted in {self.prediction_window} steps "
                        f"(predicted: {predicted_fraction*100:.1f}% memory)"
                    )

            elif predicted_fraction > self.warning_threshold:
                result['status'] = 'warning'
                self.warnings_issued += 1
                result['warning_message'] = (
                    f"WARNING: High memory usage predicted "
                    f"(current: {memory_fraction*100:.1f}%, "
                    f"predicted: {predicted_fraction*100:.1f}%)"
                )

        # Check for memory spikes
        if len(self.memory_history) >= 5:
            recent_avg = np.mean(list(self.memory_history)[-5:-1])
            if memory_fraction > recent_avg * 1.3:  # 30% spike
                result['warning_message'] = (
                    f"WARNING: Memory spike detected "
                    f"({recent_avg*100:.1f}% → {memory_fraction*100:.1f}%)"
                )

        return result

    def _predict_memory(self) -> Optional[float]:
        """
        Predict future memory usage using linear regression.

        Returns:
            Predicted memory fraction, or None if prediction not possible
        """
        if len(self.memory_history) < self.min_history_for_prediction:
            return None

        # Convert deque to numpy arrays
        steps = np.array(list(self.step_history))
        memory = np.array(list(self.memory_history))

        # Perform linear regression
        try:
            # Fit linear model: y = mx + b
            coeffs = np.polyfit(steps, memory, 1)
            slope, intercept = coeffs[0], coeffs[1]

            # Predict future memory
            future_step = self.step_count + self.prediction_window
            predicted_memory = slope * future_step + intercept

            # Clamp to valid range [0, 1]
            return max(0.0, min(1.0, predicted_memory))
        except Exception as e:
            logger.warning(f"Memory prediction failed: {e}")
            return None

    def _calculate_reduction_factor(
        self,
        current_memory_fraction: float,
        predicted_memory_fraction: float
    ) -> float:
        """
        Calculate batch size reduction factor based on memory prediction.

        Args:
            current_memory_fraction: Current memory usage (0-1)
            predicted_memory_fraction: Predicted memory usage (0-1)

        Returns:
            Reduction factor (e.g., 0.8 = reduce to 80% of current batch size)
        """
        # How far over threshold are we predicted to go?
        overage = predicted_memory_fraction - self.critical_threshold

        # Conservative reduction: aim for 10% below warning threshold
        target_fraction = self.warning_threshold * 0.9

        if predicted_memory_fraction > 0:
            reduction = target_fraction / predicted_memory_fraction
            # Clamp reduction to reasonable range (50-90%)
            return max(0.5, min(0.9, reduction))

        return 0.8  # Default 20% reduction

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if len(self.memory_history) == 0:
            return {
                'peak_memory_gb': 0,
                'avg_memory_fraction': 0,
                'warnings_issued': 0,
                'critical_alerts_issued': 0,
                'adjustments_made': 0,
            }

        return {
            'peak_memory_gb': self.peak_memory / (1024**3),
            'current_memory_gb': (self.allocated_history[-1] if self.allocated_history else 0) / (1024**3),
            'avg_memory_fraction': np.mean(list(self.memory_history)),
            'max_memory_fraction': np.max(list(self.memory_history)),
            'min_memory_fraction': np.min(list(self.memory_history)),
            'warnings_issued': self.warnings_issued,
            'critical_alerts_issued': self.critical_alerts_issued,
            'adjustments_made': self.adjustments_made,
            'last_adjustment_step': self.last_adjustment_step,
            'samples_collected': len(self.memory_history),
        }

    def reset(self):
        """Reset monitoring history and statistics."""
        self.memory_history.clear()
        self.allocated_history.clear()
        self.step_history.clear()
        self.peak_memory = 0.0
        self.warnings_issued = 0
        self.critical_alerts_issued = 0
        self.adjustments_made = 0
        self.last_adjustment_step = -1
        logger.info("PredictiveOOMMonitor reset")

    def should_check_this_step(self, step: int, check_frequency: int = 10) -> bool:
        """
        Determine if we should check memory on this step.

        For performance, we can check less frequently after warmup.

        Args:
            step: Current step
            check_frequency: Check every N steps

        Returns:
            True if should check this step
        """
        # Always check first 100 steps
        if step < 100:
            return True

        # After warmup, check every N steps
        return step % check_frequency == 0
