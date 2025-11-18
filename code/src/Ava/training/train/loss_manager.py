"""
LossComputationManager - Handles loss calculation and gradient operations.

Responsibilities:
- Loss function composition and weighting
- Forward pass with loss computation
- Gradient surgery (clipping, scaling)
- Gradient health monitoring
- Numerical stability checks
- NaN detection and handling
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import TrainingContext, ManagerInterface

logger = logging.getLogger(__name__)


class LossComputationManager(ManagerInterface):
    """
    Manages all aspects of loss computation and gradient operations.

    Handles multiple loss functions, gradient surgery, health monitoring,
    and numerical stability.
    """

    def __init__(self, context: TrainingContext):
        """Initialize loss computation manager."""
        super().__init__(context)
        self.loss_functions: Dict[str, nn.Module] = {}
        self.loss_weights: Dict[str, float] = {}
        self.gradient_scaler: Optional[Any] = None
        self.gradient_clip_value: float = 1.0
        self.max_consecutive_nan_losses = 5
        self.max_total_nan_losses = 20

        # NaN tracking
        self.consecutive_nan_losses = 0
        self.total_nan_losses = 0

        # Loss statistics
        self.loss_history: List[float] = []
        self.max_loss_history = 100

    def initialize(self) -> None:
        """Initialize loss computation manager."""
        # Setup gradient scaling if using fp16
        self._init_gradient_scaler()

        # Load config settings
        if self.config:
            self.gradient_clip_value = getattr(
                self.config.training, "gradient_clip_value", 1.0
            )
            self.max_consecutive_nan_losses = getattr(
                self.config.training, "max_consecutive_nan_losses", 5
            )
            self.max_total_nan_losses = getattr(
                self.config.training, "max_total_nan_losses", 20
            )

        logger.info(
            f"Loss manager initialized: "
            f"clip_value={self.gradient_clip_value}, "
            f"max_consecutive_nan={self.max_consecutive_nan_losses}, "
            f"max_total_nan={self.max_total_nan_losses}"
        )

        super().initialize()

    def cleanup(self) -> None:
        """Cleanup loss computation resources."""
        if self.gradient_scaler:
            # No explicit cleanup needed for GradScaler
            pass

    def register_loss_function(
        self, name: str, loss_fn: nn.Module, weight: float = 1.0
    ) -> None:
        """
        Register a loss function.

        Args:
            name: Name of loss function
            loss_fn: Loss function module
            weight: Weight in composite loss
        """
        self.loss_functions[name] = loss_fn
        self.loss_weights[name] = weight
        logger.info(f"Registered loss function '{name}' with weight {weight}")

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss from multiple loss functions.

        Args:
            outputs: Model outputs (logits, etc.)
            targets: Target labels

        Returns:
            (total_loss, loss_breakdown_dict)

        Raises:
            RuntimeError: If loss becomes NaN too many times
        """
        if not self.loss_functions:
            raise ValueError("No loss functions registered")

        loss_breakdown: Dict[str, float] = {}
        total_loss = None

        # Compute weighted sum of loss functions
        for name, loss_fn in self.loss_functions.items():
            try:
                loss_value = loss_fn(outputs, targets)
                weight = self.loss_weights.get(name, 1.0)
                weighted_loss = loss_value * weight

                loss_breakdown[name] = loss_value.item()

                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

            except Exception as e:
                logger.error(f"Error computing loss '{name}': {e}")
                raise

        if total_loss is None:
            raise RuntimeError("No valid losses computed")

        # Check for NaN
        if torch.isnan(total_loss):
            self._handle_nan_loss(loss_breakdown)
            raise RuntimeError(
                f"Loss is NaN. Loss breakdown: {loss_breakdown}. "
                f"Consecutive NaN: {self.consecutive_nan_losses}/{self.max_consecutive_nan_losses}"
            )

        # Reset NaN counter on valid loss
        self.consecutive_nan_losses = 0
        loss_breakdown["total"] = total_loss.item()

        # Track loss history
        self.loss_history.append(total_loss.item())
        if len(self.loss_history) > self.max_loss_history:
            self.loss_history.pop(0)

        return total_loss, loss_breakdown

    def backward(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass with optional gradient scaling.

        Args:
            loss: Loss tensor to backpropagate
        """
        if self.gradient_scaler:
            self.gradient_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_gradients(self, parameters: List[nn.Parameter]) -> float:
        """
        Clip gradients and return clipping norm.

        Args:
            parameters: Parameters to clip gradients for

        Returns:
            Gradient norm before clipping
        """
        if self.gradient_scaler:
            self.gradient_scaler.unscale_(
                self.context.optimizer
            ) if self.context.optimizer else None

        # Compute gradient norm
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        # Clip gradients
        if self.gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, self.gradient_clip_value
            )

        logger.debug(f"Gradient norm: {total_norm:.4f}")
        return total_norm

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient scaling."""
        if self.context.optimizer is None:
            raise RuntimeError("Optimizer not set in context")

        if self.gradient_scaler:
            self.gradient_scaler.step(self.context.optimizer)
            self.gradient_scaler.update()
        else:
            self.context.optimizer.step()

        self.context.optimizer.zero_grad()

    def _init_gradient_scaler(self) -> None:
        """Initialize gradient scaler for mixed precision."""
        try:
            precision = getattr(
                self.config.deepspeed if self.config else None,
                "precision_type",
                "fp32",
            )

            if precision == "fp16" and torch.cuda.is_available():
                from torch.amp import GradScaler

                self.gradient_scaler = GradScaler(
                    init_scale=2.0 ** 16,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                logger.info("Gradient scaler initialized for fp16")

        except Exception as e:
            logger.warning(f"Failed to initialize gradient scaler: {e}")

    def _handle_nan_loss(self, loss_breakdown: Dict[str, float]) -> None:
        """Handle NaN loss detection."""
        self.consecutive_nan_losses += 1
        self.total_nan_losses += 1

        logger.error(
            f"NaN loss detected! "
            f"Breakdown: {loss_breakdown} "
            f"Consecutive: {self.consecutive_nan_losses}/{self.max_consecutive_nan_losses} "
            f"Total: {self.total_nan_losses}/{self.max_total_nan_losses}"
        )

        # Fail fast if too many NaN losses
        if self.consecutive_nan_losses >= self.max_consecutive_nan_losses:
            raise RuntimeError(
                f"Too many consecutive NaN losses: {self.consecutive_nan_losses}"
            )

        if self.total_nan_losses >= self.max_total_nan_losses:
            raise RuntimeError(
                f"Too many total NaN losses: {self.total_nan_losses}"
            )

    def get_average_loss(self, window_size: Optional[int] = None) -> float:
        """
        Get average loss over recent steps.

        Args:
            window_size: Number of recent losses to average (default: all)

        Returns:
            Average loss
        """
        if not self.loss_history:
            return float("inf")

        if window_size:
            window = self.loss_history[-window_size:]
        else:
            window = self.loss_history

        return sum(window) / len(window) if window else float("inf")

    def on_step_end(self, step: int, loss: float) -> None:
        """Update loss context."""
        self.context.current_loss = loss

    def get_status(self) -> Dict[str, Any]:
        """Return loss manager status."""
        return {
            "loss_functions": list(self.loss_functions.keys()),
            "loss_weights": self.loss_weights,
            "current_loss": self.context.current_loss,
            "average_loss": self.get_average_loss(window_size=10),
            "consecutive_nan": self.consecutive_nan_losses,
            "total_nan": self.total_nan_losses,
            "gradient_clip_value": self.gradient_clip_value,
        }
