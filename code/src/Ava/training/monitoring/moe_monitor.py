"""
MoE Expert Load Balance Monitor

Monitors expert utilization and provides suggestions for load_balance_loss_coef adjustments.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging


@dataclass
class ExpertLoadStats:
    """Statistics for expert load balancing."""
    expert_usage: torch.Tensor  # Usage count per expert
    total_tokens: int
    balance_score: float  # 0.0 (perfectly imbalanced) to 1.0 (perfectly balanced)
    coefficient_of_variation: float  # CV of expert usage distribution
    max_min_ratio: float  # Ratio of max to min usage
    suggested_load_balance_coef: Optional[float] = None
    suggestion_reason: Optional[str] = None


class MoELoadBalanceMonitor:
    """
    Monitors MoE expert load balance and suggests hyperparameter adjustments.

    Tracks expert utilization across training and provides actionable suggestions
    when load imbalance is detected.

    Args:
        num_experts: Number of experts in the MoE layer
        current_load_balance_coef: Current load balance loss coefficient
        check_interval: How often to check balance (in steps)
        target_balance_score: Target balance score (0.6-0.9 is typical)
        logger: Optional logger for warnings/suggestions
    """

    def __init__(
        self,
        num_experts: int,
        current_load_balance_coef: float = 0.01,
        check_interval: int = 100,
        target_balance_score: float = 0.7,
        logger: Optional[logging.Logger] = None,
    ):
        self.num_experts = num_experts
        self.current_load_balance_coef = current_load_balance_coef
        self.check_interval = check_interval
        self.target_balance_score = target_balance_score
        self.logger = logger or logging.getLogger(__name__)

        # History tracking
        self.balance_history: List[float] = []
        self.coef_history: List[float] = [current_load_balance_coef]
        self.step_count = 0

        # Expert usage accumulator
        self.cumulative_expert_usage = torch.zeros(num_experts)
        self.cumulative_tokens = 0

    def update(
        self,
        expert_indices: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
        moe_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExpertLoadStats]:
        """
        Update expert usage statistics.

        Args:
            expert_indices: Expert indices selected [num_tokens, k]
            expert_weights: Expert weights [num_tokens, k]
            moe_metrics: Dictionary of MoE metrics from model forward pass

        Returns:
            ExpertLoadStats if it's time to check balance, None otherwise
        """
        self.step_count += 1

        # Extract expert usage from different sources
        if expert_indices is not None:
            # Count expert usage from indices
            num_tokens = expert_indices.shape[0]
            for expert_id in range(self.num_experts):
                count = (expert_indices == expert_id).sum().item()
                self.cumulative_expert_usage[expert_id] += count
            self.cumulative_tokens += num_tokens

        elif moe_metrics is not None and 'expert_usage' in moe_metrics:
            # Use pre-computed expert usage from metrics
            usage = moe_metrics['expert_usage']
            if isinstance(usage, torch.Tensor):
                self.cumulative_expert_usage += usage.cpu()
            self.cumulative_tokens += moe_metrics.get('num_tokens', 0)

        # Check balance at specified interval
        if self.step_count % self.check_interval == 0 and self.cumulative_tokens > 0:
            stats = self._compute_balance_stats()
            self.balance_history.append(stats.balance_score)

            # Check if adjustment is needed
            if stats.balance_score < self.target_balance_score:
                stats.suggested_load_balance_coef = self._suggest_coefficient(stats)
                stats.suggestion_reason = self._get_suggestion_reason(stats)

                # Log warning
                self.logger.warning(
                    f"\n  Expert Load Imbalance Detected (Step {self.step_count}):\n"
                    f"   Balance Score: {stats.balance_score:.3f} (target: {self.target_balance_score:.3f})\n"
                    f"   Coefficient of Variation: {stats.coefficient_of_variation:.3f}\n"
                    f"   Max/Min Ratio: {stats.max_min_ratio:.2f}x\n"
                    f"   {stats.suggestion_reason}\n"
                    f"    Suggestion: Increase load_balance_loss_coef from "
                    f"{self.current_load_balance_coef:.4f} to {stats.suggested_load_balance_coef:.4f}"
                )

            # Reset accumulators
            self.cumulative_expert_usage.zero_()
            self.cumulative_tokens = 0

            return stats

        return None

    def _compute_balance_stats(self) -> ExpertLoadStats:
        """Compute expert load balance statistics."""
        usage = self.cumulative_expert_usage.numpy()
        total_tokens = self.cumulative_tokens

        # Normalize to get distribution
        if total_tokens > 0:
            distribution = usage / total_tokens
        else:
            distribution = np.ones(self.num_experts) / self.num_experts

        # Compute balance score (entropy-based)
        # Perfect balance = log(num_experts), worst = 0
        # Normalize to [0, 1]
        epsilon = 1e-10
        entropy = -np.sum(distribution * np.log(distribution + epsilon))
        max_entropy = np.log(self.num_experts)
        balance_score = entropy / max_entropy if max_entropy > 0 else 1.0

        # Coefficient of variation (std / mean)
        mean_usage = np.mean(usage)
        std_usage = np.std(usage)
        cv = std_usage / (mean_usage + epsilon)

        # Max/min ratio
        max_usage = np.max(usage)
        min_usage = np.min(usage)
        max_min_ratio = max_usage / (min_usage + epsilon)

        return ExpertLoadStats(
            expert_usage=self.cumulative_expert_usage.clone(),
            total_tokens=total_tokens,
            balance_score=balance_score,
            coefficient_of_variation=float(cv),
            max_min_ratio=max_min_ratio,
        )

    def _suggest_coefficient(self, stats: ExpertLoadStats) -> float:
        """Suggest new load_balance_loss_coef based on imbalance severity."""
        # Compute imbalance severity
        balance_deficit = self.target_balance_score - stats.balance_score

        # Adjustment strategy:
        # - Mild imbalance (0.05-0.15 deficit): Increase by 25%
        # - Moderate imbalance (0.15-0.30 deficit): Increase by 50%
        # - Severe imbalance (>0.30 deficit): Double the coefficient

        if balance_deficit < 0.15:
            multiplier = 1.25
        elif balance_deficit < 0.30:
            multiplier = 1.5
        else:
            multiplier = 2.0

        # Cap at reasonable maximum (0.1)
        suggested = min(self.current_load_balance_coef * multiplier, 0.1)

        return round(suggested, 5)

    def _get_suggestion_reason(self, stats: ExpertLoadStats) -> str:
        """Get human-readable reason for suggestion."""
        if stats.max_min_ratio > 10:
            return (
                f" SEVERE: Some experts are used {stats.max_min_ratio:.1f}x more than others. "
                "This indicates expert collapse."
            )
        elif stats.coefficient_of_variation > 1.0:
            return (
                f" MODERATE: High variation in expert usage (CV={stats.coefficient_of_variation:.2f}). "
                "Load balancing needs improvement."
            )
        else:
            return (
                f" MILD: Balance score below target but not critical. "
                "Small adjustment recommended."
            )

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return {
            'step_count': self.step_count,
            'current_load_balance_coef': self.current_load_balance_coef,
            'target_balance_score': self.target_balance_score,
            'balance_history': self.balance_history[-10:],  # Last 10 checks
            'num_checks': len(self.balance_history),
            'avg_balance_score': np.mean(self.balance_history) if self.balance_history else 0.0,
        }

    def update_coefficient(self, new_coef: float) -> None:
        """Update the current load balance coefficient."""
        self.current_load_balance_coef = new_coef
        self.coef_history.append(new_coef)
        self.logger.info(f" Updated load_balance_loss_coef to {new_coef:.5f}")
