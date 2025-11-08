#!/usr/bin/env python3
"""
Example: MoE Load Balance Monitoring

This script demonstrates how to use the MoELoadBalanceMonitor
to track expert utilization and get automatic suggestions.

Usage:
    python monitor_moe_balance.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.Ava.training.monitoring import MoELoadBalanceMonitor


def simulate_expert_routing(num_experts: int, num_tokens: int, k: int, imbalance_factor: float = 1.0):
    """
    Simulate expert routing for demonstration.

    Args:
        num_experts: Number of experts
        num_tokens: Number of tokens to route
        k: Number of experts per token
        imbalance_factor: 1.0 = balanced, >1.0 = imbalanced

    Returns:
        expert_indices: [num_tokens, k]
        expert_weights: [num_tokens, k]
    """
    # Create biased distribution if imbalance_factor > 1
    if imbalance_factor > 1.0:
        # Some experts are much more likely to be selected
        probs = np.array([1.0 / (i + 1) ** imbalance_factor for i in range(num_experts)])
        probs = probs / probs.sum()
    else:
        # Uniform distribution
        probs = np.ones(num_experts) / num_experts

    # Sample experts for each token
    expert_indices = np.random.choice(num_experts, size=(num_tokens, k), p=probs)

    # Random weights (normalized)
    expert_weights = np.random.rand(num_tokens, k)
    expert_weights = expert_weights / expert_weights.sum(axis=1, keepdims=True)

    return torch.tensor(expert_indices), torch.tensor(expert_weights)


def main():
    print("="*80)
    print("MoE Load Balance Monitor - Example")
    print("="*80)

    # Configuration
    num_experts = 16
    num_tokens_per_batch = 1024
    k = 2  # experts per token
    check_interval = 10  # Check every 10 steps

    print(f"\nConfiguration:")
    print(f"  Experts: {num_experts}")
    print(f"  Tokens per batch: {num_tokens_per_batch}")
    print(f"  Experts per token: {k}")
    print(f"  Check interval: {check_interval} steps\n")

    # Initialize monitor
    monitor = MoELoadBalanceMonitor(
        num_experts=num_experts,
        current_load_balance_coef=0.01,
        check_interval=check_interval,
        target_balance_score=0.7,
    )

    print("="*80)
    print("Scenario 1: Balanced Expert Usage")
    print("="*80)

    # Simulate balanced routing
    for step in range(30):
        expert_indices, expert_weights = simulate_expert_routing(
            num_experts, num_tokens_per_batch, k, imbalance_factor=1.0
        )

        stats = monitor.update(
            expert_indices=expert_indices,
            expert_weights=expert_weights
        )

        if stats:
            print(f"\n✅ Step {step}: Balance Score = {stats.balance_score:.3f} (GOOD)")
            print(f"   Max/Min Ratio: {stats.max_min_ratio:.2f}x")
            print(f"   Coefficient of Variation: {stats.coefficient_of_variation:.3f}")

    print("\n" + "="*80)
    print("Scenario 2: Imbalanced Expert Usage (Expert Collapse)")
    print("="*80)

    # Reset monitor
    monitor = MoELoadBalanceMonitor(
        num_experts=num_experts,
        current_load_balance_coef=0.01,
        check_interval=check_interval,
        target_balance_score=0.7,
    )

    # Simulate imbalanced routing (expert collapse)
    for step in range(30):
        expert_indices, expert_weights = simulate_expert_routing(
            num_experts, num_tokens_per_batch, k, imbalance_factor=2.0  # Strong imbalance
        )

        stats = monitor.update(
            expert_indices=expert_indices,
            expert_weights=expert_weights
        )

        if stats:
            print(f"\n⚠️  Step {step}: Balance Score = {stats.balance_score:.3f} (POOR)")
            print(f"   Max/Min Ratio: {stats.max_min_ratio:.2f}x")
            print(f"   Coefficient of Variation: {stats.coefficient_of_variation:.3f}")

            if stats.suggested_load_balance_coef:
                print(f"\n   💡 SUGGESTION:")
                print(f"   {stats.suggestion_reason}")
                print(f"   Increase load_balance_loss_coef: "
                      f"{monitor.current_load_balance_coef:.4f} → {stats.suggested_load_balance_coef:.4f}")

    print("\n" + "="*80)
    print("Scenario 3: Progressive Improvement After Adjustment")
    print("="*80)

    # Apply suggested coefficient
    monitor.update_coefficient(0.03)  # Increase from 0.01 to 0.03

    # Simulate improved routing after coefficient increase
    for step in range(30):
        # Gradually improve balance
        imbalance = max(1.0, 2.0 - step * 0.05)  # Linearly improve
        expert_indices, expert_weights = simulate_expert_routing(
            num_experts, num_tokens_per_batch, k, imbalance_factor=imbalance
        )

        stats = monitor.update(
            expert_indices=expert_indices,
            expert_weights=expert_weights
        )

        if stats:
            if stats.balance_score >= 0.7:
                emoji = "✅"
                status = "GOOD"
            elif stats.balance_score >= 0.5:
                emoji = "🟡"
                status = "FAIR"
            else:
                emoji = "🔴"
                status = "POOR"

            print(f"\n{emoji} Step {step}: Balance Score = {stats.balance_score:.3f} ({status})")
            print(f"   Max/Min Ratio: {stats.max_min_ratio:.2f}x")
            print(f"   Imbalance Factor: {imbalance:.2f}")

    # Final statistics
    print("\n" + "="*80)
    print("Final Statistics")
    print("="*80)

    final_stats = monitor.get_current_stats()
    print(f"\nTotal steps monitored: {final_stats['step_count']}")
    print(f"Number of balance checks: {final_stats['num_checks']}")
    print(f"Average balance score: {final_stats['avg_balance_score']:.3f}")
    print(f"Current load_balance_coef: {final_stats['current_load_balance_coef']:.5f}")
    print(f"Target balance score: {final_stats['target_balance_score']:.3f}")

    print("\nBalance history (last 10 checks):")
    for i, score in enumerate(final_stats['balance_history']):
        print(f"  Check {i+1}: {score:.3f}")

    print("\n" + "="*80)
    print("Example complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Monitor tracks balance score and provides suggestions")
    print("  2. Balance score < 0.7 triggers warnings")
    print("  3. Suggestions are graded by severity (mild/moderate/severe)")
    print("  4. Increasing load_balance_loss_coef helps improve balance")
    print("  5. Integration is simple - just call update() in training loop")


if __name__ == '__main__':
    main()
