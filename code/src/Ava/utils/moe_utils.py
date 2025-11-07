"""
Utilities for MoE training, analysis, and monitoring.

This module provides helper functions for:
- Expert utilization analysis
- Routing entropy computation
- Expert specialization measurement
- Memory estimation
- Throughput benchmarking
- Checkpoint management
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def calculate_expert_utilization(
    expert_counts: torch.Tensor,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Calculate expert utilization statistics.

    Args:
        expert_counts: Token counts per expert [num_experts]
        normalize: Whether to normalize by total count

    Returns:
        Dictionary with utilization stats
    """
    num_experts = expert_counts.shape[0]
    total_tokens = expert_counts.sum().item()

    if total_tokens == 0:
        return {
            'utilization': expert_counts,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'balance_score': 0.0,
        }

    if normalize:
        utilization = expert_counts / total_tokens
    else:
        utilization = expert_counts

    # Statistics
    mean_util = utilization.mean().item()
    std_util = utilization.std().item()
    min_util = utilization.min().item()
    max_util = utilization.max().item()

    # Balance score: 1.0 = perfect balance, 0.0 = completely imbalanced
    ideal_util = 1.0 / num_experts if normalize else total_tokens / num_experts
    balance_score = 1.0 - (utilization - ideal_util).abs().sum().item() / 2.0

    return {
        'utilization': utilization,
        'mean': mean_util,
        'std': std_util,
        'min': min_util,
        'max': max_util,
        'balance_score': balance_score,
        'num_unused_experts': (expert_counts == 0).sum().item(),
    }


def compute_routing_entropy(router_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute routing entropy (measure of diversity).

    Higher entropy = more uniform routing = better load balancing

    Args:
        router_probs: Router probabilities [num_tokens, num_experts]

    Returns:
        Scalar entropy value
    """
    # Entropy: -sum(p * log(p))
    entropy = -(router_probs * (router_probs + 1e-10).log()).sum(dim=-1).mean()
    return entropy


def analyze_expert_specialization(
    expert_outputs: Dict[int, torch.Tensor],
    method: str = 'activation_similarity'
) -> Dict[str, float]:
    """
    Analyze how specialized each expert is.

    High specialization = experts learn different functions
    Low specialization = experts are redundant

    Args:
        expert_outputs: Dictionary mapping expert_id to output tensors
        method: Analysis method ('activation_similarity', 'weight_similarity')

    Returns:
        Dictionary with specialization metrics
    """
    if len(expert_outputs) < 2:
        return {'specialization_score': 1.0}

    if method == 'activation_similarity':
        # Compute pairwise cosine similarity of expert outputs
        expert_ids = list(expert_outputs.keys())
        num_experts = len(expert_ids)
        similarities = []

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                out_i = expert_outputs[expert_ids[i]].flatten()
                out_j = expert_outputs[expert_ids[j]].flatten()

                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    out_i.unsqueeze(0),
                    out_j.unsqueeze(0)
                ).item()
                similarities.append(abs(sim))

        avg_similarity = np.mean(similarities)
        specialization_score = 1.0 - avg_similarity  # Lower similarity = higher specialization

        return {
            'specialization_score': specialization_score,
            'avg_similarity': avg_similarity,
            'pairwise_similarities': similarities,
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_moe_memory(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_experts_per_token: int = 2,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Estimate memory footprint of MoE model.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        num_experts: Number of experts per layer
        num_layers: Number of transformer layers
        batch_size: Batch size
        seq_len: Sequence length
        num_experts_per_token: Active experts per token
        dtype: Parameter dtype

    Returns:
        Dictionary with memory estimates (in GB)
    """
    # Bytes per parameter
    bytes_per_param = 4 if dtype == torch.float32 else 2  # fp32 or fp16/bf16

    # Parameters per expert (2 linear layers with gated activation)
    params_per_expert = (
        hidden_size * intermediate_size * 2 +  # gate_up_proj
        intermediate_size * hidden_size  # down_proj
    )

    # Total expert parameters
    expert_params = params_per_expert * num_experts * num_layers

    # Non-expert parameters (attention, embeddings, etc.)
    # Rough estimate: ~0.3x of expert parameters for similar model size
    non_expert_params = expert_params * 0.3

    total_params = expert_params + non_expert_params

    # Memory for parameters
    param_memory = total_params * bytes_per_param / 1e9  # GB

    # Activation memory (forward + backward)
    num_tokens = batch_size * seq_len
    activation_per_token = (
        hidden_size +  # Input
        hidden_size * num_experts_per_token +  # Router outputs
        intermediate_size * num_experts_per_token  # Expert activations
    )
    activation_memory = activation_per_token * num_tokens * num_layers * bytes_per_param * 2 / 1e9  # GB

    # Optimizer state (Adam: 2x parameters for first and second moments)
    optimizer_memory = param_memory * 2

    # Total memory
    total_memory = param_memory + activation_memory + optimizer_memory

    return {
        'param_memory_gb': param_memory,
        'activation_memory_gb': activation_memory,
        'optimizer_memory_gb': optimizer_memory,
        'total_memory_gb': total_memory,
        'total_params_billions': total_params / 1e9,
    }


def benchmark_moe_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Benchmark MoE model throughput.

    Args:
        model: MoE model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        num_iterations: Number of iterations to average
        warmup_iterations: Warmup iterations
        device: Device to run on

    Returns:
        Dictionary with throughput metrics
    """
    model = model.to(device)
    model.eval()

    # Create dummy input
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_ids)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    import time
    times = []

    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_ids)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    tokens_per_sec = batch_size * seq_len / avg_time

    return {
        'avg_time_sec': avg_time,
        'std_time_sec': std_time,
        'tokens_per_sec': tokens_per_sec,
        'samples_per_sec': batch_size / avg_time,
    }


def load_balancing_metrics(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> Dict[str, float]:
    """
    Comprehensive load balancing metrics.

    Args:
        router_probs: Router probabilities [num_tokens, num_experts]
        expert_indices: Selected experts [num_tokens, k]
        num_experts: Total number of experts

    Returns:
        Dictionary of metrics
    """
    num_tokens = router_probs.shape[0]
    k = expert_indices.shape[1]

    # Expert token counts
    expert_counts = torch.zeros(num_experts, device=router_probs.device)
    for i in range(num_experts):
        expert_counts[i] = (expert_indices == i).sum()

    # Probability fractions
    prob_fractions = router_probs.sum(dim=0) / num_tokens

    # Token fractions
    token_fractions = expert_counts / (num_tokens * k)

    # Gini coefficient (inequality measure)
    # 0 = perfect equality, 1 = perfect inequality
    sorted_counts, _ = torch.sort(expert_counts)
    n = num_experts
    index = torch.arange(1, n + 1, device=expert_counts.device).float()
    gini = ((2 * index - n - 1) * sorted_counts).sum() / (n * expert_counts.sum() + 1e-10)
    gini = gini.item()

    # Coefficient of variation
    cv = (expert_counts.std() / (expert_counts.mean() + 1e-10)).item()

    return {
        'gini_coefficient': gini,
        'coefficient_of_variation': cv,
        'min_expert_usage': expert_counts.min().item(),
        'max_expert_usage': expert_counts.max().item(),
        'prob_fraction_std': prob_fractions.std().item(),
        'token_fraction_std': token_fractions.std().item(),
    }


def visualize_expert_routing(
    expert_indices: torch.Tensor,
    num_experts: int,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create expert routing visualization matrix.

    Args:
        expert_indices: Expert assignments [num_tokens, k]
        num_experts: Total number of experts
        save_path: Optional path to save visualization

    Returns:
        Routing matrix [num_tokens, num_experts]
    """
    num_tokens, k = expert_indices.shape

    # Create routing matrix
    routing_matrix = torch.zeros(num_tokens, num_experts)
    for token_idx in range(num_tokens):
        for expert_idx in expert_indices[token_idx]:
            routing_matrix[token_idx, expert_idx] = 1

    routing_matrix_np = routing_matrix.cpu().numpy()

    if save_path:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(routing_matrix_np, aspect='auto', cmap='viridis')
            plt.xlabel('Expert ID')
            plt.ylabel('Token ID')
            plt.title('Expert Routing Pattern')
            plt.colorbar(label='Selected')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("matplotlib not available, skipping visualization save")

    return routing_matrix_np


def checkpoint_moe_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    save_path: str,
) -> None:
    """
    Save MoE model checkpoint with expert statistics.

    Args:
        model: MoE model
        optimizer: Optimizer
        epoch: Current epoch
        step: Current step
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'config': model.config if hasattr(model, 'config') else None,
    }

    # Add expert usage stats if available
    if hasattr(model, 'get_expert_usage_stats'):
        checkpoint['expert_usage_stats'] = model.get_expert_usage_stats()

    torch.save(checkpoint, save_path)
    print(f"Saved MoE checkpoint to {save_path}")
