"""
Gradient Surgery for Multi-Task Learning.

This module implements gradient manipulation techniques for handling
conflicting gradients in multi-task or multi-loss training scenarios.

Available methods:
1. PCGrad - Projected Conflicting Gradients
2. CAGrad - Conflict-Averse Gradient descent
3. GradNorm - Gradient normalization
4. MGDA - Multiple Gradient Descent Algorithm

These methods help when different loss terms produce conflicting gradients
that can destabilize training or lead to task imbalance.

Example:
    >>> pcgrad = PCGrad()
    >>> gradients = [grad1, grad2, grad3]  # Gradients from different losses
    >>> modified_grads = pcgrad(gradients)

    >>> # Or use the GradientSurgeon for automatic application
    >>> surgeon = GradientSurgeon(model, method='pcgrad')
    >>> surgeon.step([loss1, loss2, loss3])
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GradientSurgeryConfig:
    """Configuration for gradient surgery."""
    method: str = 'pcgrad'  # 'pcgrad', 'cagrad', 'gradnorm', 'mgda'
    # PCGrad
    pcgrad_reduction: str = 'mean'  # How to combine after projection

    # CAGrad
    cagrad_c: float = 0.5  # Constraint coefficient
    cagrad_rescale: float = 1.0  # Gradient rescaling factor

    # GradNorm
    gradnorm_alpha: float = 1.5  # Restoring force strength
    gradnorm_update_freq: int = 1  # How often to update weights

    # MGDA
    mgda_normalize: bool = True  # Normalize task gradients

    # General
    conflict_threshold: float = 0.0  # Cosine similarity threshold for conflict
    enable_monitoring: bool = True  # Track gradient statistics


class PCGrad:
    """
    Projected Conflicting Gradients (PCGrad).

    When gradients from different tasks conflict (negative cosine similarity),
    project each gradient onto the normal plane of the conflicting gradient.

    This removes the conflicting component while preserving the non-conflicting parts.

    Reference:
        "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)

    Example:
        >>> pcgrad = PCGrad()
        >>> grad1 = torch.randn(100)
        >>> grad2 = torch.randn(100)
        >>> modified = pcgrad([grad1, grad2])
    """

    def __init__(
        self,
        reduction: str = 'mean',
        conflict_threshold: float = 0.0,
    ):
        self.reduction = reduction
        self.conflict_threshold = conflict_threshold

    def __call__(
        self,
        gradients: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply PCGrad to a list of gradients.

        Args:
            gradients: List of gradient tensors (same shape)

        Returns:
            List of modified gradients with conflicts resolved
        """
        if len(gradients) <= 1:
            return gradients

        num_tasks = len(gradients)
        device = gradients[0].device

        # Clone gradients to avoid modifying originals
        grads = [g.clone() for g in gradients]

        # Project each gradient onto others
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue

                # Check for conflict (negative cosine similarity)
                g_i = grads[i]
                g_j = gradients[j]  # Use original gradient for projection

                dot = torch.dot(g_i.flatten(), g_j.flatten())
                norm_j_sq = torch.dot(g_j.flatten(), g_j.flatten())

                # Cosine similarity
                norm_i = torch.norm(g_i)
                norm_j = torch.norm(g_j)
                if norm_i > 0 and norm_j > 0:
                    cos_sim = dot / (norm_i * norm_j)
                else:
                    cos_sim = torch.tensor(0.0, device=device)

                # Project if conflicting
                if cos_sim < self.conflict_threshold:
                    # Project g_i onto plane perpendicular to g_j
                    # g_i' = g_i - (g_i · g_j / ||g_j||^2) * g_j
                    if norm_j_sq > 1e-9:
                        proj = dot / norm_j_sq
                        grads[i] = g_i - proj * g_j

        return grads

    def compute_shared_gradient(
        self,
        gradients: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute a single shared gradient from multiple task gradients.

        Args:
            gradients: List of gradient tensors

        Returns:
            Combined gradient tensor
        """
        modified = self(gradients)

        if self.reduction == 'mean':
            return torch.stack(modified).mean(dim=0)
        elif self.reduction == 'sum':
            return torch.stack(modified).sum(dim=0)
        else:
            return torch.stack(modified).mean(dim=0)


class CAGrad:
    """
    Conflict-Averse Gradient descent (CAGrad).

    Finds a gradient direction that minimizes the worst-case loss among all tasks.
    This is more aggressive than PCGrad in avoiding conflict.

    Reference:
        "Conflict-Averse Gradient Descent for Multi-task Learning" (Liu et al., 2021)

    Example:
        >>> cagrad = CAGrad(c=0.5)
        >>> gradients = [grad1, grad2, grad3]
        >>> combined = cagrad(gradients)
    """

    def __init__(
        self,
        c: float = 0.5,
        rescale: float = 1.0,
    ):
        self.c = c  # Constraint coefficient
        self.rescale = rescale

    def __call__(
        self,
        gradients: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute conflict-averse gradient.

        Args:
            gradients: List of gradient tensors from different tasks

        Returns:
            Single combined gradient tensor
        """
        if len(gradients) == 1:
            return gradients[0]

        num_tasks = len(gradients)
        device = gradients[0].device
        dtype = gradients[0].dtype

        # Stack gradients: [num_tasks, grad_dim]
        grads = torch.stack([g.flatten() for g in gradients])
        grad_dim = grads.shape[1]

        # Compute average gradient
        g_avg = grads.mean(dim=0)
        g_avg_norm = torch.norm(g_avg)

        if g_avg_norm < 1e-9:
            return gradients[0].clone()

        # Compute task gradient deviations from average
        # g_i - g_avg for each task
        g_diff = grads - g_avg.unsqueeze(0)

        # Find the gradient that maximizes improvement while staying within constraint
        # Solve: max_g min_i <g, g_i> s.t. ||g - g_avg|| <= c * ||g_avg||

        # Compute Gram matrix G[i,j] = <g_i, g_j>
        G = torch.matmul(grads, grads.T)

        # Solve quadratic program to find optimal weights
        # For simplicity, use closed-form solution assuming convex combination
        try:
            # Compute optimal direction using gradient ascent on dual
            weights = self._solve_cagrad_weights(grads, g_avg, self.c)
            g_cagrad = torch.matmul(weights.unsqueeze(0), grads).squeeze(0)
        except Exception:
            # Fallback to average gradient
            g_cagrad = g_avg

        # Rescale
        g_cagrad = g_cagrad * self.rescale

        # Reshape back to original shape
        return g_cagrad.view_as(gradients[0])

    def _solve_cagrad_weights(
        self,
        grads: torch.Tensor,
        g_avg: torch.Tensor,
        c: float,
        max_iter: int = 50,
    ) -> torch.Tensor:
        """Solve for optimal task weights using projected gradient descent."""
        num_tasks = grads.shape[0]
        device = grads.device
        dtype = grads.dtype

        # Initialize uniform weights
        weights = torch.ones(num_tasks, device=device, dtype=dtype) / num_tasks

        # Compute constraint
        g_avg_norm = torch.norm(g_avg)
        constraint_radius = c * g_avg_norm

        lr = 0.1

        for _ in range(max_iter):
            # Current gradient
            g_current = torch.matmul(weights.unsqueeze(0), grads).squeeze(0)

            # Compute gradient of objective (maximize worst-case improvement)
            task_dots = torch.matmul(grads, g_current)  # [num_tasks]
            min_idx = task_dots.argmin()

            # Gradient update
            grad_weights = torch.zeros_like(weights)
            grad_weights[min_idx] = 1.0

            # Gradient step
            weights = weights + lr * grad_weights

            # Project to simplex
            weights = self._project_simplex(weights)

            # Check constraint
            g_new = torch.matmul(weights.unsqueeze(0), grads).squeeze(0)
            if torch.norm(g_new - g_avg) > constraint_radius:
                # Scale back
                direction = g_new - g_avg
                if torch.norm(direction) > 1e-9:
                    g_new = g_avg + constraint_radius * direction / torch.norm(direction)
                    # Recompute weights (approximate)
                    weights = torch.ones(num_tasks, device=device, dtype=dtype) / num_tasks

        return weights

    def _project_simplex(self, v: torch.Tensor) -> torch.Tensor:
        """Project vector onto probability simplex."""
        n = v.shape[0]
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0)
        rho = (u * torch.arange(1, n + 1, device=v.device, dtype=v.dtype) > (cssv - 1)).sum() - 1
        theta = (cssv[rho] - 1) / (rho + 1)
        return torch.clamp(v - theta, min=0)


class GradNorm:
    """
    Gradient Normalization for Multi-Task Learning.

    Dynamically adjusts task weights to balance gradient magnitudes
    across tasks, ensuring no single task dominates training.

    Reference:
        "GradNorm: Gradient Normalization for Adaptive Loss Balancing
        in Deep Multitask Networks" (Chen et al., 2018)

    Example:
        >>> gradnorm = GradNorm(num_tasks=3, alpha=1.5)
        >>> weights = gradnorm.update(task_losses, shared_grad)
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        initial_weights: Optional[List[float]] = None,
    ):
        self.num_tasks = num_tasks
        self.alpha = alpha  # Restoring force strength

        # Initialize task weights (learnable)
        if initial_weights is not None:
            self.weights = torch.tensor(initial_weights, requires_grad=True)
        else:
            self.weights = torch.ones(num_tasks, requires_grad=True)

        # Track initial losses for relative loss computation
        self.initial_losses: Optional[torch.Tensor] = None
        self.step_count = 0

    def get_weights(self) -> torch.Tensor:
        """Get normalized task weights."""
        return F.softmax(self.weights, dim=0) * self.num_tasks

    def update(
        self,
        task_losses: List[torch.Tensor],
        task_grads: List[torch.Tensor],
        shared_params: Optional[torch.Tensor] = None,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """
        Update task weights based on gradient norms.

        Args:
            task_losses: List of loss values per task
            task_grads: List of gradient tensors per task
            shared_params: Optional shared parameters for computing gradient norms
            lr: Learning rate for weight updates

        Returns:
            Updated task weights
        """
        self.step_count += 1
        device = task_losses[0].device if isinstance(task_losses[0], torch.Tensor) else 'cpu'

        # Convert to tensors if needed
        losses = torch.stack([
            l if isinstance(l, torch.Tensor) else torch.tensor(l, device=device)
            for l in task_losses
        ])

        # Initialize baseline losses
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()

        # Compute relative inverse training rate
        loss_ratios = losses / (self.initial_losses + 1e-9)
        avg_loss_ratio = loss_ratios.mean()
        relative_inverse_rate = loss_ratios / (avg_loss_ratio + 1e-9)

        # Compute gradient norms
        grad_norms = torch.stack([
            torch.norm(g.flatten()) for g in task_grads
        ])
        avg_grad_norm = grad_norms.mean()

        # Compute GradNorm target
        target_grad_norms = avg_grad_norm * (relative_inverse_rate ** self.alpha)

        # Compute GradNorm loss
        gradnorm_loss = (grad_norms - target_grad_norms).abs().sum()

        # Update weights
        if self.weights.grad is not None:
            self.weights.grad.zero_()

        gradnorm_loss.backward()

        with torch.no_grad():
            if self.weights.grad is not None:
                self.weights -= lr * self.weights.grad
                # Renormalize
                self.weights.data = self.weights.data - self.weights.data.mean() + 1.0

        return self.get_weights()


class MGDA:
    """
    Multiple Gradient Descent Algorithm (MGDA).

    Finds a Pareto-optimal gradient direction that improves all tasks
    or at least doesn't worsen any task.

    Reference:
        "Multi-Task Learning as Multi-Objective Optimization" (Sener & Koltun, 2018)

    Example:
        >>> mgda = MGDA()
        >>> combined_grad = mgda(gradients)
    """

    def __init__(
        self,
        normalize: bool = True,
        max_iter: int = 100,
    ):
        self.normalize = normalize
        self.max_iter = max_iter

    def __call__(
        self,
        gradients: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute MGDA gradient.

        Args:
            gradients: List of gradient tensors from different tasks

        Returns:
            Combined gradient tensor
        """
        if len(gradients) == 1:
            return gradients[0]

        num_tasks = len(gradients)
        device = gradients[0].device
        dtype = gradients[0].dtype

        # Stack and optionally normalize
        grads = torch.stack([g.flatten() for g in gradients])

        if self.normalize:
            norms = torch.norm(grads, dim=1, keepdim=True)
            grads = grads / (norms + 1e-9)

        # Find minimum-norm point in convex hull (Frank-Wolfe algorithm)
        weights = self._find_min_norm_weights(grads)

        # Compute combined gradient
        combined = torch.matmul(weights.unsqueeze(0), grads).squeeze(0)

        # Reshape
        return combined.view_as(gradients[0])

    def _find_min_norm_weights(
        self,
        grads: torch.Tensor,
    ) -> torch.Tensor:
        """Find weights that minimize ||sum(w_i * g_i)||^2."""
        num_tasks = grads.shape[0]
        device = grads.device
        dtype = grads.dtype

        # Compute Gram matrix
        G = torch.matmul(grads, grads.T)

        # Initialize uniform
        weights = torch.ones(num_tasks, device=device, dtype=dtype) / num_tasks

        for _ in range(self.max_iter):
            # Current gradient
            current = torch.matmul(G, weights)

            # Find improving direction (minimum element)
            idx = current.argmin()

            # Line search
            direction = torch.zeros_like(weights)
            direction[idx] = 1.0
            direction = direction - weights

            # Optimal step size (closed form for quadratic)
            numerator = torch.dot(current, direction)
            denominator = torch.dot(torch.matmul(G, direction), direction)

            if denominator > 1e-9:
                step = max(0, min(1, -numerator / denominator))
            else:
                step = 0

            # Update
            weights = weights + step * direction

            # Check convergence
            if step < 1e-6:
                break

        return weights


class GradientSurgeon:
    """
    High-level interface for gradient surgery.

    Automatically handles gradient computation and modification
    for multi-task learning scenarios.

    Args:
        model: The model being trained
        method: Gradient surgery method ('pcgrad', 'cagrad', 'gradnorm', 'mgda')
        config: Optional GradientSurgeryConfig

    Example:
        >>> surgeon = GradientSurgeon(model, method='pcgrad')
        >>> # During training:
        >>> losses = [loss1, loss2, loss3]
        >>> surgeon.backward(losses)
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = 'pcgrad',
        config: Optional[GradientSurgeryConfig] = None,
    ):
        self.model = model
        self.method = method
        self.config = config or GradientSurgeryConfig()

        # Initialize method
        if method == 'pcgrad':
            self.surgery = PCGrad(
                reduction=self.config.pcgrad_reduction,
                conflict_threshold=self.config.conflict_threshold,
            )
        elif method == 'cagrad':
            self.surgery = CAGrad(
                c=self.config.cagrad_c,
                rescale=self.config.cagrad_rescale,
            )
        elif method == 'mgda':
            self.surgery = MGDA(
                normalize=self.config.mgda_normalize,
            )
        elif method == 'gradnorm':
            # GradNorm needs to be initialized with num_tasks
            self.surgery = None
            self._gradnorm_weights = None
        else:
            raise ValueError(f"Unknown method: {method}")

        # Statistics
        self.stats = {
            'conflicts': 0,
            'total_updates': 0,
            'avg_cosine_similarity': 0.0,
        }

    def backward(
        self,
        losses: List[torch.Tensor],
        retain_graph: bool = False,
    ):
        """
        Compute and apply gradient surgery.

        Args:
            losses: List of loss tensors from different tasks
            retain_graph: Whether to retain computation graph
        """
        if len(losses) == 1:
            losses[0].backward(retain_graph=retain_graph)
            return

        # Compute gradients for each loss
        task_grads = []
        for i, loss in enumerate(losses):
            self.model.zero_grad()
            loss.backward(retain_graph=(i < len(losses) - 1) or retain_graph)

            # Collect gradients
            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.clone())
                else:
                    grads.append(torch.zeros_like(param))
            task_grads.append(torch.cat([g.flatten() for g in grads]))

        # Apply gradient surgery
        if self.method == 'gradnorm':
            if self.surgery is None:
                self.surgery = GradNorm(
                    num_tasks=len(losses),
                    alpha=self.config.gradnorm_alpha,
                )
            weights = self.surgery.get_weights()
            combined = sum(w * g for w, g in zip(weights, task_grads))
            # Update GradNorm weights periodically
            if self.stats['total_updates'] % self.config.gradnorm_update_freq == 0:
                self.surgery.update(losses, task_grads)
        else:
            # PCGrad, CAGrad, MGDA
            if isinstance(self.surgery, (PCGrad,)):
                modified = self.surgery(task_grads)
                combined = torch.stack(modified).mean(dim=0)
            else:
                combined = self.surgery(task_grads)

        # Apply combined gradients back to model
        self.model.zero_grad()
        idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            if param.grad is not None or True:
                param.grad = combined[idx:idx + numel].view_as(param)
            idx += numel

        # Update statistics
        self._update_stats(task_grads)

    def _update_stats(self, task_grads: List[torch.Tensor]):
        """Update gradient statistics."""
        self.stats['total_updates'] += 1

        if self.config.enable_monitoring and len(task_grads) >= 2:
            # Compute pairwise cosine similarities
            cos_sims = []
            conflicts = 0
            for i in range(len(task_grads)):
                for j in range(i + 1, len(task_grads)):
                    cos = F.cosine_similarity(
                        task_grads[i].unsqueeze(0),
                        task_grads[j].unsqueeze(0),
                    ).item()
                    cos_sims.append(cos)
                    if cos < self.config.conflict_threshold:
                        conflicts += 1

            self.stats['conflicts'] = conflicts
            self.stats['avg_cosine_similarity'] = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get gradient surgery statistics."""
        return self.stats.copy()


def apply_gradient_surgery(
    gradients: List[torch.Tensor],
    method: str = 'pcgrad',
    **kwargs,
) -> torch.Tensor:
    """
    Convenience function to apply gradient surgery.

    Args:
        gradients: List of gradient tensors
        method: Method to use ('pcgrad', 'cagrad', 'mgda')
        **kwargs: Additional arguments for the method

    Returns:
        Combined gradient tensor
    """
    if method == 'pcgrad':
        surgery = PCGrad(**kwargs)
        return surgery.compute_shared_gradient(gradients)
    elif method == 'cagrad':
        surgery = CAGrad(**kwargs)
        return surgery(gradients)
    elif method == 'mgda':
        surgery = MGDA(**kwargs)
        return surgery(gradients)
    else:
        # Default to mean
        return torch.stack(gradients).mean(dim=0)


__all__ = [
    'GradientSurgeryConfig',
    'PCGrad',
    'CAGrad',
    'GradNorm',
    'MGDA',
    'GradientSurgeon',
    'apply_gradient_surgery',
]
