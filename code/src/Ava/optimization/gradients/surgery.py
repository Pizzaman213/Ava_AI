"""
Gradient Surgery for Multi-Task Learning.

This module implements gradient surgery techniques to resolve gradient conflicts
in multi-task learning scenarios, ensuring that optimization for one task
doesn't negatively interfere with other tasks.
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from collections import defaultdict
import math


class GradientSurgeon:
    """
    Gradient Surgery implementation for multi-task learning.

    This class implements various gradient surgery techniques to handle
    conflicting gradients in multi-task optimization.
    """

    def __init__(
        self,
        method: str = "pcgrad",
        cosine_similarity_threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize gradient surgeon.

        Args:
            method: Surgery method ('pcgrad', 'graddrop', 'gradnorm', 'cagrad')
            cosine_similarity_threshold: Threshold for detecting conflicts
            reduction: How to reduce gradients ('mean', 'sum')
        """
        self.method = method.lower()
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.reduction = reduction

        # Supported methods
        self.supported_methods = {
            'pcgrad': self._pcgrad,
            'graddrop': self._graddrop,
            'gradnorm': self._gradnorm,
            'cagrad': self._cagrad,
            'mgda': self._mgda
        }

        if self.method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {method}")

    def apply_surgery(
        self,
        gradients: Dict[str, List[torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> List[torch.Tensor]:
        """
        Apply gradient surgery to resolve conflicts.

        Args:
            gradients: Dictionary mapping task names to gradient lists
            losses: Optional task losses for some methods
            task_weights: Optional task importance weights

        Returns:
            List of surgically modified gradients
        """
        # Convert gradients to matrix format
        grad_matrix = self._gradients_to_matrix(gradients)

        # Apply the selected method
        modified_grads = self.supported_methods[self.method](
            grad_matrix, gradients, losses, task_weights
        )

        return modified_grads

    def _gradients_to_matrix(self, gradients: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        """Convert gradient dictionary to matrix format."""
        # Flatten and concatenate gradients for each task
        task_grads = []
        for task_name, grad_list in gradients.items():
            flattened = torch.cat([g.flatten() for g in grad_list])
            task_grads.append(flattened)

        # Stack into matrix [num_tasks, num_params]
        return torch.stack(task_grads)

    def _matrix_to_gradients(
        self,
        grad_matrix: torch.Tensor,
        reference_gradients: Dict[str, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Convert matrix back to gradient list format."""
        # Get the structure from reference gradients
        task_names = list(reference_gradients.keys())
        ref_grad_list = reference_gradients[task_names[0]]

        # Compute parameter shapes and sizes
        param_shapes = [g.shape for g in ref_grad_list]
        param_sizes = [g.numel() for g in ref_grad_list]

        # Aggregate gradients across tasks
        if self.reduction == "mean":
            aggregated_grad = grad_matrix.mean(dim=0)
        elif self.reduction == "sum":
            aggregated_grad = grad_matrix.sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Reshape back to original parameter structure
        result_grads = []
        start_idx = 0
        for shape, size in zip(param_shapes, param_sizes):
            param_grad = aggregated_grad[start_idx:start_idx + size].reshape(shape)
            result_grads.append(param_grad)
            start_idx += size

        return result_grads

    def _pcgrad(
        self,
        grad_matrix: torch.Tensor,
        gradients: Dict[str, List[torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> List[torch.Tensor]:
        """
        PCGrad: Project Conflicting Gradients.

        Projects gradients that have negative cosine similarity.
        """
        num_tasks, num_params = grad_matrix.shape
        modified_grads = grad_matrix.clone()

        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    g_i = modified_grads[i]
                    g_j = grad_matrix[j]

                    # Compute cosine similarity
                    cos_sim = torch.dot(g_i, g_j) / (g_i.norm() * g_j.norm() + 1e-8)

                    # If conflicting (negative cosine similarity), project
                    if cos_sim < 0:
                        # Project g_i onto the orthogonal complement of g_j
                        proj_g_j = torch.dot(g_i, g_j) / (g_j.norm() ** 2 + 1e-8) * g_j
                        modified_grads[i] = g_i - proj_g_j

        return self._matrix_to_gradients(modified_grads, gradients)

    def _graddrop(
        self,
        grad_matrix: torch.Tensor,
        gradients: Dict[str, List[torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> List[torch.Tensor]:
        """
        GradDrop: Drop conflicting gradient components.

        Randomly drops gradient components that conflict with other tasks.
        """
        num_tasks, num_params = grad_matrix.shape
        modified_grads = grad_matrix.clone()

        # Compute pairwise cosine similarities
        similarities = torch.zeros(num_tasks, num_tasks)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    g_i = grad_matrix[i]
                    g_j = grad_matrix[j]
                    cos_sim = torch.dot(g_i, g_j) / (g_i.norm() * g_j.norm() + 1e-8)
                    similarities[i, j] = cos_sim

        # For each task, drop components that conflict with others
        for i in range(num_tasks):
            conflict_mask = similarities[i] < self.cosine_similarity_threshold

            if conflict_mask.any():
                # Randomly drop some gradient components
                drop_prob = 0.5 * conflict_mask.float().mean().item()
                dropout_mask = torch.bernoulli(torch.full((num_params,), float(1 - drop_prob)))
                modified_grads[i] = modified_grads[i] * dropout_mask

        return self._matrix_to_gradients(modified_grads, gradients)

    def _gradnorm(
        self,
        grad_matrix: torch.Tensor,
        gradients: Dict[str, List[torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> List[torch.Tensor]:
        """
        GradNorm: Gradient normalization for multi-task learning.

        Balances training by normalizing gradient magnitudes.
        """
        if losses is None:
            # Without losses, just normalize gradient magnitudes
            grad_norms = grad_matrix.norm(dim=1, keepdim=True)
            normalized_grads = grad_matrix / (grad_norms + 1e-8)
            return self._matrix_to_gradients(normalized_grads, gradients)

        num_tasks = grad_matrix.shape[0]
        task_names = list(losses.keys())

        # Compute gradient norms
        grad_norms = grad_matrix.norm(dim=1)

        # Compute loss ratios
        loss_values = torch.stack([losses[name] for name in task_names])
        loss_ratios = loss_values / loss_values.sum()

        # Compute target gradient ratios (inversely proportional to loss ratios)
        target_ratios = 1.0 / (loss_ratios + 1e-8)
        target_ratios = target_ratios / target_ratios.sum()

        # Compute current gradient ratios
        current_ratios = grad_norms / grad_norms.sum()

        # Compute reweighting factors
        reweight_factors = target_ratios / (current_ratios + 1e-8)

        # Apply reweighting
        reweighted_grads = grad_matrix * reweight_factors.unsqueeze(1)

        return self._matrix_to_gradients(reweighted_grads, gradients)

    def _cagrad(
        self,
        grad_matrix: torch.Tensor,
        gradients: Dict[str, List[torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        c: float = 0.5
    ) -> List[torch.Tensor]:
        """
        CAGrad: Conflict-Averse Gradient descent.

        Minimizes the maximum cosine similarity between gradients.
        """
        num_tasks, num_params = grad_matrix.shape
        GG = grad_matrix @ grad_matrix.T  # Gram matrix

        # Solve for optimal combination weights
        def solve_min_norm_point(GG):
            """Solve for minimum norm point in the convex hull."""
            n = GG.shape[0]
            ones = torch.ones(n, 1).to(GG.device)

            # Solve: minimize 0.5 * x^T G x subject to sum(x) = 1, x >= 0
            # Using a simple iterative algorithm
            x = torch.ones(n).to(GG.device) / n

            for _ in range(100):  # Max iterations
                # Gradient of the objective
                grad = GG @ x

                # Find the most violating constraint
                min_idx = torch.argmin(grad)

                # Update step
                gamma = torch.clamp((grad[min_idx] - grad).max() / (2 * GG.diagonal().max()), 0, 1)

                # Update weights
                x_new = (1 - gamma) * x
                x_new[min_idx] += gamma
                x = x_new

            return x

        # Get optimal weights
        weights = solve_min_norm_point(GG)

        # Compute weighted gradient
        weighted_grad = (weights.unsqueeze(1) * grad_matrix).sum(dim=0)

        # Apply conflict-averse correction
        for i in range(num_tasks):
            g_i = grad_matrix[i]
            cos_sim = torch.dot(g_i, weighted_grad) / (g_i.norm() * weighted_grad.norm() + 1e-8)

            if cos_sim < c:
                # Correct the gradient to reduce conflict
                correction = (c - cos_sim) * weighted_grad / (weighted_grad.norm() + 1e-8)
                grad_matrix[i] = g_i + correction

        return self._matrix_to_gradients(grad_matrix, gradients)

    def _mgda(
        self,
        grad_matrix: torch.Tensor,
        gradients: Dict[str, List[torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> List[torch.Tensor]:
        """
        MGDA: Multi-Gradient Descent Algorithm.

        Finds Pareto optimal direction for multi-objective optimization.
        """
        # Solve for the minimum norm point in the convex hull of gradients
        num_tasks = grad_matrix.shape[0]

        # Compute Gram matrix
        GG = grad_matrix @ grad_matrix.T

        # Use Frank-Wolfe algorithm to solve the constrained optimization
        def frank_wolfe_solver(GG, max_iter=1000, tolerance=1e-6):
            n = GG.shape[0]
            x = torch.ones(n).to(GG.device) / n

            for iteration in range(max_iter):
                # Compute gradient
                grad = GG @ x

                # Find vertex that minimizes linear approximation
                min_idx = torch.argmin(grad)

                # Check optimality condition
                gap = torch.max(grad) - grad[min_idx]
                if gap < tolerance:
                    break

                # Compute step size
                d = torch.zeros_like(x)
                d[min_idx] = 1.0
                direction = d - x

                # Line search
                numerator = torch.dot(grad, direction)
                denominator = torch.dot(direction, GG @ direction)

                if denominator > 0:
                    step_size = torch.clamp(-numerator / denominator, 0, 1)
                else:
                    step_size = 1.0

                # Update
                x = x + step_size * direction

            return x

        # Get optimal weights
        alpha = frank_wolfe_solver(GG)

        # Compute the Pareto optimal gradient
        pareto_grad = (alpha.unsqueeze(1) * grad_matrix).sum(dim=0)

        # Return as single aggregated gradient
        return self._matrix_to_gradients(pareto_grad.unsqueeze(0), gradients)


class AdaptiveGradientSurgeon:
    """
    Adaptive gradient surgeon that selects the best method based on gradient conflicts.

    This class automatically detects the level of gradient conflicts and
    applies the most appropriate gradient surgery technique.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        conflict_threshold: float = 0.3,
        adaptation_frequency: int = 100
    ):
        """
        Initialize adaptive gradient surgeon.

        Args:
            methods: List of methods to choose from
            conflict_threshold: Threshold for detecting significant conflicts
            adaptation_frequency: How often to re-evaluate method selection
        """
        if methods is None:
            methods = ['pcgrad', 'graddrop', 'gradnorm']

        self.methods = methods
        self.conflict_threshold = conflict_threshold
        self.adaptation_frequency = adaptation_frequency

        # Initialize surgeons for each method
        self.surgeons = {
            method: GradientSurgeon(method=method)
            for method in methods
        }

        # Performance tracking
        self.method_performance = defaultdict(list)
        self.step_count = 0
        self.current_method = methods[0]

    def compute_gradient_conflicts(self, gradients: Dict[str, List[torch.Tensor]]) -> float:
        """Compute the average gradient conflict score."""
        grad_matrix = self._gradients_to_matrix(gradients)
        num_tasks = grad_matrix.shape[0]

        if num_tasks < 2:
            return 0.0

        conflicts = []
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                g_i = grad_matrix[i]
                g_j = grad_matrix[j]
                cos_sim = torch.dot(g_i, g_j) / (g_i.norm() * g_j.norm() + 1e-8)
                conflicts.append(max(0, -cos_sim.item()))  # Only negative similarities

        return float(np.mean(conflicts)) if conflicts else 0.0

    def _gradients_to_matrix(self, gradients: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        """Convert gradient dictionary to matrix format."""
        task_grads = []
        for task_name, grad_list in gradients.items():
            flattened = torch.cat([g.flatten() for g in grad_list])
            task_grads.append(flattened)
        return torch.stack(task_grads)

    def select_method(self, gradients: Dict[str, List[torch.Tensor]]) -> str:
        """Select the best method based on current gradient state."""
        conflict_score = self.compute_gradient_conflicts(gradients)

        if conflict_score < 0.1:
            # Low conflict - use simple averaging
            return 'gradnorm'
        elif conflict_score < 0.5:
            # Medium conflict - use PCGrad
            return 'pcgrad'
        else:
            # High conflict - use more aggressive methods
            return 'cagrad'

    def apply_surgery(
        self,
        gradients: Dict[str, List[torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> List[torch.Tensor]:
        """Apply adaptive gradient surgery."""
        # Select method every N steps
        if self.step_count % self.adaptation_frequency == 0:
            self.current_method = self.select_method(gradients)

        # Apply selected method
        surgeon = self.surgeons[self.current_method]
        result = surgeon.apply_surgery(gradients, losses, task_weights)

        # Track performance (simplified - could be more sophisticated)
        conflict_score = self.compute_gradient_conflicts(gradients)
        self.method_performance[self.current_method].append(conflict_score)

        self.step_count += 1
        return result


class GradientConflictAnalyzer:
    """
    Analyzer for understanding gradient conflicts in multi-task learning.

    This class provides tools for visualizing and understanding
    gradient conflicts between tasks.
    """

    def __init__(self):
        self.conflict_history = []
        self.method_history = []

    def analyze_gradients(
        self,
        gradients: Dict[str, List[torch.Tensor]],
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze gradient conflicts and provide diagnostic information.

        Args:
            gradients: Task gradients
            task_names: Optional task names for better reporting

        Returns:
            Dictionary with conflict analysis results
        """
        if task_names is None:
            task_names = list(gradients.keys())

        # Convert to matrix format
        grad_matrix = self._gradients_to_matrix(gradients)
        num_tasks, num_params = grad_matrix.shape

        # Compute pairwise similarities
        similarities = torch.zeros(num_tasks, num_tasks)
        conflicts = torch.zeros(num_tasks, num_tasks)

        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    g_i = grad_matrix[i]
                    g_j = grad_matrix[j]
                    cos_sim = torch.dot(g_i, g_j) / (g_i.norm() * g_j.norm() + 1e-8)
                    similarities[i, j] = cos_sim
                    conflicts[i, j] = max(0, -cos_sim)

        # Compute statistics
        analysis = {
            'num_tasks': num_tasks,
            'num_parameters': num_params,
            'similarity_matrix': similarities.cpu().numpy(),
            'conflict_matrix': conflicts.cpu().numpy(),
            'average_conflict': conflicts.mean().item(),
            'max_conflict': conflicts.max().item(),
            'gradient_norms': grad_matrix.norm(dim=1).cpu().numpy(),
            'task_names': task_names
        }

        # Find most conflicting task pairs
        conflict_pairs = []
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                conflict_score = conflicts[i, j].item()
                if conflict_score > 0.1:  # Threshold for significant conflict
                    conflict_pairs.append({
                        'task1': task_names[i],
                        'task2': task_names[j],
                        'conflict_score': conflict_score,
                        'similarity': similarities[i, j].item()
                    })

        # Sort by conflict score
        conflict_pairs.sort(key=lambda x: x['conflict_score'], reverse=True)
        analysis['conflict_pairs'] = conflict_pairs

        # Recommendations
        recommendations = []
        if analysis['average_conflict'] > 0.3:
            recommendations.append("High gradient conflicts detected. Consider using PCGrad or CAGrad.")
        if analysis['max_conflict'] > 0.7:
            recommendations.append("Severe conflicts present. Task weighting may be needed.")
        if len(conflict_pairs) > num_tasks:
            recommendations.append("Many conflicting task pairs. Consider task grouping.")

        analysis['recommendations'] = recommendations

        return analysis

    def _gradients_to_matrix(self, gradients: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        """Convert gradient dictionary to matrix format."""
        task_grads = []
        for task_name, grad_list in gradients.items():
            flattened = torch.cat([g.flatten() for g in grad_list])
            task_grads.append(flattened)
        return torch.stack(task_grads)

    def track_conflict_over_time(self, analysis: Dict[str, Any]):
        """Track conflict metrics over time for trend analysis."""
        self.conflict_history.append({
            'step': len(self.conflict_history),
            'average_conflict': analysis['average_conflict'],
            'max_conflict': analysis['max_conflict'],
            'num_conflicts': len(analysis['conflict_pairs'])
        })

    def get_conflict_trends(self) -> Dict[str, List[float]]:
        """Get conflict trends over training steps."""
        if not self.conflict_history:
            return {}

        trends = {
            'steps': [entry['step'] for entry in self.conflict_history],
            'average_conflicts': [entry['average_conflict'] for entry in self.conflict_history],
            'max_conflicts': [entry['max_conflict'] for entry in self.conflict_history],
            'num_conflicts': [entry['num_conflicts'] for entry in self.conflict_history]
        }

        return trends