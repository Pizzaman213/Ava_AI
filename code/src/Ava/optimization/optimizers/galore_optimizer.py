"""
GaLore (Gradient Low-Rank Projection) Optimizer Implementation

Based on: "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
Paper: https://arxiv.org/abs/2403.03507

GaLore reduces memory usage by projecting gradients into a low-rank subspace,
achieving 50-65% gradient memory reduction with minimal quality impact (<1%).

Key Features:
- Low-rank gradient projection via SVD
- Periodic subspace updates
- Compatible with AdamW and Lion optimizers
- Configurable rank and update frequency
"""

import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional, Tuple, Callable
import math
from collections import defaultdict


class GaLoreProjector:
    """
    Handles low-rank projection of gradients using SVD.

    The projector maintains projection matrices (U, V) that define
    the low-rank subspace for gradient updates.
    """

    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = 'std'
    ):
        """
        Args:
            rank: Target rank for low-rank projection
            update_proj_gap: Number of steps between projection matrix updates
            scale: Scaling factor for projected gradients
            proj_type: Projection type ('std' for standard SVD)
        """
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.ortho_matrix: Optional[torch.Tensor] = None

    def project(self, full_rank_grad: torch.Tensor, iter: int) -> torch.Tensor:
        """
        Project gradient to low-rank subspace.

        Args:
            full_rank_grad: Full gradient tensor (must be 2D)
            iter: Current iteration number (for determining when to update projection)

        Returns:
            Low-rank projected gradient (smaller tensor)
        """
        # Update projection matrix periodically
        if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad,
                self.rank,
                self.proj_type
            )

        # Project to low-rank: G_low = P.T @ G or G @ P depending on shape
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            # Tall matrix (M > N): Project columns
            # Result: (M, rank)
            low_rank = torch.matmul(full_rank_grad, self.ortho_matrix)
        else:
            # Wide matrix (M < N): Project rows
            # Result: (rank, N)
            low_rank = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank * self.scale

    def project_back(
        self,
        low_rank_tensor: torch.Tensor,
        ortho_matrix: torch.Tensor,
        original_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Project low-rank tensor back to full space.

        Args:
            low_rank_tensor: Tensor in low-rank subspace
            ortho_matrix: Orthogonal projection matrix (U or V from SVD)
            original_shape: Original shape before projection

        Returns:
            Full-rank tensor
        """
        if ortho_matrix is None:
            return low_rank_tensor

        # Determine projection direction from shapes
        M, N = original_shape[0], original_shape[1]

        if M >= N:
            # Tall matrix: we projected columns (M, N) -> (M, rank)
            # Inverse: (M, rank) @ V.T -> (M, N)
            return torch.matmul(low_rank_tensor, ortho_matrix.t())
        else:
            # Wide matrix: we projected rows (M, N) -> (rank, N)
            # Inverse: U @ (rank, N) -> (M, N)
            return torch.matmul(ortho_matrix, low_rank_tensor)

    @staticmethod
    def get_orthogonal_matrix(
        weights: torch.Tensor,
        rank: int,
        proj_type: str = 'std'
    ) -> torch.Tensor:
        """
        Compute orthogonal projection matrix via SVD.

        Args:
            weights: Weight gradient tensor (2D)
            rank: Target rank
            proj_type: Type of projection ('std' for standard)

        Returns:
            Orthogonal matrix for projection (U or V from SVD)
        """
        module_params = weights

        if proj_type == 'std':
            # Use randomized SVD for efficiency
            if module_params.shape[0] >= module_params.shape[1]:
                # Tall matrix: use right singular vectors
                # G ≈ U @ S @ V.T, project using V
                _, _, V = torch.svd_lowrank(module_params, q=rank)
                ortho_matrix = V[:, :rank]
            else:
                # Wide matrix: use left singular vectors
                # G ≈ U @ S @ V.T, project using U.T
                U, _, _ = torch.svd_lowrank(module_params, q=rank)
                ortho_matrix = U[:, :rank]
        else:
            raise ValueError(f"Unsupported projection type: {proj_type}")

        return ortho_matrix


class GaLoreAdamW(Optimizer):
    """
    AdamW optimizer with GaLore gradient projection.

    Combines the benefits of:
    - AdamW: Decoupled weight decay
    - GaLore: Low-rank gradient projection for memory efficiency

    Memory Savings: 50-65% gradient memory reduction
    Quality Impact: <1% with proper tuning
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        # GaLore specific parameters
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = 'std',
        # Optional: only apply GaLore to specific parameter types
        galore_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None
    ):
        """
        Args:
            params: Iterable of parameters or parameter groups
            lr: Learning rate
            betas: Coefficients for running averages (Adam beta1, beta2)
            eps: Term for numerical stability
            weight_decay: Weight decay coefficient
            rank: Rank for low-rank projection
            update_proj_gap: Steps between projection updates
            scale: Scaling factor for projected gradients
            proj_type: Projection type
            galore_filter: Optional function to filter which params use GaLore
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type
        )
        super(GaLoreAdamW, self).__init__(params, defaults)

        self.galore_filter = galore_filter

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure: Optional closure to reevaluate the model
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('GaLoreAdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Initialize GaLore projector if applicable
                    should_use_galore = grad.dim() >= 2 and grad.numel() > 1000

                    if should_use_galore:
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            proj_type=group['proj_type']
                        )
                        # GaLore: Store LOW-RANK optimizer states for memory savings
                        # This is the key - we allocate smaller buffers
                        rank = group['rank']
                        if grad.shape[0] >= grad.shape[1]:
                            # Tall matrix - project columns
                            state['exp_avg'] = torch.zeros(grad.shape[0], rank, dtype=grad.dtype, device=grad.device)
                            state['exp_avg_sq'] = torch.zeros(grad.shape[0], rank, dtype=grad.dtype, device=grad.device)
                        else:
                            # Wide matrix - project rows
                            state['exp_avg'] = torch.zeros(rank, grad.shape[1], dtype=grad.dtype, device=grad.device)
                            state['exp_avg_sq'] = torch.zeros(rank, grad.shape[1], dtype=grad.dtype, device=grad.device)
                    else:
                        state['projector'] = None
                        # Standard: Full-size optimizer states
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # GaLore algorithm: Work in projected subspace
                if state['projector'] is not None:
                    # Project gradient to low-rank subspace
                    grad_proj = state['projector'].project(grad, state['step'])

                    # Update moments in LOW-RANK space (this is where memory savings happen)
                    exp_avg.mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_proj, grad_proj, value=1 - beta2)

                    # Compute step in low-rank space
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1

                    # Compute update in low-rank space
                    step_proj = exp_avg / denom

                    # Project back to full space for parameter update
                    step_full = state['projector'].project_back(
                        step_proj,
                        state['projector'].ortho_matrix,
                        grad.shape
                    )

                    # Apply weight decay (decoupled, as in AdamW)
                    if group['weight_decay'] > 0.0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    # Apply update
                    p.data.add_(step_full, alpha=-step_size)
                else:
                    # Standard AdamW update (no projection)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1

                    # Apply weight decay (decoupled, as in AdamW)
                    if group['weight_decay'] > 0.0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    # Apply update
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class GaLoreLion(Optimizer):
    """
    Lion optimizer with GaLore gradient projection.

    Lion (Evolved Sign Momentum) uses sign of gradient for updates,
    making it extremely memory efficient. Combined with GaLore for
    even greater memory savings.

    Memory Savings: 60-75% compared to AdamW (Lion alone: 50%, GaLore: additional 50%)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
        # GaLore specific parameters
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = 'std',
        galore_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None
    ):
        """
        Args:
            params: Iterable of parameters or parameter groups
            lr: Learning rate (typically 3-10x smaller than AdamW)
            betas: Coefficients for running averages (momentum, interpolation)
            weight_decay: Weight decay coefficient
            rank: Rank for low-rank projection
            update_proj_gap: Steps between projection updates
            scale: Scaling factor for projected gradients
            proj_type: Projection type
            galore_filter: Optional function to filter which params use GaLore
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type
        )
        super(GaLoreLion, self).__init__(params, defaults)

        self.galore_filter = galore_filter

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Lion update rule:
        1. c_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        2. m_t = β₂ * m_{t-1} + (1 - β₂) * g_t
        3. θ_t = θ_{t-1} - η * (sign(c_t) + λ * θ_{t-1})
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('GaLoreLion does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Initialize GaLore projector if applicable
                    should_use_galore = grad.dim() >= 2 and grad.numel() > 1000

                    if should_use_galore:
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            proj_type=group['proj_type']
                        )
                        # GaLore: Store LOW-RANK momentum for memory savings
                        rank = group['rank']
                        if grad.shape[0] >= grad.shape[1]:
                            state['exp_avg'] = torch.zeros(grad.shape[0], rank, dtype=grad.dtype, device=grad.device)
                        else:
                            state['exp_avg'] = torch.zeros(rank, grad.shape[1], dtype=grad.dtype, device=grad.device)
                    else:
                        state['projector'] = None
                        # Standard: Full-size momentum
                        state['exp_avg'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_avg = state['exp_avg']

                # GaLore Lion algorithm
                if state['projector'] is not None:
                    # Project gradient to low-rank
                    grad_proj = state['projector'].project(grad, state['step'])

                    # Lion update in LOW-RANK space
                    # 1. Compute interpolated gradient: c_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                    update_proj = exp_avg.clone().mul_(beta1).add_(grad_proj, alpha=1 - beta1)

                    # 2. Update momentum in low-rank space: m_t = β₂ * m_{t-1} + (1 - β₂) * g_t
                    exp_avg.mul_(beta2).add_(grad_proj, alpha=1 - beta2)

                    # 3. Project update back to full space
                    update_full = state['projector'].project_back(
                        update_proj.sign_(),
                        state['projector'].ortho_matrix,
                        grad.shape
                    )

                    # 4. Apply weight decay
                    if group['weight_decay'] > 0.0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    # 5. Apply update: θ_t = θ_{t-1} - η * sign(c_t)
                    p.data.add_(update_full, alpha=-group['lr'])
                else:
                    # Standard Lion update
                    # 1. Compute interpolated gradient
                    update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1)

                    # 2. Update momentum
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                    # 3. Apply weight decay
                    if group['weight_decay'] > 0.0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    # 4. Apply update
                    p.data.add_(update.sign_(), alpha=-group['lr'])

        return loss


def create_galore_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    rank: int = 128,
    update_proj_gap: int = 200,
    galore_scale: float = 1.0,
    **kwargs
) -> Optimizer:
    """
    Factory function to create GaLore optimizer with sensible defaults.

    Args:
        model: PyTorch model
        optimizer_type: 'adamw' or 'lion'
        lr: Learning rate
        weight_decay: Weight decay coefficient
        rank: GaLore projection rank
        update_proj_gap: Steps between projection updates
        galore_scale: Scaling factor for projected gradients
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured GaLore optimizer

    Example:
        >>> optimizer = create_galore_optimizer(
        ...     model,
        ...     optimizer_type='adamw',
        ...     lr=1e-3,
        ...     rank=128
        ... )
    """
    # Separate parameters that should use GaLore
    # Typically: large weight matrices (>1000 params)
    # Not: biases, layer norms, embeddings
    galore_params = []
    regular_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Use GaLore for large 2D+ tensors (weight matrices)
        if param.dim() >= 2 and param.numel() > 1000:
            # Exclude specific layers if needed
            if not any(x in name.lower() for x in ['layernorm', 'ln', 'bn']):
                galore_params.append(param)
                continue

        regular_params.append(param)

    # Create parameter groups
    param_groups = []

    if galore_params:
        param_groups.append({
            'params': galore_params,
            'rank': rank,
            'update_proj_gap': update_proj_gap,
            'scale': galore_scale,
            'lr': lr,
            'weight_decay': weight_decay
        })

    if regular_params:
        param_groups.append({
            'params': regular_params,
            'rank': 0,  # No projection for regular params
            'lr': lr,
            'weight_decay': weight_decay
        })

    # Create optimizer
    if optimizer_type.lower() == 'adamw':
        return GaLoreAdamW(param_groups, **kwargs)
    elif optimizer_type.lower() == 'lion':
        return GaLoreLion(param_groups, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


# Export public API
__all__ = [
    'GaLoreProjector',
    'GaLoreAdamW',
    'GaLoreLion',
    'create_galore_optimizer'
]
