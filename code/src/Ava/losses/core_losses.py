"""
Core Loss Functions

This module contains fundamental training objectives:
- Temperature-scaled cross-entropy with adaptive temperature
- Focal loss for class imbalance
- Label smoothing
- Perplexity tracking
- Adaptive loss scaling
- Composite loss wrapper

Consolidated from: deepseek_loss.py, advanced_losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math


class TemperatureScaledCrossEntropy(nn.Module):
    """
    Temperature-scaled cross-entropy loss with adaptive temperature and label smoothing.

    This loss improves gradient flow and training stability through temperature
    scaling and optional label smoothing. Includes EOS penalty to prevent early termination.

    From: deepseek_loss.py
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        label_smoothing: float = 0.1,
        vocab_size: Optional[int] = None,
        temperature_bounds: Tuple[float, float] = (0.5, 2.0),
        adaptation_rate: float = 0.01,
        eos_token_id: Optional[int] = None,
        min_sequence_length: int = 20,
        eos_penalty_weight: float = 5.0
    ):
        """
        Initialize temperature-scaled cross-entropy loss.

        Args:
            initial_temperature: Starting temperature value
            adaptive_temperature: Whether to adapt temperature based on training
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            vocab_size: Vocabulary size (required for label smoothing)
            temperature_bounds: Min and max temperature values
            adaptation_rate: Rate of temperature adaptation
            eos_token_id: EOS token ID for early EOS penalty
            min_sequence_length: Minimum sequence length before allowing EOS
            eos_penalty_weight: Penalty weight for early EOS tokens
        """
        super().__init__()
        self.register_buffer('temperature', torch.tensor(initial_temperature))
        self.adaptive_temperature = adaptive_temperature
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.temperature_bounds = temperature_bounds
        self.adaptation_rate = adaptation_rate

        # EOS penalty settings
        self.eos_token_id = eos_token_id
        self.min_sequence_length = min_sequence_length
        self.eos_penalty_weight = eos_penalty_weight

        # Track loss statistics for adaptive temperature
        self.register_buffer('loss_history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('history_size', torch.tensor(0))

        if label_smoothing > 0 and vocab_size is None:
            raise ValueError("vocab_size must be provided when using label smoothing")

    def update_temperature(self, current_loss: torch.Tensor):
        """
        Update temperature based on loss trends.

        Lower temperature when loss is stable (encourage confidence).
        Higher temperature when loss is volatile (encourage exploration).
        """
        if not self.adaptive_temperature:
            return

        # Update loss history
        ptr = self.history_ptr.item() if isinstance(self.history_ptr, torch.Tensor) else int(self.history_ptr)
        self.loss_history[ptr] = current_loss.item()  # type: ignore[index]
        self.history_ptr = torch.tensor((ptr + 1) % 100)
        self.history_size = torch.min(self.history_size + 1, torch.tensor(100))

        # Need sufficient history for adaptation
        if self.history_size < 10:
            return

        # Calculate loss variance over recent history
        history_size_val = self.history_size.item() if isinstance(self.history_size, torch.Tensor) else int(self.history_size)
        recent_losses = self.loss_history[:history_size_val]  # type: ignore[index]
        loss_variance = torch.var(recent_losses)
        loss_mean = torch.mean(recent_losses)

        # Calculate coefficient of variation (normalized variance)
        if loss_mean > 0:
            cv = torch.sqrt(loss_variance) / loss_mean

            # High variance -> increase temperature (more exploration)
            # Low variance -> decrease temperature (more confidence)
            if cv > 0.1:  # High variance threshold
                temp_delta = self.adaptation_rate
            elif cv < 0.05:  # Low variance threshold
                temp_delta = -self.adaptation_rate
            else:
                temp_delta = 0.0

            # Update temperature with bounds
            new_temp = self.temperature + temp_delta
            self.temperature = torch.clamp(
                new_temp,
                self.temperature_bounds[0],
                self.temperature_bounds[1]
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> Dict[str, Any]:
        """
        Compute temperature-scaled cross-entropy loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Dictionary containing loss and temperature info
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Reshape for loss computation
        batch_size, seq_len = targets.shape
        vocab_size = logits.shape[-1]

        logits_flat = scaled_logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Apply label smoothing if configured
        if self.label_smoothing > 0:
            with torch.no_grad():
                # Create smoothed target distribution
                smoothed_targets = torch.zeros_like(logits_flat)
                smoothed_targets.fill_(self.label_smoothing / (vocab_size - 1))
                smoothed_targets.scatter_(1, targets_flat.unsqueeze(1),
                                        1.0 - self.label_smoothing)

            # Compute loss with smoothed targets
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Apply EOS penalty for early termination
        if self.eos_token_id is not None and self.eos_penalty_weight > 0:
            # Reshape to [batch_size, seq_len]
            loss_2d = loss.view(batch_size, seq_len)
            targets_2d = targets.view(batch_size, seq_len)

            # Create position indices [batch_size, seq_len]
            positions = torch.arange(seq_len, device=targets.device).unsqueeze(0).expand(batch_size, -1)

            # Find positions where target is EOS
            eos_mask = (targets_2d == self.eos_token_id)

            # Find positions before min_sequence_length
            early_mask = (positions < self.min_sequence_length)

            # Apply penalty to early EOS tokens
            early_eos_mask = eos_mask & early_mask
            loss_2d = loss_2d + early_eos_mask.float() * self.eos_penalty_weight

            # Flatten back
            loss = loss_2d.view(-1)

        # Apply attention mask if provided
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            loss = loss * mask_flat

            if reduction == 'mean':
                loss = loss.sum() / mask_flat.sum()
            elif reduction == 'sum':
                loss = loss.sum()
        else:
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()

        # Update temperature based on loss
        if self.training and reduction == 'mean':
            self.update_temperature(loss.detach())

        return {
            'loss': loss,
            'temperature': self.temperature.item(),
            'label_smoothing': self.label_smoothing
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples.

    From: advanced_losses.py
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions [batch_size, num_classes] or [batch_size, seq_len, num_classes]
            targets: Ground truth labels [batch_size] or [batch_size, seq_len]

        Returns:
            Focal loss value
        """
        # Flatten if needed
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Compute alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[targets]

        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization.

    Prevents the model from becoming too confident on training data.

    From: advanced_losses.py
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class PerplexityLoss(nn.Module):
    """
    Perplexity-based loss for language modeling evaluation.

    This loss computes perplexity and can be used as an auxiliary loss.

    From: advanced_losses.py
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute perplexity and cross-entropy loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]

        Returns:
            Dictionary with 'loss' and 'perplexity'
        """
        # Flatten logits and targets
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.ignore_index)

        # Compute perplexity
        perplexity = torch.exp(loss)

        return {
            'loss': loss,
            'perplexity': perplexity
        }


class AdaptiveLossScaling(nn.Module):
    """
    Adaptive loss scaling for balancing multiple loss components.

    This module learns to weight different loss components dynamically.

    From: advanced_losses.py
    """

    def __init__(self, num_losses: int, init_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_losses = num_losses

        if init_weights is None:
            init_weights = [1.0] * num_losses

        # Learnable loss weights (in log space for stability)
        self.log_weights = nn.Parameter(torch.tensor(init_weights).log())

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptively weighted loss.

        Args:
            losses: List of individual loss values

        Returns:
            Tuple of (combined weighted loss, normalized weights)
        """
        weights = torch.exp(self.log_weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Compute weighted loss
        weighted_loss_val = sum(w * loss for w, loss in zip(weights, losses))
        # Ensure it's a tensor, not just 0
        if not isinstance(weighted_loss_val, torch.Tensor):
            weighted_loss_val = torch.tensor(0.0, device=weights.device)

        return weighted_loss_val, weights


class CompositeLoss(nn.Module):
    """
    Composite loss that combines multiple loss functions.

    This is a convenient wrapper for combining different loss types.

    From: advanced_losses.py
    """

    def __init__(self, loss_config: Dict[str, Dict]):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        # Import here to avoid circular dependencies
        from .regularization_losses import ContrastiveLoss, DiversityLoss, ConsistencyLoss
        from .mtp_moe_losses import AuxiliaryLoss

        for loss_name, config in loss_config.items():
            loss_type = config.pop('type')
            weight = config.pop('weight', 1.0)

            self.weights[loss_name] = weight

            if loss_type == 'focal':
                self.losses[loss_name] = FocalLoss(**config)
            elif loss_type == 'contrastive':
                self.losses[loss_name] = ContrastiveLoss(**config)
            elif loss_type == 'label_smoothing':
                self.losses[loss_name] = LabelSmoothingLoss(**config)
            elif loss_type == 'auxiliary':
                self.losses[loss_name] = AuxiliaryLoss(**config)
            elif loss_type == 'consistency':
                self.losses[loss_name] = ConsistencyLoss(**config)
            elif loss_type == 'diversity':
                self.losses[loss_name] = DiversityLoss(**config)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute all configured losses.

        Args:
            **kwargs: Arguments for different loss functions

        Returns:
            Dictionary of computed losses
        """
        computed_losses = {}
        total_loss = 0.0

        for loss_name, loss_fn in self.losses.items():
            try:
                if loss_name == 'focal' and 'inputs' in kwargs and 'targets' in kwargs:
                    loss_value = loss_fn(kwargs['inputs'], kwargs['targets'])
                elif loss_name == 'contrastive' and 'embeddings' in kwargs:
                    loss_value = loss_fn(kwargs['embeddings'], kwargs.get('labels'))
                elif loss_name == 'auxiliary':
                    loss_dict = loss_fn(
                        gate_logits=kwargs.get('gate_logits'),
                        expert_indices=kwargs.get('expert_indices'),
                        expert_outputs=kwargs.get('expert_outputs'),
                        num_experts=kwargs.get('num_experts')
                    )
                    loss_value = sum(loss_dict.values())
                    computed_losses.update({f"aux_{k}": v for k, v in loss_dict.items()})
                else:
                    continue  # Skip if required args not available

                computed_losses[loss_name] = loss_value
                total_loss += self.weights[loss_name] * loss_value

            except Exception:
                # Skip losses that can't be computed with available inputs
                continue

        computed_losses['total'] = total_loss
        return computed_losses
