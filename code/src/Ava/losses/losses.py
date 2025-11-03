"""
Unified Loss Module for Ava AI Training Pipeline

This module consolidates all loss functions into a single file:
- DeepSeek-style losses (temperature-scaled cross-entropy, multi-token prediction, MoE balancing)
- Adaptive MTP loss (confidence-weighted multi-token prediction)
- Repetition penalties (n-gram, immediate repetition)
- Anti-repetition loss
- Advanced losses (focal, contrastive, diversity, auxiliary, consistency, perplexity)
- Unified loss interface

All loss components are available in this single module for easier maintenance and imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from collections import Counter


# ============================================================================
# DEEPSEEK-STYLE LOSSES
# ============================================================================


class MultiTokenPredictionLoss(nn.Module):
    """
    Multi-Token Prediction (MTP) loss for improved long-range dependency learning.

    This loss predicts multiple future tokens simultaneously, helping the model
    learn better representations of future context.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_future_tokens: int = 3,
        mtp_weight: float = 0.1,
        shared_projection: bool = False,
        temperature: float = 1.0
    ):
        """
        Initialize Multi-Token Prediction loss.

        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Hidden dimension of the model
            num_future_tokens: Number of future tokens to predict (2-4 recommended)
            mtp_weight: Weight for MTP loss relative to main loss
            shared_projection: Whether to share projection heads across tokens
            temperature: Temperature for softmax scaling
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_future_tokens = num_future_tokens
        self.mtp_weight = mtp_weight
        self.temperature = temperature

        # Create projection heads for each future token
        if shared_projection:
            # Single shared projection for all future tokens
            self.projection = nn.Linear(hidden_size, vocab_size * num_future_tokens)
        else:
            # Separate projection for each future token
            self.projections = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size)
                for _ in range(num_future_tokens)
            ])
        self.shared_projection = shared_projection

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute multi-token prediction loss.

        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]
            target_ids: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary containing MTP loss and per-token losses
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Normalize hidden states for stability
        hidden_states = self.layer_norm(hidden_states)

        # Initialize losses
        total_mtp_loss = torch.tensor(0.0, device=device)
        per_token_losses = []

        # Compute predictions for each future token
        for future_idx in range(1, self.num_future_tokens + 1):
            # Skip if we don't have enough future tokens
            if future_idx >= seq_len:
                continue

            # Get hidden states for predicting future_idx tokens ahead
            pred_hidden = hidden_states[:, :-future_idx, :]

            # Get target IDs for future_idx tokens ahead
            future_targets = target_ids[:, future_idx:]

            # Project to vocabulary size
            if self.shared_projection:
                # Extract the appropriate slice from shared projection
                start_idx = (future_idx - 1) * self.vocab_size
                end_idx = future_idx * self.vocab_size
                logits = self.projection(pred_hidden)[:, :, start_idx:end_idx]
            else:
                logits = self.projections[future_idx - 1](pred_hidden)

            # Apply temperature scaling
            logits = logits / self.temperature

            # Reshape for loss computation
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = future_targets.reshape(-1)

            # Apply attention mask if provided
            if attention_mask is not None:
                mask_flat = attention_mask[:, future_idx:].reshape(-1)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

            # CRITICAL FIX: Accumulate losses as tensors to preserve gradient flow
            total_mtp_loss = total_mtp_loss + loss
            # Store detached scalar for logging only - don't break gradient flow
            per_token_losses.append(loss.detach().item())

        # Average over number of future tokens
        if self.num_future_tokens > 0:
            total_mtp_loss = total_mtp_loss / min(self.num_future_tokens, seq_len - 1)

        return {
            'mtp_loss': total_mtp_loss * self.mtp_weight,
            'per_token_losses': per_token_losses,
            'mtp_weight': self.mtp_weight
        }


class TemperatureScaledCrossEntropy(nn.Module):
    """
    Temperature-scaled cross-entropy loss with adaptive temperature and label smoothing.

    This loss improves gradient flow and training stability through temperature
    scaling and optional label smoothing.
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

        # CRITICAL FIX: Apply EOS penalty for early termination
        # This prevents the model from learning to output EOS immediately
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


class AuxiliaryFreeMoEBalancer(nn.Module):
    """
    Auxiliary-loss-free load balancing for Mixture of Experts.

    Instead of using auxiliary losses that can hurt performance, this module
    uses gradient manipulation to achieve expert load balancing.
    """

    def __init__(
        self,
        num_experts: int,
        balance_loss_weight: float = 0.0,  # Set to 0 for auxiliary-free
        gradient_balance_weight: float = 0.1,
        target_balance_ratio: float = 1.0,
        momentum: float = 0.9
    ):
        """
        Initialize auxiliary-free MoE balancer.

        Args:
            num_experts: Number of experts in the MoE layer
            balance_loss_weight: Weight for traditional balance loss (0 for auxiliary-free)
            gradient_balance_weight: Weight for gradient-based balancing
            target_balance_ratio: Target ratio for expert utilization
            momentum: Momentum for tracking expert statistics
        """
        super().__init__()
        self.num_experts = num_experts
        self.balance_loss_weight = balance_loss_weight
        self.gradient_balance_weight = gradient_balance_weight
        self.target_balance_ratio = target_balance_ratio
        self.momentum = momentum

        # Track expert utilization statistics
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('expert_scores', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

        # FIXED: Accumulation buffers for gradient accumulation support
        # These accumulate statistics across micro-steps before applying momentum
        self.register_buffer('_accumulated_counts', torch.zeros(num_experts))
        self.register_buffer('_accumulated_scores', torch.zeros(num_experts))
        self.register_buffer('_accumulated_tokens', torch.tensor(0.0))
        self.register_buffer('_accumulation_steps', torch.tensor(0))

    def update_statistics(
        self,
        expert_indices: torch.Tensor,
        expert_scores: torch.Tensor,
        is_optimizer_step: bool = True
    ):
        """
        Update expert utilization statistics with gradient accumulation support.

        Args:
            expert_indices: Selected expert indices [batch_size * seq_len, top_k]
            expert_scores: Expert selection scores [batch_size * seq_len, num_experts]
            is_optimizer_step: Whether this is an actual optimizer step (not just micro-step)
                              Set to True when optimizer.step() is called, False during gradient accumulation.
        """
        with torch.no_grad():
            # FIXED: Accumulate statistics across micro-steps, apply momentum only on optimizer steps
            # This prevents skewing momentum-based tracking when using gradient accumulation

            # Count expert usage
            for i in range(self.num_experts):
                expert_mask = (expert_indices == i).float()
                count = expert_mask.sum()
                self._accumulated_counts[i] += count  # type: ignore[index]

            # Track average scores (accumulate)
            avg_scores = expert_scores.mean(dim=0)
            self._accumulated_scores += avg_scores

            # Track total token count (accumulate)
            batch_tokens = float(expert_indices.shape[0])
            self._accumulated_tokens += batch_tokens

            # Increment accumulation step counter
            self._accumulation_steps += 1

            # When optimizer steps, apply momentum update and reset accumulators
            if is_optimizer_step:
                # Average accumulated statistics over all micro-steps
                # CRITICAL: Cast accumulation steps to int for division
                # Type annotation: _accumulation_steps is always a Tensor (registered buffer)
                accum_steps_tensor: torch.Tensor = self._accumulation_steps  # type: ignore[assignment]
                num_steps: int = max(1, int(accum_steps_tensor.item()))

                # Ensure num_steps is int for float conversion
                num_steps_float: float = float(num_steps)

                # Type annotation: these are always Tensors (registered buffers)
                avg_counts: torch.Tensor = self._accumulated_counts / num_steps_float  # type: ignore[assignment]
                avg_scores_per_step: torch.Tensor = self._accumulated_scores / num_steps_float  # type: ignore[assignment]
                avg_tokens: torch.Tensor = self._accumulated_tokens / num_steps_float  # type: ignore[assignment]

                # Apply momentum update with averaged statistics
                for i in range(self.num_experts):
                    self.expert_counts[i] = (  # type: ignore[index]
                        self.momentum * self.expert_counts[i] +  # type: ignore[index]
                        (1 - self.momentum) * avg_counts[i]
                    )

                self.expert_scores = (
                    self.momentum * self.expert_scores +
                    (1 - self.momentum) * avg_scores_per_step
                )

                self.total_tokens = (
                    self.momentum * self.total_tokens +
                    (1 - self.momentum) * avg_tokens
                )

                # Reset accumulators (all are tensors via register_buffer)
                # Type: registered buffers are always Tensors
                self._accumulated_counts.zero_()  # type: ignore[union-attr]
                self._accumulated_scores.zero_()  # type: ignore[union-attr]
                self._accumulated_tokens.zero_()  # type: ignore[union-attr]
                self._accumulation_steps.zero_()  # type: ignore[union-attr]

    def compute_balance_gradients(
        self,
        gate_logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute gradient adjustments for load balancing without auxiliary loss.

        Args:
            gate_logits: Router logits [batch_size * seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size * seq_len, top_k]

        Returns:
            Gradient adjustment tensor
        """
        with torch.no_grad():
            # Calculate target distribution
            target_count = self.total_tokens / self.num_experts

            # Calculate imbalance for each expert
            imbalance = self.expert_counts - target_count  # type: ignore[operator]

            # Normalize imbalance
            if self.total_tokens > 0:
                imbalance = imbalance / self.total_tokens

            # Create gradient adjustments
            # Overused experts get negative gradients (discourage selection)
            # Underused experts get positive gradients (encourage selection)
            grad_adjustment = -imbalance * self.gradient_balance_weight

            # Expand to match gate_logits shape
            grad_adjustment = grad_adjustment.unsqueeze(0).expand_as(gate_logits)

        return grad_adjustment

    def forward(
        self,
        gate_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        compute_loss: bool = False
    ) -> Dict[str, Any]:
        """
        Apply auxiliary-free load balancing.

        Args:
            gate_logits: Router logits [batch_size * seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size * seq_len, top_k]
            expert_outputs: Optional expert outputs for diversity
            compute_loss: Whether to compute traditional balance loss (for comparison)

        Returns:
            Dictionary with balancing information
        """
        # Update statistics
        expert_scores = F.softmax(gate_logits, dim=-1)
        self.update_statistics(expert_indices, expert_scores)

        # Compute gradient adjustments for balancing
        balancing_term = None
        if self.training and gate_logits.requires_grad:
            grad_adjustment = self.compute_balance_gradients(gate_logits, expert_indices)

            # CRITICAL FIX: Create effective gradient signal for load balancing
            # Previous implementation: (adjusted_logits - adjusted_logits.detach()) = 0 (no signal!)
            # New approach: Use gate_logits directly with gradient adjustment as steering signal
            # The gradient adjustment encourages underused experts and discourages overused ones
            if grad_adjustment is not None:
                adjusted_logits = gate_logits + grad_adjustment.detach()
                # Create loss that pulls gate_logits toward balanced distribution
                # This preserves gradients through gate_logits while steering toward balance
                balancing_term = F.mse_loss(gate_logits, adjusted_logits.detach())
                balancing_term = balancing_term * self.gradient_balance_weight
        else:
            balancing_term = torch.tensor(0.0, device=gate_logits.device, requires_grad=False)

        # Optionally compute traditional balance loss for comparison
        balance_loss = torch.tensor(0.0, device=gate_logits.device)
        if compute_loss and self.balance_loss_weight > 0:
            # Traditional load balancing loss (for comparison/debugging)
            expert_mask = F.one_hot(expert_indices, self.num_experts).float()
            expert_usage = expert_mask.sum(dim=0).sum(dim=0)
            gate_prob_sums = expert_scores.sum(dim=0)

            total_tokens = gate_logits.shape[0] * expert_indices.shape[1]
            balance_loss = self.num_experts * torch.sum(gate_prob_sums * expert_usage) / (total_tokens ** 2)
            balance_loss = balance_loss * self.balance_loss_weight

        # Calculate load balance statistics
        with torch.no_grad():
            if self.total_tokens > 0:
                expected_count = self.total_tokens / self.num_experts
                balance_ratio = self.expert_counts / (expected_count + 1e-6)  # type: ignore[operator]
                cv = torch.std(balance_ratio) / (torch.mean(balance_ratio) + 1e-6)
            else:
                balance_ratio = torch.ones(self.num_experts, device=gate_logits.device)
                cv = torch.tensor(0.0, device=gate_logits.device)

        return {
            'balance_loss': balance_loss,
            'balancing_term': balancing_term,
            'expert_counts': self.expert_counts.clone(),  # type: ignore[union-attr]
            'expert_balance_ratio': balance_ratio,
            'coefficient_of_variation': cv.item(),
            'gradient_weight': self.gradient_balance_weight
        }


class DeepSeekLoss(nn.Module):
    """
    Combined DeepSeek-style loss incorporating all components.

    This is the main loss class that combines temperature-scaled cross-entropy,
    multi-token prediction, and auxiliary-free MoE balancing.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_experts: Optional[int] = None,
        # Multi-token prediction settings
        use_mtp: bool = True,
        num_future_tokens: int = 3,
        mtp_weight: float = 0.1,
        # Temperature scaling settings
        initial_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        label_smoothing: float = 0.1,
        # MoE balancing settings
        use_moe_balancing: bool = True,
        gradient_balance_weight: float = 0.1,
        # EOS penalty settings
        eos_token_id: Optional[int] = None,
        min_sequence_length: int = 20,
        eos_penalty_weight: float = 5.0
    ):
        """
        Initialize DeepSeek-style loss.

        Args:
            vocab_size: Vocabulary size
            hidden_size: Model hidden dimension
            num_experts: Number of experts (for MoE models)
            use_mtp: Whether to use multi-token prediction
            num_future_tokens: Number of future tokens to predict
            mtp_weight: Weight for MTP loss
            initial_temperature: Initial temperature for scaling
            adaptive_temperature: Whether to adapt temperature
            label_smoothing: Label smoothing factor
            use_moe_balancing: Whether to use MoE balancing
            gradient_balance_weight: Weight for gradient-based balancing
            eos_token_id: EOS token ID for early EOS penalty
            min_sequence_length: Minimum sequence length before allowing EOS
            eos_penalty_weight: Penalty weight for early EOS tokens
        """
        super().__init__()

        # Main loss: temperature-scaled cross-entropy with EOS penalty
        self.main_loss = TemperatureScaledCrossEntropy(
            initial_temperature=initial_temperature,
            adaptive_temperature=adaptive_temperature,
            label_smoothing=label_smoothing,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            min_sequence_length=min_sequence_length,
            eos_penalty_weight=eos_penalty_weight
        )

        # Multi-token prediction loss
        self.use_mtp = use_mtp
        if use_mtp:
            self.mtp_loss = MultiTokenPredictionLoss(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_future_tokens=num_future_tokens,
                mtp_weight=mtp_weight
            )

        # MoE load balancing (auxiliary-free)
        self.use_moe_balancing = use_moe_balancing and num_experts is not None
        if self.use_moe_balancing:
            assert num_experts is not None, "num_experts must be provided when use_moe_balancing is True"
            self.moe_balancer = AuxiliaryFreeMoEBalancer(
                num_experts=num_experts,
                gradient_balance_weight=gradient_balance_weight
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined DeepSeek-style loss.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            hidden_states: Hidden states for MTP [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            gate_logits: MoE gate logits [batch_size * seq_len, num_experts]
            expert_indices: Selected experts [batch_size * seq_len, top_k]

        Returns:
            Dictionary with all loss components
        """
        losses = {}

        # Compute main loss (temperature-scaled cross-entropy)
        main_loss_dict = self.main_loss(logits, targets, attention_mask)
        losses.update({f'main_{k}': v for k, v in main_loss_dict.items()})
        total_loss = main_loss_dict['loss']

        # Add multi-token prediction loss
        if self.use_mtp and hidden_states is not None:
            mtp_dict = self.mtp_loss(hidden_states, targets, attention_mask)
            losses.update({f'mtp_{k}': v for k, v in mtp_dict.items()})
            total_loss = total_loss + mtp_dict['mtp_loss']

        # Apply MoE balancing (gradient-based, no auxiliary loss)
        if self.use_moe_balancing and gate_logits is not None and expert_indices is not None:
            balance_dict = self.moe_balancer(gate_logits, expert_indices)
            losses.update({f'moe_{k}': v for k, v in balance_dict.items()})
            # FIXED: balancing_term now properly contributes gradients
            total_loss = total_loss + balance_dict['balancing_term']

        # CRITICAL FIX: Add main_loss key that the trainer expects
        # The trainer at enhanced_trainer.py:1895 looks for 'main_loss'
        losses['main_loss'] = main_loss_dict['loss']
        losses['total_loss'] = total_loss

        return losses


# ============================================================================
# ADAPTIVE MTP LOSS
# ============================================================================


class AdaptiveMTPLoss(nn.Module):
    """
    Loss function for Adaptive Multi-Token Prediction.

    Combines:
    - Primary token prediction loss (always full weight)
    - Confidence-weighted additional token losses
    - Regularization to encourage confident predictions
    """

    def __init__(
        self,
        vocab_size: int,
        primary_loss_weight: float = 1.0,
        additional_loss_base_weight: float = 0.1,
        confidence_reg_strength: float = 0.01,
        use_confidence_weighting: bool = True,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Initialize Adaptive MTP Loss.

        Args:
            vocab_size: Size of vocabulary
            primary_loss_weight: Weight for primary token loss (always 1.0)
            additional_loss_base_weight: Base weight for additional tokens before confidence scaling
            confidence_reg_strength: Strength of confidence regularization
            use_confidence_weighting: Whether to weight additional losses by confidence
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation (padding)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.primary_loss_weight = primary_loss_weight
        self.additional_loss_base_weight = additional_loss_base_weight
        self.confidence_reg_strength = confidence_reg_strength
        self.use_confidence_weighting = use_confidence_weighting
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # Track statistics
        self.register_buffer('total_primary_loss', torch.tensor(0.0))
        self.register_buffer('total_additional_loss', torch.tensor(0.0))
        self.register_buffer('total_confidence_reg', torch.tensor(0.0))
        self.register_buffer('loss_count', torch.tensor(0))

    def compute_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional label smoothing.

        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for loss computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        if self.label_smoothing > 0:
            # Label smoothing
            log_probs = F.log_softmax(logits_flat, dim=-1)

            # Create smoothed target distribution
            with torch.no_grad():
                smoothed_targets = torch.zeros_like(log_probs)
                smoothed_targets.fill_(self.label_smoothing / (vocab_size - 1))
                smoothed_targets.scatter_(
                    1,
                    targets_flat.unsqueeze(1),
                    1.0 - self.label_smoothing
                )

                # Mask padding tokens
                if self.ignore_index >= 0:
                    padding_mask = targets_flat == self.ignore_index
                    smoothed_targets[padding_mask] = 0.0

            loss = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.ignore_index,
                reduction='none'
            )

        # Apply attention mask if provided
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum().clamp(min=1.0)
        else:
            if self.ignore_index >= 0:
                # Count non-ignored tokens
                valid_tokens = (targets_flat != self.ignore_index).float()
                loss = loss.sum() / valid_tokens.sum().clamp(min=1.0)
            else:
                loss = loss.mean()

        return loss

    def compute_confidence_regularization(
        self,
        confidence_scores: torch.Tensor,
        target_mode: str = 'binary'
    ) -> torch.Tensor:
        """
        Regularization to encourage confident predictions.

        Pushes confidence scores away from uncertain middle ground (0.5)
        toward either high confidence (1.0) or low confidence (0.0).

        Args:
            confidence_scores: Confidence scores [batch_size, 1] or [batch_size, seq_len, 1]
            target_mode: 'binary' (push to 0 or 1) or 'high' (push toward 1)

        Returns:
            Regularization loss
        """
        if target_mode == 'binary':
            # Penalize scores near 0.5 (uncertain)
            # Loss is minimal at 0 and 1, maximal at 0.5
            # Use: -log(|2*conf - 1|) which is high when conf ≈ 0.5
            epsilon = 1e-7
            deviation_from_half = torch.abs(2 * confidence_scores - 1).clamp(min=epsilon)
            reg_loss = -torch.log(deviation_from_half).mean()

        elif target_mode == 'high':
            # Encourage high confidence
            # Negative log likelihood of confidence
            epsilon = 1e-7
            reg_loss = -torch.log(confidence_scores.clamp(min=epsilon)).mean()

        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

        return reg_loss

    def forward(
        self,
        primary_logits: torch.Tensor,
        targets: torch.Tensor,
        additional_logits: Optional[List[torch.Tensor]] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mtp_active: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute adaptive MTP loss.

        Args:
            primary_logits: Logits for next token [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            additional_logits: List of logits for future positions (if MTP active)
            confidence_scores: Confidence scores [batch_size, 1]
            attention_mask: Attention mask [batch_size, seq_len]
            mtp_active: Whether MTP was activated

        Returns:
            Dictionary containing:
                - loss: Total loss
                - primary_loss: Loss for primary token prediction
                - additional_loss: Loss for additional tokens (if MTP active)
                - confidence_reg: Confidence regularization loss
                - avg_confidence: Average confidence score
                - effective_mtp_weight: Effective weight applied to additional losses
        """
        # Always compute primary token loss
        primary_loss = self.compute_cross_entropy(
            primary_logits, targets, attention_mask
        )
        primary_loss = primary_loss * self.primary_loss_weight

        # Initialize additional loss and regularization
        additional_loss = torch.tensor(0.0, device=primary_logits.device)
        confidence_reg = torch.tensor(0.0, device=primary_logits.device)
        avg_confidence = torch.tensor(0.0, device=primary_logits.device)
        effective_mtp_weight = 0.0

        # Compute additional losses if MTP is active
        if mtp_active and additional_logits is not None:
            num_heads = len(additional_logits)

            # Compute loss for each future position
            head_losses = []
            for i, logits in enumerate(additional_logits):
                # Shift targets for future position (i+1 positions ahead)
                future_offset = i + 1

                # Ensure we have enough positions
                if future_offset >= targets.shape[1]:
                    continue

                # Get shifted targets
                shifted_targets = targets[:, future_offset:]

                # Get corresponding logits (remove last few positions)
                shifted_logits = logits[:, :-future_offset, :]

                # Shifted attention mask
                if attention_mask is not None:
                    shifted_mask = attention_mask[:, future_offset:]
                else:
                    shifted_mask = None

                # Compute loss for this head
                head_loss = self.compute_cross_entropy(
                    shifted_logits, shifted_targets, shifted_mask
                )
                head_losses.append(head_loss)

            # Average losses across heads
            if head_losses:
                additional_loss = torch.stack(head_losses).mean()

                # Apply confidence weighting
                if self.use_confidence_weighting and confidence_scores is not None:
                    avg_confidence = confidence_scores.mean()
                    confidence_weight = avg_confidence.clamp(min=0.0, max=1.0)
                    effective_mtp_weight = (
                        self.additional_loss_base_weight * confidence_weight
                    )
                else:
                    effective_mtp_weight = self.additional_loss_base_weight

                # Weight the additional loss
                additional_loss = additional_loss * effective_mtp_weight

        # Compute confidence regularization
        if confidence_scores is not None and self.confidence_reg_strength > 0:
            confidence_reg = self.compute_confidence_regularization(
                confidence_scores, target_mode='binary'
            )
            confidence_reg = confidence_reg * self.confidence_reg_strength

            if confidence_scores.numel() > 0:
                avg_confidence = confidence_scores.mean()

        # Total loss
        total_loss = primary_loss + additional_loss + confidence_reg

        # Update statistics
        with torch.no_grad():
            self.total_primary_loss += primary_loss.item()
            self.total_additional_loss += additional_loss.item()
            self.total_confidence_reg += confidence_reg.item()
            self.loss_count += 1

        return {
            'loss': total_loss,
            'primary_loss': primary_loss,
            'additional_loss': additional_loss,
            'confidence_reg': confidence_reg,
            'avg_confidence': avg_confidence,
            'effective_mtp_weight': effective_mtp_weight,
            'mtp_active': mtp_active,
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        Get loss statistics for monitoring.

        Returns:
            Dictionary with average losses
        """
        if self.loss_count > 0:
            avg_primary = (self.total_primary_loss / self.loss_count)  # type: ignore[operator]
            avg_additional = (self.total_additional_loss / self.loss_count)  # type: ignore[operator]
            avg_conf_reg = (self.total_confidence_reg / self.loss_count)  # type: ignore[operator]
            # Convert tensors to floats
            avg_primary = avg_primary.item() if isinstance(avg_primary, torch.Tensor) else float(avg_primary)
            avg_additional = avg_additional.item() if isinstance(avg_additional, torch.Tensor) else float(avg_additional)
            avg_conf_reg = avg_conf_reg.item() if isinstance(avg_conf_reg, torch.Tensor) else float(avg_conf_reg)
        else:
            avg_primary = 0.0
            avg_additional = 0.0
            avg_conf_reg = 0.0

        loss_count_val = self.loss_count.item() if isinstance(self.loss_count, torch.Tensor) else int(self.loss_count)
        return {
            'avg_primary_loss': avg_primary,
            'avg_additional_loss': avg_additional,
            'avg_confidence_reg': avg_conf_reg,
            'total_computations': loss_count_val,
        }

    def reset_statistics(self):
        """Reset tracking statistics."""
        self.total_primary_loss.zero_()  # type: ignore[attr-defined]
        self.total_additional_loss.zero_()  # type: ignore[attr-defined]
        self.total_confidence_reg.zero_()  # type: ignore[attr-defined]
        self.loss_count.zero_()  # type: ignore[attr-defined]


# ============================================================================
# REPETITION PENALTY LOSSES
# ============================================================================


class NGramRepetitionPenalty(nn.Module):
    """
    Penalizes n-gram repetitions in generated sequences.

    This helps prevent mode collapse where models learn to repeat
    the same tokens/phrases instead of generating diverse text.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        penalty_weight: float = 0.1,
        vocab_size: int = 50257,
        ignore_index: int = -100
    ):
        """
        Initialize n-gram repetition penalty.

        Args:
            ngram_size: Size of n-grams to track (2-4 recommended)
            penalty_weight: Weight of penalty loss (0.01-0.1)
            vocab_size: Vocabulary size
            ignore_index: Token ID to ignore (padding)
        """
        super().__init__()
        self.ngram_size = ngram_size
        self.penalty_weight = penalty_weight
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def compute_ngram_repetition_penalty(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute penalty for repeated n-grams.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Penalty value (higher = more repetition)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        if seq_len < self.ngram_size:
            return torch.tensor(0.0, device=device)

        # Track unique n-grams and repetition counts
        total_penalty = torch.tensor(0.0, device=device)
        total_ngrams = 0

        for b in range(batch_size):
            sequence = token_ids[b]
            mask = attention_mask[b] if attention_mask is not None else None

            # Extract n-grams
            ngrams = []
            for i in range(seq_len - self.ngram_size + 1):
                # Skip if any token in ngram is masked
                if mask is not None and mask[i:i+self.ngram_size].sum() < self.ngram_size:
                    continue

                ngram = tuple(sequence[i:i+self.ngram_size].tolist())
                ngrams.append(ngram)

            if len(ngrams) == 0:
                continue

            # Count repetitions
            ngram_counts = Counter(ngrams)

            # Penalize repeated n-grams
            for ngram, count in ngram_counts.items():
                if count > 1:
                    # Penalty increases with repetition count
                    # count=2: penalty=1, count=3: penalty=3, count=4: penalty=6
                    repetition_penalty = (count - 1) * count / 2
                    total_penalty = total_penalty + repetition_penalty

            total_ngrams += len(ngrams)

        # Normalize by number of n-grams
        if total_ngrams > 0:
            normalized_penalty = total_penalty / total_ngrams
        else:
            normalized_penalty = torch.tensor(0.0, device=device)

        return normalized_penalty * self.penalty_weight

    def compute_token_diversity_penalty(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Penalize low entropy (non-diverse) predictions.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            temperature: Temperature for softmax

        Returns:
            Diversity penalty (lower entropy = higher penalty)
        """
        # Apply temperature
        scaled_logits = logits / temperature

        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Compute entropy (higher entropy = more diverse)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Normalize entropy (max entropy = log(vocab_size))
        max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy

        # Penalty for low entropy (1 - entropy)
        # High entropy (diverse) = low penalty
        # Low entropy (repetitive) = high penalty
        diversity_penalty = (1.0 - normalized_entropy).mean()

        return diversity_penalty * self.penalty_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Compute repetition penalties.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with penalty losses
        """
        # Compute n-gram repetition penalty on targets
        ngram_penalty = self.compute_ngram_repetition_penalty(
            targets, attention_mask
        )

        # Compute diversity penalty on predictions
        diversity_penalty = self.compute_token_diversity_penalty(logits)

        # Total penalty
        total_penalty = ngram_penalty + diversity_penalty

        return {
            'ngram_penalty': ngram_penalty,
            'diversity_penalty': diversity_penalty,
            'total_repetition_penalty': total_penalty,
            'penalty_weight': self.penalty_weight
        }


class SequenceRepetitionDetector(nn.Module):
    """
    Detects and heavily penalizes immediate token repetition.

    This is a stronger version that specifically targets the
    "time time time" style repetition we're seeing.
    """

    def __init__(
        self,
        penalty_weight: float = 1.0,
        max_repeat_length: int = 10
    ):
        """
        Initialize sequence repetition detector.

        Args:
            penalty_weight: Weight of penalty (higher = stronger)
            max_repeat_length: Maximum repetition sequence to detect
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self.max_repeat_length = max_repeat_length

    def detect_immediate_repetition(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Detect immediate token repetition (e.g., "time time time").

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Penalty value
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        total_penalty = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            sequence = token_ids[b]
            mask = attention_mask[b] if attention_mask is not None else None

            consecutive_count = 1
            max_consecutive = 1

            for i in range(1, seq_len):
                # Skip masked tokens
                if mask is not None and mask[i] == 0:
                    continue

                # Check if same as previous token
                if sequence[i] == sequence[i-1]:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1

            # Heavy penalty for consecutive repetitions
            # 2 repeats: penalty=1, 3 repeats: penalty=4, 4 repeats: penalty=9
            if max_consecutive > 1:
                repetition_penalty = (max_consecutive - 1) ** 2
                total_penalty = total_penalty + repetition_penalty

        # Average over batch
        normalized_penalty = total_penalty / batch_size

        return normalized_penalty * self.penalty_weight

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Detect and penalize sequence repetition.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with repetition penalties
        """
        immediate_penalty = self.detect_immediate_repetition(
            token_ids, attention_mask
        )

        return {
            'immediate_repetition_penalty': immediate_penalty,
            'repetition_detector_weight': self.penalty_weight
        }


# ============================================================================
# ANTI-REPETITION LOSSES
# ============================================================================


class AntiRepetitionLoss(nn.Module):
    """
    Enhanced loss function that penalizes repetitive outputs.

    Combines:
    - Base cross-entropy loss
    - N-gram repetition penalty
    - EOS token over-use penalty
    - Diversity bonus
    """

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        pad_token_id: int,
        repetition_penalty_weight: float = 0.1,
        eos_penalty_weight: float = 0.05,
        diversity_bonus_weight: float = 0.05,
        ngram_size: int = 4,
        ignore_index: int = -100
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            eos_token_id: ID of EOS token to penalize
            pad_token_id: ID of padding token (ignored in calculations)
            repetition_penalty_weight: Weight for n-gram repetition penalty (0.1 = 10%)
            eos_penalty_weight: Weight for EOS over-use penalty (0.05 = 5%)
            diversity_bonus_weight: Weight for diversity bonus (0.05 = 5%)
            ngram_size: Size of n-grams to check (default: 4)
            ignore_index: Token ID to ignore in loss calculation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.repetition_penalty_weight = repetition_penalty_weight
        self.eos_penalty_weight = eos_penalty_weight
        self.diversity_bonus_weight = diversity_bonus_weight
        self.ngram_size = ngram_size
        self.ignore_index = ignore_index

        # Base loss
        self.base_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def calculate_ngram_repetition(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate n-gram repetition ratio for each sequence.

        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)

        Returns:
            repetition_scores: [batch_size] - ratio of repeated n-grams
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        if seq_len < self.ngram_size:
            return torch.zeros(batch_size, device=device)

        repetition_scores = []

        for i in range(batch_size):
            tokens = token_ids[i]

            # Apply attention mask if provided
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                tokens = tokens[:valid_len]

            # Skip if too short
            if len(tokens) < self.ngram_size:
                repetition_scores.append(0.0)
                continue

            # Extract n-grams
            ngrams = []
            for j in range(len(tokens) - self.ngram_size + 1):
                ngram = tuple(tokens[j:j+self.ngram_size].tolist())
                # Skip if contains padding
                if self.pad_token_id not in ngram:
                    ngrams.append(ngram)

            if len(ngrams) == 0:
                repetition_scores.append(0.0)
                continue

            # Calculate repetition ratio
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            repetition = 1.0 - (unique_ngrams / total_ngrams)
            repetition_scores.append(repetition)

        return torch.tensor(repetition_scores, device=device)

    def calculate_eos_penalty(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate penalty for excessive EOS token usage.

        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)

        Returns:
            eos_penalties: [batch_size] - ratio of EOS tokens
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device

        eos_ratios = []

        for i in range(batch_size):
            tokens = token_ids[i]

            # Apply attention mask if provided
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                tokens = tokens[:valid_len]

            if len(tokens) == 0:
                eos_ratios.append(0.0)
                continue

            # Count EOS tokens (excluding the final legitimate one)
            eos_count = (tokens == self.eos_token_id).sum().item()

            # If EOS appears, it should ideally be only at the end
            # Penalize if it appears multiple times or early
            if eos_count > 1:
                # Multiple EOS tokens = problem
                penalty = eos_count / len(tokens)
            elif eos_count == 1 and tokens[-1] != self.eos_token_id:
                # EOS in middle of sequence = problem
                penalty = 0.5 / len(tokens)
            else:
                # Normal case: single EOS at end (or no EOS)
                penalty = 0.0

            eos_ratios.append(penalty)

        return torch.tensor(eos_ratios, device=device)

    def calculate_diversity_bonus(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate diversity bonus (negative penalty) for diverse outputs.
        Uses unique token ratio as a simple diversity measure.

        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)

        Returns:
            diversity_scores: [batch_size] - higher = more diverse (bonus)
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device

        diversity_scores = []

        for i in range(batch_size):
            tokens = token_ids[i]

            # Apply attention mask if provided
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
                tokens = tokens[:valid_len]

            if len(tokens) == 0:
                diversity_scores.append(0.0)
                continue

            # Filter out special tokens
            valid_tokens = tokens[
                (tokens != self.pad_token_id) &
                (tokens != self.eos_token_id)
            ]

            if len(valid_tokens) == 0:
                diversity_scores.append(0.0)
                continue

            # Diversity = ratio of unique tokens
            unique_tokens = len(torch.unique(valid_tokens))
            diversity = unique_tokens / len(valid_tokens)
            diversity_scores.append(diversity)

        return torch.tensor(diversity_scores, device=device)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Calculate combined loss with anti-repetition penalties.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)
            return_components: If True, return dict with loss components

        Returns:
            loss: Scalar loss
            components: Dict with loss components (if return_components=True)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 1. Base cross-entropy loss
        base_loss_per_token = self.base_loss(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )
        base_loss_per_token = base_loss_per_token.view(batch_size, seq_len)

        # Average over valid tokens
        if attention_mask is not None:
            # Mask out padding
            mask = (labels != self.ignore_index).float()
            base_loss = (base_loss_per_token * mask).sum() / mask.sum()
        else:
            base_loss = base_loss_per_token.mean()

        # 2. CRITICAL FIX: Use ground truth labels for penalty calculation, not predictions
        # Using predictions during early training is meaningless since they're random
        # We want to penalize repetitive patterns in the training data itself

        # 3. Calculate penalties on LABELS (ground truth)
        repetition_scores = self.calculate_ngram_repetition(
            labels, attention_mask
        )
        repetition_penalty = repetition_scores.mean()

        eos_penalties = self.calculate_eos_penalty(
            labels, attention_mask
        )
        eos_penalty = eos_penalties.mean()

        diversity_scores = self.calculate_diversity_bonus(
            labels, attention_mask
        )
        diversity_bonus = diversity_scores.mean()

        # 4. Combined loss
        # Base loss + repetition penalty + EOS penalty - diversity bonus
        total_loss = (
            base_loss +
            self.repetition_penalty_weight * repetition_penalty +
            self.eos_penalty_weight * eos_penalty -
            self.diversity_bonus_weight * diversity_bonus
        )

        if return_components:
            components = {
                'base_loss': base_loss.item(),
                'repetition_penalty': repetition_penalty.item(),
                'eos_penalty': eos_penalty.item(),
                'diversity_bonus': diversity_bonus.item(),
                'total_loss': total_loss.item()
            }
            return total_loss, components

        return total_loss, None


class AdaptiveAntiRepetitionLoss(AntiRepetitionLoss):
    """
    Adaptive version that adjusts penalty weights based on training progress.

    Starts with high penalties and gradually reduces them as model improves.
    """

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        pad_token_id: int,
        initial_repetition_weight: float = 0.2,  # Start high
        final_repetition_weight: float = 0.05,    # End low
        initial_eos_weight: float = 0.1,
        final_eos_weight: float = 0.02,
        warmup_steps: int = 10000,
        **kwargs
    ):
        """
        Args:
            warmup_steps: Number of steps to linearly reduce penalties
            Other args same as AntiRepetitionLoss
        """
        super().__init__(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty_weight=initial_repetition_weight,
            eos_penalty_weight=initial_eos_weight,
            **kwargs
        )

        self.initial_repetition_weight = initial_repetition_weight
        self.final_repetition_weight = final_repetition_weight
        self.initial_eos_weight = initial_eos_weight
        self.final_eos_weight = final_eos_weight
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def update_weights(self, step: int):
        """Update penalty weights based on training step."""
        self.current_step = step

        if step >= self.warmup_steps:
            # Use final weights
            self.repetition_penalty_weight = self.final_repetition_weight
            self.eos_penalty_weight = self.final_eos_weight
        else:
            # Linear interpolation
            progress = step / self.warmup_steps

            self.repetition_penalty_weight = (
                self.initial_repetition_weight * (1 - progress) +
                self.final_repetition_weight * progress
            )

            self.eos_penalty_weight = (
                self.initial_eos_weight * (1 - progress) +
                self.final_eos_weight * progress
            )


# ============================================================================
# ADVANCED LOSSES
# ============================================================================


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning representations by contrasting positive and negative pairs.

    This implementation supports both standard contrastive loss and InfoNCE-style loss.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        loss_type: str = "infonce",
        normalize_embeddings: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.loss_type = loss_type.lower()
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Embeddings tensor [batch_size, embedding_dim]
            labels: Labels for creating positive/negative pairs [batch_size]
            positive_pairs: Explicit positive pairs [num_pairs, 2]
            negative_pairs: Explicit negative pairs [num_pairs, 2]

        Returns:
            Contrastive loss value
        """
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        if self.loss_type == "infonce":
            return self._infonce_loss(embeddings, labels, positive_pairs)
        elif self.loss_type == "standard":
            return self._standard_contrastive_loss(embeddings, labels, positive_pairs, negative_pairs)
        elif self.loss_type == "triplet":
            if labels is None:
                raise ValueError("Labels required for triplet loss")
            return self._triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _infonce_loss(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """InfoNCE loss implementation."""
        batch_size = embeddings.shape[0]

        if positive_pairs is not None:
            # Use explicit positive pairs
            anchor_indices = positive_pairs[:, 0]
            positive_indices = positive_pairs[:, 1]
            anchor_embeds = embeddings[anchor_indices]
            positive_embeds = embeddings[positive_indices]
        elif labels is not None:
            # Create positive pairs from labels
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask = mask.float() - torch.eye(batch_size, device=mask.device)

            # For simplicity, use consecutive augmentations as positive pairs
            anchor_embeds = embeddings[::2]  # Even indices
            positive_embeds = embeddings[1::2]  # Odd indices
            batch_size = min(anchor_embeds.shape[0], positive_embeds.shape[0])
            anchor_embeds = anchor_embeds[:batch_size]
            positive_embeds = positive_embeds[:batch_size]
        else:
            # Assume first half are anchors, second half are positives
            mid_point = batch_size // 2
            anchor_embeds = embeddings[:mid_point]
            positive_embeds = embeddings[mid_point:mid_point*2]

        # Compute similarities
        positive_sim = F.cosine_similarity(anchor_embeds, positive_embeds, dim=-1) / self.temperature

        # Compute similarities with all negatives
        all_similarities = torch.matmul(anchor_embeds, embeddings.T) / self.temperature

        # Create labels (positive indices)
        if positive_pairs is not None:
            pos_labels = positive_pairs[:len(anchor_embeds), 1]
        else:
            pos_labels = torch.arange(len(anchor_embeds), len(anchor_embeds)*2, device=embeddings.device)

        # InfoNCE loss
        loss = F.cross_entropy(all_similarities, pos_labels)
        return loss

    def _standard_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard contrastive loss with margin."""
        if positive_pairs is None or negative_pairs is None:
            # Generate pairs from labels
            if labels is None:
                raise ValueError("Either pairs or labels must be provided")
            positive_pairs, negative_pairs = self._generate_pairs_from_labels(labels)

        # At this point, pairs are guaranteed to be tensors
        pos_pairs: torch.Tensor = positive_pairs
        neg_pairs: torch.Tensor = negative_pairs

        # Compute distances for positive pairs
        pos_distances = torch.norm(
            embeddings[pos_pairs[:, 0]] - embeddings[pos_pairs[:, 1]],
            p=2, dim=-1
        )

        # Compute distances for negative pairs
        neg_distances = torch.norm(
            embeddings[neg_pairs[:, 0]] - embeddings[neg_pairs[:, 1]],
            p=2, dim=-1
        )

        # Contrastive loss
        pos_loss = pos_distances.pow(2)
        neg_loss = F.relu(self.margin - neg_distances).pow(2)

        return (pos_loss.mean() + neg_loss.mean()) / 2

    def _triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Triplet loss implementation."""
        return F.triplet_margin_loss(
            anchor=embeddings,
            positive=embeddings,  # Simplified - would need proper positive mining
            negative=embeddings,  # Simplified - would need proper negative mining
            margin=self.margin
        )

    def _generate_pairs_from_labels(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate positive and negative pairs from labels."""
        batch_size = labels.shape[0]
        positive_pairs = []
        negative_pairs = []

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if labels[i] == labels[j]:
                    positive_pairs.append([i, j])
                else:
                    negative_pairs.append([i, j])

        positive_pairs = torch.tensor(positive_pairs, device=labels.device)
        negative_pairs = torch.tensor(negative_pairs, device=labels.device)

        return positive_pairs, negative_pairs


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples.
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


class DiversityLoss(nn.Module):
    """
    Diversity loss to encourage diverse representations in MoE models.

    This loss encourages different experts to learn diverse representations.
    """

    def __init__(self, similarity_metric: str = "cosine", diversity_weight: float = 1.0):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.diversity_weight = diversity_weight

    def forward(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute diversity loss across expert outputs.

        Args:
            expert_outputs: List of expert outputs [batch_size, hidden_dim]

        Returns:
            Diversity loss encouraging different experts to be different
        """
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        device = expert_outputs[0].device
        diversity_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                expert_i = expert_outputs[i]
                expert_j = expert_outputs[j]

                if self.similarity_metric == "cosine":
                    # Cosine similarity - we want this to be low (diverse)
                    similarity = F.cosine_similarity(expert_i, expert_j, dim=-1).mean()
                elif self.similarity_metric == "l2":
                    # L2 distance - we want experts to be far apart
                    distance = torch.norm(expert_i - expert_j, p=2, dim=-1).mean()
                    similarity = 1.0 / (1.0 + distance)  # Convert to similarity (high=bad)
                else:
                    # Dot product similarity
                    similarity = (expert_i * expert_j).sum(dim=-1).mean()

                diversity_loss += similarity
                num_pairs += 1

        if num_pairs > 0:
            return self.diversity_weight * diversity_loss / num_pairs
        else:
            device = expert_outputs[0].device if expert_outputs else 'cpu'
            return torch.tensor(0.0, device=device)


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for MoE routing and other auxiliary objectives.

    This implements various auxiliary losses commonly used in MoE models.
    """

    def __init__(
        self,
        load_balancing_weight: float = 0.0001,
        router_z_weight: float = 0.001,
        expert_diversity_weight: float = 0.0001
    ):
        super().__init__()
        self.load_balancing_weight = load_balancing_weight
        self.router_z_weight = router_z_weight
        self.expert_diversity_weight = expert_diversity_weight

    def load_balancing_loss(
        self,
        gate_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int
    ) -> torch.Tensor:
        """
        Load balancing loss to encourage uniform expert usage.

        Args:
            gate_logits: Router logits [batch_size * seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size * seq_len, top_k]
            num_experts: Total number of experts

        Returns:
            Load balancing loss
        """
        # Gate probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Expert usage frequency
        expert_mask = F.one_hot(expert_indices, num_experts).float()
        expert_usage = expert_mask.sum(dim=0).sum(dim=0)

        # Gate probability sums
        gate_prob_sums = gate_probs.sum(dim=0)

        # Load balancing loss (CV^2 - coefficient of variation squared)
        total_tokens = gate_logits.shape[0] * expert_indices.shape[1]
        load_loss = num_experts * torch.sum(gate_prob_sums * expert_usage) / (total_tokens ** 2)

        # Cap load balancing loss to prevent runaway values
        load_loss = torch.clamp(load_loss, max=10.0)

        return load_loss

    def router_z_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Router Z-loss for numerical stability.

        Args:
            gate_logits: Router logits [batch_size * seq_len, num_experts]

        Returns:
            Router Z-loss
        """
        # Clip gate logits to prevent extreme values
        gate_logits = torch.clamp(gate_logits, min=-10.0, max=10.0)

        # Z-loss encourages smaller logits to prevent overflow
        logsumexp_vals = torch.logsumexp(gate_logits, dim=-1)
        z_loss = torch.mean(logsumexp_vals ** 2)

        # Cap the z-loss to prevent runaway values
        z_loss = torch.clamp(z_loss, max=100.0)

        return z_loss

    def expert_diversity_loss(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Expert diversity loss to encourage specialization.

        Args:
            expert_outputs: List of expert outputs

        Returns:
            Expert diversity loss
        """
        diversity_loss_fn = DiversityLoss()
        return diversity_loss_fn(expert_outputs)

    def forward(
        self,
        gate_logits: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        num_experts: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses.

        Returns:
            Dictionary of auxiliary losses
        """
        losses = {}

        if gate_logits is not None and expert_indices is not None and num_experts is not None:
            losses['load_balancing'] = self.load_balancing_weight * self.load_balancing_loss(
                gate_logits, expert_indices, num_experts
            )

        if gate_logits is not None:
            losses['router_z'] = self.router_z_weight * self.router_z_loss(gate_logits)

        if expert_outputs is not None:
            losses['expert_diversity'] = self.expert_diversity_weight * self.expert_diversity_loss(expert_outputs)

        return losses


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for semi-supervised learning and augmentation consistency.

    Encourages the model to produce consistent predictions for augmented versions
    of the same input.
    """

    def __init__(
        self,
        consistency_type: str = "mse",
        temperature: float = 1.0,
        threshold: float = 0.95
    ):
        super().__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
        self.threshold = threshold

    def forward(
        self,
        outputs_original: torch.Tensor,
        outputs_augmented: torch.Tensor,
        confidence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute consistency loss between original and augmented outputs.

        Args:
            outputs_original: Outputs for original inputs
            outputs_augmented: Outputs for augmented inputs
            confidence_mask: Optional mask for high-confidence predictions

        Returns:
            Consistency loss
        """
        if self.consistency_type == "mse":
            # MSE between softmax outputs
            probs_orig = F.softmax(outputs_original / self.temperature, dim=-1)
            probs_aug = F.softmax(outputs_augmented / self.temperature, dim=-1)
            consistency_loss = F.mse_loss(probs_aug, probs_orig, reduction='none').mean(dim=-1)

        elif self.consistency_type == "kl":
            # KL divergence
            log_probs_orig = F.log_softmax(outputs_original / self.temperature, dim=-1)
            probs_aug = F.softmax(outputs_augmented / self.temperature, dim=-1)
            consistency_loss = F.kl_div(log_probs_orig, probs_aug, reduction='none').sum(dim=-1)

        elif self.consistency_type == "ce":
            # Cross-entropy with original as pseudo-labels
            pseudo_labels = torch.argmax(outputs_original, dim=-1)
            consistency_loss = F.cross_entropy(outputs_augmented, pseudo_labels, reduction='none')

        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")

        # Apply confidence mask if provided
        if confidence_mask is not None:
            consistency_loss = consistency_loss * confidence_mask

        return consistency_loss.mean()


class PerplexityLoss(nn.Module):
    """
    Perplexity-based loss for language modeling evaluation.

    This loss computes perplexity and can be used as an auxiliary loss.
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
    """

    def __init__(self, loss_config: Dict[str, Dict]):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

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

            except Exception as e:
                # Skip losses that can't be computed with available inputs
                continue

        computed_losses['total'] = total_loss
        return computed_losses


# ============================================================================
# UNIFIED LOSS INTERFACE
# ============================================================================


class UnifiedLoss(nn.Module):
    """
    Unified loss function combining all available loss components.

    This is the main loss class for the Ava AI training pipeline, providing
    a single interface to all loss functions with flexible configuration.

    Features:
    - Temperature-scaled cross-entropy with adaptive temperature
    - Multi-token prediction (both DeepSeek-style and Adaptive MTP)
    - N-gram repetition penalties
    - Immediate repetition detection
    - EOS token penalties
    - MoE load balancing (auxiliary-free)
    - Diversity and contrastive losses
    - Label smoothing
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: Optional[int] = None,
        # Primary loss settings
        primary_loss_type: str = "deepseek",  # "deepseek", "adaptive_mtp", "standard"
        ignore_index: int = -100,
        # Temperature scaling
        initial_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        label_smoothing: float = 0.1,
        # Multi-token prediction
        use_mtp: bool = False,
        num_future_tokens: int = 3,
        mtp_weight: float = 0.1,
        mtp_type: str = "deepseek",  # "deepseek" or "adaptive"
        # Adaptive MTP specific
        use_confidence_weighting: bool = True,
        confidence_reg_strength: float = 0.01,
        # Repetition penalties
        use_ngram_penalty: bool = True,
        ngram_size: int = 4,
        ngram_penalty_weight: float = 0.1,
        use_immediate_repetition_penalty: bool = True,
        immediate_repetition_weight: float = 0.5,
        # EOS penalties
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        min_sequence_length: int = 20,
        eos_penalty_weight: float = 0.05,
        # MoE balancing
        num_experts: Optional[int] = None,
        use_moe_balancing: bool = False,
        gradient_balance_weight: float = 0.1,
        # Advanced losses
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        use_diversity_loss: bool = False,
        diversity_weight: float = 0.01,
        # Auxiliary losses
        use_auxiliary_loss: bool = False,
        load_balancing_weight: float = 0.0001,
        router_z_weight: float = 0.001,
    ):
        """
        Initialize unified loss function.

        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension (required for MTP)
            primary_loss_type: Type of primary loss ("deepseek", "adaptive_mtp", "standard")
            ignore_index: Token ID to ignore in loss calculation
            initial_temperature: Starting temperature for scaling
            adaptive_temperature: Whether to adapt temperature during training
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            use_mtp: Whether to use multi-token prediction
            num_future_tokens: Number of future tokens to predict
            mtp_weight: Weight for MTP loss
            mtp_type: Type of MTP ("deepseek" or "adaptive")
            use_confidence_weighting: Whether to weight MTP by confidence
            confidence_reg_strength: Strength of confidence regularization
            use_ngram_penalty: Whether to penalize n-gram repetitions
            ngram_size: Size of n-grams to track
            ngram_penalty_weight: Weight for n-gram penalty
            use_immediate_repetition_penalty: Whether to penalize immediate repetition
            immediate_repetition_weight: Weight for immediate repetition penalty
            eos_token_id: EOS token ID for penalties
            pad_token_id: Padding token ID
            min_sequence_length: Minimum length before allowing EOS
            eos_penalty_weight: Weight for EOS penalty
            num_experts: Number of experts (for MoE)
            use_moe_balancing: Whether to use MoE balancing
            gradient_balance_weight: Weight for gradient-based balancing
            use_focal_loss: Whether to use focal loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            use_diversity_loss: Whether to use diversity loss
            diversity_weight: Weight for diversity loss
            use_auxiliary_loss: Whether to use auxiliary loss
            load_balancing_weight: Weight for load balancing loss
            router_z_weight: Weight for router z-loss
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.primary_loss_type = primary_loss_type
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Initialize primary loss based on type
        if primary_loss_type == "deepseek":
            # DeepSeek-style loss with temperature scaling
            self.main_loss = TemperatureScaledCrossEntropy(
                initial_temperature=initial_temperature,
                adaptive_temperature=adaptive_temperature,
                label_smoothing=label_smoothing,
                vocab_size=vocab_size,
                eos_token_id=eos_token_id,
                min_sequence_length=min_sequence_length,
                eos_penalty_weight=eos_penalty_weight
            )
        elif primary_loss_type == "adaptive_mtp":
            # Adaptive MTP as primary loss
            if hidden_size is None:
                raise ValueError("hidden_size required for adaptive_mtp loss")
            self.main_loss = AdaptiveMTPLoss(
                vocab_size=vocab_size,
                primary_loss_weight=1.0,
                additional_loss_base_weight=mtp_weight,
                confidence_reg_strength=confidence_reg_strength,
                use_confidence_weighting=use_confidence_weighting,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index
            )
        else:
            # Standard cross-entropy with label smoothing
            if label_smoothing > 0:
                self.main_loss = LabelSmoothingLoss(
                    num_classes=vocab_size,
                    smoothing=label_smoothing
                )
            else:
                self.main_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Multi-token prediction
        self.use_mtp = use_mtp
        self.mtp_type = mtp_type
        if use_mtp:
            if hidden_size is None:
                raise ValueError("hidden_size required for multi-token prediction")

            if mtp_type == "deepseek":
                self.mtp_loss = MultiTokenPredictionLoss(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_future_tokens=num_future_tokens,
                    mtp_weight=mtp_weight
                )
            elif mtp_type == "adaptive":
                self.mtp_loss = AdaptiveMTPLoss(
                    vocab_size=vocab_size,
                    primary_loss_weight=0.0,  # Only use MTP component
                    additional_loss_base_weight=mtp_weight,
                    confidence_reg_strength=confidence_reg_strength,
                    use_confidence_weighting=use_confidence_weighting,
                    label_smoothing=label_smoothing,
                    ignore_index=ignore_index
                )

        # N-gram repetition penalty
        self.use_ngram_penalty = use_ngram_penalty
        if use_ngram_penalty:
            self.ngram_penalty = NGramRepetitionPenalty(
                ngram_size=ngram_size,
                penalty_weight=ngram_penalty_weight,
                vocab_size=vocab_size,
                ignore_index=ignore_index
            )

        # Immediate repetition detector
        self.use_immediate_repetition_penalty = use_immediate_repetition_penalty
        if use_immediate_repetition_penalty:
            self.immediate_repetition_detector = SequenceRepetitionDetector(
                penalty_weight=immediate_repetition_weight
            )

        # MoE load balancing
        self.use_moe_balancing = use_moe_balancing and num_experts is not None
        if self.use_moe_balancing:
            assert num_experts is not None
            self.moe_balancer = AuxiliaryFreeMoEBalancer(
                num_experts=num_experts,
                gradient_balance_weight=gradient_balance_weight
            )

        # Focal loss
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma
            )

        # Diversity loss
        self.use_diversity_loss = use_diversity_loss
        if use_diversity_loss:
            self.diversity_loss = DiversityLoss(
                diversity_weight=diversity_weight
            )

        # Auxiliary loss
        self.use_auxiliary_loss = use_auxiliary_loss
        if use_auxiliary_loss:
            self.auxiliary_loss = AuxiliaryLoss(
                load_balancing_weight=load_balancing_weight,
                router_z_weight=router_z_weight
            )

    def compute_main_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        primary_logits: Optional[torch.Tensor] = None,
        additional_logits: Optional[List[torch.Tensor]] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        mtp_active: bool = False
    ) -> Dict[str, Any]:
        """Compute primary loss based on configured type."""
        if self.primary_loss_type == "deepseek":
            # DeepSeek-style temperature-scaled cross-entropy
            result = self.main_loss(logits, targets, attention_mask)
            return {
                'loss': result['loss'],
                'temperature': result.get('temperature', 1.0)
            }
        elif self.primary_loss_type == "adaptive_mtp":
            # Adaptive MTP as primary loss
            if primary_logits is None:
                primary_logits = logits
            result = self.main_loss(
                primary_logits=primary_logits,
                targets=targets,
                additional_logits=additional_logits,
                confidence_scores=confidence_scores,
                attention_mask=attention_mask,
                mtp_active=mtp_active
            )
            return result
        else:
            # Standard cross-entropy or label smoothing
            if isinstance(self.main_loss, LabelSmoothingLoss):
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss = self.main_loss(logits_flat, targets_flat)
            else:
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss = self.main_loss(logits_flat, targets_flat)

            return {'loss': loss}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        # Adaptive MTP specific
        primary_logits: Optional[torch.Tensor] = None,
        additional_logits: Optional[List[torch.Tensor]] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        mtp_active: bool = False,
        # MoE specific
        gate_logits: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        # Control flags
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Compute unified loss with all configured components.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            hidden_states: Hidden states for MTP [batch_size, seq_len, hidden_size]
            primary_logits: Primary token logits (for Adaptive MTP)
            additional_logits: Additional token logits (for Adaptive MTP)
            confidence_scores: Confidence scores (for Adaptive MTP)
            mtp_active: Whether MTP was activated
            gate_logits: MoE gate logits [batch_size * seq_len, num_experts]
            expert_indices: Selected experts [batch_size * seq_len, top_k]
            expert_outputs: Expert outputs for diversity loss
            return_detailed: Whether to return detailed loss breakdown

        Returns:
            If return_detailed=False: Total loss tensor
            If return_detailed=True: Dictionary with all loss components
        """
        losses = {}

        # 1. Compute main loss
        main_result = self.compute_main_loss(
            logits=logits,
            targets=targets,
            attention_mask=attention_mask,
            primary_logits=primary_logits,
            additional_logits=additional_logits,
            confidence_scores=confidence_scores,
            mtp_active=mtp_active
        )

        losses['main_loss'] = main_result['loss']
        total_loss = main_result['loss']

        # Store additional main loss info
        for key, value in main_result.items():
            if key != 'loss':
                losses[f'main_{key}'] = value

        # 2. Add multi-token prediction loss (if not already primary)
        if self.use_mtp and self.primary_loss_type != "adaptive_mtp":
            if hidden_states is None:
                losses['mtp_warning'] = "MTP enabled but hidden_states not provided"
            else:
                if self.mtp_type == "deepseek":
                    mtp_result = self.mtp_loss(hidden_states, targets, attention_mask)
                    losses['mtp_loss'] = mtp_result['mtp_loss']
                    total_loss = total_loss + mtp_result['mtp_loss']
                    losses['mtp_per_token_losses'] = mtp_result['per_token_losses']
                elif self.mtp_type == "adaptive":
                    if primary_logits is not None and additional_logits is not None:
                        mtp_result = self.mtp_loss(
                            primary_logits=primary_logits,
                            targets=targets,
                            additional_logits=additional_logits,
                            confidence_scores=confidence_scores,
                            attention_mask=attention_mask,
                            mtp_active=mtp_active
                        )
                        losses['adaptive_mtp_loss'] = mtp_result['loss']
                        total_loss = total_loss + mtp_result['loss']
                        losses.update({f'adaptive_mtp_{k}': v for k, v in mtp_result.items() if k != 'loss'})

        # 3. Add n-gram repetition penalty
        if self.use_ngram_penalty:
            ngram_result = self.ngram_penalty(logits, targets, attention_mask)
            losses['ngram_penalty'] = ngram_result['total_repetition_penalty']
            total_loss = total_loss + ngram_result['total_repetition_penalty']
            losses['ngram_penalty_breakdown'] = {
                'ngram': ngram_result['ngram_penalty'],
                'diversity': ngram_result['diversity_penalty']
            }

        # 4. Add immediate repetition penalty
        if self.use_immediate_repetition_penalty:
            immediate_result = self.immediate_repetition_detector(targets, attention_mask)
            losses['immediate_repetition_penalty'] = immediate_result['immediate_repetition_penalty']
            total_loss = total_loss + immediate_result['immediate_repetition_penalty']

        # 5. Add MoE balancing
        if self.use_moe_balancing and gate_logits is not None and expert_indices is not None:
            balance_result = self.moe_balancer(gate_logits, expert_indices, expert_outputs)
            losses['moe_balancing_term'] = balance_result['balancing_term']
            total_loss = total_loss + balance_result['balancing_term']
            losses['moe_balance_loss'] = balance_result['balance_loss']
            losses['moe_expert_balance_ratio'] = balance_result['expert_balance_ratio']
            losses['moe_cv'] = balance_result['coefficient_of_variation']

        # 6. Add focal loss
        if self.use_focal_loss:
            focal_result = self.focal_loss(logits, targets)
            losses['focal_loss'] = focal_result
            # Focal loss replaces main loss, don't add to total

        # 7. Add diversity loss
        if self.use_diversity_loss and expert_outputs is not None:
            diversity_result = self.diversity_loss(expert_outputs)
            losses['diversity_loss'] = diversity_result
            total_loss = total_loss + diversity_result

        # 8. Add auxiliary loss
        if self.use_auxiliary_loss:
            aux_result = self.auxiliary_loss(
                gate_logits=gate_logits,
                expert_indices=expert_indices,
                expert_outputs=expert_outputs,
                num_experts=self.moe_balancer.num_experts if self.use_moe_balancing else None
            )
            for key, value in aux_result.items():
                losses[f'aux_{key}'] = value
                total_loss = total_loss + value

        # Store total loss
        losses['total_loss'] = total_loss

        if return_detailed:
            return losses
        else:
            return total_loss

    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get statistics from all loss components."""
        stats = {}

        # Adaptive MTP statistics
        if self.primary_loss_type == "adaptive_mtp":
            stats['adaptive_mtp'] = self.main_loss.get_statistics()
        elif self.use_mtp and self.mtp_type == "adaptive":
            stats['adaptive_mtp'] = self.mtp_loss.get_statistics()

        # MoE balancing statistics
        if self.use_moe_balancing:
            stats['moe_balancing'] = {
                'expert_counts': self.moe_balancer.expert_counts.tolist(),
                'expert_scores': self.moe_balancer.expert_scores.tolist(),
                'total_tokens': self.moe_balancer.total_tokens.item()
            }

        return stats

    def reset_statistics(self):
        """Reset statistics in all loss components."""
        if self.primary_loss_type == "adaptive_mtp":
            self.main_loss.reset_statistics()
        elif self.use_mtp and self.mtp_type == "adaptive":
            self.mtp_loss.reset_statistics()


# Convenience function for creating unified loss
def create_unified_loss(config: Dict[str, Any]) -> UnifiedLoss:
    """
    Create a unified loss from a configuration dictionary.

    Args:
        config: Configuration dictionary with loss parameters

    Returns:
        Configured UnifiedLoss instance
    """
    return UnifiedLoss(**config)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Unified interface
    'UnifiedLoss',
    'create_unified_loss',
    # DeepSeek components
    'MultiTokenPredictionLoss',
    'TemperatureScaledCrossEntropy',
    'AuxiliaryFreeMoEBalancer',
    'DeepSeekLoss',
    # Adaptive MTP
    'AdaptiveMTPLoss',
    # Repetition penalties
    'NGramRepetitionPenalty',
    'SequenceRepetitionDetector',
    # Anti-repetition
    'AntiRepetitionLoss',
    'AdaptiveAntiRepetitionLoss',
    # Advanced losses
    'ContrastiveLoss',
    'FocalLoss',
    'LabelSmoothingLoss',
    'DiversityLoss',
    'AuxiliaryLoss',
    'ConsistencyLoss',
    'PerplexityLoss',
    'AdaptiveLossScaling',
    'CompositeLoss',
]
