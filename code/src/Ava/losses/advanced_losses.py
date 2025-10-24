"""
Advanced Loss Functions Module

Advanced and specialized loss functions including:
- Adaptive Multi-Token Prediction (MTP)
- DeepSeek-style losses (MTP, Temperature Scaling, MoE Balancing)
- Anti-Repetition losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import math


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
# DEEPSEEK MULTI-TOKEN PREDICTION
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

            # Accumulate losses as tensors to preserve gradient flow
            total_mtp_loss = total_mtp_loss + loss
            # Store detached scalar for logging only
            per_token_losses.append(loss.detach().item())

        # Average over number of future tokens
        if self.num_future_tokens > 0:
            total_mtp_loss = total_mtp_loss / min(self.num_future_tokens, seq_len - 1)

        return {
            'mtp_loss': total_mtp_loss * self.mtp_weight,
            'per_token_losses': per_token_losses,
            'mtp_weight': self.mtp_weight
        }


# ============================================================================
# TEMPERATURE-SCALED CROSS ENTROPY
# ============================================================================

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


# ============================================================================
# AUXILIARY-FREE MOE BALANCER
# ============================================================================

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

        # Accumulation buffers for gradient accumulation support
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
            # Accumulate statistics across micro-steps, apply momentum only on optimizer steps

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
                accum_steps_tensor: torch.Tensor = self._accumulation_steps  # type: ignore[assignment]
                num_steps: int = max(1, int(accum_steps_tensor.item()))
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

                # Reset accumulators
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

            # Create effective gradient signal for load balancing
            if grad_adjustment is not None:
                adjusted_logits = gate_logits + grad_adjustment.detach()
                # Create loss that pulls gate_logits toward balanced distribution
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


# ============================================================================
# DEEPSEEK COMBINED LOSS
# ============================================================================

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
            # balancing_term properly contributes gradients
            total_loss = total_loss + balance_dict['balancing_term']

        # Add main_loss key that the trainer expects
        losses['main_loss'] = main_loss_dict['loss']
        losses['total_loss'] = total_loss

        return losses


# ============================================================================
# ANTI-REPETITION LOSS
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

        # 2. Calculate penalties on LABELS (ground truth)
        repetition_scores = self.calculate_ngram_repetition(
            labels, attention_mask
        )
        repetition_penalty = repetition_scores.mean()

        # 3. Combined loss
        total_loss = (
            base_loss +
            self.repetition_penalty_weight * repetition_penalty
        )

        if return_components:
            components = {
                'base_loss': base_loss.item(),
                'repetition_penalty': repetition_penalty.item(),
                'total_loss': total_loss.item()
            }
            return total_loss, components

        return total_loss, None
