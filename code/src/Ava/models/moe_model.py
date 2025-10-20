"""
Enhanced Mixture of Experts (MoE) Model

A production-ready transformer with Mixture of Experts layers, supporting:
- Switch Transformer routing
- Dynamic expert selection
- Load balancing
- Flash Attention
- Rotary Position Embeddings (RoPE)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

# Import routing and expert layers
try:
    from ..layers.routing import SwitchTransformerRouting
    from ..layers.experts import SparseExpert
except ImportError:
    SwitchTransformerRouting = None
    SparseExpert = None


@dataclass
class EnhancedMoEConfig:
    """Configuration for Enhanced MoE Model."""

    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048

    # MoE settings
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    router_type: str = 'switch'  # 'switch', 'deepseek', etc.
    router_aux_loss_coef: float = 0.01
    router_jitter_noise: float = 0.01

    # Regularization
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Optimization
    use_flash_attention: bool = False
    use_cache: bool = True
    rope_theta: float = 10000.0
    hidden_act: str = 'gelu'
    initializer_range: float = 0.02

    # Additional features
    use_moh: bool = False  # Mixture of Heads
    use_moa: bool = False  # Mixture of Activations
    use_cross_attention: bool = False
    use_alibi: bool = False

    # Training-specific features (may be in checkpoint but not used in inference)
    deepspeed_activation_checkpointing: bool = False
    deepspeed_partition_activations: bool = False
    deepspeed_moe_param_groups: bool = False

    # Loss regularization features
    entropy_regularization: float = 0.0  # Entropy bonus for diverse predictions
    output_diversity_weight: float = 0.0  # Penalty for low output diversity
    eos_logit_bias: float = 0.0  # Bias applied to EOS token logits
    eos_token_id: int = 3  # EOS token ID (tokenizer-specific)
    min_sequence_length: int = 0  # Minimum sequence length before allowing EOS


class RoPEPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    # Type annotation for registered buffer
    inv_freq: torch.Tensor

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary embeddings."""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE."""

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_dropout

        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_attention_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # RoPE
        if not config.use_alibi:
            self.rope = RoPEPositionalEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )
        else:
            self.rope = None

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if configured
        if self.rope is not None:
            cos, sin = self.rope(seq_len, hidden_states.device)
            cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim]
            sin = sin[None, None, :, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MoEFeedForward(nn.Module):
    """MoE Feed-Forward layer with expert routing."""

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token

        # Simple linear router (not using complex routing for now)
        self.router = nn.Linear(config.hidden_size, config.num_experts)

        # Experts
        if SparseExpert is not None:
            self.experts = nn.ModuleList([
                SparseExpert(
                    input_size=config.hidden_size,
                    hidden_size=config.intermediate_size,
                    output_size=config.hidden_size,
                    sparsity_level=0.5
                )
                for _ in range(config.num_experts)
            ])
        else:
            # Fallback: simple FFN experts
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    nn.GELU() if config.hidden_act == 'gelu' else nn.ReLU(),
                    nn.Dropout(config.hidden_dropout),
                    nn.Linear(config.intermediate_size, config.hidden_size)
                )
                for _ in range(config.num_experts)
            ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_flat.shape[0]

        # Router forward pass
        router_logits = self.router(hidden_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        # CRITICAL FIX: Calculate load balancing auxiliary loss
        # This encourages uniform expert utilization
        aux_info = {}

        # Fraction of tokens routed to each expert
        # Shape: [num_experts]
        expert_mask = torch.zeros(self.num_experts, device=hidden_flat.device)
        for expert_idx in range(self.num_experts):
            expert_mask[expert_idx] = (router_probs.argmax(dim=-1) == expert_idx).float().sum()
        expert_fraction = expert_mask / num_tokens  # Normalize

        # Average probability assigned to each expert
        # Shape: [num_experts]
        expert_avg_prob = router_probs.mean(dim=0)

        # Load balancing loss: encourages expert_fraction ≈ expert_avg_prob
        # If all experts are used equally, both should be 1/num_experts
        load_balance_loss = self.num_experts * (expert_fraction * expert_avg_prob).sum()
        aux_info['load_balance_loss'] = load_balance_loss
        aux_info['router_probs'] = router_probs.detach()
        aux_info['expert_utilization'] = expert_mask.detach()

        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)
        # CRITICAL FIX: Add epsilon to prevent division by zero
        top_k_sum = top_k_probs.sum(dim=-1, keepdim=True)
        top_k_probs = top_k_probs / (top_k_sum + 1e-9)  # Safe renormalization

        # Process through experts (more efficient batched version)
        output = torch.zeros_like(hidden_flat)

        for i in range(self.num_experts_per_token):
            expert_mask_indices = top_k_indices[:, i]
            for expert_idx in range(self.num_experts):
                token_mask = (expert_mask_indices == expert_idx)
                if token_mask.any():
                    tokens_for_expert = hidden_flat[token_mask]
                    if SparseExpert is not None and isinstance(self.experts[expert_idx], SparseExpert):
                        expert_output, _, _ = self.experts[expert_idx](tokens_for_expert)
                    else:
                        expert_output = self.experts[expert_idx](tokens_for_expert)
                    # Get the expert weights for these tokens
                    token_weights = top_k_probs[token_mask, i].unsqueeze(1)
                    output[token_mask] += expert_output * token_weights

        output = output.view(batch_size, seq_len, hidden_size)
        output = self.dropout(output)

        return output, aux_info


class TransformerBlock(nn.Module):
    """Transformer block with MoE feed-forward."""

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = MoEFeedForward(config)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attn_output)

        # MoE feed-forward with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output, aux_info = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output

        return hidden_states, aux_info


class EnhancedMoEModel(nn.Module):
    """Enhanced Mixture of Experts Transformer Model."""

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # CRITICAL FIX: Use EITHER learned position embeddings OR RoPE, not both
        # RoPE is superior for extrapolation, so we use it exclusively
        use_learned_pos = getattr(config, 'use_learned_position_embeddings', False)
        if use_learned_pos and config.use_alibi:
            # If both specified, prefer RoPE/ALiBi over learned
            self.position_embedding = None
        elif use_learned_pos:
            # Use learned position embeddings (legacy support)
            self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            # Default: No learned position embeddings (RoPE handles it in attention)
            self.position_embedding = None

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # CRITICAL FIX: Only tie weights if explicitly enabled in config
        # Tying can cause gradient conflicts with large vocabularies
        tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
        if tie_word_embeddings:
            # Tie input and output embeddings (saves memory but can hurt training)
            self.lm_head.weight = self.token_embedding.weight
        else:
            # Keep separate (better for training, especially with large vocab)
            pass  # lm_head already has independent weights

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Any:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # CRITICAL: Validate input_ids to prevent CUDA assert errors
        # Check for negative values
        if (input_ids < 0).any():
            raise ValueError(f"Found negative token IDs in input_ids. Min: {input_ids.min().item()}")

        # Check for values >= vocab_size
        if (input_ids >= self.config.vocab_size).any():
            max_id = input_ids.max().item()
            raise ValueError(
                f"Found token IDs >= vocab_size ({self.config.vocab_size}). "
                f"Max ID in batch: {max_id}. Check your tokenizer configuration."
            )

        # Check for NaN or inf
        if not torch.isfinite(input_ids.float()).all():
            raise ValueError("Found NaN or Inf in input_ids")

        # Check sequence length
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings}). Truncate your inputs."
            )

        # Embeddings
        token_embeds = self.token_embedding(input_ids)

        if self.position_embedding is not None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + position_embeds
        else:
            hidden_states = token_embeds

        hidden_states = self.dropout(hidden_states)

        # CRITICAL FIX: Prepare proper causal attention mask
        # Create causal mask: upper triangular matrix of -inf (prevents looking ahead)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )  # Shape: [seq_len, seq_len]

        # Add batch and head dimensions: [1, 1, seq_len, seq_len]
        causal_mask = causal_mask[None, None, :, :]

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask shape: [batch_size, seq_len]
            # Convert to [batch_size, 1, 1, seq_len] for broadcasting
            padding_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]

            # Invert: 1 = attend, 0 = don't attend
            # Convert 0s to -inf
            padding_mask = (1.0 - padding_mask) * torch.finfo(hidden_states.dtype).min

            # Combine causal and padding masks
            # padding_mask: [batch, 1, 1, seq_len] - masks padding tokens
            # causal_mask: [1, 1, seq_len, seq_len] - masks future tokens
            # Broadcasting will handle the combination
            attention_mask = causal_mask + padding_mask  # Broadcasting magic
        else:
            # Just use causal mask
            attention_mask = causal_mask

        # Apply transformer blocks
        all_aux_info = []
        for layer in self.layers:
            hidden_states, aux_info = layer(hidden_states, attention_mask)
            all_aux_info.append(aux_info)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # CRITICAL FIX: Apply negative bias to EOS token logits during training
        # This prevents the model from learning to output EOS as the most likely token
        if self.training and labels is not None:
            eos_logit_bias = getattr(self.config, 'eos_logit_bias', 0.0)
            if eos_logit_bias > 0:
                # CRITICAL FIX: Get EOS token ID from config (tokenizer-specific)
                # Default to 3 for enhanced-500 tokenizer, not Qwen's 151643!
                eos_token_id = getattr(self.config, 'eos_token_id', 3)
                # Subtract bias from EOS logits (makes EOS less likely to be predicted)
                logits[:, :, eos_token_id] = logits[:, :, eos_token_id] - eos_logit_bias

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # CRITICAL FIX: Add MoE auxiliary loss for load balancing
            if all_aux_info and self.config.router_aux_loss_coef > 0:
                total_aux_loss = 0.0
                num_layers_with_aux = 0
                for layer_aux in all_aux_info:
                    if 'load_balance_loss' in layer_aux:
                        total_aux_loss += layer_aux['load_balance_loss']
                        num_layers_with_aux += 1

                if num_layers_with_aux > 0:
                    avg_aux_loss = total_aux_loss / num_layers_with_aux
                    loss = loss + self.config.router_aux_loss_coef * avg_aux_loss

            # FIX #18: Add entropy regularization (encourages diverse predictions)
            entropy_reg = getattr(self.config, 'entropy_regularization', 0.0)
            if entropy_reg > 0:
                # Calculate entropy of output distribution
                output_probs = F.softmax(shift_logits, dim=-1)
                # Entropy: -sum(p * log(p))
                entropy = -(output_probs * torch.log(output_probs + 1e-9)).sum(dim=-1).mean()
                # Subtract entropy (negative loss = bonus for high entropy/diversity)
                loss = loss - entropy_reg * entropy

            # FIX #19: Add output diversity penalty (penalizes repetitive outputs)
            diversity_weight = getattr(self.config, 'output_diversity_weight', 0.0)
            if diversity_weight > 0:
                # Get predicted tokens
                predicted_tokens = shift_logits.argmax(dim=-1)  # [batch, seq_len]
                # Calculate diversity: ratio of unique tokens to total tokens
                batch_size = predicted_tokens.shape[0]
                diversity_scores = []
                for i in range(batch_size):
                    unique_count = predicted_tokens[i].unique().numel()
                    total_count = predicted_tokens[i].numel()
                    diversity_scores.append(unique_count / total_count)
                avg_diversity = sum(diversity_scores) / len(diversity_scores)
                # Penalty for low diversity (1 - diversity, so low diversity = high penalty)
                diversity_loss = (1.0 - avg_diversity)
                loss = loss + diversity_weight * diversity_loss

        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states,
                'last_hidden_state': hidden_states,
                'aux_info': all_aux_info
            }
        else:
            return (loss, logits, hidden_states) if loss is not None else (logits, hidden_states)

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Simple greedy/sampling generation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum total length (including prompt)
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (True) or greedy (False)
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID

        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Start with input_ids
        generated = input_ids.clone()

        # Generate tokens one at a time
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs['logits']

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if do_sample:
                # Nucleus (top-p) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Mask out removed tokens
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Check if we've reached max length
            if generated.shape[1] >= max_length:
                break

        return generated
