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
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

logger = logging.getLogger(__name__)

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
    quantize_kv_cache: bool = False  # INT8 quantization for KV cache (75% memory savings)
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
    """Multi-head attention with optional RoPE and Flash Attention support."""

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_dropout
        self.use_flash_attention = getattr(config, 'use_flash_attention', True)
        self.quantize_kv_cache = getattr(config, 'quantize_kv_cache', False)

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
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        **kwargs
    ) -> tuple:
        batch_size, seq_len, _ = hidden_states.shape

        # Store the input dtype for consistency
        input_dtype = hidden_states.dtype

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
            # For KV cache, we need to account for the position offset
            if past_key_value is not None:
                # past_key_value contains (past_k, past_v)
                past_seq_len = past_key_value[0].shape[2]
                # Apply RoPE with correct position offsets
                cos, sin = self.rope(past_seq_len + seq_len, hidden_states.device)
                # Only use RoPE embeddings for current sequence
                cos = cos[past_seq_len:past_seq_len + seq_len].to(dtype=input_dtype)[None, None, :, :]
                sin = sin[past_seq_len:past_seq_len + seq_len].to(dtype=input_dtype)[None, None, :, :]
            else:
                cos, sin = self.rope(seq_len, hidden_states.device)
                cos = cos.to(dtype=input_dtype)[None, None, :, :]
                sin = sin.to(dtype=input_dtype)[None, None, :, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Concatenate with past key-values if provided (KV cache for generation)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # Dequantize if cache was quantized
            if self.quantize_kv_cache and past_k.dtype == torch.int8:
                past_k = past_k.to(k.dtype) / 127.0
                past_v = past_v.to(v.dtype) / 127.0
            k = torch.cat([past_k, k], dim=2)  # Concatenate on sequence dimension
            v = torch.cat([past_v, v], dim=2)

        # Store current key-values for next iteration if caching
        if use_cache:
            if self.quantize_kv_cache:
                # Quantize to INT8 for 75% memory savings
                # Scale to [-127, 127] range and convert to int8
                k_quantized = (k * 127.0).clamp(-127, 127).to(torch.int8)
                v_quantized = (v * 127.0).clamp(-127, 127).to(torch.int8)
                present_key_value = (k_quantized, v_quantized)
            else:
                present_key_value = (k, v)
        else:
            present_key_value = None

        # Use Flash Attention if enabled (2-3x faster, 3-4x less memory)
        if self.use_flash_attention:
            # F.scaled_dot_product_attention expects [batch, heads, seq, head_dim]
            # Our tensors are already in this format
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False  # Set to True if you want causal masking
            )
        else:
            # Standard attention implementation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # Ensure dtype consistency before matmul - critical for BF16 mixed precision
            attn_weights = attn_weights.to(dtype=input_dtype)

            # Compute attention output
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


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
        # CRITICAL FIX: Use appropriate epsilon for precision mode
        # 1e-9 is too small for fp16 (min normal: 6e-5), can still cause NaN
        # Use 1e-6 for fp16/bf16 compatibility
        top_k_sum = top_k_probs.sum(dim=-1, keepdim=True)
        epsilon = 1e-6 if top_k_probs.dtype in (torch.float16, torch.bfloat16) else 1e-9
        top_k_probs = top_k_probs / (top_k_sum + epsilon)  # Safe renormalization

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
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict, Optional[tuple]]:
        # Self-attention with residual
        residual = hidden_states
        # CUDA GRAPH FIX: Clone after layer norm to prevent tensor overwrite errors
        hidden_states = self.ln1(hidden_states).clone()
        attn_output, present_key_value = self.attention(
            hidden_states,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + self.dropout(attn_output)

        # MoE feed-forward with residual
        residual = hidden_states
        # CUDA GRAPH FIX: Clone after layer norm to prevent tensor overwrite errors
        hidden_states = self.ln2(hidden_states).clone()
        ff_output, aux_info = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output

        return hidden_states, aux_info, present_key_value


class EnhancedMoEModel(nn.Module):
    """Enhanced Mixture of Experts Transformer Model."""

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # OPTIMIZATION: Clear position embedding precedence with single source of truth
        # Priority: RoPE/ALiBi > Learned > None
        use_rope = getattr(config, 'use_rope', True)  # RoPE is default
        use_alibi = getattr(config, 'use_alibi', False)
        use_learned_pos = getattr(config, 'use_learned_position_embeddings', False)

        if use_rope or use_alibi:
            # RoPE/ALiBi handle positions in attention, no separate embeddings needed
            self.position_embedding = None
            if use_learned_pos:
                logger.warning("Ignoring use_learned_position_embeddings=True because RoPE/ALiBi is enabled")
        elif use_learned_pos:
            # Fallback to learned position embeddings if RoPE/ALiBi disabled
            self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            logger.info("Using learned position embeddings (consider RoPE for better extrapolation)")
        else:
            # No position encoding specified
            self.position_embedding = None
            logger.warning("No position encoding specified - model may not learn positions properly")

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
        past_key_values: Optional[List[tuple]] = None,
        use_cache: bool = False,
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
        # IMPORTANT: Match the dtype of hidden_states to avoid dtype mismatch in scaled_dot_product_attention
        # Use float32 for mask construction to avoid precision issues, then cast to model dtype
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32),
            diagonal=1
        )  # Shape: [seq_len, seq_len]

        # Cast to match hidden_states dtype AFTER construction
        causal_mask = causal_mask.to(dtype=hidden_states.dtype)

        # Add batch and head dimensions: [1, 1, seq_len, seq_len]
        causal_mask = causal_mask[None, None, :, :]

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask shape: [batch_size, seq_len]
            # Convert to [batch_size, 1, 1, seq_len] for broadcasting
            padding_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]

            # Invert: 1 = attend, 0 = don't attend
            # Convert 0s to -inf, ensuring dtype matches
            padding_mask = ((1.0 - padding_mask) * torch.finfo(hidden_states.dtype).min).to(dtype=hidden_states.dtype)

            # Combine causal and padding masks
            # padding_mask: [batch, 1, 1, seq_len] - masks padding tokens
            # causal_mask: [1, 1, seq_len, seq_len] - masks future tokens
            # Broadcasting will handle the combination
            attention_mask = causal_mask + padding_mask  # Broadcasting magic
        else:
            # Just use causal mask
            attention_mask = causal_mask

        # Apply transformer blocks with KV caching
        all_aux_info = []
        present_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            # Get past key-value for this layer if available
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            hidden_states, aux_info, present_key_value = layer(
                hidden_states,
                attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            all_aux_info.append(aux_info)

            if use_cache:
                present_key_values.append(present_key_value)  # type: ignore[union-attr]

        # Final layer norm
        # CUDA GRAPH FIX: Clone after final layer norm to prevent tensor overwrite errors
        hidden_states = self.ln_f(hidden_states).clone()

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
                'aux_info': all_aux_info,
                'past_key_values': present_key_values
            }
        else:
            if loss is not None:
                return (loss, logits, hidden_states, present_key_values)
            else:
                return (logits, hidden_states, present_key_values)

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @torch.no_grad()
    @torch.compiler.disable(recursive=True)  # CRITICAL: Disable compile for dynamic shapes in generation
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Simple greedy/sampling generation without KV cache.

        Note: KV caching is not yet implemented in the current forward() method.
        This version re-processes the entire sequence at each step, which is slower
        but guaranteed to work correctly.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum total length (including prompt)
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k filtering (keep only top k tokens). If None, no filtering
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            no_repeat_ngram_size: If > 0, prevents repetition of n-grams of this size
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
        for step_idx in range(max_length - input_ids.shape[1]):
            # Forward pass - process entire sequence each time (no KV cache yet)
            outputs = self.forward(
                input_ids=generated,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs['logits']

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # For each token in the generated sequence, apply penalty
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        # If score < 0, multiply by penalty (make more negative)
                        # If score > 0, divide by penalty (make less positive)
                        # This discourages repetition regardless of original score
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Apply n-gram blocking
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                # For each sequence in batch
                for i in range(batch_size):
                    # Get the last (n-1) tokens
                    ngram_prefix = generated[i, -(no_repeat_ngram_size - 1):].tolist()

                    # Find all n-grams in the generated sequence that start with this prefix
                    banned_tokens = set()
                    for j in range(generated.shape[1] - no_repeat_ngram_size + 1):
                        # Check if this position matches our prefix
                        current_ngram_prefix = generated[i, j:j + no_repeat_ngram_size - 1].tolist()
                        if current_ngram_prefix == ngram_prefix:
                            # Ban the token that completes this n-gram
                            banned_token = generated[i, j + no_repeat_ngram_size - 1].item()
                            banned_tokens.add(banned_token)

                    # Set banned tokens to -inf
                    for token_id in banned_tokens:
                        next_token_logits[i, token_id] = float('-inf')

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                # Remove all tokens with a probability less than the top k tokens
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < top_k_values[:, -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

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


# ============================================================================
# NEW: Optimized MoE Transformer with Production-Grade Performance
# ============================================================================

from dataclasses import dataclass, field
from typing import Optional

try:
    from .moe_layer import SparseMoELayer
except ImportError:
    SparseMoELayer = None


@dataclass
class OptimizedMoEConfig:
    """
    Configuration for OptimizedMoETransformer with all performance features.
    """
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 14336  # 3.5x hidden_size (Mixtral-style)
    max_position_embeddings: int = 4096

    # MoE settings
    num_experts: int = 32
    num_experts_per_token: int = 2
    router_type: str = 'mixtral'  # 'mixtral' or 'deepseek'
    capacity_factor: float = 1.25
    expert_dropout: float = 0.0
    activation: str = 'swiglu'  # 'swiglu', 'geglu', 'gelu'

    # MoE performance
    use_grouped_gemm: bool = True
    use_triton_kernels: bool = True
    use_torch_compile: bool = True
    gradient_checkpointing: bool = False

    # MoE auxiliary losses
    router_z_loss_coef: float = 0.001
    load_balance_loss_coef: float = 0.01
    diversity_loss_coef: float = 0.001
    expert_dropout_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    # DeepSeek-style shared expert
    use_shared_expert: bool = False
    shared_expert_weight: float = 0.5

    # Memory optimization (All phases)
    # Phase 1: LoRA
    use_lora_experts: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    freeze_lora_base: bool = False
    # Phase 2: CPU Offloading
    use_expert_offloading: bool = False
    max_active_experts_gpu: int = 4
    offload_eviction_policy: str = 'lru'
    # Phase 4: Quantization
    use_expert_quantization: bool = False
    expert_quantization_bits: int = 8

    # Attention settings
    attention_dropout: float = 0.0
    use_flash_attention: bool = False
    rope_theta: float = 10000.0

    # Regularization
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Distributed training
    expert_parallel_size: int = 1

    # Type hints
    dtype: Optional[torch.dtype] = None


class OptimizedTransformerBlock(nn.Module):
    """
    Transformer block using OptimizedMoELayer for FFN.
    """

    def __init__(self, config: OptimizedMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Self-attention (reuse from existing implementation)
        self.attention = MultiHeadAttention(
            # Convert config to EnhancedMoEConfig format
            type('Config', (), {
                'hidden_size': config.hidden_size,
                'num_attention_heads': config.num_attention_heads,
                'attention_dropout': config.attention_dropout,
                'max_position_embeddings': config.max_position_embeddings,
                'rope_theta': config.rope_theta,
                'use_alibi': False,
                'use_flash_attention': getattr(config, 'use_flash_attention', False),
            })()
        )

        # Sparse MoE layer (replaces standard FFN)
        if SparseMoELayer is not None:
            self.moe = SparseMoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                router_type=config.router_type,
                capacity_factor=config.capacity_factor,
                expert_dropout=config.expert_dropout,
                activation=config.activation,
                use_grouped_gemm=config.use_grouped_gemm,
                use_triton_kernels=config.use_triton_kernels,
                use_torch_compile=config.use_torch_compile,
                router_z_loss_coef=config.router_z_loss_coef,
                load_balance_loss_coef=config.load_balance_loss_coef,
                diversity_loss_coef=config.diversity_loss_coef,
                expert_dropout_loss_coef=config.expert_dropout_loss_coef,
                router_jitter_noise=config.router_jitter_noise,
                use_shared_expert=config.use_shared_expert,
                shared_expert_weight=config.shared_expert_weight,
                gradient_checkpointing=config.gradient_checkpointing,
                dtype=config.dtype,
                # Memory optimization parameters (All phases)
                use_lora_experts=config.use_lora_experts,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                freeze_lora_base=config.freeze_lora_base,
                use_expert_offloading=config.use_expert_offloading,
                max_active_experts_gpu=config.max_active_experts_gpu,
                offload_eviction_policy=config.offload_eviction_policy,
                use_expert_quantization=config.use_expert_quantization,
                expert_quantization_bits=config.expert_quantization_bits,
            )
        else:
            raise ImportError("SparseMoELayer not available. Cannot create OptimizedMoETransformer")

        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        # Self-attention with residual
        residual = hidden_states
        # CUDA GRAPH FIX: Clone after layer norm to prevent tensor overwrite errors
        hidden_states = self.ln1(hidden_states).clone()
        attn_output, _ = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attn_output)

        # MoE with residual
        residual = hidden_states
        moe_output, aux_loss, moe_metrics = self.moe(hidden_states, training=training)
        hidden_states = residual + moe_output

        return hidden_states, {'aux_loss': aux_loss, **moe_metrics}


class OptimizedMoETransformer(nn.Module):
    """
    Production-grade MoE Transformer with all performance optimizations.

    Features:
    - Mixtral or DeepSeek routing
    - Grouped GEMM for expert computation
    - Triton kernels for routing
    - torch.compile optimization
    - Expert parallelism support
    - 4 auxiliary losses for stability
    - Gradient checkpointing

    Args:
        config: OptimizedMoEConfig instance

    Example:
        >>> config = OptimizedMoEConfig(
        ...     hidden_size=4096,
        ...     num_experts=32,
        ...     num_experts_per_token=2,
        ...     router_type='mixtral'
        ... )
        >>> model = OptimizedMoETransformer(config)
        >>> x = torch.randint(0, 32000, (2, 128))
        >>> output = model(x)
    """

    def __init__(self, config: OptimizedMoEConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks with MoE
        self.layers = nn.ModuleList([
            OptimizedTransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with proper scaling."""
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
    ) -> Any:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], optional
            labels: [batch_size, seq_len], optional (for training)
            return_dict: Whether to return dict or tuple

        Returns:
            Dict with 'loss', 'logits', 'hidden_states', 'aux_info' if return_dict=True
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Validate inputs
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError(f"Invalid token IDs. Must be in [0, {self.config.vocab_size})")

        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Create causal attention mask
        # Use float32 for mask construction to avoid precision issues, then cast to model dtype
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32),
            diagonal=1
        )
        # Cast to match hidden_states dtype AFTER construction
        causal_mask = causal_mask.to(dtype=hidden_states.dtype)[None, None, :, :]

        if attention_mask is not None:
            padding_mask = ((1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min).to(dtype=hidden_states.dtype)
            attention_mask = causal_mask + padding_mask
        else:
            attention_mask = causal_mask

        # Apply transformer blocks
        all_aux_info = []
        for layer in self.layers:
            hidden_states, aux_info = layer(
                hidden_states,
                attention_mask=attention_mask,
                training=self.training,
            )
            all_aux_info.append(aux_info)

        # Final layer norm
        # CUDA GRAPH FIX: Clone after layer norm to prevent tensor overwrite errors
        hidden_states = self.ln_f(hidden_states).clone()

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Standard cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Add MoE auxiliary losses
            if all_aux_info:
                total_aux_loss = sum(info['aux_loss'] for info in all_aux_info) / len(all_aux_info)
                loss = loss + total_aux_loss

        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states,
                'last_hidden_state': hidden_states,
                'aux_info': all_aux_info,
            }
        else:
            return (loss, logits, hidden_states) if loss is not None else (logits, hidden_states)

    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """Get expert utilization statistics across all layers."""
        all_stats = {}
        for i, layer in enumerate(self.layers):
            layer_stats = layer.moe.get_expert_usage_stats()  # type: ignore[attr-defined]
            for key, value in layer_stats.items():
                all_stats[f'layer_{i}_{key}'] = value
        return all_stats

    def reset_expert_counts(self):
        """Reset expert utilization counters."""
        for layer in self.layers:
            layer.moe.reset_expert_counts()  # type: ignore[attr-defined]

    @torch.no_grad()
    @torch.compiler.disable(recursive=True)  # CRITICAL: Disable compile for dynamic shapes in generation
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Simple greedy/sampling generation.

        IMPORTANT: This method is decorated with @torch.compiler.disable() to prevent
        torch.compile from creating static graphs. Generation requires dynamic sequence
        lengths as tokens are added one by one, which would cause shape mismatch errors.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum total length (including prompt)
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k filtering (keep only top k tokens). If None, no filtering
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            no_repeat_ngram_size: If > 0, prevents repetition of n-grams of this size
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
        for step_idx in range(max_length - input_ids.shape[1]):
            # Forward pass with error handling
            try:
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                logits = outputs['logits']
            except Exception as e:
                # CRITICAL FIX: Add comprehensive error context for debugging
                error_msg = (
                    f"Generation failed at step {step_idx}:\n"
                    f"  Generated shape: {generated.shape}\n"
                    f"  Attention mask shape: {attention_mask.shape if attention_mask is not None else 'None'}\n"
                    f"  Batch size: {batch_size}\n"
                    f"  Current sequence length: {generated.shape[1]}\n"
                    f"  Error: {str(e)}\n"
                    f"  Error type: {type(e).__name__}"
                )
                raise RuntimeError(error_msg) from e

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # For each token in the generated sequence, apply penalty
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        # If score < 0, multiply by penalty (make more negative)
                        # If score > 0, divide by penalty (make less positive)
                        # This discourages repetition regardless of original score
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Apply n-gram blocking
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                # For each sequence in batch
                for i in range(batch_size):
                    # Get the last (n-1) tokens
                    ngram_prefix = generated[i, -(no_repeat_ngram_size - 1):].tolist()

                    # Find all n-grams in the generated sequence that start with this prefix
                    banned_tokens = set()
                    for j in range(generated.shape[1] - no_repeat_ngram_size + 1):
                        # Check if this position matches our prefix
                        current_ngram_prefix = generated[i, j:j + no_repeat_ngram_size - 1].tolist()
                        if current_ngram_prefix == ngram_prefix:
                            # Ban the token that completes this n-gram
                            banned_token = generated[i, j + no_repeat_ngram_size - 1].item()
                            banned_tokens.add(banned_token)

                    # Set banned tokens to -inf
                    for token_id in banned_tokens:
                        next_token_logits[i, token_id] = float('-inf')

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                # Remove all tokens with a probability less than the top k tokens
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < top_k_values[:, -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

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
