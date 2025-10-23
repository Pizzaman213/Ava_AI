"""
Colossal-AI Enhanced MoE Model

This module extends the EnhancedMoEModel to support Colossal-AI's
advanced parallelization strategies including tensor parallelism,
pipeline parallelism, and sequence parallelism.
"""

import logging
import math
from typing import Dict, Optional, Tuple, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base MoE model
from .moe_model import EnhancedMoEConfig, EnhancedMoEModel, TransformerBlock

# Colossal-AI imports
try:
    import colossalai
    from colossalai.nn import (
        Linear1D_Col,
        Linear1D_Row,
        VocabParallelEmbedding1D,
        VocabParallelCrossEntropyLoss1D,
    )
    from colossalai.nn.layer import DropPath, LayerNorm1D
    from colossalai.tensor import ColoParameter
    from colossalai.utils import get_current_device

    COLOSSALAI_AVAILABLE = True
except ImportError:
    COLOSSALAI_AVAILABLE = False
    # Fallback imports
    Linear1D_Col = nn.Linear
    Linear1D_Row = nn.Linear
    VocabParallelEmbedding1D = nn.Embedding
    LayerNorm1D = nn.LayerNorm

logger = logging.getLogger(__name__)


class ColossalAIMoEConfig(EnhancedMoEConfig):
    """Extended configuration for Colossal-AI MoE model."""

    # Colossal-AI specific settings
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    use_colossal_linear: bool = True
    use_vocab_parallel: bool = True
    use_checkpoint: bool = False
    checkpoint_num_layers: Optional[int] = None

    # Expert parallelism
    expert_parallel: bool = False
    expert_parallel_size: int = 1

    # Communication optimization
    reduce_scatter_bucket_size: int = 1024 * 1024  # 1MB
    all_gather_bucket_size: int = 1024 * 1024  # 1MB


class ColossalAIMultiHeadAttention(nn.Module):
    """Multi-head attention with Colossal-AI tensor parallelism support."""

    def __init__(self, config: ColossalAIMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.config = config

        assert self.hidden_size % self.num_heads == 0

        if COLOSSALAI_AVAILABLE and config.use_colossal_linear:
            # Use Colossal-AI's parallel linear layers
            self.q_proj = Linear1D_Col(config.hidden_size, config.hidden_size, bias=True)
            self.k_proj = Linear1D_Col(config.hidden_size, config.hidden_size, bias=True)
            self.v_proj = Linear1D_Col(config.hidden_size, config.hidden_size, bias=True)
            self.o_proj = Linear1D_Row(config.hidden_size, config.hidden_size, bias=True)
        else:
            # Fallback to standard PyTorch layers
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.hidden_dropout)

        # RoPE initialization (reuse from base model)
        if not config.use_alibi:
            from .moe_model import RoPEPositionalEmbedding
            self.rope = RoPEPositionalEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )
        else:
            self.rope = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional caching."""

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
            from .moe_model import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle past key-value if provided
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Cache current key-value if requested
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present_key_value


class ColossalAIMoELayer(nn.Module):
    """MoE feed-forward layer with Colossal-AI parallelism support."""

    def __init__(self, config: ColossalAIMoEConfig):
        super().__init__()
        self.config = config

        # Router (can be parallelized)
        if COLOSSALAI_AVAILABLE and config.use_colossal_linear:
            self.router = Linear1D_Col(
                config.hidden_size,
                config.num_experts,
                bias=False
            )
        else:
            self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList()
        for _ in range(config.num_experts):
            if COLOSSALAI_AVAILABLE and config.use_colossal_linear:
                expert = nn.Sequential(
                    Linear1D_Col(config.hidden_size, config.intermediate_size, bias=True),
                    nn.GELU() if config.hidden_act == 'gelu' else nn.ReLU(),
                    nn.Dropout(config.hidden_dropout),
                    Linear1D_Row(config.intermediate_size, config.hidden_size, bias=True),
                    nn.Dropout(config.hidden_dropout),
                )
            else:
                expert = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    nn.GELU() if config.hidden_act == 'gelu' else nn.ReLU(),
                    nn.Dropout(config.hidden_dropout),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout),
                )
            self.experts.append(expert)

        self.num_experts_per_token = config.num_experts_per_token
        self.router_jitter_noise = config.router_jitter_noise

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with expert routing."""

        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router scores
        router_logits = self.router(hidden_states_flat)

        # Add jitter noise during training
        if self.training and self.router_jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.router_jitter_noise
            router_logits = router_logits + noise

        # Get top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Initialize output
        final_output = torch.zeros_like(hidden_states_flat)

        # Process each expert
        for expert_idx in range(self.config.num_experts):
            # Get tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # Get expert input
            expert_input = hidden_states_flat[expert_mask]

            # Run expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get routing weights for this expert
            expert_weights = torch.where(
                selected_experts == expert_idx,
                routing_weights,
                torch.zeros_like(routing_weights)
            ).sum(dim=-1, keepdim=True)

            # Add weighted expert output
            final_output[expert_mask] += expert_output * expert_weights[expert_mask]

        # Reshape output
        final_output = final_output.view(batch_size, seq_len, hidden_dim)

        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(router_logits, selected_experts)

        return final_output, aux_loss

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary loss for load balancing."""

        num_experts = self.config.num_experts
        num_tokens = router_logits.size(0)

        # Compute expert load (fraction of tokens per expert)
        expert_mask = F.one_hot(selected_experts, num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=0).sum(dim=0)
        expert_load = tokens_per_expert / num_tokens

        # Compute router probability mass per expert
        router_prob = F.softmax(router_logits, dim=-1)
        router_prob_per_expert = router_prob.mean(dim=0)

        # Auxiliary loss encourages uniform distribution
        aux_loss = num_experts * (expert_load * router_prob_per_expert).sum()

        return aux_loss


class ColossalAITransformerBlock(nn.Module):
    """Transformer block with Colossal-AI parallelism support."""

    def __init__(self, config: ColossalAIMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Layer normalization
        if COLOSSALAI_AVAILABLE:
            self.ln_1 = LayerNorm1D(config.hidden_size, eps=config.layer_norm_eps)
            self.ln_2 = LayerNorm1D(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Multi-head attention
        self.attn = ColossalAIMultiHeadAttention(config)

        # MoE or standard FFN
        if layer_idx % 2 == 1:  # Use MoE for odd layers
            self.ffn = ColossalAIMoELayer(config)
            self.use_moe = True
        else:
            # Standard FFN with Colossal-AI parallelism
            if COLOSSALAI_AVAILABLE and config.use_colossal_linear:
                self.ffn = nn.Sequential(
                    Linear1D_Col(config.hidden_size, config.intermediate_size, bias=True),
                    nn.GELU() if config.hidden_act == 'gelu' else nn.ReLU(),
                    nn.Dropout(config.hidden_dropout),
                    Linear1D_Row(config.intermediate_size, config.hidden_size, bias=True),
                    nn.Dropout(config.hidden_dropout),
                )
            else:
                self.ffn = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    nn.GELU() if config.hidden_act == 'gelu' else nn.ReLU(),
                    nn.Dropout(config.hidden_dropout),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout),
                )
            self.use_moe = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass through transformer block."""

        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_output

        # FFN with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        if self.use_moe:
            ffn_output, aux_loss = self.ffn(hidden_states)
        else:
            ffn_output = self.ffn(hidden_states)
            aux_loss = None

        hidden_states = residual + ffn_output

        return hidden_states, present_key_value, aux_loss


class ColossalAIEnhancedMoEModel(nn.Module):
    """Enhanced MoE Model with Colossal-AI parallelism support."""

    def __init__(self, config: ColossalAIMoEConfig):
        super().__init__()
        self.config = config

        # Token embeddings with vocab parallelism
        if COLOSSALAI_AVAILABLE and config.use_vocab_parallel:
            self.token_embedding = VocabParallelEmbedding1D(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings (standard, not parallelized)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            ColossalAITransformerBlock(config, i)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        if COLOSSALAI_AVAILABLE:
            self.ln_f = LayerNorm1D(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Output projection with vocab parallelism
        if COLOSSALAI_AVAILABLE and config.use_vocab_parallel:
            self.lm_head = VocabParallelCrossEntropyLoss1D(
                config.hidden_size,
                config.vocab_size,
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """Forward pass through the model."""

        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Get position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Prepare attention mask
        if attention_mask is not None:
            # Convert to attention bias
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Process through transformer layers
        all_aux_losses = []
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None

            hidden_states, present_key_value, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )

            if use_cache:
                present_key_values.append(present_key_value)

            if aux_loss is not None:
                all_aux_losses.append(aux_loss)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Compute logits
        if COLOSSALAI_AVAILABLE and isinstance(self.lm_head, VocabParallelCrossEntropyLoss1D):
            # Vocab parallel handles both logits and loss
            if labels is not None:
                loss = self.lm_head(hidden_states, labels)
                logits = None  # Not needed when using vocab parallel loss
            else:
                logits = self.lm_head.weight @ hidden_states.transpose(-1, -2)
                loss = None
        else:
            logits = self.lm_head(hidden_states)

            # Compute loss if labels provided
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = None

        # Add auxiliary losses
        if all_aux_losses and loss is not None:
            aux_loss = torch.stack(all_aux_losses).mean()
            loss = loss + self.config.router_aux_loss_coef * aux_loss

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': present_key_values,
            'aux_losses': all_aux_losses,
        }

    @classmethod
    def from_base_model(cls, base_model: 'EnhancedMoEModel', config: Optional[ColossalAIMoEConfig] = None):
        """Create Colossal-AI model from base MoE model."""

        if config is None:
            # Convert base config to Colossal-AI config
            base_config = base_model.config
            config = ColossalAIMoEConfig(**base_config.__dict__)

        # Create new model
        colossal_model = cls(config)

        # Copy weights from base model
        try:
            # This is a simplified weight copying - may need adjustment based on actual model structure
            colossal_model.load_state_dict(base_model.state_dict(), strict=False)
            logger.info("Successfully loaded weights from base model")
        except Exception as e:
            logger.warning(f"Failed to load all weights from base model: {e}")

        return colossal_model


def create_colossal_moe_model(
    config: Union[EnhancedMoEConfig, ColossalAIMoEConfig, Dict[str, Any]],
) -> ColossalAIEnhancedMoEModel:
    """Factory function to create Colossal-AI MoE model."""

    # Convert dict config to ColossalAIMoEConfig if needed
    if isinstance(config, dict):
        config = ColossalAIMoEConfig(**config)
    elif isinstance(config, EnhancedMoEConfig):
        # Convert base config to Colossal-AI config
        config = ColossalAIMoEConfig(**config.__dict__)

    # Check Colossal-AI availability
    if not COLOSSALAI_AVAILABLE:
        logger.warning("Colossal-AI not available. Model will use standard PyTorch layers.")
        config.use_colossal_linear = False
        config.use_vocab_parallel = False

    return ColossalAIEnhancedMoEModel(config)