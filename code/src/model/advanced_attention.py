"""
Advanced Attention Mechanisms for MoE++ Models

Includes Differential Attention, Mixture of Attention Heads,
and other research-oriented attention variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math
import numpy as np

from .attention import BaseAttention, create_attention_layer


@dataclass
class AdvancedAttentionConfig:
    """Configuration for advanced attention mechanisms"""
    # Differential Attention
    use_differential_attention: bool = False
    lambda_diff: float = 0.5
    diff_temperature: float = 1.0
    
    # Mixture of Attention Heads
    use_mixture_of_heads: bool = False
    num_attention_experts: int = 4
    attention_expert_types: List[str] = None
    
    # Dilated Attention
    use_dilated_attention: bool = False
    dilation_rates: List[int] = None
    
    # Synthesizer Attention
    use_synthesizer: bool = False
    synthesizer_mode: str = "dense"  # "dense", "random", "factorized"
    
    def __post_init__(self):
        if self.attention_expert_types is None:
            self.attention_expert_types = [
                "standard", "sliding_window", "sparse", "linear"
            ]
            
        if self.dilation_rates is None:
            self.dilation_rates = [1, 2, 4, 8]


class DifferentialAttention(BaseAttention):
    """
    Differential Attention that learns what NOT to attend to
    
    Computes attention as: Attention_pos - λ * Attention_neg
    This allows the model to explicitly learn negative attention patterns.
    """
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        
        # Separate projections for positive and negative attention
        self.q_proj_neg = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj_neg = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj_neg = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Learnable lambda parameter
        self.lambda_diff = nn.Parameter(torch.tensor(config.lambda_diff))
        self.temperature = config.diff_temperature
        
        # Gating mechanism for adaptive lambda
        self.lambda_gate = nn.Linear(self.hidden_size, 1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Positive attention (standard)
        query_states_pos = self.q_proj(hidden_states)
        key_states_pos = self.k_proj(hidden_states)
        value_states_pos = self.v_proj(hidden_states)
        
        # Negative attention
        query_states_neg = self.q_proj_neg(hidden_states)
        key_states_neg = self.k_proj_neg(hidden_states)
        value_states_neg = self.v_proj_neg(hidden_states)
        
        # Reshape for multi-head attention
        query_states_pos = self._reshape(query_states_pos, batch_size, seq_length)
        key_states_pos = self._reshape(key_states_pos, batch_size, seq_length)
        value_states_pos = self._reshape(value_states_pos, batch_size, seq_length)
        
        query_states_neg = self._reshape(query_states_neg, batch_size, seq_length)
        key_states_neg = self._reshape(key_states_neg, batch_size, seq_length)
        value_states_neg = self._reshape(value_states_neg, batch_size, seq_length)
        
        # Apply rotary embeddings if configured
        if self.position_embedding is not None:
            cos, sin = self.position_embedding(value_states_pos, seq_length)
            query_states_pos, key_states_pos = apply_rotary_pos_emb(
                query_states_pos, key_states_pos, cos, sin, position_ids
            )
            query_states_neg, key_states_neg = apply_rotary_pos_emb(
                query_states_neg, key_states_neg, cos, sin, position_ids
            )
        
        # Compute positive attention scores
        attn_weights_pos = torch.matmul(
            query_states_pos, key_states_pos.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        # Compute negative attention scores
        attn_weights_neg = torch.matmul(
            query_states_neg, key_states_neg.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        # Apply temperature scaling
        attn_weights_pos = attn_weights_pos / self.temperature
        attn_weights_neg = attn_weights_neg / self.temperature
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights_pos = attn_weights_pos + attention_mask
            attn_weights_neg = attn_weights_neg + attention_mask
        
        # Softmax
        attn_probs_pos = F.softmax(attn_weights_pos, dim=-1)
        attn_probs_neg = F.softmax(attn_weights_neg, dim=-1)
        
        # Apply dropout
        attn_probs_pos = self.attention_dropout(attn_probs_pos)
        attn_probs_neg = self.attention_dropout(attn_probs_neg)
        
        # Compute attention outputs
        attn_output_pos = torch.matmul(attn_probs_pos, value_states_pos)
        attn_output_neg = torch.matmul(attn_probs_neg, value_states_neg)
        
        # Compute adaptive lambda
        lambda_gate = torch.sigmoid(self.lambda_gate(hidden_states))
        effective_lambda = self.lambda_diff * lambda_gate
        
        # Differential attention: positive - lambda * negative
        attn_output = attn_output_pos - effective_lambda.unsqueeze(1) * attn_output_neg
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Compute combined attention weights for visualization
        if output_attentions:
            attn_weights = attn_probs_pos - effective_lambda.unsqueeze(1).unsqueeze(1) * attn_probs_neg
        else:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
        
    def _reshape(self, tensor, batch_size, seq_length):
        """Reshape tensor for multi-head attention"""
        return tensor.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)


class MixtureOfAttentionHeads(BaseAttention):
    """
    Routes different queries to different attention patterns
    
    Each head can use a different attention mechanism (sliding window,
    global, sparse, etc.) based on the input.
    """
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        
        self.num_attention_experts = config.num_attention_experts
        self.attention_expert_types = config.attention_expert_types
        
        # Create different attention mechanisms
        self.attention_experts = nn.ModuleList()
        for expert_type in self.attention_expert_types:
            # Create config for this expert
            expert_config = type(config)(**vars(config))
            expert_config.attention_variant = expert_type
            expert_config.num_attention_heads = self.num_heads // len(self.attention_expert_types)
            
            # Create attention layer
            attention_layer = create_attention_layer(expert_config, layer_idx)
            self.attention_experts.append(attention_layer)
        
        # Router for selecting attention patterns
        self.attention_router = nn.Linear(
            self.hidden_size,
            len(self.attention_experts)
        )
        
        # Learnable temperature for routing
        self.router_temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Compute routing probabilities
        router_input = hidden_states.mean(dim=1)  # Average pooling
        router_logits = self.attention_router(router_input)
        router_probs = F.softmax(router_logits / self.router_temperature, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        combined_attn_weights = None
        
        # Process each attention expert
        for idx, (expert, prob) in enumerate(zip(self.attention_experts, router_probs.unbind(dim=-1))):
            # Compute attention with this expert
            expert_output, expert_attn_weights, _ = expert(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            
            # Weight by routing probability
            output += expert_output * prob.unsqueeze(1).unsqueeze(2)
            
            # Combine attention weights if needed
            if output_attentions and expert_attn_weights is not None:
                if combined_attn_weights is None:
                    combined_attn_weights = expert_attn_weights * prob.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    combined_attn_weights += expert_attn_weights * prob.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        return output, combined_attn_weights, past_key_value


class DilatedAttention(BaseAttention):
    """
    Attention with dilated (strided) patterns for long-range dependencies
    
    Uses different dilation rates for different heads to capture
    multi-scale patterns efficiently.
    """
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        
        self.dilation_rates = config.dilation_rates
        self.heads_per_dilation = self.num_heads // len(self.dilation_rates)
        
        # Ensure we have enough heads
        assert self.num_heads >= len(self.dilation_rates), \
            f"Need at least {len(self.dilation_rates)} heads for dilated attention"
    
    def _create_dilated_attention_mask(
        self,
        seq_length: int,
        dilation_rate: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create attention mask for dilated attention pattern"""
        mask = torch.full((seq_length, seq_length), float('-inf'), device=device)
        
        for i in range(seq_length):
            # Attend to positions at dilated intervals
            for j in range(i, -1, -dilation_rate):
                mask[i, j] = 0
                
            # Also attend to recent tokens (within dilation_rate)
            start = max(0, i - dilation_rate + 1)
            mask[i, start:i+1] = 0
            
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Standard projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._reshape(query_states, batch_size, seq_length)
        key_states = self._reshape(key_states, batch_size, seq_length)
        value_states = self._reshape(value_states, batch_size, seq_length)
        
        # Apply rotary embeddings if configured
        if self.position_embedding is not None:
            cos, sin = self.position_embedding(value_states, seq_length)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        
        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        # Apply different dilation masks to different heads
        device = attn_weights.device
        for i, dilation_rate in enumerate(self.dilation_rates):
            start_head = i * self.heads_per_dilation
            end_head = start_head + self.heads_per_dilation
            
            # Create dilated mask
            dilated_mask = self._create_dilated_attention_mask(
                seq_length, dilation_rate, device
            )
            
            # Apply to specific heads
            attn_weights[:, start_head:end_head] += dilated_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply user-provided attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attention_dropout(attn_probs)
        
        # Compute output
        attn_output = torch.matmul(attn_probs, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_probs if output_attentions else None, past_key_value
    
    def _reshape(self, tensor, batch_size, seq_length):
        """Reshape tensor for multi-head attention"""
        return tensor.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)


class SynthesizerAttention(BaseAttention):
    """
    Synthesizer: Rethinking Self-Attention in Transformer Models
    
    Generates attention patterns without computing token interactions,
    using learned or random patterns instead.
    """
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        
        self.synthesizer_mode = config.synthesizer_mode
        
        if self.synthesizer_mode == "dense":
            # Learn a dense attention matrix directly
            self.attention_weights = nn.Parameter(
                torch.randn(self.num_heads, config.max_position_embeddings, config.max_position_embeddings)
            )
        elif self.synthesizer_mode == "factorized":
            # Factorized version: learn two smaller matrices
            self.attention_left = nn.Parameter(
                torch.randn(self.num_heads, config.max_position_embeddings, 32)
            )
            self.attention_right = nn.Parameter(
                torch.randn(self.num_heads, 32, config.max_position_embeddings)
            )
        elif self.synthesizer_mode == "random":
            # Use fixed random patterns
            self.register_buffer(
                "random_attention",
                torch.randn(self.num_heads, config.max_position_embeddings, config.max_position_embeddings)
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Only need value projection for synthesizer
        value_states = self.v_proj(hidden_states)
        value_states = self._reshape(value_states, batch_size, seq_length)
        
        # Get synthesized attention weights
        if self.synthesizer_mode == "dense":
            attn_weights = self.attention_weights[:, :seq_length, :seq_length]
        elif self.synthesizer_mode == "factorized":
            attn_weights = torch.matmul(
                self.attention_left[:, :seq_length],
                self.attention_right[:, :, :seq_length]
            )
        else:  # random
            attn_weights = self.random_attention[:, :seq_length, :seq_length]
        
        # Expand for batch size
        attn_weights = attn_weights.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=hidden_states.device),
            diagonal=1
        ).bool()
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply user-provided attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attention_dropout(attn_probs)
        
        # Compute output
        attn_output = torch.matmul(attn_probs, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_probs if output_attentions else None, past_key_value
    
    def _reshape(self, tensor, batch_size, seq_length):
        """Reshape tensor for multi-head attention"""
        return tensor.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)


# Helper function for rotary embeddings
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings to query and key tensors"""
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)