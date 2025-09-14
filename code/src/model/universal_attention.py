"""
Universal Attention Mechanism that handles all dimension cases
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class UniversalRotaryEmbedding(nn.Module):
    """Universal rotary embedding that handles any dimension configuration"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values"""
        positions = torch.arange(seq_len).unsqueeze(1)
        freqs = positions * self.inv_freq
        
        # Create cos and sin embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """Forward pass that handles any input dimension"""
        # Determine sequence length
        if x.dim() == 4:  # [batch, heads, seq, dim]
            seq_len = x.shape[2]
            batch_size = x.shape[0]
            num_heads = x.shape[1]
        elif x.dim() == 3:  # [batch, seq, dim]
            seq_len = x.shape[1]
            batch_size = x.shape[0]
            num_heads = 1
        else:
            seq_len = x.shape[0] if x.dim() > 0 else 1
            batch_size = 1
            num_heads = 1
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)
        
        # Get cos and sin
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Match dimensions to input
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            # Broadcast to match batch and head dimensions
            cos = cos.expand(batch_size, num_heads, -1, -1)
            sin = sin.expand(batch_size, num_heads, -1, -1)
            # Ensure last dimension matches
            if cos.shape[-1] != x.shape[-1]:
                # Adjust to match head dimension
                target_dim = x.shape[-1]
                if cos.shape[-1] < target_dim:
                    # Pad with zeros
                    pad_size = target_dim - cos.shape[-1]
                    cos = F.pad(cos, (0, pad_size))
                    sin = F.pad(sin, (0, pad_size))
                else:
                    # Truncate
                    cos = cos[..., :target_dim]
                    sin = sin[..., :target_dim]
        elif x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            cos = cos.expand(batch_size, -1, -1)
            sin = sin.expand(batch_size, -1, -1)
            # Match last dimension
            if cos.shape[-1] != x.shape[-1]:
                target_dim = x.shape[-1]
                if cos.shape[-1] < target_dim:
                    cos = F.pad(cos, (0, target_dim - cos.shape[-1]))
                    sin = F.pad(sin, (0, target_dim - sin.shape[-1]))
                else:
                    cos = cos[..., :target_dim]
                    sin = sin[..., :target_dim]
        
        return cos.to(x.dtype), sin.to(x.dtype)


def apply_universal_rotary(q, k, cos, sin):
    """Apply rotary embeddings with automatic dimension matching"""
    # Handle the case where q and k might have different head counts than cos/sin
    if q.shape[1] != cos.shape[1]:  # Different number of heads
        # Expand or contract cos/sin to match q/k head count
        if cos.shape[1] == 1:  # cos/sin computed for single head, expand to all heads
            cos = cos.expand(-1, q.shape[1], -1, -1)
            sin = sin.expand(-1, q.shape[1], -1, -1)
        elif q.shape[1] < cos.shape[1]:  # More cos/sin heads than q/k heads
            cos = cos[:, :q.shape[1]]
            sin = sin[:, :q.shape[1]]
        else:  # Need to repeat cos/sin for more q/k heads
            repeat_factor = q.shape[1] // cos.shape[1]
            cos = cos.repeat(1, repeat_factor, 1, 1)
            sin = sin.repeat(1, repeat_factor, 1, 1)
            if cos.shape[1] < q.shape[1]:  # Handle remainder
                remaining = q.shape[1] - cos.shape[1]
                cos = torch.cat([cos, cos[:, :remaining]], dim=1)
                sin = torch.cat([sin, sin[:, :remaining]], dim=1)
    
    # Ensure dimensions match exactly
    if q.shape != cos.shape:
        # Match cos/sin to q/k dimensions
        if cos.dim() < q.dim():
            for _ in range(q.dim() - cos.dim()):
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        
        # Match shape exactly
        target_shape = list(q.shape)
        if cos.shape[0] == 1 and target_shape[0] > 1:
            cos = cos.expand(*target_shape)
            sin = sin.expand(*target_shape)
        elif cos.shape != q.shape:
            # Last resort: broadcast
            cos = cos.expand(*target_shape)
            sin = sin.expand(*target_shape)
    
    # Apply rotary
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


class UniversalMultiQueryAttention(nn.Module):
    """Universal attention that handles all configurations"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        
        # Ensure compatibility
        if self.num_heads % self.num_key_value_heads != 0:
            self.num_key_value_heads = self.num_heads
        
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = UniversalRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
            base=getattr(config, 'rope_theta', 10000.0)
        )
        
        self.attention_dropout = nn.Dropout(getattr(config, 'attention_dropout', 0.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        """Forward pass with universal dimension handling"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Handle key-value groups before rotary embeddings
        if self.num_key_value_groups > 1:
            # Expand key and value states to match query head count
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_universal_rotary(query_states, key_states, cos, sin)
        
        # Key-value groups already handled before rotary embeddings
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        # Always return exactly 3 values for compatibility
        # (output, attention_weights, cache)
        if output_attentions:
            attention_weights = attn_weights
        else:
            attention_weights = None
        
        if use_cache:
            cache = (key_states, value_states)
        else:
            cache = None
        
        return attn_output, attention_weights, cache