"""
Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) Implementation
With Flash Attention 2 support and memory optimizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import warnings

# Import apply_rotary_pos_emb function
def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    # Only warn once at module import, not every time
    import os
    if os.environ.get('FLASH_ATTN_WARNING_SHOWN') != '1':
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Flash Attention not available. Install with: pip install flash-attn --no-build-isolation")
        logger.info("Using standard attention implementation for now.")
        os.environ['FLASH_ATTN_WARNING_SHOWN'] = '1'

# Import PagedAttention if available
try:
    from ..optimization.paged_attention import PagedAttention, PagedAttentionConfig
    PAGED_ATTENTION_AVAILABLE = True
except ImportError:
    PAGED_ATTENTION_AVAILABLE = False

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) layer
    Supports standard MHA, MQA (single KV head), and GQA (grouped KV heads)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_type = config.attention_type  # mha, mqa, gqa
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
        
        # Check if PagedAttention should be used
        self.use_paged_attention = getattr(config, 'use_paged_attention', False) and PAGED_ATTENTION_AVAILABLE
        
        if self.use_paged_attention:
            # Initialize PagedAttention
            paged_config = PagedAttentionConfig(
                block_size=getattr(config, 'paged_attention_block_size', 16),
                num_blocks=getattr(config, 'paged_attention_num_blocks', 256),
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                device=getattr(config, 'device', 'cuda')
            )
            self.paged_attention = PagedAttention(
                self.hidden_size,
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim,
                config=paged_config
            )
            # Use paged attention projections
            self.q_proj = self.paged_attention.q_proj
            self.k_proj = self.paged_attention.k_proj
            self.v_proj = self.paged_attention.v_proj
            self.o_proj = self.paged_attention.o_proj
        else:
            # Standard projections
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=getattr(config, 'attention_bias', False))
        
        # Dropout
        self.attention_dropout = nn.Dropout(getattr(config, 'attention_dropout', 0.0))
        
        # Flash Attention settings
        self.use_flash_attn = config.use_flash_attn and FLASH_ATTENTION_AVAILABLE and not self.use_paged_attention
        if self.use_flash_attn:
            self.flash_attn_dropout = config.attention_dropout if self.training else 0.0
        
        # Initialize rotary embeddings
        from .moe_transformer import RotaryEmbedding
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Cache for inference (not used with PagedAttention)
        if not self.use_paged_attention:
            self.cache_k = None
            self.cache_v = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        mod_mask: Optional[torch.Tensor] = None,
        seq_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for MQA/GQA attention
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs for rotary embeddings
            past_key_value: Cached key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache key/value states
            cache_position: Cache position for inference
            mod_mask: Mixture of Depths mask
            seq_id: Sequence ID for PagedAttention
            
        Returns:
            attn_output: Attention output [batch_size, seq_len, hidden_size]
            attn_weights: Attention weights (if output_attentions=True)
            past_key_value: Updated cache (if use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use PagedAttention if enabled
        if self.use_paged_attention:
            attn_output, _ = self.paged_attention(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                seq_id=seq_id,
                use_cache=use_cache
            )
            return attn_output, None, None
        
        # Apply MoD mask if provided
        if mod_mask is not None:
            # Process only active tokens
            active_hidden_states = hidden_states[mod_mask]
            if active_hidden_states.shape[0] == 0:
                # No active tokens, return zeros
                return torch.zeros_like(hidden_states), None, past_key_value
        else:
            active_hidden_states = hidden_states
        
        # Compute QKV projections
        query_states = self.q_proj(active_hidden_states)
        key_states = self.k_proj(active_hidden_states)
        value_states = self.v_proj(active_hidden_states)
        
        # Reshape for multi-head attention
        if mod_mask is not None:
            num_active_tokens = active_hidden_states.shape[0]
            query_states = query_states.view(num_active_tokens, self.num_heads, self.head_dim)
            key_states = key_states.view(num_active_tokens, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(num_active_tokens, self.num_key_value_heads, self.head_dim)
        else:
            query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if available
        if self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(query_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # Repeat KV heads for GQA
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        if self.use_flash_attn and not output_attentions and mod_mask is None:
            # Use Flash Attention
            attn_output = self._flash_attention(query_states, key_states, value_states, attention_mask)
            attn_weights = None
        else:
            # Standard attention computation
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape output
        if mod_mask is not None:
            attn_output = attn_output.view(num_active_tokens, self.hidden_size)
            final_output = torch.zeros(batch_size, seq_len, self.hidden_size, device=hidden_states.device, dtype=attn_output.dtype)
            final_output[mod_mask] = attn_output
            attn_output = final_output
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads for grouped-query attention
        From: [batch, n_kv_heads, seq_len, head_dim] or [num_tokens, n_kv_heads, head_dim]
        To: [batch, n_heads, seq_len, head_dim] or [num_tokens, n_heads, head_dim]
        """
        if n_rep == 1:
            return hidden_states
        
        if hidden_states.dim() == 4:
            # Standard case: [batch, n_kv_heads, seq_len, head_dim]
            batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
            hidden_states = hidden_states[:, :, None, :, :].expand(
                batch, n_kv_heads, n_rep, seq_len, head_dim
            )
            return hidden_states.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
        else:
            # MoD case: [num_tokens, n_kv_heads, head_dim]
            num_tokens, n_kv_heads, head_dim = hidden_states.shape
            hidden_states = hidden_states[:, :, None, :].expand(
                num_tokens, n_kv_heads, n_rep, head_dim
            )
            return hidden_states.reshape(num_tokens, n_kv_heads * n_rep, head_dim)

    def _flash_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention using Flash Attention 2
        """
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        # Reshape for flash attention: [batch * seq_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)
        key_states = key_states.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)
        value_states = value_states.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)
        
        # Handle padding if attention mask is provided
        if attention_mask is not None:
            # Convert attention mask to indices for flash attention
            # This is a simplified version - full implementation would handle various mask types
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.flash_attn_dropout,
                causal=True,
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.flash_attn_dropout,
                causal=True,
            )
        
        # Reshape back
        attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        return attn_output

    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard scaled dot-product attention
        """
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask for autoregressive modeling
        if not use_cache:  # Only apply causal mask during training
            seq_len = query_states.shape[-2]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query_states.device), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            # Handle different tensor dimensions
            if attn_weights.dim() == 3 and attention_mask.dim() == 4:
                # MoD case: attn_weights is [num_tokens, num_heads, seq_len]
                # Reshape mask to match: take first batch dimension
                mask_reshaped = attention_mask[0, 0, :, :].unsqueeze(0)  # [1, seq_len, seq_len]
                if mask_reshaped.shape[-1] == attn_weights.shape[-1]:
                    attn_weights = attn_weights + mask_reshaped
            elif attn_weights.shape[-2:] == attention_mask.shape[-2:]:
                attn_weights = attn_weights + attention_mask
            elif attention_mask.shape[-1] == attn_weights.shape[-1]:
                # Broadcast mask if needed
                attn_weights = attn_weights + attention_mask.unsqueeze(1)
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output, attn_weights

class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for Flash Attention with additional features
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_flash_attn = config.use_flash_attn and FLASH_ATTENTION_AVAILABLE
        
        if not self.use_flash_attn:
            warnings.warn("Flash Attention requested but not available. Using standard attention.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention or fallback
        """
        if self.use_flash_attn:
            return self._flash_forward(query, key, value, attention_mask, dropout_p, is_causal, sliding_window)
        else:
            return self._standard_forward(query, key, value, attention_mask, dropout_p, is_causal)

    def _flash_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Flash Attention forward pass"""
        batch_size, seq_len = query.shape[:2]
        
        # Handle different input formats
        if query.dim() == 4:  # [batch, heads, seq, dim]
            query = query.transpose(1, 2).reshape(batch_size * seq_len, -1, query.shape[-1])
            key = key.transpose(1, 2).reshape(batch_size * seq_len, -1, key.shape[-1])
            value = value.transpose(1, 2).reshape(batch_size * seq_len, -1, value.shape[-1])
        
        # Apply flash attention
        if sliding_window is not None:
            # Use sliding window attention if available in flash_attn version
            output = flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                causal=is_causal,
                window_size=(sliding_window, -1) if sliding_window else (-1, -1),
            )
        else:
            output = flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                causal=is_causal,
            )
        
        # Reshape back if needed
        if output.shape[0] == batch_size * seq_len:
            output = output.view(batch_size, seq_len, -1, output.shape[-1]).transpose(1, 2)
        
        return output

    def _standard_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Standard attention fallback"""
        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if is_causal:
            seq_len = query.shape[-2]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        
        output = torch.matmul(attn_weights, value)
        
        return output

class OptimizedAttention(nn.Module):
    """
    Optimized attention implementation with multiple backend support
    Automatically selects the best implementation based on hardware
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Determine best attention implementation
        self.use_flash_attn = config.use_flash_attn and FLASH_ATTENTION_AVAILABLE
        self.use_xformers = False
        
        try:
            import xformers.ops
            self.use_xformers = True
        except ImportError:
            pass
        
        # Create attention layer
        self.attention = MultiQueryAttention(config)
        
        # Set implementation priority
        if self.use_flash_attn:
            self.attn_impl = "flash"
        elif self.use_xformers:
            self.attn_impl = "xformers"
        else:
            self.attn_impl = "standard"
        
        print(f"Using {self.attn_impl} attention implementation")

    def forward(self, *args, **kwargs):
        """Forward with optimized implementation"""
        return self.attention(*args, **kwargs)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors"""
    # Ensure cos and sin have compatible dimensions with q and k
    if cos.dim() != q.dim():
        # Expand cos and sin to match q and k dimensions
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
    
    # Ensure the sequence dimension matches
    if cos.shape[-2] != q.shape[-2]:
        seq_len = q.shape[-2]
        cos = cos[..., :seq_len, :]
        sin = sin[..., :seq_len, :]
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention - limits attention to a local window
    Reduces complexity from O(n²) to O(n*w) where w is window size
    """
    def __init__(self, config, window_size: int = 256):
        super().__init__()
        self.window_size = window_size
        self.base_attention = MultiQueryAttention(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Apply sliding window attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create sliding window mask
        if seq_len <= self.window_size:
            # If sequence is shorter than window, use standard attention
            return self.base_attention(hidden_states, attention_mask, **kwargs)
        
        # Create band matrix mask for sliding window
        window_mask = self._create_sliding_window_mask(seq_len, self.window_size, hidden_states.device)
        
        # Combine with existing mask if present
        if attention_mask is not None:
            attention_mask = attention_mask * window_mask
        else:
            attention_mask = window_mask
            
        return self.base_attention(hidden_states, attention_mask, **kwargs)
    
    def _create_sliding_window_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Create a sliding window attention mask"""
        # Create indices
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Create band matrix - attend only within window
        mask = (col_idx >= row_idx - window_size) & (col_idx <= row_idx)
        mask = mask.float()
        
        # Convert to attention mask format (0 for attend, -inf for mask)
        mask = (1.0 - mask) * torch.finfo(torch.float32).min
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        return mask


class SparseAttention(nn.Module):
    """
    Sparse Attention with BigBird/Longformer style patterns
    Combines local attention, global tokens, and random attention
    """
    def __init__(self, config, window_size: int = 256, num_global_tokens: int = 64, num_random_blocks: int = 3):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.num_random_blocks = num_random_blocks
        self.block_size = 64  # Size of random attention blocks
        
        self.base_attention = MultiQueryAttention(config)
        
        # Global token projections
        self.global_key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.global_value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Apply sparse attention pattern"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if seq_len <= self.window_size * 2:
            # For short sequences, use standard attention
            return self.base_attention(hidden_states, attention_mask, **kwargs)
        
        # Create sparse attention mask
        sparse_mask = self._create_sparse_attention_mask(seq_len, hidden_states.device)
        
        # Combine with existing mask
        if attention_mask is not None:
            attention_mask = attention_mask + sparse_mask
        else:
            attention_mask = sparse_mask
            
        # Process global tokens separately for efficiency
        if self.num_global_tokens > 0:
            global_hidden = hidden_states[:, :self.num_global_tokens]
            local_hidden = hidden_states[:, self.num_global_tokens:]
            
            # Global tokens attend to everything, everything attends to global
            global_attn_out, _, _ = self.base_attention(global_hidden, None, **kwargs)
            
            # Local tokens with sparse pattern
            local_attn_out, _, cache = self.base_attention(local_hidden, attention_mask[:, :, self.num_global_tokens:, self.num_global_tokens:], **kwargs)
            
            # Combine outputs
            attn_output = torch.cat([global_attn_out, local_attn_out], dim=1)
            return attn_output, None, cache
        else:
            return self.base_attention(hidden_states, attention_mask, **kwargs)
    
    def _create_sparse_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create BigBird-style sparse attention mask"""
        # Initialize with all -inf (no attention)
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        # 1. Local sliding window attention
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0
        
        # 2. Global tokens - first num_global_tokens attend to all and are attended by all
        if self.num_global_tokens > 0:
            mask[:self.num_global_tokens, :] = 0  # Global tokens attend to all
            mask[:, :self.num_global_tokens] = 0  # All attend to global tokens
        
        # 3. Random attention blocks
        if self.num_random_blocks > 0 and seq_len > self.block_size:
            for i in range(self.num_global_tokens, seq_len, self.block_size):
                # Select random blocks for this query block to attend to
                block_indices = torch.randperm(seq_len // self.block_size)[:self.num_random_blocks] * self.block_size
                for block_idx in block_indices:
                    end_idx = min(i + self.block_size, seq_len)
                    end_block = min(int(block_idx) + self.block_size, seq_len)
                    mask[i:end_idx, block_idx:end_block] = 0
        
        # Ensure causal mask for autoregressive models
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * float('-inf')
        mask = mask + causal_mask
        
        return mask.unsqueeze(0).unsqueeze(0)


class StreamingAttentionWithSinks(nn.Module):
    """
    Attention with sink tokens for streaming/continuous generation
    Maintains attention to initial tokens to prevent degradation
    """
    def __init__(self, config, num_sink_tokens: int = 4):
        super().__init__()
        self.num_sink_tokens = num_sink_tokens
        self.base_attention = MultiQueryAttention(config)
        self.sink_importance = nn.Parameter(torch.ones(num_sink_tokens))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Apply attention with sink tokens"""
        batch_size, seq_len, _ = hidden_states.shape
        
        if seq_len <= self.num_sink_tokens:
            return self.base_attention(hidden_states, attention_mask, **kwargs)
        
        # Create sink-aware attention mask
        sink_mask = self._create_sink_mask(seq_len, hidden_states.device)
        
        # Weight sink tokens by learned importance
        if self.training:
            hidden_states = hidden_states.clone()
            hidden_states[:, :self.num_sink_tokens] *= self.sink_importance.view(1, -1, 1)
        
        # Combine masks
        if attention_mask is not None:
            attention_mask = attention_mask + sink_mask
        else:
            attention_mask = sink_mask
            
        return self.base_attention(hidden_states, attention_mask, **kwargs)
    
    def _create_sink_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create attention mask that ensures sink tokens are always attended"""
        mask = torch.zeros((seq_len, seq_len), device=device)
        
        # All positions attend to sink tokens
        mask[:, :self.num_sink_tokens] = 0
        
        # Apply causal mask to rest
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask + causal_mask * float('-inf')
        
        return mask.unsqueeze(0).unsqueeze(0)


class ALiBiAttention(nn.Module):
    """
    Attention with Linear Biases (ALiBi) - no position embeddings needed
    Better length extrapolation than standard position embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.base_attention = MultiQueryAttention(config)
        
        # ALiBi slopes for each head
        slopes = self._get_alibi_slopes(self.num_heads)
        self.register_buffer('alibi_slopes', slopes)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Apply ALiBi attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create ALiBi bias
        alibi_bias = self._create_alibi_bias(seq_len, hidden_states.device)
        
        # Add to attention mask
        if attention_mask is not None:
            # Expand alibi_bias to match attention_mask dimensions
            if attention_mask.dim() == 4 and alibi_bias.dim() == 3:
                alibi_bias = alibi_bias.unsqueeze(0)
            attention_mask = attention_mask + alibi_bias
        else:
            attention_mask = alibi_bias.unsqueeze(0) if alibi_bias.dim() == 3 else alibi_bias
            
        # Disable rotary embeddings as ALiBi replaces them
        kwargs['position_ids'] = None
        
        return self.base_attention(hidden_states, attention_mask, **kwargs)
    
    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Calculate ALiBi slopes for each attention head"""
        # Geometric sequence of slopes
        slopes = torch.tensor([2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)])
        return slopes
    
    def _create_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create ALiBi positional bias matrix"""
        # Distance matrix
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        distance = positions - positions.transpose(0, 1)
        
        # Apply slopes to create per-head biases
        alibi_bias = distance.float().unsqueeze(0) * self.alibi_slopes.view(-1, 1, 1)
        
        # Mask future positions
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        alibi_bias = alibi_bias.masked_fill(causal_mask.bool(), float('-inf'))
        
        return alibi_bias


class CachedAttention(nn.Module):
    """
    Attention with pattern caching for repeated computations
    Useful for tasks with similar attention patterns
    """
    def __init__(self, config, cache_size: int = 32):
        super().__init__()
        self.config = config
        self.cache_size = cache_size
        self.base_attention = MultiQueryAttention(config)
        
        # Cache for attention patterns
        self.pattern_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache_lookup: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward with optional cache lookup"""
        
        if not use_cache_lookup or not self.training:
            return self.base_attention(hidden_states, attention_mask, **kwargs)
        
        # Create cache key from input shape and mask
        cache_key = self._create_cache_key(hidden_states, attention_mask)
        
        if cache_key in self.pattern_cache:
            # Cache hit - reuse attention pattern
            self.cache_hits += 1
            cached_pattern, cached_stats = self.pattern_cache[cache_key]
            return self._apply_cached_pattern(hidden_states, cached_pattern, cached_stats, **kwargs)
        
        # Cache miss - compute normally
        self.cache_misses += 1
        output, attn_weights, cache = self.base_attention(
            hidden_states, attention_mask, output_attentions=True, **kwargs
        )
        
        # Store in cache if there's room
        if len(self.pattern_cache) < self.cache_size and attn_weights is not None:
            pattern_stats = {
                'mean': attn_weights.mean(dim=[0, 1]),
                'std': attn_weights.std(dim=[0, 1])
            }
            self.pattern_cache[cache_key] = (attn_weights.detach(), pattern_stats)
        
        # Log cache statistics periodically
        if (self.cache_hits + self.cache_misses) % 100 == 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            print(f"Attention cache hit rate: {hit_rate:.2%}")
        
        return output, attn_weights, cache
    
    def _create_cache_key(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> str:
        """Create a cache key from input characteristics"""
        shape_key = f"{hidden_states.shape}"
        mask_key = f"{attention_mask.shape}_{attention_mask.sum().item()}" if attention_mask is not None else "none"
        return f"{shape_key}_{mask_key}"
    
    def _apply_cached_pattern(
        self,
        hidden_states: torch.Tensor,
        cached_pattern: torch.Tensor,
        pattern_stats: dict,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Apply cached attention pattern with slight variations"""
        # Add small noise to cached pattern for diversity
        noise = torch.randn_like(cached_pattern) * 0.01
        attention_weights = cached_pattern + noise
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply attention using cached weights
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to QKV
        q = self.base_attention.q_proj(hidden_states).view(batch_size, seq_len, self.base_attention.num_heads, -1).transpose(1, 2)
        v = self.base_attention.v_proj(hidden_states).view(batch_size, seq_len, self.base_attention.num_key_value_heads, -1).transpose(1, 2)
        
        # Repeat KV heads if needed
        if self.base_attention.num_key_value_groups > 1:
            v = self.base_attention._repeat_kv(v, self.base_attention.num_key_value_groups)
        
        # Apply cached attention pattern
        attn_output = torch.matmul(attention_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.base_attention.o_proj(attn_output)
        
        return attn_output, attention_weights, None


class xPosRotaryEmbedding(nn.Module):
    """
    Extrapolatable Position Embedding (xPos) - improved RoPE for length extrapolation
    Combines RoPE with exponential decay for better stability on long sequences
    """
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, scale_base: float = 512):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale_base = scale_base
        
        # Standard RoPE frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # xPos scale factors
        scale = (torch.arange(0, self.dim, 2).float() + 0.4 * self.dim) / (1.4 * self.dim)
        self.register_buffer("scale", scale, persistent=False)
        
        # Pre-compute for efficiency
        self._set_cos_sin_cache(max_position_embeddings)
        
    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cos and sin values with xPos scaling"""
        positions = torch.arange(seq_len).unsqueeze(1)
        freqs = positions * self.inv_freq
        
        # Standard cos and sin
        cos = freqs.cos()
        sin = freqs.sin()
        
        # xPos scaling
        power = (positions - seq_len // 2) / self.scale_base
        scale_cos = self.scale ** power.clamp(-1, 1)
        scale_sin = self.scale ** power.clamp(-1, 1)
        
        # Apply scaling
        cos = cos * scale_cos
        sin = sin * scale_sin
        
        # Cache values
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given positions with xPos scaling"""
        # Get sequence length from input
        if x.dim() == 4:  # [batch, heads, seq, dim]
            batch_size = x.shape[0]
            seq_len = x.shape[2]
        elif x.dim() == 3:  # [batch, seq, dim]
            batch_size = x.shape[0]
            seq_len = x.shape[1]
        else:
            batch_size = 1
            seq_len = x.shape[0] if x.dim() > 0 else position_ids.shape[-1]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)
        
        # Get cached values
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Reshape to match input dimensions
        if x.dim() == 4:  # [batch, heads, seq, dim]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
            # Expand to match batch and head dimensions
            cos = cos.expand(batch_size, -1, -1, -1)
            sin = sin.expand(batch_size, -1, -1, -1)
        else:
            cos = cos.unsqueeze(0)  # [1, seq, dim]
            sin = sin.unsqueeze(0)
            if batch_size > 1:
                cos = cos.expand(batch_size, -1, -1)
                sin = sin.expand(batch_size, -1, -1)
        
        return cos.to(x.dtype), sin.to(x.dtype)


class LinearAttention(nn.Module):
    """
    Linear Attention - O(n) complexity using kernel trick
    Efficient for very long sequences but may sacrifice some quality
    """
    def __init__(self, config, feature_map: str = "elu"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.feature_map = feature_map
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Feature map parameters
        if feature_map == "elu":
            self.feature_fn = lambda x: F.elu(x) + 1
        elif feature_map == "relu":
            self.feature_fn = F.relu
        else:
            self.feature_fn = lambda x: x
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Apply linear attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to QKV
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map
        q = self.feature_fn(q)
        k = self.feature_fn(k)
        
        # Linear attention: (Q @ (K^T @ V)) instead of ((Q @ K^T) @ V)
        # This changes complexity from O(n²d) to O(nd²)
        
        # Compute K^T @ V first: [batch, heads, dim, seq] @ [batch, heads, seq, dim] = [batch, heads, dim, dim]
        kv = torch.matmul(k.transpose(-2, -1), v)
        
        # Apply causal mask by cumulative sum
        if attention_mask is not None or kwargs.get('is_causal', True):
            # For causal attention, use cumulative sum trick
            # Accumulate kv products over sequence dimension
            kv_cumsum = kv.cumsum(dim=-2)  # [batch, heads, seq, dim]
            
            # Compute normalizer: sum of keys
            k_cumsum = k.cumsum(dim=-2)  # [batch, heads, seq, dim]
            normalizer = (q * k_cumsum).sum(dim=-1, keepdim=True) + 1e-8
            
            # Compute attention output
            attn_output = torch.matmul(q, kv_cumsum.transpose(-2, -1)) / normalizer
        else:
            # Non-causal attention
            attn_output = torch.matmul(q, kv) / (torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-8)
            
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, None


class RingAttention(nn.Module):
    """
    Ring Attention - Distributed attention for extremely long sequences
    Splits sequence across devices in a ring topology
    """
    def __init__(self, config, num_devices: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_devices = num_devices or torch.cuda.device_count()
        self.base_attention = MultiQueryAttention(config)
        
        # Ring communication parameters
        self.chunk_size = getattr(config, 'ring_chunk_size', 512)
        self.overlap_size = getattr(config, 'ring_overlap_size', 64)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Apply ring attention across devices"""
        
        if self.num_devices <= 1 or not torch.cuda.is_available():
            # Fall back to standard attention
            return self.base_attention(hidden_states, attention_mask, **kwargs)
            
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Split sequence into chunks for each device
        chunk_size = (seq_len + self.num_devices - 1) // self.num_devices
        chunks = []
        
        for i in range(self.num_devices):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk = hidden_states[:, start:end]
            chunks.append(chunk)
            
        # TODO: Implement actual ring communication
        # This would involve:
        # 1. Distributing chunks to different devices
        # 2. Computing partial attention on each device
        # 3. Ring-passing key-value pairs between devices
        # 4. Aggregating results
        
        # For now, fall back to standard attention
        return self.base_attention(hidden_states, attention_mask, **kwargs)


def create_attention_layer(config, attention_type: Optional[str] = None) -> nn.Module:
    """
    Factory function to create the appropriate attention layer based on config
    """
    if attention_type is None:
        attention_type = getattr(config, 'attention_variant', 'standard')
        
    if attention_type == 'sliding_window':
        window_size = getattr(config, 'sliding_window_size', 256)
        return SlidingWindowAttention(config, window_size=window_size)
    elif attention_type == 'sparse':
        return SparseAttention(
            config,
            window_size=getattr(config, 'sparse_window_size', 256),
            num_global_tokens=getattr(config, 'sparse_global_tokens', 64),
            num_random_blocks=getattr(config, 'sparse_random_blocks', 3)
        )
    elif attention_type == 'streaming':
        num_sink_tokens = getattr(config, 'num_sink_tokens', 4)
        return StreamingAttentionWithSinks(config, num_sink_tokens=num_sink_tokens)
    elif attention_type == 'alibi':
        return ALiBiAttention(config)
    elif attention_type == 'cached':
        cache_size = getattr(config, 'attention_cache_size', 32)
        return CachedAttention(config, cache_size=cache_size)
    elif attention_type == 'linear':
        feature_map = getattr(config, 'linear_attention_feature_map', 'elu')
        return LinearAttention(config, feature_map=feature_map)
    elif attention_type == 'ring':
        return RingAttention(config)
    else:
        # Default to standard multi-query attention
        return MultiQueryAttention(config)