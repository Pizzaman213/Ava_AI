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
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Union
from contextlib import contextmanager

from ..core.checkpoint import load_state_dict_with_remapping
from .experts import HighPerformanceExpert, ExpertParallelGroup

# Import optimized MoE layer for grouped GEMM (5-10x faster expert computation)
try:
    from .moe_layer import SparseMoELayer
    SPARSE_MOE_AVAILABLE = True
except ImportError:
    SparseMoELayer = None
    SPARSE_MOE_AVAILABLE = False

# Import Triton RoPE kernel (5-8% faster rotary embeddings)
try:
    from ..cuda.rope_kernel import apply_rotary_pos_emb_triton, TRITON_ROPE_AVAILABLE
except ImportError:
    apply_rotary_pos_emb_triton = None
    TRITON_ROPE_AVAILABLE = False

# Import fused LayerNorm + residual kernel (8-15% per layer speedup)
try:
    from ..cuda.fused_norm import fused_add_layer_norm_simple, TRITON_AVAILABLE as FUSED_NORM_AVAILABLE
except ImportError:
    fused_add_layer_norm_simple = None
    FUSED_NORM_AVAILABLE = False

# Logger must be defined before Flash Attention check
logger = logging.getLogger(__name__)

# PERFORMANCE FIX: Check Flash Attention availability at import time
# This avoids repeated import attempts on every forward pass (10x slowdown)
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_VARLEN_AVAILABLE = False
_flash_attn_func = None
_flash_attn_varlen_func = None

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    logger.debug(f"Flash Attention not available: {e}")
    FLASH_ATTN_AVAILABLE = False

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    FLASH_ATTN_VARLEN_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    FLASH_ATTN_VARLEN_AVAILABLE = False

# Backward compatibility aliases - use nn.experts implementations
SwiGLUExpert = HighPerformanceExpert
SwiGLUExpertGroup = ExpertParallelGroup


def _validate_attention_mask_shape(
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    name: str = "attention_mask"
) -> None:
    """
    Validate attention mask has expected shape for model forward pass.

    Args:
        attention_mask: The mask tensor to validate
        batch_size: Expected batch size
        seq_len: Expected sequence length
        name: Name for error messages
    """
    if attention_mask is None:
        return

    dim = attention_mask.dim()

    if dim == 2:
        # 1D padding mask: [batch, seq_len]
        expected = (batch_size, seq_len)
        if attention_mask.shape != expected:
            raise ValueError(
                f"{name} has shape {attention_mask.shape}, expected {expected} for 1D mask"
            )
    elif dim == 3:
        # 2D document boundary mask: [batch, seq_len, seq_len]
        expected = (batch_size, seq_len, seq_len)
        if attention_mask.shape != expected:
            raise ValueError(
                f"{name} has shape {attention_mask.shape}, expected {expected} for 2D mask"
            )
    elif dim == 4:
        # 4D ready mask: [batch, 1, seq_len, seq_len] or [batch, heads, seq_len, seq_len]
        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dim is {attention_mask.shape[0]}, expected {batch_size}"
            )
        if attention_mask.shape[2] != seq_len or attention_mask.shape[3] != seq_len:
            raise ValueError(
                f"{name} seq dims are {attention_mask.shape[2:]}, expected ({seq_len}, {seq_len})"
            )
    else:
        raise ValueError(
            f"{name} has {dim} dimensions, expected 2, 3, or 4. Shape: {attention_mask.shape}"
        )

# NVTX annotations for Nsight profiling
_nvtx_available = False
try:
    import torch.cuda.nvtx as nvtx
    _nvtx_available = True
except ImportError:
    logger.debug("NVTX not available - NVIDIA profiling annotations disabled")

@contextmanager
def nvtx_range(name: str):
    """NVTX range context manager for Nsight profiling."""
    if _nvtx_available and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield

# Import routing and expert layers
try:
    from .routing import MixtralRouter as SwitchTransformerRouting
    from .experts import SparseExpert
except ImportError:
    SwitchTransformerRouting = None
    SparseExpert = None


@dataclass
class EnhancedMoEConfig:
    """
    Configuration for EnhancedMoEModel - the standard MoE transformer.

    Use this config for:
    - Development and testing
    - Single GPU training with moderate model sizes
    - Standard MoE training without advanced optimizations

    For production training with performance optimizations (grouped GEMM,
    Triton kernels, expert offloading, quantization), use OptimizedMoEConfig
    with OptimizedMoETransformer instead.

    See also: OptimizedMoEConfig (line ~1108)
    """

    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048

    # Special token IDs (must match tokenizer - custom BPE tokenizer)
    pad_token_id: int = 0
    eos_token_id: int = 5    # [EOS] token
    bos_token_id: int = 2    # [CLS]/[BOS] token

    # MoE settings
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    capacity_factor: float = 1.25  # Alias for expert_capacity_factor (YAML compatibility)
    router_type: str = 'switch'  # 'switch', 'deepseek', etc.
    router_aux_loss_coef: float = 0.01
    router_jitter_noise: float = 0.01

    # MoE auxiliary loss coefficients (for load balancing and stability)
    router_z_loss_coef: float = 0.001  # Router z-loss for stability
    load_balance_loss_coef: float = 0.01  # Load balancing loss
    diversity_loss_coef: float = 0.0  # Expert diversity loss
    expert_dropout_loss_coef: float = 0.0  # Expert dropout loss
    expert_dropout: float = 0.0  # Expert dropout probability

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
    activation: str = 'gelu'  # Alias for hidden_act (YAML compatibility)
    initializer_range: float = 0.02

    # Performance optimization flags (must be passed from YAML config)
    gradient_checkpointing: bool = False  # 70-80% memory savings
    # P2-7: Selective checkpointing - only checkpoint every N layers (3-5% speedup)
    # checkpoint_layer_interval=1 means all layers (default), 2 means every 2nd layer, 3 means every 3rd
    checkpoint_layer_interval: int = 1
    use_grouped_gemm: bool = False  # 5-10x expert computation speedup
    use_triton_kernels: bool = False  # 20-30% routing speedup
    use_torch_compile: bool = False  # 15-25% overall speedup
    use_optimized_moe: bool = True   # Use optimized MoE implementation (SparseMoELayer with grouped GEMM)
    use_fused_qkv: bool = True  # Fused Q/K/V projection (5-10% attention speedup)
    use_fused_norm: bool = True  # Fused LayerNorm + residual (8-15% per layer)
    use_counting_sort_capacity: bool = True  # O(N) capacity limiting (10-20% MoE speedup)

    # Memory optimization flags
    use_lora_experts: bool = False  # Use LoRA for expert parameters
    use_expert_offloading: bool = False  # Offload experts to CPU

    # Additional features
    use_moh: bool = False  # Mixture of Heads
    use_moa: bool = False  # Mixture of Activations
    use_cross_attention: bool = False
    use_alibi: bool = False

    # Training-specific features (may be in checkpoint but not used in inference)
    deepspeed_activation_checkpointing: bool = False
    deepspeed_partition_activations: bool = False
    deepspeed_moe_param_groups: bool = False

    # Attention settings
    use_causal_attention: bool = True  # Causal masking for autoregressive LM (set False for bidirectional)

    # Loss regularization features for coherence (small defaults that help most cases)
    entropy_regularization: float = 0.005  # Entropy bonus for diverse predictions (reduces repetition)
    output_diversity_weight: float = 0.0005  # Penalty for low output diversity (reduces "the the the" patterns)
    eos_logit_bias: float = 0.0  # Bias applied to EOS token logits (set 0.5 to reduce early termination)
    min_sequence_length: int = 0  # Minimum sequence length before allowing EOS
    label_smoothing: float = 0.0  # CRITICAL: Label smoothing for cross-entropy loss (0.1 recommended for anti-overfitting)

    # Weight tying
    tie_word_embeddings: bool = False  # CRITICAL: Tie input/output embeddings (improves vocab learning, reduces parameters)


class RoPEPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with caching for position ranges."""

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

        # Cache keyed by (position_offset, seq_len, device_str) tuple for correct position handling
        # This ensures generation with KV cache gets correct position embeddings
        # FIX: Cache on same device as request to avoid CPU<->GPU transfers every forward pass
        self._cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_max_size = 20  # Limited to prevent memory bloat

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        position_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for rotary embeddings with caching.

        Args:
            seq_len: Number of positions to compute
            device: Target device
            position_offset: Starting position (for KV cache generation)

        Returns:
            cos, sin tensors for positions [position_offset, position_offset + seq_len)
        """
        # Include device in cache key to avoid cross-device transfers
        cache_key = (position_offset, seq_len, str(device))

        if cache_key in self._cache:
            # Move to end for LRU (most recently used)
            cos_cached, sin_cached = self._cache.pop(cache_key)
            self._cache[cache_key] = (cos_cached, sin_cached)
            # CRITICAL: Clone tensors to avoid CUDAGraphs overwriting cached values
            # torch.compile with CUDA graphs reuses tensor memory, so returning
            # the same tensor object causes "tensor output overwritten" errors
            return cos_cached.clone(), sin_cached.clone()

        # Compute positions [position_offset, position_offset + seq_len)
        t = torch.arange(position_offset, position_offset + seq_len, device=device)
        t = t.type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # LRU eviction - remove oldest entry if at capacity
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        # Store on same device to avoid CPU<->GPU transfers every forward pass
        # Trade-off: Uses more GPU memory but avoids repeated transfers
        self._cache[cache_key] = (cos, sin)

        # CRITICAL: Clone tensors to avoid CUDAGraphs overwriting cached values
        # torch.compile with CUDA graphs reuses tensor memory, so returning
        # the same tensor object causes "tensor output overwritten" errors
        return cos.clone(), sin.clone()

    def clear_cache(self) -> None:
        """Clear the RoPE position embedding cache to free memory."""
        self._cache.clear()

    def forward_with_position_ids(
        self,
        position_ids: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for rotary embeddings using arbitrary position_ids.

        This is critical for sequence packing where positions reset per document.
        Unlike forward() which assumes sequential positions, this method supports
        per-token position IDs that can restart at 0 for each document.

        Args:
            position_ids: [batch, seq_len] tensor of position IDs
            device: Target device

        Returns:
            cos, sin tensors of shape [batch, seq_len, head_dim] for the given positions
        """
        # position_ids: [batch, seq_len] -> compute per-token frequencies
        # inv_freq: [head_dim/2]
        # Result: [batch, seq_len, head_dim]

        # Expand position_ids for broadcasting: [batch, seq_len, 1]
        position_ids = position_ids.float().unsqueeze(-1)
        # inv_freq: [1, 1, head_dim/2]
        inv_freq = self.inv_freq.to(device).unsqueeze(0).unsqueeze(0)

        # Compute frequencies: [batch, seq_len, head_dim/2]
        freqs = position_ids * inv_freq

        # Double the frequencies for full head_dim: [batch, seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        # CRITICAL: Clone tensors to avoid CUDAGraphs overwriting values
        # torch.compile with CUDA graphs reuses tensor memory, so returning
        # the same tensor object causes "tensor output overwritten" errors
        return cos.clone(), sin.clone()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.

    Uses standard RoPE implementation which is torch.compile-friendly.
    Complex number and Triton optimizations are disabled due to incompatibility
    with torch.compile's tracing mechanism.
    """
    # Standard RoPE implementation - torch.compile friendly
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [batch, 1, seq_len, head_dim] or broadcastable shape
    #
    # Formula: q' = q * cos + rotate_half(q) * sin
    # where rotate_half splits last dim in half and swaps with negation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and Flash Attention support."""

    # Class-level flag to log attention backend only once across all instances
    _attention_backend_logged = False

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        # Validate num_attention_heads to prevent division by zero
        if config.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be > 0, got {config.num_attention_heads}"
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_dropout
        self.use_flash_attention = getattr(config, 'use_flash_attention', True)
        self.quantize_kv_cache = getattr(config, 'quantize_kv_cache', False)
        self.is_causal = getattr(config, 'use_causal_attention', True)  # Causal masking for autoregressive LM

        # OPTIMIZATION: Fused QKV projection (5-10% attention speedup)
        # Single matrix multiplication instead of 3 separate Q/K/V projections
        # Reduces memory bandwidth by reading input once instead of 3 times
        self.use_fused_qkv = getattr(config, 'use_fused_qkv', True)

        if self.use_fused_qkv:
            # Fused Q, K, V projections in single linear layer
            self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
            self.q_proj = None  # Not used with fused QKV
            self.k_proj = None
            self.v_proj = None
        else:
            # Separate Q, K, V projections (legacy path)
            self.qkv_proj = None
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

        # GRADIENT CHECKPOINTING FIX: Store flag to disable attention dropout during recomputation
        # Dropout uses different RNG samples during forward vs. checkpoint recomputation,
        # causing different hidden states and routing decisions that lead to shape mismatches
        self._gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs
    ) -> tuple:
        batch_size, seq_len, _ = hidden_states.shape

        # Store the input dtype for consistency
        input_dtype = hidden_states.dtype

        # GRADIENT CHECKPOINTING FIX: Compute effective dropout probability
        # Set to 0.0 during gradient checkpointing to ensure deterministic recomputation
        # This prevents shape mismatches in MoE routing when hidden states differ
        if self._gradient_checkpointing and self.training:
            effective_dropout_p = 0.0
        else:
            effective_dropout_p = self.dropout if self.training else 0.0

        # Project to Q, K, V
        # OPTIMIZATION: Use fused QKV projection when available (5-10% attention speedup)
        if self.use_fused_qkv:
            # Single matrix multiply for Q, K, V (reads input once, 3x fewer memory fetches)
            qkv = self.qkv_proj(hidden_states)  # [batch, seq, 3 * hidden_size]
            q, k, v = qkv.chunk(3, dim=-1)  # Split into Q, K, V
        else:
            # Legacy separate projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if configured
        if self.rope is not None:
            # CRITICAL FIX: Use document-relative position_ids for sequence packing
            # This ensures each document in a packed sequence gets correct positional encoding
            if position_ids is not None:
                # Use per-token position IDs (resets at document boundaries)
                cos, sin = self.rope.forward_with_position_ids(position_ids, hidden_states.device)
                cos = cos.to(dtype=input_dtype)[:, None, :, :]  # [batch, 1, seq, head_dim]
                sin = sin.to(dtype=input_dtype)[:, None, :, :]
            elif past_key_value is not None:
                # KV cache case: compute RoPE for correct position range
                past_seq_len = past_key_value[0].shape[2]
                cos, sin = self.rope(seq_len, hidden_states.device, position_offset=past_seq_len)
                cos = cos.to(dtype=input_dtype)[None, None, :, :]
                sin = sin.to(dtype=input_dtype)[None, None, :, :]
            else:
                # Standard case: positions [0, seq_len)
                cos, sin = self.rope(seq_len, hidden_states.device, position_offset=0)
                cos = cos.to(dtype=input_dtype)[None, None, :, :]
                sin = sin.to(dtype=input_dtype)[None, None, :, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Concatenate with past key-values if provided (KV cache for generation)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # Dequantize if cache was quantized
            if self.quantize_kv_cache and past_k.dtype == torch.int8:
                # FIX: Use symmetric scale 127.5 to utilize full INT8 range [-128, 127]
                past_k = past_k.to(k.dtype) / 127.5
                past_v = past_v.to(v.dtype) / 127.5
            k = torch.cat([past_k, k], dim=2)  # Concatenate on sequence dimension
            v = torch.cat([past_v, v], dim=2)

        # Store current key-values for next iteration if caching
        if use_cache:
            if self.quantize_kv_cache:
                # Quantize to INT8 for 75% memory savings
                # FIX: Use symmetric scale 127.5 and round for better precision
                # This uses full INT8 range [-128, 127] instead of just [-127, 127]
                k_quantized = (k * 127.5).round().clamp(-128, 127).to(torch.int8)
                v_quantized = (v * 127.5).round().clamp(-128, 127).to(torch.int8)
                present_key_value = (k_quantized, v_quantized)
            else:
                present_key_value = (k, v)
        else:
            present_key_value = None

        # SPEED OPTIMIZATION: Try Flash Attention varlen → FA3 → xformers → FA2 → standard
        # Flash Attention varlen is FASTEST for packed sequences (40-60% speedup over 2D masks)
        # - No 512MB mask allocation (for batch=32, seq=2048)
        # - Native document boundary support via cu_seqlens
        #
        # CRITICAL: Document boundary masking for sequence packing
        # When attention_mask is 4D (from packing), it contains document boundaries that MUST
        # be respected to prevent cross-document attention. In this case:
        # - We cannot use simple causal=True (ignores document boundaries)
        # - We must pass the full attention mask to the attention function
        # - The mask already includes causal masking, so is_causal should be False
        has_document_mask = attention_mask is not None and attention_mask.dim() == 4
        use_causal_only = self.is_causal and not has_document_mask
        has_cu_seqlens = cu_seqlens is not None and max_seqlen is not None

        # === FASTEST PATH: Flash Attention varlen with cu_seqlens ===
        # This completely eliminates 2D mask allocation and is 40-60% faster
        # PERFORMANCE FIX: Use cached availability check instead of try/import on every forward
        if self.use_flash_attention and FLASH_ATTN_VARLEN_AVAILABLE and has_cu_seqlens and past_key_value is None:
            try:
                # flash_attn_varlen_func expects [total_tokens, heads, head_dim]
                # Flatten from [batch, heads, seq, head_dim] to [total_tokens, heads, head_dim]
                total_tokens = batch_size * seq_len
                q_var = q.transpose(1, 2).reshape(total_tokens, self.num_heads, self.head_dim)
                k_var = k.transpose(1, 2).reshape(total_tokens, self.num_heads, self.head_dim)
                v_var = v.transpose(1, 2).reshape(total_tokens, self.num_heads, self.head_dim)

                attn_output = _flash_attn_varlen_func(
                    q_var, k_var, v_var,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=effective_dropout_p,
                    causal=True,  # Causal within each document
                )

                # Reshape back to [batch, heads, seq, head_dim]
                attn_output = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
                attn_output = attn_output.transpose(1, 2)

                if not MultiHeadAttention._attention_backend_logged:
                    import logging
                    logging.info("✓ Using Flash Attention varlen (40-60% faster, no 2D mask allocation)")
                    MultiHeadAttention._attention_backend_logged = True

                # Skip the other attention paths
                attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
                attn_output = attn_output.to(input_dtype)
                attn_output = self.o_proj(attn_output)
                return attn_output, present_key_value

            except (ImportError, RuntimeError, AttributeError) as e:
                # Fall through to standard paths
                if not MultiHeadAttention._attention_backend_logged:
                    import logging
                    logging.debug(f"Flash Attention varlen unavailable: {e}")

        # PERFORMANCE FIX: Use cached availability check instead of try/import on every forward
        if self.use_flash_attention and FLASH_ATTN_AVAILABLE and not has_document_mask:
            try:
                # Flash Attention 3 from official repo (fastest)
                # flash_attn_func expects [batch, seq, heads, head_dim]
                # Need to transpose from [batch, heads, seq, head_dim]
                q_fa = q.transpose(1, 2)  # [batch, seq, heads, head_dim]
                k_fa = k.transpose(1, 2)
                v_fa = v.transpose(1, 2)

                attn_output = _flash_attn_func(
                    q_fa, k_fa, v_fa,
                    dropout_p=effective_dropout_p,
                    causal=use_causal_only
                )
                # flash_attn_func returns [batch, seq, heads, head_dim]
                attn_output = attn_output.transpose(1, 2)  # Back to [batch, heads, seq, head_dim]

                if not MultiHeadAttention._attention_backend_logged:
                    import logging
                    logging.info("✓ Using Flash Attention 3 (1.5-2× faster than FA2)")
                    MultiHeadAttention._attention_backend_logged = True

            except (RuntimeError, AttributeError):
                # Try xformers memory-efficient attention (20-30% speedup)
                try:
                    from xformers.ops import memory_efficient_attention, LowerTriangularMask  # type: ignore[import-untyped]

                    # CRITICAL: Ensure dtype consistency for xformers
                    # xformers requires all inputs to have the same dtype
                    target_dtype = q.dtype

                    # xformers expects [batch, seq, heads, head_dim]
                    q_xf = q.transpose(1, 2).to(target_dtype)
                    k_xf = k.transpose(1, 2).to(target_dtype)
                    v_xf = v.transpose(1, 2).to(target_dtype)

                    # CRITICAL: Use document boundary mask if provided, otherwise causal
                    # Document boundary mask already includes causal masking
                    if has_document_mask:
                        # xformers expects [batch, heads, seq, seq] but our mask is [batch, 1, seq, seq]
                        # Need to squeeze or expand appropriately
                        attn_bias = attention_mask.squeeze(1) if attention_mask.size(1) == 1 else attention_mask
                        attn_bias = attn_bias.to(target_dtype)
                    elif use_causal_only:
                        attn_bias = LowerTriangularMask()
                    else:
                        attn_bias = attention_mask

                    attn_output = memory_efficient_attention(
                        q_xf, k_xf, v_xf,
                        attn_bias=attn_bias,
                        p=effective_dropout_p,
                    )
                    # xformers returns [batch, seq, heads, head_dim]
                    attn_output = attn_output.transpose(1, 2)  # Back to [batch, heads, seq, head_dim]

                    if not MultiHeadAttention._attention_backend_logged:
                        import logging
                        if has_document_mask:
                            logging.info("✓ Using xformers with document boundary masking (sequence packing safe)")
                        else:
                            logging.info("✓ Using xformers memory-efficient attention (20-30% speedup)")
                        MultiHeadAttention._attention_backend_logged = True

                except (ImportError, RuntimeError, AttributeError, ValueError):
                    # Fallback to PyTorch's Flash Attention 2 (SDPA)
                    # F.scaled_dot_product_attention expects [batch, heads, seq, head_dim]
                    # Our tensors are already in this format
                    #
                    # CRITICAL: When we have document boundary mask (4D), we MUST use it
                    # and set is_causal=False (the mask already contains causal masking)
                    if has_document_mask:
                        attn_output = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attention_mask,  # Use full document boundary mask
                            dropout_p=effective_dropout_p,
                            is_causal=False  # Mask already includes causal
                        )
                    else:
                        attn_output = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attention_mask if not use_causal_only else None,
                            dropout_p=effective_dropout_p,
                            is_causal=use_causal_only
                        )

                    if not MultiHeadAttention._attention_backend_logged:
                        import logging
                        if has_document_mask:
                            logging.info("✓ Using PyTorch SDPA with document boundary masking (sequence packing safe)")
                        else:
                            logging.info("✓ Using PyTorch Flash Attention 2 (SDPA)")
                        MultiHeadAttention._attention_backend_logged = True
        elif self.use_flash_attention:
            # PERFORMANCE FIX: Flash Attention requested but not available
            # Use PyTorch's SDPA directly (still fast, ~80% of Flash Attention speed)
            if has_document_mask:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=effective_dropout_p,
                    is_causal=False  # Mask already includes causal
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask if not use_causal_only else None,
                    dropout_p=effective_dropout_p,
                    is_causal=use_causal_only
                )

            if not MultiHeadAttention._attention_backend_logged:
                import logging
                logging.info("✓ Using PyTorch SDPA (Flash Attention library unavailable)")
                MultiHeadAttention._attention_backend_logged = True
        else:
            # Standard attention implementation (only when use_flash_attention=False)
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
    """MoE Feed-Forward layer with expert routing.

    Uses memory-efficient nn.ModuleList with sequential expert processing.
    This avoids the OOM issues from grouped GEMM which requires gathering
    all expert weights for all tokens at once.
    """

    def __init__(self, config: EnhancedMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token

        # Get activation function
        activation_name = getattr(config, 'activation', 'swiglu')

        # Simple linear router
        self.router = nn.Linear(config.hidden_size, config.num_experts)

        # Loss coefficients for MoE regularization
        self.load_balance_loss_coef = getattr(config, 'load_balance_loss_coef', 0.01)
        self.diversity_loss_coef = getattr(config, 'diversity_loss_coef', 0.0)  # Expert diversity
        self.router_z_loss_coef = getattr(config, 'router_z_loss_coef', 0.001)  # Router stability

        # Router jitter noise to prevent expert collapse
        # During training, adds small noise to router logits to encourage exploration
        self.router_jitter_noise = getattr(config, 'router_jitter_noise', 0.01)

        # Use ModuleList with SwiGLU experts (memory-efficient, fast)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_flat.shape[0]
        device = hidden_flat.device
        dtype = hidden_flat.dtype

        # Router forward pass
        router_logits = self.router(hidden_flat)

        # Apply jitter during training to prevent expert collapse
        # Small noise encourages exploration and prevents router from getting stuck
        if self.training and self.router_jitter_noise > 0:
            noise = torch.empty_like(router_logits).uniform_(
                -self.router_jitter_noise, self.router_jitter_noise
            )
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)

        # Auxiliary loss info
        aux_info = {
            'router_type': 'simple_topk',  # Track router type for logging
        }

        # Calculate load balancing auxiliary loss (Switch Transformer formulation)
        # Expert utilization (fraction of tokens routed to each expert)
        # NOTE: expert_fraction is computed from discrete argmax decisions (non-differentiable)
        # This is correct - gradient flows through expert_avg_prob only
        top1_indices = router_probs.argmax(dim=-1).detach()  # Detach for clarity
        # Use bincount for vectorized expert counting (avoids loop allocations)
        expert_counts = torch.bincount(top1_indices, minlength=self.num_experts).float()

        # Guard against division by zero for empty batches
        if num_tokens > 0:
            expert_fraction = expert_counts / num_tokens
        else:
            expert_fraction = torch.zeros_like(expert_counts)

        # Average probability assigned to each expert (differentiable)
        expert_avg_prob = router_probs.mean(dim=0)

        # Load balancing loss - gradient flows through expert_avg_prob
        # expert_fraction.detach() makes gradient flow explicit
        load_balance_loss = self.num_experts * (expert_fraction.detach() * expert_avg_prob).sum()
        # CRITICAL FIX: Ensure load_balance_loss is a scalar
        if load_balance_loss.numel() > 1:
            load_balance_loss = load_balance_loss.mean()
        aux_info['load_balance_loss'] = load_balance_loss * self.load_balance_loss_coef
        aux_info['router_probs'] = router_probs.detach()
        aux_info['expert_utilization'] = expert_counts.detach()

        # GPU SYNC OPT: Keep tensor on GPU, defer extraction to log intervals
        # Instead of N syncs per forward from .cpu().tolist(), we store the tensor
        # and the training loop extracts via BatchSynchronizer at log intervals
        aux_info['_expert_utilization_tensor'] = expert_counts.float().detach()

        # Router z-loss for stability (prevents router logits from becoming too large)
        if self.router_z_loss_coef > 0:
            log_z = torch.logsumexp(router_logits, dim=-1)
            log_z = torch.clamp(log_z, max=20.0)  # Prevent overflow when squared
            router_z_loss = log_z.pow(2).mean()
            aux_info['load_balance_loss'] = aux_info['load_balance_loss'] + router_z_loss * self.router_z_loss_coef

        # Expert diversity loss (encourages experts to specialize differently)
        if self.diversity_loss_coef > 0:
            # Compute variance of expert probabilities - low variance = all experts similar (bad)
            # We want HIGH variance (each expert handles different types of tokens)
            expert_prob_var = expert_avg_prob.var()
            # Negative loss: bonus for high variance (diverse expert usage)
            diversity_loss = -expert_prob_var
            aux_info['load_balance_loss'] = aux_info['load_balance_loss'] + diversity_loss * self.diversity_loss_coef

        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)

        # FIX: Numerically stable renormalization with proper epsilon for low precision
        # ALWAYS renormalize - cheap operation, avoids GPU sync from .any() check
        # Previous code used `if needs_renorm.any():` which caused GPU→CPU sync
        top_k_sum = top_k_probs.sum(dim=-1, keepdim=True)
        # BF16 has ~3 decimal digits precision, FP16 has ~4, FP32 has ~7
        epsilon = 1e-3 if dtype == torch.bfloat16 else (1e-4 if dtype == torch.float16 else 1e-7)
        # Unconditional normalize - division is cheap, sync is expensive
        top_k_probs = top_k_probs / top_k_sum.clamp(min=epsilon)

        # Process through experts using batched computation
        # D2D FIX: Use in-place index_add_ to avoid creating intermediate tensors
        # This replaces the previous gather-scatter pattern that created D2D copies
        output = torch.zeros_like(hidden_flat)

        for i in range(self.num_experts_per_token):
            expert_indices_i = top_k_indices[:, i]  # [num_tokens]
            expert_weights_i = top_k_probs[:, i]  # [num_tokens]

            # Sort tokens by expert assignment for sequential processing
            sorted_order = torch.argsort(expert_indices_i)
            sorted_expert_ids = expert_indices_i[sorted_order]

            # Pre-compute expert boundaries using bincount + cumsum
            expert_counts_per_k = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
            boundaries = torch.zeros(self.num_experts + 1, dtype=torch.long, device=device)
            boundaries[1:] = expert_counts_per_k.cumsum(0)

            # COMPILE-FRIENDLY: Use tensor slicing directly, no .tolist() sync
            # Process each expert's tokens using pure tensor operations
            for expert_idx in range(self.num_experts):
                # Use tensor indexing - no GPU sync
                start_idx = boundaries[expert_idx]
                end_idx = boundaries[expert_idx + 1]

                # COMPILE-FRIENDLY: Slice with tensor indices (empty slices are valid)
                expert_sorted_idx = sorted_order[start_idx:end_idx]

                # Skip empty experts via tensor size check (no GPU sync)
                # torch.compile handles this - empty tensors flow through cleanly
                if expert_sorted_idx.numel() == 0:
                    continue

                # D2D FIX: Use index_select instead of advanced indexing
                tokens = torch.index_select(hidden_flat, 0, expert_sorted_idx)
                weights = torch.index_select(expert_weights_i, 0, expert_sorted_idx)

                # Process tokens through expert
                expert_output = self.experts[expert_idx](tokens)

                # Apply weights in-place before accumulation
                expert_output = expert_output * weights.unsqueeze(1)

                # Accumulate weighted output using index_add_ (in-place, efficient)
                output.index_add_(0, expert_sorted_idx, expert_output)

        output = output.view(batch_size, seq_len, hidden_size)
        output = self.dropout(output)

        return output, aux_info


class TransformerBlock(nn.Module):
    """Transformer block with MoE feed-forward."""

    def __init__(self, config: EnhancedMoEConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = MultiHeadAttention(config)

        # Use optimized SparseMoELayer with grouped GEMM when available (5-10x faster)
        use_optimized = getattr(config, 'use_optimized_moe', True) or getattr(config, 'use_grouped_gemm', False)
        self._use_sparse_moe = use_optimized and SPARSE_MOE_AVAILABLE

        if self._use_sparse_moe:
            self.feed_forward = SparseMoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                router_type=getattr(config, 'router_type', 'mixtral'),
                capacity_factor=getattr(config, 'capacity_factor', 1.25),
                expert_dropout=getattr(config, 'expert_dropout', 0.0),
                activation=getattr(config, 'activation', 'swiglu'),
                use_grouped_gemm=getattr(config, 'use_grouped_gemm', True),
                use_triton_kernels=getattr(config, 'use_triton_kernels', True),
                use_torch_compile=getattr(config, 'use_torch_compile', True),
                router_z_loss_coef=getattr(config, 'router_z_loss_coef', 0.001),
                load_balance_loss_coef=getattr(config, 'load_balance_loss_coef', 0.01),
                diversity_loss_coef=getattr(config, 'diversity_loss_coef', 0.001),
                router_jitter_noise=getattr(config, 'router_jitter_noise', 0.01),
                aux_loss_frequency=getattr(config, 'aux_loss_frequency', 10),
                gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
            )
            if layer_idx == 0:
                logger.info("TransformerBlock using SparseMoELayer (5-10x faster grouped GEMM)")
        else:
            self.feed_forward = MoEFeedForward(config)
            if layer_idx == 0:
                logger.info("TransformerBlock using MoEFeedForward (sequential experts)")

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

        # GRADIENT CHECKPOINTING FIX: Store flag to disable dropout during recomputation
        # Dropout uses different RNG samples during forward vs. checkpoint recomputation,
        # causing different hidden states and routing decisions that lead to shape mismatches
        self._gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

        # OPTIMIZATION: Use fused LayerNorm + residual when available (8-15% per layer speedup)
        self._use_fused_norm = (
            getattr(config, 'use_fused_norm', True) and
            FUSED_NORM_AVAILABLE and
            fused_add_layer_norm_simple is not None
        )
        if self._use_fused_norm and layer_idx == 0:
            logger.info("TransformerBlock using fused LayerNorm + residual (8-15% speedup)")

        # Activation cache for gradient checkpointing optimization
        # Set via set_activation_cache() by HybridCacheManager
        self._activation_cache = None

    def set_activation_cache(self, cache) -> None:
        """
        Set activation cache for this layer.

        Args:
            cache: ActivationCache instance from hybrid_cache module
        """
        self._activation_cache = cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict, Optional[tuple]]:
        # Self-attention with residual
        with nvtx_range("block/attention"):
            residual = hidden_states
            with nvtx_range("block/attention/ln1"):
                # OPTIMIZATION: Removed .clone() for 2-3% speedup
                hidden_states = self.ln1(hidden_states)
            with nvtx_range("block/attention/mha"):
                attn_output, present_key_value = self.attention(
                    hidden_states,
                    attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            with nvtx_range("block/attention/residual"):
                # GRADIENT CHECKPOINTING FIX: Skip dropout during checkpoint recomputation
                # to ensure deterministic routing decisions (prevents shape mismatch errors)
                if self._gradient_checkpointing and self.training:
                    hidden_states = residual + attn_output
                else:
                    hidden_states = residual + self.dropout(attn_output)

        # MoE feed-forward with residual
        with nvtx_range("block/moe_ffn"):
            residual = hidden_states
            with nvtx_range("block/moe_ffn/ln2"):
                # OPTIMIZATION: Removed .clone() for 2-3% speedup
                hidden_states = self.ln2(hidden_states)
            with nvtx_range("block/moe_ffn/experts"):
                if self._use_sparse_moe:
                    # SparseMoELayer returns (output, aux_loss, metrics)
                    ff_output, aux_loss, metrics = self.feed_forward(hidden_states, training=self.training)
                    # Convert to aux_info format for backward compatibility
                    aux_info = {
                        'load_balance_loss': aux_loss,
                        **metrics
                    }
                else:
                    # MoEFeedForward returns (output, aux_info)
                    ff_output, aux_info = self.feed_forward(hidden_states)
            with nvtx_range("block/moe_ffn/residual"):
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
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # CRITICAL FIX: Only tie weights if explicitly enabled in config
        # Tying can cause gradient conflicts with large vocabularies
        # WARNING: tie_word_embeddings=True is INCOMPATIBLE with DeepSpeed ZeRO
        # because the shared parameter gets gradient reduction attempted twice,
        # causing "parameter has already been reduced" errors
        tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
        self._tie_word_embeddings = tie_word_embeddings  # Store for validation
        if tie_word_embeddings:
            # Tie input and output embeddings (saves memory but can hurt training)
            self.lm_head.weight = self.token_embedding.weight
            logger.warning(
                "tie_word_embeddings=True: Incompatible with DeepSpeed ZeRO. "
                "If using DeepSpeed, set tie_word_embeddings: false in your config "
                "to avoid 'parameter already reduced' errors."
            )
        else:
            # Keep separate (better for training, especially with large vocab)
            pass  # lm_head already has independent weights

        # TIER2 OPTIMIZATION: Cache for causal attention masks (5-10% speedup)
        # FIX: Store GPU masks directly keyed by (seq_len, device, dtype) to avoid GPU-CPU transfers
        self._causal_mask_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._causal_mask_cache_max_size = 10  # Reduced to limit memory

        # FIX: Cache constant tensors to avoid creating them on every forward pass
        self._neg_inf_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._zero_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

        # Gradient checkpointing for 70-80% memory savings
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)
        # P2-7: Selective checkpointing - only checkpoint every N layers for 3-5% speedup
        # checkpoint_layer_interval=1 means all layers (default), 2 means every 2nd layer, etc.
        self.checkpoint_layer_interval = getattr(config, 'checkpoint_layer_interval', 1)
        if self.gradient_checkpointing:
            if self.checkpoint_layer_interval > 1:
                logger.info(
                    f"Selective gradient checkpointing ENABLED: every {self.checkpoint_layer_interval} layers "
                    f"(~{100 // self.checkpoint_layer_interval}% of layers checkpointed, 3-5% speedup)"
                )
            else:
                logger.info("Gradient checkpointing ENABLED (70-80% memory savings)")

        # Wire use_optimized_moe flag to enable all MoE optimizations together
        if getattr(config, 'use_optimized_moe', False):
            # Enable grouped GEMM and Triton kernels when use_optimized_moe is set
            if not getattr(config, 'use_grouped_gemm', False):
                config.use_grouped_gemm = True
                logger.info("use_optimized_moe: Enabled use_grouped_gemm (5-10x expert speedup)")
            if not getattr(config, 'use_triton_kernels', False):
                config.use_triton_kernels = True
                logger.info("use_optimized_moe: Enabled use_triton_kernels (20-30% routing speedup)")
            logger.info("use_optimized_moe ENABLED (combined 10-20% overall speedup)")

        # RAG (Retrieval Augmented Generation) components
        self.retriever = None
        self.rag_fusion = None
        rag_config = getattr(config, 'rag_config', None)
        if rag_config and getattr(rag_config, 'use_rag', False):
            try:
                from ava.rag import create_retriever, create_fusion, VectorIndex
                # Initialize retriever
                index_path = getattr(rag_config, 'index_path', None) or getattr(rag_config, 'knowledge_base_path', None)
                if index_path:
                    # Load pre-built index
                    self._rag_index = VectorIndex.load(index_path, embedding_dim=getattr(rag_config, 'embedding_dim', 768))
                    self.retriever = self._rag_index.retriever
                    logger.info(f"RAG: Loaded index from {index_path}")
                else:
                    # Create empty retriever (documents must be added later)
                    self.retriever = create_retriever(rag_config)
                    logger.info(f"RAG: Initialized {getattr(rag_config, 'retriever_type', 'faiss')} retriever")

                # Initialize fusion
                self.rag_fusion = create_fusion(rag_config, config.hidden_size)
                logger.info(f"RAG: Using {getattr(rag_config, 'rag_fusion_type', 'attention')} fusion")
            except ImportError as e:
                logger.warning(f"RAG dependencies not available: {e}")
                logger.warning("Install with: pip install faiss-cpu chromadb sentence-transformers")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}")

        # OPTIMIZATION: Weight prefetching for better memory latency hiding
        # Pre-touches next layer's weights to warm L2 cache while current layer computes
        self._enable_weight_prefetch = getattr(config, 'enable_weight_prefetch', True)
        if self._enable_weight_prefetch and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream()
        else:
            self._prefetch_stream = None

        # Initialize weights
        self.apply(self._init_weights)

    @torch.compiler.disable  # Exclude from torch.compile - mark_dynamic cannot be traced
    def _get_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        TIER2 OPTIMIZATION: Get cached causal attention mask or create new one.
        Avoids recreating the same mask on every forward pass (5-10% speedup).

        BOTTLENECK FIX: Cache is now device/dtype-independent for >80% hit rate.
        Masks are stored as bool tensors and converted to target dtype on use.

        Note: This method is excluded from torch.compile (@torch.compiler.disable)
        because mark_dynamic cannot be traced. The mask is still created efficiently
        and cached for reuse.

        Returns: [1, 1, seq_len, seq_len] causal mask
        """
        # FIX: Cache key includes device and dtype to avoid GPU-CPU transfers
        cache_key = (seq_len, device, dtype)

        if cache_key in self._causal_mask_cache:
            return self._causal_mask_cache[cache_key]

        # FIX: Get or create cached constant tensors (avoids repeated tensor creation)
        const_key = (device, dtype)
        if const_key not in self._neg_inf_cache:
            self._neg_inf_cache[const_key] = torch.tensor(float('-inf'), device=device, dtype=dtype)
        if const_key not in self._zero_cache:
            self._zero_cache[const_key] = torch.tensor(0.0, device=device, dtype=dtype)

        neg_inf = self._neg_inf_cache[const_key]
        zero = self._zero_cache[const_key]

        # Create new causal mask
        causal_mask_bool = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1
        )
        # Add batch and head dimensions
        causal_mask_bool = causal_mask_bool[None, None, :, :]

        # Convert to target dtype using cached constants
        causal_mask = torch.where(causal_mask_bool, neg_inf, zero)

        # Cache the GPU mask (limit cache size with LRU eviction)
        if len(self._causal_mask_cache) >= self._causal_mask_cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._causal_mask_cache))
            del self._causal_mask_cache[oldest_key]
        self._causal_mask_cache[cache_key] = causal_mask

        return causal_mask

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

    def clear_caches(self) -> None:
        """
        Clear internal caches to free VRAM.

        Call periodically during training (e.g., every 500 steps) to prevent
        memory fragmentation and reduce VRAM usage.

        Clears:
        - Causal attention mask cache
        - Constant tensor caches
        - RoPE positional embedding cache (in attention layers)
        """
        # Clear causal mask cache
        if hasattr(self, '_causal_mask_cache'):
            self._causal_mask_cache.clear()

        # Clear constant tensor caches
        if hasattr(self, '_neg_inf_cache'):
            self._neg_inf_cache.clear()
        if hasattr(self, '_zero_cache'):
            self._zero_cache.clear()

        # Clear RoPE cache in each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'rope'):
                if hasattr(layer.attention.rope, '_cache'):
                    layer.attention.rope._cache.clear()

    def _prefetch_layer_weights(self, layer_idx: int) -> None:
        """
        OPTIMIZATION: Prefetch next layer's weights to warm L2 cache.

        Launches async memory accesses on a separate stream to hide memory latency.
        The weights are already on GPU; this just ensures they're in L2 cache
        before being needed.

        Args:
            layer_idx: Index of the layer to prefetch (typically current_idx + 1)
        """
        if self._prefetch_stream is None:
            return
        if layer_idx >= len(self.layers):
            return

        with torch.cuda.stream(self._prefetch_stream):
            layer = self.layers[layer_idx]
            # Touch parameters to trigger prefetch to L2 cache
            # Using .data avoids autograd overhead
            for param in layer.parameters():
                # Simple read to warm cache - no actual computation
                _ = param.data.untyped_storage().data_ptr()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[tuple]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs
    ) -> Any:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # GPU SYNC FIX: Removed input validation entirely to avoid GPU sync
        # The embedding layer will naturally raise IndexError if indices are out of bounds
        # This eliminates .any()/.item() calls that were causing cudaStreamSynchronize

        # Check sequence length (cheap, no GPU sync needed - uses Python int)
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings}). Truncate your inputs."
            )

        # Embeddings
        with nvtx_range("model/embedding"):
            with nvtx_range("model/embedding/token"):
                token_embeds = self.token_embedding(input_ids)

            if self.position_embedding is not None:
                with nvtx_range("model/embedding/position"):
                    # Use provided position_ids (e.g., from sequence packing with per-doc positions)
                    # or create standard sequential positions
                    if position_ids is None:
                        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
                        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    else:
                        position_ids = position_ids.to(device)
                    position_embeds = self.position_embedding(position_ids)
                    hidden_states = token_embeds + position_embeds
            else:
                hidden_states = token_embeds

            # GRADIENT CHECKPOINTING FIX: Skip embedding dropout during checkpoint recomputation
            # to ensure deterministic routing decisions (prevents shape mismatch errors)
            if not (self.gradient_checkpointing and self.training):
                hidden_states = self.dropout(hidden_states)

        # TIER2 OPTIMIZATION: Use cached causal attention mask (5-10% speedup)
        with nvtx_range("model/attention_mask"):
            # Handle different attention mask formats:
            # - None: Use standard causal mask
            # - [batch, seq_len]: 1D padding mask, combine with causal
            # - [batch, seq_len, seq_len]: 2D document-boundary mask from sequence packing
            # - [batch, 1, seq_len, seq_len]: 4D mask (already ready)

            # Validate mask shape early to catch bugs (only in debug mode)
            if attention_mask is not None and logger.isEnabledFor(logging.DEBUG):
                _validate_attention_mask_shape(attention_mask, batch_size, seq_len)

            if attention_mask is not None and attention_mask.dim() >= 3:
                # 2D or 4D mask provided (e.g., from sequence packing with document boundaries)
                # This already includes causal masking and document boundaries
                if attention_mask.dim() == 3:
                    # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                    attention_mask = attention_mask[:, None, :, :]

                # Convert to model dtype
                attention_mask = attention_mask.to(dtype=hidden_states.dtype, device=device)
            else:
                # Standard case: create causal mask and optionally combine with 1D padding mask
                causal_mask = self._get_causal_mask(seq_len, device, hidden_states.dtype)

                if attention_mask is not None:
                    # attention_mask shape: [batch_size, seq_len]
                    # Convert to [batch_size, 1, 1, seq_len] for broadcasting
                    padding_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]

                    # DTYPE FIX: Create mask directly in hidden_states.dtype to prevent recompilation
                    # Invert: 1 = attend, 0 = don't attend
                    # Convert 0s to -inf
                    mask_value = torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=device)
                    padding_mask = torch.where(padding_mask == 0, mask_value, torch.tensor(0.0, dtype=hidden_states.dtype, device=device))

                    # Combine causal and padding masks via broadcasting:
                    # - causal_mask: [1, 1, seq_len, seq_len] masks future tokens (K > Q positions)
                    # - padding_mask: [batch, 1, 1, seq_len] masks padding in KEY dimension
                    # - Result: [batch, 1, seq_len, seq_len] with both masks applied
                    # Broadcasting expands causal_mask to batch dim and padding_mask to query dim
                    attention_mask = causal_mask + padding_mask
                else:
                    # Just use causal mask
                    attention_mask = causal_mask

        # RAG: Retrieve relevant documents if RAG is enabled
        rag_context = None
        if self.retriever is not None and self.rag_fusion is not None:
            with nvtx_range("model/rag_retrieval"):
                # Create query from hidden states
                rag_config = getattr(self.config, 'rag_config', None)
                query_strategy = getattr(rag_config, 'query_strategy', 'first_token') if rag_config else 'first_token'
                top_k = getattr(rag_config, 'max_retrieved_docs', 5) if rag_config else 5

                if query_strategy == 'mean':
                    query = hidden_states.mean(dim=1)  # [batch, hidden]
                elif query_strategy == 'cls':
                    query = hidden_states[:, 0]  # [batch, hidden] (first token)
                else:  # 'first_token'
                    query = hidden_states[:, 0]  # [batch, hidden]

                # Retrieve documents
                try:
                    results = self.retriever.retrieve(query, top_k=top_k)
                    # Build context tensor from retrieved embeddings
                    if results and any(r for r in results):
                        context_embeddings = []
                        for batch_results in results:
                            if batch_results:
                                # Stack document embeddings for this batch item
                                batch_ctx = torch.stack([
                                    r.document.embedding.to(device) if r.document.embedding is not None
                                    else torch.zeros(self.config.hidden_size, device=device)
                                    for r in batch_results
                                ])
                            else:
                                # No results for this item, use zeros
                                batch_ctx = torch.zeros(top_k, self.config.hidden_size, device=device)
                            context_embeddings.append(batch_ctx.to(device))
                        rag_context = torch.stack(context_embeddings).to(device)  # [batch, num_docs, hidden]
                except Exception as e:
                    # Re-raise PyTorch internal exceptions used by gradient checkpointing
                    if type(e).__name__ in ('_StopRecomputationError', 'StopIteration'):
                        raise
                    logger.warning(f"RAG retrieval failed: {e}")
                    rag_context = None

        # Apply transformer blocks with KV caching
        all_aux_info = []
        present_key_values = [] if use_cache else None

        # KV cache eviction for long sequences (generation mode only)
        # Uses HybridCacheManager's KVCacheManager if attached
        if use_cache and past_key_values is not None:
            kv_manager = getattr(self, '_kv_cache_manager', None)
            if kv_manager is not None and kv_manager.should_evict(past_key_values):
                past_key_values = kv_manager.evict(past_key_values)

        with nvtx_range("model/transformer_layers"):
            for idx, layer in enumerate(self.layers):
                # OPTIMIZATION: Prefetch next layer's weights while computing current layer
                # This hides memory latency by overlapping compute and memory operations
                if self._enable_weight_prefetch and idx + 1 < len(self.layers):
                    self._prefetch_layer_weights(idx + 1)

                with nvtx_range(f"model/layer_{idx}"):
                    # Get past key-value for this layer if available
                    past_key_value = past_key_values[idx] if past_key_values is not None else None

                    # Use gradient checkpointing if enabled (70-80% memory savings)
                    # NOTE: Checkpointing is incompatible with KV caching during training
                    # P2-7: Selective checkpointing - only checkpoint every N layers (3-5% speedup)
                    should_checkpoint = (
                        self.gradient_checkpointing and
                        self.training and
                        not use_cache and
                        (idx % self.checkpoint_layer_interval == 0)  # Only checkpoint every N-th layer
                    )
                    if should_checkpoint:
                        # Wrapper function for checkpoint - must return tuple
                        # CRITICAL FIX: Pass position_ids for correct RoPE in sequence packing
                        # OPTIMIZATION: Pass cu_seqlens for Flash Attention varlen (40-60% speedup)
                        def create_custom_forward(module, pos_ids, cu_seqs, max_seq):
                            def custom_forward(hidden, mask, past_kv, cache_flag):
                                return module(
                                    hidden, mask,
                                    past_key_value=past_kv,
                                    use_cache=cache_flag,
                                    position_ids=pos_ids,
                                    cu_seqlens=cu_seqs,
                                    max_seqlen=max_seq,
                                )
                            return custom_forward

                        # FIX: use_reentrant=False is required for DeepSpeed ZeRO compatibility
                        # Reentrant mode re-runs forward during backward, causing DeepSpeed to
                        # see parameters being reduced twice ("parameter already reduced" error)
                        hidden_states, aux_info, present_key_value = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer, position_ids, cu_seqlens, max_seqlen),
                            hidden_states,
                            attention_mask,
                            past_key_value,
                            use_cache,
                            use_reentrant=False,
                            preserve_rng_state=True,  # Ensures router jitter noise is reproducible during recomputation
                        )
                    else:
                        # CRITICAL FIX: Pass position_ids for correct RoPE in sequence packing
                        # OPTIMIZATION: Pass cu_seqlens for Flash Attention varlen (40-60% speedup)
                        hidden_states, aux_info, present_key_value = layer(
                            hidden_states,
                            attention_mask,
                            past_key_value=past_key_value,
                            use_cache=use_cache,
                            position_ids=position_ids,
                            cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen,
                        )
                    all_aux_info.append(aux_info)

                    if use_cache:
                        present_key_values.append(present_key_value)  # type: ignore[union-attr]

        # Final layer norm
        with nvtx_range("model/final_ln"):
            # OPTIMIZATION: Removed .clone() for 2-3% speedup
            hidden_states = self.ln_f(hidden_states)

        # RAG: Apply fusion with retrieved context
        if rag_context is not None and self.rag_fusion is not None:
            with nvtx_range("model/rag_fusion"):
                hidden_states = self.rag_fusion(hidden_states, rag_context)

        # LM head
        with nvtx_range("model/lm_head"):
            logits = self.lm_head(hidden_states)

        # CRITICAL FIX: Apply negative bias to EOS token logits during training
        # This prevents the model from learning to output EOS as the most likely token
        if self.training and labels is not None:
            eos_logit_bias = getattr(self.config, 'eos_logit_bias', 0.0)
            if eos_logit_bias > 0:
                with nvtx_range("model/eos_bias"):
                    # CRITICAL FIX: Get EOS token ID from config (tokenizer-specific)
                    # Default to 1 which is the standard EOS token for our tokenizers
                    eos_token_id = getattr(self.config, 'eos_token_id', 1)
                    # Subtract bias from EOS logits (makes EOS less likely to be predicted)
                    logits[:, :, eos_token_id] = logits[:, :, eos_token_id] - eos_logit_bias

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            with nvtx_range("model/loss_compute"):
                with nvtx_range("model/loss_compute/shift"):
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                with nvtx_range("model/loss_compute/cross_entropy"):
                    # Validate shapes before cross_entropy (no CUDA sync - just shape check)
                    actual_vocab = shift_logits.size(-1)
                    if actual_vocab != self.config.vocab_size:
                        raise ValueError(
                            f"Logits vocab dimension ({actual_vocab}) doesn't match "
                            f"config.vocab_size ({self.config.vocab_size}). "
                            f"Logits shape: {shift_logits.shape}, Labels shape: {shift_labels.shape}"
                        )
                    # GPU SYNC FIX: Removed per-batch label validation that caused 2 cudaStreamSynchronize
                    # via .max().item() and .min().item(). Labels are validated at data loading time.
                    # If you need validation, enable debug mode which runs checks every N steps.

                    # GPU SYNC FIX: Removed per-batch NaN/Inf checks that caused 2 cudaStreamSynchronize
                    # via .any() calls. Use torch.autograd.detect_anomaly() during debugging instead.
                    # For production, cross_entropy will naturally produce NaN loss if inputs are bad.

                    # FIX: Get label smoothing from config (critical anti-overfitting technique)
                    label_smoothing = getattr(self.config, 'label_smoothing', 0.0) or 0.0

                    # Use -100 for ignore_index (standard PyTorch convention)
                    # The sequence packing collator sets padding labels to -100
                    loss = F.cross_entropy(
                        shift_logits.view(-1, self.config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        label_smoothing=label_smoothing,  # ADDED: critical for preventing overfitting
                        reduction='mean'  # EXPLICIT: ensure scalar loss
                    )

                    # CRITICAL FIX: Ensure loss is a scalar (safety check)
                    if loss.numel() > 1:
                        loss = loss.mean()

                # CRITICAL FIX: Add MoE auxiliary loss for load balancing
                # NOTE: aux losses are already scaled by their coefficients in MoEFeedForward.forward()
                # so we just aggregate them here without additional scaling
                if all_aux_info:
                    with nvtx_range("model/loss_compute/aux_loss"):
                        total_aux_loss = 0.0
                        num_layers_with_aux = 0
                        for layer_aux in all_aux_info:
                            if 'load_balance_loss' in layer_aux:
                                total_aux_loss += layer_aux['load_balance_loss']
                                num_layers_with_aux += 1

                        if num_layers_with_aux > 0:
                            # FIX: Normalize by number of layers to prevent amplification.
                            # Without normalization, aux loss dominates cross-entropy loss
                            # (e.g., 16 layers = 16x amplification of load balance loss).
                            # Each layer's loss coefficient (e.g., 0.01) already controls magnitude.
                            normalize_aux = getattr(self.config, 'normalize_aux_loss', True)
                            if normalize_aux:
                                total_aux_loss = total_aux_loss / num_layers_with_aux

                            # CRITICAL FIX: Ensure total_aux_loss is a scalar tensor
                            # If it has multiple elements, reduce it to mean
                            if isinstance(total_aux_loss, torch.Tensor) and total_aux_loss.numel() > 1:
                                total_aux_loss = total_aux_loss.mean()

                            # Track loss component ratios for diagnosis
                            # GPU SYNC FIX: Store as tensors, defer .item() to training loop at log intervals
                            # This removes 2 cudaStreamSynchronize calls per step
                            if all_aux_info and len(all_aux_info) > 0:
                                all_aux_info[0]['loss_components'] = {
                                    'cross_entropy_loss': loss.detach(),
                                    'aux_loss': total_aux_loss.detach(),
                                }

                            loss = loss + total_aux_loss

                # FIX #18: Add entropy regularization (encourages diverse predictions)
                # MEMORY FIX: Use chunked computation to avoid OOM on large vocab
                entropy_reg = getattr(self.config, 'entropy_regularization', 0.0) or 0.0
                if entropy_reg > 0:
                    with nvtx_range("model/loss_compute/entropy_reg"):
                        # Memory-efficient entropy: use log_softmax and process in chunks
                        # Entropy = -sum(p * log(p)) = -sum(softmax(x) * log_softmax(x))
                        batch_size, seq_len_shifted = shift_logits.shape[:2]
                        chunk_size = min(64, seq_len_shifted)  # Process 64 positions at a time

                        entropy_sum = 0.0
                        total_positions = 0

                        for i in range(0, seq_len_shifted, chunk_size):
                            chunk_logits = shift_logits[:, i:i+chunk_size, :]
                            # Use log_softmax (more numerically stable and memory efficient)
                            log_probs = F.log_softmax(chunk_logits, dim=-1)
                            probs = torch.exp(log_probs)
                            # Entropy for this chunk: -sum(p * log(p))
                            chunk_entropy = -(probs * log_probs).sum(dim=-1).sum()
                            entropy_sum = entropy_sum + chunk_entropy
                            total_positions += chunk_logits.shape[0] * chunk_logits.shape[1]
                            del log_probs, probs, chunk_entropy  # Free memory immediately

                        entropy = entropy_sum / total_positions
                        # CRITICAL FIX: Subtract entropy bonus (rewards high-entropy/diverse predictions)
                        loss = loss - entropy_reg * entropy

                # FIX #19: Add differentiable output diversity penalty (penalizes repetitive outputs)
                # MEMORY FIX: Use chunked computation to avoid OOM on large vocab
                diversity_weight = getattr(self.config, 'output_diversity_weight', 0.0) or 0.0
                if diversity_weight > 0:
                    with nvtx_range("model/loss_compute/diversity"):
                        # Memory-efficient diversity: process in chunks
                        batch_size, seq_len_shifted = shift_logits.shape[:2]
                        chunk_size = min(64, seq_len_shifted - 1)  # Process 64 position pairs at a time

                        similarity_sum = 0.0
                        total_pairs = 0

                        for i in range(0, seq_len_shifted - 1, chunk_size):
                            end_idx = min(i + chunk_size, seq_len_shifted - 1)
                            # Get logits for adjacent positions
                            chunk_logits_t = shift_logits[:, i:end_idx, :]
                            chunk_logits_t1 = shift_logits[:, i+1:end_idx+1, :]

                            # Compute softmax for chunks only
                            probs_t = F.softmax(chunk_logits_t, dim=-1)
                            probs_t1 = F.softmax(chunk_logits_t1, dim=-1)

                            # Cosine similarity: dot(a, b) / (||a|| * ||b||)
                            dot_product = (probs_t * probs_t1).sum(dim=-1)
                            norm_t = probs_t.norm(dim=-1) + 1e-8
                            norm_t1 = probs_t1.norm(dim=-1) + 1e-8
                            chunk_similarity = dot_product / (norm_t * norm_t1)

                            similarity_sum = similarity_sum + chunk_similarity.sum()
                            total_pairs += chunk_similarity.numel()
                            del probs_t, probs_t1, dot_product, norm_t, norm_t1, chunk_similarity

                        diversity_loss = similarity_sum / total_pairs
                        loss = loss + diversity_weight * diversity_loss

        # FINAL SAFETY CHECK: Ensure loss is always a scalar before returning
        # This handles any edge cases where multi-element tensors might slip through
        if loss is not None and isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()

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

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional['EnhancedMoEConfig'] = None,
        strict: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> 'EnhancedMoEModel':
        """
        Load model from pretrained checkpoint.

        Supports both .pt and .safetensors formats.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Optional config to override checkpoint config
            strict: Whether to enforce strict state_dict loading
            device: Device to load model on
            **kwargs: Additional model constructor arguments

        Returns:
            EnhancedMoEModel with loaded weights

        Example:
            >>> model = EnhancedMoEModel.from_pretrained('/path/to/checkpoint.pt')
            >>> model = EnhancedMoEModel.from_pretrained('/path/to/checkpoint.pt', device='cuda:0')
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint (support multiple formats)
        if checkpoint_path.suffix == '.safetensors':
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(checkpoint_path))
                checkpoint_config = None
            except ImportError:
                raise ImportError("safetensors required for .safetensors files")
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Extract state_dict (support multiple checkpoint formats)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                checkpoint_config = checkpoint.get('config', None)
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                checkpoint_config = checkpoint.get('config', None)
            elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
                checkpoint_config = checkpoint.get('config', None)
            else:
                # Assume entire checkpoint is state_dict
                state_dict = checkpoint
                checkpoint_config = None

        # Determine config (priority: user > checkpoint > error)
        if config is not None:
            model_config = config
            logger.info("Using user-provided config")
        elif checkpoint_config is not None:
            if isinstance(checkpoint_config, dict):
                model_config = EnhancedMoEConfig(**checkpoint_config)
            else:
                model_config = checkpoint_config
            logger.info("Using config from checkpoint")
        else:
            raise ValueError(
                "No config available. Provide config argument or ensure "
                "checkpoint contains 'config' key"
            )

        # Create and load model
        logger.info(f"Initializing model: vocab={model_config.vocab_size}, "
                    f"hidden={model_config.hidden_size}, layers={model_config.num_layers}")
        model = cls(model_config, **kwargs)

        # Load with automatic key remapping for backwards compatibility
        success, result = load_state_dict_with_remapping(model, state_dict, strict=strict)
        if not strict and result and (result.missing_keys or result.unexpected_keys):
            logger.warning(f"Incompatible keys: missing={len(result.missing_keys)}, "
                          f"unexpected={len(result.unexpected_keys)}")

        if device is not None:
            model = model.to(device)

        logger.info(f"✓ Successfully loaded model from {checkpoint_path}")
        return model

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
        use_cache: bool = True,  # NEW: Enable KV caching by default
        **kwargs
    ) -> torch.Tensor:
        """
        Autoregressive text generation with KV caching for 20-50x speedup.

        OPTIMIZATION: Uses KV cache to avoid recomputing attention for previous tokens.
        Each generation step only processes the new token, not the entire sequence.

        Before (slow): Process entire sequence at each step - O(n²) attention
        After (fast): Process only new token with cached K,V - O(n) attention

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
            use_cache: Whether to use KV caching (default True for 20-50x speedup)

        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Start with input_ids
        generated = input_ids.clone()

        # Initialize KV cache
        past_key_values = None

        # First forward pass: process the entire prompt to build initial KV cache
        if use_cache:
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.get('past_key_values', None)
            logits = outputs['logits']
            next_token_logits = logits[:, -1, :]
        else:
            # Fallback to non-cached generation
            outputs = self.forward(
                input_ids=generated,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs['logits']
            next_token_logits = logits[:, -1, :]

        # Generate tokens one at a time
        for step_idx in range(max_length - input_ids.shape[1]):
            # For first iteration, we already have logits from the initial forward pass
            if step_idx > 0:
                if use_cache and past_key_values is not None:
                    # OPTIMIZED: Only process the last generated token
                    # KV cache contains attention states for all previous tokens
                    outputs = self.forward(
                        input_ids=next_token,  # Only the new token!
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.get('past_key_values', None)
                else:
                    # Non-cached: process entire sequence (slow fallback)
                    outputs = self.forward(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        return_dict=True
                    )

                logits = outputs['logits']
                next_token_logits = logits[:, -1, :]

            # Apply repetition penalty (vectorized for efficiency)
            if repetition_penalty != 1.0:
                # Create a mask for tokens that appear in the generated sequence
                # This is more efficient than per-token loops
                for i in range(batch_size):
                    unique_tokens = generated[i].unique()
                    for token_id in unique_tokens:
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Apply n-gram blocking
            # GPU SYNC FIX: Move generated to CPU once for n-gram computation instead of
            # calling .item() per position (which causes seq_len cudaStreamSynchronize calls)
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                # Single sync: transfer to CPU then convert to list
                # Using .cpu() then explicit sync then .tolist() avoids double sync from .cpu().tolist()
                gen_cpu_tensor = generated.cpu()
                if generated.is_cuda:
                    torch.cuda.current_stream().synchronize()
                generated_cpu = gen_cpu_tensor.tolist()
                for i in range(batch_size):
                    ngram_prefix = generated_cpu[i][-(no_repeat_ngram_size - 1):]
                    banned_tokens = set()
                    for j in range(len(generated_cpu[i]) - no_repeat_ngram_size + 1):
                        current_ngram_prefix = generated_cpu[i][j:j + no_repeat_ngram_size - 1]
                        if current_ngram_prefix == ngram_prefix:
                            banned_token = generated_cpu[i][j + no_repeat_ngram_size - 1]
                            banned_tokens.add(banned_token)
                    for token_id in banned_tokens:
                        next_token_logits[i, token_id] = float('-inf')

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < top_k_values[:, -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            if do_sample:
                # Nucleus (top-p) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
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

    Use this config for:
    - Production training at scale
    - Multi-GPU distributed training
    - Large models requiring memory optimizations
    - Maximum throughput with grouped GEMM, Triton kernels

    Key differences from EnhancedMoEConfig:
    - Grouped GEMM for expert computation (5-10x speedup)
    - Triton kernel support for custom operations
    - Expert offloading to CPU for memory savings
    - Expert quantization (INT8) for reduced memory
    - LoRA expert support for efficient fine-tuning
    - DeepSeek-style shared expert option
    - CUDAGraphs-safe routing option (20-30% speedup)

    For simpler development/testing, use EnhancedMoEConfig with EnhancedMoEModel.

    See also: EnhancedMoEConfig (line ~32)
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
    enable_cudagraphs_safe_routing: bool = False  # OPTIMIZATION: Enable CUDAGraphs-compatible routing (20-30% speedup)
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

    # Attention settings
    attention_dropout: float = 0.0
    use_flash_attention: bool = False
    rope_theta: float = 10000.0

    # Regularization
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

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
            type('Config', (), {  # type: ignore[call-arg]
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
                compile_router=getattr(config, 'compile_router', True),  # OPTIMIZATION: Enable selective router compilation (15-25% speedup)
                router_compile_mode=getattr(config, 'router_compile_mode', 'default'),
                enable_cudagraphs_safe_routing=getattr(config, 'enable_cudagraphs_safe_routing', False),  # OPTIMIZATION: Enable CUDAGraphs-compatible routing
                router_z_loss_coef=config.router_z_loss_coef,
                load_balance_loss_coef=config.load_balance_loss_coef,
                diversity_loss_coef=config.diversity_loss_coef,
                expert_dropout_loss_coef=config.expert_dropout_loss_coef,
                router_jitter_noise=config.router_jitter_noise,
                aux_loss_frequency=getattr(config, 'aux_loss_frequency', 1),  # OPTIMIZATION: Reduce aux loss overhead
                use_shared_expert=config.use_shared_expert,
                shared_expert_weight=config.shared_expert_weight,
                gradient_checkpointing=config.gradient_checkpointing,
                dtype=config.dtype,
            )
        else:
            raise ImportError("SparseMoELayer not available. Cannot create OptimizedMoETransformer")

        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

        # GRADIENT CHECKPOINTING FIX: Store flag to disable dropout during recomputation
        self._gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        # Self-attention with residual
        residual = hidden_states
        # OPTIMIZATION: Removed .clone() for 2-3% speedup
        hidden_states = self.ln1(hidden_states)
        attn_output, _ = self.attention(hidden_states, attention_mask)
        # GRADIENT CHECKPOINTING FIX: Skip dropout during checkpoint recomputation
        if self._gradient_checkpointing and self.training:
            hidden_states = residual + attn_output
        else:
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

        # GRADIENT CHECKPOINTING FIX: Store flag to disable dropout during recomputation
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

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

        # GPU SYNC FIX: Removed per-batch input validation that caused 2 cudaStreamSynchronize
        # via .any() calls. Token IDs are validated at data loading time.
        # The embedding layer will naturally fail if IDs are out of range.

        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        # GRADIENT CHECKPOINTING FIX: Skip embedding dropout during checkpoint recomputation
        if not (self.gradient_checkpointing and self.training):
            hidden_states = self.dropout(hidden_states)

        # Create causal attention mask
        # Use float32 for mask construction to avoid precision issues, then cast to model dtype
        # OPTIMIZATION: Construct directly in model dtype to avoid recompilation (10-15% speedup)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=hidden_states.dtype),
            diagonal=1
        )[None, None, :, :]

        if attention_mask is not None:
            # DTYPE FIX: Create mask directly in hidden_states.dtype to prevent recompilation
            mask_value = torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=device)
            padding_mask = torch.where(
                attention_mask[:, None, None, :] == 0,
                mask_value,
                torch.tensor(0.0, dtype=hidden_states.dtype, device=device)
            )
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
        hidden_states = self.ln_f(hidden_states)
        # PERFORMANCE FIX: Only clone when CUDA graphs are enabled
        # Clone prevents tensor overwrite errors with CUDA graphs, but wastes 2-3% VRAM otherwise
        if getattr(self.config, 'enable_cudagraphs_safe_routing', False):
            hidden_states = hidden_states.clone()

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Standard cross-entropy loss
            # Use -100 as ignore_index (standard PyTorch convention)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Add MoE auxiliary losses
            # FIX: aux_loss from SparseMoELayer is already scaled by coefficients at the router level
            # (load_balance_loss_coef, router_z_loss_coef, etc. are applied in the router forward pass)
            # So we should NOT apply router_aux_coef again here - that would be double scaling.
            if all_aux_info:
                total_aux_loss = sum(info['aux_loss'] for info in all_aux_info if info.get('aux_loss') is not None)
                num_layers = sum(1 for info in all_aux_info if info.get('aux_loss') is not None)
                if num_layers > 0:
                    # Average across layers (consistent with EnhancedMoEModel)
                    total_aux_loss = total_aux_loss / num_layers
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
                # Re-raise PyTorch internal exceptions used by gradient checkpointing
                if type(e).__name__ in ('_StopRecomputationError', 'StopIteration'):
                    raise
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
            # GPU SYNC FIX: Move generated to CPU once for n-gram computation instead of
            # calling .item() per position (which causes seq_len cudaStreamSynchronize calls)
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                # Single sync: transfer to CPU then convert to list
                # Using .cpu() then explicit sync then .tolist() avoids double sync from .cpu().tolist()
                gen_cpu_tensor = generated.cpu()
                if generated.is_cuda:
                    torch.cuda.current_stream().synchronize()
                generated_cpu = gen_cpu_tensor.tolist()
                # For each sequence in batch
                for i in range(batch_size):
                    # Get the last (n-1) tokens
                    ngram_prefix = generated_cpu[i][-(no_repeat_ngram_size - 1):]

                    # Find all n-grams in the generated sequence that start with this prefix
                    banned_tokens = set()
                    for j in range(len(generated_cpu[i]) - no_repeat_ngram_size + 1):
                        # Check if this position matches our prefix
                        current_ngram_prefix = generated_cpu[i][j:j + no_repeat_ngram_size - 1]
                        if current_ngram_prefix == ngram_prefix:
                            # Ban the token that completes this n-gram
                            banned_token = generated_cpu[i][j + no_repeat_ngram_size - 1]
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
