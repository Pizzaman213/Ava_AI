"""
Activation Compression and Gradient Checkpointing
Memory-efficient training with activation recomputation
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple, Any, Optional, Callable, Dict
import logging
import functools
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing"""
    # Basic settings
    enable_checkpointing: bool = True
    checkpoint_method: str = "uniform"  # uniform, selective, adaptive
    
    # Selective checkpointing
    checkpoint_ratio: float = 0.5  # Fraction of layers to checkpoint
    min_checkpoint_interval: int = 1
    max_checkpoint_interval: int = 4
    
    # Memory target
    target_memory_gb: float = 16.0
    
    # Performance
    use_reentrant: bool = True
    preserve_rng_state: bool = True
    
    # Adaptive settings
    profile_memory: bool = True
    adjust_dynamically: bool = False

class SelectiveCheckpointer:
    """Selective gradient checkpointing based on memory usage"""
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.layer_memory_usage = {}
        self.checkpoint_decisions = {}
        
    def profile_layer(self, layer_name: str, layer: nn.Module, input_shape: Tuple[int, ...]):
        """Profile memory usage of a layer"""
        if not self.config.profile_memory:
            return
        
        # Create dummy input
        device = next(layer.parameters()).device
        dummy_input = torch.randn(input_shape, device=device, requires_grad=True)
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.enable_grad():
            output = layer(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
        
        # Measure memory after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()
            activation_memory = mem_after - mem_before
        else:
            # Estimate for CPU
            activation_memory = output.numel() * output.element_size()
        
        self.layer_memory_usage[layer_name] = activation_memory
        
        # Cleanup
        del dummy_input, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def decide_checkpoints(self, total_layers: int) -> List[bool]:
        """Decide which layers to checkpoint"""
        if self.config.checkpoint_method == "uniform":
            # Uniform checkpointing
            interval = max(1, int(1 / self.config.checkpoint_ratio))
            return [i % interval == 0 for i in range(total_layers)]
        
        elif self.config.checkpoint_method == "selective":
            # Checkpoint layers with highest memory usage
            if not self.layer_memory_usage:
                # Fallback to uniform if no profiling data
                return self.decide_checkpoints_uniform(total_layers)
            
            # Sort layers by memory usage
            sorted_layers = sorted(
                self.layer_memory_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top memory consumers
            num_checkpoint = int(total_layers * self.config.checkpoint_ratio)
            checkpoint_names = set(name for name, _ in sorted_layers[:num_checkpoint])
            
            return [f"layer_{i}" in checkpoint_names for i in range(total_layers)]
        
        elif self.config.checkpoint_method == "adaptive":
            # Adaptive checkpointing based on memory target
            return self._adaptive_checkpointing(total_layers)
        
        else:
            raise ValueError(f"Unknown checkpoint method: {self.config.checkpoint_method}")
    
    def _adaptive_checkpointing(self, total_layers: int) -> List[bool]:
        """Adaptive checkpointing to meet memory target"""
        if not self.layer_memory_usage:
            return self.decide_checkpoints_uniform(total_layers)
        
        # Calculate total activation memory
        total_activation_memory = sum(self.layer_memory_usage.values())
        target_memory = self.config.target_memory_gb * 1e9  # Convert to bytes
        
        if total_activation_memory <= target_memory:
            # No checkpointing needed
            return [False] * total_layers
        
        # Binary search for optimal checkpoint ratio
        low, high = 0.0, 1.0
        best_decisions = None
        
        while high - low > 0.01:
            mid = (low + high) / 2
            self.config.checkpoint_ratio = mid
            decisions = self.decide_checkpoints_uniform(total_layers)
            
            # Estimate memory with these decisions
            saved_memory = sum(
                mem for i, (name, mem) in enumerate(self.layer_memory_usage.items())
                if decisions[i]
            )
            
            remaining_memory = total_activation_memory - saved_memory
            
            if remaining_memory <= target_memory:
                best_decisions = decisions
                high = mid
            else:
                low = mid
        
        return best_decisions or [True] * total_layers
    
    def decide_checkpoints_uniform(self, total_layers: int) -> List[bool]:
        """Uniform checkpoint spacing"""
        interval = max(1, int(1 / self.config.checkpoint_ratio))
        return [i % interval == 0 for i in range(total_layers)]

class CheckpointedModule(nn.Module):
    """Wrapper for checkpointed execution of a module"""
    def __init__(
        self,
        module: nn.Module,
        use_checkpoint: bool = True,
        use_reentrant: bool = True,
        preserve_rng_state: bool = True,
    ):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state
    
    def forward(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing
            if self.use_reentrant:
                return checkpoint(
                    self.module,
                    *args,
                    preserve_rng_state=self.preserve_rng_state,
                    **kwargs
                )
            else:
                # Non-reentrant checkpointing (PyTorch 2.0+)
                return checkpoint(
                    self.module,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=self.preserve_rng_state,
                    **kwargs
                )
        else:
            # Normal forward pass
            return self.module(*args, **kwargs)

class ActivationCompressor:
    """Compress activations during forward pass"""
    def __init__(
        self,
        compression_ratio: float = 0.5,
        compression_method: str = "quantize",  # quantize, svd, sparse
        bits: int = 8,
    ):
        self.compression_ratio = compression_ratio
        self.compression_method = compression_method
        self.bits = bits
        self.compressed_activations = {}
    
    def compress(self, name: str, activation: torch.Tensor) -> Any:
        """Compress activation tensor"""
        if self.compression_method == "quantize":
            return self._quantize(activation)
        elif self.compression_method == "svd":
            return self._svd_compress(activation)
        elif self.compression_method == "sparse":
            return self._sparse_compress(activation)
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
    
    def decompress(self, name: str, compressed: Any) -> torch.Tensor:
        """Decompress activation tensor"""
        if self.compression_method == "quantize":
            return self._dequantize(compressed)
        elif self.compression_method == "svd":
            return self._svd_decompress(compressed)
        elif self.compression_method == "sparse":
            return self._sparse_decompress(compressed)
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
    
    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Quantize tensor to reduced precision"""
        # Calculate scale and zero point
        min_val = tensor.min()
        max_val = tensor.max()
        
        scale = (max_val - min_val) / (2 ** self.bits - 1)
        zero_point = min_val
        
        # Quantize
        quantized = ((tensor - zero_point) / scale).round().to(torch.int8)
        
        return quantized, scale.item(), zero_point.item()
    
    def _dequantize(self, compressed: Tuple[torch.Tensor, float, float]) -> torch.Tensor:
        """Dequantize tensor"""
        quantized, scale, zero_point = compressed
        return quantized.float() * scale + zero_point
    
    def _svd_compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress using SVD decomposition"""
        # Reshape to 2D
        original_shape = tensor.shape
        matrix = tensor.view(tensor.size(0), -1)
        
        # SVD
        U, S, V = torch.svd(matrix, some=True)
        
        # Keep top k components
        k = int(S.size(0) * self.compression_ratio)
        U_compressed = U[:, :k]
        S_compressed = S[:k]
        V_compressed = V[:, :k]
        
        return (U_compressed, S_compressed, V_compressed, original_shape)
    
    def _svd_decompress(self, compressed: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Decompress SVD representation"""
        U, S, V, original_shape = compressed
        
        # Reconstruct matrix
        matrix = torch.mm(torch.mm(U, torch.diag(S)), V.t())
        
        # Reshape to original
        return matrix.view(original_shape)
    
    def _sparse_compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress by keeping only top-k values"""
        # Flatten tensor
        flat = tensor.view(-1)
        
        # Keep top-k values
        k = int(flat.numel() * self.compression_ratio)
        values, indices = torch.topk(flat.abs(), k)
        
        # Get actual values (with sign)
        values = flat[indices]
        
        return values, indices, tensor.shape
    
    def _sparse_decompress(self, compressed: Tuple[torch.Tensor, torch.Tensor, torch.Size]) -> torch.Tensor:
        """Decompress sparse representation"""
        values, indices, shape = compressed
        
        # Reconstruct tensor
        flat = torch.zeros(shape.numel(), dtype=values.dtype, device=values.device)
        flat[indices] = values
        
        return flat.view(shape)

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention implementation"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_chunked_attention: bool = False,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_dropout = attention_dropout
        self.use_flash_attention = use_flash_attention
        self.use_chunked_attention = use_chunked_attention
        self.chunk_size = chunk_size
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with memory-efficient attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute QKV
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use Flash Attention (PyTorch 2.0+)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scale,
            )
        elif self.use_chunked_attention:
            # Chunked attention for long sequences
            attn_output = self._chunked_attention(q, k, v, attention_mask)
        else:
            # Standard attention
            attn_output = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard scaled dot-product attention"""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        probs = torch.softmax(scores, dim=-1)
        probs = torch.dropout(probs, self.attention_dropout, self.training)
        
        return torch.matmul(probs, v)
    
    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Chunked attention for memory efficiency"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        chunk_size = min(self.chunk_size, seq_len)
        
        # Process attention in chunks
        output_chunks = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]
            
            # Compute attention for this chunk against all keys
            chunk_scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, i:end_i, :]
                chunk_scores = chunk_scores + chunk_mask
            
            chunk_probs = torch.softmax(chunk_scores, dim=-1)
            chunk_probs = torch.dropout(chunk_probs, self.attention_dropout, self.training)
            
            chunk_output = torch.matmul(chunk_probs, v)
            output_chunks.append(chunk_output)
        
        return torch.cat(output_chunks, dim=2)

def apply_gradient_checkpointing(
    model: nn.Module,
    config: CheckpointConfig,
) -> nn.Module:
    """
    Apply gradient checkpointing to a model
    
    Args:
        model: Model to modify
        config: Checkpointing configuration
        
    Returns:
        Modified model with checkpointing
    """
    # Create checkpointer
    checkpointer = SelectiveCheckpointer(config)
    
    # Find transformer layers
    transformer_layers = []
    for name, module in model.named_modules():
        if "layer" in name and len(list(module.children())) > 0:
            transformer_layers.append((name, module))
    
    # Profile layers if needed
    if config.profile_memory:
        logger.info("Profiling layer memory usage...")
        for name, layer in transformer_layers:
            # Assume typical input shape
            input_shape = (1, 512, model.config.hidden_size)
            checkpointer.profile_layer(name, layer, input_shape)
    
    # Decide which layers to checkpoint
    checkpoint_decisions = checkpointer.decide_checkpoints(len(transformer_layers))
    
    # Apply checkpointing
    for i, (name, layer) in enumerate(transformer_layers):
        if checkpoint_decisions[i]:
            # Wrap layer with checkpointing
            parent_name, layer_name = name.rsplit('.', 1)
            parent = model
            
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            wrapped = CheckpointedModule(
                layer,
                use_checkpoint=True,
                use_reentrant=config.use_reentrant,
                preserve_rng_state=config.preserve_rng_state,
            )
            
            setattr(parent, layer_name, wrapped)
            logger.debug(f"Checkpointed layer: {name}")
    
    logger.info(f"Applied gradient checkpointing to {sum(checkpoint_decisions)} / {len(transformer_layers)} layers")
    
    return model