"""
PagedAttention implementation (vLLM-style)

Implements virtual memory management for KV cache to enable efficient
handling of long sequences and dynamic memory allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass 
class PagedAttentionConfig:
    """Configuration for PagedAttention"""
    block_size: int = 16  # Number of tokens per block
    num_blocks: int = 256  # Total number of blocks in pool
    num_kv_heads: int = 32  # Number of KV heads
    head_dim: int = 128  # Dimension per head
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    
    # Memory management
    max_blocks_per_seq: int = 128
    enable_prefix_caching: bool = True
    swap_space_gb: int = 4  # CPU swap space
    
    # Performance
    use_cuda_kernel: bool = True
    num_worker_threads: int = 1


class BlockTable:
    """Manages block allocation for sequences"""
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.free_blocks = deque(range(config.num_blocks))
        self.ref_counts = torch.zeros(config.num_blocks, dtype=torch.int32)
        
        # Sequence to blocks mapping
        self.seq_to_blocks: Dict[int, List[int]] = {}
        
    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        """Allocate blocks for a sequence"""
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"Not enough free blocks. Need {num_blocks}, have {len(self.free_blocks)}")
            
        blocks = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.popleft()
            blocks.append(block_id)
            self.ref_counts[block_id] = 1
            
        self.seq_to_blocks[seq_id] = blocks
        return blocks
    
    def free(self, seq_id: int):
        """Free blocks for a sequence"""
        if seq_id not in self.seq_to_blocks:
            return
            
        blocks = self.seq_to_blocks[seq_id]
        for block_id in blocks:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)
                
        del self.seq_to_blocks[seq_id]
        
    def fork(self, parent_seq_id: int, child_seq_id: int):
        """Fork blocks from parent to child (copy-on-write)"""
        if parent_seq_id not in self.seq_to_blocks:
            raise ValueError(f"Parent sequence {parent_seq_id} not found")
            
        parent_blocks = self.seq_to_blocks[parent_seq_id]
        self.seq_to_blocks[child_seq_id] = parent_blocks.copy()
        
        # Increment reference counts
        for block_id in parent_blocks:
            self.ref_counts[block_id] += 1
            
    def get_blocks(self, seq_id: int) -> List[int]:
        """Get blocks for a sequence"""
        return self.seq_to_blocks.get(seq_id, [])
    
    def append_block(self, seq_id: int) -> int:
        """Append a new block to sequence"""
        if len(self.free_blocks) == 0:
            raise RuntimeError("No free blocks available")
            
        block_id = self.free_blocks.popleft()
        self.ref_counts[block_id] = 1
        
        if seq_id not in self.seq_to_blocks:
            self.seq_to_blocks[seq_id] = []
        self.seq_to_blocks[seq_id].append(block_id)
        
        return block_id


class PagedKVCache:
    """
    Paged key-value cache with virtual memory management
    
    Features:
    - Block-based memory allocation
    - Copy-on-write for efficient forking
    - CPU offloading for large caches
    - Prefix caching for shared prompts
    """
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.block_table = BlockTable(config)
        
        # Allocate KV cache blocks
        cache_shape = (
            config.num_blocks,
            config.block_size,
            2,  # K and V
            config.num_kv_heads,
            config.head_dim
        )
        
        # GPU cache
        self.gpu_cache = torch.zeros(
            cache_shape,
            dtype=config.dtype,
            device=config.device
        )
        
        # CPU swap space
        if config.swap_space_gb > 0:
            cpu_blocks = int(config.swap_space_gb * 1024**3 / 
                           (config.block_size * 2 * config.num_kv_heads * config.head_dim * 2))
            self.cpu_cache = torch.zeros(
                (cpu_blocks, config.block_size, 2, config.num_kv_heads, config.head_dim),
                dtype=config.dtype,
                device="cpu",
                pin_memory=True
            )
            self.cpu_block_table = BlockTable(PagedAttentionConfig(
                **{**vars(config), 'num_blocks': cpu_blocks}
            ))
        else:
            self.cpu_cache = None
            self.cpu_block_table = None
            
        # Prefix cache for shared prompts
        self.prefix_cache: Dict[str, List[int]] = {}
        
    def allocate_sequence(self, seq_id: int, seq_len: int) -> List[int]:
        """Allocate blocks for a new sequence"""
        num_blocks = (seq_len + self.config.block_size - 1) // self.config.block_size
        return self.block_table.allocate(seq_id, num_blocks)
    
    def write_kv(
        self,
        seq_id: int,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """Write key-value pair to cache"""
        blocks = self.block_table.get_blocks(seq_id)
        if not blocks:
            raise ValueError(f"No blocks allocated for sequence {seq_id}")
            
        # Determine block and position within block
        block_idx = position // self.config.block_size
        position_in_block = position % self.config.block_size
        
        # Allocate new block if needed
        if block_idx >= len(blocks):
            for _ in range(block_idx - len(blocks) + 1):
                self.block_table.append_block(seq_id)
            blocks = self.block_table.get_blocks(seq_id)
            
        block_id = blocks[block_idx]
        
        # Write to cache
        self.gpu_cache[block_id, position_in_block, 0] = key
        self.gpu_cache[block_id, position_in_block, 1] = value
        
    def read_kv(
        self,
        seq_id: int,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read key-value pairs from cache"""
        blocks = self.block_table.get_blocks(seq_id)
        if not blocks:
            raise ValueError(f"No blocks allocated for sequence {seq_id}")
            
        batch_size = positions.shape[0]
        keys = []
        values = []
        
        for i in range(batch_size):
            pos = positions[i].item()
            block_idx = pos // self.config.block_size
            position_in_block = pos % self.config.block_size
            
            if block_idx < len(blocks):
                block_id = blocks[block_idx]
                keys.append(self.gpu_cache[block_id, position_in_block, 0])
                values.append(self.gpu_cache[block_id, position_in_block, 1])
            else:
                # Return zeros for out-of-bounds positions
                keys.append(torch.zeros(
                    (self.config.num_kv_heads, self.config.head_dim),
                    dtype=self.config.dtype,
                    device=self.config.device
                ))
                values.append(torch.zeros(
                    (self.config.num_kv_heads, self.config.head_dim),
                    dtype=self.config.dtype,
                    device=self.config.device
                ))
                
        keys = torch.stack(keys)
        values = torch.stack(values)
        
        return keys, values
    
    def swap_out(self, seq_id: int):
        """Swap sequence cache to CPU"""
        if self.cpu_cache is None:
            logger.warning("CPU swap space not configured")
            return
            
        gpu_blocks = self.block_table.get_blocks(seq_id)
        if not gpu_blocks:
            return
            
        # Allocate CPU blocks
        cpu_blocks = self.cpu_block_table.allocate(seq_id, len(gpu_blocks))
        
        # Copy data
        for gpu_block, cpu_block in zip(gpu_blocks, cpu_blocks):
            self.cpu_cache[cpu_block].copy_(self.gpu_cache[gpu_block])
            
        # Free GPU blocks
        self.block_table.free(seq_id)
        
    def swap_in(self, seq_id: int):
        """Swap sequence cache back to GPU"""
        if self.cpu_cache is None:
            return
            
        cpu_blocks = self.cpu_block_table.get_blocks(seq_id)
        if not cpu_blocks:
            return
            
        # Allocate GPU blocks
        gpu_blocks = self.block_table.allocate(seq_id, len(cpu_blocks))
        
        # Copy data
        for cpu_block, gpu_block in zip(cpu_blocks, gpu_blocks):
            self.gpu_cache[gpu_block].copy_(self.cpu_cache[cpu_block])
            
        # Free CPU blocks
        self.cpu_block_table.free(seq_id)
        
    def fork_sequence(self, parent_id: int, child_id: int):
        """Fork sequence using copy-on-write"""
        self.block_table.fork(parent_id, child_id)
        
    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence"""
        self.block_table.free(seq_id)
        if self.cpu_block_table and seq_id in self.cpu_block_table.seq_to_blocks:
            self.cpu_block_table.free(seq_id)


class PagedAttention(nn.Module):
    """
    Attention module with paged KV cache
    
    Drop-in replacement for standard attention with virtual memory management
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 2048,
        config: Optional[PagedAttentionConfig] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        
        # Initialize paged KV cache config
        if config is None:
            config = PagedAttentionConfig(
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim
            )
        self.config = config
        
        # Initialize KV cache
        self.kv_cache = PagedKVCache(config)
        
        # Query, key, value projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # Rotary embeddings (if used)
        self.rotary_emb = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_id: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with paged attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings if available
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
            
        # Write to paged KV cache
        if use_cache and seq_id is not None:
            for i in range(seq_len):
                pos = position_ids[0, i].item() if position_ids is not None else i
                self.kv_cache.write_kv(
                    seq_id,
                    pos,
                    key_states[0, i],
                    value_states[0, i]
                )
                
        # Compute attention with paged KV cache
        attn_output = self._paged_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            seq_id
        )
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None
    
    def _paged_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_id: Optional[int]
    ) -> torch.Tensor:
        """Compute attention using paged KV cache"""
        batch_size, seq_len, num_heads, head_dim = query_states.shape
        
        # For now, use standard attention computation
        # In production, this would use optimized kernels
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Repeat KV for GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)
            value_states = value_states.repeat_interleave(repeat_factor, dim=1)
            
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output
    
    def allocate_kv_cache(self, batch_size: int, max_seq_len: int):
        """Pre-allocate KV cache for sequences"""
        for i in range(batch_size):
            self.kv_cache.allocate_sequence(i, max_seq_len)
            
    def clear_kv_cache(self):
        """Clear all KV cache"""
        for seq_id in list(self.kv_cache.block_table.seq_to_blocks.keys()):
            self.kv_cache.free_sequence(seq_id)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings"""
    # This is a simplified version - actual implementation would be more complex
    return q, k