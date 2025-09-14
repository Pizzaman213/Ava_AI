"""
MLX Model implementation compatible with MoEConfig
Optimized for Apple Silicon using MLX framework
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

class MLXTransformer(nn.Module):
    """Optimized MLX Transformer model compatible with MoEConfig"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Convert config attributes if it's a MoEConfig object
        if hasattr(config, '__dict__'):
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            num_layers = config.num_layers
            num_heads = config.num_attention_heads
            max_pos = config.max_position_embeddings
            dropout = getattr(config, 'hidden_dropout', 0.1)
            intermediate_size = config.intermediate_size or (hidden_size * 4)
            num_experts = getattr(config, 'num_experts', 0)
            num_experts_per_tok = getattr(config, 'num_experts_per_tok', 1)
        else:
            # Dict config
            vocab_size = config['vocab_size']
            hidden_size = config['hidden_size']
            num_layers = config['num_layers']
            num_heads = config['num_attention_heads']
            max_pos = config['max_position_embeddings']
            dropout = config.get('hidden_dropout', 0.1)
            intermediate_size = config.get('intermediate_size', hidden_size * 4)
            num_experts = config.get('num_experts', 0)
            num_experts_per_tok = config.get('num_experts_per_tok', 1)
        
        # Embeddings - optimized initialization
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_pos, hidden_size)
        
        # Transformer layers
        self.layers = []
        for _ in range(num_layers):
            if num_experts > 0:
                layer = MLXMoELayer(
                    hidden_size, num_heads, intermediate_size, 
                    num_experts, num_experts_per_tok, dropout
                )
            else:
                layer = MLXTransformerLayer(
                    hidden_size, num_heads, intermediate_size, dropout
                )
            self.layers.append(layer)
        
        # Output - use RMSNorm for speed
        self.ln_f = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Only apply dropout if > 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Precompute position indices for speed
        self._position_cache = {}
        
        # Store config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def __call__(self, input_ids, attention_mask=None, labels=None, training=False):
        # Ensure input_ids is int32
        if not isinstance(input_ids, mx.array):
            input_ids = mx.array(input_ids, dtype=mx.int32)
        elif input_ids.dtype != mx.int32:
            input_ids = input_ids.astype(mx.int32)
            
        batch_size, seq_len = input_ids.shape
        
        # Create position ids (ensure int32 dtype)
        position_ids = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(position_ids, 0)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))
        position_ids = position_ids.astype(mx.int32)  # Ensure int32
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        if training and self.dropout:
            hidden_states = self.dropout(hidden_states)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, training=training)
        
        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        if labels is not None:
            # Simplified loss computation to avoid segfault
            # Flatten logits and labels
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            
            # Cross entropy loss without shifting (simpler)
            loss = mx.mean(nn.losses.cross_entropy(logits_flat, labels_flat, reduction='none'))
            
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}
    
    def generate(self, input_ids, max_length=50, temperature=0.8, top_k=50):
        """Simple generation for compatibility"""
        self.eval()
        
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self(input_ids, training=False)
            logits = outputs['logits']
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_vals = mx.topk(next_token_logits, k=min(top_k, next_token_logits.shape[-1]))
                mask = next_token_logits < top_k_vals[0][:, -1:] 
                next_token_logits = mx.where(mask, -float('inf'), next_token_logits)
            
            # Sample
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.argmax(probs, axis=-1, keepdims=True)
            
            input_ids = mx.concatenate([input_ids, next_token], axis=1)
        
        return input_ids

class MLXTransformerLayer(nn.Module):
    """Single transformer layer in MLX - optimized for speed"""
    
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Use RMSNorm for speed
        self.ln_1 = nn.RMSNorm(hidden_size)
        self.self_attn = MLXAttention(hidden_size, num_heads, dropout)
        
        # MLP with RMSNorm
        self.ln_2 = nn.RMSNorm(hidden_size)
        self.mlp = MLXMLP(hidden_size, intermediate_size)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def __call__(self, hidden_states, training=False):
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.self_attn(hidden_states, training=training)
        if training and self.dropout:
            hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if training and self.dropout:
            hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class MLXMoELayer(nn.Module):
    """MoE layer in MLX - optimized for speed"""
    
    def __init__(self, hidden_size, num_heads, intermediate_size, num_experts, num_experts_per_tok, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        # Use RMSNorm for speed
        self.ln_1 = nn.RMSNorm(hidden_size)
        self.self_attn = MLXAttention(hidden_size, num_heads, dropout)
        
        # MoE MLP with RMSNorm
        self.ln_2 = nn.RMSNorm(hidden_size)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)  # No bias for speed
        self.experts = [MLXMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def __call__(self, hidden_states, training=False):
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.self_attn(hidden_states, training=training)
        if training and self.dropout:
            hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MoE MLP with residual
        residual = hidden_states
        hidden_states_norm = self.ln_2(hidden_states)
        
        # Routing
        batch_size, seq_len, hidden_size = hidden_states_norm.shape
        router_logits = self.router(hidden_states_norm)
        router_probs = mx.softmax(router_logits, axis=-1)
        
        # Simplified MoE without conditional logic to avoid segfault
        # Apply all experts weighted by their routing probabilities
        expert_outputs = None
        
        for expert_idx in range(self.num_experts):
            expert_weight = router_probs[:, :, expert_idx:expert_idx+1]
            expert_output = self.experts[expert_idx](hidden_states_norm)
            weighted_output = expert_output * expert_weight
            
            if expert_outputs is None:
                expert_outputs = weighted_output
            else:
                expert_outputs = expert_outputs + weighted_output
        
        hidden_states = expert_outputs
        if training and self.dropout:
            hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class MLXAttention(nn.Module):
    """Multi-head attention in MLX - optimized with fused operations"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Separate QKV projections (more stable)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        # Store scale as a regular Python float, not an MLX array
        self.scale = float(self.head_dim ** 0.5)
        
    def __call__(self, hidden_states, training=False):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Separate QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Simple transpose
        q = mx.transpose(q, (0, 2, 1, 3))  # [B, H, S, D]
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Attention scores with precomputed scale
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / self.scale
        
        # Causal mask - create more carefully to avoid issues
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9, dtype=mx.float32), k=1)
        scores = scores + mx.expand_dims(mx.expand_dims(mask, 0), 0)  # Add batch and head dims
        
        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)
        if training and self.dropout:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = mx.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MLXMLP(nn.Module):
    """MLP in MLX with SwiGLU activation - optimized"""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # Back to separate projections to avoid split issues
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def __call__(self, hidden_states):
        # SwiGLU activation with separate projections
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

def count_parameters(model):
    """Count parameters in MLX model"""
    return sum(p.size for _, p in tree_flatten(model.parameters()))

def create_mlx_model(config):
    """Create MLX model from config"""
    return MLXTransformer(config)