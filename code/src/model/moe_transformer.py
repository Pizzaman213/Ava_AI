"""
MoE++ Transformer with Mixture of Depths (MoD) and Multi-Query Attention (MQA/GQA)
State-of-the-art implementation with hierarchical expert routing and advanced optimizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math
from src.model.experts import HierarchicalExpertLayer, ExpertRouter, MemoryEfficientMoELayer, AdaptiveCapacityRouter
from src.model.attention import MultiQueryAttention, FlashAttentionWrapper, create_attention_layer, xPosRotaryEmbedding
try:
    from src.model.universal_attention import UniversalMultiQueryAttention, UniversalRotaryEmbedding
    UNIVERSAL_ATTENTION_AVAILABLE = True
except ImportError:
    UNIVERSAL_ATTENTION_AVAILABLE = False
from src.model.mod import MixtureOfDepths, DepthRouter
import logging
import numpy as np
import os
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class MoEConfig:
    """Configuration for MoE++ Transformer"""
    vocab_size: int = 50257
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For GQA
    intermediate_size: Optional[int] = None  # Will be set based on hidden_size if not specified
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_capacity_factor: float = 1.25
    hidden_act: str = "swiglu"
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.01  # Reduced for better stability in deep networks
    use_cache: bool = True
    tie_word_embeddings: bool = False
    
    # MoD configuration
    use_mod: bool = True
    mod_mode: str = "learned"  # learned, fixed, adaptive
    mod_capacity_factor: float = 0.8
    mod_skip_fraction: float = 0.2
    
    # MQA/GQA configuration
    attention_type: str = "gqa"  # mha, mqa, gqa
    use_flash_attn: bool = True
    attention_dropout: float = 0.1  # Default attention dropout for regularization
    
    # Expert configuration
    expert_type: str = "hierarchical"  # standard, hierarchical
    expert_routing_type: str = "switch"  # switch, gshard, base
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    
    # Advanced features
    use_speculative_decoding: bool = True
    speculative_draft_layers: int = 4
    use_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    
    # Activation options
    hidden_dropout: float = 0.1  # Default hidden dropout for regularization
    attention_bias: bool = False
    mlp_bias: bool = False
    label_smoothing: float = 0.1  # Label smoothing for loss regularization
    
    # Optimization features
    use_fused_kernels: bool = False
    
    # Output options
    output_attentions: bool = False
    output_hidden_states: bool = False
    
    # New optimization features
    use_parallel_experts: bool = True
    use_memory_efficient_moe: bool = False
    expert_dropout: float = 0.1  # Default expert dropout for regularization
    use_adaptive_capacity: bool = False
    use_torch_compile: bool = True
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_backend: str = "inductor"
    checkpoint_experts: bool = False
    activation_pool_size: int = 4
    
    # Advanced Architecture Features from FUTURE_FEATURES.md
    # MoD++ Configuration
    use_mod_plus_plus: bool = False
    mod_confidence_threshold: float = 0.95
    mod_min_layers: int = 3
    mod_adaptive_depth: bool = True
    
    # Hierarchical MoE Configuration
    use_hierarchical_moe: bool = False
    num_expert_groups: int = 4
    experts_per_group: int = 4
    
    # Continuous Experts Configuration
    use_continuous_experts: bool = False
    continuous_expert_temperature: float = 1.0
    
    # NAS Expert Search Configuration
    use_nas_experts: bool = False
    nas_population_size: int = 20
    nas_num_generations: int = 50
    
    # Mixture of Tokenizers Configuration
    use_mixture_tokenizers: bool = False
    tokenizer_types: Optional[List[str]] = None  # ['byte', 'word', 'char']
    
    # Missing attribute that's being accessed
    use_return_dict: bool = True
    
    def __post_init__(self):
        """Set dynamic defaults based on other parameters"""
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # Only convert if needed
        if hidden_states.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Advanced Architecture Components from FUTURE_FEATURES.md

class MoDPlusPlus(nn.Module):
    """Mixture of Depths v2 with learnable depth routing and early exit"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.depth_predictor = nn.Linear(config.hidden_size, config.num_layers)
        self.confidence_scorer = nn.Linear(config.hidden_size, 1)
        
        self.layer_importance = nn.Parameter(
            torch.ones(config.num_layers) / config.num_layers
        )
        
        self.depth_embedding = nn.Embedding(config.num_layers, config.hidden_size)
        
    def forward(self, x: torch.Tensor, layers: nn.ModuleList) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len = x.shape[:2]
        
        depth_logits = self.depth_predictor(x.mean(dim=1))
        depth_probs = F.softmax(depth_logits, dim=-1)
        
        if self.training:
            selected_depths = torch.multinomial(depth_probs, 1).squeeze(-1)
        else:
            selected_depths = depth_probs.argmax(dim=-1)
        
        output = x
        exit_points = []
        confidence = None
        
        for i, layer in enumerate(layers):
            if i >= self.config.mod_min_layers:
                confidence = torch.sigmoid(self.confidence_scorer(output.mean(dim=1)))
                
                should_exit = (confidence > self.config.mod_confidence_threshold) | (i >= selected_depths.unsqueeze(1))
                
                if should_exit.all() and not self.training:
                    break
            
            # Fix: Handle layer output properly
            if hasattr(layer, 'forward'):
                layer_output = layer(output)
                # Handle tuple outputs from layers
                if isinstance(layer_output, tuple):
                    output = layer_output[0]
                else:
                    output = layer_output
            else:
                output = layer(output)
            
            if self.config.mod_adaptive_depth:
                depth_emb = self.depth_embedding(torch.tensor(i, device=x.device))
                output = output + depth_emb.unsqueeze(0).unsqueeze(0) * self.layer_importance[i]
            
            exit_points.append(i)
        
        stats = {
            'average_depth': float(selected_depths.float().mean()),
            'depth_distribution': depth_probs.detach(),
            'exit_points': exit_points,
            'confidence_scores': confidence.detach() if confidence is not None else None
        }
        
        return output, stats


class HierarchicalMoEPlus(nn.Module):
    """Hierarchical Mixture of Experts with two-stage routing"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_expert_groups = config.num_expert_groups
        self.experts_per_group = config.experts_per_group
        
        self.coarse_router = nn.Linear(config.hidden_size, config.num_expert_groups)
        
        self.fine_routers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.experts_per_group) 
            for _ in range(config.num_expert_groups)
        ])
        
        expert_dim = config.intermediate_size if config.intermediate_size is not None else config.hidden_size * 4
        self.expert_groups = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_size, expert_dim),
                    nn.ReLU() if config.hidden_act == "relu" else nn.GELU(),
                    nn.Linear(expert_dim, config.hidden_size)
                ) for _ in range(config.experts_per_group)
            ]) for _ in range(config.num_expert_groups)
        ])
        
        self.group_embeddings = nn.Embedding(config.num_expert_groups, config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, hidden_size = x.shape
        
        # Fix: Reshape for routing
        x_flat = x.reshape(-1, hidden_size)
        
        coarse_logits = self.coarse_router(x_flat)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        
        if self.training:
            group_indices = torch.multinomial(coarse_probs, 1).squeeze(-1)
        else:
            group_indices = coarse_probs.argmax(dim=-1)
        
        output_flat = torch.zeros_like(x_flat)
        total_aux_loss = 0.0
        
        for group_idx in range(self.num_expert_groups):
            group_mask = (group_indices == group_idx)
            
            if not group_mask.any():
                continue
                
            group_input = x_flat[group_mask]
            
            fine_logits = self.fine_routers[group_idx](group_input)
            fine_probs = F.softmax(fine_logits, dim=-1)
            
            if self.training:
                expert_indices = torch.multinomial(fine_probs, 1).squeeze(-1)
            else:
                expert_indices = fine_probs.argmax(dim=-1)
            
            group_output = torch.zeros_like(group_input)
            
            for expert_idx in range(self.experts_per_group):
                expert_mask = (expert_indices == expert_idx)
                
                if not expert_mask.any():
                    continue
                    
                expert_input = group_input[expert_mask]
                expert_output = self.expert_groups[group_idx][expert_idx](expert_input)
                group_output[expert_mask] = expert_output
            
            group_emb = self.group_embeddings(torch.tensor(group_idx, device=x.device))
            group_output = group_output + group_emb
            
            output_flat[group_mask] = group_output
        
        # Fix: Reshape back to original dimensions
        output = output_flat.reshape(batch_size, seq_len, hidden_size)
        
        stats = {
            'coarse_routing': coarse_probs.detach().reshape(batch_size, seq_len, -1),
            'group_distribution': group_indices.float().reshape(batch_size, seq_len).mean(dim=1),
            'auxiliary_loss': total_aux_loss
        }
        
        # Return aux_loss as float for compatibility with MLP interface
        return output, total_aux_loss


class ContinuousExperts(nn.Module):
    """Continuous expert interpolation with smooth blending"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.temperature = config.continuous_expert_temperature
        
        # Use dynamic intermediate size based on hidden size
        if config.intermediate_size is None:
            expert_dim = self.hidden_size * 4
        else:
            expert_dim = config.intermediate_size
            # Ensure compatibility
            if expert_dim / self.hidden_size > 32:
                expert_dim = self.hidden_size * 4
        
        self.expert_embeddings = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size) / math.sqrt(self.hidden_size)
        )
        
        self.expert_params = nn.ParameterList([
            nn.Parameter(torch.randn(expert_dim, self.hidden_size) / math.sqrt(self.hidden_size))
            for _ in range(self.num_experts)
        ])
        
        self.expert_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(expert_dim))
            for _ in range(self.num_experts)
        ])
        
        self.output_params = nn.ParameterList([
            nn.Parameter(torch.randn(self.hidden_size, expert_dim) / math.sqrt(expert_dim))
            for _ in range(self.num_experts)
        ])
        
        self.output_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.num_experts)
        ])
        
        self.router = nn.Linear(self.hidden_size, self.num_experts)
        
    def forward(self, x: torch.Tensor, mod_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        batch_size, seq_len, hidden_size = x.shape
        
        # Fix: Simplified implementation
        x_flat = x.reshape(-1, hidden_size)
        
        routing_logits = self.router(x_flat) / self.temperature
        expert_weights = F.softmax(routing_logits, dim=-1)
        
        # Simple weighted sum of expert outputs
        output_flat = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            # Apply expert i
            hidden = F.relu(F.linear(x_flat, self.expert_params[i], self.expert_biases[i]))
            expert_out = F.linear(hidden, self.output_params[i], self.output_biases[i])
            
            # Weight by expert weight
            weight = expert_weights[:, i:i+1]
            output_flat += weight * expert_out
        
        output = output_flat.reshape(batch_size, seq_len, hidden_size)
        
        stats = {
            'expert_weights': expert_weights.detach().reshape(batch_size, seq_len, -1),
            'weight_entropy': -(expert_weights * torch.log(expert_weights + 1e-10)).sum(dim=-1).mean(),
            'temperature': self.temperature
        }
        
        # Return 0.0 as aux_loss for compatibility
        return output, 0.0


class MixtureOfTokenizers(nn.Module):
    """Multiple tokenization strategies with dynamic selection"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # Three tokenizer types
        self.byte_embedding = nn.Embedding(256, config.hidden_size)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.char_embedding = nn.Embedding(1000, config.hidden_size)
        
        # Fix: Use hidden_size for router input
        self.tokenizer_selector = nn.Linear(config.hidden_size, 3)
        self.fusion_layer = nn.Linear(config.hidden_size * 3, config.hidden_size)
        
    def forward(self, input_ids: torch.Tensor, input_type: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        # Get embeddings from different tokenizers
        byte_embeddings = self.byte_embedding(torch.clamp(input_ids, 0, 255))
        word_embeddings = self.word_embedding(input_ids)
        char_embeddings = self.char_embedding(torch.clamp(input_ids, 0, 999))
        
        # Fix: Use mean of embeddings for selection
        mean_embedding = (byte_embeddings + word_embeddings + char_embeddings) / 3.0
        
        selection_logits = self.tokenizer_selector(mean_embedding.mean(dim=1))
        selection_weights = F.softmax(selection_logits, dim=-1)
        
        combined = torch.cat([byte_embeddings, word_embeddings, char_embeddings], dim=-1)
        
        weighted_embeddings = (
            selection_weights[:, 0:1].unsqueeze(1) * byte_embeddings +
            selection_weights[:, 1:2].unsqueeze(1) * word_embeddings +
            selection_weights[:, 2:3].unsqueeze(1) * char_embeddings
        )
        
        output = self.fusion_layer(combined) + weighted_embeddings
        
        stats = {
            'tokenizer_weights': selection_weights.detach(),
            'dominant_tokenizer': selection_weights.argmax(dim=-1),
        }
        
        return output, stats


class MoELayer(nn.Module):
    """Single MoE++ Transformer layer with MoD and MQA/GQA"""
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Multi-Query/Grouped-Query Attention with variants
        # Use universal attention if available for better compatibility
        if UNIVERSAL_ATTENTION_AVAILABLE:
            self.self_attn = UniversalMultiQueryAttention(config)
        else:
            attention_variant = getattr(config, 'attention_variant', 'standard')
            if attention_variant == 'standard':
                self.self_attn = MultiQueryAttention(config)
            else:
                self.self_attn = create_attention_layer(config, attention_type=attention_variant)
        
        # Mixture of Experts FFN
        # Use memory-efficient layer if requested
        if getattr(config, 'use_memory_efficient_moe', False):
            self.mlp = MemoryEfficientMoELayer(config)
        else:
            self.mlp = HierarchicalExpertLayer(config)
        
        # Normalization layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Dropout layers for regularization
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Mixture of Depths components
        if config.use_mod:
            self.depth_router = DepthRouter(config.hidden_size, mode=config.mod_mode)
            self.mod = MixtureOfDepths(
                config.hidden_size,
                capacity_factor=getattr(config, 'mod_capacity_factor', 1.25),
                skip_fraction=config.mod_skip_fraction
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass with MoD routing and expert selection
        
        Returns:
            hidden_states: Output tensor
            self_attn_weights: Attention weights (if output_attentions=True)
            present_key_value: Cache for next step (if use_cache=True)
            aux_losses: Dictionary of auxiliary losses
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.hidden_dropout(hidden_states)  # Apply dropout after norm
        
        # Mixture of Depths routing decision
        if self.config.use_mod:
            depth_scores = self.depth_router(hidden_states)
            mod_mask, mod_aux_loss = self.mod(hidden_states, depth_scores)
        else:
            mod_mask = None
            mod_aux_loss = 0.0
        
        # Self-attention with MQA/GQA
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            mod_mask=mod_mask,
        )
        
        # Apply dropout and residual connection
        attn_output = self.attention_dropout(attn_output)
        hidden_states = residual + attn_output
        
        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.hidden_dropout(hidden_states)  # Apply dropout after norm
        
        # Expert routing and computation
        mlp_output, expert_aux_loss = self.mlp(hidden_states, mod_mask)
        hidden_states = residual + mlp_output
        
        aux_losses = {
            "mod_loss": mod_aux_loss,
            "expert_loss": expert_aux_loss,
        }
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
            
        outputs += (aux_losses,)
        
        return outputs

class MoEModel(nn.Module):
    """MoE++ Transformer Model with all advanced features"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        
        # Token embeddings or Mixture of Tokenizers
        if config.use_mixture_tokenizers:
            self.embed_tokens = MixtureOfTokenizers(config)
            self.use_mixture_tokenizers = True
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, 
                config.hidden_size, 
                self.padding_idx
            )
            self.use_mixture_tokenizers = False
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MoELayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # Advanced Architecture Components
        if config.use_mod_plus_plus:
            self.mod_plus_plus = MoDPlusPlus(config)
        else:
            self.mod_plus_plus = None
            
        if config.use_hierarchical_moe:
            # Replace standard expert layers with hierarchical ones
            for layer in self.layers:
                # Wrap to match expected interface
                original_mlp = HierarchicalMoEPlus(config)
                layer.mlp = self._create_mlp_wrapper(original_mlp)
                
        if config.use_continuous_experts:
            # Replace with continuous expert layers
            for layer in self.layers:
                original_experts = ContinuousExperts(config)
                layer.mlp = self._create_mlp_wrapper(original_experts)
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Rotary embeddings - use universal if available
        if UNIVERSAL_ATTENTION_AVAILABLE:
            self.rotary_emb = UniversalRotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )
        elif getattr(config, 'use_xpos', False):
            self.rotary_emb = xPosRotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scale_base=getattr(config, 'xpos_scale_base', 512)
            )
        else:
            # Fallback to xPos as it's the only one available
            self.rotary_emb = xPosRotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scale_base=getattr(config, 'xpos_scale_base', 512)
            )
        
        # Add lm_head for compatibility with training code expecting logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights with proper scaling"""
        def _init_weights(module):
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        
        self.apply(_init_weights)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a pretrained model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load model on (auto-detect if None)
            
        Returns:
            Loaded model instance
        """
        import torch
        from pathlib import Path
        from LLM.src.utils.config import load_config
        
        checkpoint_path = Path(checkpoint_path)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Load config
        config = None
        for config_name in ["config.yaml", "config.json"]:
            config_path = checkpoint_path / config_name
            if config_path.exists():
                config = load_config(config_path)
                break
        
        if config is None:
            # Try parent directory
            config_path = checkpoint_path.parent.parent / "config.yaml"
            if config_path.exists():
                config = load_config(config_path)
            else:
                raise ValueError(f"Could not find config in {checkpoint_path}")
        
        # Initialize model
        model = cls(config.model)
        
        # Find model file
        model_file = None
        for name in ["pytorch_model.bin", "model.pt", "checkpoint.pt"]:
            path = checkpoint_path / name
            if path.exists():
                model_file = path
                break
        
        if model_file is None:
            # Look for any .pt or .bin file
            pt_files = list(checkpoint_path.glob("*.pt"))
            bin_files = list(checkpoint_path.glob("*.bin"))
            if pt_files:
                model_file = pt_files[0]
            elif bin_files:
                model_file = bin_files[0]
            else:
                raise ValueError(f"No model file found in {checkpoint_path}")
        
        # Load state dict
        state_dict = torch.load(model_file, map_location=device)
        
        # Handle different state dict formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Remove module. prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Load weights
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        
        return model
    
    def _create_mlp_wrapper(self, mlp_module):
        """Create a wrapper that adapts any MLP module to the expected interface"""
        class MLPWrapper(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            
            def forward(self, hidden_states, mod_mask=None):
                # Try to call with both arguments
                try:
                    result = self.wrapped(hidden_states, mod_mask)
                except:
                    # Fallback to single argument
                    result = self.wrapped(hidden_states)
                
                # Ensure we return (output, aux_loss) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                elif isinstance(result, tuple):
                    return result[0], result[1] if len(result) > 1 else 0.0
                else:
                    return result, 0.0
        
        return MLPWrapper(mlp_module)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the MoE++ transformer
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Input embeddings
        if inputs_embeds is None:
            if self.use_mixture_tokenizers:
                inputs_embeds, tokenizer_stats = self.embed_tokens(input_ids)
            else:
                inputs_embeds = self.embed_tokens(input_ids)
                tokenizer_stats = None
        else:
            tokenizer_stats = None
        
        hidden_states = inputs_embeds
        
        # Position embeddings
        seq_length = hidden_states.shape[1]
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        
        # Attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_4d_causal_attention_mask(
                attention_mask, hidden_states.shape[:2], hidden_states.dtype
            )
        
        # Pass rotary embeddings to attention layers
        for layer in self.layers:
            if hasattr(layer.self_attn, 'rotary_emb'):
                layer.self_attn.rotary_emb = self.rotary_emb
        
        # Initialize cache
        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * len(self.layers)
        
        # Collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_aux_losses = []
        next_cache = [] if use_cache else None
        
        # Forward through layers (with optional MoD++)
        mod_stats = None
        if self.config.use_mod_plus_plus and self.mod_plus_plus is not None:
            # Use MoD++ for dynamic depth routing
            hidden_states, mod_stats = self.mod_plus_plus(hidden_states, self.layers)
            # Skip regular layer processing since MoD++ handles it
        else:
            # Regular forward through layers
            for idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                
                past_key_value = past_key_values[idx] if past_key_values is not None else None
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_value,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                
                hidden_states = layer_outputs[0]
                
                if use_cache:
                    next_cache.append(layer_outputs[2 if output_attentions else 1])
                
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                
                # Collect auxiliary losses
                aux_losses = layer_outputs[-1]
                all_aux_losses.append(aux_losses)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Compute total auxiliary loss
        total_aux_loss = 0.0
        for aux_losses in all_aux_losses:
            total_aux_loss += aux_losses.get("mod_loss", 0.0) * self.config.aux_loss_coef
            total_aux_loss += aux_losses.get("expert_loss", 0.0) * self.config.router_aux_loss_coef
        
        # Compute logits for compatibility
        logits = self.lm_head(hidden_states)
        
        # Prepare output dictionary
        output_dict = {
            "last_hidden_state": hidden_states,
            "logits": logits,  # Add logits for compatibility
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "aux_loss": total_aux_loss,
        }
        
        # Add optional statistics
        if tokenizer_stats is not None:
            output_dict["tokenizer_stats"] = tokenizer_stats
        if mod_stats is not None:
            output_dict["mod_stats"] = mod_stats
            
        return output_dict

    def _prepare_4d_causal_attention_mask(self, attention_mask, input_shape, dtype):
        """Prepare 4D causal attention mask"""
        batch_size, seq_length = input_shape
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool), 
            diagonal=1
        )
        causal_mask = causal_mask.to(dtype).masked_fill(causal_mask, float("-inf"))
        causal_mask = causal_mask.to(attention_mask.device)
        
        # Expand for batch and heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            # Expand attention_mask
            expanded_mask = attention_mask[:, None, None, :].to(dtype)
            inverted_mask = (1.0 - expanded_mask) * -10000.0  # Use large negative value instead of -inf
            causal_mask = causal_mask.masked_fill(causal_mask == float("-inf"), -10000.0)  # Replace -inf with large negative
            causal_mask = causal_mask + inverted_mask
        
        return causal_mask

class MoEForCausalLM(nn.Module):
    """MoE++ Model for Causal Language Modeling"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.model = MoEModel(config)
        self.vocab_size = config.vocab_size
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight
        
        # Initialize weights
        self.post_init()
        
        # Enable PyTorch 2.0 compile if available and requested
        # Check if using advanced features that may have dimension issues
        has_advanced_features = any([
            getattr(config, 'use_hierarchical_moe', False),
            getattr(config, 'use_continuous_experts', False),
            getattr(config, 'use_mod_plus_plus', False),
            getattr(config, 'use_mixture_of_tokenizers', False),
        ])
        
        # Disable torch.compile for MPS devices as it causes errors
        is_mps = torch.backends.mps.is_available() and torch.cuda.is_available() == False
        if getattr(config, 'use_torch_compile', False) and hasattr(torch, '__version__') and torch.__version__ >= '2.0' and not has_advanced_features and not is_mps:
            try:
                compile_mode = getattr(config, 'torch_compile_mode', 'reduce-overhead')
                compile_backend = getattr(config, 'torch_compile_backend', 'inductor')
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    backend=compile_backend,
                    fullgraph=False  # Set to False for better compatibility
                )
                logger.info(f"Enabled PyTorch 2.0 compile with mode='{compile_mode}', backend='{compile_backend}'")
            except Exception as e:
                logger.warning(f"Failed to compile model with PyTorch 2.0: {e}")
        elif has_advanced_features and getattr(config, 'use_torch_compile', False):
            logger.info("torch.compile disabled due to advanced features that may have dimension incompatibilities")

    def post_init(self):
        """Initialize lm_head weights"""
        if not getattr(self.config, 'tie_word_embeddings', False):
            self.lm_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: Optional[str] = None):
        """Load a pretrained model from checkpoint"""
        # Use the base model's from_pretrained to load config and find files
        import torch
        from pathlib import Path
        from LLM.src.utils.config import load_config
        
        checkpoint_path = Path(checkpoint_path)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Load config
        config = None
        for config_name in ["config.yaml", "config.json"]:
            config_path = checkpoint_path / config_name
            if config_path.exists():
                config = load_config(config_path)
                break
        
        if config is None:
            config_path = checkpoint_path.parent.parent / "config.yaml"
            if config_path.exists():
                config = load_config(config_path)
            else:
                raise ValueError(f"Could not find config in {checkpoint_path}")
        
        # Initialize model
        model = cls(config.model)
        
        # Find and load checkpoint
        model_file = None
        for name in ["pytorch_model.bin", "model.pt", "checkpoint.pt"]:
            path = checkpoint_path / name
            if path.exists():
                model_file = path
                break
        
        if model_file is None:
            pt_files = list(checkpoint_path.glob("*.pt"))
            bin_files = list(checkpoint_path.glob("*.bin"))
            if pt_files:
                model_file = pt_files[0]
            elif bin_files:
                model_file = bin_files[0]
            else:
                raise ValueError(f"No model file found in {checkpoint_path}")
        
        state_dict = torch.load(model_file, map_location=device)
        
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Remove module. prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional loss computation
        """
        # Model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss with label smoothing, ignoring padding tokens
            label_smoothing = getattr(self.config, 'label_smoothing', 0.1)
            # Ignore padding tokens (assumed to be -100 or pad_token_id)
            pad_token_id = getattr(self.config, 'pad_token_id', -100)
            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                ignore_index=pad_token_id
            )
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), 
                shift_labels.view(-1)
            )
            
            # Add auxiliary losses (check for NaN)
            aux_loss = outputs.get("aux_loss", 0.0)
            if aux_loss is not None and aux_loss != 0.0:
                # Convert to tensor if needed
                if not isinstance(aux_loss, torch.Tensor):
                    aux_loss = torch.tensor(aux_loss, device=loss.device, dtype=loss.dtype)
                # Handle NaN auxiliary loss
                if torch.isnan(aux_loss):
                    print(f"Warning: NaN auxiliary loss detected, using 0.0")
                    aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                loss = loss + aux_loss * 0.01  # Scale down aux loss to prevent domination
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
            "aux_loss": outputs.get("aux_loss"),
        }

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs
    ):
        """Prepare inputs for generation step"""
        # Only use last token for generation with cache
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # Position ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        
        # Cache position
        if cache_position is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values else 0
            cache_position = torch.arange(
                past_seen_tokens, 
                past_seen_tokens + input_ids.shape[1], 
                device=input_ids.device
            )
        
        # Handle inputs_embeds
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        })
        
        return model_inputs

    @staticmethod
    def from_config(config: MoEConfig):
        """Create model from configuration"""
        return MoEForCausalLM(config)

    def save_pretrained(self, save_directory: str):
        """Save model and configuration"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save model state
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load model from pretrained weights"""
        import os
        import json
        
        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = MoEConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model