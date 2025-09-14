"""
Progressive Model Growing and Training

Implements progressive training strategies where models start small
and grow during training for better convergence and efficiency.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import copy
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass 
class ProgressiveConfig:
    """Configuration for progressive training"""
    # Growth schedule
    growth_schedule: List[Dict[str, Any]] = None
    
    # Layer growth
    start_num_layers: int = 6
    final_num_layers: int = 24
    layer_growth_steps: List[int] = None
    
    # Hidden size growth
    start_hidden_size: int = 512
    final_hidden_size: int = 2048
    hidden_growth_steps: List[int] = None
    
    # Expert growth
    start_num_experts: int = 8
    final_num_experts: int = 64
    expert_growth_steps: List[int] = None
    
    # Attention head growth
    start_num_heads: int = 8
    final_num_heads: int = 32
    head_growth_steps: List[int] = None
    
    # Knowledge transfer
    use_knowledge_distillation: bool = True
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    
    # Initialization strategy
    init_strategy: str = "inherit"  # "inherit", "random", "function_preserving"
    
    # Training stabilization
    stabilization_steps: int = 1000  # Steps after growth before next growth
    lr_reset_on_growth: bool = True
    warmup_after_growth: int = 500
    
    def __post_init__(self):
        if self.growth_schedule is None:
            self.growth_schedule = [
                {"step": 0, "num_layers": 6, "hidden_size": 512, "num_experts": 8, "num_heads": 8},
                {"step": 10000, "num_layers": 12, "hidden_size": 768, "num_experts": 16, "num_heads": 12},
                {"step": 50000, "num_layers": 18, "hidden_size": 1024, "num_experts": 32, "num_heads": 16},
                {"step": 100000, "num_layers": 24, "hidden_size": 1536, "num_experts": 48, "num_heads": 24},
                {"step": 200000, "num_layers": 24, "hidden_size": 2048, "num_experts": 64, "num_heads": 32},
            ]


class ProgressiveModelGrower:
    """
    Manages progressive model growth during training
    """
    
    def __init__(
        self,
        base_config: Any,
        progressive_config: ProgressiveConfig,
        model_class: type
    ):
        self.base_config = base_config
        self.progressive_config = progressive_config
        self.model_class = model_class
        
        self.current_stage = 0
        self.current_model = None
        self.previous_model = None
        self.growth_history = []
        
        # Initialize with smallest model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model with the smallest configuration"""
        first_stage = self.progressive_config.growth_schedule[0]
        
        # Create config for first stage
        config = copy.deepcopy(self.base_config)
        self._update_config(config, first_stage)
        
        # Create model
        self.current_model = self.model_class(config)
        logger.info(f"Initialized model with config: {first_stage}")
        
    def _update_config(self, config: Any, stage: Dict[str, Any]):
        """Update model config based on growth stage"""
        if "num_layers" in stage:
            config.num_hidden_layers = stage["num_layers"]
        if "hidden_size" in stage:
            config.hidden_size = stage["hidden_size"]
        if "num_experts" in stage:
            config.num_experts = stage["num_experts"]
        if "num_heads" in stage:
            config.num_attention_heads = stage["num_heads"]
            
    def should_grow(self, step: int) -> bool:
        """Check if model should grow at current step"""
        if self.current_stage >= len(self.progressive_config.growth_schedule) - 1:
            return False
            
        next_stage = self.progressive_config.growth_schedule[self.current_stage + 1]
        return step >= next_stage["step"]
        
    def grow_model(self, step: int) -> Tuple[nn.Module, Dict[str, Any]]:
        """Grow the model to the next stage"""
        self.current_stage += 1
        stage_config = self.progressive_config.growth_schedule[self.current_stage]
        
        logger.info(f"Growing model at step {step} to stage {self.current_stage}: {stage_config}")
        
        # Save previous model for distillation
        if self.progressive_config.use_knowledge_distillation:
            self.previous_model = copy.deepcopy(self.current_model)
            self.previous_model.eval()
            
        # Create new model config
        new_config = copy.deepcopy(self.base_config)
        self._update_config(new_config, stage_config)
        
        # Create new model
        new_model = self.model_class(new_config)
        
        # Transfer knowledge from old model
        self._transfer_knowledge(self.current_model, new_model, stage_config)
        
        # Update current model
        self.current_model = new_model
        
        # Record growth event
        growth_info = {
            "step": step,
            "stage": self.current_stage,
            "config": stage_config,
            "param_count": sum(p.numel() for p in new_model.parameters())
        }
        self.growth_history.append(growth_info)
        
        return self.current_model, growth_info
        
    def _transfer_knowledge(
        self,
        old_model: nn.Module,
        new_model: nn.Module,
        stage_config: Dict[str, Any]
    ):
        """Transfer knowledge from old model to new model"""
        if self.progressive_config.init_strategy == "inherit":
            self._inherit_weights(old_model, new_model)
        elif self.progressive_config.init_strategy == "function_preserving":
            self._function_preserving_init(old_model, new_model, stage_config)
        # else: random initialization
        
    def _inherit_weights(self, old_model: nn.Module, new_model: nn.Module):
        """Inherit weights from old model where possible"""
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        
        for name, param in old_state.items():
            if name in new_state:
                old_shape = param.shape
                new_shape = new_state[name].shape
                
                if old_shape == new_shape:
                    # Direct copy
                    new_state[name] = param
                else:
                    # Need to resize
                    new_state[name] = self._resize_weight(param, new_shape)
                    
        new_model.load_state_dict(new_state)
        
    def _resize_weight(self, weight: torch.Tensor, new_shape: Tuple[int, ...]) -> torch.Tensor:
        """Resize weight tensor to new shape"""
        # Initialize new weight
        new_weight = torch.zeros(new_shape, dtype=weight.dtype, device=weight.device)
        
        # Copy over the overlapping region
        slices = []
        for old_dim, new_dim in zip(weight.shape, new_shape):
            slices.append(slice(0, min(old_dim, new_dim)))
            
        new_weight[tuple(slices)] = weight[tuple(slices)]
        
        # Initialize new parts
        if any(n > o for n, o in zip(new_shape, weight.shape)):
            # Use Xavier/He initialization for new parts
            fan_in = new_shape[0] if len(new_shape) > 1 else 1
            fan_out = new_shape[1] if len(new_shape) > 1 else 1
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            
            # Initialize only the new parts
            for i, (old_dim, new_dim) in enumerate(zip(weight.shape, new_shape)):
                if new_dim > old_dim:
                    idx = [slice(None)] * len(new_shape)
                    idx[i] = slice(old_dim, new_dim)
                    new_weight[tuple(idx)] = torch.randn_like(new_weight[tuple(idx)]) * std
                    
        return new_weight
        
    def _function_preserving_init(
        self,
        old_model: nn.Module,
        new_model: nn.Module,
        stage_config: Dict[str, Any]
    ):
        """Initialize new model to preserve function of old model"""
        # This is more complex and depends on the specific architecture
        # For now, fall back to inherit strategy
        self._inherit_weights(old_model, new_model)
        
        # Additional function-preserving adjustments
        if "hidden_size" in stage_config:
            old_hidden = old_model.config.hidden_size
            new_hidden = stage_config["hidden_size"]
            
            # Scale weights to preserve activation magnitudes
            scale = (old_hidden / new_hidden) ** 0.5
            
            for name, param in new_model.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    param.data *= scale
                    
    def get_distillation_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute distillation loss if previous model exists"""
        if self.previous_model is None:
            return 0.0
            
        with torch.no_grad():
            teacher_outputs = self.previous_model(**inputs)
            
        # KL divergence loss
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs.logits
        
        # Match dimensions if needed
        if student_logits.shape != teacher_logits.shape:
            # Truncate or pad as needed
            min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]
            
        # Compute KL divergence
        student_log_probs = F.log_softmax(
            student_logits / self.progressive_config.distillation_temperature,
            dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.progressive_config.distillation_temperature,
            dim=-1
        )
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (self.progressive_config.distillation_temperature ** 2)
        
        return kl_loss * self.progressive_config.distillation_alpha
        
    def get_current_config(self) -> Dict[str, Any]:
        """Get current model configuration"""
        return self.progressive_config.growth_schedule[self.current_stage]
        
    def get_growth_factor(self) -> Dict[str, float]:
        """Get growth factors compared to initial model"""
        initial = self.progressive_config.growth_schedule[0]
        current = self.progressive_config.growth_schedule[self.current_stage]
        
        factors = {}
        for key in ["num_layers", "hidden_size", "num_experts", "num_heads"]:
            if key in initial and key in current:
                factors[key] = current[key] / initial[key]
                
        return factors


class LayerStackingGrower:
    """
    Specialized grower for layer stacking (depth growth)
    """
    
    def __init__(self, base_model: nn.Module, target_layers: int):
        self.base_model = base_model
        self.target_layers = target_layers
        self.current_layers = len(base_model.layers)
        
    def add_layers(self, num_new_layers: int) -> nn.Module:
        """Add new layers to the model"""
        if self.current_layers >= self.target_layers:
            return self.base_model
            
        num_to_add = min(num_new_layers, self.target_layers - self.current_layers)
        
        # Create new layers by copying existing ones
        new_layers = []
        for i in range(num_to_add):
            # Copy from existing layers in round-robin fashion
            source_idx = i % self.current_layers
            new_layer = copy.deepcopy(self.base_model.layers[source_idx])
            
            # Add noise to break symmetry
            for param in new_layer.parameters():
                param.data += torch.randn_like(param) * 0.01
                
            new_layers.append(new_layer)
            
        # Insert new layers
        # Strategy: distribute evenly throughout the model
        insert_positions = self._compute_insert_positions(num_to_add)
        
        for i, (pos, layer) in enumerate(zip(insert_positions, new_layers)):
            self.base_model.layers.insert(pos + i, layer)
            
        self.current_layers += num_to_add
        logger.info(f"Added {num_to_add} layers, total now: {self.current_layers}")
        
        return self.base_model
        
    def _compute_insert_positions(self, num_new_layers: int) -> List[int]:
        """Compute positions to insert new layers"""
        # Distribute evenly
        positions = []
        step = self.current_layers / (num_new_layers + 1)
        
        for i in range(num_new_layers):
            pos = int((i + 1) * step)
            positions.append(pos)
            
        return positions


class ExpertGrower:
    """
    Specialized grower for expert growth
    """
    
    def __init__(self, moe_layer: nn.Module, target_experts: int):
        self.moe_layer = moe_layer
        self.target_experts = target_experts
        self.current_experts = len(moe_layer.experts)
        
    def add_experts(self, num_new_experts: int) -> None:
        """Add new experts to MoE layer"""
        if self.current_experts >= self.target_experts:
            return
            
        num_to_add = min(num_new_experts, self.target_experts - self.current_experts)
        
        # Create new experts
        for i in range(num_to_add):
            # Clone from existing expert
            source_idx = i % self.current_experts
            new_expert = copy.deepcopy(self.moe_layer.experts[source_idx])
            
            # Reinitialize with some noise
            for param in new_expert.parameters():
                param.data = torch.randn_like(param) * 0.02
                
            self.moe_layer.experts.append(new_expert)
            
        # Update router dimensions
        old_router = self.moe_layer.router
        new_router = nn.Linear(
            old_router.in_features,
            self.current_experts + num_to_add,
            bias=old_router.bias is not None
        )
        
        # Copy old router weights
        with torch.no_grad():
            new_router.weight[:self.current_experts] = old_router.weight
            if old_router.bias is not None:
                new_router.bias[:self.current_experts] = old_router.bias
                
        self.moe_layer.router = new_router
        self.moe_layer.num_experts = self.current_experts + num_to_add
        self.current_experts += num_to_add
        
        logger.info(f"Added {num_to_add} experts, total now: {self.current_experts}")