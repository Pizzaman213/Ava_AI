"""
Mixture of LoRA Experts for efficient fine-tuning

Combines Low-Rank Adaptation (LoRA) with Mixture of Experts
for parameter-efficient and task-specific fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math

from ..model.experts import HierarchicalRouter


@dataclass
class LoRAConfig:
    """Configuration for LoRA experts"""
    r: int = 16  # Rank
    alpha: int = 32  # Scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None
    
    # MoLoRA specific
    num_lora_experts: int = 8
    experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    
    # Task-specific experts
    task_specific_experts: Dict[str, List[int]] = None
    share_base_weights: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
        if self.task_specific_experts is None:
            self.task_specific_experts = {
                "general": [0, 1],
                "code": [2, 3],
                "math": [4, 5],
                "chat": [6, 7]
            }
    
    @property
    def scaling(self):
        return self.alpha / self.r


class LoRAExpert(nn.Module):
    """
    Single LoRA expert module
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        expert_id: int
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.expert_id = expert_id
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros((config.r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, config.r)))
        self.lora_dropout = nn.Dropout(p=config.dropout)
        
        # Task embedding for task-conditioned experts
        self.task_embedding = nn.Embedding(10, in_features)  # Support up to 10 tasks
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA weights"""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
        
    def forward(
        self,
        x: torch.Tensor,
        base_output: Optional[torch.Tensor] = None,
        task_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through LoRA expert
        
        Args:
            x: Input tensor
            base_output: Output from base model (if share_base_weights=True)
            task_id: Task identifier for task-specific adaptation
        """
        # Apply dropout
        x = self.lora_dropout(x)
        
        # Task conditioning
        if task_id is not None:
            task_emb = self.task_embedding(torch.tensor([task_id], device=x.device))
            x = x + task_emb.expand_as(x)
        
        # LoRA computation: (x @ A^T) @ B^T
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        
        # Scale by alpha/r
        lora_output = lora_output * self.config.scaling
        
        # Add to base output if provided
        if base_output is not None:
            return base_output + lora_output
        else:
            return lora_output
            
    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights with base weights for inference"""
        # Compute BA
        delta_weight = self.lora_B @ self.lora_A
        # Scale and add to base
        return base_weight + (delta_weight * self.config.scaling)


class MixtureOfLoRAExperts(nn.Module):
    """
    Mixture of LoRA Experts layer
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        base_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.base_layer = base_layer
        
        # Create LoRA experts
        self.experts = nn.ModuleList([
            LoRAExpert(in_features, out_features, config, i)
            for i in range(config.num_lora_experts)
        ])
        
        # Router for expert selection
        self.router = nn.Linear(in_features, config.num_lora_experts)
        
        # Load balancing loss
        self.load_balancing_loss_weight = 0.01
        
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        return_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through MoLoRA layer
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Get base output if available
        base_output = None
        if self.base_layer is not None and self.config.share_base_weights:
            base_output = self.base_layer(x)
            
        # Flatten for routing
        x_flat = x.view(-1, hidden_size)
        
        # Compute routing probabilities
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            router_probs, 
            self.config.experts_per_token,
            dim=-1
        )
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Task-specific expert filtering
        if task_id is not None and task_id in self.config.task_specific_experts:
            allowed_experts = self.config.task_specific_experts[task_id]
            # Mask out non-allowed experts
            mask = torch.zeros_like(router_probs)
            mask[:, allowed_experts] = 1.0
            router_probs = router_probs * mask
            router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)
            
            # Re-select top-k from allowed experts
            top_k_probs, top_k_indices = torch.topk(
                router_probs,
                min(self.config.experts_per_token, len(allowed_experts)),
                dim=-1
            )
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros(
            (batch_size * seq_len, self.out_features),
            device=x.device,
            dtype=x.dtype
        )
        
        # Process each expert
        for i in range(self.config.experts_per_token):
            expert_mask = F.one_hot(
                top_k_indices[:, i],
                num_classes=self.config.num_lora_experts
            ).float()
            
            # Compute expert outputs
            for expert_idx, expert in enumerate(self.experts):
                # Get tokens routed to this expert
                expert_mask_i = expert_mask[:, expert_idx]
                if expert_mask_i.sum() > 0:
                    expert_input = x_flat[expert_mask_i.bool()]
                    
                    # Get base output for these tokens if needed
                    expert_base = None
                    if base_output is not None:
                        expert_base = base_output.view(-1, self.out_features)[expert_mask_i.bool()]
                    
                    expert_output = expert(expert_input, expert_base, task_id)
                    
                    # Weighted combine
                    output[expert_mask_i.bool()] += (
                        expert_output * top_k_probs[expert_mask_i.bool(), i].unsqueeze(-1)
                    )
        
        # Add base output for tokens that didn't go through experts (if any)
        if base_output is not None and self.config.share_base_weights:
            no_expert_mask = (top_k_probs.sum(dim=-1) == 0)
            if no_expert_mask.any():
                output[no_expert_mask] = base_output.view(-1, self.out_features)[no_expert_mask]
        
        # Reshape output
        output = output.view(batch_size, seq_len, self.out_features)
        
        # Compute auxiliary losses
        aux_loss = self._compute_load_balancing_loss(router_probs)
        
        if return_router_logits:
            return output, {
                "router_logits": router_logits.view(batch_size, seq_len, -1),
                "selected_experts": top_k_indices.view(batch_size, seq_len, -1),
                "aux_loss": aux_loss
            }
        
        return output, aux_loss
        
    def _compute_load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss"""
        # Compute expert load
        expert_load = router_probs.mean(dim=0)
        
        # Target uniform distribution
        uniform_load = 1.0 / self.config.num_lora_experts
        
        # L2 loss
        loss = ((expert_load - uniform_load) ** 2).sum()
        
        return loss * self.load_balancing_loss_weight
        
    def merge_all_experts(self) -> torch.Tensor:
        """Merge all LoRA experts for deployment"""
        if self.base_layer is None:
            raise ValueError("Cannot merge without base layer")
            
        base_weight = self.base_layer.weight.data
        merged_weight = base_weight.clone()
        
        # Average all expert deltas
        for expert in self.experts:
            merged_weight += expert.merge_weights(torch.zeros_like(base_weight)) / len(self.experts)
            
        return merged_weight
        
    def select_best_experts(self, num_experts: int = 1) -> List[int]:
        """Select best performing experts based on routing statistics"""
        # Get average routing probabilities
        router_stats = self.router.weight.data.abs().mean(dim=0)
        
        # Select top experts
        _, top_experts = torch.topk(router_stats, num_experts)
        
        return top_experts.tolist()


class LoRAMoEModel(nn.Module):
    """
    Complete model with Mixture of LoRA Experts
    """
    
    def __init__(self, base_model: nn.Module, lora_config: LoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config
        
        # Replace target modules with MoLoRA
        self.lora_layers = nn.ModuleDict()
        self._replace_with_lora()
        
        # Freeze base model parameters
        if lora_config.share_base_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
    def _replace_with_lora(self):
        """Replace target modules with MoLoRA layers"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.lora_config.target_modules):
                if isinstance(module, nn.Linear):
                    # Create MoLoRA layer
                    lora_layer = MixtureOfLoRAExperts(
                        module.in_features,
                        module.out_features,
                        self.lora_config,
                        base_layer=module if self.lora_config.share_base_weights else None
                    )
                    
                    # Store for later access
                    self.lora_layers[name] = lora_layer
                    
                    # Replace in model
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.base_model
                    
                    if parent_name:
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                            
                    setattr(parent_module, child_name, lora_layer)
                    
    def forward(self, *args, task_id: Optional[int] = None, **kwargs):
        """Forward pass with optional task conditioning"""
        # Set task_id for all LoRA layers
        for lora_layer in self.lora_layers.values():
            lora_layer.current_task_id = task_id
            
        # Forward through base model (which now includes LoRA layers)
        return self.base_model(*args, **kwargs)
        
    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights and return base model"""
        for name, lora_layer in self.lora_layers.items():
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = self.base_model
            
            if parent_name:
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
                    
            # Create merged linear layer
            merged_weight = lora_layer.merge_all_experts()
            merged_layer = nn.Linear(
                lora_layer.in_features,
                lora_layer.out_features,
                bias=lora_layer.base_layer.bias is not None
            )
            merged_layer.weight.data = merged_weight
            
            if lora_layer.base_layer.bias is not None:
                merged_layer.bias.data = lora_layer.base_layer.bias.data
                
            # Replace in model
            setattr(parent_module, child_name, merged_layer)
            
        # Unfreeze parameters
        for param in self.base_model.parameters():
            param.requires_grad = True
            
        return self.base_model
        
    def save_lora_weights(self, path: str):
        """Save only LoRA weights"""
        lora_state_dict = {}
        
        for name, lora_layer in self.lora_layers.items():
            for expert_idx, expert in enumerate(lora_layer.experts):
                prefix = f"{name}.expert_{expert_idx}"
                lora_state_dict[f"{prefix}.lora_A"] = expert.lora_A
                lora_state_dict[f"{prefix}.lora_B"] = expert.lora_B
                lora_state_dict[f"{prefix}.task_embedding"] = expert.task_embedding.weight
                
            lora_state_dict[f"{name}.router"] = lora_layer.router.weight
            
        torch.save({
            "lora_config": self.lora_config,
            "lora_weights": lora_state_dict
        }, path)
        
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        checkpoint = torch.load(path)
        
        # Load weights
        for name, param in checkpoint["lora_weights"].items():
            module_path = name.split('.')
            module = self
            
            for part in module_path[:-1]:
                module = getattr(module, part)
                
            setattr(module, module_path[-1], param)