"""
LoRA/QLoRA Integration for Parameter-Efficient Fine-Tuning
Implements Low-Rank Adaptation for reasoning-aware RLHF training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
import math
from collections import OrderedDict
import bitsandbytes as bnb

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """Configuration for LoRA/QLoRA adapters"""
    # LoRA parameters
    r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # QLoRA parameters
    use_qlora: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Task-specific adapters
    use_task_adapters: bool = True
    task_adapter_layers: List[str] = None
    num_tasks: int = 10
    
    # Reasoning-specific
    reasoning_adapter_r: int = 32  # Higher rank for reasoning
    reasoning_modules: List[str] = None
    
    # Progressive unfreezing
    progressive_unfreeze: bool = True
    unfreeze_schedule: List[int] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        if self.task_adapter_layers is None:
            self.task_adapter_layers = ["gate_proj", "up_proj", "down_proj"]
        
        if self.reasoning_modules is None:
            self.reasoning_modules = ["q_proj", "v_proj", "o_proj"]
        
        if self.unfreeze_schedule is None:
            self.unfreeze_schedule = [1000, 2000, 3000, 4000]  # Steps at which to unfreeze layers

class LoRALayer(nn.Module):
    """Base LoRA layer implementation"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA weights"""
        # Initialize A with normal distribution
        nn.init.normal_(self.lora_A, std=1 / math.sqrt(self.r))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation to base output"""
        # LoRA forward: y = Wx + (BA)x * scaling
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        lora_output = self.lora_dropout(lora_output)
        
        return base_output + lora_output * self.scaling

class QLoRALayer(LoRALayer):
    """QLoRA layer with 4-bit quantization"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        bnb_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(in_features, out_features, r, lora_alpha, lora_dropout)
        
        # Store quantization config
        self.bnb_config = bnb_config or {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    
    def quantize_base_layer(self, base_layer: nn.Linear) -> nn.Module:
        """Quantize base layer to 4-bit"""
        # Create 4-bit linear layer
        bnb_layer = bnb.nn.Linear4bit(
            base_layer.in_features,
            base_layer.out_features,
            bias=base_layer.bias is not None,
            compute_dtype=self.bnb_config["bnb_4bit_compute_dtype"],
            compress_statistics=self.bnb_config["bnb_4bit_use_double_quant"],
            quant_type=self.bnb_config["bnb_4bit_quant_type"],
        )
        
        # Copy weights
        bnb_layer.weight.data = base_layer.weight.data
        if base_layer.bias is not None:
            bnb_layer.bias.data = base_layer.bias.data
        
        return bnb_layer

class ReasoningLoRA(nn.Module):
    """Specialized LoRA adapter for reasoning tasks"""
    def __init__(
        self,
        base_model_dim: int,
        config: LoRAConfig,
        reasoning_type: str = "general"
    ):
        super().__init__()
        self.config = config
        self.reasoning_type = reasoning_type
        
        # Higher rank for reasoning complexity
        r = config.reasoning_adapter_r
        
        # Main reasoning adapter
        self.reasoning_adapter = LoRALayer(
            base_model_dim,
            base_model_dim,
            r=r,
            lora_alpha=config.lora_alpha * 2,  # Higher scaling for reasoning
            lora_dropout=config.lora_dropout
        )
        
        # Step-wise reasoning adapter
        self.step_adapter = LoRALayer(
            base_model_dim,
            base_model_dim,
            r=r // 2,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        
        # Reasoning type embeddings
        self.reasoning_type_embeddings = nn.Embedding(5, base_model_dim)
        
        # Reasoning gate
        self.reasoning_gate = nn.Sequential(
            nn.Linear(base_model_dim * 2, base_model_dim),
            nn.ReLU(),
            nn.Linear(base_model_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        reasoning_step: Optional[int] = None
    ) -> torch.Tensor:
        """Apply reasoning-aware LoRA adaptation"""
        # Get reasoning type embedding
        type_idx = self._get_reasoning_type_idx()
        type_embedding = self.reasoning_type_embeddings(
            torch.tensor([type_idx], device=x.device)
        ).expand(x.size(0), -1)
        
        # Apply main reasoning adapter
        reasoning_output = self.reasoning_adapter(x, base_output)
        
        # Apply step-wise adaptation if in multi-step reasoning
        if reasoning_step is not None and reasoning_step > 0:
            step_output = self.step_adapter(x, reasoning_output)
            
            # Gate between reasoning and step outputs
            gate_input = torch.cat([x.mean(dim=1), type_embedding], dim=-1)
            gate_value = self.reasoning_gate(gate_input)
            
            reasoning_output = gate_value * step_output + (1 - gate_value) * reasoning_output
        
        return reasoning_output
    
    def _get_reasoning_type_idx(self) -> int:
        """Map reasoning type to index"""
        type_map = {
            "general": 0,
            "mathematical": 1,
            "logical": 2,
            "scientific": 3,
            "code": 4
        }
        return type_map.get(self.reasoning_type, 0)

class TaskSpecificLoRA(nn.Module):
    """Task-specific LoRA adapters with routing"""
    def __init__(
        self,
        base_model_dim: int,
        config: LoRAConfig,
        num_tasks: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.num_tasks = num_tasks or config.num_tasks
        
        # Create task-specific adapters
        self.task_adapters = nn.ModuleList([
            LoRALayer(
                base_model_dim,
                base_model_dim,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            )
            for _ in range(self.num_tasks)
        ])
        
        # Task router
        self.task_router = nn.Sequential(
            nn.Linear(base_model_dim, base_model_dim // 2),
            nn.ReLU(),
            nn.Linear(base_model_dim // 2, self.num_tasks),
            nn.Softmax(dim=-1)
        )
        
        # Shared adapter for common patterns
        self.shared_adapter = LoRALayer(
            base_model_dim,
            base_model_dim,
            r=config.r * 2,  # Larger shared adapter
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        task_id: Optional[int] = None
    ) -> torch.Tensor:
        """Apply task-specific adaptation"""
        # Apply shared adapter first
        shared_output = self.shared_adapter(x, base_output)
        
        if task_id is not None:
            # Use specific task adapter
            task_output = self.task_adapters[task_id](x, shared_output)
            return task_output
        else:
            # Route to task adapters
            router_input = x.mean(dim=1) if x.dim() > 2 else x
            routing_weights = self.task_router(router_input)
            
            # Weighted combination of task adapters
            task_output = torch.zeros_like(shared_output)
            for i, adapter in enumerate(self.task_adapters):
                weight = routing_weights[:, i:i+1]
                if weight.max() > 0.01:  # Skip negligible weights
                    task_output += weight.unsqueeze(-1) * adapter(x, shared_output)
            
            return task_output

class ProgressiveUnfreezing:
    """Manages progressive unfreezing of model layers during training"""
    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.current_stage = 0
        
        # Identify layer groups
        self.layer_groups = self._identify_layer_groups()
        
        # Initially freeze all base model parameters
        self._freeze_base_model()
    
    def _identify_layer_groups(self) -> List[List[str]]:
        """Identify groups of layers for progressive unfreezing"""
        layer_groups = []
        
        # Group by layer depth (simplified)
        layers_by_depth = defaultdict(list)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Estimate depth by counting dots in name
                depth = name.count('.')
                layers_by_depth[depth].append(name)
        
        # Sort by depth and create groups
        max_depth = max(layers_by_depth.keys()) if layers_by_depth else 0
        num_groups = min(len(self.config.unfreeze_schedule), max_depth + 1)
        
        for i in range(num_groups):
            group = []
            for depth in range(i * (max_depth // num_groups), (i + 1) * (max_depth // num_groups)):
                group.extend(layers_by_depth.get(depth, []))
            if group:
                layer_groups.append(group)
        
        return layer_groups
    
    def _freeze_base_model(self):
        """Freeze all base model parameters"""
        for name, param in self.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
    
    def step(self, global_step: int):
        """Update unfreezing based on training progress"""
        if not self.config.progressive_unfreeze:
            return
        
        # Check if should advance to next stage
        if self.current_stage < len(self.config.unfreeze_schedule):
            if global_step >= self.config.unfreeze_schedule[self.current_stage]:
                self._unfreeze_next_group()
                self.current_stage += 1
    
    def _unfreeze_next_group(self):
        """Unfreeze next group of layers"""
        if self.current_stage >= len(self.layer_groups):
            return
        
        group = self.layer_groups[self.current_stage]
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            # Check if parameter belongs to current group
            for layer_name in group:
                if layer_name in name and "lora" not in name.lower():
                    param.requires_grad = True
                    unfrozen_count += 1
        
        logger.info(f"Unfroze {unfrozen_count} parameters in stage {self.current_stage}")

class LoRAAdapter:
    """Main LoRA adapter manager"""
    def __init__(
        self,
        base_model: nn.Module,
        config: LoRAConfig
    ):
        self.base_model = base_model
        self.config = config
        self.lora_layers = OrderedDict()
        
        # Apply LoRA to target modules
        self._apply_lora()
        
        # Setup progressive unfreezing
        self.unfreezer = ProgressiveUnfreezing(base_model, config)
        
        # Task routers if using task adapters
        if config.use_task_adapters:
            self._setup_task_adapters()
    
    def _apply_lora(self):
        """Apply LoRA to target modules"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    # Determine adapter type
                    if any(reasoning in name for reasoning in self.config.reasoning_modules):
                        # Use reasoning adapter
                        lora_layer = ReasoningLoRA(
                            module.in_features,
                            self.config
                        )
                    else:
                        # Use standard LoRA
                        if self.config.use_qlora:
                            lora_layer = QLoRALayer(
                                module.in_features,
                                module.out_features,
                                r=self.config.r,
                                lora_alpha=self.config.lora_alpha,
                                lora_dropout=self.config.lora_dropout
                            )
                            # Quantize base layer
                            quantized = lora_layer.quantize_base_layer(module)
                            self._replace_module(name, quantized)
                        else:
                            lora_layer = LoRALayer(
                                module.in_features,
                                module.out_features,
                                r=self.config.r,
                                lora_alpha=self.config.lora_alpha,
                                lora_dropout=self.config.lora_dropout
                            )
                    
                    self.lora_layers[name] = lora_layer
    
    def _setup_task_adapters(self):
        """Setup task-specific adapters"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.config.task_adapter_layers):
                if isinstance(module, nn.Linear) and name not in self.lora_layers:
                    task_adapter = TaskSpecificLoRA(
                        module.in_features,
                        self.config
                    )
                    self.lora_layers[f"{name}_task"] = task_adapter
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace module in model"""
        parts = name.split('.')
        parent = self.base_model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def forward_with_adapters(
        self,
        x: torch.Tensor,
        layer_outputs: Dict[str, torch.Tensor],
        task_id: Optional[int] = None,
        reasoning_step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply LoRA adapters to layer outputs"""
        adapted_outputs = {}
        
        for name, output in layer_outputs.items():
            if name in self.lora_layers:
                adapter = self.lora_layers[name]
                
                if isinstance(adapter, ReasoningLoRA):
                    adapted = adapter(x, output, reasoning_step)
                elif isinstance(adapter, TaskSpecificLoRA):
                    adapted = adapter(x, output, task_id)
                else:
                    adapted = adapter(x, output)
                
                adapted_outputs[name] = adapted
            else:
                adapted_outputs[name] = output
        
        return adapted_outputs
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable LoRA parameters"""
        params = []
        
        for adapter in self.lora_layers.values():
            params.extend(adapter.parameters())
        
        return params
    
    def save_adapters(self, path: str):
        """Save LoRA adapter weights"""
        adapter_state = {
            name: adapter.state_dict()
            for name, adapter in self.lora_layers.items()
        }
        
        torch.save({
            "adapter_state": adapter_state,
            "config": self.config
        }, path)
        
        logger.info(f"Saved LoRA adapters to {path}")
    
    def load_adapters(self, path: str):
        """Load LoRA adapter weights"""
        checkpoint = torch.load(path, map_location="cpu")
        
        adapter_state = checkpoint["adapter_state"]
        for name, state_dict in adapter_state.items():
            if name in self.lora_layers:
                self.lora_layers[name].load_state_dict(state_dict)
        
        logger.info(f"Loaded LoRA adapters from {path}")
    
    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights into base model and remove adapters"""
        for name, module in self.base_model.named_modules():
            if name in self.lora_layers:
                adapter = self.lora_layers[name]
                
                if isinstance(module, nn.Linear) and isinstance(adapter, LoRALayer):
                    # Merge weights: W' = W + BA * scaling
                    delta_weight = adapter.lora_B @ adapter.lora_A * adapter.scaling
                    module.weight.data += delta_weight
        
        # Remove adapters
        self.lora_layers.clear()
        
        return self.base_model

def create_peft_model(
    base_model: nn.Module,
    config: LoRAConfig,
    adapter_name: str = "default"
) -> Tuple[nn.Module, LoRAAdapter]:
    """Create model with LoRA adapters"""
    # Create adapter
    adapter = LoRAAdapter(base_model, config)
    
    # Modify forward pass to use adapters
    original_forward = base_model.forward
    
    def forward_with_lora(self, *args, **kwargs):
        # Get intermediate outputs
        outputs = original_forward(*args, **kwargs)
        
        # Apply adapters if layer outputs are available
        if hasattr(outputs, "layer_outputs"):
            outputs.layer_outputs = adapter.forward_with_adapters(
                args[0] if args else kwargs.get("input_ids"),
                outputs.layer_outputs,
                task_id=kwargs.get("task_id"),
                reasoning_step=kwargs.get("reasoning_step")
            )
        
        return outputs
    
    # Monkey-patch forward method
    import types
    base_model.forward = types.MethodType(forward_with_lora, base_model)
    
    return base_model, adapter