"""
Hierarchical Expert Layers with Advanced Routing
Implements Switch Transformer, GLaM-style routing, and custom hierarchical routing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w3 = nn.Linear(dim_out, dim_in, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Expert(nn.Module):
    """Single Expert with SwiGLU activation"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.net = SwiGLU(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.net(x))

class ExpertRouter(nn.Module):
    """Advanced router for expert selection with load balancing"""
    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        routing_type: str = "switch",  # switch, gshard, base
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        dropout: float = 0.1,  # Default router dropout for regularization
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routing_type = routing_type
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        
        # Router network
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize gate weights with smaller values for stability
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts
        
        Returns:
            router_weights: Softmax router weights [batch_size, seq_len, num_experts]
            selected_experts: Selected expert indices [batch_size, seq_len, num_experts_per_tok]
            aux_loss: Load balancing auxiliary loss
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply dropout to input before routing
        x_dropout = self.dropout(x)
        
        # Compute router scores
        router_logits = self.gate(x_dropout)  # [batch_size, seq_len, num_experts]
        
        # Add noise for exploration during training
        if self.training and self.routing_type == "gshard":
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise
        
        # Apply temperature scaling for better exploration
        temperature = 1.0 if not self.training else 0.8  # Slightly sharper during training
        router_logits = router_logits / temperature
        
        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        if self.num_experts_per_tok == 1:
            # Switch routing (top-1)
            router_weights, selected_experts = router_probs.max(dim=-1)
            router_weights = router_weights.unsqueeze(-1)  # Add dimension for consistency
            selected_experts = selected_experts.unsqueeze(-1)
            # Don't normalize for single expert - weight indicates confidence
        else:
            # Top-k routing
            router_weights, selected_experts = torch.topk(
                router_probs, self.num_experts_per_tok, dim=-1
            )
            # Normalize weights
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary losses
        aux_loss = self._compute_aux_loss(router_probs, selected_experts)
        z_loss = self._compute_z_loss(router_logits)
        
        total_aux_loss = self.aux_loss_coef * aux_loss + self.z_loss_coef * z_loss
        
        return router_weights, selected_experts, total_aux_loss

    def _compute_aux_loss(self, router_probs: torch.Tensor, selected_experts: torch.Tensor, token_importance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute load balancing auxiliary loss with optional importance weighting"""
        # Compute expert load (fraction of tokens routed to each expert)
        num_tokens = router_probs.shape[0] * router_probs.shape[1]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        
        # Apply token importance if provided
        if token_importance is not None:
            # Reshape importance to match expert_mask dimensions
            if expert_mask.dim() == 4:
                importance = token_importance.view(-1, 1, 1, 1).expand_as(expert_mask)
            else:
                importance = token_importance.view(-1, 1).expand_as(expert_mask)
            expert_mask = expert_mask * importance
        
        # Handle both 3D and 4D tensors
        if expert_mask.dim() == 4:
            expert_load = expert_mask.sum(dim=[0, 1, 2])  # Count per expert
        else:
            expert_load = expert_mask.sum(dim=[0, 1])  # Count per expert
        
        # Normalize by total importance or token count
        if token_importance is not None:
            total_importance = token_importance.sum()
            expert_load = expert_load / total_importance.clamp(min=1e-8)
        else:
            expert_load = expert_load.float() / num_tokens
        
        # Compute ideal uniform distribution
        ideal_load = 1.0 / self.num_experts
        
        # Use smooth L1 loss for robustness
        load_balance_loss = F.smooth_l1_loss(
            expert_load, 
            torch.full_like(expert_load, ideal_load),
            reduction='sum'
        )
        
        # Add entropy regularization to encourage exploration
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1).mean()
        
        return load_balance_loss - 0.01 * entropy

    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Router z-loss for numerical stability"""
        z_loss = torch.logsumexp(router_logits, dim=-1).mean()
        return z_loss

class HierarchicalRouter(nn.Module):
    """Hierarchical router with coarse-to-fine expert selection"""
    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_groups: int = 4,
        num_experts_per_tok: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_groups = num_groups
        self.num_experts_per_tok = num_experts_per_tok
        self.temperature = temperature
        self.experts_per_group = num_experts // num_groups
        
        # Coarse router (select expert groups)
        self.coarse_router = nn.Linear(dim, num_groups, bias=False)
        
        # Fine routers (one per group)
        self.fine_routers = nn.ModuleList([
            nn.Linear(dim, self.experts_per_group, bias=False)
            for _ in range(num_groups)
        ])
        
        # Initialize weights
        for router in [self.coarse_router] + list(self.fine_routers):
            nn.init.kaiming_uniform_(router.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchical routing: first select groups, then experts within groups
        """
        batch_size, seq_len, _ = x.shape
        
        # Coarse routing - select top groups
        coarse_logits = self.coarse_router(x) / self.temperature
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        
        # Select top-2 groups
        top_groups_weights, top_groups = torch.topk(coarse_probs, k=2, dim=-1)
        
        # Fine routing within selected groups
        all_expert_weights = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        all_expert_indices = []
        
        for i in range(2):  # For each of top-2 groups
            group_idx = top_groups[:, :, i]  # [batch_size, seq_len]
            group_weight = top_groups_weights[:, :, i]  # [batch_size, seq_len]
            
            # Compute fine routing for all tokens
            fine_logits_list = []
            for g in range(self.num_groups):
                mask = (group_idx == g).unsqueeze(-1)
                fine_logits = self.fine_routers[g](x) / self.temperature
                fine_logits_list.append(fine_logits * mask.float())
            
            # Combine fine logits
            fine_logits = torch.stack(fine_logits_list, dim=2)  # [batch, seq, groups, experts_per_group]
            fine_logits = fine_logits.view(batch_size, seq_len, -1)
            
            # Select top expert within group
            fine_probs = F.softmax(fine_logits, dim=-1)
            # Handle empty sequences (all masked by MoD)
            if fine_probs.shape[-1] == 0:
                expert_weight = torch.zeros(batch_size, seq_len, device=x.device)
                expert_idx = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
            else:
                expert_weight, expert_idx = fine_probs.max(dim=-1)
            
            # Combine coarse and fine weights
            combined_weight = group_weight * expert_weight
            
            # Update global expert weights
            all_expert_weights.scatter_add_(
                2, expert_idx.unsqueeze(-1), combined_weight.unsqueeze(-1)
            )
            all_expert_indices.append(expert_idx)
        
        # Stack expert indices
        selected_experts = torch.stack(all_expert_indices, dim=-1)
        
        # Normalize weights
        router_weights = all_expert_weights.gather(2, selected_experts)
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Compute auxiliary loss
        aux_loss = self._compute_load_balance_loss(all_expert_weights)
        
        return router_weights, selected_experts, aux_loss

    def _compute_load_balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss for hierarchical routing"""
        # Average load per expert
        expert_load = expert_weights.mean(dim=[0, 1])
        ideal_load = 1.0 / self.num_experts
        
        # Variance from ideal load
        load_balance_loss = torch.sum((expert_load - ideal_load) ** 2)
        
        return load_balance_loss

class HierarchicalExpertLayer(nn.Module):
    """Hierarchical Mixture of Experts layer with advanced routing"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_capacity_factor = config.expert_capacity_factor
        self.use_parallel_experts = getattr(config, 'use_parallel_experts', True)  # Enable parallel processing by default
        self.expert_dropout = getattr(config, 'expert_dropout', 0.1)  # Expert dropout for regularization
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(
                self.hidden_size,
                self.intermediate_size,
                dropout=getattr(config, 'hidden_dropout', 0.1),  # Use hidden dropout from config
                bias=getattr(config, 'mlp_bias', False)
            )
            for _ in range(self.num_experts)
        ])
        
        # Create router
        expert_type = getattr(config, 'expert_type', 'standard')
        if expert_type == "hierarchical":
            self.router = HierarchicalRouter(
                self.hidden_size,
                self.num_experts,
                num_groups=4,
                num_experts_per_tok=self.num_experts_per_tok,
            )
        else:
            self.router = ExpertRouter(
                self.hidden_size,
                self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                routing_type=config.expert_routing_type,
                aux_loss_coef=getattr(config, 'aux_loss_coef', 0.01),
                z_loss_coef=getattr(config, 'router_z_loss_coef', 0.001),
                dropout=getattr(config, 'attention_dropout', 0.1),  # Add router dropout
            )

    def forward(self, x: torch.Tensor, mod_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            mod_mask: Mixture of Depths mask (optional)
            
        Returns:
            output: Output tensor [batch_size, seq_len, hidden_size]
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply MoD mask if provided
        if mod_mask is not None:
            active_tokens = x[mod_mask]
            if active_tokens.shape[0] == 0:
                return torch.zeros_like(x), torch.tensor(0.0, device=x.device)
        else:
            active_tokens = x.view(-1, hidden_size)
        
        # Apply expert dropout during training (at token-expert level, not expert level)
        if self.training and self.expert_dropout > 0:
            # Create dropout mask per token-expert pair for better regularization
            expert_dropout_mask = torch.rand(active_tokens.shape[0], self.num_experts, device=x.device) > self.expert_dropout
        else:
            expert_dropout_mask = None
        
        # Route tokens to experts
        if mod_mask is not None:
            # Route only active tokens
            router_input = active_tokens.view(-1, 1, hidden_size)
            router_weights, selected_experts, aux_loss = self.router(router_input)
            router_weights = router_weights.squeeze(1)
            selected_experts = selected_experts.squeeze(1)
        else:
            # Route all tokens
            router_weights, selected_experts, aux_loss = self.router(x)
            # Flatten for expert computation
            router_weights = router_weights.view(-1, router_weights.shape[-1])
            selected_experts = selected_experts.view(-1, selected_experts.shape[-1])
        
        # Expert computation with capacity constraints
        output = self._compute_experts(active_tokens, router_weights, selected_experts, expert_dropout_mask)
        
        # Reshape output
        if mod_mask is not None:
            final_output = torch.zeros_like(x)
            final_output[mod_mask] = output
            output = final_output
        else:
            output = output.view(batch_size, seq_len, hidden_size)
        
        return output, aux_loss

    def _compute_experts(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_dropout_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute expert outputs with capacity constraints - using parallel processing"""
        # Use new parallel implementation if available
        if hasattr(self, 'use_parallel_experts') and self.use_parallel_experts:
            return self._compute_experts_parallel(x, router_weights, selected_experts, expert_dropout_mask)
        
        # Fall back to sequential processing
        num_tokens = x.shape[0]
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Skip if expert is dropped out (check all tokens for this expert)
            if expert_dropout_mask is not None and not expert_dropout_mask[:, expert_idx].any():
                continue
                
            # Find tokens routed to this expert
            if selected_experts.dim() == 2:
                # 2D: [num_tokens, num_experts_per_tok]
                expert_mask = (selected_experts == expert_idx).any(dim=-1)
            else:
                # 1D: [num_tokens] (shouldn't happen with current code)
                expert_mask = (selected_experts == expert_idx)
            
            if not expert_mask.any():
                continue
            
            # Apply capacity constraint
            expert_capacity = int(self.expert_capacity_factor * num_tokens / self.num_experts)
            num_tokens_to_expert = expert_mask.sum().item()
            
            if num_tokens_to_expert > expert_capacity:
                # Randomly drop tokens exceeding capacity
                indices = torch.where(expert_mask)[0]
                perm = torch.randperm(len(indices))[:expert_capacity]
                expert_mask = torch.zeros_like(expert_mask)
                expert_mask[indices[perm]] = True
            
            # Compute expert output
            expert_input = x[expert_mask]
            if expert_input.shape[0] > 0:
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for this expert's tokens
                expert_token_indices = expert_mask.nonzero().squeeze(-1)
                
                if self.num_experts_per_tok == 1:
                    # For single expert: weights are just the router weights
                    weights = router_weights[expert_token_indices, 0]
                else:
                    # For multiple experts: find which position this expert was selected
                    # and use the corresponding weight
                    selected_positions = (selected_experts[expert_token_indices] == expert_idx)
                    weights = (router_weights[expert_token_indices] * selected_positions.float()).sum(dim=-1)
                
                # Ensure weights have correct shape
                if weights.dim() == 0:
                    weights = weights.unsqueeze(0)
                
                # Accumulate weighted expert outputs
                weighted_output = expert_output * weights.unsqueeze(-1)
                output[expert_mask] += weighted_output
        
        return output

    def _compute_experts_parallel(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_dropout_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Parallel expert computation for improved performance"""
        num_tokens, hidden_size = x.shape
        
        # Group tokens by expert for batch processing
        expert_batches = [[] for _ in range(self.num_experts)]
        expert_batch_indices = [[] for _ in range(self.num_experts)]
        expert_batch_weights = [[] for _ in range(self.num_experts)]
        
        # Build batches for each expert
        for token_idx in range(num_tokens):
            if selected_experts.dim() == 2:
                # Multiple experts per token
                for k, expert_idx in enumerate(selected_experts[token_idx]):
                    expert_idx_val = int(expert_idx.item())
                    expert_batches[expert_idx_val].append(x[token_idx])
                    expert_batch_indices[expert_idx_val].append(token_idx)
                    expert_batch_weights[expert_idx_val].append(router_weights[token_idx, k])
            else:
                # Single expert per token
                expert_idx_val = int(selected_experts[token_idx].item())
                expert_batches[expert_idx_val].append(x[token_idx])
                expert_batch_indices[expert_idx_val].append(token_idx)
                expert_batch_weights[expert_idx_val].append(router_weights[token_idx])
        
        # Apply capacity constraints
        expert_capacity = int(self.expert_capacity_factor * num_tokens / self.num_experts)
        for expert_idx in range(self.num_experts):
            if len(expert_batches[expert_idx]) > expert_capacity:
                # Randomly select tokens up to capacity
                indices = torch.randperm(len(expert_batches[expert_idx]))[:expert_capacity]
                expert_batches[expert_idx] = [expert_batches[expert_idx][i] for i in indices]
                expert_batch_indices[expert_idx] = [expert_batch_indices[expert_idx][i] for i in indices]
                expert_batch_weights[expert_idx] = [expert_batch_weights[expert_idx][i] for i in indices]
        
        # Process all experts in parallel (future: can use torch.multiprocessing)
        output = torch.zeros_like(x)
        expert_outputs = []
        
        # Batch process for each expert
        for expert_idx, expert in enumerate(self.experts):
            # Skip if expert is dropped out (check all tokens for this expert)
            if expert_dropout_mask is not None and not expert_dropout_mask[:, expert_idx].any():
                continue
                
            if expert_batches[expert_idx]:
                # Stack batch for this expert
                batch_tensor = torch.stack(expert_batches[expert_idx])
                batch_weights = torch.stack(expert_batch_weights[expert_idx])
                
                # Apply additional noise during training for exploration
                if self.training and hasattr(self.router, 'routing_type') and self.router.routing_type == "gshard":
                    noise = torch.randn_like(batch_weights) * 0.01
                    batch_weights = batch_weights + noise
                    batch_weights = batch_weights / batch_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                
                # Process batch through expert
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.training):
                    expert_output = expert(batch_tensor)
                
                # Store results
                expert_outputs.append((
                    expert_idx,
                    expert_output,
                    expert_batch_indices[expert_idx],
                    batch_weights
                ))
        
        # Combine outputs from all experts
        for expert_idx, expert_out, indices, weights in expert_outputs:
            for i, token_idx in enumerate(indices):
                output[token_idx] += expert_out[i] * weights[i].unsqueeze(-1)
        
        return output

class ExpertParallelWrapper(nn.Module):
    """Wrapper for expert parallel execution across devices"""
    def __init__(self, expert_layer: HierarchicalExpertLayer, device_ids: Optional[list] = None):
        super().__init__()
        self.expert_layer = expert_layer
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.num_devices = len(self.device_ids)
        
        if self.num_devices > 1:
            # Distribute experts across devices
            experts_per_device = self.expert_layer.num_experts // self.num_devices
            for i, device_id in enumerate(self.device_ids):
                start_idx = i * experts_per_device
                end_idx = (i + 1) * experts_per_device if i < self.num_devices - 1 else self.expert_layer.num_experts
                
                for j in range(start_idx, end_idx):
                    self.expert_layer.experts[j] = self.expert_layer.experts[j].to(f'cuda:{device_id}')

    def forward(self, x: torch.Tensor, mod_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with expert parallelism"""
        if self.num_devices == 1:
            return self.expert_layer(x, mod_mask)
        
        # Multi-device expert parallel execution
        # This is a simplified version - full implementation would use
        # more sophisticated communication patterns
        return self.expert_layer(x, mod_mask)


class AdaptiveCapacityRouter(nn.Module):
    """Router with adaptive capacity based on actual usage"""
    def __init__(self, dim: int, num_experts: int, num_experts_per_tok: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        # Main router
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Capacity predictor
        self.capacity_predictor = nn.Linear(dim, num_experts)
        
        # EMA tracking of expert usage
        self.register_buffer('ema_usage', torch.zeros(num_experts))
        self.register_buffer('ema_capacity', torch.ones(num_experts))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.capacity_predictor.weight, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with adaptive capacity
        
        Returns:
            router_weights: Softmax router weights
            selected_experts: Selected expert indices
            aux_loss: Load balancing auxiliary loss
            adaptive_capacity: Predicted capacity per expert
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute router scores
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Predict required capacity based on input
        capacity_logits = self.capacity_predictor(x.mean(dim=1))  # [batch_size, num_experts]
        capacity = F.softmax(capacity_logits, dim=-1).mean(dim=0)  # Average over batch
        
        # Update EMA of usage
        if self.training:
            current_usage = router_probs.mean(dim=[0, 1])
            self.ema_usage = 0.9 * self.ema_usage + 0.1 * current_usage
            
            # Adjust capacity based on historical usage
            # If an expert is underused, increase its capacity
            # If overused, slightly decrease to encourage load balancing
            usage_ratio = current_usage / (self.ema_usage + 1e-6)
            adjusted_capacity = capacity * (2.0 - usage_ratio.clamp(0.5, 1.5))
            self.ema_capacity = 0.9 * self.ema_capacity + 0.1 * adjusted_capacity
        else:
            adjusted_capacity = self.ema_capacity
        
        # Select experts
        router_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize weights
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary loss
        aux_loss = self._compute_adaptive_aux_loss(router_probs, adjusted_capacity)
        
        return router_weights, selected_experts, aux_loss, adjusted_capacity
    
    def _compute_adaptive_aux_loss(self, router_probs: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss with adaptive targets"""
        # Expert load
        expert_load = router_probs.mean(dim=[0, 1])
        
        # Target load based on adaptive capacity
        target_load = capacity / capacity.sum()
        
        # Adaptive loss - encourage matching the predicted capacity
        aux_loss = F.smooth_l1_loss(expert_load, target_load, reduction='sum')
        
        # Add entropy for exploration
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1).mean()
        
        return aux_loss - 0.01 * entropy


class MemoryEfficientMoELayer(HierarchicalExpertLayer):
    """Memory-efficient MoE layer with gradient checkpointing and activation pooling"""
    def __init__(self, config):
        super().__init__(config)
        self.use_checkpoint = getattr(config, 'gradient_checkpointing', True)
        self.checkpoint_experts = getattr(config, 'checkpoint_experts', True)
        
        # Activation pooling for memory reuse
        self.activation_pool_size = getattr(config, 'activation_pool_size', 4)
        self.activation_pool = []
        
    def forward(self, x: torch.Tensor, mod_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional gradient checkpointing"""
        if self.use_checkpoint and self.training:
            # Checkpoint the entire MoE computation
            return checkpoint(self._forward_impl, x, mod_mask, use_reentrant=False)
        return self._forward_impl(x, mod_mask)
    
    def _forward_impl(self, x: torch.Tensor, mod_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actual forward implementation"""
        batch_size, seq_len, hidden_size = x.shape
        
        # Try to reuse activation buffer from pool
        if self.activation_pool and x.shape == self.activation_pool[0].shape:
            active_buffer = self.activation_pool.pop(0)
            active_buffer.copy_(x.view(-1, hidden_size))
            active_tokens = active_buffer
        else:
            active_tokens = x.view(-1, hidden_size)
        
        # Apply MoD mask if provided
        if mod_mask is not None:
            active_tokens = x[mod_mask]
            if active_tokens.shape[0] == 0:
                return torch.zeros_like(x), torch.tensor(0.0, device=x.device)
        
        # Apply expert dropout during training
        if self.training and self.expert_dropout > 0:
            expert_dropout_mask = torch.rand(self.num_experts, device=x.device) > self.expert_dropout
        else:
            expert_dropout_mask = None
        
        # Route tokens to experts
        router_weights, selected_experts, aux_loss = self.router(active_tokens.unsqueeze(1) if mod_mask else x)
        
        if mod_mask:
            router_weights = router_weights.squeeze(1)
            selected_experts = selected_experts.squeeze(1)
        else:
            router_weights = router_weights.view(-1, router_weights.shape[-1])
            selected_experts = selected_experts.view(-1, selected_experts.shape[-1])
        
        # Expert computation with optional checkpointing
        if self.checkpoint_experts and self.training:
            output = checkpoint(
                self._compute_experts,
                active_tokens,
                router_weights,
                selected_experts,
                expert_dropout_mask,
                use_reentrant=False
            )
        else:
            output = self._compute_experts(active_tokens, router_weights, selected_experts, expert_dropout_mask)
        
        # Reshape output
        if mod_mask is not None:
            final_output = torch.zeros_like(x)
            final_output[mod_mask] = output
            output = final_output
        else:
            output = output.view(batch_size, seq_len, hidden_size)
        
        # Add activation buffer back to pool for reuse
        if len(self.activation_pool) < self.activation_pool_size and not mod_mask:
            self.activation_pool.append(active_tokens.detach())
        
        return output, aux_loss