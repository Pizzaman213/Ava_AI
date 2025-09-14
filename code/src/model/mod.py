"""
Mixture of Depths (MoD) Implementation
Dynamic computation allocation based on token complexity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

class DepthRouter(nn.Module):
    """
    Router for Mixture of Depths - decides which tokens to process at each layer
    """
    def __init__(
        self,
        hidden_size: int,
        mode: str = "learned",  # learned, fixed, adaptive
        temperature: float = 1.0,
        min_depth: float = 0.1,
        max_depth: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        self.temperature = temperature
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        if mode == "learned":
            # Learned routing network
            self.depth_predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
            
            # Initialize for balanced routing
            nn.init.constant_(self.depth_predictor[-2].bias, 0.0)
        
        elif mode == "adaptive":
            # Adaptive routing based on token statistics
            self.register_buffer("running_mean", torch.zeros(hidden_size))
            self.register_buffer("running_var", torch.ones(hidden_size))
            self.momentum = 0.9

    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute depth scores for each token
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            layer_idx: Current layer index (for fixed mode)
            
        Returns:
            depth_scores: Scores indicating processing priority [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        if self.mode == "learned":
            # Learned routing
            depth_scores = self.depth_predictor(x).squeeze(-1)
            
        elif self.mode == "fixed":
            # Fixed pattern (e.g., process every other token at deeper layers)
            if layer_idx is None:
                depth_scores = torch.ones(batch_size, seq_len, device=x.device)
            else:
                # Decrease processing probability at deeper layers
                base_prob = 1.0 - (layer_idx * 0.03)  # Reduce by 3% per layer
                depth_scores = torch.full((batch_size, seq_len), base_prob, device=x.device)
                
                # Add some randomness
                if self.training:
                    noise = torch.randn_like(depth_scores) * 0.1
                    depth_scores = depth_scores + noise
                    depth_scores = torch.sigmoid(depth_scores)
        
        elif self.mode == "adaptive":
            # Adaptive routing based on token complexity
            # Update running statistics
            if self.training:
                with torch.no_grad():
                    mean = x.mean(dim=[0, 1])
                    var = x.var(dim=[0, 1])
                    self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                    self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Compute token complexity (deviation from mean)
            normalized = (x - self.running_mean) / (self.running_var.sqrt() + 1e-5)
            complexity = normalized.abs().mean(dim=-1)
            
            # Convert complexity to depth scores
            depth_scores = torch.sigmoid(complexity / self.temperature)
        
        else:
            raise ValueError(f"Unknown routing mode: {self.mode}")
        
        # Clamp scores to valid range
        depth_scores = torch.clamp(depth_scores, self.min_depth, self.max_depth)
        
        return depth_scores

class MixtureOfDepths(nn.Module):
    """
    Mixture of Depths layer - dynamically allocates computation to tokens
    """
    def __init__(
        self,
        hidden_size: int,
        capacity_factor: float = 0.8,
        skip_fraction: float = 0.2,
        aux_loss_weight: float = 0.01,
        straight_through: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        self.skip_fraction = skip_fraction
        self.aux_loss_weight = aux_loss_weight
        self.straight_through = straight_through

    def forward(
        self,
        x: torch.Tensor,
        depth_scores: torch.Tensor,
        force_process: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixture of Depths routing
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            depth_scores: Routing scores from DepthRouter [batch_size, seq_len]
            force_process: Optional mask to force processing certain tokens
            
        Returns:
            routing_mask: Binary mask indicating which tokens to process [batch_size, seq_len]
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len = depth_scores.shape
        
        # Compute capacity (number of tokens to process)
        total_tokens = batch_size * seq_len
        capacity = int(total_tokens * self.capacity_factor)
        skip_tokens = int(total_tokens * self.skip_fraction)
        process_tokens = capacity - skip_tokens
        
        # Get top-k tokens to process based on depth scores
        flat_scores = depth_scores.view(-1)
        
        if force_process is not None:
            # Ensure forced tokens have high scores
            force_mask_flat = force_process.view(-1)
            flat_scores = torch.where(force_mask_flat, torch.ones_like(flat_scores), flat_scores)
        
        # Select top tokens
        _, indices = torch.topk(flat_scores, k=min(process_tokens, total_tokens))
        
        # Create routing mask
        routing_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        routing_mask[indices] = True
        routing_mask = routing_mask.view(batch_size, seq_len)
        
        # Apply straight-through estimator for gradients
        if self.straight_through and self.training:
            # Use continuous scores in forward, binary mask in backward
            routing_mask_float = routing_mask.float()
            routing_mask_float = depth_scores + (routing_mask_float - depth_scores).detach()
            routing_mask = routing_mask_float > 0.5
        
        # Compute auxiliary loss (encourage balanced routing)
        actual_capacity = routing_mask.float().mean()
        target_capacity = self.capacity_factor
        aux_loss = self.aux_loss_weight * (actual_capacity - target_capacity) ** 2
        
        return routing_mask, aux_loss

class AdaptiveMoD(nn.Module):
    """
    Adaptive Mixture of Depths with learned capacity allocation
    """
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        base_capacity: float = 0.8,
        min_capacity: float = 0.5,
        max_capacity: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.base_capacity = base_capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        
        # Learned capacity predictor
        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_layers),
            nn.Sigmoid()
        )
        
        # Layer-specific MoD modules
        self.mod_layers = nn.ModuleList([
            MixtureOfDepths(
                hidden_size,
                capacity_factor=base_capacity,
                skip_fraction=0.1 + 0.01 * i,  # Increase skip fraction at deeper layers
            )
            for i in range(num_layers)
        ])

    def get_layer_capacity(self, x: torch.Tensor, layer_idx: int) -> float:
        """
        Compute dynamic capacity for a specific layer
        """
        # Global features for capacity prediction
        global_features = x.mean(dim=[0, 1])  # [hidden_size]
        
        # Predict capacity for all layers
        capacities = self.capacity_predictor(global_features)  # [num_layers]
        
        # Scale to valid range
        capacity = capacities[layer_idx].item()
        capacity = self.min_capacity + capacity * (self.max_capacity - self.min_capacity)
        
        return capacity

    def forward(
        self,
        x: torch.Tensor,
        depth_scores: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive MoD for a specific layer
        """
        # Get dynamic capacity
        if self.training:
            capacity = self.get_layer_capacity(x, layer_idx)
            self.mod_layers[layer_idx].capacity_factor = capacity
        
        # Apply MoD
        return self.mod_layers[layer_idx](x, depth_scores)

class HierarchicalMoD(nn.Module):
    """
    Hierarchical Mixture of Depths with multi-scale routing
    """
    def __init__(
        self,
        hidden_size: int,
        num_levels: int = 3,
        base_capacity: float = 0.8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.base_capacity = base_capacity
        
        # Multi-scale routers
        self.routers = nn.ModuleList([
            DepthRouter(hidden_size, mode="learned")
            for _ in range(num_levels)
        ])
        
        # Level-specific capacities
        self.level_capacities = [
            base_capacity * (0.5 + 0.5 * i / num_levels)
            for i in range(num_levels)
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical routing with multi-scale decisions
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize routing mask
        routing_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        total_aux_loss = 0.0
        
        # Apply routing at each level
        for level, (router, capacity) in enumerate(zip(self.routers, self.level_capacities)):
            # Get routing scores for current level
            depth_scores = router(x)
            
            # Apply routing only to currently active tokens
            active_mask = routing_mask.clone()
            num_active = active_mask.sum()
            
            if num_active > 0:
                # Compute how many tokens to keep at this level
                keep_tokens = int(num_active * capacity)
                
                # Get scores for active tokens
                active_scores = depth_scores[active_mask]
                
                # Select top tokens
                if keep_tokens < num_active:
                    _, keep_indices = torch.topk(active_scores, k=keep_tokens)
                    
                    # Update routing mask
                    active_indices = torch.where(active_mask)[0]
                    keep_mask = torch.zeros_like(active_mask)
                    keep_mask[active_indices[keep_indices]] = True
                    routing_mask = routing_mask & keep_mask
                
                # Compute auxiliary loss
                actual_kept = routing_mask.sum() / (batch_size * seq_len)
                target_kept = capacity ** (level + 1)  # Exponential decay
                aux_loss = 0.01 * (actual_kept - target_kept) ** 2
                total_aux_loss += aux_loss
        
        return routing_mask, total_aux_loss

class TokenImportanceScorer(nn.Module):
    """
    Score token importance for MoD routing decisions
    """
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head importance scoring
        self.importance_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1)
            for _ in range(num_heads)
        ])
        
        # Attention-based importance
        self.query_proj = nn.Linear(hidden_size, hidden_size // 4)
        self.key_proj = nn.Linear(hidden_size, hidden_size // 4)
        
        # Feature extractors
        self.norm_score = nn.LayerNorm(hidden_size)
        self.gradient_estimator = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, return_all_scores: bool = False) -> torch.Tensor:
        """
        Compute token importance scores
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            return_all_scores: Whether to return individual score components
            
        Returns:
            importance_scores: Combined importance scores [batch_size, seq_len]
            all_scores: Dict of individual scores (if return_all_scores=True)
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x_norm = self.norm_score(x)
        
        # Multi-head importance scores
        head_scores = []
        for head in self.importance_heads:
            score = head(x_norm).squeeze(-1)  # [batch_size, seq_len]
            head_scores.append(score)
        
        head_scores = torch.stack(head_scores, dim=-1)  # [batch_size, seq_len, num_heads]
        avg_head_score = head_scores.mean(dim=-1)
        
        # Attention-based importance (how much other tokens attend to this token)
        queries = self.query_proj(x_norm)  # [batch_size, seq_len, hidden_size//4]
        keys = self.key_proj(x_norm)
        
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        attn_scores = F.softmax(attn_scores / math.sqrt(queries.size(-1)), dim=-1)
        
        # Importance = how much other tokens attend to this token
        attn_importance = attn_scores.sum(dim=1)  # [batch_size, seq_len]
        
        # Gradient-based importance estimation
        grad_importance = torch.sigmoid(self.gradient_estimator(x_norm)).squeeze(-1)
        
        # Combine scores
        importance_scores = (avg_head_score + attn_importance + grad_importance) / 3.0
        
        if return_all_scores:
            all_scores = {
                "head_scores": head_scores,
                "attention_importance": attn_importance,
                "gradient_importance": grad_importance,
            }
            return importance_scores, all_scores
        
        return importance_scores