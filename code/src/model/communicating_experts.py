"""
Communicating Experts for MoE++ Models

Implements inter-expert communication mechanisms allowing experts
to share information during forward passes for better coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)


@dataclass
class CommunicationConfig:
    """Configuration for expert communication"""
    # Communication patterns
    communication_type: str = "attention"  # attention, message_passing, graph
    communication_rounds: int = 2  # Number of communication rounds
    
    # Attention-based communication
    num_comm_heads: int = 8
    comm_hidden_dim: int = 256
    comm_dropout: float = 0.1
    
    # Message passing
    message_dim: int = 128
    aggregation_method: str = "mean"  # mean, max, sum, attention
    
    # Graph-based communication
    use_learned_topology: bool = True
    sparsity_ratio: float = 0.3  # For sparse communication graphs
    
    # Efficiency settings
    use_sparse_communication: bool = True
    top_k_experts: int = 4  # Communicate only with top-k experts
    
    # Residual connections
    communication_residual: bool = True
    residual_weight: float = 0.7


class ExpertCommunicator(nn.Module):
    """
    Base class for expert communication mechanisms
    """
    
    def __init__(self, config: CommunicationConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        routing_weights: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Communicate between experts"""
        raise NotImplementedError


class AttentionCommunicator(ExpertCommunicator):
    """
    Attention-based communication between experts
    """
    
    def __init__(self, config: CommunicationConfig, hidden_size: int):
        super().__init__(config, hidden_size)
        
        # Multi-head attention for inter-expert communication
        self.cross_expert_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=config.num_comm_heads,
            dropout=config.comm_dropout,
            batch_first=True
        )
        
        # Projection layers
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        routing_weights: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Perform attention-based communication
        
        Args:
            expert_outputs: List of [batch, seq_len, hidden] tensors
            routing_weights: [batch, seq_len, num_experts] routing probabilities
        """
        num_experts = len(expert_outputs)
        batch_size, seq_len, hidden_size = expert_outputs[0].shape
        
        # Stack expert outputs: [batch, num_experts, seq_len, hidden]
        stacked_outputs = torch.stack(expert_outputs, dim=1)
        
        # Prepare for attention: [batch * seq_len, num_experts, hidden]
        stacked_outputs = rearrange(
            stacked_outputs,
            'b e s h -> (b s) e h'
        )
        
        # Apply communication rounds
        communicated = stacked_outputs
        for _ in range(self.config.communication_rounds):
            # Project to Q, K, V
            queries = self.query_proj(communicated)
            keys = self.key_proj(communicated)
            values = self.value_proj(communicated)
            
            # Apply attention
            attn_output, _ = self.cross_expert_attention(
                queries, keys, values
            )
            
            # Residual connection
            if self.config.communication_residual:
                communicated = (
                    self.config.residual_weight * communicated +
                    (1 - self.config.residual_weight) * attn_output
                )
            else:
                communicated = attn_output
                
            # Layer norm
            communicated = self.layer_norm(communicated)
            
        # Project back
        communicated = self.output_proj(communicated)
        
        # Reshape back to list of expert outputs
        communicated = rearrange(
            communicated,
            '(b s) e h -> b e s h',
            b=batch_size,
            s=seq_len
        )
        
        # Convert back to list
        updated_outputs = [
            communicated[:, i] for i in range(num_experts)
        ]
        
        return updated_outputs


class MessagePassingCommunicator(ExpertCommunicator):
    """
    Message passing between experts
    """
    
    def __init__(self, config: CommunicationConfig, hidden_size: int):
        super().__init__(config, hidden_size)
        
        # Message generation
        self.message_generator = nn.Sequential(
            nn.Linear(hidden_size, config.message_dim),
            nn.ReLU(),
            nn.Linear(config.message_dim, config.message_dim)
        )
        
        # Message aggregation
        if config.aggregation_method == "attention":
            self.message_attention = nn.MultiheadAttention(
                config.message_dim,
                num_heads=4,
                batch_first=True
            )
            
        # Message incorporation
        self.message_incorporator = nn.Sequential(
            nn.Linear(hidden_size + config.message_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        routing_weights: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Perform message passing communication
        """
        num_experts = len(expert_outputs)
        
        updated_outputs = []
        
        for round_idx in range(self.config.communication_rounds):
            # Generate messages from each expert
            messages = []
            for expert_output in expert_outputs:
                message = self.message_generator(expert_output)
                messages.append(message)
                
            # Aggregate messages for each expert
            aggregated_messages = []
            for i in range(num_experts):
                # Collect messages from other experts
                other_messages = [
                    messages[j] for j in range(num_experts) if j != i
                ]
                
                if not other_messages:
                    # No other experts
                    aggregated_messages.append(
                        torch.zeros_like(messages[0])
                    )
                    continue
                    
                # Stack messages
                stacked_messages = torch.stack(other_messages, dim=1)
                
                # Aggregate
                if self.config.aggregation_method == "mean":
                    aggregated = stacked_messages.mean(dim=1)
                elif self.config.aggregation_method == "max":
                    aggregated = stacked_messages.max(dim=1)[0]
                elif self.config.aggregation_method == "sum":
                    aggregated = stacked_messages.sum(dim=1)
                elif self.config.aggregation_method == "attention":
                    # Use attention to aggregate
                    query = messages[i].unsqueeze(1)  # [batch, 1, seq_len, msg_dim]
                    aggregated, _ = self.message_attention(
                        query,
                        stacked_messages,
                        stacked_messages
                    )
                    aggregated = aggregated.squeeze(1)
                    
                aggregated_messages.append(aggregated)
                
            # Incorporate messages into expert outputs
            new_outputs = []
            for i, (expert_output, agg_message) in enumerate(
                zip(expert_outputs, aggregated_messages)
            ):
                # Concatenate and process
                combined = torch.cat([expert_output, agg_message], dim=-1)
                updated = self.message_incorporator(combined)
                
                # Residual connection
                if self.config.communication_residual:
                    updated = (
                        self.config.residual_weight * expert_output +
                        (1 - self.config.residual_weight) * updated
                    )
                    
                new_outputs.append(self.layer_norm(updated))
                
            expert_outputs = new_outputs
            
        return expert_outputs


class GraphCommunicator(ExpertCommunicator):
    """
    Graph-based communication with learned topology
    """
    
    def __init__(
        self,
        config: CommunicationConfig,
        hidden_size: int,
        num_experts: int
    ):
        super().__init__(config, hidden_size)
        self.num_experts = num_experts
        
        # Learned adjacency matrix
        if config.use_learned_topology:
            self.adjacency_logits = nn.Parameter(
                torch.randn(num_experts, num_experts)
            )
            
        # Graph convolution layers
        self.graph_conv = nn.ModuleList([
            GraphConvLayer(hidden_size, hidden_size)
            for _ in range(config.communication_rounds)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(config.communication_rounds)
        ])
        
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        routing_weights: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Perform graph-based communication
        """
        batch_size, seq_len, hidden_size = expert_outputs[0].shape
        
        # Stack expert outputs: [batch, seq_len, num_experts, hidden]
        node_features = torch.stack(expert_outputs, dim=2)
        
        # Get adjacency matrix
        if self.config.use_learned_topology:
            # Apply sparsity
            if self.config.use_sparse_communication:
                adjacency = self._get_sparse_adjacency()
            else:
                adjacency = torch.sigmoid(self.adjacency_logits)
        else:
            # Use routing weights as adjacency
            if routing_weights is not None:
                # Compute expert co-activation
                adjacency = self._compute_coactivation_adjacency(
                    routing_weights
                )
            else:
                # Fully connected
                adjacency = torch.ones(
                    self.num_experts,
                    self.num_experts,
                    device=node_features.device
                )
                
        # Apply graph convolutions
        for i, (conv, norm) in enumerate(
            zip(self.graph_conv, self.layer_norms)
        ):
            # Reshape for graph conv
            node_features_flat = rearrange(
                node_features,
                'b s e h -> (b s) e h'
            )
            
            # Apply convolution
            updated = conv(node_features_flat, adjacency)
            
            # Reshape back
            updated = rearrange(
                updated,
                '(b s) e h -> b s e h',
                b=batch_size,
                s=seq_len
            )
            
            # Residual and norm
            if self.config.communication_residual:
                node_features = (
                    self.config.residual_weight * node_features +
                    (1 - self.config.residual_weight) * updated
                )
            else:
                node_features = updated
                
            node_features = norm(node_features)
            
        # Convert back to list
        updated_outputs = [
            node_features[:, :, i] for i in range(self.num_experts)
        ]
        
        return updated_outputs
        
    def _get_sparse_adjacency(self) -> torch.Tensor:
        """Get sparse adjacency matrix"""
        # Keep only top-k connections per expert
        k = int(self.num_experts * self.config.sparsity_ratio)
        
        adjacency = torch.sigmoid(self.adjacency_logits)
        
        # Make sparse
        topk_values, _ = torch.topk(adjacency, k, dim=1)
        threshold = topk_values[:, -1].unsqueeze(1)
        sparse_adjacency = torch.where(
            adjacency >= threshold,
            adjacency,
            torch.zeros_like(adjacency)
        )
        
        # Make symmetric
        sparse_adjacency = (sparse_adjacency + sparse_adjacency.T) / 2
        
        return sparse_adjacency
        
    def _compute_coactivation_adjacency(
        self,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute adjacency based on expert co-activation"""
        # routing_weights: [batch, seq_len, num_experts]
        
        # Average over batch and sequence
        avg_routing = routing_weights.mean(dim=[0, 1])  # [num_experts]
        
        # Compute co-activation matrix
        adjacency = torch.outer(avg_routing, avg_routing)
        
        # Normalize
        adjacency = adjacency / adjacency.max()
        
        return adjacency


class GraphConvLayer(nn.Module):
    """
    Single graph convolution layer
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply graph convolution
        
        Args:
            node_features: [batch, num_nodes, features]
            adjacency: [num_nodes, num_nodes]
        """
        # Normalize adjacency
        row_sum = adjacency.sum(dim=1, keepdim=True)
        norm_adjacency = adjacency / (row_sum + 1e-8)
        
        # Apply convolution: A @ X @ W
        aggregated = torch.matmul(norm_adjacency, node_features)
        output = self.linear(aggregated)
        
        return F.relu(output)


class CommunicatingMoE(nn.Module):
    """
    MoE layer with communicating experts
    """
    
    def __init__(
        self,
        num_experts: int,
        expert_class: type,
        expert_config: Any,
        communication_config: CommunicationConfig,
        hidden_size: int
    ):
        super().__init__()
        self.num_experts = num_experts
        self.config = communication_config
        
        # Create experts
        self.experts = nn.ModuleList([
            expert_class(expert_config)
            for _ in range(num_experts)
        ])
        
        # Create router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Create communicator
        if communication_config.communication_type == "attention":
            self.communicator = AttentionCommunicator(
                communication_config,
                hidden_size
            )
        elif communication_config.communication_type == "message_passing":
            self.communicator = MessagePassingCommunicator(
                communication_config,
                hidden_size
            )
        elif communication_config.communication_type == "graph":
            self.communicator = GraphCommunicator(
                communication_config,
                hidden_size,
                num_experts
            )
        else:
            raise ValueError(
                f"Unknown communication type: {communication_config.communication_type}"
            )
            
        # Top-k selection for efficiency
        self.use_top_k = communication_config.use_sparse_communication
        self.top_k = min(communication_config.top_k_experts, num_experts)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with expert communication
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts if using sparse communication
        if self.use_top_k:
            topk_weights, topk_indices = torch.topk(
                routing_weights,
                self.top_k,
                dim=-1
            )
            # Renormalize
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_indices = None
            topk_weights = routing_weights
            
        # Get expert outputs
        expert_outputs = []
        active_experts = []
        
        for i in range(self.num_experts):
            if topk_indices is not None:
                # Check if expert is in top-k for any token
                is_active = (topk_indices == i).any()
                if not is_active:
                    continue
                    
            expert_output = self.experts[i](hidden_states)
            expert_outputs.append(expert_output)
            active_experts.append(i)
            
        # Apply communication between active experts
        if len(expert_outputs) > 1:
            communicated_outputs = self.communicator(
                expert_outputs,
                routing_weights[:, :, active_experts] if active_experts else routing_weights
            )
        else:
            communicated_outputs = expert_outputs
            
        # Combine expert outputs
        if topk_indices is not None:
            # Sparse combination
            combined_output = torch.zeros_like(hidden_states)
            
            for i, (expert_idx, expert_output) in enumerate(
                zip(active_experts, communicated_outputs)
            ):
                # Find positions where this expert is selected
                expert_mask = (topk_indices == expert_idx)
                expert_weights = torch.where(
                    expert_mask,
                    topk_weights,
                    torch.zeros_like(topk_weights)
                )
                
                # Add weighted expert output
                combined_output += expert_weights.unsqueeze(-1) * expert_output
        else:
            # Dense combination
            stacked_outputs = torch.stack(communicated_outputs, dim=2)
            routing_weights_expanded = routing_weights.unsqueeze(-1)
            combined_output = (stacked_outputs * routing_weights_expanded).sum(dim=2)
            
        # Prepare auxiliary outputs
        aux_outputs = None
        if return_router_logits:
            aux_outputs = {
                "router_logits": router_logits,
                "routing_weights": routing_weights,
                "active_experts": active_experts,
                "communication_rounds": self.config.communication_rounds
            }
            
        return combined_output, aux_outputs


class HierarchicalCommunicatingMoE(nn.Module):
    """
    Hierarchical MoE with expert groups that communicate
    """
    
    def __init__(
        self,
        num_expert_groups: int,
        experts_per_group: int,
        expert_class: type,
        expert_config: Any,
        communication_config: CommunicationConfig,
        hidden_size: int
    ):
        super().__init__()
        self.num_expert_groups = num_expert_groups
        self.experts_per_group = experts_per_group
        self.total_experts = num_expert_groups * experts_per_group
        
        # Create expert groups
        self.expert_groups = nn.ModuleList([
            CommunicatingMoE(
                experts_per_group,
                expert_class,
                expert_config,
                communication_config,
                hidden_size
            )
            for _ in range(num_expert_groups)
        ])
        
        # Group-level router
        self.group_router = nn.Linear(hidden_size, num_expert_groups)
        
        # Inter-group communicator
        self.group_communicator = AttentionCommunicator(
            communication_config,
            hidden_size
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Hierarchical forward pass with two-level routing
        """
        # Group-level routing
        group_logits = self.group_router(hidden_states)
        group_weights = F.softmax(group_logits, dim=-1)
        
        # Get outputs from each group
        group_outputs = []
        group_aux_outputs = []
        
        for group in self.expert_groups:
            output, aux = group(hidden_states, return_router_logits)
            group_outputs.append(output)
            if aux:
                group_aux_outputs.append(aux)
                
        # Apply inter-group communication
        if len(group_outputs) > 1:
            communicated_group_outputs = self.group_communicator(
                group_outputs,
                group_weights
            )
        else:
            communicated_group_outputs = group_outputs
            
        # Combine group outputs
        stacked_outputs = torch.stack(communicated_group_outputs, dim=2)
        group_weights_expanded = group_weights.unsqueeze(-1)
        combined_output = (stacked_outputs * group_weights_expanded).sum(dim=2)
        
        # Prepare auxiliary outputs
        aux_outputs = None
        if return_router_logits:
            aux_outputs = {
                "group_logits": group_logits,
                "group_weights": group_weights,
                "group_aux_outputs": group_aux_outputs
            }
            
        return combined_output, aux_outputs