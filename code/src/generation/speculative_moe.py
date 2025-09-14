"""
Speculative Mixture of Experts

Implements speculative decoding with small draft experts
for faster inference while maintaining quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative MoE"""
    # Draft model settings
    num_draft_experts: int = 4
    draft_expert_size_ratio: float = 0.1  # Draft experts are 10% of target size
    draft_sequence_length: int = 5  # Number of tokens to draft
    
    # Verification settings
    acceptance_threshold: float = 0.9
    temperature_matching: bool = True
    
    # Tree speculation
    use_tree_speculation: bool = False
    tree_depth: int = 3
    branching_factor: int = 2
    
    # Self-speculation (using early layers)
    use_self_speculation: bool = False
    draft_exit_layer: int = 6  # Use first 6 layers as draft
    
    # Performance tracking
    track_acceptance_rate: bool = True
    target_acceptance_rate: float = 0.7


class DraftExpert(nn.Module):
    """
    Lightweight expert for draft generation
    """
    
    def __init__(self, config: Any, draft_config: SpeculativeConfig):
        super().__init__()
        
        # Smaller dimensions
        self.hidden_size = int(config.hidden_size * draft_config.draft_expert_size_ratio)
        self.intermediate_size = int(config.intermediate_size * draft_config.draft_expert_size_ratio)
        
        # Simple feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.ReLU(),
            nn.Linear(self.intermediate_size, config.vocab_size)
        )
        
        # Lightweight attention (optional)
        self.use_attention = False
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                self.hidden_size,
                num_heads=max(1, config.num_attention_heads // 8),
                batch_first=True
            )
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fast forward pass for drafting"""
        # Simple MLP-based prediction
        logits = self.mlp(hidden_states)
        return logits


class SpeculativeMoE(nn.Module):
    """
    MoE with speculative decoding using small draft experts
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        config: SpeculativeConfig
    ):
        super().__init__()
        self.target_model = target_model
        self.config = config
        
        # Create draft experts
        self.draft_experts = nn.ModuleList([
            DraftExpert(target_model.config, config)
            for _ in range(config.num_draft_experts)
        ])
        
        # Router for selecting draft expert
        self.draft_router = nn.Linear(
            target_model.config.hidden_size,
            config.num_draft_experts
        )
        
        # Statistics tracking
        self.acceptance_history = []
        self.speedup_history = []
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate text using speculative decoding
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Initialize
        generated_tokens = []
        total_draft_tokens = 0
        total_accepted_tokens = 0
        
        # Timing
        start_time = time.time()
        
        while len(generated_tokens) < max_new_tokens:
            # Draft phase
            draft_tokens, draft_probs = self._draft_tokens(
                input_ids,
                self.config.draft_sequence_length,
                temperature
            )
            total_draft_tokens += draft_tokens.size(1)
            
            # Verification phase
            accepted_tokens, accept_mask = self._verify_tokens(
                input_ids,
                draft_tokens,
                draft_probs,
                temperature
            )
            
            # Update statistics
            num_accepted = accept_mask.sum().item()
            total_accepted_tokens += num_accepted
            self.acceptance_history.append(num_accepted / draft_tokens.size(1))
            
            # Append accepted tokens
            if num_accepted > 0:
                generated_tokens.extend(accepted_tokens[:num_accepted].tolist())
                input_ids = torch.cat([
                    input_ids,
                    accepted_tokens[:num_accepted].unsqueeze(0)
                ], dim=1)
                
            # If no tokens accepted, generate one with target model
            if num_accepted == 0:
                target_token = self._generate_with_target(
                    input_ids,
                    temperature,
                    top_k,
                    top_p
                )
                generated_tokens.append(target_token.item())
                input_ids = torch.cat([input_ids, target_token.unsqueeze(0)], dim=1)
                
        # Calculate statistics
        elapsed_time = time.time() - start_time
        acceptance_rate = total_accepted_tokens / max(total_draft_tokens, 1)
        
        # Estimate speedup
        draft_cost_ratio = self.config.draft_expert_size_ratio
        effective_speedup = (1 + acceptance_rate * (self.config.draft_sequence_length - 1)) / \
                          (1 + draft_cost_ratio * self.config.draft_sequence_length)
        
        stats = {
            "acceptance_rate": acceptance_rate,
            "effective_speedup": effective_speedup,
            "elapsed_time": elapsed_time,
            "tokens_per_second": len(generated_tokens) / elapsed_time
        }
        
        return torch.tensor(generated_tokens, device=device), stats
        
    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft multiple tokens using small experts"""
        device = input_ids.device
        drafted_tokens = []
        drafted_probs = []
        
        # Get hidden states from early layers of target model
        with torch.no_grad():
            if self.config.use_self_speculation:
                # Use early exit from target model
                hidden_states = self._get_early_hidden_states(input_ids)
            else:
                # Use last hidden state
                outputs = self.target_model(
                    input_ids,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
                
        # Select draft expert
        router_logits = self.draft_router(hidden_states[:, -1, :])
        expert_idx = router_logits.argmax(dim=-1)
        
        # Generate draft tokens autoregressively
        current_hidden = hidden_states[:, -1, :].unsqueeze(1)
        
        for _ in range(num_tokens):
            # Get logits from draft expert
            draft_logits = self.draft_experts[expert_idx](current_hidden)
            
            # Sample token
            if temperature > 0:
                probs = F.softmax(draft_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = draft_logits[:, -1].argmax(dim=-1, keepdim=True)
                
            drafted_tokens.append(next_token)
            drafted_probs.append(probs if temperature > 0 else 
                                F.softmax(draft_logits[:, -1], dim=-1))
            
            # Update hidden state (simplified)
            # In practice, you'd run through more of the model
            current_hidden = self._update_hidden_state(current_hidden, next_token)
            
        # Stack drafted tokens and probabilities
        drafted_tokens = torch.cat(drafted_tokens, dim=1)
        drafted_probs = torch.stack(drafted_probs, dim=1)
        
        return drafted_tokens, drafted_probs
        
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify drafted tokens with target model"""
        device = input_ids.device
        batch_size = input_ids.size(0)
        num_draft = draft_tokens.size(1)
        
        # Concatenate draft tokens to input
        extended_input = torch.cat([input_ids, draft_tokens], dim=1)
        
        # Run target model on extended sequence
        with torch.no_grad():
            target_outputs = self.target_model(extended_input)
            target_logits = target_outputs.logits
            
        # Get logits for draft positions
        draft_position_logits = target_logits[:, input_ids.size(1)-1:-1]
        
        # Compute target probabilities
        if temperature > 0:
            target_probs = F.softmax(draft_position_logits / temperature, dim=-1)
        else:
            target_probs = F.softmax(draft_position_logits, dim=-1)
            
        # Verify each drafted token
        accept_mask = torch.ones(num_draft, dtype=torch.bool, device=device)
        
        for i in range(num_draft):
            if self.config.temperature_matching:
                # Check if draft probability is close to target
                draft_prob = draft_probs[:, i].gather(1, draft_tokens[:, i:i+1])
                target_prob = target_probs[:, i].gather(1, draft_tokens[:, i:i+1])
                
                accept = (draft_prob / target_prob) >= self.config.acceptance_threshold
            else:
                # Simple threshold on target probability
                target_prob = target_probs[:, i].gather(1, draft_tokens[:, i:i+1])
                accept = target_prob >= self.config.acceptance_threshold
                
            accept_mask[i] = accept.squeeze()
            
            # Stop at first rejection
            if not accept_mask[i]:
                accept_mask[i+1:] = False
                break
                
        # Get accepted tokens
        num_accepted = accept_mask.sum().item()
        accepted_tokens = draft_tokens[:, :num_accepted] if num_accepted > 0 else draft_tokens[:, :0]
        
        return accepted_tokens.squeeze(0), accept_mask
        
    def _generate_with_target(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> torch.Tensor:
        """Generate single token with target model"""
        with torch.no_grad():
            outputs = self.target_model(input_ids)
            logits = outputs.logits[:, -1]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
            # Apply top-k/top-p filtering
            if top_k is not None:
                logits = top_k_filtering(logits, top_k)
            if top_p is not None:
                logits = top_p_filtering(logits, top_p)
                
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
        return next_token
        
    def _get_early_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states from early layers for self-speculation"""
        # Run through only early layers
        with torch.no_grad():
            # This is model-specific - simplified version
            embeddings = self.target_model.get_input_embeddings()(input_ids)
            hidden_states = embeddings
            
            # Run through first N layers
            for i in range(self.config.draft_exit_layer):
                layer = self.target_model.layers[i] if hasattr(self.target_model, 'layers') else None
                if layer is not None:
                    hidden_states = layer(hidden_states)[0]
                    
        return hidden_states
        
    def _update_hidden_state(
        self,
        hidden_state: torch.Tensor,
        next_token: torch.Tensor
    ) -> torch.Tensor:
        """Simple hidden state update for draft model"""
        # This is a simplified version
        # In practice, you'd properly update the hidden state
        return hidden_state  # Placeholder
        
    def adapt_acceptance_threshold(self):
        """Adapt acceptance threshold based on performance"""
        if not self.acceptance_history:
            return
            
        recent_rate = np.mean(self.acceptance_history[-100:])
        
        if recent_rate < self.config.target_acceptance_rate:
            # Lower threshold to accept more
            self.config.acceptance_threshold *= 0.95
            logger.info(f"Lowered acceptance threshold to {self.config.acceptance_threshold:.3f}")
        elif recent_rate > self.config.target_acceptance_rate + 0.1:
            # Raise threshold to be more selective
            self.config.acceptance_threshold *= 1.05
            logger.info(f"Raised acceptance threshold to {self.config.acceptance_threshold:.3f}")


class TreeSpeculativeMoE(SpeculativeMoE):
    """
    Tree-based speculative decoding with branching
    """
    
    def __init__(self, target_model: nn.Module, config: SpeculativeConfig):
        super().__init__(target_model, config)
        
        # Enable tree speculation
        self.config.use_tree_speculation = True
        
    def _draft_tree(
        self,
        input_ids: torch.Tensor,
        depth: int,
        branching_factor: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Draft a tree of possible continuations"""
        tree = {"tokens": [], "probs": [], "children": []}
        
        # Draft multiple branches at each level
        for level in range(depth):
            if level == 0:
                # First level: branch from original sequence
                branches = self._draft_branches(
                    input_ids,
                    branching_factor,
                    temperature
                )
                tree["tokens"] = branches["tokens"]
                tree["probs"] = branches["probs"]
                tree["children"] = [
                    self._draft_tree(
                        torch.cat([input_ids, tok.unsqueeze(0)], dim=1),
                        depth - 1,
                        branching_factor,
                        temperature
                    )
                    for tok in branches["tokens"]
                ]
            else:
                # Recursive branching
                break  # Simplified for now
                
        return tree
        
    def _draft_branches(
        self,
        input_ids: torch.Tensor,
        num_branches: int,
        temperature: float
    ) -> Dict[str, torch.Tensor]:
        """Draft multiple token options"""
        # Get hidden states
        with torch.no_grad():
            outputs = self.target_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
        # Select draft expert
        router_logits = self.draft_router(hidden_states[:, -1, :])
        expert_idx = router_logits.argmax(dim=-1)
        
        # Get logits from draft expert
        draft_logits = self.draft_experts[expert_idx](hidden_states[:, -1:, :])
        
        # Sample multiple tokens
        if temperature > 0:
            probs = F.softmax(draft_logits[:, -1] / temperature, dim=-1)
            tokens = torch.multinomial(probs, num_branches, replacement=True)
        else:
            # Top-k tokens
            tokens = draft_logits[:, -1].topk(num_branches, dim=-1).indices
            probs = F.softmax(draft_logits[:, -1], dim=-1)
            
        return {
            "tokens": tokens.squeeze(0),
            "probs": probs
        }


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits"""
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    
    return logits