"""
Speculative Decoding Engine for faster inference
Implements draft model generation and verification with acceptance sampling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math
from dataclasses import dataclass

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    draft_model_layers: int = 4
    draft_model_dim: int = 512
    speculation_length: int = 5
    acceptance_threshold: float = 0.9
    temperature_matching: bool = True
    use_tree_speculation: bool = False
    tree_branches: int = 2
    adaptive_speculation: bool = True
    min_speculation_length: int = 2
    max_speculation_length: int = 10

class DraftModel(nn.Module):
    """
    Lightweight draft model for speculative token generation
    Uses fewer layers and smaller dimensions than the main model
    """
    def __init__(self, vocab_size: int, config: SpeculativeConfig):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Token embeddings (shared with main model if possible)
        self.embed_tokens = nn.Embedding(vocab_size, config.draft_model_dim)
        
        # Lightweight transformer layers
        self.layers = nn.ModuleList([
            DraftTransformerLayer(config.draft_model_dim)
            for _ in range(config.draft_model_layers)
        ])
        
        # Output projection
        self.lm_head = nn.Linear(config.draft_model_dim, vocab_size, bias=False)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(8192, config.draft_model_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(config.draft_model_dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Fast forward pass for draft token generation
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        pos_embeddings = self.pos_encoding(position_ids)
        hidden_states = hidden_states + pos_embeddings
        
        # Process through transformer layers
        all_hidden_states = []
        all_kv_caches = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, kv_cache = layer(hidden_states, past_kv, use_cache)
            
            all_hidden_states.append(hidden_states)
            if use_cache:
                all_kv_caches.append(kv_cache)
        
        # Final norm and projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "past_key_values": all_kv_caches,
            "hidden_states": all_hidden_states,
        }

class DraftTransformerLayer(nn.Module):
    """Lightweight transformer layer for draft model"""
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=0.0,
            batch_first=True
        )
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """Forward pass with optional caching"""
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        attn_output, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = residual + attn_output
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        # Simple KV cache (full implementation would be more sophisticated)
        kv_cache = (x, x) if use_cache else None
        
        return x, kv_cache

class SpeculativeDecoder(nn.Module):
    """
    Main speculative decoding engine
    Coordinates draft model and target model for faster generation
    """
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: Optional[DraftModel] = None,
        config: Optional[SpeculativeConfig] = None
    ):
        super().__init__()
        self.target_model = target_model
        self.config = config or SpeculativeConfig()
        
        # Create draft model if not provided
        if draft_model is None:
            vocab_size = target_model.config.vocab_size
            self.draft_model = DraftModel(vocab_size, self.config)
        else:
            self.draft_model = draft_model
        
        # Statistics tracking
        self.acceptance_history = []
        self.speculation_lengths = []

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        return_stats: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate tokens using speculative decoding
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            return_stats: Whether to return generation statistics
            
        Returns:
            Dictionary with generated tokens and optional statistics
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize
        generated_tokens = input_ids.clone()
        draft_cache = None
        target_cache = None
        
        # Statistics
        total_draft_tokens = 0
        total_accepted_tokens = 0
        num_iterations = 0
        
        with torch.no_grad():
            while generated_tokens.shape[1] - input_ids.shape[1] < max_new_tokens:
                # Determine speculation length (adaptive)
                if self.config.adaptive_speculation:
                    speculation_length = self._get_adaptive_speculation_length()
                else:
                    speculation_length = self.config.speculation_length
                
                # Generate draft tokens
                draft_tokens, draft_logits, draft_cache = self._generate_draft_tokens(
                    generated_tokens,
                    speculation_length,
                    temperature,
                    draft_cache
                )
                
                # Verify with target model
                accepted_tokens, target_cache, acceptance_mask = self._verify_tokens(
                    generated_tokens,
                    draft_tokens,
                    draft_logits,
                    temperature,
                    target_cache
                )
                
                # Update generated tokens
                if accepted_tokens.shape[1] > 0:
                    generated_tokens = torch.cat([generated_tokens, accepted_tokens], dim=1)
                    total_accepted_tokens += accepted_tokens.shape[1]
                else:
                    # If no tokens accepted, generate one token with target model
                    target_token, target_cache = self._generate_target_token(
                        generated_tokens,
                        temperature,
                        top_p,
                        top_k,
                        target_cache
                    )
                    generated_tokens = torch.cat([generated_tokens, target_token], dim=1)
                    total_accepted_tokens += 1
                
                # Update statistics
                total_draft_tokens += speculation_length
                num_iterations += 1
                
                # Update acceptance history for adaptive speculation
                if self.config.adaptive_speculation:
                    acceptance_rate = acceptance_mask.float().mean().item()
                    self.acceptance_history.append(acceptance_rate)
                    self.speculation_lengths.append(speculation_length)
        
        # Prepare output
        output = {
            "sequences": generated_tokens,
            "generated_tokens": generated_tokens[:, input_ids.shape[1]:],
        }
        
        if return_stats:
            output["stats"] = {
                "total_draft_tokens": total_draft_tokens,
                "total_accepted_tokens": total_accepted_tokens,
                "acceptance_rate": total_accepted_tokens / max(total_draft_tokens, 1),
                "num_iterations": num_iterations,
                "speedup": (generated_tokens.shape[1] - input_ids.shape[1]) / max(num_iterations, 1),
            }
        
        return output

    def _generate_draft_tokens(
        self,
        input_ids: torch.LongTensor,
        num_tokens: int,
        temperature: float,
        past_key_values: Optional[List] = None
    ) -> Tuple[torch.LongTensor, torch.Tensor, List]:
        """Generate draft tokens using the lightweight model"""
        draft_tokens = []
        draft_logits = []
        current_ids = input_ids
        
        for _ in range(num_tokens):
            # Forward pass through draft model
            outputs = self.draft_model(
                current_ids if past_key_values is None else current_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Sample token
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            draft_tokens.append(next_token)
            draft_logits.append(logits)
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        draft_tokens = torch.cat(draft_tokens, dim=1)
        draft_logits = torch.stack(draft_logits, dim=1)
        
        return draft_tokens, draft_logits, past_key_values

    def _verify_tokens(
        self,
        input_ids: torch.LongTensor,
        draft_tokens: torch.LongTensor,
        draft_logits: torch.Tensor,
        temperature: float,
        past_key_values: Optional[List] = None
    ) -> Tuple[torch.LongTensor, List, torch.Tensor]:
        """Verify draft tokens with target model"""
        batch_size = input_ids.shape[0]
        num_draft_tokens = draft_tokens.shape[1]
        
        # Prepare input for target model (original + draft tokens)
        full_input = torch.cat([input_ids, draft_tokens], dim=1)
        
        # Forward pass through target model
        target_outputs = self.target_model(
            full_input if past_key_values is None else full_input[:, -num_draft_tokens-1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        target_logits = target_outputs["logits"]
        target_cache = target_outputs["past_key_values"]
        
        # Extract logits for positions where draft tokens were generated
        start_pos = input_ids.shape[1]
        verification_logits = target_logits[:, start_pos-1:start_pos+num_draft_tokens-1]
        
        # Compute acceptance probabilities
        if temperature > 0:
            draft_probs = F.softmax(draft_logits / temperature, dim=-1)
            target_probs = F.softmax(verification_logits / temperature, dim=-1)
            
            # Acceptance probability for each token
            draft_token_probs = torch.gather(draft_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            target_token_probs = torch.gather(target_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Accept if target probability is high enough
            acceptance_probs = torch.minimum(
                torch.ones_like(target_token_probs),
                target_token_probs / (draft_token_probs + 1e-10)
            )
            
            # Sample acceptance decisions
            uniform_samples = torch.rand_like(acceptance_probs)
            accept_mask = uniform_samples < acceptance_probs
        else:
            # Greedy decoding - accept if target model would have generated the same token
            target_tokens = verification_logits.argmax(dim=-1)
            accept_mask = target_tokens == draft_tokens
        
        # Find first rejection point
        # Accept all tokens up to first rejection
        accept_mask_cumulative = accept_mask.cumprod(dim=1)
        num_accepted = accept_mask_cumulative.sum(dim=1)
        
        # Get accepted tokens
        max_accepted = num_accepted.max().item()
        if max_accepted > 0:
            accepted_tokens = draft_tokens[:, :max_accepted]
        else:
            accepted_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=input_ids.device)
        
        return accepted_tokens, target_cache, accept_mask

    def _generate_target_token(
        self,
        input_ids: torch.LongTensor,
        temperature: float,
        top_p: float,
        top_k: int,
        past_key_values: Optional[List] = None
    ) -> Tuple[torch.LongTensor, List]:
        """Generate a single token using the target model"""
        outputs = self.target_model(
            input_ids if past_key_values is None else input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        logits = outputs["logits"][:, -1, :]
        past_key_values = outputs["past_key_values"]
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        # Sample token
        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        
        return next_token, past_key_values

    def _get_adaptive_speculation_length(self) -> int:
        """Compute adaptive speculation length based on acceptance history"""
        if len(self.acceptance_history) < 10:
            return self.config.speculation_length
        
        # Use recent acceptance rates
        recent_rates = self.acceptance_history[-20:]
        avg_rate = sum(recent_rates) / len(recent_rates)
        
        # Adjust speculation length based on acceptance rate
        if avg_rate > 0.9:
            # High acceptance - increase speculation
            new_length = min(
                self.config.max_speculation_length,
                int(self.config.speculation_length * 1.2)
            )
        elif avg_rate < 0.5:
            # Low acceptance - decrease speculation
            new_length = max(
                self.config.min_speculation_length,
                int(self.config.speculation_length * 0.8)
            )
        else:
            # Moderate acceptance - maintain or slightly adjust
            new_length = self.config.speculation_length
        
        return new_length

class TreeSpeculativeDecoder(SpeculativeDecoder):
    """
    Tree-based speculative decoding for exploring multiple paths
    """
    def __init__(self, target_model: nn.Module, draft_model: Optional[DraftModel] = None):
        super().__init__(target_model, draft_model)
        self.tree_branches = self.config.tree_branches if self.config.use_tree_speculation else 1

    def _generate_tree_candidates(
        self,
        input_ids: torch.LongTensor,
        depth: int,
        temperature: float
    ) -> List[torch.LongTensor]:
        """Generate tree of candidate sequences"""
        candidates = [input_ids]
        
        for level in range(depth):
            new_candidates = []
            
            for candidate in candidates:
                # Generate multiple continuations
                outputs = self.draft_model(candidate)
                logits = outputs["logits"][:, -1, :]
                
                # Sample multiple tokens
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=self.tree_branches)
                else:
                    # Top-k for greedy
                    _, next_tokens = torch.topk(logits, k=self.tree_branches, dim=-1)
                
                # Create new candidates
                for i in range(self.tree_branches):
                    new_candidate = torch.cat([candidate, next_tokens[:, i:i+1]], dim=1)
                    new_candidates.append(new_candidate)
            
            candidates = new_candidates
        
        return candidates