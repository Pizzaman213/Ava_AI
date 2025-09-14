"""
Advanced Inference Techniques for MoE++ Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class InferenceConfig:
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0


class DiverseBeamSearch:
    """Diverse beam search for better generation diversity"""
    
    def __init__(self, model: nn.Module, beam_size: int = 4, diversity_penalty: float = 0.5):
        self.model = model
        self.beam_size = beam_size
        self.diversity_penalty = diversity_penalty
    
    def search(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        **kwargs
    ) -> List[torch.Tensor]:
        """Perform diverse beam search"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize beams
        beams = [[input_ids[i].clone()] for i in range(batch_size)]
        beam_scores = [[0.0] for _ in range(batch_size)]
        
        for _ in range(max_length - input_ids.shape[1]):
            new_beams = []
            new_scores = []
            
            for batch_idx in range(batch_size):
                batch_new_beams = []
                batch_new_scores = []
                
                for beam_idx, (beam, score) in enumerate(zip(beams[batch_idx], beam_scores[batch_idx])):
                    # Get model predictions
                    with torch.no_grad():
                        outputs = self.model(beam.unsqueeze(0))
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits', outputs.get('loss', None))
                            if logits is None:
                                # Fallback for models that return tuples
                                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        else:
                            logits = outputs
                        
                        # Take last token logits
                        if logits.dim() == 3:
                            next_token_logits = logits[0, -1, :]
                        else:
                            next_token_logits = logits[-1, :]
                    
                    # Apply diversity penalty
                    if beam_idx > 0:
                        for prev_beam in batch_new_beams:
                            if len(prev_beam) > 0:
                                prev_token = prev_beam[-1][-1]
                                next_token_logits[prev_token] -= self.diversity_penalty
                    
                    # Get top k tokens
                    topk_scores, topk_indices = torch.topk(next_token_logits, k=self.beam_size)
                    
                    for token_score, token_idx in zip(topk_scores, topk_indices):
                        new_beam = torch.cat([beam, token_idx.unsqueeze(0)])
                        batch_new_beams.append(new_beam)
                        batch_new_scores.append(score + token_score.item())
                
                # Select top beams
                sorted_beams = sorted(zip(batch_new_scores, batch_new_beams), reverse=True)
                new_beams.append([beam for _, beam in sorted_beams[:self.beam_size]])
                new_scores.append([score for score, _ in sorted_beams[:self.beam_size]])
            
            beams = new_beams
            beam_scores = new_scores
        
        # Return best beam for each batch
        return [beams[i][0] for i in range(batch_size)]


class AdaptiveInference:
    """Adaptive inference with early exit and dynamic compute"""
    
    def __init__(self, model: nn.Module, confidence_threshold: float = 0.95):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.exit_stats = []
    
    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive compute"""
        # Check if model supports early exit
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_mod_plus_plus'):
            # Use MoD++ adaptive depth if available
            outputs = self.model(input_ids, **kwargs)
            
            if isinstance(outputs, dict) and 'exit_layer' in outputs:
                self.exit_stats.append(outputs['exit_layer'])
            
            return outputs
        else:
            # Standard forward pass
            return self.model(input_ids, **kwargs)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get adaptive inference statistics"""
        if not self.exit_stats:
            return {}
        
        return {
            'avg_exit_layer': np.mean(self.exit_stats),
            'min_exit_layer': min(self.exit_stats),
            'max_exit_layer': max(self.exit_stats),
            'early_exit_rate': sum(1 for e in self.exit_stats if e < self.model.config.num_layers) / len(self.exit_stats)
        }


class StreamingGeneration:
    """Streaming text generation for real-time output"""
    
    def __init__(self, model: nn.Module, buffer_size: int = 10):
        self.model = model
        self.buffer_size = buffer_size
        self.generation_buffer = []
    
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        callback: Optional[callable] = None
    ):
        """Generate tokens in streaming fashion"""
        current_ids = input_ids.clone()
        
        for step in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                outputs = self.model(current_ids)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('loss', None))
                    if logits is None:
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    logits = outputs
                
                # Get next token logits
                if logits.dim() == 3:
                    next_token_logits = logits[0, -1, :] / temperature
                else:
                    next_token_logits = logits[-1, :] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
                # Add to buffer
                self.generation_buffer.append(next_token.item())
                
                # Call callback if provided
                if callback:
                    callback(next_token.item())
                
                # Yield tokens when buffer is full
                if len(self.generation_buffer) >= self.buffer_size:
                    yield self.generation_buffer.copy()
                    self.generation_buffer = []
        
        # Yield remaining tokens
        if self.generation_buffer:
            yield self.generation_buffer


class SpeculativeDecoding:
    """Speculative decoding with draft model"""
    
    def __init__(self, target_model: nn.Module, draft_model: Optional[nn.Module] = None):
        self.target_model = target_model
        self.draft_model = draft_model or self._create_draft_model()
        self.acceptance_rate = []
    
    def _create_draft_model(self):
        """Create a smaller draft model"""
        # For now, just use the same model (in practice, would be smaller)
        return self.target_model
    
    def decode(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        draft_steps: int = 4
    ) -> torch.Tensor:
        """Speculative decoding with draft model"""
        current_ids = input_ids.clone()
        
        while current_ids.shape[1] < max_length:
            # Generate draft tokens
            draft_ids = current_ids.clone()
            draft_logits = []
            
            with torch.no_grad():
                for _ in range(draft_steps):
                    outputs = self.draft_model(draft_ids)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs[0] if isinstance(outputs, tuple) else outputs)
                    else:
                        logits = outputs
                    
                    if logits.dim() == 3:
                        next_logits = logits[0, -1, :]
                    else:
                        next_logits = logits[-1, :]
                    
                    draft_logits.append(next_logits)
                    next_token = torch.argmax(next_logits)
                    draft_ids = torch.cat([draft_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Verify with target model
            with torch.no_grad():
                target_outputs = self.target_model(draft_ids)
                if isinstance(target_outputs, dict):
                    target_logits = target_outputs.get('logits', target_outputs[0])
                else:
                    target_logits = target_outputs
            
            # Accept/reject draft tokens
            accepted = 0
            for i in range(draft_steps):
                if i < target_logits.shape[1] - current_ids.shape[1]:
                    target_probs = F.softmax(target_logits[0, current_ids.shape[1] + i - 1, :], dim=-1)
                    draft_probs = F.softmax(draft_logits[i], dim=-1)
                    
                    # Calculate acceptance probability
                    draft_token = draft_ids[0, current_ids.shape[1] + i]
                    accept_prob = min(1.0, target_probs[draft_token] / (draft_probs[draft_token] + 1e-8))
                    
                    if torch.rand(1).item() < accept_prob:
                        accepted += 1
                    else:
                        break
            
            # Update sequence with accepted tokens
            if accepted > 0:
                current_ids = draft_ids[:, :current_ids.shape[1] + accepted]
                self.acceptance_rate.append(accepted / draft_steps)
            else:
                # Sample from target model
                with torch.no_grad():
                    outputs = self.target_model(current_ids)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs[0])
                    else:
                        logits = outputs
                    
                    if logits.dim() == 3:
                        next_logits = logits[0, -1, :]
                    else:
                        next_logits = logits[-1, :]
                    
                    next_token = torch.argmax(next_logits)
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    self.acceptance_rate.append(0.0)
        
        return current_ids
    
    def get_statistics(self) -> Dict[str, float]:
        """Get speculative decoding statistics"""
        if not self.acceptance_rate:
            return {}
        
        return {
            'avg_acceptance_rate': np.mean(self.acceptance_rate),
            'min_acceptance_rate': min(self.acceptance_rate),
            'max_acceptance_rate': max(self.acceptance_rate)
        }