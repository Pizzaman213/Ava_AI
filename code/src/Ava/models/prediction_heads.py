"""
Multi-Token Prediction Heads for Adaptive MTP

This module implements lightweight prediction heads that predict tokens
at future positions (t+1, t+2, t+3, etc.) from shared transformer backbone.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple


class MultiTokenPredictionHeads(nn.Module):
    """
    Lightweight prediction heads for predicting multiple future tokens.

    These heads share the same transformer backbone but branch off at the
    final layers to predict different future positions efficiently.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 3,
        head_type: str = 'linear',
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        share_projections: bool = False,
        use_layer_norm: bool = True,
    ):
        """
        Initialize multi-token prediction heads.

        Args:
            hidden_size: Dimension of hidden states from backbone
            vocab_size: Size of vocabulary
            num_heads: Number of future positions to predict (2-4 recommended)
            head_type: Type of head ('linear' or 'mlp')
            intermediate_size: Intermediate dimension for MLP heads
            dropout: Dropout probability
            share_projections: Whether to share weights across heads
            use_layer_norm: Whether to apply layer normalization
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_type = head_type
        self.share_projections = share_projections

        # Input layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        # Create prediction heads
        if share_projections:
            # Single shared head for all positions
            self.shared_head: Optional[nn.Module] = self._create_head(
                hidden_size, vocab_size * num_heads,
                intermediate_size, dropout
            )
            self.heads: Optional[nn.ModuleList] = None
        else:
            # Separate head for each future position
            self.heads = nn.ModuleList([
                self._create_head(hidden_size, vocab_size, intermediate_size, dropout)
                for _ in range(num_heads)
            ])
            self.shared_head = None

    def _create_head(
        self,
        input_size: int,
        output_size: int,
        intermediate_size: Optional[int],
        dropout: float
    ) -> nn.Module:
        """
        Create a single prediction head.

        Args:
            input_size: Input dimension
            output_size: Output dimension (vocab_size)
            intermediate_size: Intermediate dimension for MLP
            dropout: Dropout probability

        Returns:
            Prediction head module
        """
        if self.head_type == 'linear':
            # Simple linear projection
            return nn.Linear(input_size, output_size)

        elif self.head_type == 'mlp':
            # Small MLP for richer representations
            if intermediate_size is None:
                intermediate_size = input_size * 2

            return nn.Sequential(
                nn.Linear(input_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, output_size)
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_logits: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict multiple future tokens from hidden states.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            return_all_logits: Return logits for all positions separately

        Returns:
            Dictionary containing:
                - all_logits: List of logits for each future position
                  Each: [batch_size, seq_len, vocab_size]
                - combined_logits: Optional combined logits if needed
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Apply layer normalization if configured
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        if self.share_projections:
            # Single forward pass for all heads
            assert self.shared_head is not None, "shared_head should not be None when share_projections is True"
            combined_logits = self.shared_head(hidden_states)
            # [batch_size, seq_len, vocab_size * num_heads]

            # Split into separate predictions
            all_logits = torch.chunk(combined_logits, self.num_heads, dim=-1)
            # List of [batch_size, seq_len, vocab_size]

        else:
            # Separate forward pass for each head
            assert self.heads is not None, "heads should not be None when share_projections is False"
            all_logits = [
                head(hidden_states)  # [batch_size, seq_len, vocab_size]
                for head in self.heads
            ]

        result = {
            'all_logits': all_logits,
            'num_heads': self.num_heads,
        }

        # Optionally return combined logits
        if return_all_logits:
            result['logits_list'] = all_logits

        return result

    def predict_tokens(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict future tokens with sampling strategies.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Dictionary with predicted tokens for each position
        """
        # Get logits for all heads
        outputs = self.forward(hidden_states)
        all_logits = outputs['all_logits']

        predictions = []
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        for logits in all_logits:
            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            predicted_tokens = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1
            ).view(batch_size, seq_len)

            predictions.append(predicted_tokens)

        return {
            'predictions': predictions,  # List of [batch_size, seq_len]
            'num_tokens': len(predictions),
        }

    def get_parameters_count(self) -> Dict[str, Any]:
        """
        Get the number of parameters in prediction heads.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())

        if self.share_projections:
            assert self.shared_head is not None
            shared_params = sum(p.numel() for p in self.shared_head.parameters())
            per_head_params = shared_params // self.num_heads
        else:
            assert self.heads is not None
            per_head_params = sum(p.numel() for p in self.heads[0].parameters())
            shared_params = 0

        return {
            'total_params': total_params,
            'per_head_params': per_head_params,
            'shared_params': shared_params,
            'num_heads': self.num_heads,
        }
