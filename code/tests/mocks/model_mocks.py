"""
Mock models for testing training pipeline without full model initialization.

These mocks provide the same interface as real models but with minimal
computation, allowing fast testing of training components.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MockMoEConfig:
    """Minimal config for mock MoE models."""

    vocab_size: int = 1000
    hidden_size: int = 64
    num_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 256
    max_position_embeddings: int = 128
    num_experts: int = 4
    num_experts_per_token: int = 2
    capacity_factor: float = 1.25
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    router_z_loss_coef: float = 0.001
    load_balance_loss_coef: float = 0.01
    gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    use_fused_qkv: bool = False
    use_fused_norm: bool = False
    use_triton_kernels: bool = False


class MockMoEForward:
    """
    Mock forward pass output for MoE models.

    Generates realistic-looking outputs without actual computation.
    """

    def __init__(
        self,
        batch_size: int = 4,
        seq_len: int = 64,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_experts: int = 4,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate mock forward outputs."""
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Generate realistic logits
        logits = torch.randn(
            batch_size, seq_len, self.vocab_size,
            device=self.device, dtype=self.dtype
        )

        result = {'logits': logits}

        # Compute loss if labels provided
        if labels is not None:
            # Simple cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss

            # Add auxiliary losses
            result['router_loss'] = torch.tensor(0.01, device=self.device, dtype=self.dtype, requires_grad=True)
            result['load_balance_loss'] = torch.tensor(0.005, device=self.device, dtype=self.dtype, requires_grad=True)
            result['router_z_loss'] = torch.tensor(0.001, device=self.device, dtype=self.dtype, requires_grad=True)

        return result


class SimpleMoEModel(nn.Module):
    """
    A simple MoE model for testing that actually computes gradients.

    This is a minimal but functional model suitable for testing
    training loops, optimizers, and gradient computations.
    """

    def __init__(self, config: Optional[MockMoEConfig] = None):
        super().__init__()
        config = config or MockMoEConfig()
        self.config = config

        # Simple embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Simple transformer layer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True,
            )
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Mock router parameters for aux loss testing
        self.router_gate = nn.Linear(config.hidden_size, config.num_experts)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with loss computation."""
        # Embed
        hidden = self.embedding(input_ids)

        # Create causal mask if needed
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )

        # Transform
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask)

        # Project to vocab
        logits = self.lm_head(hidden)

        result = {'logits': logits}

        if labels is not None:
            # Compute language modeling loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss

            # Compute mock auxiliary losses
            router_logits = self.router_gate(hidden.mean(dim=1))  # [batch, num_experts]
            router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() * 0.001
            result['router_z_loss'] = router_z_loss

            load_balance_loss = (router_logits.softmax(dim=-1).var(dim=-1).mean()) * 0.01
            result['load_balance_loss'] = load_balance_loss

        return result


def create_mock_moe_model(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    device: torch.device = torch.device('cpu'),
    requires_grad: bool = True,
) -> Mock:
    """
    Create a fully mocked MoE model.

    The mock model has the same interface as a real model but with
    no actual computation. Useful for testing component integration.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        device: Target device
        requires_grad: Whether mock tensors require grad

    Returns:
        Mock model with forward(), parameters(), etc.
    """
    model = Mock(spec=nn.Module)

    # Create mock forward that returns loss
    mock_forward = MockMoEForward(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        device=device,
    )
    model.forward = Mock(side_effect=mock_forward)
    model.__call__ = model.forward

    # Create mock parameters
    mock_params = [
        torch.randn(hidden_size, hidden_size, requires_grad=requires_grad),
        torch.randn(hidden_size, requires_grad=requires_grad),
    ]

    model.parameters = Mock(return_value=iter(mock_params))
    model.named_parameters = Mock(return_value=[
        ('layer.weight', mock_params[0]),
        ('layer.bias', mock_params[1]),
    ])

    # Training mode
    model.training = True
    model.train = Mock(return_value=model)
    model.eval = Mock(return_value=model)

    # Device management
    model.to = Mock(return_value=model)
    model.cuda = Mock(return_value=model)
    model.cpu = Mock(return_value=model)

    # State dict
    model.state_dict = Mock(return_value={
        'layer.weight': mock_params[0],
        'layer.bias': mock_params[1],
    })
    model.load_state_dict = Mock()

    # Module list/children
    model.modules = Mock(return_value=iter([model]))
    model.children = Mock(return_value=iter([]))
    model.named_modules = Mock(return_value=[('', model)])

    return model


def create_simple_moe_model(
    config: Optional[MockMoEConfig] = None,
    device: torch.device = torch.device('cpu'),
) -> SimpleMoEModel:
    """
    Create a simple but functional MoE model for testing.

    This model actually computes gradients and can be used for
    testing training loops end-to-end.

    Args:
        config: Model configuration
        device: Target device

    Returns:
        Functional SimpleMoEModel
    """
    config = config or MockMoEConfig()
    model = SimpleMoEModel(config)
    return model.to(device)


class MockRouter(nn.Module):
    """Mock router for testing routing logic."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_experts: int = 4,
        num_selected: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Returns:
            expert_indices: [num_tokens, k] - selected expert indices
            expert_weights: [num_tokens, k] - routing weights
            aux_loss: Scalar auxiliary loss
        """
        # Flatten to [num_tokens, hidden]
        batch_size, seq_len, _ = hidden_states.shape
        hidden_flat = hidden_states.view(-1, self.hidden_size)

        # Compute routing logits
        logits = self.gate(hidden_flat)  # [num_tokens, num_experts]

        # Top-k selection
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.num_selected, dim=-1)

        # Renormalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Compute aux loss
        aux_loss = torch.logsumexp(logits, dim=-1).pow(2).mean() * 0.001

        return indices, weights, aux_loss


def create_mock_router(
    hidden_size: int = 64,
    num_experts: int = 4,
    num_selected: int = 2,
) -> MockRouter:
    """Create a mock router for testing."""
    return MockRouter(hidden_size, num_experts, num_selected)


class MockExpert(nn.Module):
    """Mock expert for testing expert computation."""

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def create_mock_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    """Create a mock optimizer for testing."""
    return torch.optim.AdamW(model.parameters(), lr=lr)


def create_mock_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps: int = 1000,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a mock scheduler for testing."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps
    )
