#!/usr/bin/env python3
"""Quick test of model initialization on CPU."""

import torch
import sys
sys.path.insert(0, '/project/code')

from src.Ava.models.moe_model import OptimizedMoETransformer
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50680
    hidden_size: int = 512
    num_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    max_position_embeddings: int = 512
    num_experts: int = 4
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.0
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    expert_routing: str = "switch"
    use_expert_offloading: bool = True
    use_lora_experts: bool = True
    lora_rank: int = 8
    use_torch_compile: bool = False  # Disable for quick test
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_rotary_embeddings: bool = True

def test_model_init():
    print("Testing model initialization on CPU...")
    print(f"CUDA available: {torch.cuda.is_available()}")

    config = ModelConfig()
    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Expert offloading: {config.use_expert_offloading}")
    print(f"  LoRA experts: {config.use_lora_experts}")

    print("\nCreating model...")
    model = OptimizedMoETransformer(config)

    print(f"✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move to CPU and set dtype
    print("\nMoving to CPU with bfloat16...")
    model = model.to(device='cpu', dtype=torch.bfloat16)
    print(f"✓ Model on CPU with dtype {model.embed_tokens.weight.dtype}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    try:
        test_model_init()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
