#!/usr/bin/env python3
"""Profile training to find bottlenecks."""

import torch
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, '/project/code/src')
sys.path.insert(0, '/project/code/scripts/5_training')

# Import training components
from Ava.config.training_config import TrainingConfig
from Ava.models.optimized_moe_transformer import OptimizedMoETransformer

def profile_single_step():
    """Profile a single training step to find bottlenecks."""

    print("Loading config...")
    config_path = Path("/project/code/configs/moe/tiny_moe.yaml")
    config = TrainingConfig.from_yaml(config_path)

    print(f"Creating model...")
    model = OptimizedMoETransformer(config).cuda().to(torch.bfloat16)

    print(f"Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Create dummy batch
    batch_size = config.training.batch_size
    seq_len = config.model.max_position_embeddings
    vocab_size = config.model.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    print(f"\nBatch shape: {input_ids.shape}")
    print(f"Starting profiling...")
    print("=" * 60)

    # Warmup
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask, labels=labels)
    torch.cuda.synchronize()

    # Profile forward pass
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    torch.cuda.synchronize()
    forward_time = time.time() - start

    print(f"✓ Forward pass: {forward_time:.3f}s")
    print(f"  Loss: {loss.item():.4f}")

    # Profile backward pass
    start = time.time()

    loss.backward()

    torch.cuda.synchronize()
    backward_time = time.time() - start

    print(f"✓ Backward pass: {backward_time:.3f}s")

    # Memory usage
    memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"✓ Peak GPU memory: {memory_allocated:.2f} GB")

    print("=" * 60)
    print(f"Total time per step: {forward_time + backward_time:.3f}s")
    print(f"Expected throughput: {1 / (forward_time + backward_time):.2f} it/s")

    if (forward_time + backward_time) > 5.0:
        print("\n⚠️  WARNING: Step time >5s is TOO SLOW")
        print("Expected step time: <2s")
    else:
        print("\n✓ Step time is acceptable")

if __name__ == '__main__':
    profile_single_step()
