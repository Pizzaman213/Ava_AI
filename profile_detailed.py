#!/usr/bin/env python3
"""Detailed profiling to find bottlenecks."""

import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, '/project/code/src')

from Ava.config.training_config import TrainingConfig

def profile_model():
    """Profile model forward/backward in detail."""

    config_path = Path("/project/code/configs/moe/tiny_moe.yaml")
    config = TrainingConfig.from_yaml(config_path)

    print("Creating model...")
    from Ava.models import moe_model
    model_class = moe_model.OptimizedMoETransformer
    model = model_class(config).cuda().to(torch.bfloat16)
    model.train()

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Create batch
    batch_size = 16
    seq_len = 256
    vocab_size = config.model.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    labels = input_ids.clone()

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        outputs = model(input_ids, labels=labels)
        outputs.loss.backward()
    torch.cuda.synchronize()

    # Profile
    print("\nProfiling...")
    print("=" * 60)

    # Forward pass
    torch.cuda.synchronize()
    start = time.time()

    outputs = model(input_ids, labels=labels)

    torch.cuda.synchronize()
    forward_time = time.time() - start

    print(f"Forward pass: {forward_time:.3f}s")

    # Backward pass
    torch.cuda.synchronize()
    start = time.time()

    outputs.loss.backward()

    torch.cuda.synchronize()
    backward_time = time.time() - start

    print(f"Backward pass: {backward_time:.3f}s")
    print(f"Total: {forward_time + backward_time:.3f}s")
    print("=" * 60)

    total_time = forward_time + backward_time
    if total_time > 5.0:
        print(f"\n⚠️  SLOW: {total_time:.3f}s per step (expected <2s)")
        print(f"Forward: {forward_time/total_time*100:.1f}%")
        print(f"Backward: {backward_time/total_time*100:.1f}%")
    else:
        print(f"\n✓ FAST: {total_time:.3f}s per step")

if __name__ == '__main__':
    profile_model()
