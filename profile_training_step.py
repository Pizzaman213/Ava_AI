#!/usr/bin/env python
"""Profile a single training step to find bottlenecks."""

import sys
sys.path.insert(0, '/project/code')

import time
import torch
from src.Ava.data.pretokenized_loader import create_ultra_fast_dataloaders
from src.Ava.models.moe_model import OptimizedMoETransformer, OptimizedMoEConfig
from transformers import AutoTokenizer

if __name__ == '__main__':
    print("Initializing model and data...")

    # Load model
    config = OptimizedMoEConfig(
        vocab_size=50680,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        num_experts=8,
        expert_capacity=96,
        max_position_embeddings=2048,
    )

    model = OptimizedMoETransformer(config).cuda().bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Load data
    train_loader, _ = create_ultra_fast_dataloaders(
        batch_size=96,
        max_length=256,
        data_dir='/project/code/data/processed',
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
    )

    print("\nProfiling training step...")

    # Get first batch
    data_iter = iter(train_loader)

    # Skip first batch (worker warmup)
    print("  Warming up workers...")
    t0 = time.time()
    batch = next(data_iter)
    print(f"    First batch: {time.time() - t0:.3f}s")

    # Profile second batch with detailed timing
    print("\n  Profiling second batch:")

    t_total = time.time()

    # Data loading
    t0 = time.time()
    batch = next(data_iter)
    t_data = time.time() - t0
    print(f"    Data loading: {t_data:.3f}s")

    # Move to GPU
    t0 = time.time()
    input_ids = batch['input_ids'].cuda()
    labels = batch['labels'].cuda()
    attention_mask = batch['attention_mask'].cuda()
    t_gpu_transfer = time.time() - t0
    print(f"    GPU transfer: {t_gpu_transfer:.3f}s")

    # Forward pass
    t0 = time.time()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    t_forward = time.time() - t0
    print(f"    Forward pass: {t_forward:.3f}s")

    # Backward pass
    t0 = time.time()
    loss.backward()
    t_backward = time.time() - t0
    print(f"    Backward pass: {t_backward:.3f}s")

    # Optimizer step
    t0 = time.time()
    optimizer.step()
    optimizer.zero_grad()
    t_optimizer = time.time() - t0
    print(f"    Optimizer step: {t_optimizer:.3f}s")

    total_time = time.time() - t_total

    print(f"\n  Total iteration: {total_time:.3f}s")
    print(f"  Expected throughput: {1/total_time:.1f} it/s")

    print("\n  Breakdown:")
    print(f"    Data loading:  {t_data/total_time*100:5.1f}%")
    print(f"    GPU transfer:  {t_gpu_transfer/total_time*100:5.1f}%")
    print(f"    Forward pass:  {t_forward/total_time*100:5.1f}%")
    print(f"    Backward pass: {t_backward/total_time*100:5.1f}%")
    print(f"    Optimizer:     {t_optimizer/total_time*100:5.1f}%")
