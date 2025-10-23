#!/usr/bin/env python3
"""
Benchmark Script: Colossal-AI Speed Comparison

This script benchmarks training speed with different Colossal-AI configurations
to demonstrate the performance improvements.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.Ava.models.colossalai_moe_model import (
    ColossalAIMoEConfig,
    create_colossal_moe_model,
)


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    config_name: str
    steps_per_second: float
    memory_used_gb: float
    time_per_step_ms: float
    tokens_per_second: float
    effective_batch_size: int


def create_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, device: str = "cuda"):
    """Create dummy training batch"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def benchmark_configuration(
    config_name: str,
    model_config: ColossalAIMoEConfig,
    batch_size: int,
    seq_len: int,
    num_steps: int = 100,
    warmup_steps: int = 10,
) -> BenchmarkResult:
    """Benchmark a specific configuration"""

    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = create_colossal_moe_model(model_config)
    model = model.to(device)
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    print(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        batch = create_dummy_batch(batch_size, seq_len, model_config.vocab_size, device)

        optimizer.zero_grad()
        outputs = model(batch["input_ids"], labels=batch["labels"])
        loss = outputs["loss"]
        if loss is not None:
            loss.backward()
            optimizer.step()

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_steps} benchmark steps...")
    start_time = time.perf_counter()

    for step in range(num_steps):
        batch = create_dummy_batch(batch_size, seq_len, model_config.vocab_size, device)

        optimizer.zero_grad()
        outputs = model(batch["input_ids"], labels=batch["labels"])
        loss = outputs["loss"]
        if loss is not None:
            loss.backward()
            optimizer.step()

        # Progress
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    # Synchronize after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate metrics
    steps_per_second = num_steps / total_time
    time_per_step_ms = (total_time / num_steps) * 1000
    tokens_per_step = batch_size * seq_len
    tokens_per_second = tokens_per_step * steps_per_second

    # Memory usage
    if torch.cuda.is_available():
        memory_used_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        memory_used_gb = 0

    result = BenchmarkResult(
        config_name=config_name,
        steps_per_second=steps_per_second,
        memory_used_gb=memory_used_gb,
        time_per_step_ms=time_per_step_ms,
        tokens_per_second=tokens_per_second,
        effective_batch_size=batch_size,
    )

    # Print results
    print(f"\nResults for {config_name}:")
    print(f"  Steps/second: {steps_per_second:.2f}")
    print(f"  Time/step: {time_per_step_ms:.2f} ms")
    print(f"  Tokens/second: {tokens_per_second:,.0f}")
    print(f"  Memory used: {memory_used_gb:.2f} GB")

    # Cleanup
    del model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_benchmark_suite():
    """Run complete benchmark suite"""

    print("\n" + "="*60)
    print("COLOSSAL-AI SPEED BENCHMARK")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nGPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")
    else:
        print("\nWARNING: No GPU available, running on CPU")

    # Common settings
    seq_len = 256
    num_steps = 50
    warmup_steps = 5

    # Test configurations
    configs = []

    # 1. Baseline - No optimizations
    baseline_config = ColossalAIMoEConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_experts=4,
        num_experts_per_token=2,
        use_flash_attention=False,  # Disabled
        use_colossal_linear=False,  # Standard PyTorch
        use_vocab_parallel=False,
    )
    configs.append(("Baseline (No Optimizations)", baseline_config, 16))

    # 2. Flash Attention only
    flash_config = ColossalAIMoEConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_experts=4,
        num_experts_per_token=2,
        use_flash_attention=True,  # Enabled
        use_colossal_linear=False,
        use_vocab_parallel=False,
    )
    configs.append(("Flash Attention Only", flash_config, 24))

    # 3. Colossal-AI Basic
    colossal_basic = ColossalAIMoEConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_experts=4,
        num_experts_per_token=2,
        use_flash_attention=True,
        use_colossal_linear=True,  # Colossal-AI layers
        use_vocab_parallel=True,  # Vocab parallelism
        tensor_parallel_size=1,
        zero_stage=1,  # ZeRO-1
    )
    configs.append(("Colossal-AI Basic (ZeRO-1)", colossal_basic, 32))

    # 4. Colossal-AI Optimized
    colossal_optimized = ColossalAIMoEConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_experts=4,
        num_experts_per_token=2,
        use_flash_attention=True,
        use_colossal_linear=True,
        use_vocab_parallel=True,
        tensor_parallel_size=1,
        zero_stage=1,
        enable_jit_fused=True,  # JIT compilation
        enable_fused_normalization=True,  # Fused ops
        enable_xformers=True,  # xFormers
    )
    configs.append(("Colossal-AI Optimized (All Features)", colossal_optimized, 48))

    # 5. Ultra-Fast Configuration
    ultrafast_config = ColossalAIMoEConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_experts=4,
        num_experts_per_token=2,
        use_flash_attention=True,
        use_colossal_linear=True,
        use_vocab_parallel=True,
        dropout=0.0,  # No dropout for speed
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_parallel_size=1,
        zero_stage=1,
        enable_jit_fused=True,
        enable_fused_normalization=True,
        enable_xformers=True,
        enable_cuda_graph=True,  # CUDA graphs
        enable_tensor_core=True,  # Tensor cores
        enable_tf32=True,  # TF32
    )
    configs.append(("Ultra-Fast (Maximum Speed)", ultrafast_config, 64))

    # Run benchmarks
    results = []
    for config_name, model_config, batch_size in configs:
        try:
            result = benchmark_configuration(
                config_name=config_name,
                model_config=model_config,
                batch_size=batch_size,
                seq_len=seq_len,
                num_steps=num_steps,
                warmup_steps=warmup_steps,
            )
            results.append(result)
        except Exception as e:
            print(f"\nFailed to benchmark {config_name}: {e}")
            continue

    # Print comparison table
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)

    if results:
        # Find baseline for speedup calculation
        baseline_result = results[0] if results else None

        # Table header
        print(f"\n{'Configuration':<40} {'Steps/s':<12} {'ms/step':<12} {'Tokens/s':<15} {'Memory GB':<12} {'Speedup':<10}")
        print("-" * 120)

        # Table rows
        for result in results:
            speedup = result.steps_per_second / baseline_result.steps_per_second if baseline_result else 1.0

            print(f"{result.config_name:<40} "
                  f"{result.steps_per_second:<12.2f} "
                  f"{result.time_per_step_ms:<12.2f} "
                  f"{result.tokens_per_second:<15,.0f} "
                  f"{result.memory_used_gb:<12.2f} "
                  f"{speedup:<10.2f}x")

        # Best configuration
        best_result = max(results, key=lambda x: x.steps_per_second)
        print(f"\n🏆 Fastest Configuration: {best_result.config_name}")
        print(f"   Achieved {best_result.steps_per_second:.2f} steps/second")

        # Calculate improvement
        if baseline_result:
            best_speedup = best_result.steps_per_second / baseline_result.steps_per_second
            print(f"   {best_speedup:.2f}x faster than baseline")

            # Memory efficiency
            memory_ratio = baseline_result.memory_used_gb / best_result.memory_used_gb if best_result.memory_used_gb > 0 else 1
            print(f"   {memory_ratio:.2f}x more memory efficient")

            # Throughput improvement
            throughput_improvement = best_result.tokens_per_second / baseline_result.tokens_per_second
            print(f"   {throughput_improvement:.2f}x higher throughput")

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


def quick_speed_test():
    """Quick speed test with current configuration"""
    print("\n" + "="*60)
    print("QUICK SPEED TEST")
    print("="*60)

    # Simple config for quick test
    config = ColossalAIMoEConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        num_experts=2,
        num_experts_per_token=1,
        use_flash_attention=True,
    )

    batch_size = 32
    seq_len = 128
    num_steps = 20

    result = benchmark_configuration(
        "Quick Test",
        config,
        batch_size,
        seq_len,
        num_steps,
        warmup_steps=2,
    )

    print(f"\n✓ Quick test complete: {result.steps_per_second:.2f} steps/second")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Colossal-AI Speed")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")

    args = parser.parse_args()

    if args.quick:
        quick_speed_test()
    else:
        run_benchmark_suite()