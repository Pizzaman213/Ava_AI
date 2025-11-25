#!/usr/bin/env python3
"""
Benchmark script to test Phase 1 optimizations
Compares performance before and after enabling optimizations
"""

import time
import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def benchmark_training_step(config_path: str, num_steps: int = 100):
    """
    Run a quick benchmark of training performance

    Args:
        config_path: Path to config file
        num_steps: Number of steps to benchmark

    Returns:
        Dict with benchmark results
    """
    from src.Ava.training.train.trainer import Trainer
    from src.Ava.config.yaml_loader import load_config

    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_path}")
    print(f"{'='*60}\n")

    # Load config
    config = load_config(config_path)

    # Override some settings for benchmarking
    config['training']['max_steps'] = num_steps
    config['training']['eval_steps'] = num_steps + 1  # Disable eval
    config['training']['save_steps'] = num_steps + 1  # Disable saving
    config['training']['logging_steps'] = 10
    config['wandb']['use_wandb'] = False

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Run training
    trainer = Trainer(config)

    # Time the training
    start_time = time.time()
    start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    # Calculate metrics
    elapsed_time = end_time - start_time
    steps_per_second = num_steps / elapsed_time
    seconds_per_step = elapsed_time / num_steps

    results = {
        'config': config_path,
        'total_time_sec': elapsed_time,
        'steps_per_second': steps_per_second,
        'seconds_per_step': seconds_per_step,
        'peak_memory_gb': peak_mem / 1e9,
        'avg_memory_gb': start_mem / 1e9,
    }

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Steps per second: {steps_per_second:.3f}")
    print(f"Seconds per step: {seconds_per_step:.3f}")
    print(f"Peak memory: {peak_mem/1e9:.2f} GB")
    print(f"{'='*60}\n")

    return results


def compare_configs(baseline_config: str, optimized_config: str, num_steps: int = 100):
    """
    Compare two configurations and report speedup

    Args:
        baseline_config: Path to baseline config
        optimized_config: Path to optimized config
        num_steps: Number of steps to benchmark
    """
    print("\n" + "="*80)
    print("PHASE 1 OPTIMIZATION BENCHMARK")
    print("="*80)
    print(f"Comparing baseline vs optimized configurations")
    print(f"Running {num_steps} training steps each\n")

    # Run baseline
    print("Running BASELINE configuration...")
    baseline_results = benchmark_training_step(baseline_config, num_steps)

    # Clear everything
    torch.cuda.empty_cache()
    time.sleep(2)

    # Run optimized
    print("\nRunning OPTIMIZED configuration...")
    optimized_results = benchmark_training_step(optimized_config, num_steps)

    # Calculate speedup
    speedup = optimized_results['steps_per_second'] / baseline_results['steps_per_second']
    time_reduction = (1 - optimized_results['seconds_per_step'] / baseline_results['seconds_per_step']) * 100

    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"Baseline: {baseline_results['steps_per_second']:.3f} steps/sec")
    print(f"Optimized: {optimized_results['steps_per_second']:.3f} steps/sec")
    print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster ({time_reduction:.1f}% time reduction)")
    print(f"\nMemory usage:")
    print(f"  Baseline:  {baseline_results['peak_memory_gb']:.2f} GB")
    print(f"  Optimized: {optimized_results['peak_memory_gb']:.2f} GB")

    memory_change = optimized_results['peak_memory_gb'] - baseline_results['peak_memory_gb']
    if memory_change > 0:
        print(f"  Change:    +{memory_change:.2f} GB ({memory_change/baseline_results['peak_memory_gb']*100:.1f}% increase)")
    else:
        print(f"  Change:    {memory_change:.2f} GB ({abs(memory_change)/baseline_results['peak_memory_gb']*100:.1f}% reduction)")

    print("="*80)

    return {
        'baseline': baseline_results,
        'optimized': optimized_results,
        'speedup': speedup,
        'time_reduction_pct': time_reduction
    }


def quick_test():
    """
    Quick test with just the optimized config to verify it works
    """
    config_path = "code/configs/moe/minimal_working.yaml"
    print(f"\n{'='*80}")
    print("QUICK VERIFICATION TEST")
    print(f"{'='*80}")
    print(f"Testing optimized config with 50 steps...")

    results = benchmark_training_step(config_path, num_steps=50)

    print("\n✅ Configuration test completed successfully!")
    print(f"Performance: {results['steps_per_second']:.3f} steps/second")
    print(f"Memory usage: {results['peak_memory_gb']:.2f} GB")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Phase 1 optimizations")
    parser.add_argument("--mode", choices=["quick", "compare"], default="quick",
                      help="Run quick test or full comparison")
    parser.add_argument("--steps", type=int, default=50,
                      help="Number of training steps to benchmark")
    parser.add_argument("--baseline", type=str,
                      help="Path to baseline config (for compare mode)")
    parser.add_argument("--optimized", type=str,
                      default="code/configs/moe/minimal_working.yaml",
                      help="Path to optimized config")

    args = parser.parse_args()

    try:
        if args.mode == "quick":
            quick_test()
        else:
            if not args.baseline:
                print("ERROR: --baseline config required for compare mode")
                sys.exit(1)
            compare_configs(args.baseline, args.optimized, args.steps)
    except Exception as e:
        print(f"\n❌ Benchmark failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
