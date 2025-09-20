#!/usr/bin/env python3
"""
Comprehensive A100 GPU Benchmark and Profiling Suite

This script provides comprehensive benchmarking and profiling tools specifically
designed for NVIDIA A100 GPUs, measuring all optimization features.

Features:
- Tensor Core utilization benchmarks
- Memory bandwidth measurements
- FlashAttention performance comparison
- Mixed precision speedup analysis
- NVLink communication benchmarks
- CUDA graphs overhead reduction
- torch.compile optimization impact
- Structured sparsity performance
- Multi-GPU scaling analysis

Usage:
    # Run comprehensive benchmark suite
    python a100_benchmark.py --full-benchmark

    # Benchmark specific optimization
    python a100_benchmark.py --benchmark-flash-attention --seq-lengths 512,1024,2048

    # Multi-GPU communication benchmark
    torchrun --nproc_per_node=8 a100_benchmark.py --benchmark-nvlink

    # Memory optimization benchmarks
    python a100_benchmark.py --benchmark-memory --model-sizes small,medium,large

Examples:
    # Quick A100 feature test
    python a100_benchmark.py --quick-test

    # Deep performance analysis
    python a100_benchmark.py --full-benchmark --save-results results/a100_analysis.json

    # Compare optimizations impact
    python a100_benchmark.py --compare-optimizations --iterations 100
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import logging

# Add project root to path
sys.path.append('/project/code')

# Import A100 optimization modules
from src.Ava.optimization.a100_optimizer import A100Optimizer, StructuredSparsityOptimizer, benchmark_a100_optimizations
from src.Ava.optimization.flash_attention_v3 import FlashAttentionV3, benchmark_flash_attention
from src.Ava.optimization.memory_optimizer import A100MemoryOptimizer, profile_memory_usage
from src.Ava.optimization.nvlink_optimizer import NVLinkOptimizer, benchmark_nvlink_communication

# Import test models
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class A100BenchmarkSuite:
    """
    Comprehensive benchmark suite for A100 optimizations.
    """

    def __init__(
        self,
        device: torch.device,
        output_dir: str = "benchmark_results",
        save_plots: bool = True,
        verbose: bool = True
    ):
        """
        Initialize benchmark suite.

        Args:
            device: CUDA device for benchmarking
            output_dir: Directory to save results
            save_plots: Whether to save plots
            verbose: Enable verbose logging
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_plots = save_plots
        self.verbose = verbose

        # Initialize optimizers
        self.a100_optimizer = A100Optimizer(profile_mode=True)
        self.memory_optimizer = A100MemoryOptimizer(profile_memory=True)
        self.sparsity_optimizer = StructuredSparsityOptimizer()

        # Results storage
        self.results = {
            'device_info': self._get_device_info(),
            'benchmarks': {},
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"A100 Benchmark Suite initialized for {self.device}")
        logger.info(f"Results will be saved to: {self.output_dir}")

    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "multiprocessor_count": props.multiprocessor_count,
            "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
            "max_shared_memory_per_multiprocessor": props.max_shared_memory_per_multiprocessor,
            "memory_clock_rate": props.memory_clock_rate,
            "memory_bus_width": props.memory_bus_width,
            "l2_cache_size": props.l2_cache_size,
            "max_threads_per_block": props.max_threads_per_block,
            "warp_size": props.warp_size
        }

    def benchmark_tensor_cores(
        self,
        matrix_sizes: List[Tuple[int, int, int]] = None,
        precisions: List[str] = None,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark Tensor Core performance with different precisions.

        Args:
            matrix_sizes: List of (M, N, K) matrix dimensions
            precisions: List of precision types
            iterations: Number of iterations per test

        Returns:
            Benchmark results
        """
        logger.info("Running Tensor Core benchmarks...")

        if matrix_sizes is None:
            # A100-optimized sizes (multiples of 8 for Tensor Cores)
            matrix_sizes = [
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (8192, 8192, 8192)
            ]

        if precisions is None:
            precisions = ['fp32', 'tf32', 'fp16', 'bf16']

        results = {'matrix_sizes': matrix_sizes, 'results': {}}

        for precision in precisions:
            precision_results = []

            for m, n, k in matrix_sizes:
                # Create test matrices
                if precision == 'fp32':
                    a = torch.randn(m, k, dtype=torch.float32, device=self.device)
                    b = torch.randn(k, n, dtype=torch.float32, device=self.device)
                    torch.backends.cuda.matmul.allow_tf32 = False
                elif precision == 'tf32':
                    a = torch.randn(m, k, dtype=torch.float32, device=self.device)
                    b = torch.randn(k, n, dtype=torch.float32, device=self.device)
                    torch.backends.cuda.matmul.allow_tf32 = True
                elif precision == 'fp16':
                    a = torch.randn(m, k, dtype=torch.float16, device=self.device)
                    b = torch.randn(k, n, dtype=torch.float16, device=self.device)
                elif precision == 'bf16':
                    a = torch.randn(m, k, dtype=torch.bfloat16, device=self.device)
                    b = torch.randn(k, n, dtype=torch.bfloat16, device=self.device)

                # Warmup
                for _ in range(10):
                    _ = torch.matmul(a, b)

                # Benchmark
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(iterations):
                    _ = torch.matmul(a, b)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

                # Calculate TFLOPS
                flops = 2 * m * n * k * iterations  # 2 for multiply-add
                tflops = flops / elapsed_time / 1e12

                precision_results.append({
                    'matrix_size': (m, n, k),
                    'elapsed_time': elapsed_time,
                    'tflops': tflops,
                    'avg_time_per_op': elapsed_time / iterations
                })

                if self.verbose:
                    logger.info(f"  {precision} {m}x{n}x{k}: {tflops:.2f} TFLOPS")

            results['results'][precision] = precision_results

        # Calculate speedups relative to FP32
        if 'fp32' in results['results']:
            fp32_results = {(r['matrix_size']): r['tflops'] for r in results['results']['fp32']}
            for precision in precisions:
                if precision != 'fp32':
                    for result in results['results'][precision]:
                        matrix_size = result['matrix_size']
                        if matrix_size in fp32_results:
                            result['speedup_vs_fp32'] = result['tflops'] / fp32_results[matrix_size]

        self.results['benchmarks']['tensor_cores'] = results
        logger.info("Tensor Core benchmarks completed")
        return results

    def benchmark_mixed_precision(
        self,
        model_configs: List[Dict[str, Any]] = None,
        batch_sizes: List[int] = None,
        sequence_lengths: List[int] = None,
        iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark mixed precision training performance.

        Args:
            model_configs: List of model configurations
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths
            iterations: Number of iterations per test

        Returns:
            Benchmark results
        """
        logger.info("Running mixed precision benchmarks...")

        if model_configs is None:
            model_configs = [
                {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},  # Small
                {'hidden_size': 1024, 'num_layers': 24, 'num_heads': 16},  # Medium
                {'hidden_size': 1536, 'num_layers': 48, 'num_heads': 24}   # Large
            ]

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]

        if sequence_lengths is None:
            sequence_lengths = [128, 256, 512, 1024]

        results = {'configurations': [], 'results': {}}
        precisions = ['fp32', 'fp16', 'bf16']

        for config_idx, config in enumerate(model_configs):
            config_name = f"model_{config_idx}"
            results['configurations'].append({config_name: config})

            # Create model
            model_config = EnhancedMoEConfig(
                vocab_size=50257,
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_layers'],
                num_attention_heads=config['num_heads'],
                intermediate_size=config['hidden_size'] * 4,
                num_experts=8,
                top_k=2
            )

            for precision in precisions:
                precision_results = []

                for batch_size in batch_sizes:
                    for seq_len in sequence_lengths:
                        try:
                            # Create model and optimizer
                            model = EnhancedMoEModel(model_config).to(self.device)

                            if precision == 'fp16':
                                model = model.half()
                            elif precision == 'bf16':
                                model = model.to(torch.bfloat16)

                            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                            # Create sample data
                            input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)
                            labels = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)

                            # Warmup
                            for _ in range(5):
                                with torch.cuda.amp.autocast(enabled=(precision != 'fp32')):
                                    outputs = model(input_ids, labels=labels)
                                    loss = outputs.loss
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()

                            # Benchmark forward + backward
                            torch.cuda.synchronize()
                            start_time = time.time()

                            for _ in range(iterations):
                                with torch.cuda.amp.autocast(enabled=(precision != 'fp32')):
                                    outputs = model(input_ids, labels=labels)
                                    loss = outputs.loss
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()

                            torch.cuda.synchronize()
                            elapsed_time = time.time() - start_time

                            # Memory usage
                            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

                            precision_results.append({
                                'batch_size': batch_size,
                                'sequence_length': seq_len,
                                'elapsed_time': elapsed_time,
                                'avg_time_per_step': elapsed_time / iterations,
                                'steps_per_second': iterations / elapsed_time,
                                'max_memory_gb': max_memory,
                                'parameters': sum(p.numel() for p in model.parameters())
                            })

                            # Clean up
                            del model, optimizer
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()

                            if self.verbose:
                                logger.info(f"  {config_name} {precision} B{batch_size} S{seq_len}: "
                                           f"{iterations/elapsed_time:.1f} steps/s, {max_memory:.1f}GB")

                        except torch.cuda.OutOfMemoryError:
                            logger.warning(f"OOM: {config_name} {precision} B{batch_size} S{seq_len}")
                            continue

                results['results'][f"{config_name}_{precision}"] = precision_results

        self.results['benchmarks']['mixed_precision'] = results
        logger.info("Mixed precision benchmarks completed")
        return results

    def benchmark_flash_attention(
        self,
        sequence_lengths: List[int] = None,
        head_dims: List[int] = None,
        num_heads_list: List[int] = None,
        batch_sizes: List[int] = None,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark FlashAttention v3 performance.

        Args:
            sequence_lengths: List of sequence lengths
            head_dims: List of head dimensions
            num_heads_list: List of number of heads
            batch_sizes: List of batch sizes
            iterations: Number of iterations

        Returns:
            Benchmark results
        """
        logger.info("Running FlashAttention benchmarks...")

        if sequence_lengths is None:
            sequence_lengths = [128, 256, 512, 1024, 2048, 4096]

        if head_dims is None:
            head_dims = [64, 128]

        if num_heads_list is None:
            num_heads_list = [8, 12, 16]

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        results = {'configurations': [], 'results': []}

        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                for num_heads in num_heads_list:
                    for head_dim in head_dims:
                        embed_dim = num_heads * head_dim

                        try:
                            # Standard attention
                            std_attn = nn.MultiheadAttention(
                                embed_dim, num_heads, batch_first=True
                            ).to(self.device)

                            # FlashAttention v3
                            flash_attn = FlashAttentionV3(
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                causal=True,
                                use_tensor_cores=True,
                                profile_mode=True
                            ).to(self.device)

                            # Create test data
                            x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

                            # Benchmark standard attention
                            torch.cuda.synchronize()
                            start_time = time.time()
                            for _ in range(iterations):
                                _ = std_attn(x, x, x)
                            torch.cuda.synchronize()
                            std_time = time.time() - start_time

                            # Benchmark FlashAttention
                            torch.cuda.synchronize()
                            start_time = time.time()
                            for _ in range(iterations):
                                _ = flash_attn(x, x, x)
                            torch.cuda.synchronize()
                            flash_time = time.time() - start_time

                            # Memory usage
                            torch.cuda.reset_peak_memory_stats()
                            _ = std_attn(x, x, x)
                            std_memory = torch.cuda.max_memory_allocated()

                            torch.cuda.reset_peak_memory_stats()
                            _ = flash_attn(x, x, x)
                            flash_memory = torch.cuda.max_memory_allocated()

                            # Calculate metrics
                            speedup = std_time / flash_time
                            memory_savings = (std_memory - flash_memory) / std_memory * 100

                            result = {
                                'batch_size': batch_size,
                                'sequence_length': seq_len,
                                'num_heads': num_heads,
                                'head_dim': head_dim,
                                'embed_dim': embed_dim,
                                'standard_time': std_time,
                                'flash_time': flash_time,
                                'speedup': speedup,
                                'standard_memory_mb': std_memory / (1024 ** 2),
                                'flash_memory_mb': flash_memory / (1024 ** 2),
                                'memory_savings_percent': memory_savings
                            }

                            results['results'].append(result)

                            if self.verbose:
                                logger.info(f"  B{batch_size} S{seq_len} H{num_heads}x{head_dim}: "
                                           f"{speedup:.2f}x speedup, {memory_savings:.1f}% memory saved")

                            # Clean up
                            del std_attn, flash_attn, x
                            torch.cuda.empty_cache()

                        except torch.cuda.OutOfMemoryError:
                            logger.warning(f"OOM: B{batch_size} S{seq_len} H{num_heads}x{head_dim}")
                            continue

        self.results['benchmarks']['flash_attention'] = results
        logger.info("FlashAttention benchmarks completed")
        return results

    def benchmark_memory_optimization(
        self,
        model_sizes: List[str] = None,
        optimization_levels: List[str] = None,
        iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Benchmark memory optimization techniques.

        Args:
            model_sizes: List of model sizes ('small', 'medium', 'large')
            optimization_levels: List of optimization levels
            iterations: Number of iterations

        Returns:
            Benchmark results
        """
        logger.info("Running memory optimization benchmarks...")

        if model_sizes is None:
            model_sizes = ['small', 'medium', 'large']

        if optimization_levels is None:
            optimization_levels = ['none', 'checkpointing', 'offloading', 'full']

        model_configs = {
            'small': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
            'medium': {'hidden_size': 1024, 'num_layers': 24, 'num_heads': 16},
            'large': {'hidden_size': 1536, 'num_layers': 48, 'num_heads': 24}
        }

        results = {'model_configs': model_configs, 'results': []}

        for model_size in model_sizes:
            config = model_configs[model_size]

            for opt_level in optimization_levels:
                try:
                    # Create model config
                    model_config = EnhancedMoEConfig(
                        vocab_size=50257,
                        hidden_size=config['hidden_size'],
                        num_hidden_layers=config['num_layers'],
                        num_attention_heads=config['num_heads'],
                        intermediate_size=config['hidden_size'] * 4,
                        num_experts=8,
                        top_k=2
                    )

                    # Create model
                    model = EnhancedMoEModel(model_config).to(self.device)

                    # Apply optimizations
                    if opt_level != 'none':
                        memory_optimizer = A100MemoryOptimizer(
                            enable_gradient_checkpointing=(opt_level in ['checkpointing', 'full']),
                            enable_cpu_offload=(opt_level in ['offloading', 'full']),
                            checkpoint_policy='selective',
                            profile_memory=True
                        )
                        model = memory_optimizer.optimize_model_memory(model)

                    # Create optimizer
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                    # Test with different batch sizes
                    max_batch_size = 1
                    for batch_size in [1, 2, 4, 8, 16, 32]:
                        try:
                            # Create sample data
                            input_ids = torch.randint(0, 50257, (batch_size, 512), device=self.device)
                            labels = torch.randint(0, 50257, (batch_size, 512), device=self.device)

                            # Test forward + backward
                            torch.cuda.reset_peak_memory_stats()

                            outputs = model(input_ids, labels=labels)
                            loss = outputs.loss
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                            max_batch_size = batch_size
                            torch.cuda.empty_cache()

                        except torch.cuda.OutOfMemoryError:
                            break

                    # Measure training time
                    batch_size = max_batch_size
                    input_ids = torch.randint(0, 50257, (batch_size, 512), device=self.device)
                    labels = torch.randint(0, 50257, (batch_size, 512), device=self.device)

                    torch.cuda.synchronize()
                    start_time = time.time()

                    for _ in range(iterations):
                        outputs = model(input_ids, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start_time

                    # Get memory stats
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

                    result = {
                        'model_size': model_size,
                        'optimization_level': opt_level,
                        'max_batch_size': max_batch_size,
                        'peak_memory_gb': peak_memory,
                        'training_time': elapsed_time,
                        'steps_per_second': iterations / elapsed_time,
                        'parameters': sum(p.numel() for p in model.parameters())
                    }

                    results['results'].append(result)

                    if self.verbose:
                        logger.info(f"  {model_size} {opt_level}: "
                                   f"batch_size={max_batch_size}, "
                                   f"memory={peak_memory:.1f}GB, "
                                   f"speed={iterations/elapsed_time:.1f} steps/s")

                    # Clean up
                    del model, optimizer
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                except Exception as e:
                    logger.error(f"Error in {model_size} {opt_level}: {e}")
                    continue

        self.results['benchmarks']['memory_optimization'] = results
        logger.info("Memory optimization benchmarks completed")
        return results

    def benchmark_structured_sparsity(
        self,
        model_sizes: List[str] = None,
        sparsity_levels: List[float] = None,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark structured sparsity performance.

        Args:
            model_sizes: List of model sizes
            sparsity_levels: List of sparsity levels (only 0.5 gets hardware acceleration)
            iterations: Number of iterations

        Returns:
            Benchmark results
        """
        logger.info("Running structured sparsity benchmarks...")

        if model_sizes is None:
            model_sizes = ['small', 'medium']

        if sparsity_levels is None:
            sparsity_levels = [0.0, 0.5]  # 0.5 is 2:4 sparsity for A100

        model_configs = {
            'small': {'hidden_size': 768, 'num_layers': 12},
            'medium': {'hidden_size': 1024, 'num_layers': 24}
        }

        results = {'model_configs': model_configs, 'results': []}

        for model_size in model_sizes:
            config = model_configs[model_size]

            for sparsity_level in sparsity_levels:
                try:
                    # Create simple model for testing
                    model = nn.Sequential(
                        nn.Linear(config['hidden_size'], config['hidden_size'] * 4),
                        nn.GELU(),
                        nn.Linear(config['hidden_size'] * 4, config['hidden_size'])
                    ).to(self.device)

                    # Apply sparsity
                    if sparsity_level > 0:
                        sparsity_optimizer = StructuredSparsityOptimizer(sparsity_level=sparsity_level)
                        sparsity_info = sparsity_optimizer.sparsify_model(model)
                    else:
                        sparsity_info = []

                    # Create test data
                    batch_size = 32
                    x = torch.randn(batch_size, 512, config['hidden_size'], device=self.device)

                    # Warmup
                    for _ in range(10):
                        _ = model(x)

                    # Benchmark forward pass
                    torch.cuda.synchronize()
                    start_time = time.time()

                    for _ in range(iterations):
                        _ = model(x)

                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start_time

                    # Calculate actual sparsity
                    total_params = sum(p.numel() for p in model.parameters())
                    zero_params = sum((p == 0).sum().item() for p in model.parameters())
                    actual_sparsity = zero_params / total_params

                    result = {
                        'model_size': model_size,
                        'target_sparsity': sparsity_level,
                        'actual_sparsity': actual_sparsity,
                        'elapsed_time': elapsed_time,
                        'throughput': iterations / elapsed_time,
                        'parameters': total_params,
                        'zero_parameters': zero_params,
                        'layers_sparsified': len(sparsity_info)
                    }

                    results['results'].append(result)

                    if self.verbose:
                        logger.info(f"  {model_size} sparsity={sparsity_level:.1f}: "
                                   f"{iterations/elapsed_time:.1f} iter/s, "
                                   f"actual_sparsity={actual_sparsity:.1%}")

                    # Clean up
                    del model, x
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error in {model_size} sparsity={sparsity_level}: {e}")
                    continue

        # Calculate speedups relative to dense models
        dense_results = {r['model_size']: r['throughput'] for r in results['results'] if r['target_sparsity'] == 0.0}
        for result in results['results']:
            if result['target_sparsity'] > 0.0:
                model_size = result['model_size']
                if model_size in dense_results:
                    result['speedup_vs_dense'] = result['throughput'] / dense_results[model_size]

        self.results['benchmarks']['structured_sparsity'] = results
        logger.info("Structured sparsity benchmarks completed")
        return results

    def benchmark_compilation(
        self,
        model_sizes: List[str] = None,
        compile_modes: List[str] = None,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark torch.compile performance.

        Args:
            model_sizes: List of model sizes
            compile_modes: List of compilation modes
            iterations: Number of iterations

        Returns:
            Benchmark results
        """
        logger.info("Running torch.compile benchmarks...")

        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available, skipping compilation benchmarks")
            return {}

        if model_sizes is None:
            model_sizes = ['small', 'medium']

        if compile_modes is None:
            compile_modes = ['none', 'default', 'reduce-overhead', 'max-autotune']

        model_configs = {
            'small': {'hidden_size': 768, 'num_layers': 6},
            'medium': {'hidden_size': 1024, 'num_layers': 12}
        }

        results = {'model_configs': model_configs, 'results': []}

        for model_size in model_sizes:
            config = model_configs[model_size]

            for compile_mode in compile_modes:
                try:
                    # Create model config
                    model_config = EnhancedMoEConfig(
                        vocab_size=50257,
                        hidden_size=config['hidden_size'],
                        num_hidden_layers=config['num_layers'],
                        num_attention_heads=config['hidden_size'] // 64,
                        intermediate_size=config['hidden_size'] * 4,
                        num_experts=4,
                        top_k=2
                    )

                    # Create model
                    model = EnhancedMoEModel(model_config).to(self.device)

                    # Apply compilation
                    if compile_mode != 'none':
                        model = torch.compile(model, mode=compile_mode)

                    # Create test data
                    batch_size = 4
                    seq_len = 512
                    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)

                    # Warmup (important for compilation)
                    for _ in range(10):
                        _ = model(input_ids)

                    # Benchmark
                    torch.cuda.synchronize()
                    start_time = time.time()

                    for _ in range(iterations):
                        _ = model(input_ids)

                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start_time

                    result = {
                        'model_size': model_size,
                        'compile_mode': compile_mode,
                        'elapsed_time': elapsed_time,
                        'throughput': iterations / elapsed_time,
                        'avg_time_per_forward': elapsed_time / iterations
                    }

                    results['results'].append(result)

                    if self.verbose:
                        logger.info(f"  {model_size} {compile_mode}: "
                                   f"{iterations/elapsed_time:.1f} iter/s")

                    # Clean up
                    del model, input_ids
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error in {model_size} {compile_mode}: {e}")
                    continue

        # Calculate speedups relative to uncompiled models
        uncompiled_results = {r['model_size']: r['throughput'] for r in results['results'] if r['compile_mode'] == 'none'}
        for result in results['results']:
            if result['compile_mode'] != 'none':
                model_size = result['model_size']
                if model_size in uncompiled_results:
                    result['speedup_vs_uncompiled'] = result['throughput'] / uncompiled_results[model_size]

        self.results['benchmarks']['torch_compile'] = results
        logger.info("torch.compile benchmarks completed")
        return results

    def create_summary_report(self) -> Dict[str, Any]:
        """Create a comprehensive summary report."""
        logger.info("Creating summary report...")

        summary = {
            'device_info': self.results['device_info'],
            'benchmark_summary': {},
            'recommendations': [],
            'timestamp': self.results['timestamp']
        }

        # Tensor Core summary
        if 'tensor_cores' in self.results['benchmarks']:
            tc_results = self.results['benchmarks']['tensor_cores']['results']
            if 'tf32' in tc_results and 'fp32' in tc_results:
                # Find best TF32 vs FP32 speedup
                tf32_tflops = [r['tflops'] for r in tc_results['tf32']]
                fp32_tflops = [r['tflops'] for r in tc_results['fp32']]
                avg_speedup = np.mean([tf / fp for tf, fp in zip(tf32_tflops, fp32_tflops)])

                summary['benchmark_summary']['tensor_cores'] = {
                    'tf32_vs_fp32_speedup': avg_speedup,
                    'max_tf32_tflops': max(tf32_tflops),
                    'max_fp32_tflops': max(fp32_tflops)
                }

                if avg_speedup > 5.0:
                    summary['recommendations'].append("Excellent TF32 performance - use TF32 for all training")
                elif avg_speedup > 2.0:
                    summary['recommendations'].append("Good TF32 speedup - enable TF32 for faster training")

        # FlashAttention summary
        if 'flash_attention' in self.results['benchmarks']:
            fa_results = self.results['benchmarks']['flash_attention']['results']
            if fa_results:
                speedups = [r['speedup'] for r in fa_results]
                memory_savings = [r['memory_savings_percent'] for r in fa_results]

                summary['benchmark_summary']['flash_attention'] = {
                    'avg_speedup': np.mean(speedups),
                    'max_speedup': max(speedups),
                    'avg_memory_savings': np.mean(memory_savings),
                    'max_memory_savings': max(memory_savings)
                }

                if np.mean(speedups) > 2.0:
                    summary['recommendations'].append("FlashAttention provides excellent speedup - enable for all attention layers")

        # Mixed precision summary
        if 'mixed_precision' in self.results['benchmarks']:
            mp_results = self.results['benchmarks']['mixed_precision']['results']

            # Find BF16 vs FP32 comparison
            bf16_results = [k for k in mp_results.keys() if 'bf16' in k]
            fp32_results = [k for k in mp_results.keys() if 'fp32' in k]

            if bf16_results and fp32_results:
                summary['recommendations'].append("Use BF16 mixed precision for optimal A100 performance")

        # Compilation summary
        if 'torch_compile' in self.results['benchmarks']:
            tc_results = self.results['benchmarks']['torch_compile']['results']
            max_autotune_results = [r for r in tc_results if r['compile_mode'] == 'max-autotune']

            if max_autotune_results:
                speedups = [r.get('speedup_vs_uncompiled', 1.0) for r in max_autotune_results]
                avg_speedup = np.mean(speedups)

                summary['benchmark_summary']['torch_compile'] = {
                    'max_autotune_speedup': avg_speedup
                }

                if avg_speedup > 1.5:
                    summary['recommendations'].append("torch.compile with max-autotune provides good speedup")

        # General A100 recommendations
        summary['recommendations'].extend([
            "Use structured sparsity (2:4) for 2x theoretical speedup on compatible layers",
            "Enable gradient checkpointing for memory-intensive training",
            "Use NVLink optimization for multi-GPU training",
            "Configure CUDA memory pools for reduced allocation overhead"
        ])

        return summary

    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"a100_benchmark_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Create summary report
        summary = self.create_summary_report()
        full_results = {
            'summary': summary,
            'detailed_results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

        # Also save a human-readable summary
        summary_file = filepath.with_suffix('.txt')
        with open(summary_file, 'w') as f:
            f.write("A100 GPU Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Device: {summary['device_info']['name']}\n")
            f.write(f"Memory: {summary['device_info']['total_memory_gb']:.1f} GB\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")

            f.write("Benchmark Results:\n")
            f.write("-" * 20 + "\n")
            for key, value in summary['benchmark_summary'].items():
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
                f.write("\n")

            f.write("Recommendations:\n")
            f.write("-" * 20 + "\n")
            for rec in summary['recommendations']:
                f.write(f"• {rec}\n")

        logger.info(f"Summary saved to {summary_file}")

    def run_full_benchmark(self, quick_mode: bool = False):
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive A100 benchmark suite...")

        iterations = 50 if quick_mode else 100

        # Run all benchmarks
        self.benchmark_tensor_cores(iterations=iterations)
        self.benchmark_mixed_precision(iterations=iterations//2)
        self.benchmark_flash_attention(iterations=iterations)
        self.benchmark_memory_optimization(iterations=iterations//5)
        self.benchmark_structured_sparsity(iterations=iterations)
        self.benchmark_compilation(iterations=iterations)

        # Save results
        self.save_results()

        logger.info("Comprehensive benchmark completed!")

    def run_quick_test(self):
        """Run quick A100 feature test."""
        logger.info("Running quick A100 feature test...")

        # Quick tensor core test
        self.benchmark_tensor_cores(
            matrix_sizes=[(1024, 1024, 1024)],
            precisions=['fp32', 'tf32', 'bf16'],
            iterations=20
        )

        # Quick FlashAttention test
        self.benchmark_flash_attention(
            sequence_lengths=[512, 1024],
            head_dims=[64],
            num_heads_list=[12],
            batch_sizes=[4],
            iterations=20
        )

        # Save results
        self.save_results("quick_test_results.json")

        logger.info("Quick test completed!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='A100 GPU Benchmark Suite')

    parser.add_argument('--full-benchmark', action='store_true',
                       help='Run comprehensive benchmark suite')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick A100 feature test')
    parser.add_argument('--benchmark-tensor-cores', action='store_true',
                       help='Benchmark Tensor Core performance')
    parser.add_argument('--benchmark-mixed-precision', action='store_true',
                       help='Benchmark mixed precision training')
    parser.add_argument('--benchmark-flash-attention', action='store_true',
                       help='Benchmark FlashAttention performance')
    parser.add_argument('--benchmark-memory', action='store_true',
                       help='Benchmark memory optimizations')
    parser.add_argument('--benchmark-sparsity', action='store_true',
                       help='Benchmark structured sparsity')
    parser.add_argument('--benchmark-compilation', action='store_true',
                       help='Benchmark torch.compile performance')
    parser.add_argument('--benchmark-nvlink', action='store_true',
                       help='Benchmark NVLink communication')

    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per benchmark')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Filename to save results')
    parser.add_argument('--model-sizes', type=str, default='small,medium',
                       help='Comma-separated model sizes to test')
    parser.add_argument('--seq-lengths', type=str, default='512,1024,2048',
                       help='Comma-separated sequence lengths to test')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. A100 benchmarks require CUDA.")
        return

    device = torch.device('cuda')

    # Initialize benchmark suite
    benchmark_suite = A100BenchmarkSuite(
        device=device,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    try:
        # Run requested benchmarks
        if args.full_benchmark:
            benchmark_suite.run_full_benchmark()
        elif args.quick_test:
            benchmark_suite.run_quick_test()
        else:
            # Run individual benchmarks
            if args.benchmark_tensor_cores:
                benchmark_suite.benchmark_tensor_cores(iterations=args.iterations)

            if args.benchmark_mixed_precision:
                model_sizes = args.model_sizes.split(',') if args.model_sizes else None
                benchmark_suite.benchmark_mixed_precision(iterations=args.iterations)

            if args.benchmark_flash_attention:
                seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
                benchmark_suite.benchmark_flash_attention(
                    sequence_lengths=seq_lengths,
                    iterations=args.iterations
                )

            if args.benchmark_memory:
                model_sizes = args.model_sizes.split(',') if args.model_sizes else None
                benchmark_suite.benchmark_memory_optimization(
                    model_sizes=model_sizes,
                    iterations=args.iterations//5
                )

            if args.benchmark_sparsity:
                model_sizes = args.model_sizes.split(',') if args.model_sizes else None
                benchmark_suite.benchmark_structured_sparsity(
                    model_sizes=model_sizes,
                    iterations=args.iterations
                )

            if args.benchmark_compilation:
                model_sizes = args.model_sizes.split(',') if args.model_sizes else None
                benchmark_suite.benchmark_compilation(
                    model_sizes=model_sizes,
                    iterations=args.iterations
                )

            if args.benchmark_nvlink and dist.is_available():
                # Run NVLink benchmarks (requires distributed setup)
                nvlink_results = benchmark_nvlink_communication()
                logger.info(f"NVLink benchmark results: {nvlink_results}")

            # Save results if any benchmarks were run
            if any([args.benchmark_tensor_cores, args.benchmark_mixed_precision,
                   args.benchmark_flash_attention, args.benchmark_memory,
                   args.benchmark_sparsity, args.benchmark_compilation]):
                benchmark_suite.save_results(args.save_results)

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}", exc_info=True)


if __name__ == '__main__':
    main()