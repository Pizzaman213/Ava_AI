"""
Benchmarks for data pipeline optimization.

Measures performance of key data loading functions before/after optimization.
Run with: python -m ava.data.benchmarks

Usage:
    from ava.data.benchmarks import run_all_benchmarks, benchmark_packing

    # Run all benchmarks
    results = run_all_benchmarks()

    # Run specific benchmark
    packing_results = benchmark_packing(num_sequences=1000)
"""

import logging
import os
import random
import time
from typing import Dict, List, Optional, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _generate_test_sequences(
    num_sequences: int,
    min_length: int = 50,
    max_length: int = 500,
    vocab_size: int = 50680,
    seed: int = 42,
) -> tuple[List[torch.Tensor], List[int]]:
    """Generate random sequences for benchmarking."""
    random.seed(seed)
    np.random.seed(seed)

    sequences = []
    lengths = []

    for _ in range(num_sequences):
        length = random.randint(min_length, max_length)
        seq = torch.randint(0, vocab_size, (length,), dtype=torch.long)
        sequences.append(seq)
        lengths.append(length)

    return sequences, lengths


def benchmark_packing(
    num_sequences: int = 1000,
    max_length: int = 2048,
    num_iterations: int = 20,
    warmup_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark _pack_sequences_greedy function.

    Args:
        num_sequences: Number of sequences per benchmark run
        max_length: Maximum packed sequence length
        num_iterations: Number of timed iterations
        warmup_iterations: Warmup iterations (not timed)

    Returns:
        Dict with timing statistics
    """
    from .packing import SequencePackingCollator

    # Generate test data
    sequences, lengths = _generate_test_sequences(
        num_sequences=num_sequences,
        min_length=50,
        max_length=min(500, max_length // 2),
    )

    collator = SequencePackingCollator(
        max_length=max_length,
        pad_token_id=0,
        eos_token_id=1,
        sort_by_length=True,
    )

    # Warmup
    for _ in range(warmup_iterations):
        collator._pack_sequences_greedy(sequences, lengths)

    # Timed runs
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        packed, doc_ids = collator._pack_sequences_greedy(sequences, lengths)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]

    return {
        'function': '_pack_sequences_greedy',
        'num_sequences': num_sequences,
        'max_length': max_length,
        'num_packed': len(packed),
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'p50_ms': np.percentile(times_ms, 50),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99),
        'throughput_seqs_per_sec': num_sequences / np.mean(times),
    }


def benchmark_pack_assignment_numba(
    num_sequences: int = 1000,
    max_length: int = 2048,
    num_iterations: int = 20,
    warmup_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Benchmark the Numba-optimized pack assignment function.

    Returns empty dict if Numba is not available.
    """
    try:
        from .packing import _compute_pack_assignments_numba, NUMBA_AVAILABLE
        if not NUMBA_AVAILABLE:
            return {'error': 'Numba not available'}
    except ImportError:
        return {'error': 'Numba pack assignment not found'}

    # Generate test lengths
    random.seed(42)
    lengths = np.array([random.randint(50, 500) for _ in range(num_sequences)], dtype=np.int64)

    # Warmup (triggers JIT compilation)
    for _ in range(warmup_iterations):
        _compute_pack_assignments_numba(lengths, max_length)

    # Timed runs
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        pack_ids = _compute_pack_assignments_numba(lengths, max_length)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]
    num_packs = int(pack_ids.max()) + 1 if len(pack_ids) > 0 else 0

    return {
        'function': '_compute_pack_assignments_numba',
        'num_sequences': num_sequences,
        'max_length': max_length,
        'num_packs': num_packs,
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'p50_ms': np.percentile(times_ms, 50),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99),
        'throughput_seqs_per_sec': num_sequences / np.mean(times),
    }


def benchmark_binned_sampler(
    num_samples: int = 100000,
    batch_size: int = 32,
    num_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Benchmark LengthBinnedSampler iteration.

    Args:
        num_samples: Total samples to shuffle
        batch_size: Batch size for sampler
        num_iterations: Number of full iterations to time

    Returns:
        Dict with timing statistics
    """
    from .indexed import LengthBinnedSampler

    # Generate random lengths
    random.seed(42)
    lengths = [random.randint(50, 2000) for _ in range(num_samples)]

    sampler = LengthBinnedSampler(
        lengths=lengths,
        batch_size=batch_size,
        num_bins=8,
        drop_last=True,
        seed=42,
    )

    # Time full iterations
    times = []
    for epoch in range(num_iterations):
        sampler.set_epoch(epoch)
        start = time.perf_counter()
        indices = list(sampler)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]

    return {
        'function': 'LengthBinnedSampler.__iter__',
        'num_samples': num_samples,
        'batch_size': batch_size,
        'indices_yielded': len(indices),
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'throughput_samples_per_sec': num_samples / np.mean(times),
    }


def benchmark_collation(
    batch_size: int = 32,
    max_length: int = 2048,
    num_iterations: int = 50,
    warmup_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Benchmark DynamicPaddingCollator.

    Args:
        batch_size: Number of samples per batch
        max_length: Maximum sequence length
        num_iterations: Timed iterations
        warmup_iterations: Warmup iterations

    Returns:
        Dict with timing statistics
    """
    from .collators import DynamicPaddingCollator

    # Generate batch of samples
    random.seed(42)
    batch = []
    for _ in range(batch_size):
        length = random.randint(100, max_length // 2)
        sample = {
            'input_ids': torch.randint(0, 50680, (length,)),
            'attention_mask': torch.ones(length),
            'labels': torch.randint(0, 50680, (length,)),
            'length': length,
        }
        batch.append(sample)

    collator = DynamicPaddingCollator(pad_token_id=0, max_length=max_length)

    # Warmup
    for _ in range(warmup_iterations):
        collator(batch)

    # Timed runs
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = collator(batch)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]

    return {
        'function': 'DynamicPaddingCollator.__call__',
        'batch_size': batch_size,
        'max_length': max_length,
        'output_shape': list(result['input_ids'].shape),
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'p50_ms': np.percentile(times_ms, 50),
        'p95_ms': np.percentile(times_ms, 95),
        'throughput_batches_per_sec': 1.0 / np.mean(times),
    }


def run_all_benchmarks(verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Run all benchmarks and return results.

    Args:
        verbose: Print results as they complete

    Returns:
        Dict mapping benchmark name to results
    """
    results = {}

    benchmarks = [
        ('packing_1k', lambda: benchmark_packing(num_sequences=1000)),
        ('packing_10k', lambda: benchmark_packing(num_sequences=10000)),
        ('packing_numba_1k', lambda: benchmark_pack_assignment_numba(num_sequences=1000)),
        ('packing_numba_10k', lambda: benchmark_pack_assignment_numba(num_sequences=10000)),
        ('binned_sampler_100k', lambda: benchmark_binned_sampler(num_samples=100000)),
        ('collation_bs32', lambda: benchmark_collation(batch_size=32)),
        ('collation_bs64', lambda: benchmark_collation(batch_size=64)),
    ]

    for name, bench_fn in benchmarks:
        if verbose:
            print(f"Running {name}...", end=" ", flush=True)

        try:
            result = bench_fn()
            results[name] = result

            if verbose:
                if 'error' in result:
                    print(f"SKIPPED ({result['error']})")
                elif 'mean_ms' in result:
                    print(f"{result['mean_ms']:.2f} ms")
                else:
                    print("done")
        except Exception as e:
            results[name] = {'error': str(e)}
            if verbose:
                print(f"ERROR: {e}")

    return results


def print_benchmark_report(results: Dict[str, Dict[str, Any]]):
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("BENCHMARK REPORT")
    print("=" * 70)

    for name, result in results.items():
        print(f"\n{name}:")
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            continue

        for key, value in result.items():
            if key == 'function':
                continue
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)

    # Print comparison if both Python and Numba results exist
    if 'packing_1k' in results and 'packing_numba_1k' in results:
        py_result = results['packing_1k']
        numba_result = results['packing_numba_1k']

        if 'mean_ms' in py_result and 'mean_ms' in numba_result:
            speedup = py_result['mean_ms'] / numba_result['mean_ms']
            print(f"\nNumba speedup (1k seqs): {speedup:.2f}x")

    if 'packing_10k' in results and 'packing_numba_10k' in results:
        py_result = results['packing_10k']
        numba_result = results['packing_numba_10k']

        if 'mean_ms' in py_result and 'mean_ms' in numba_result:
            speedup = py_result['mean_ms'] / numba_result['mean_ms']
            print(f"Numba speedup (10k seqs): {speedup:.2f}x")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    results = run_all_benchmarks(verbose=True)
    print_benchmark_report(results)
