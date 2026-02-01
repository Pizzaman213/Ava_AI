#!/usr/bin/env python3
"""
Benchmark script for data pipeline optimizations.
Measures memory usage, throughput, and data loading speed.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import tracemalloc

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def benchmark_numpy_index_memory():
    """Benchmark memory usage of NumPy arrays vs List[Tuple]."""
    print("\n" + "="*60)
    print("BENCHMARK 1: Index Memory Usage (NumPy vs List[Tuple])")
    print("="*60)

    num_samples = 1_000_000

    # Old approach: List[Tuple[int, int]]
    tracemalloc.start()
    old_index = [(i // 1000, i % 1000) for i in range(num_samples)]
    old_current, old_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del old_index
    gc.collect()

    # New approach: NumPy int32 arrays
    tracemalloc.start()
    file_indices = np.arange(num_samples, dtype=np.int32) // 1000
    row_indices = np.arange(num_samples, dtype=np.int32) % 1000
    new_current, new_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Samples: {num_samples:,}")
    print(f"  List[Tuple] peak memory: {old_peak / 1e6:.2f} MB")
    print(f"  NumPy int32 peak memory: {new_peak / 1e6:.2f} MB")
    print(f"  Memory reduction: {(1 - new_peak / old_peak) * 100:.1f}%")

    return {
        "old_memory_mb": old_peak / 1e6,
        "new_memory_mb": new_peak / 1e6,
        "reduction_pct": (1 - new_peak / old_peak) * 100
    }


def benchmark_heap_vs_linear_rank_selection():
    """Benchmark min-heap vs linear scan for rank selection."""
    import heapq

    print("\n" + "="*60)
    print("BENCHMARK 2: Rank Selection (Min-Heap vs Linear Scan)")
    print("="*60)

    world_size = 8
    iterations = 100_000

    # Warmup
    for _ in range(1000):
        pass

    # Linear scan (old)
    token_counts = [0] * world_size
    start = time.perf_counter()
    for i in range(iterations):
        sample_tokens = 100 + (i % 500)
        best_rank = min(range(world_size), key=lambda r: token_counts[r])
        token_counts[best_rank] += sample_tokens
    linear_time = time.perf_counter() - start

    # Min-heap (new)
    rank_heap = [(0, r) for r in range(world_size)]
    heapq.heapify(rank_heap)
    start = time.perf_counter()
    for i in range(iterations):
        sample_tokens = 100 + (i % 500)
        tokens, rank = heapq.heappop(rank_heap)
        heapq.heappush(rank_heap, (tokens + sample_tokens, rank))
    heap_time = time.perf_counter() - start

    print(f"  World size: {world_size}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Linear scan: {linear_time*1000:.2f} ms")
    print(f"  Min-heap: {heap_time*1000:.2f} ms")
    print(f"  Speedup: {linear_time/heap_time:.2f}x")

    return {
        "linear_ms": linear_time * 1000,
        "heap_ms": heap_time * 1000,
        "speedup": linear_time / heap_time
    }


def benchmark_lru_eviction():
    """Benchmark pure LRU vs LFU+LRU cache eviction."""
    import math
    from collections import OrderedDict

    print("\n" + "="*60)
    print("BENCHMARK 3: Cache Eviction (Pure LRU vs LFU+LRU)")
    print("="*60)

    cache_size = 100
    iterations = 50_000

    # Simulate cache entries: (key, access_count, last_access_time)
    current_time = time.time()
    accesses = [(i, np.random.randint(1, 100), current_time - np.random.random() * 3600)
                for i in range(cache_size)]

    # Old LFU+LRU scoring
    start = time.perf_counter()
    for _ in range(iterations):
        worst = min(range(len(accesses)),
                    key=lambda i: math.log1p(accesses[i][1]) - (current_time - accesses[i][2]) / 3600)
    lfu_lru_time = time.perf_counter() - start

    # New pure LRU (just pop first)
    ordered = OrderedDict([(i, accesses[i]) for i in range(cache_size)])
    start = time.perf_counter()
    for _ in range(iterations):
        if ordered:
            ordered.popitem(last=False)
            ordered[len(ordered)] = accesses[0]
    pure_lru_time = time.perf_counter() - start

    print(f"  Cache size: {cache_size}")
    print(f"  Eviction operations: {iterations:,}")
    print(f"  LFU+LRU scoring: {lfu_lru_time*1000:.2f} ms")
    print(f"  Pure LRU popitem: {pure_lru_time*1000:.2f} ms")
    print(f"  Speedup: {lfu_lru_time/pure_lru_time:.1f}x")

    return {
        "lfu_lru_ms": lfu_lru_time * 1000,
        "pure_lru_ms": pure_lru_time * 1000,
        "speedup": lfu_lru_time / pure_lru_time
    }


def benchmark_document_ids_memory():
    """Benchmark int16 vs int64 for document IDs."""
    print("\n" + "="*60)
    print("BENCHMARK 4: Document IDs Memory (int16 vs int64)")
    print("="*60)

    max_length = 2048
    batch_size = 32

    # Old: int64
    doc_ids_old = torch.full((batch_size, max_length), -1, dtype=torch.long)
    old_bytes = doc_ids_old.numel() * doc_ids_old.element_size()

    # New: int16
    doc_ids_new = torch.full((batch_size, max_length), -1, dtype=torch.int16)
    new_bytes = doc_ids_new.numel() * doc_ids_new.element_size()

    print(f"  Batch size: {batch_size}, Seq length: {max_length}")
    print(f"  int64 memory: {old_bytes / 1024:.1f} KB")
    print(f"  int16 memory: {new_bytes / 1024:.1f} KB")
    print(f"  Memory reduction: {(1 - new_bytes / old_bytes) * 100:.0f}%")

    return {
        "int64_kb": old_bytes / 1024,
        "int16_kb": new_bytes / 1024,
        "reduction_pct": (1 - new_bytes / old_bytes) * 100
    }


def benchmark_array_copy_optimization():
    """Benchmark asarray vs always-copy approach."""
    print("\n" + "="*60)
    print("BENCHMARK 5: Array Copy Optimization (asarray vs copy)")
    print("="*60)

    # Simulate data from cache (contiguous, owned)
    num_rows = 10000
    seq_len = 512
    data = np.random.randint(0, 50000, size=(num_rows, seq_len), dtype=np.int64)

    iterations = num_rows

    # Old: Always copy
    start = time.perf_counter()
    for i in range(iterations):
        row = data[i]
        copied = np.array(row, dtype=np.int64, copy=True)
    always_copy_time = time.perf_counter() - start

    # New: asarray first, copy only if needed
    start = time.perf_counter()
    for i in range(iterations):
        row = data[i]
        arr = np.asarray(row, dtype=np.int64)
        if not arr.flags['OWNDATA'] or not arr.flags['WRITEABLE']:
            arr = arr.copy()
    smart_copy_time = time.perf_counter() - start

    print(f"  Rows processed: {iterations:,}")
    print(f"  Seq length: {seq_len}")
    print(f"  Always copy: {always_copy_time*1000:.2f} ms")
    print(f"  Smart copy: {smart_copy_time*1000:.2f} ms")
    print(f"  Speedup: {always_copy_time/smart_copy_time:.2f}x")

    return {
        "always_copy_ms": always_copy_time * 1000,
        "smart_copy_ms": smart_copy_time * 1000,
        "speedup": always_copy_time / smart_copy_time
    }


def benchmark_packing_collator():
    """Benchmark packing collator with cached bucket boundaries."""
    print("\n" + "="*60)
    print("BENCHMARK 6: Packing Collator (Bucket Caching)")
    print("="*60)

    from collections import defaultdict
    import random

    max_length = 2048
    num_buckets = 8
    batch_size = 64
    iterations = 1000

    # Simulate sequence lengths
    lengths = [random.randint(50, max_length) for _ in range(batch_size)]

    # Old: Compute boundaries every batch
    start = time.perf_counter()
    for _ in range(iterations):
        max_len = max(lengths)
        min_len = min(lengths)
        bucket_size = max(1, (max_len - min_len) // num_buckets)
        buckets = defaultdict(list)
        for idx, length in enumerate(lengths):
            bucket_idx = min(num_buckets - 1, (length - min_len) // bucket_size)
            buckets[bucket_idx].append(idx)
    no_cache_time = time.perf_counter() - start

    # New: Cache boundaries
    cached_boundaries = None
    cached_min = None
    cached_max = None

    start = time.perf_counter()
    for _ in range(iterations):
        max_len = max(lengths)
        min_len = min(lengths)

        # Cache check
        if cached_boundaries is None or cached_min != min_len or cached_max != max_len:
            bucket_size = max(1, (max_len - min_len) // num_buckets)
            cached_boundaries = [min_len + i * bucket_size for i in range(num_buckets + 1)]
            cached_min = min_len
            cached_max = max_len

        bucket_size = max(1, (max_len - min_len) // num_buckets)
        buckets = defaultdict(list)
        for idx, length in enumerate(lengths):
            bucket_idx = min(num_buckets - 1, (length - min_len) // bucket_size)
            buckets[bucket_idx].append(idx)
    with_cache_time = time.perf_counter() - start

    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Without cache: {no_cache_time*1000:.2f} ms")
    print(f"  With cache: {with_cache_time*1000:.2f} ms")
    print(f"  Speedup: {no_cache_time/with_cache_time:.2f}x")

    return {
        "no_cache_ms": no_cache_time * 1000,
        "with_cache_ms": with_cache_time * 1000,
        "speedup": no_cache_time / with_cache_time
    }


def benchmark_gpu_tensor_operations():
    """Benchmark GPU tensor operations if available."""
    if not torch.cuda.is_available():
        print("\n[SKIPPED] GPU benchmarks - no CUDA available")
        return None

    print("\n" + "="*60)
    print("BENCHMARK 7: GPU Tensor Operations")
    print("="*60)

    device = torch.device("cuda")
    batch_size = 32
    seq_len = 2048
    iterations = 100

    # Warmup
    warmup = torch.randn(batch_size, seq_len, device=device)
    del warmup
    torch.cuda.synchronize()

    # Old: torch.long for document IDs
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        doc_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
        # Simulate some operations
        mask = (doc_ids != -1)
        del doc_ids, mask
    torch.cuda.synchronize()
    long_time = time.perf_counter() - start
    long_peak_mem = torch.cuda.max_memory_allocated() / 1e6

    # New: torch.int16 for document IDs, convert to long only when needed
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        doc_ids = torch.full((batch_size, seq_len), -1, dtype=torch.int16, device=device)
        # Simulate some operations - convert to long only for comparison
        doc_ids_long = doc_ids.long()  # Convert only when needed
        mask = (doc_ids_long != -1)
        del doc_ids, doc_ids_long, mask
    torch.cuda.synchronize()
    int16_time = time.perf_counter() - start
    int16_peak_mem = torch.cuda.max_memory_allocated() / 1e6

    print(f"  Batch size: {batch_size}, Seq length: {seq_len}")
    print(f"  Iterations: {iterations}")
    print(f"  torch.long time: {long_time*1000:.2f} ms (peak: {long_peak_mem:.1f} MB)")
    print(f"  torch.int16 time: {int16_time*1000:.2f} ms (peak: {int16_peak_mem:.1f} MB)")

    return {
        "long_time_ms": long_time * 1000,
        "int16_time_ms": int16_time * 1000,
        "long_peak_mb": long_peak_mem,
        "int16_peak_mb": int16_peak_mem
    }


def main():
    print("="*60)
    print("DATA PIPELINE OPTIMIZATION BENCHMARKS")
    print("="*60)

    results = {}

    # Run all benchmarks
    results["numpy_index"] = benchmark_numpy_index_memory()
    results["heap_rank_selection"] = benchmark_heap_vs_linear_rank_selection()
    results["lru_eviction"] = benchmark_lru_eviction()
    results["document_ids"] = benchmark_document_ids_memory()
    results["array_copy"] = benchmark_array_copy_optimization()
    results["packing_cache"] = benchmark_packing_collator()
    results["gpu_operations"] = benchmark_gpu_tensor_operations()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nMemory Improvements:")
    print(f"  • Index storage: {results['numpy_index']['reduction_pct']:.0f}% reduction")
    print(f"  • Document IDs: {results['document_ids']['reduction_pct']:.0f}% reduction")

    print("\nSpeed Improvements:")
    print(f"  • Rank selection: {results['heap_rank_selection']['speedup']:.1f}x faster")
    print(f"  • Cache eviction: {results['lru_eviction']['speedup']:.0f}x faster")
    print(f"  • Array copying: {results['array_copy']['speedup']:.2f}x faster")
    print(f"  • Bucket caching: {results['packing_cache']['speedup']:.2f}x faster")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "after_optimization.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
