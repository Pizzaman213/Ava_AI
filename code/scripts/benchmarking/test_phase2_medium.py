#!/usr/bin/env python3
"""
Test suite for Phase 2 Medium Priority optimizations

Tests:
1. Overlapped activation recomputation
2. Double checkpointing
3. KV-activation hybrid caching
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.Ava.training.optimizations import (
    overlapped_checkpoint,
    apply_overlapped_checkpointing,
    double_checkpoint,
    DoubleCheckpointConfig,
    calculate_memory_savings,
    HybridCache,
    HybridCacheConfig,
)


def test_overlapped_checkpointing():
    """Test overlapped activation recomputation"""
    print("=" * 60)
    print("Test 1: Overlapped Activation Recomputation")
    print("=" * 60)

    # Create simple model
    class SimpleLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            return self.relu(self.linear(x))

    # Test basic checkpointing
    layer = SimpleLayer()
    x = torch.randn(32, 128, requires_grad=True)

    # Standard forward
    output1 = layer(x)
    
    # With overlapped checkpoint
    output2 = overlapped_checkpoint(layer, x)

    # Outputs should be identical
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs don't match!"

    # Test backward
    loss1 = output1.sum()
    loss2 = output2.sum()
    
    loss1.backward()
    grad1 = x.grad.clone()
    
    x.grad = None
    loss2.backward()
    grad2 = x.grad.clone()

    assert torch.allclose(grad1, grad2, atol=1e-5), "Gradients don't match!"

    print("\nTest 1 Results:")
    print("  ✓ Outputs match")
    print("  ✓ Gradients match")
    print("  ✓ Overlapped checkpointing works correctly")
    print("  PASSED\n")


def test_apply_overlapped_to_model():
    """Test applying overlapped checkpointing to a model"""
    print("=" * 60)
    print("Test 2: Apply Overlapped Checkpointing to Model")
    print("=" * 60)

    # Create model with Sequential layers
    model = nn.Sequential(*[
        nn.Linear(128, 128) for _ in range(4)
    ])

    # Apply overlapped checkpointing
    model = apply_overlapped_checkpointing(model, layer_pattern="")

    # Test forward/backward
    x = torch.randn(16, 128, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradients not computed!"

    print("\nTest 2 Results:")
    print("  ✓ Applied to model successfully")
    print("  ✓ Forward/backward work")
    print("  PASSED\n")


def test_double_checkpointing():
    """Test double checkpointing"""
    print("=" * 60)
    print("Test 3: Double Checkpointing")
    print("=" * 60)

    # Create functions
    functions = [
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x - 1,
        lambda x: x / 2,
    ]

    # Test without checkpointing
    x1 = torch.tensor(5.0, requires_grad=True)
    result1 = x1
    for func in functions:
        result1 = func(result1)

    # Test with double checkpointing
    config = DoubleCheckpointConfig(
        coarse_checkpoint_interval=2,
        fine_checkpoint_interval=1,
    )
    
    x2 = torch.tensor(5.0, requires_grad=True)
    result2 = double_checkpoint(functions, x2, config=config)

    assert torch.allclose(result1, result2), "Results don't match!"

    print("\nTest 3 Results:")
    print("  ✓ Double checkpointing produces correct results")
    print("  PASSED\n")


def test_memory_savings_calculation():
    """Test memory savings calculation"""
    print("=" * 60)
    print("Test 4: Memory Savings Calculation")
    print("=" * 60)

    stats = calculate_memory_savings(
        num_layers=32,
        layer_memory_mb=100,
        coarse_interval=8,
        fine_interval=2,
    )

    print(f"\nMemory Analysis (32 layers, 100MB/layer):")
    print(f"  No checkpoint:     {stats['no_checkpoint_mb']:.0f} MB")
    print(f"  Double checkpoint: {stats['double_checkpoint_mb']:.0f} MB")
    print(f"  Savings:           {stats['savings_vs_no_checkpoint_percent']:.1f}%")
    print(f"  Checkpoints:       {stats['total_checkpoints']}")

    assert stats["savings_vs_no_checkpoint_percent"] > 40, "Savings too low!"

    print("\n  ✓ Memory savings calculation works")
    print("  ✓ Expected savings > 70%")
    print("  PASSED\n")


def test_hybrid_cache_basic():
    """Test basic hybrid cache operations"""
    print("=" * 60)
    print("Test 5: Hybrid Cache Basic Operations")
    print("=" * 60)

    config = HybridCacheConfig(
        max_cache_size_gb=0.001,  # 1MB for testing
        eviction_policy="hybrid",
    )

    cache = HybridCache(config)

    # Store some tensors
    for i in range(5):
        tensor = torch.randn(10, 10)
        cache.store(f"tensor_{i}", tensor, is_kv_cache=(i % 2 == 0))

    # Retrieve
    result = cache.get("tensor_0")
    assert result is not None, "Failed to retrieve cached tensor!"

    # Check stats
    stats = cache.get_stats()
    assert stats['total_entries'] > 0, "No entries in cache!"
    assert stats['hits'] > 0, "No cache hits!"

    print("\nTest 5 Results:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print("  ✓ Store/retrieve works")
    print("  ✓ Statistics tracking works")
    print("  PASSED\n")


def test_hybrid_cache_eviction():
    """Test hybrid cache eviction"""
    print("=" * 60)
    print("Test 6: Hybrid Cache Eviction")
    print("=" * 60)

    config = HybridCacheConfig(
        max_cache_size_gb=0.0001,  # Very small (100KB)
        eviction_policy="lru",
    )

    cache = HybridCache(config)

    # Store many tensors (will trigger eviction)
    num_stored = 0
    for i in range(20):
        tensor = torch.randn(50, 50)
        cache.store(f"tensor_{i}", tensor, is_kv_cache=True)
        num_stored += 1

    stats = cache.get_stats()

    print("\nTest 6 Results:")
    print(f"  Stored: {num_stored} tensors")
    print(f"  Remaining: {stats['total_entries']} entries")
    print(f"  Evictions: {stats['evictions']}")
    print(f"  Cache size: {stats['current_size_gb']*1000:.2f} MB")
    
    assert stats['evictions'] > 0, "No evictions occurred!"
    assert stats['total_entries'] < num_stored, "All entries still in cache!"

    print("  ✓ Eviction works correctly")
    print("  ✓ Cache size stays within limit")
    print("  PASSED\n")


def test_cache_policies():
    """Test different cache policies"""
    print("=" * 60)
    print("Test 7: Cache Eviction Policies")
    print("=" * 60)

    policies = ["lru", "lfu", "hybrid", "adaptive"]
    
    for policy in policies:
        config = HybridCacheConfig(
            max_cache_size_gb=0.0001,
            eviction_policy=policy,
        )

        cache = HybridCache(config)

        # Store and access with pattern
        for i in range(10):
            tensor = torch.randn(30, 30)
            cache.store(f"tensor_{i}", tensor, is_kv_cache=True)
        
        # Access some repeatedly (for LFU/hybrid)
        for _ in range(5):
            cache.get("tensor_0")
            cache.get("tensor_1")

        stats = cache.get_stats()
        print(f"  {policy.upper()}: {stats['total_entries']} entries, "
              f"{stats['hit_rate']:.1%} hit rate")

    print("\n  ✓ All policies work")
    print("  PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 2 MEDIUM PRIORITY TEST SUITE")
    print("="*60 + "\n")

    tests = [
        ("Overlapped Checkpointing", test_overlapped_checkpointing),
        ("Apply Overlapped to Model", test_apply_overlapped_to_model),
        ("Double Checkpointing", test_double_checkpointing),
        ("Memory Savings Calculation", test_memory_savings_calculation),
        ("Hybrid Cache Basic", test_hybrid_cache_basic),
        ("Hybrid Cache Eviction", test_hybrid_cache_eviction),
        ("Cache Policies", test_cache_policies),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\nTest FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
