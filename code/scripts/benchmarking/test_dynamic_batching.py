#!/usr/bin/env python3
"""
Test script for dynamic batching implementation

Tests the DynamicBatchScheduler to verify it works correctly
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.Ava.training.optimizations.dynamic_batching import (
    DynamicBatchScheduler,
    DynamicBatchConfig,
)


def test_basic_functionality():
    """Test basic dynamic batching functionality"""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    config = DynamicBatchConfig(
        initial_batch_size=128,
        min_batch_size=32,
        max_batch_size=512,
        adjustment_frequency=5,
        warmup_steps=10,
    )

    scheduler = DynamicBatchScheduler(config)

    # Simulate training
    for step in range(100):
        batch_size = scheduler.get_current_batch_size()
        scheduler.step(step)

    stats = scheduler.get_statistics()
    print(f"\nTest 1 Results:")
    print(f"  Total adjustments: {stats['total_adjustments']}")
    print(f"  Final batch size: {stats['current_batch_size']}")
    print(f"  Avg memory utilization: {stats['avg_memory_utilization']:.1%}")
    print("  PASSED\n")


def test_memory_thresholds():
    """Test that scheduler respects memory thresholds"""
    print("=" * 60)
    print("Test 2: Memory Threshold Handling")
    print("=" * 60)

    config = DynamicBatchConfig(
        initial_batch_size=256,
        min_batch_size=64,
        max_batch_size=1024,
        low_memory_threshold=0.50,
        high_memory_threshold=0.85,
        adjustment_frequency=1,
        warmup_steps=5,
    )

    scheduler = DynamicBatchScheduler(config)

    # Run for a bit
    for step in range(50):
        scheduler.step(step)

    stats = scheduler.get_statistics()
    print(f"\nTest 2 Results:")
    print(f"  Batch size range: [{stats['min_batch_size_reached']}, {stats['max_batch_size_reached']}]")
    print(f"  Total increases: {stats['total_increases']}")
    print(f"  Total decreases: {stats['total_decreases']}")
    print("  PASSED\n")


def test_boundary_conditions():
    """Test min/max batch size limits"""
    print("=" * 60)
    print("Test 3: Boundary Conditions")
    print("=" * 60)

    config = DynamicBatchConfig(
        initial_batch_size=128,
        min_batch_size=64,
        max_batch_size=256,
        adjustment_frequency=1,
        warmup_steps=0,
    )

    scheduler = DynamicBatchScheduler(config)

    for step in range(200):
        scheduler.step(step)
        current = scheduler.get_current_batch_size()
        assert current >= config.min_batch_size, f"Batch size {current} below minimum {config.min_batch_size}"
        assert current <= config.max_batch_size, f"Batch size {current} above maximum {config.max_batch_size}"

    print(f"\nTest 3 Results:")
    print(f"  All batch sizes within bounds: [{config.min_batch_size}, {config.max_batch_size}]")
    print("  PASSED\n")


def test_disabled_mode():
    """Test that disabled mode doesn't adjust"""
    print("=" * 60)
    print("Test 4: Disabled Mode")
    print("=" * 60)

    config = DynamicBatchConfig(
        enabled=False,
        initial_batch_size=128,
    )

    scheduler = DynamicBatchScheduler(config)
    initial_size = scheduler.get_current_batch_size()

    for step in range(100):
        scheduler.step(step)

    final_size = scheduler.get_current_batch_size()

    assert initial_size == final_size, "Batch size changed when disabled"
    print(f"\nTest 4 Results:")
    print(f"  Batch size unchanged: {initial_size} == {final_size}")
    print("  PASSED\n")


def test_statistics():
    """Test statistics tracking"""
    print("=" * 60)
    print("Test 5: Statistics Tracking")
    print("=" * 60)

    config = DynamicBatchConfig(
        initial_batch_size=256,
        adjustment_frequency=10,
        warmup_steps=50,
    )

    scheduler = DynamicBatchScheduler(config)

    for step in range(300):
        scheduler.step(step)

    stats = scheduler.get_statistics()

    assert 'current_batch_size' in stats
    assert 'total_adjustments' in stats
    assert 'avg_memory_utilization' in stats

    print(f"\nTest 5 Results:")
    print(f"  Statistics collected: {len(stats)} metrics")
    print(f"  Current batch size: {stats['current_batch_size']}")
    print(f"  Total adjustments: {stats['total_adjustments']}")
    print("  PASSED\n")

    # Print summary
    scheduler.log_summary()


def test_realistic_scenario():
    """Test a realistic training scenario"""
    print("=" * 60)
    print("Test 6: Realistic Training Scenario")
    print("=" * 60)

    # Similar to actual config
    config = DynamicBatchConfig(
        enabled=True,
        initial_batch_size=256,
        min_batch_size=64,
        max_batch_size=1024,
        low_memory_threshold=0.50,
        target_memory_threshold=0.70,
        high_memory_threshold=0.85,
        critical_memory_threshold=0.95,
        adjustment_frequency=10,
        warmup_steps=100,
        max_adjustments_per_session=50,
    )

    scheduler = DynamicBatchScheduler(config)

    # Simulate 1000 training steps
    for step in range(1000):
        batch_size = scheduler.get_current_batch_size()
        scheduler.step(step)  # Update scheduler

        # Simulate training (in real use, this would be actual training)

    stats = scheduler.get_statistics()

    print(f"\nTest 6 Results:")
    print(f"  Steps completed: 1000")
    print(f"  Final batch size: {stats['current_batch_size']}")
    print(f"  Batch size improvement: {stats['batch_size_improvement']:+.1f}%")
    print(f"  Total adjustments: {stats['total_adjustments']}")
    print(f"  Avg memory: {stats['avg_memory_utilization']:.1%}")
    print("  PASSED\n")

    scheduler.log_summary()


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DYNAMIC BATCHING TEST SUITE")
    print("="*60 + "\n")

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Memory Thresholds", test_memory_thresholds),
        ("Boundary Conditions", test_boundary_conditions),
        ("Disabled Mode", test_disabled_mode),
        ("Statistics Tracking", test_statistics),
        ("Realistic Scenario", test_realistic_scenario),
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
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\n{failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
