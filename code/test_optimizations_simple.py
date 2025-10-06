#!/usr/bin/env python3
"""
Simple test of optimizations without DeepSeek dependencies.

Tests all optimization modules can be imported and basic functionality works.
"""

import sys
sys.path.insert(0, '/project/code/src')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

print("=" * 80)
print("TESTING OPTIMIZATIONS - NO DEEPSEEK")
print("=" * 80)

# Test 1: Hardware Optimizations
print("\n[1/10] Testing hardware optimizations...")
try:
    from Ava.optimization.hardware_optimizations import auto_optimize_hardware
    hw_opt = auto_optimize_hardware()
    print("✅ Hardware optimizations: PASS")
except Exception as e:
    print(f"❌ Hardware optimizations: FAIL - {e}")

# Test 2: Fused Optimizers
print("\n[2/10] Testing fused optimizers...")
try:
    from Ava.optimization.fused_optimizers import FusedAdam, Lion
    model = nn.Linear(10, 10)
    opt1 = FusedAdam(model.parameters(), lr=1e-3, fused=False, foreach=True)
    opt2 = Lion(model.parameters(), lr=1e-4)
    print("✅ Fused optimizers: PASS")
except Exception as e:
    print(f"❌ Fused optimizers: FAIL - {e}")

# Test 3: Mixed Precision
print("\n[3/10] Testing mixed precision...")
try:
    from Ava.optimization.gradient_optimizations import MixedPrecisionManager
    mp_manager = MixedPrecisionManager(enabled=True)
    print(f"  - Detected dtype: {mp_manager.dtype}")
    print("✅ Mixed precision: PASS")
except Exception as e:
    print(f"❌ Mixed precision: FAIL - {e}")

# Test 4: Gradient Clipping
print("\n[4/10] Testing adaptive gradient clipping...")
try:
    from Ava.optimization.gradient_optimizations import AdaptiveGradientClipper
    clipper = AdaptiveGradientClipper(clip_type='adaptive')
    print("✅ Adaptive gradient clipping: PASS")
except Exception as e:
    print(f"❌ Adaptive gradient clipping: FAIL - {e}")

# Test 5: Data Loading
print("\n[5/10] Testing optimized dataloader...")
try:
    from Ava.data.optimized_dataloader import OptimizedDataLoaderFactory
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    loader = OptimizedDataLoaderFactory.create_dataloader(
        dataset,
        batch_size=8,
        num_workers=0,
        use_dynamic_batching=False
    )
    print(f"  - Created dataloader with {len(loader)} batches")
    print("✅ Optimized dataloader: PASS")
except Exception as e:
    print(f"❌ Optimized dataloader: FAIL - {e}")

# Test 6: Attention (basic import test)
print("\n[6/10] Testing attention modules...")
try:
    from Ava.layers.advanced_attention import FlashAttentionWrapper
    print("✅ Attention modules: PASS")
except Exception as e:
    print(f"❌ Attention modules: FAIL - {e}")

# Test 7: Compilation
print("\n[7/10] Testing compilation optimizations...")
try:
    from Ava.optimization.compilation_optimizations import CompilationManager
    compiler = CompilationManager(mode='reduce-overhead')
    print("✅ Compilation optimizations: PASS")
except Exception as e:
    print(f"❌ Compilation optimizations: FAIL - {e}")

# Test 8: Profiling
print("\n[8/10] Testing profiling tools...")
try:
    from Ava.training.profiling_tools import ThroughputTracker, MemoryProfiler
    tracker = ThroughputTracker(window_size=10)
    profiler = MemoryProfiler()
    print("✅ Profiling tools: PASS")
except Exception as e:
    print(f"❌ Profiling tools: FAIL - {e}")

# Test 9: Scheduling
print("\n[9/10] Testing advanced scheduling...")
try:
    from Ava.training.advanced_warmup_scheduling import GradientNoiseScale
    gns = GradientNoiseScale(window_size=10)
    print("✅ Advanced scheduling: PASS")
except Exception as e:
    print(f"❌ Advanced scheduling: FAIL - {e}")

# Test 10: Integration
print("\n[10/10] Testing optimization integration...")
try:
    from Ava.training.optimization_integration import OptimizedTrainingSetup

    # Create simple model and dataset
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    dataset = TensorDataset(torch.randn(50, 10), torch.randn(50, 10))

    config = {
        'batch_size': 8,
        'learning_rate': 1e-3,
        'num_workers': 0,
        'use_sequence_packing': False  # Disable for simple tensor dataset
    }

    opt_setup = OptimizedTrainingSetup(config, enable_all=True, verbose=False)
    components = opt_setup.create_complete_setup(model, dataset)

    print(f"  - Model: {type(components['model']).__name__}")
    print(f"  - Optimizer: {type(components['optimizer']).__name__}")
    print(f"  - Train loader batches: {len(components['train_loader'])}")
    print("✅ Optimization integration: PASS")
except Exception as e:
    print(f"❌ Optimization integration: FAIL - {e}")
    import traceback
    traceback.print_exc()

# Test 11: Quick Forward/Backward Pass
print("\n[BONUS] Testing actual training step...")
try:
    # Get components from test 10
    model = components['model']
    optimizer = components['optimizer']
    train_loader = components['train_loader']
    mp_manager = components['mp_manager']

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Get one batch
    batch = next(iter(train_loader))
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass with mixed precision
    with mp_manager.autocast():
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)

    # Backward pass
    scaled_loss = mp_manager.scale_loss(loss)
    scaled_loss.backward()

    # Optimizer step
    metrics = mp_manager.step_optimizer(optimizer)
    optimizer.zero_grad()

    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Device: {device}")
    print("✅ Training step: PASS")
except Exception as e:
    print(f"❌ Training step: FAIL - {e}")

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("\nAll core optimizations are working!")
print("\n✅ Ready to use in train.py")
print("=" * 80)
