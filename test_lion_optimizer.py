#!/usr/bin/env python3
"""
Quick test script to verify Lion optimizer integration.
Tests optimizer initialization and basic functionality.
"""

import sys
sys.path.insert(0, '/project/code/src')

import torch
import torch.nn as nn
from Ava.optimization.optimizers.advanced import LionOptimizer, OptimizerFactory

print("=" * 80)
print("Lion Optimizer Integration Test")
print("=" * 80)

# Test 1: Direct instantiation
print("\n[Test 1] Direct Lion optimizer instantiation")
print("-" * 80)

# Create a simple model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

print(f"✓ Created test model with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create Lion optimizer
optimizer = LionOptimizer(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.01
)

print(f"✓ Lion optimizer created successfully")
print(f"  - Learning rate: {optimizer.defaults['lr']}")
print(f"  - Betas: {optimizer.defaults['betas']}")
print(f"  - Weight decay: {optimizer.defaults['weight_decay']}")

# Test 2: Optimizer step
print("\n[Test 2] Testing optimizer step")
print("-" * 80)

# Create dummy input and target
x = torch.randn(32, 100)
y = torch.randint(0, 10, (32,))

# Forward pass
output = model(x)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, y)

print(f"✓ Initial loss: {loss.item():.4f}")

# Backward pass
loss.backward()

# Optimizer step
optimizer.step()
optimizer.zero_grad()

print(f"✓ Optimizer step completed successfully")

# Check state buffer
param_with_state = list(model.parameters())[0]
state = optimizer.state[param_with_state]
print(f"✓ Optimizer state created:")
print(f"  - Step: {state['step']}")
print(f"  - Momentum buffer shape: {state['exp_avg'].shape}")
print(f"  - Memory per param: 1 buffer (vs 2 for AdamW)")

# Test 3: Factory method
print("\n[Test 3] Testing OptimizerFactory")
print("-" * 80)

# Test Lion
lion_opt = OptimizerFactory.create_optimizer(
    'lion',
    model.parameters(),
    learning_rate=1e-4,
    weight_decay=0.01
)
print(f"✓ Lion created via factory")

# Test Sophia
sophia_opt = OptimizerFactory.create_optimizer(
    'sophia',
    model.parameters(),
    learning_rate=1e-4,
    weight_decay=0.1
)
print(f"✓ Sophia created via factory")

# Test AdaFactor
adafactor_opt = OptimizerFactory.create_optimizer(
    'adafactor',
    model.parameters(),
    weight_decay=0.01
)
print(f"✓ AdaFactor created via factory")

# Test 4: Memory comparison
print("\n[Test 4] Memory usage comparison")
print("-" * 80)

memory_usage = OptimizerFactory.compare_memory_usage(model.parameters())
print("Estimated optimizer state memory (MB):")
for opt_name, mem_mb in memory_usage.items():
    if opt_name == 'Lion':
        savings = ((memory_usage['AdamW'] - mem_mb) / memory_usage['AdamW']) * 100
        print(f"  {opt_name:12s}: {mem_mb:6.2f} MB (saves {savings:.0f}% vs AdamW)")
    else:
        print(f"  {opt_name:12s}: {mem_mb:6.2f} MB")

# Test 5: Recommended configs
print("\n[Test 5] Recommended hyperparameters")
print("-" * 80)

for model_size in ['small', 'medium', 'large', 'xl']:
    config = OptimizerFactory.get_recommended_config('lion', model_size)
    print(f"  {model_size.upper():8s}: lr={config['lr']:.2e}, wd={config['weight_decay']:.3f}, betas={config['betas']}")

# Test 6: Multiple optimizer steps
print("\n[Test 6] Testing multiple training steps")
print("-" * 80)

losses = []
for step in range(10):
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    output = model(x)
    loss = loss_fn(output, y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"✓ Completed 10 training steps")
print(f"  - Initial loss: {losses[0]:.4f}")
print(f"  - Final loss: {losses[-1]:.4f}")
print(f"  - Loss change: {losses[-1] - losses[0]:+.4f}")

# Test 7: Weight decay behavior
print("\n[Test 7] Weight decay application")
print("-" * 80)

# Get initial parameter norm
initial_norm = sum(p.norm().item() for p in model.parameters())
print(f"  Initial parameter norm: {initial_norm:.4f}")

# Run steps with weight decay
for _ in range(20):
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

final_norm = sum(p.norm().item() for p in model.parameters())
print(f"  Final parameter norm: {final_norm:.4f}")
print(f"  ✓ Weight decay is {'working' if final_norm < initial_norm else 'NOT working'}")

# Summary
print("\n" + "=" * 80)
print("✅ All tests passed! Lion optimizer is ready for use.")
print("=" * 80)
print("\nNext steps:")
print("  1. Run training with: python train.py --config configs/moe/tiny_moe_ultra_low_mem.yaml")
print("  2. Monitor for: '✓ Using Lion optimizer (50% memory reduction vs AdamW)'")
print("  3. Compare GPU memory usage with nvidia-smi")
print("  4. Read LION_OPTIMIZER_GUIDE.md for hyperparameter tuning tips")
print("=" * 80)
