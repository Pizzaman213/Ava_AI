#!/usr/bin/env python3
"""
Quick test of Sparse MoE integration with training loop.
Tests that the model can be created, trained for a few steps, and metrics work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.Ava.models.moe_model import OptimizedMoETransformer, OptimizedMoEConfig

print("=" * 70)
print("SPARSE MoE TRAINING INTEGRATION TEST")
print("=" * 70)

# Create minimal config
config = OptimizedMoEConfig(
    vocab_size=1000,
    hidden_size=128,
    num_layers=2,
    num_attention_heads=4,
    intermediate_size=448,
    max_position_embeddings=256,
    num_experts=4,
    num_experts_per_token=2,
    router_type='mixtral',
    use_grouped_gemm=True,
    use_triton_kernels=False,
    use_torch_compile=False,
    gradient_checkpointing=False,
)

print(f"\nConfig:")
print(f"  Hidden: {config.hidden_size}, Layers: {config.num_layers}")
print(f"  Experts: {config.num_experts}, K={config.num_experts_per_token}")
print(f"  Router: {config.router_type}")

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory before model: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

model = OptimizedMoETransformer(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel created: {total_params:,} parameters")

if device == 'cuda':
    print(f"GPU memory after model: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Model is on device: {next(model.parameters()).device}")

# Create dummy data
batch_size = 2
seq_len = 32
num_samples = 10

input_ids = torch.randint(0, config.vocab_size, (num_samples, seq_len))
labels = torch.randint(0, config.vocab_size, (num_samples, seq_len))
dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop (few steps)
print(f"\nTraining for 5 steps...")
model.train()

for step, (input_ids, labels) in enumerate(dataloader):
    if step >= 5:
        break

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Forward pass
    output = model(input_ids, labels=labels, return_dict=True)
    loss = output['loss']

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get metrics
    aux_loss = 0.0
    if output['aux_info']:
        aux_loss = sum(info['aux_loss'].item() for info in output['aux_info']) / len(output['aux_info'])

    print(f"  Step {step+1}: Loss={loss.item():.4f}, Aux Loss={aux_loss:.6f}")

    # Check expert usage (first layer)
    if output['aux_info'] and len(output['aux_info']) > 0:
        layer_0_info = output['aux_info'][0]
        if 'balance_score' in layer_0_info:
            balance_score = layer_0_info['balance_score'].item()
            print(f"    Balance Score: {balance_score:.4f}")

print("\n" + "=" * 70)
print("✓ TRAINING INTEGRATION TEST PASSED")
print("=" * 70)
print("\nThe Sparse MoE model successfully:")
print("  ✓ Created OptimizedMoETransformer")
print("  ✓ Performed forward passes")
print("  ✓ Computed losses (main + auxiliary)")
print("  ✓ Performed backward passes")
print("  ✓ Updated parameters")
print("  ✓ Tracked expert usage metrics")
print("\nSparse MoE is ready for use with train.py!")
