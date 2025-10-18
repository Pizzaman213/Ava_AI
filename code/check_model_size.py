#!/usr/bin/env python3
"""Check model size with current configuration."""

import torch
import yaml
from pathlib import Path
from src.Ava.models.moe_model import EnhancedMoEConfig, EnhancedMoEModel

# Load config
config_path = Path("/project/code/configs/gpu/small.yaml")
with open(config_path) as f:
    yaml_config = yaml.safe_load(f)

model_config = yaml_config['model']

# Create model config
config = EnhancedMoEConfig(
    vocab_size=model_config['vocab_size'],
    hidden_size=model_config['hidden_size'],
    num_layers=model_config['num_layers'],
    num_attention_heads=model_config['num_attention_heads'],
    intermediate_size=model_config['intermediate_size'],
    max_position_embeddings=model_config['max_position_embeddings'],
    num_experts=model_config['num_experts'],
    num_experts_per_token=model_config['num_experts_per_token'],
    router_type=model_config.get('router_type', 'deepseek'),
    initializer_range=model_config.get('initializer_range', 0.02),
)

print("="*80)
print("MODEL SIZE CHECK")
print("="*80)
print("\nConfiguration:")
print(f"  Vocab size: {config.vocab_size}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Num layers: {config.num_layers}")
print(f"  Num heads: {config.num_attention_heads}")
print(f"  Intermediate size: {config.intermediate_size}")
print(f"  Num experts: {config.num_experts}")
print(f"  Experts per token: {config.num_experts_per_token}")

# Create model
print("\nInitializing model...")
model = EnhancedMoEModel(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*80}")
print("PARAMETER COUNT:")
print(f"{'='*80}")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Total (human): {total_params/1000:.1f}K")

# Estimate model size
param_size_bytes = total_params * 4  # FP32
param_size_fp16_bytes = total_params * 2  # FP16

print(f"\n{'='*80}")
print("MODEL SIZE ESTIMATES:")
print(f"{'='*80}")
print(f"  FP32 size: {param_size_bytes / (1024**2):.2f} MB")
print(f"  FP16 size: {param_size_fp16_bytes / (1024**2):.2f} MB")

# Save a test checkpoint to check actual size
test_checkpoint_path = Path("/project/code/test_model_checkpoint.pt")
print(f"\nSaving test checkpoint to {test_checkpoint_path}...")
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, test_checkpoint_path)

actual_size = test_checkpoint_path.stat().st_size
print(f"  Actual checkpoint size: {actual_size / (1024**2):.2f} MB")
print(f"  Actual checkpoint size: {actual_size / 1024:.2f} KB")

# Compare with NeuML
print(f"\n{'='*80}")
print("COMPARISON WITH NEUML:")
print(f"{'='*80}")
print(f"  NeuML original: 94,450 params (0.386 MB)")
print(f"  Our model: {total_params:,} params ({actual_size / (1024**2):.2f} MB)")
print(f"  Size ratio: {total_params / 94450:.1f}x larger")
print(f"  File ratio: {actual_size / (386*1024):.1f}x larger")

# Clean up
test_checkpoint_path.unlink()
print(f"\nTest checkpoint removed.")
print("="*80)
