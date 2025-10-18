#!/usr/bin/env python3
"""Detailed parameter count breakdown."""

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
print("DETAILED PARAMETER BREAKDOWN")
print("="*80)

# Create model
model = EnhancedMoEModel(config)

# Group parameters by component
param_groups = {}
for name, param in model.named_parameters():
    parts = name.split('.')
    if 'token_embedding' in name:
        group = 'token_embedding'
    elif 'lm_head' in name:
        group = 'lm_head'
    elif 'final_layer_norm' in name:
        group = 'final_layer_norm'
    elif 'layers' in name:
        layer_num = parts[1]
        component = parts[2] if len(parts) > 2 else 'unknown'
        group = f'layer_{layer_num}_{component}'
    else:
        group = 'other'

    if group not in param_groups:
        param_groups[group] = 0
    param_groups[group] += param.numel()

# Sort and display
print("\nParameter breakdown by component:\n")
total = 0
for group in sorted(param_groups.keys()):
    count = param_groups[group]
    total += count
    print(f"  {group:40s}: {count:10,} ({count/1000:7.1f}K)")

print(f"\n{'='*80}")
print(f"  {'TOTAL':40s}: {total:10,} ({total/1000:7.1f}K)")
print(f"{'='*80}")

# Show layer-wise breakdown
print("\nPer-layer breakdown:")
layer_totals = {}
for group, count in param_groups.items():
    if group.startswith('layer_'):
        layer_num = group.split('_')[1]
        if layer_num not in layer_totals:
            layer_totals[layer_num] = 0
        layer_totals[layer_num] += count

for layer in sorted(layer_totals.keys()):
    print(f"  Layer {layer}: {layer_totals[layer]:,} ({layer_totals[layer]/1000:.1f}K)")

print(f"\nAverage per layer: {sum(layer_totals.values())/len(layer_totals):,.0f} "
      f"({sum(layer_totals.values())/len(layer_totals)/1000:.1f}K)")
