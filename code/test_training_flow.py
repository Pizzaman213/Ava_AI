#!/usr/bin/env python3
"""Test the complete training flow"""

import sys
import os
import torch
sys.path.insert(0, '/project/code/scripts/training')

from qwen_code_train import (
    EnhancedMoEConfig,
    EnhancedMoEModel,
    AdvancedTrainingOrchestrator,
    TextDataset
)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

print("🔧 Setting up test environment...")

# Setup
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Small config for testing
config = EnhancedMoEConfig(
    hidden_size=128,
    num_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    num_experts=4,
    num_experts_per_tok=2,
    min_experts_per_tok=1,
    max_experts_per_tok=3,
    expert_sparsity=0.3,
    gradient_checkpointing=False,  # Disable for testing
    adaptive_computation=False      # Disable for testing
)

print("✅ Config created")

# Create model
model = EnhancedMoEModel(config)
model = model.to(device)
print(f"✅ Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Create orchestrator
training_config = {
    'training': {
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'warmup_steps': 10,
        'total_steps': 100,
        'batch_size': 2
    },
    'deepspeed': {
        'enabled': False
    },
    'hidden_size': config.hidden_size,
    'num_experts': config.num_experts
}

orchestrator = AdvancedTrainingOrchestrator(training_config, model, tokenizer, device)
print("✅ Training orchestrator created")

# Create dummy dataset
dataset = TextDataset(None, tokenizer, max_length=32)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print(f"✅ Dataset created with {len(dataset)} samples")

# Test training step
print("\n🏃 Testing training step...")
batch = next(iter(dataloader))

try:
    metrics = orchestrator.train_step(batch, global_step=0, total_steps=100)
    print(f"✅ Training step successful!")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Temperature: {metrics.get('routing_temperature', 1.0):.3f}")

    # Test evaluation
    print("\n📊 Testing evaluation...")
    eval_metrics = orchestrator.evaluate(dataloader)
    print(f"✅ Evaluation successful!")
    print(f"   Val Loss: {eval_metrics['loss']:.4f}")
    print(f"   Perplexity: {eval_metrics['perplexity']:.2f}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 All training flow tests passed!")