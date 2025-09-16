#!/usr/bin/env python3
"""Quick test of the training pipeline."""

import sys
sys.path.append('/project/code')

import torch
from transformers import AutoTokenizer
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.data import create_dataloaders

print("1. Creating tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print("2. Creating model...")
config = EnhancedMoEConfig(
    hidden_size=128,
    num_layers=2,
    num_attention_heads=4,
    num_experts=2,
    ffn_hidden_size=256,
    vocab_size=tokenizer.vocab_size
)
model = EnhancedMoEModel(config)
print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")

print("3. Creating dataloaders...")
train_loader, val_loader = create_dataloaders(
    tokenizer=tokenizer,
    batch_size=2,
    max_length=32,
    data_dir='/project/code/data/pretraining/processed',
    num_workers=0,  # Use 0 workers for debugging
    max_train_samples=10,
    max_val_samples=5
)

print("4. Testing one training step...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 1:
        break

    print(f"   Batch {batch_idx}: input_ids shape = {batch['input_ids'].shape}")

    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
    )

    loss = outputs['loss']
    print(f"   Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("   ✓ Training step successful!")

print("\n✅ All tests passed!")
print("\nTo run full training:")
print("python /project/code/scripts/training/train.py --config /project/code/configs/cpu/small.yaml")
print("\nOutput will be saved to: /project/code/outputs/")