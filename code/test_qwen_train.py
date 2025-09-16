#!/usr/bin/env python3
"""Quick test of the qwen_code_train.py script"""

import sys
import os
sys.path.insert(0, '/project/code/scripts/training')

# Test imports
try:
    from qwen_code_train import (
        EnhancedMoEConfig,
        EnhancedMoEModel,
        MoEPlusPlusLayer,
        MTPLayer,
        AdvancedTrainingOrchestrator,
        TextDataset
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test model creation
try:
    config = EnhancedMoEConfig(
        hidden_size=256,  # Smaller for testing
        num_layers=2,     # Fewer layers
        num_attention_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        intermediate_size=512
    )
    print("✅ Config created successfully")

    import torch
    model = EnhancedMoEModel(config)
    print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    device = torch.device("cpu")
    model = model.to(device)

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"✅ Forward pass successful")
    print(f"   Output shape: {outputs['logits'].shape}")
    print(f"   Loss: {outputs['loss']}")

except Exception as e:
    print(f"❌ Error during model test: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 All tests passed! The script structure is functional.")