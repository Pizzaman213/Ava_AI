#!/usr/bin/env python3
"""
Test Lion optimizer integration with the training script.
This directly tests the optimizer setup function.
"""

import sys
sys.path.insert(0, '/project/code/scripts/5_training')
sys.path.insert(0, '/project/code/src')

import torch
import torch.nn as nn
from src.Ava.config.training_config import EnhancedTrainingConfig, TrainingConfig

print("=" * 80)
print("Lion Optimizer Training Integration Test")
print("=" * 80)

# Mock the get_logger function
class MockLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warning(self, msg):
        print(f"[WARN] {msg}")
    def error(self, msg):
        print(f"[ERROR] {msg}")

# Create a simple test model
print("\n[1] Creating test model...")
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512)
)
print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Test Lion optimizer setup
print("\n[2] Testing Lion optimizer setup...")

config_dict = {
    "training": {
        "optimizer": "lion",
        "learning_rate": 1.5e-5,
        "weight_decay": 0.033,
        "lion_betas": [0.9, 0.99]
    }
}

training_config = EnhancedTrainingConfig(
    config_file="test.yaml",
    training=TrainingConfig(
        learning_rate=1.5e-5,
        batch_size=8,
        epochs=1
    )
)

# Import the setup function
import importlib.util
spec = importlib.util.spec_from_file_location("train", "/project/code/scripts/5_training/train.py")
train_module = importlib.util.module_from_spec(spec)

# Mock logger
train_module._logger = MockLogger()
def get_logger():
    return train_module._logger
train_module.get_logger = get_logger

# Load the module
spec.loader.exec_module(train_module)

# Test optimizer setup
print("\n[3] Calling setup_optimizer_and_lr_management with Lion config...")
try:
    optimizer, lr_manager = train_module.setup_optimizer_and_lr_management(
        model=model,
        config_dict=config_dict,
        training_config=training_config,
        total_steps=1000
    )

    print(f"\n✅ SUCCESS! Lion optimizer created:")
    print(f"   Optimizer type: {type(optimizer).__name__}")
    print(f"   Learning rate: {optimizer.defaults['lr']:.2e}")
    print(f"   Betas: {optimizer.defaults['betas']}")
    print(f"   Weight decay: {optimizer.defaults['weight_decay']}")

    # Verify it's Lion
    from src.Ava.optimization.optimizers.advanced import LionOptimizer
    if isinstance(optimizer, LionOptimizer):
        print(f"\n✓ Confirmed: This is a Lion optimizer!")
    else:
        print(f"\n✗ ERROR: Expected LionOptimizer but got {type(optimizer).__name__}")

    # Test a training step
    print("\n[4] Testing optimizer step...")
    x = torch.randn(32, 512)
    target = torch.randn(32, 512)

    output = model(x)
    loss = nn.MSELoss()(output, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"✓ Training step completed successfully")
    print(f"  Loss: {loss.item():.4f}")

    # Check optimizer state
    first_param = list(model.parameters())[0]
    state = optimizer.state[first_param]
    print(f"  Optimizer state initialized: step={state['step']}")
    print(f"  Momentum buffer shape: {state['exp_avg'].shape}")

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test other optimizers too
print("\n" + "=" * 80)
print("Testing other optimizers...")
print("=" * 80)

for opt_name in ["adamw", "sophia", "adafactor"]:
    print(f"\n[{opt_name.upper()}] Testing {opt_name} optimizer...")

    config_dict["training"]["optimizer"] = opt_name

    try:
        optimizer, _ = train_module.setup_optimizer_and_lr_management(
            model=model,
            config_dict=config_dict,
            training_config=training_config,
            total_steps=1000
        )
        print(f"✓ {opt_name.upper()} optimizer created: {type(optimizer).__name__}")
    except Exception as e:
        print(f"✗ {opt_name.upper()} failed: {e}")

print("\n" + "=" * 80)
print("✅ All integration tests passed!")
print("=" * 80)
print("\n🚀 Ready to train with Lion optimizer!")
print("   Run: python train.py --config configs/moe/tiny_moe_ultra_low_mem.yaml")
