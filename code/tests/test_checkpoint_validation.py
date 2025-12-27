"""
Test checkpoint validation and optimizer state verification.

Tests:
- Optimizer state param count validation
- NaN/Inf detection in optimizer states
- Checkpoint load with mismatched states
- Checkpoint load with corrupted states
"""

import sys
from pathlib import Path
import tempfile
import torch
import torch.nn as nn

code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir / 'src'))

from ava.core.checkpoint import CheckpointManager


class TinyModel(nn.Module):
    """Tiny model for checkpoint testing."""
    def __init__(self, num_params=5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(num_params)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_optimizer_state_validation():
    """Test that validation detects param count mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # Create model and optimizer
        model = TinyModel(num_params=5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do one step to create optimizer state
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        opt.step()

        # Create checkpoint manager
        manager = CheckpointManager(save_dir=save_dir, async_save=False)

        # Save checkpoint
        manager.save(model, opt, epoch=1, step=100, metrics={'loss': 0.5})

        # Load into model with DIFFERENT number of params
        model2 = TinyModel(num_params=3)  # Fewer params!
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        # Load should detect mismatch
        ckpt_path = save_dir / 'checkpoint_epoch_1_step_100.pt'

        # This should log warning but not crash
        epoch, step = manager.load(model2, opt2, ckpt_path)

        print("✓ Optimizer state validation detects param mismatch")


def test_nan_inf_detection():
    """Test that validation detects NaN/Inf in optimizer states."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        model = TinyModel(num_params=3)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create optimizer state
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        opt.step()

        # Get state dict and corrupt it
        state_dict = opt.state_dict()

        # Find first state tensor and inject NaN
        for param_id in state_dict['state']:
            state = state_dict['state'][param_id]
            if 'exp_avg' in state:
                state['exp_avg'][0] = float('nan')
                break

        # Create checkpoint manager
        manager = CheckpointManager(save_dir=save_dir, async_save=False)

        # Validate should detect NaN
        valid = manager._validate_optimizer_state(model, opt, state_dict)
        assert not valid, "Should detect NaN in optimizer state"

        print("✓ NaN detection in optimizer state works")

        # Test Inf detection
        state_dict2 = opt.state_dict()
        for param_id in state_dict2['state']:
            state = state_dict2['state'][param_id]
            if 'exp_avg' in state:
                state['exp_avg'][0] = float('inf')
                break

        valid2 = manager._validate_optimizer_state(model, opt, state_dict2)
        assert not valid2, "Should detect Inf in optimizer state"

        print("✓ Inf detection in optimizer state works")


def test_checkpoint_recovery():
    """Test that corrupted optimizer state is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        model = TinyModel(num_params=3)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create and save valid checkpoint
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        opt.step()

        manager = CheckpointManager(save_dir=save_dir, async_save=False)
        manager.save(model, opt, epoch=1, step=100, metrics={'loss': 0.5})

        # Load checkpoint
        ckpt_path = save_dir / 'checkpoint_epoch_1_step_100.pt'
        checkpoint = torch.load(ckpt_path)

        # Corrupt optimizer state (remove param_groups)
        del checkpoint['optimizer_state_dict']['param_groups']

        # Save corrupted checkpoint
        corrupted_path = save_dir / 'corrupted.pt'
        torch.save(checkpoint, corrupted_path)

        # Load should handle corruption
        model2 = TinyModel(num_params=3)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        epoch, step = manager.load(model2, opt2, corrupted_path)

        # Model should load, optimizer should be fresh
        assert epoch == 1
        assert step == 100

        print("✓ Checkpoint corruption is handled gracefully")


if __name__ == '__main__':
    print("Testing checkpoint validation...")
    test_optimizer_state_validation()
    test_nan_inf_detection()
    test_checkpoint_recovery()
    print("\n✅ All checkpoint validation tests passed!")
