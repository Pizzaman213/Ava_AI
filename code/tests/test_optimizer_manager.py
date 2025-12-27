"""
Test OptimizerManager functionality.

Tests:
- BF16 epsilon auto-detection
- Config path fallback (nested vs flat)
- Optimizer creation for all types
- Scheduler creation and step tracking
- Scheduler double-step prevention
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir / 'src'))

from ava.training.context import TrainingContext
from ava.training.optimizer import OptimizerManager


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)


def test_bf16_epsilon_detection():
    """Test that BF16 uses eps=1e-6, FP16/FP32 use eps=1e-8."""
    model = DummyModel()
    context = TrainingContext(model=model, device=torch.device('cpu'))
    manager = OptimizerManager(context)
    manager.initialize()

    # Test BF16 config (nested structure)
    config_bf16 = {
        'training': {
            'precision': {'mixed_precision': 'bf16'}
        },
        'optimizer': {'type': 'adamw'}
    }

    opt = manager.create_optimizer(model, config_bf16, learning_rate=1e-4)

    # Check epsilon
    assert opt.param_groups[0]['eps'] == 1e-6, \
        f"BF16 should use eps=1e-6, got {opt.param_groups[0]['eps']}"

    print("✓ BF16 epsilon auto-detection works (eps=1e-6)")

    # Test FP16 config
    config_fp16 = {
        'training': {
            'precision': {'mixed_precision': 'fp16'}
        },
        'optimizer': {'type': 'adamw'}
    }

    opt2 = manager.create_optimizer(model, config_fp16, learning_rate=1e-4)
    assert opt2.param_groups[0]['eps'] == 1e-8, \
        f"FP16 should use eps=1e-8, got {opt2.param_groups[0]['eps']}"

    print("✓ FP16 epsilon auto-detection works (eps=1e-8)")


def test_epsilon_override():
    """Test that explicit epsilon in config overrides auto-detection."""
    model = DummyModel()
    context = TrainingContext(model=model, device=torch.device('cpu'))
    manager = OptimizerManager(context)
    manager.initialize()

    # BF16 with explicit (bad) epsilon
    config = {
        'training': {
            'precision': {'mixed_precision': 'bf16'}
        },
        'optimizer': {
            'type': 'adamw',
            'eps': 1e-8  # Override (should warn in docs)
        }
    }

    opt = manager.create_optimizer(model, config, learning_rate=1e-4)

    # Explicit value should override auto-detection
    assert opt.param_groups[0]['eps'] == 1e-8, \
        "Explicit eps should override auto-detection"

    print("✓ Explicit epsilon override works")


def test_config_path_fallback():
    """Test backward compatibility with flat mixed_precision path."""
    model = DummyModel()
    context = TrainingContext(model=model, device=torch.device('cpu'))
    manager = OptimizerManager(context)
    manager.initialize()

    # Legacy flat path
    config_legacy = {
        'training': {
            'mixed_precision': 'bf16'  # Flat (old format)
        },
        'optimizer': {'type': 'adamw'}
    }

    opt = manager.create_optimizer(model, config_legacy, learning_rate=1e-4)

    # Should still detect BF16 and use correct epsilon
    assert opt.param_groups[0]['eps'] == 1e-6, \
        "Legacy config path should work with BF16 detection"

    print("✓ Legacy config path fallback works")

    # Nested path should take precedence
    config_both = {
        'training': {
            'mixed_precision': 'fp16',  # Legacy (ignored)
            'precision': {'mixed_precision': 'bf16'}  # New (used)
        },
        'optimizer': {'type': 'adamw'}
    }

    opt2 = manager.create_optimizer(model, config_both, learning_rate=1e-4)
    assert opt2.param_groups[0]['eps'] == 1e-6, \
        "Nested path should override flat path"

    print("✓ Nested config path takes precedence")


def test_scheduler_creation():
    """Test LR scheduler creation with warmup and decay."""
    model = DummyModel()
    context = TrainingContext(model=model, device=torch.device('cpu'))
    manager = OptimizerManager(context)
    manager.initialize()

    config = {
        'training': {'precision': {'mixed_precision': 'bf16'}},
        'optimizer': {'type': 'adamw'}
    }

    opt = manager.create_optimizer(model, config, learning_rate=1e-3)
    scheduler = manager.create_scheduler(
        opt, warmup_steps=100, total_steps=1000, min_lr=1e-5
    )

    # Initial LR should be low (warmup start)
    initial_lr = opt.param_groups[0]['lr']
    assert initial_lr < 1e-3, f"Warmup should start with low LR, got {initial_lr}"

    # Step through warmup
    for _ in range(100):
        scheduler.step()

    # After warmup, LR should be near base rate
    post_warmup_lr = opt.param_groups[0]['lr']
    assert abs(post_warmup_lr - 1e-3) < 1e-4, \
        f"After warmup, LR should be ~1e-3, got {post_warmup_lr}"

    print("✓ LR scheduler warmup works correctly")


def test_scheduler_no_double_step():
    """Test that on_step_end doesn't double-step scheduler."""
    model = DummyModel()
    context = TrainingContext(model=model, device=torch.device('cpu'))
    manager = OptimizerManager(context)
    manager.initialize()

    config = {
        'training': {'precision': {'mixed_precision': 'bf16'}},
        'optimizer': {'type': 'adamw'}
    }

    opt = manager.create_optimizer(model, config, learning_rate=1e-3)
    scheduler = manager.create_scheduler(
        opt, warmup_steps=10, total_steps=100, min_lr=1e-5
    )

    # Manually step scheduler
    scheduler.step()
    lr_after_manual_step = opt.param_groups[0]['lr']

    # Call on_step_end (should NOT step scheduler again)
    manager.on_step_end(step=1, loss=0.5)
    lr_after_callback = opt.param_groups[0]['lr']

    # LR should be unchanged
    assert lr_after_callback == lr_after_manual_step, \
        "on_step_end should not step scheduler (prevents 2x speed)"

    print("✓ Scheduler double-step prevention works")


def test_optimizer_types():
    """Test creation of different optimizer types."""
    model = DummyModel()
    context = TrainingContext(model=model, device=torch.device('cpu'))
    manager = OptimizerManager(context)
    manager.initialize()

    # Test AdamW (always available)
    config_adamw = {
        'training': {'precision': {'mixed_precision': 'bf16'}},
        'optimizer': {'type': 'adamw'}
    }
    opt = manager.create_optimizer(model, config_adamw, learning_rate=1e-4)
    assert isinstance(opt, torch.optim.AdamW)
    print("✓ AdamW optimizer creation works")

    # Test Lion (if available)
    from ava.training.optimizer import LION_AVAILABLE
    if LION_AVAILABLE:
        config_lion = {
            'training': {'precision': {'mixed_precision': 'bf16'}},
            'optimizer': {'type': 'lion'}
        }
        opt_lion = manager.create_optimizer(model, config_lion, learning_rate=1e-4)
        print("✓ Lion optimizer creation works")
    else:
        print("⊗ Lion optimizer not available (skip)")

    # Test GaLore (if available)
    from ava.training.optimizer import GALORE_AVAILABLE
    if GALORE_AVAILABLE:
        config_galore = {
            'training': {'precision': {'mixed_precision': 'bf16'}},
            'optimizer': {'type': 'galore', 'eps': 1e-6}
        }
        opt_galore = manager.create_optimizer(model, config_galore, learning_rate=1e-4)
        print("✓ GaLore optimizer creation works")
    else:
        print("⊗ GaLore optimizer not available (skip)")


if __name__ == '__main__':
    print("Testing OptimizerManager...")
    test_bf16_epsilon_detection()
    test_epsilon_override()
    test_config_path_fallback()
    test_scheduler_creation()
    test_scheduler_no_double_step()
    test_optimizer_types()
    print("\n✅ All optimizer tests passed!")
