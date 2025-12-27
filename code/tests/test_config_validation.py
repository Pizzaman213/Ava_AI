"""
Test configuration validation for ModelConfig and TrainingConfig.

Tests that invalid configurations are caught early with clear error messages.
"""

import sys
from pathlib import Path

# Add code/src to path for src.ava imports
code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir))


def test_config_imports():
    """Test that config modules import correctly."""
    from src.ava.config.training_config import ModelConfig, TrainingConfig
    print("✓ Config modules import successfully")


# ============================================================================
# ModelConfig Validation Tests
# ============================================================================

def test_model_config_valid():
    """Test that valid ModelConfig passes validation."""
    from src.ava.config.training_config import ModelConfig

    config = ModelConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_layers=6,
        num_attention_heads=8,
        num_experts=4,
        num_experts_per_token=2,
        capacity_factor=1.25,
    )
    assert config.hidden_size == 512
    print("✓ Valid ModelConfig passes validation")


def test_model_config_hidden_size_zero():
    """Test that hidden_size=0 raises ValueError."""
    from src.ava.config.training_config import ModelConfig

    try:
        ModelConfig(hidden_size=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "hidden_size" in str(e)
        print("✓ hidden_size=0 raises ValueError")


def test_model_config_hidden_size_negative():
    """Test that negative hidden_size raises ValueError."""
    from src.ava.config.training_config import ModelConfig

    try:
        ModelConfig(hidden_size=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "hidden_size" in str(e)
        print("✓ Negative hidden_size raises ValueError")


def test_model_config_num_layers_zero():
    """Test that num_layers=0 raises ValueError."""
    from src.ava.config.training_config import ModelConfig

    try:
        ModelConfig(num_layers=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_layers" in str(e)
        print("✓ num_layers=0 raises ValueError")


def test_model_config_num_experts_per_token_exceeds_num_experts():
    """Test that num_experts_per_token > num_experts raises ValueError."""
    from src.ava.config.training_config import ModelConfig

    try:
        ModelConfig(num_experts=4, num_experts_per_token=8)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_experts_per_token" in str(e)
        print("✓ num_experts_per_token > num_experts raises ValueError")


def test_model_config_capacity_factor_zero():
    """Test that capacity_factor=0 raises ValueError."""
    from src.ava.config.training_config import ModelConfig

    try:
        ModelConfig(capacity_factor=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "capacity_factor" in str(e)
        print("✓ capacity_factor=0 raises ValueError")


def test_model_config_negative_router_z_loss():
    """Test that negative router_z_loss_coef raises ValueError."""
    from src.ava.config.training_config import ModelConfig

    try:
        ModelConfig(router_z_loss_coef=-0.001)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "router_z_loss_coef" in str(e)
        print("✓ Negative router_z_loss_coef raises ValueError")


def test_model_config_hidden_size_not_divisible():
    """Test that hidden_size not divisible by num_attention_heads raises ValueError."""
    from src.ava.config.training_config import ModelConfig

    try:
        ModelConfig(hidden_size=100, num_attention_heads=8)  # 100 % 8 != 0
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "divisible" in str(e)
        print("✓ hidden_size not divisible by num_attention_heads raises ValueError")


# ============================================================================
# TrainingConfig Validation Tests
# ============================================================================

def test_training_config_valid():
    """Test that valid TrainingConfig passes validation."""
    from src.ava.config.training_config import TrainingConfig

    config = TrainingConfig(
        batch_size=32,
        epochs=5,
        learning_rate=1e-4,
        gradient_accumulation_steps=4,
        warmup_steps=1000,
    )
    assert config.batch_size == 32
    print("✓ Valid TrainingConfig passes validation")


def test_training_config_batch_size_zero():
    """Test that batch_size=0 raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(batch_size=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "batch_size" in str(e)
        print("✓ batch_size=0 raises ValueError")


def test_training_config_batch_size_negative():
    """Test that negative batch_size raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(batch_size=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "batch_size" in str(e)
        print("✓ Negative batch_size raises ValueError")


def test_training_config_gradient_accumulation_zero():
    """Test that gradient_accumulation_steps=0 raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(gradient_accumulation_steps=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "gradient_accumulation_steps" in str(e)
        print("✓ gradient_accumulation_steps=0 raises ValueError")


def test_training_config_learning_rate_negative():
    """Test that negative learning_rate raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(learning_rate=-1e-4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "learning_rate" in str(e)
        print("✓ Negative learning_rate raises ValueError")


def test_training_config_warmup_steps_negative():
    """Test that negative warmup_steps raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(warmup_steps=-100)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "warmup_steps" in str(e)
        print("✓ Negative warmup_steps raises ValueError")


def test_training_config_max_steps_zero():
    """Test that max_steps=0 raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(max_steps=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_steps" in str(e)
        print("✓ max_steps=0 raises ValueError")


def test_training_config_epochs_negative():
    """Test that negative epochs raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(epochs=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "epochs" in str(e)
        print("✓ Negative epochs raises ValueError")


def test_training_config_max_gradient_norm_zero():
    """Test that max_gradient_norm=0 raises ValueError."""
    from src.ava.config.training_config import TrainingConfig

    try:
        TrainingConfig(max_gradient_norm=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_gradient_norm" in str(e)
        print("✓ max_gradient_norm=0 raises ValueError")


def test_training_config_none_values_allowed():
    """Test that None values for optional fields are allowed."""
    from src.ava.config.training_config import TrainingConfig

    config = TrainingConfig(
        batch_size=None,
        epochs=None,
        learning_rate=None,
        max_steps=None,
    )
    assert config.batch_size is None
    print("✓ None values for optional fields are allowed")


def run_all_tests():
    """Run all tests."""
    print("\n=== Config Validation Tests ===\n")

    test_config_imports()

    print("\n--- ModelConfig Tests ---")
    test_model_config_valid()
    test_model_config_hidden_size_zero()
    test_model_config_hidden_size_negative()
    test_model_config_num_layers_zero()
    test_model_config_num_experts_per_token_exceeds_num_experts()
    test_model_config_capacity_factor_zero()
    test_model_config_negative_router_z_loss()
    test_model_config_hidden_size_not_divisible()

    print("\n--- TrainingConfig Tests ---")
    test_training_config_valid()
    test_training_config_batch_size_zero()
    test_training_config_batch_size_negative()
    test_training_config_gradient_accumulation_zero()
    test_training_config_learning_rate_negative()
    test_training_config_warmup_steps_negative()
    test_training_config_max_steps_zero()
    test_training_config_epochs_negative()
    test_training_config_max_gradient_norm_zero()
    test_training_config_none_values_allowed()

    print("\n=== All Config Validation tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
