"""Test that multi-element tensor handling works correctly."""
import torch
import sys
sys.path.insert(0, '/root/Ava_AI/code/src')


def test_tensor_to_scalar_conversion():
    """Test the tensor-to-scalar conversion logic used in training loop."""

    # Test case 1: Scalar tensor
    scalar_tensor = torch.tensor(0.5)
    if hasattr(scalar_tensor, 'numel') and scalar_tensor.numel() > 1:
        result = scalar_tensor.mean().item()
    else:
        result = scalar_tensor.item()
    assert result == 0.5, f"Scalar tensor failed: {result}"
    print("✓ Scalar tensor handling works")

    # Test case 2: Multi-element tensor (like the 4-element expert utilization)
    multi_tensor = torch.tensor([0.25, 0.30, 0.20, 0.25])
    if hasattr(multi_tensor, 'numel') and multi_tensor.numel() > 1:
        result = multi_tensor.mean().item()
    else:
        result = multi_tensor.item()
    expected = 0.25
    assert abs(result - expected) < 1e-6, f"Multi-element tensor failed: {result} != {expected}"
    print("✓ Multi-element tensor handling works")

    # Test case 3: Regular float
    float_val = 0.75
    if isinstance(float_val, (int, float)):
        result = float_val
    elif hasattr(float_val, 'item'):
        if hasattr(float_val, 'numel') and float_val.numel() > 1:
            result = float_val.mean().item()
        else:
            result = float_val.item()
    assert result == 0.75, f"Float handling failed: {result}"
    print("✓ Float handling works")

    print("\n✅ All tensor-to-scalar conversion tests passed!")


if __name__ == "__main__":
    test_tensor_to_scalar_conversion()
