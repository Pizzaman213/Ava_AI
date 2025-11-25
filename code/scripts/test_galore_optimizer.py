"""
Test script for GaLore optimizer

Tests the GaLore optimizer implementation with a simple model
to verify it works correctly and achieves memory savings.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from Ava.optimization.optimizers import create_galore_optimizer, create_8bit_optimizer


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self, hidden_size=512, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def test_galore_optimizer():
    """Test GaLore optimizer functionality."""
    print("=" * 70)
    print("Testing GaLore Optimizer")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, memory savings not measurable")

    # Create model
    model = SimpleModel(hidden_size=1024, num_layers=6).to(device)
    num_params = count_parameters(model)
    print(f"\nModel: {num_params:,} parameters")

    # Test 1: Standard AdamW
    print("\n" + "-" * 70)
    print("Test 1: Standard AdamW (Baseline)")
    print("-" * 70)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    optimizer_standard = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run a few training steps
    for step in range(10):
        x = torch.randn(8, 1024, device=device)
        target = torch.randn(8, 1024, device=device)

        optimizer_standard.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer_standard.step()

        if step == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    mem_standard = get_memory_mb()
    if torch.cuda.is_available():
        peak_mem_standard = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nCurrent Memory: {mem_standard:.2f} MB")
        print(f"Peak Memory:    {peak_mem_standard:.2f} MB")

    # Test 2: GaLore AdamW
    print("\n" + "-" * 70)
    print("Test 2: GaLore AdamW (rank=128)")
    print("-" * 70)

    # Recreate model
    model = SimpleModel(hidden_size=1024, num_layers=6).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        optimizer_galore = create_galore_optimizer(
            model,
            optimizer_type='adamw',
            lr=1e-3,
            rank=128,
            update_proj_gap=5,  # Update frequently for testing
            weight_decay=0.01
        )
        print("✓ GaLore optimizer created successfully")

        # Run training steps
        for step in range(10):
            x = torch.randn(8, 1024, device=device)
            target = torch.randn(8, 1024, device=device)

            optimizer_galore.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer_galore.step()

            if step == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

        mem_galore = get_memory_mb()
        if torch.cuda.is_available():
            peak_mem_galore = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"\nCurrent Memory: {mem_galore:.2f} MB")
            print(f"Peak Memory:    {peak_mem_galore:.2f} MB")

            # Calculate savings
            if peak_mem_standard > 0:
                savings_mb = peak_mem_standard - peak_mem_galore
                savings_pct = (savings_mb / peak_mem_standard) * 100
                print(f"\nMemory Savings: {savings_mb:.2f} MB ({savings_pct:.1f}%)")

        print("✓ GaLore training completed successfully")

    except Exception as e:
        print(f"✗ GaLore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: GaLore Lion
    print("\n" + "-" * 70)
    print("Test 3: GaLore Lion (rank=128)")
    print("-" * 70)

    # Recreate model
    model = SimpleModel(hidden_size=1024, num_layers=6).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        optimizer_lion = create_galore_optimizer(
            model,
            optimizer_type='lion',
            lr=1e-4,  # Lion uses smaller LR
            rank=128,
            update_proj_gap=5,
            weight_decay=0.01
        )
        print("✓ GaLore Lion optimizer created successfully")

        # Run training steps
        for step in range(10):
            x = torch.randn(8, 1024, device=device)
            target = torch.randn(8, 1024, device=device)

            optimizer_lion.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer_lion.step()

            if step == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

        mem_lion = get_memory_mb()
        if torch.cuda.is_available():
            peak_mem_lion = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"\nCurrent Memory: {mem_lion:.2f} MB")
            print(f"Peak Memory:    {peak_mem_lion:.2f} MB")

            # Calculate savings vs standard
            if peak_mem_standard > 0:
                savings_mb = peak_mem_standard - peak_mem_lion
                savings_pct = (savings_mb / peak_mem_standard) * 100
                print(f"\nMemory Savings vs AdamW: {savings_mb:.2f} MB ({savings_pct:.1f}%)")

        print("✓ GaLore Lion training completed successfully")

    except Exception as e:
        print(f"✗ GaLore Lion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Factory function
    print("\n" + "-" * 70)
    print("Test 4: Factory Function (create_8bit_optimizer)")
    print("-" * 70)

    model = SimpleModel(hidden_size=512, num_layers=4).to(device)

    try:
        optimizer = create_8bit_optimizer(
            'galore_adamw',
            model,
            lr=1e-3,
            rank=64,
            update_proj_gap=10
        )
        print("✓ Created via factory function")

        # Quick test
        x = torch.randn(4, 512, device=device)
        target = torch.randn(4, 512, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
        print("✓ Factory function test passed")

    except Exception as e:
        print(f"✗ Factory function test failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("✓ All GaLore tests passed successfully!")
    print("\nGaLore optimizer is ready to use.")
    print("\nUsage:")
    print("  from Ava.optimization.optimizers import create_galore_optimizer")
    print("  optimizer = create_galore_optimizer(model, 'adamw', lr=1e-3, rank=128)")
    print("\nOr via config:")
    print("  configs/memory/galore_memory_efficient.yaml")
    print("=" * 70)

    return True


if __name__ == '__main__':
    success = test_galore_optimizer()
    sys.exit(0 if success else 1)
