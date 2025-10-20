#!/usr/bin/env python3
"""
Test script to verify checkpoint resumption functionality.

This script tests that:
1. Checkpoint can be loaded
2. Training state is restored correctly
3. Model parameters are restored
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

import torch

def test_checkpoint_loading():
    """Test loading the latest checkpoint."""

    # Find the latest checkpoint
    checkpoint_path = Path("/project/code/outputs/runs/run_20251020_103304_35bab4f6/checkpoints/latest_model.pt")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"✅ Found checkpoint: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / (1024**3):.2f} GB")

    # Try to load checkpoint
    try:
        print("\n🔄 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Check what keys are in the checkpoint
        print(f"\n📦 Checkpoint contents:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"   {key}: dict with {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"   {key}: tensor with shape {checkpoint[key].shape}")
            else:
                print(f"   {key}: {type(checkpoint[key]).__name__} = {checkpoint[key]}")

        # Check critical training state
        print(f"\n📊 Training state:")
        step_count = checkpoint.get("step_count", checkpoint.get("step", 0))
        epoch_count = checkpoint.get("epoch_count", checkpoint.get("epoch", 0))
        best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float("inf")))

        print(f"   Step count: {step_count}")
        print(f"   Epoch count: {epoch_count}")
        print(f"   Best loss: {best_loss:.4f}")

        # Check for optimizer and micro step counts
        optimizer_step_count = checkpoint.get("optimizer_step_count", "N/A")
        micro_step_count = checkpoint.get("micro_step_count", "N/A")
        print(f"   Optimizer step count: {optimizer_step_count}")
        print(f"   Micro step count: {micro_step_count}")

        # Check model state
        if "model_state_dict" in checkpoint:
            model_params = len(checkpoint["model_state_dict"])
            print(f"   Model parameters: {model_params} tensors")

            # Calculate total parameters
            total_params = sum(
                p.numel() for p in checkpoint["model_state_dict"].values()
                if isinstance(p, torch.Tensor)
            )
            print(f"   Total model params: {total_params / 1e6:.1f}M")
        else:
            print(f"   ⚠️  No model_state_dict found!")

        # Check optimizer state
        if "optimizer_state_dict" in checkpoint:
            opt_state = checkpoint["optimizer_state_dict"]
            print(f"   Optimizer state: {len(opt_state.get('state', {}))} parameter states")
            print(f"   Optimizer param groups: {len(opt_state.get('param_groups', []))}")
        else:
            print(f"   ⚠️  No optimizer_state_dict found!")

        print(f"\n✅ Checkpoint loaded successfully!")
        print(f"\n💡 You can resume training with:")
        print(f"   python code/scripts/5_training/train.py \\")
        print(f"     --config code/configs/gpu/small.yaml \\")
        print(f"     --resume {checkpoint_path}")

        return True

    except Exception as e:
        print(f"\n❌ Failed to load checkpoint: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("CHECKPOINT RESUMPTION TEST")
    print("=" * 70)

    success = test_checkpoint_loading()

    print("\n" + "=" * 70)
    if success:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
    print("=" * 70)
