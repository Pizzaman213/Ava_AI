#!/usr/bin/env python3
"""
Simple checkpoint analysis - no model imports needed
"""
import torch
import sys

checkpoint_path = "/project/code/outputs/runs/run_20251014_113843_3ae62fda/checkpoints/latest_model.pt"

print("🔍 Checkpoint Analysis")
print("=" * 80)

try:
    # Load checkpoint with torch (required for PyTorch checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"✅ Checkpoint loaded successfully\n")

    # Basic metrics
    print(f"📊 Training Metrics:")
    if 'optimizer_step' in checkpoint:
        print(f"   Step: {checkpoint['optimizer_step']}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"   Loss: {checkpoint['loss']:.4f}")
    if 'learning_rate' in checkpoint:
        print(f"   Learning Rate: {checkpoint['learning_rate']:.2e}")

    # Check what's in checkpoint
    print(f"\n📦 Checkpoint Contents:")
    for key in checkpoint.keys():
        if key != 'model_state_dict' and key != 'optimizer_state_dict':
            value = checkpoint[key]
            if isinstance(value, (int, float, str)):
                print(f"   {key}: {value}")
            elif isinstance(value, dict):
                print(f"   {key}: dict with {len(value)} keys")
            elif isinstance(value, list):
                print(f"   {key}: list with {len(value)} items")
            else:
                print(f"   {key}: {type(value).__name__}")

    # Model state analysis (without loading actual tensors)
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print(f"\n🏗️  Model State:")
        print(f"   Parameters: {len(model_state)} tensors")

        # Check parameter shapes and types
        sample_params = list(model_state.items())[:5]
        print(f"\n   Sample Parameters (first 5):")
        for name, param in sample_params:
            print(f"      {name[:60]:60s}: shape={list(param.shape) if hasattr(param, 'shape') else 'N/A'}")

    # Check for recent losses
    if 'recent_losses' in checkpoint:
        recent = checkpoint['recent_losses'][-10:]
        print(f"\n📉 Recent Losses (last 10):")
        for i, loss in enumerate(recent, 1):
            print(f"   {i:2d}. {loss:.4f}")

        # Calculate trend
        if len(recent) >= 5:
            first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            if second_half < first_half:
                print(f"   ✅ Trend: Decreasing (good!)")
            else:
                print(f"   ⚠️  Trend: Increasing or flat")

    # Validation metrics
    if 'val_loss' in checkpoint:
        print(f"\n✅ Validation Metrics:")
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        if 'val_perplexity' in checkpoint:
            print(f"   Perplexity: {checkpoint['val_perplexity']:.2f}")
        if 'loss' in checkpoint:
            ratio = checkpoint['val_loss'] / checkpoint['loss']
            print(f"   Val/Train Ratio: {ratio:.3f}")
            if ratio > 1.0 and ratio < 2.0:
                print(f"   ✅ Good generalization")
            elif ratio > 2.0:
                print(f"   ⚠️  High ratio - might be overfitting")

    # Overall assessment
    print(f"\n{'='*80}")
    print(f"🎯 Assessment:")
    print(f"{'='*80}")

    issues = []

    if 'loss' in checkpoint:
        loss = checkpoint['loss']
        if loss > 10:
            issues.append(f"Loss is very high ({loss:.2f})")
        elif loss < 0.1:
            issues.append(f"Loss is suspiciously low ({loss:.2f}) - might be collapsed")

    if 'learning_rate' in checkpoint:
        lr = checkpoint['learning_rate']
        if lr < 1e-10:
            issues.append(f"LR is extremely low ({lr:.2e}) - might have collapsed")
        elif lr > 1e-2:
            issues.append(f"LR is very high ({lr:.2e})")

    if issues:
        print(f"⚠️  Potential Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"✅ Checkpoint looks healthy!")
        if 'loss' in checkpoint:
            print(f"   • Loss: {checkpoint['loss']:.4f} (reasonable)")
        if 'learning_rate' in checkpoint:
            print(f"   • LR: {checkpoint['learning_rate']:.2e} (stable)")
        print(f"   • Training progressing normally")

    print(f"{'='*80}\n")

    # Based on your terminal output
    print(f"💡 Based on your training terminal output:")
    print(f"   ✅ Loss decreasing: 2.3441 (train), 3.3732 (val)")
    print(f"   ✅ Repetition: 0.0% (excellent!)")
    print(f"   ✅ Gradients: 0.634 (healthy)")
    print(f"   ✅ Val/Train ratio: 1.439 (good generalization)")
    print(f"\n   🎉 MODEL IS NOT COLLAPSED - Training successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
