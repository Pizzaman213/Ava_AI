#!/usr/bin/env python3
"""
Inspect the model checkpoint to understand its structure
"""
import torch
import sys

def inspect_checkpoint(checkpoint_path):
    """Load and inspect checkpoint contents"""
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # Print checkpoint structure
    print(f"\n📊 Checkpoint type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"📝 Keys in checkpoint: {list(checkpoint.keys())}")

        # Check for common patterns
        if 'module' in checkpoint:
            module_data = checkpoint['module']
            print(f"\n🔧 Module type: {type(module_data)}")
            if isinstance(module_data, dict):
                print(f"📝 Module keys (first 10): {list(module_data.keys())[:10]}")

        # Check if this is a DeepSpeed checkpoint
        if any(key in checkpoint for key in ['ds_config', 'ds_version', 'param_shapes']):
            print("🚀 This appears to be a DeepSpeed checkpoint")

            if 'param_shapes' in checkpoint and checkpoint['param_shapes'] is not None:
                param_shapes = checkpoint['param_shapes']
                print(f"\n📐 Parameter shapes (first 10):")
                for i, (name, shape) in enumerate(param_shapes.items()):
                    if i >= 10:
                        print(f"... and {len(param_shapes) - 10} more parameters")
                        break
                    print(f"  {name}: {shape}")
            else:
                print("📐 Parameter shapes not available or None")

        # Look for model state dict
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                model_state = checkpoint[key]
                print(f"\n🎯 Found model state under key '{key}'")
                if isinstance(model_state, dict):
                    print(f"📝 Model state keys (first 10): {list(model_state.keys())[:10]}")
                break

    # Check file size
    import os
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"\n📏 Checkpoint size: {size_mb:.1f} MB")

if __name__ == "__main__":
    checkpoint_path = "/project/code/outputs/checkpoint_step_5000/step_5000/mp_rank_00_model_states.pt"
    inspect_checkpoint(checkpoint_path)