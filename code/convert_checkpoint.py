#!/usr/bin/env python3
"""Convert DeepSpeed checkpoint to standard PyTorch format for generation."""

import torch
import sys
from pathlib import Path

def convert_deepspeed_checkpoint(deepspeed_path, output_path):
    """Convert DeepSpeed checkpoint to standard format."""
    print(f"Loading DeepSpeed checkpoint from {deepspeed_path}...")

    # Load the DeepSpeed checkpoint
    ds_checkpoint = torch.load(deepspeed_path, map_location='cpu', weights_only=False)

    # Extract model state dict
    if 'module' in ds_checkpoint:
        model_state_dict = ds_checkpoint['module']
    else:
        raise ValueError("No 'module' key found in DeepSpeed checkpoint")

    # Load the metadata checkpoint to get config
    meta_checkpoint_path = Path(deepspeed_path).parent.parent / "model.pt"
    if meta_checkpoint_path.exists():
        meta_checkpoint = torch.load(meta_checkpoint_path, map_location='cpu', weights_only=False)
        config = meta_checkpoint.get('config', {})
    else:
        config = {}

    # Create new checkpoint with proper format
    checkpoint = {
        'model_state_dict': model_state_dict,
        'config': config
    }

    # Save the converted checkpoint
    print(f"Saving converted checkpoint to {output_path}...")
    torch.save(checkpoint, output_path)
    print(" Conversion complete!")

    return checkpoint

if __name__ == "__main__":
    deepspeed_path = "/project/code/outputs/runs/run_20250922_233206_a23bfbf7/checkpoints/step_257000/step_257000/mp_rank_00_model_states.pt"
    output_path = "/project/code/outputs/runs/run_20250922_233206_a23bfbf7/checkpoints/step_257000/converted_model.pt"

    convert_deepspeed_checkpoint(deepspeed_path, output_path)