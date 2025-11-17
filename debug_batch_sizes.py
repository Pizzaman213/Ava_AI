"""Debug script to check batch sizes from the dataloader."""

import sys
sys.path.insert(0, '/project/code')

import torch
from src.Ava.config.training_config import load_config
from src.Ava.data.pretokenized_loader import create_dataloaders

def debug_batch_sizes(config_path: str):
    """Check what batch sizes are being created."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print(f"\nConfig settings:")
    print(f"  batch_size: {config.training.batch_size}")
    print(f"  max_length: {config.data.max_length}")
    print(f"  use_sequence_packing: {config.data.use_sequence_packing}")

    print(f"\nCreating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(config)

        print(f"\nChecking first 5 batches:")
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break

            input_ids = batch['input_ids']
            print(f"\nBatch {i}:")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  input_ids dtype: {input_ids.dtype}")
            print(f"  input_ids device: {input_ids.device}")
            print(f"  Expected size: {input_ids.shape[0] * input_ids.shape[1] * input_ids.element_size() / 1024 / 1024:.2f} MB")

            # Check for abnormal values
            if input_ids.shape[0] > 1000 or input_ids.shape[1] > 10000:
                print(f"  ⚠️ WARNING: Abnormally large batch dimensions!")
                print(f"  Max value in input_ids: {input_ids.max().item()}")
                print(f"  Min value in input_ids: {input_ids.min().item()}")

        print(f"\n✓ Batch size check complete")

    except Exception as e:
        print(f"\n✗ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_batch_sizes("/tmp/test_compile_fix.yaml")
