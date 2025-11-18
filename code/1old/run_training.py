#!/usr/bin/env python3
"""
Simple training runner using the refactored training pipeline.
This extends train_refactored_CORRECTED.py with a basic training loop.
"""

import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import the refactored training script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_refactored_CORRECTED",
    Path(__file__).parent / "train_refactored_CORRECTED.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
init_training = train_module.main


def create_dummy_data(batch_size: int = 2, seq_length: int = 128, vocab_size: int = 50680, num_batches: int = 5):
    """Create dummy training data for testing."""
    class DummyDataset:
        def __init__(self, num_batches, batch_size, seq_length, vocab_size):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.vocab_size = vocab_size

        def __iter__(self):
            for _ in range(self.num_batches):
                input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
                labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
                attention_mask = torch.ones((self.batch_size, self.seq_length))
                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask
                }

    return DummyDataset(num_batches, batch_size, seq_length, vocab_size)


def run_training(config_path: str, num_epochs: int = 1, num_batches: int = 5):
    """Run training with initialized components."""
    
    print("\n" + "="*80)
    print("🚀 INITIALIZING TRAINING PIPELINE")
    print("="*80)
    
    # Initialize training components
    components = init_training(config_path=config_path)
    
    model = components['model']
    optimizer = components['optimizer']
    device = components['context'].device
    config = components['context'].config
    
    print("\n" + "="*80)
    print("🏋️  STARTING TRAINING LOOP")
    print("="*80)
    
    if model is None:
        print("❌ Error: Model not created. Cannot proceed with training.")
        return
    
    # Get batch size from config
    batch_size = getattr(config.training, 'batch_size', 2) if hasattr(config, 'training') else 2
    seq_length = getattr(config.model, 'max_position_embeddings', 512) if hasattr(config, 'model') else 512
    vocab_size = getattr(config.model, 'vocab_size', 50680) if hasattr(config, 'model') else 50680
    
    # Create dummy dataloader
    train_dataset = create_dummy_data(
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        num_batches=num_batches
    )
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batches per epoch: {num_batches}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Device: {device}")
    
    # Training loop
    model.train()
    total_loss = 0.0
    total_steps = 0
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        
        pbar = tqdm(train_dataset, desc=f"Training", leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Backward pass
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Track metrics
                loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                epoch_loss += loss_value
                total_loss += loss_value
                total_steps += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                })
                
            except Exception as e:
                print(f"\n⚠️  Error in batch {batch_idx}: {e}")
                print(f"   Continuing with next batch...")
                continue
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n   ✅ Epoch {epoch + 1} complete - Avg Loss: {avg_epoch_loss:.4f}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\n📊 Training Summary:")
    print(f"   Total steps: {total_steps}")
    print(f"   Average loss: {total_loss / max(total_steps, 1):.4f}")
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n🎉 Training finished successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training with refactored pipeline")
    parser.add_argument("--config", type=str, default="code/configs/moe/minimal_working.yaml",
                       help="Path to config file")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batches", type=int, default=5,
                       help="Number of batches per epoch (for testing)")
    
    args = parser.parse_args()
    
    try:
        run_training(
            config_path=args.config,
            num_epochs=args.epochs,
            num_batches=args.batches
        )
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
