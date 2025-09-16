#!/usr/bin/env python3
"""
Main training script for Qwen MoE++ model.

This script trains the model using data from /project/code/data/pretraining/processed/
and configurations from /project/code/configs/

Usage:
    # Basic training with CPU config
    python train.py --config /project/code/configs/cpu/small.yaml

    # Training with custom parameters
    python train.py --config /project/code/configs/cpu/small.yaml --batch-size 4 --epochs 3

    # Resume from checkpoint
    python train.py --config /project/code/configs/cpu/small.yaml --resume outputs/checkpoint.pt

Examples:
    # Train small model on CPU
    python train.py --config /project/code/configs/cpu/small.yaml --output-dir outputs/small_model

    # Train with specific data directory
    python train.py --config /project/code/configs/cpu/small.yaml --data-dir /project/code/data/pretraining/processed
"""

import argparse
import sys
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.data import create_dataloaders
from src.Ava.data_streaming import create_streaming_dataloaders
from transformers import AutoTokenizer


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']

        # Add auxiliary losses if present
        if 'aux_losses' in outputs:
            for aux_loss in outputs['aux_losses'].values():
                if aux_loss is not None and aux_loss.numel() == 1:
                    loss = loss + 0.01 * aux_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        avg_loss = total_loss / num_batches

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Log every 100 steps
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(f"  Step {batch_idx}: Loss = {avg_loss:.4f}")

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Qwen MoE++ model')

    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (e.g., /project/code/configs/cpu/small.yaml)')

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                       default='/project/code/data/pretraining/processed',
                       help='Directory containing preprocessed training data')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of training samples to load (for testing)')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Use streaming data loader for large datasets (default: True)')
    parser.add_argument('--no-streaming', dest='streaming', action='store_false',
                       help='Disable streaming and load all data into memory')
    parser.add_argument('--buffer-size', type=int, default=1000,
                       help='Buffer size for streaming data loader')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='/project/code/outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"🔧 Using device: {device}")

    # Load configuration
    print(f"📋 Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract model config
    model_config_dict = config_dict.get('model', {})
    training_config = config_dict.get('training', {})

    # Override with command line arguments
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    if args.epochs:
        training_config['num_epochs'] = args.epochs
    if args.learning_rate:
        training_config['learning_rate'] = args.learning_rate

    # Set defaults if not in config
    batch_size = training_config.get('batch_size', 4)
    num_epochs = training_config.get('num_epochs', 3)
    learning_rate = float(training_config.get('learning_rate', 5e-4))

    print(f"📊 Training configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max length: {args.max_length}")
    print(f"  - Data directory: {args.data_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    import logging
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    # Initialize tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloaders
    print(f"📚 Loading data from {args.data_dir}...")
    if args.streaming:
        print("🌊 Using streaming data loader (memory efficient)")
        train_loader, val_loader = create_streaming_dataloaders(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=args.max_length,
            data_dir=args.data_dir,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            max_samples=args.max_samples,
            buffer_size=args.buffer_size
        )
    else:
        print("⚠️ Loading all data into memory (use --streaming for large datasets)")
        train_loader, val_loader = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=args.max_length,
            data_dir=args.data_dir,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            max_samples=args.max_samples
        )

    # Initialize model
    print("🤖 Initializing model...")
    # Filter config to only include valid fields
    from dataclasses import fields
    valid_fields = {f.name for f in fields(EnhancedMoEConfig)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_fields}
    model_config = EnhancedMoEConfig(**filtered_config)
    model = EnhancedMoEModel(model_config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"  Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n📍 Epoch {epoch}/{num_epochs}")

        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        train_time = time.time() - start_time

        print(f"  Training loss: {train_loss:.4f}")
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"  Validation loss: {val_loss:.4f}")

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'best_model.pt'
            print(f"  💾 Saving best model to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config_dict
            }, checkpoint_path)

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            print(f"  💾 Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config_dict
            }, checkpoint_path)

    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"📂 Model saved to: {output_dir}")
    print("="*60)

    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config_dict
    }, final_path)
    print(f"\n💾 Final model saved to {final_path}")

    print("\n🎯 To generate text with your trained model:")
    print(f"python /project/code/scripts/generation/generate.py --model-path {final_path} --prompt 'Your text here'")


if __name__ == "__main__":
    main()