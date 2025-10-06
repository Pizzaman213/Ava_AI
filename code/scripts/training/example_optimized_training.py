#!/usr/bin/env python3
"""
Example: Optimized Training with All Features

This is a complete, self-contained example showing how to use
all training optimizations in a real training script.

Run:
    cd /project/code
    export PYTHONPATH=/project/code/src:$PYTHONPATH
    python scripts/training/example_optimized_training.py
"""

import sys
sys.path.insert(0, '/project/code/src')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import logging

# Import optimization integration
from Ava.training.optimization_integration import quick_optimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Dummy Model and Dataset (Replace with your real ones) ===

class SimpleModel(nn.Module):
    """Simple transformer-like model for demonstration."""

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        hidden = self.embedding(input_ids)

        for layer in self.layers:
            hidden = layer(hidden)

        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return {'loss': loss, 'logits': logits}


def create_dummy_dataset(num_samples=1000, seq_length=128, vocab_size=1000):
    """Create dummy dataset for demonstration."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))

    # Wrap in dict format
    class DictDataset(Dataset):
        def __init__(self, input_ids, labels):
            self.input_ids = input_ids
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'labels': self.labels[idx]
            }

    return DictDataset(input_ids, labels)


# === Main Training Function ===

def main():
    """Main training function with all optimizations."""

    print("\n" + "=" * 80)
    print("EXAMPLE: OPTIMIZED TRAINING")
    print("=" * 80 + "\n")

    # Configuration
    config = {
        'batch_size': 16,
        'num_epochs': 3,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'log_interval': 10,
        'optimizer_type': 'fused_adam',
        'compile_model': True,
        'compile_mode': 'reduce-overhead',
        'mixed_precision': True,
        'use_sequence_packing': False,  # Not needed for fixed-length dummy data
        'num_workers': 0,  # 0 for this simple example
    }

    # Create model
    logger.info("Creating model...")
    model = SimpleModel(vocab_size=1000, hidden_size=256, num_layers=4)

    # Create dataset
    logger.info("Creating dataset...")
    train_dataset = create_dummy_dataset(num_samples=1000, seq_length=128)
    val_dataset = create_dummy_dataset(num_samples=200, seq_length=128)

    # === APPLY ALL OPTIMIZATIONS (ONE FUNCTION CALL!) ===
    logger.info("\n🚀 Setting up optimized training...")
    setup = quick_optimize(
        model=model,
        train_dataset=train_dataset,
        config=config
    )

    # Extract optimized components
    model = setup['model']
    optimizer = setup['optimizer']
    train_loader = setup['train_loader']
    mp_manager = setup['mp_manager']
    grad_clipper = setup['grad_clipper']
    monitor = setup['monitor']

    logger.info("✅ Optimizations applied!\n")

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # === TRAINING LOOP ===

    logger.info("Starting training...")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Epochs: {config['num_epochs']}")
    logger.info(f"Optimizer: {type(optimizer).__name__}\n")

    model.train()

    for epoch in range(config['num_epochs']):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        logger.info(f"{'='*80}")

        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            # Start monitoring
            if monitor:
                monitor.start_step()

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # === OPTIMIZED FORWARD PASS ===
            with mp_manager.autocast():
                outputs = model(**batch)
                loss = outputs['loss']

            # === OPTIMIZED BACKWARD PASS ===
            scaled_loss = mp_manager.scale_loss(loss)
            scaled_loss.backward()

            # === OPTIMIZED GRADIENT CLIPPING ===
            clip_stats = grad_clipper.clip_gradients(
                model.named_parameters(),
                named=True
            )

            # === OPTIMIZED OPTIMIZER STEP ===
            opt_metrics = mp_manager.step_optimizer(optimizer)

            # Zero gradients
            optimizer.zero_grad()

            # End monitoring
            if monitor:
                batch_size = batch['input_ids'].shape[0]
                seq_len = batch['input_ids'].shape[1]
                stats = monitor.end_step(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    loss=loss.item(),
                    grad_norm=clip_stats.get('grad_norm', 0.0)
                )

            # Logging
            epoch_loss += loss.item()
            num_batches += 1

            if step % config['log_interval'] == 0:
                logger.info(
                    f"  Step {step:4d}: "
                    f"loss={loss.item():.4f}, "
                    f"grad_norm={clip_stats.get('grad_norm', 0):.3f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")

        if monitor:
            summary = monitor.get_summary()
            logger.info(f"  Throughput: {summary['throughput'].get('samples_per_sec', 0):.1f} samples/s")
            logger.info(f"  Throughput: {summary['throughput'].get('tokens_per_sec', 0):.0f} tokens/s")
            if summary['throughput'].get('mfu', 0) > 0:
                logger.info(f"  MFU: {summary['throughput'].get('mfu', 0):.2%}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80 + "\n")

    # Show final stats
    if monitor:
        logger.info("Final Statistics:")
        summary = monitor.get_summary()
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Average throughput: {summary['throughput'].get('tokens_per_sec', 0):.0f} tokens/s")

        memory = summary.get('memory', {})
        if memory:
            logger.info(f"  Peak memory: {memory.get('peak_mb', 0):.1f} MB")
            logger.info(f"  Current memory: {memory.get('current_mb', 0):.1f} MB")

    logger.info("\n🎉 Example completed successfully!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
