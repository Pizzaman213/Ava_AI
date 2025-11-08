#!/usr/bin/env python3
"""
Example: Training a Sparse Mixture of Experts Model

This script demonstrates how to train a sparse MoE model using the
OptimizedMoETransformer architecture with production-ready features.

Features demonstrated:
- SparseMoELayer configuration
- Mixtral and DeepSeek routing
- Load balancing and auxiliary losses
- Expert usage monitoring
- Training loop integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
import yaml

from src.Ava.models.moe_model import OptimizedMoETransformer, OptimizedMoEConfig


def load_moe_config(config_path: str) -> OptimizedMoEConfig:
    """Load MoE configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract model config
    model_config = config_dict.get('model', {})

    # Create OptimizedMoEConfig
    config = OptimizedMoEConfig(
        vocab_size=model_config.get('vocab_size', 32000),
        hidden_size=model_config.get('hidden_size', 2048),
        num_layers=model_config.get('num_layers', 16),
        num_attention_heads=model_config.get('num_attention_heads', 16),
        intermediate_size=model_config.get('intermediate_size', 7168),
        max_position_embeddings=model_config.get('max_position_embeddings', 2048),
        num_experts=model_config.get('num_experts', 8),
        num_experts_per_token=model_config.get('num_experts_per_token', 2),
        router_type=model_config.get('router_type', 'mixtral'),
        capacity_factor=model_config.get('capacity_factor', 1.25),
        expert_dropout=model_config.get('expert_dropout', 0.0),
        activation=model_config.get('activation', 'swiglu'),
        use_grouped_gemm=model_config.get('use_grouped_gemm', True),
        use_triton_kernels=model_config.get('use_triton_kernels', True),
        use_torch_compile=model_config.get('use_torch_compile', False),
        gradient_checkpointing=model_config.get('gradient_checkpointing', False),
        router_z_loss_coef=model_config.get('router_z_loss_coef', 0.001),
        load_balance_loss_coef=model_config.get('load_balance_loss_coef', 0.01),
        diversity_loss_coef=model_config.get('diversity_loss_coef', 0.001),
        expert_dropout_loss_coef=model_config.get('expert_dropout_loss_coef', 0.0),
        router_jitter_noise=model_config.get('router_jitter_noise', 0.01),
        use_shared_expert=model_config.get('use_shared_expert', False),
        attention_dropout=model_config.get('attention_dropout', 0.0),
        dropout=model_config.get('dropout', 0.0),
        layer_norm_eps=model_config.get('layer_norm_eps', 1e-5),
        initializer_range=model_config.get('initializer_range', 0.02),
    )

    return config


def create_dummy_dataset(vocab_size: int, seq_len: int, num_samples: int = 1000):
    """Create dummy dataset for demonstration."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(input_ids, labels)


def train_moe_model(
    config: OptimizedMoEConfig,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    log_interval: int = 10,
) -> OptimizedMoETransformer:
    """
    Train MoE model with monitoring of expert usage and auxiliary losses.

    Args:
        config: Model configuration
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        log_interval: Steps between logging

    Returns:
        Trained model
    """
    print("=" * 70)
    print("TRAINING SPARSE MoE MODEL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Experts per token: {config.num_experts_per_token}")
    print(f"  Router type: {config.router_type}")
    print(f"  Grouped GEMM: {config.use_grouped_gemm}")
    print(f"  Device: {device}")
    print()

    # Create model
    model = OptimizedMoETransformer(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_aux_loss = 0.0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(input_ids, labels=labels, return_dict=True)
            loss = output['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            epoch_loss += loss.item()

            # Calculate auxiliary loss contribution
            if output['aux_info']:
                aux_loss_sum = sum(info['aux_loss'].item() for info in output['aux_info']) / len(output['aux_info'])
                epoch_aux_loss += aux_loss_sum
            else:
                aux_loss_sum = 0.0

            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_aux_loss = epoch_aux_loss / (batch_idx + 1)

                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Step {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Aux Loss: {aux_loss_sum:.6f}")

                # Expert usage stats (from first layer)
                if output['aux_info'] and len(output['aux_info']) > 0:
                    layer_0_info = output['aux_info'][0]
                    if 'balance_score' in layer_0_info:
                        balance_score = layer_0_info['balance_score'].item()
                        print(f"  Balance Score (Layer 0): {balance_score:.4f} (1.0 = perfect)")

            global_step += 1

        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_aux_loss = epoch_aux_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average Auxiliary Loss: {avg_epoch_aux_loss:.6f}")

        # Get expert usage statistics
        print(f"\n  Expert Usage Statistics:")
        for i, layer in enumerate(model.layers):
            stats = layer.moe.get_expert_usage_stats()
            if 'expert_usage_normalized' in stats:
                usage = stats['expert_usage_normalized']
                print(f"    Layer {i}: min={usage.min().item():.4f}, "
                      f"max={usage.max().item():.4f}, "
                      f"std={usage.std().item():.4f}")

        print()

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return model


def main():
    """Main training example."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Sparse MoE Model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/moe/small_moe.yaml',
        help='Path to MoE configuration file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=128,
        help='Sequence length'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of training samples (dummy data)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='Path to save trained model'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_moe_config(args.config)

    # Create dataset
    print(f"Creating dummy dataset ({args.num_samples} samples)...")
    dataset = create_dummy_dataset(config.vocab_size, args.seq_len, args.num_samples)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train model
    model = train_moe_model(
        config=config,
        train_loader=train_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )

    # Save model if requested
    if args.save_path:
        print(f"\nSaving model to {args.save_path}...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, args.save_path)
        print("Model saved successfully!")


if __name__ == "__main__":
    main()
