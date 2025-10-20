#!/usr/bin/env python3
"""
Simple script to test/validate the model on CPU.
This loads the latest checkpoint and runs validation on a small batch of data.
"""

import sys
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset


def load_model_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    print(f"\n{'='*80}")
    print(f"Loading model from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
    elif 'model_config' in checkpoint:
        config_dict = checkpoint['model_config']
    else:
        raise ValueError("No config found in checkpoint")

    # Extract only model-specific config if it's a full training config
    if 'model' in config_dict:
        model_config_dict = config_dict['model']
    else:
        model_config_dict = config_dict

    # Filter to only include valid EnhancedMoEConfig parameters
    valid_params = {
        'vocab_size', 'hidden_size', 'num_layers', 'num_attention_heads', 'intermediate_size',
        'max_position_embeddings', 'num_experts', 'num_experts_per_token', 'expert_capacity_factor',
        'router_type', 'router_aux_loss_coef', 'router_jitter_noise', 'attention_dropout',
        'hidden_dropout', 'dropout', 'layer_norm_eps', 'use_flash_attention', 'use_cache',
        'rope_theta', 'hidden_act', 'initializer_range', 'use_moh', 'use_moa',
        'use_cross_attention', 'use_alibi', 'deepspeed_activation_checkpointing',
        'deepspeed_partition_activations', 'deepspeed_moe_param_groups',
        'entropy_regularization', 'output_diversity_weight', 'eos_logit_bias',
        'eos_token_id', 'min_sequence_length'
    }

    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_params}

    # Convert string values to appropriate types
    if 'layer_norm_eps' in filtered_config and isinstance(filtered_config['layer_norm_eps'], str):
        filtered_config['layer_norm_eps'] = float(filtered_config['layer_norm_eps'])
    if 'initializer_range' in filtered_config and isinstance(filtered_config['initializer_range'], str):
        filtered_config['initializer_range'] = float(filtered_config['initializer_range'])
    if 'rope_theta' in filtered_config and isinstance(filtered_config['rope_theta'], str):
        filtered_config['rope_theta'] = float(filtered_config['rope_theta'])

    # Create model config
    model_config = EnhancedMoEConfig(**filtered_config)

    # Initialize model
    print("Initializing model...")
    model = EnhancedMoEModel(model_config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("No model state dict found in checkpoint")

    model.to(device)
    model.eval()

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint.get('step', 'N/A')}")
    if 'loss' in checkpoint:
        print(f"  Training loss: {checkpoint['loss']:.4f}")

    return model, model_config


def evaluate_model(model, dataloader, device: str = 'cpu', max_batches: int = 50):
    """Evaluate model on validation data."""
    print(f"\n{'='*80}")
    print(f"Running validation on {max_batches} batches")
    print(f"{'='*80}\n")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=max_batches, desc="Validating")):
            if batch_idx >= max_batches:
                break

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            labels = input_ids.clone()

            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']

                if torch.isfinite(loss):
                    total_loss += loss.item()
                    total_tokens += input_ids.numel()
                    num_batches += 1
                else:
                    print(f"⚠️  Warning: Non-finite loss at batch {batch_idx}")

            except Exception as e:
                print(f"❌ Error at batch {batch_idx}: {e}")
                continue

    if num_batches > 0:
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"\n{'='*80}")
        print(f"📊 Validation Results")
        print(f"{'='*80}")
        print(f"  Batches evaluated: {num_batches}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"{'='*80}\n")

        return avg_loss, perplexity
    else:
        print("❌ No valid batches processed")
        return None, None


def create_validation_dataloader(tokenizer_path: str, data_dir: str, batch_size: int = 4, max_length: int = 512):
    """Create a simple validation dataloader."""
    print(f"\n{'='*80}")
    print(f"Creating validation dataloader")
    print(f"{'='*80}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    data_path = Path(data_dir)

    # Try to find validation data
    val_files = list(data_path.glob("*val*.jsonl")) or list(data_path.glob("*test*.jsonl"))

    if not val_files:
        print("⚠️  No validation files found, using first 1000 samples from processed data")
        dataset = load_dataset('json', data_files=str(list(data_path.glob("*_processed.jsonl"))[0]), split='train[:1000]')
    else:
        print(f"Found validation file: {val_files[0]}")
        dataset = load_dataset('json', data_files=str(val_files[0]), split='train')

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for CPU testing
        pin_memory=False
    )

    print(f"✓ Created dataloader with {len(dataset)} samples")

    return dataloader


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test model on CPU')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='/project/code/configs/gpu/small.yaml', help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='/project/code/data/processed', help='Path to data directory')
    parser.add_argument('--tokenizer', type=str, default='/project/code/models/tokenizer/enhanced-27000', help='Path to tokenizer')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for validation')
    parser.add_argument('--max-batches', type=int, default=50, help='Maximum number of batches to evaluate')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')

    args = parser.parse_args()

    # Find checkpoint if not specified
    if not args.checkpoint:
        print("No checkpoint specified, searching for latest...")
        checkpoint_path = Path('/project/code/outputs/runs/run_20251019_153956_49caa62a/checkpoints/latest_model.pt')
        if not checkpoint_path.exists():
            print("❌ No checkpoint found. Please specify --checkpoint")
            return
        args.checkpoint = str(checkpoint_path)

    # Load model
    model, model_config = load_model_checkpoint(args.checkpoint, device='cpu')

    # Create dataloader
    dataloader = create_validation_dataloader(
        tokenizer_path=args.tokenizer,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Run validation
    avg_loss, perplexity = evaluate_model(
        model,
        dataloader,
        device='cpu',
        max_batches=args.max_batches
    )

    if avg_loss is not None:
        print("✅ Validation completed successfully!")
    else:
        print("❌ Validation failed")


if __name__ == '__main__':
    main()
