#!/usr/bin/env python3
"""
Diagnose NaN/Inf loss issues in training.
This script checks:
1. Data validity
2. Model initialization
3. Forward pass stability
4. Mixed precision issues
"""

import torch
import torch.nn as nn
import sys
import yaml
from pathlib import Path

# Add code to path
sys.path.insert(0, str(Path(__file__).parent / "code" / "src"))

from Ava.config.training_config import TrainingConfig
from Ava.models.moe_model import EnhancedMoEModel
from Ava.data.dataloader import DataManager

def check_tensor_health(tensor, name="tensor"):
    """Check if a tensor has NaN or Inf values."""
    if tensor is None:
        print(f"  {name}: None")
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.min().item() if not has_nan and not has_inf else float('nan')
    max_val = tensor.max().item() if not has_nan and not has_inf else float('nan')
    mean_val = tensor.float().mean().item() if not has_nan and not has_inf else float('nan')

    status = "✓" if not (has_nan or has_inf) else "✗"
    print(f"  {status} {name}: shape={tuple(tensor.shape)}, min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, has_nan={has_nan}, has_inf={has_inf}")

    return not (has_nan or has_inf)

def diagnose_training(config_path: str):
    """Run comprehensive diagnostics on training setup."""

    print("="*80)
    print("NaN/Inf Loss Diagnostic Tool")
    print("="*80)

    # 1. Load configuration
    print("\n[1] Loading configuration...")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)

    print(f"  ✓ Config loaded from {config_path}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Mixed precision: {config.hardware.mixed_precision}")
    print(f"  - Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  - Max grad norm: {config.training.max_grad_norm}")

    # 2. Check data
    print("\n[2] Checking data integrity...")
    try:
        data_manager = DataManager(config)
        train_loader, _ = data_manager.get_dataloaders()

        print(f"  ✓ Data loader created successfully")

        # Get a batch
        batch = next(iter(train_loader))
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        print(f"  ✓ Batch loaded successfully")
        print(f"    - Batch size: {input_ids.shape[0]}")
        print(f"    - Sequence length: {input_ids.shape[1]}")

        # Check for validity
        check_tensor_health(input_ids, "input_ids")
        check_tensor_health(attention_mask, "attention_mask")
        check_tensor_health(labels, "labels")

        # Check token ID ranges
        print(f"  - Input token range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        print(f"  - Label token range: [{labels.min().item()}, {labels.max().item()}]")
        print(f"  - Vocab size expected: {config.model.vocab_size}")

        if input_ids.max().item() >= config.model.vocab_size:
            print(f"  ✗ ERROR: Input tokens exceed vocab size!")
        if labels.max().item() >= config.model.vocab_size:
            print(f"  ✗ ERROR: Label tokens exceed vocab size!")

    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Initialize model
    print("\n[3] Initializing model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  - Using device: {device}")

        model = EnhancedMoEModel(config.model)
        model = model.to(device)

        print(f"  ✓ Model initialized successfully")

        # Check model weights
        print("\n  Checking model weight initialization...")
        param_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                if has_nan or has_inf:
                    print(f"  ✗ {name}: has_nan={has_nan}, has_inf={has_inf}")
                else:
                    param_stats[name] = {
                        'min': param.min().item(),
                        'max': param.max().item(),
                        'mean': param.float().mean().item(),
                        'std': param.float().std().item()
                    }

        # Show stats for a few key layers
        print("\n  Key layer statistics:")
        key_layers = ['token_embedding', 'lm_head', 'layers.0']
        for layer_prefix in key_layers:
            matching = [k for k in param_stats.keys() if layer_prefix in k]
            if matching:
                k = matching[0]
                stats = param_stats[k]
                print(f"    {k[:50]}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, range=[{stats['min']:.6f}, {stats['max']:.6f}]")

    except Exception as e:
        print(f"  ✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Test forward pass with different precisions
    print("\n[4] Testing forward pass...")

    # Move batch to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # Test with float32 first
    print("\n  Testing with FP32...")
    try:
        model = model.float()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']

        all_good = check_tensor_health(logits, "logits")

        # Compute loss manually
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        check_tensor_health(loss.unsqueeze(0), "loss")

        if all_good:
            print(f"  ✓ FP32 forward pass: SUCCESS")
        else:
            print(f"  ✗ FP32 forward pass: FAILED (NaN/Inf in outputs)")

    except Exception as e:
        print(f"  ✗ FP32 forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # Test with bf16
    if config.hardware.mixed_precision == 'bf16' and torch.cuda.is_available():
        print("\n  Testing with BF16...")
        try:
            model = model.bfloat16()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']

            all_good = check_tensor_health(logits, "logits")

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            check_tensor_health(loss.unsqueeze(0), "loss")

            if all_good:
                print(f"  ✓ BF16 forward pass: SUCCESS")
            else:
                print(f"  ✗ BF16 forward pass: FAILED (NaN/Inf in outputs)")
                print(f"    This suggests a mixed precision issue!")

        except Exception as e:
            print(f"  ✗ BF16 forward pass failed: {e}")
            import traceback
            traceback.print_exc()

    # 5. Check for common issues
    print("\n[5] Checking for common issues...")

    issues = []

    # Check learning rate
    if config.training.learning_rate > 1e-3:
        issues.append(f"Learning rate ({config.training.learning_rate}) is very high - may cause instability")

    # Check initializer range
    if config.model.initializer_range > 0.05:
        issues.append(f"Initializer range ({config.model.initializer_range}) is large - may cause initial instability")

    # Check gradient clipping
    if config.training.max_grad_norm > 5.0:
        issues.append(f"Gradient clipping threshold ({config.training.max_grad_norm}) is high")

    # Check batch size with gradient accumulation
    effective_bs = config.training.batch_size * config.training.gradient_accumulation_steps
    if effective_bs > 128:
        issues.append(f"Effective batch size ({effective_bs}) is very large")

    if issues:
        print("  Potential issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✓ No obvious configuration issues found")

    # 6. Recommendations
    print("\n[6] Recommendations:")
    print("  Based on the diagnostics, try these fixes:")
    print("    1. Reduce learning rate to 5e-5 or 1e-4")
    print("    2. Enable gradient checkpointing to reduce memory pressure")
    print("    3. Reduce max_grad_norm to 0.5 or 1.0")
    print("    4. If using bf16, try fp16 or fp32 initially")
    print("    5. Check if DeepSeek loss is properly configured")
    print("    6. Verify tokenizer is producing valid token IDs")

    print("\n" + "="*80)
    print("Diagnostic complete!")
    print("="*80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose NaN/Inf loss issues")
    parser.add_argument("--config", type=str, default="/project/code/configs/moe/tiny_moe_ultra_low_mem.yaml",
                       help="Path to training config")

    args = parser.parse_args()

    diagnose_training(args.config)
