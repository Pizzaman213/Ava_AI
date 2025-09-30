#!/usr/bin/env python3
"""
Fix gradient stability issues in all GPU configuration files.
"""

import yaml
from pathlib import Path

def fix_config(config):
    """Apply stability fixes to a configuration."""

    # Fix model parameters
    if 'model' in config:
        # Reduce initializer range for stability
        config['model']['initializer_range'] = 0.01  # Reduced from 0.02

        # Add dropout for regularization
        config['model']['attention_dropout'] = 0.1
        config['model']['hidden_dropout'] = 0.1

        # Disable router jitter during initial training
        if 'router_jitter_noise' in config['model']:
            config['model']['router_jitter_noise'] = 0.0

        # Increase router aux loss for better routing
        if 'router_aux_loss_coef' in config['model']:
            config['model']['router_aux_loss_coef'] = 0.01

    # Fix training parameters
    if 'training' in config:
        # Reduce learning rate significantly
        if 'learning_rate' in config['training']:
            old_lr = config['training']['learning_rate']
            # Convert to float if string
            if isinstance(old_lr, str):
                old_lr = float(old_lr)
            # Scale down by 4x for stability
            new_lr = old_lr / 4.0
            config['training']['learning_rate'] = min(new_lr, 1e-4)  # Cap at 1e-4

        # Reduce weight decay
        if 'weight_decay' in config['training']:
            config['training']['weight_decay'] = min(config['training']['weight_decay'], 0.01)

        # More aggressive gradient clipping
        config['training']['max_gradient_norm'] = 0.5  # Reduced from 1.0

        # Adjust optimizer betas for stability
        if 'beta2' in config['training']:
            config['training']['beta2'] = 0.999  # More stable than 0.95

        # Increase warmup for gentler start
        if 'warmup_steps' in config['training']:
            config['training']['warmup_steps'] = max(config['training']['warmup_steps'] * 2, 5000)

        # Adjust learning rate end
        if 'lr_end' in config['training']:
            # Set to 10% of new learning rate
            config['training']['lr_end'] = config['training']['learning_rate'] * 0.1

        # Enable gradient checkpointing for stability
        config['training']['gradient_checkpointing'] = True

        # Use fp16 instead of bf16 for better stability
        if config['training'].get('mixed_precision') == 'bf16':
            config['training']['mixed_precision'] = 'fp16'

        # Add gradient health monitoring
        config['training']['gradient_health'] = {
            'enabled': True,
            'initial_clip_value': 2.0,
            'final_clip_value': 0.5,
            'warmup_steps': config['training'].get('warmup_steps', 5000),
            'explosion_threshold': 5.0,
            'explosion_window': 5,
            'auto_reduce_lr': True,
            'lr_reduction_factor': 0.5
        }

    # Fix hardware settings
    if 'fp16' in config or 'bf16' in config:
        config['fp16'] = True
        config['bf16'] = False

    # Add stability features
    config['stability'] = {
        'loss_smoothing': True,
        'loss_smoothing_alpha': 0.9,
        'lr_warmup_ratio': 0.1,
        'lr_decay_ratio': 0.1,
        'sync_gradients_each_step': False,
        'use_kahan_summation': True,
        'use_stable_embedding': True,
        'stop_on_nan': True,
        'max_grad_norm_before_stop': 100.0,
        'max_loss_before_stop': 20.0
    }

    # Fix enhanced features for stability
    if 'enhanced_features' in config:
        if 'gradient' in config['enhanced_features']:
            config['enhanced_features']['gradient']['gradient_surgery'] = True
            config['enhanced_features']['gradient']['adaptive_gradient_surgery'] = True
            config['enhanced_features']['gradient']['gradient_norm_tracking'] = True
            config['enhanced_features']['gradient']['gradient_anomaly_detection'] = True

        if 'losses' in config['enhanced_features']:
            config['enhanced_features']['losses']['adaptive_loss_scaling'] = True

    return config

def main():
    """Fix all GPU configuration files."""
    config_dir = Path('/project/code/configs/gpu')

    for config_file in config_dir.glob('*.yaml'):
        print(f"Processing {config_file.name}...")

        # Read config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Apply fixes
        config = fix_config(config)

        # Write back
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=100)

        print(f"  ✓ Fixed {config_file.name}")

    print("\nAll configurations have been fixed for gradient stability!")
    print("\nKey changes applied:")
    print("- Learning rate reduced by 4x (capped at 1e-4)")
    print("- Gradient clipping reduced to 0.5")
    print("- Added dropout (0.1) for regularization")
    print("- Initializer range reduced to 0.01")
    print("- Warmup steps doubled (min 5000)")
    print("- Beta2 increased to 0.999 for stability")
    print("- Enabled gradient checkpointing")
    print("- Added gradient health monitoring")
    print("- Switched from bf16 to fp16 for stability")
    print("- Added stability features and emergency stops")

if __name__ == '__main__':
    main()