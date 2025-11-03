#!/usr/bin/env python3
"""
Enhanced Learning Rate Finder Script

This script runs the enhanced LR finder with MULTIPLE METHODS (fastai, valley, steepest)
and generates a comprehensive report. By default, it AUTOMATICALLY UPDATES your config
file with the optimal learning rate and lr_end values.

DEFAULT BEHAVIOR:
    - Runs 3 methods: fastai (conservative), valley (balanced), steepest (aggressive)
    - Uses geometric mean for final recommendation
    - Auto-updates your config with optimal LR and lr_end
    - Creates timestamped backup
    - NOW USES 1000 ITERATIONS for maximum accuracy!

Usage:
    # Auto-update config with 3 methods (DEFAULT - recommended)
    python run_lr_finder_enhanced.py --config ../configs/gpu/small.yaml

    # Use all 5 methods (slower but more comprehensive)
    python run_lr_finder_enhanced.py --config ../configs/gpu/small.yaml --methods fastai valley steepest minimum combined

    # Use single method (NOT recommended - no variance check)
    python run_lr_finder_enhanced.py --config ../configs/gpu/small.yaml --methods fastai

    # Just analyze, don't update config
    python run_lr_finder_enhanced.py --config ../configs/gpu/small.yaml --no-auto-update

    # Custom output directory
    python run_lr_finder_enhanced.py --config ../configs/gpu/small.yaml --output lr_results/

    # Skip backup creation
    python run_lr_finder_enhanced.py --config ../configs/gpu/small.yaml --no-backup

What gets auto-updated:
    - training.learning_rate: Set to optimal LR from multi-method analysis
    - training.lr_end: Set to 1/300 of peak LR (ensures proper cosine decay)
    - wandb.notes: Adds timestamp and optimization info
    - Creates timestamped backup: config_backup_YYYYMMDD_HHMMSS.yaml
    - Validates lr_end < learning_rate (prevents inverted schedule bug)

Why multiple methods?
    - Cross-validation between different algorithms
    - Variance check for confidence assessment
    - Geometric mean balances conservative vs aggressive
    - High variance (>5x) = low confidence, use conservative LR
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
import json
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.Ava.optimization import LRFinder, LRFinderConfig
from src.Ava.data.dataloader import create_streaming_dataloaders
from src.Ava.models.moe_model import EnhancedMoEModel
from src.Ava.config.training_config import ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_enhanced_lr_finder(
    config_path: str,
    output_dir: str = "lr_results",
    num_runs: int = 1,
    methods: list | None = None
) -> Dict[str, Any]:
    """
    Run enhanced LR finder with multiple methods and averaging.

    Args:
        config_path: Path to training config file
        output_dir: Directory to save results
        num_runs: Number of runs to average (default: 1)
        methods: List of suggestion methods to try (default: all)

    Returns:
        Dictionary with results from all methods
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config using yaml
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate existing config for common bugs
    current_lr = config_dict.get('training', {}).get('learning_rate')
    current_lr_end = config_dict.get('training', {}).get('lr_end')
    if current_lr and current_lr_end:
        if current_lr_end >= current_lr:
            logger.warning("")
            logger.warning("="*80)
            logger.warning("⚠️  WARNING: INVERTED LR SCHEDULE DETECTED IN CONFIG")
            logger.warning("="*80)
            logger.warning(f"Current config has lr_end ({current_lr_end:.2e}) >= learning_rate ({current_lr:.2e})")
            logger.warning(f"This breaks cosine scheduler! lr_end must be LOWER than learning_rate.")
            logger.warning(f"")
            logger.warning(f"This script will fix it automatically after finding optimal LR.")
            logger.warning("="*80)
            logger.warning("")

    # Extract relevant config sections
    model_config_dict = config_dict.get('model', {})
    training_config_dict = config_dict.get('training', {})
    data_config_dict = config_dict.get('data', {})
    lr_finder_config_dict = config_dict.get('lr_finder', {})

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Clear GPU cache and check memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # Check available GPU memory
        gpu_free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        gpu_total_mb = torch.cuda.mem_get_info()[1] / 1024**2
        gpu_used_mb = gpu_total_mb - gpu_free_mb

        logger.info(f"Cleared GPU cache")
        logger.info(f"💾 GPU Memory: {gpu_used_mb:.0f} MB used, {gpu_free_mb:.0f} MB free, {gpu_total_mb:.0f} MB total")

        # Warn if GPU memory is highly occupied
        if gpu_free_mb < 4000:  # Less than 4GB free
            logger.warning("")
            logger.warning("="*80)
            logger.warning("⚠️  WARNING: INSUFFICIENT GPU MEMORY")
            logger.warning("="*80)
            logger.warning(f"Only {gpu_free_mb:.0f} MB GPU memory available!")
            logger.warning(f"LR finder needs ~4-8 GB free memory to run safely")
            logger.warning(f"Current GPU usage: {gpu_used_mb:.0f} MB ({gpu_used_mb/gpu_total_mb*100:.1f}%)")
            logger.warning("")
            logger.warning("🛑 ACTION REQUIRED:")
            logger.warning("   If training is currently running, STOP it first:")
            logger.warning("")
            logger.warning("   Option 1: Press Ctrl+C in the training terminal")
            logger.warning("   Option 2: Run this command:")
            logger.warning("             pkill -f 'train.py'")
            logger.warning("")
            logger.warning("   Then:")
            logger.warning("   1. Wait 10 seconds for GPU memory to clear")
            logger.warning("   2. Verify with: nvidia-smi")
            logger.warning("   3. Re-run this script")
            logger.warning("")
            logger.warning("="*80)
            logger.warning("")

            # Ask user if they want to continue
            try:
                response = input("⚠️  Continue anyway? This will likely fail with OOM. [y/N]: ")
                if response.lower() != 'y':
                    logger.info("❌ Aborting. Please free up GPU memory and try again.")
                    logger.info("")
                    logger.info("Quick fix:")
                    logger.info("  pkill -f 'train.py' && sleep 10 && nvidia-smi")
                    sys.exit(1)
                else:
                    logger.warning("⚠️  Proceeding with low GPU memory. Expect potential OOM errors...")
            except (KeyboardInterrupt, EOFError):
                logger.info("\n❌ Interrupted by user. Exiting.")
                sys.exit(1)

    # Create model
    logger.info("Creating model...")
    # EnhancedMoEModel expects a config with all fields
    from types import SimpleNamespace

    # Provide defaults for any missing fields
    config_defaults = {
        'use_moh': False,
        'use_moa': False,
        'use_cross_attention': False,
        'use_alibi': False,
        'dropout': model_config_dict.get('hidden_dropout', 0.1),
        'deepspeed_activation_checkpointing': False,
        'deepspeed_partition_activations': False,
        'deepspeed_moe_param_groups': False,
        'tie_word_embeddings': False,
    }

    # Merge defaults with actual config
    full_config = {**config_defaults, **model_config_dict}

    # Convert numeric string fields to proper types
    numeric_fields = ['layer_norm_eps', 'initializer_range', 'router_aux_loss_coef',
                      'router_jitter_noise', 'expert_capacity_factor', 'attention_dropout',
                      'hidden_dropout', 'dropout', 'rope_theta']
    for field in numeric_fields:
        if field in full_config and isinstance(full_config[field], str):
            full_config[field] = float(full_config[field])

    # Import the proper config class - ModelConfig for model-specific parameters
    from src.Ava.config.training_config import ModelConfig
    model_config = ModelConfig(**full_config)
    model = EnhancedMoEModel(model_config).to(device)  # type: ignore[arg-type]
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Create optimizer
    logger.info("Creating optimizer...")
    learning_rate = float(training_config_dict.get('learning_rate', 3e-4))
    weight_decay = float(training_config_dict.get('weight_decay', 0.1))
    beta1 = float(training_config_dict.get('beta1', 0.9))
    beta2 = float(training_config_dict.get('beta2', 0.999))
    eps = float(training_config_dict.get('eps', 1e-8))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )

    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer_name = data_config_dict.get('tokenizer_name', 'Qwen/Qwen2.5-0.5B')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Create data loaders
    logger.info("Creating data loaders...")
    data_dir = data_config_dict.get('data_dir', '/project/code/data/processed')
    # Make path absolute if it's relative
    if not Path(data_dir).is_absolute():
        data_dir = str(Path(config_path).parent.parent.parent / data_dir)

    max_length = data_config_dict.get('max_length', 512)
    # Use smaller batch size for LR finder to avoid OOM
    batch_size = min(training_config_dict.get('batch_size', 16), 4)
    logger.info(f"Using batch_size={batch_size} for LR finder (reduced to avoid OOM)")
    val_split_ratio = data_config_dict.get('val_split_ratio', 0.15)
    buffer_size = data_config_dict.get('buffer_size', 50000)

    train_loader, val_loader = create_streaming_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        data_dir=data_dir,
        val_split_ratio=val_split_ratio,
        buffer_size=buffer_size
    )

    # Define methods to test (default to 3 key methods for speed + robustness)
    if methods is None or len(methods) == 0:
        methods = ['fastai', 'valley', 'steepest']

    logger.info(f"📋 Running {len(methods)} methods: {', '.join(methods)}")

    # Store results for each method
    all_results = {}

    for method in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running LR Finder with {method} method...")
        logger.info(f"{'='*80}\n")

        # IMPORTANT: Reset model and optimizer for each method
        # Otherwise model state carries over from previous run
        logger.info(f"🔄 Resetting model and optimizer for {method}...")

        # Recreate model from scratch using the ModelConfig created earlier
        model = EnhancedMoEModel(model_config).to(device)  # type: ignore[arg-type]

        # Recreate optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-7,
            betas=(float(training_config_dict.get('beta1', 0.9)),
                   float(training_config_dict.get('beta2', 0.999))),
            eps=float(training_config_dict.get('eps', 1e-8)),
            weight_decay=float(training_config_dict.get('weight_decay', 0.1))
        )

        # Configure LR finder with IMPROVED DEFAULTS (1000 iterations!)
        lr_config = LRFinderConfig(
            start_lr=float(lr_finder_config_dict.get('start_lr', 1e-8)),
            end_lr=float(lr_finder_config_dict.get('end_lr', 1e-2)),
            num_iter=int(lr_finder_config_dict.get('num_iterations', 1000)),  # Default to 1000!
            beta=float(lr_finder_config_dict.get('beta', 0.95)),               # Stronger smoothing
            suggestion_method=method,
            use_savgol_filter=lr_finder_config_dict.get('use_savgol_filter', True),
            savgol_window=int(lr_finder_config_dict.get('savgol_window', 31)),  # Larger window
            savgol_polyorder=3,
            momentum_cycling=lr_finder_config_dict.get('momentum_cycling', False),
            track_validation=lr_finder_config_dict.get('track_validation', False),
            num_runs=num_runs,
            save_plot=True,
            plot_path=str(output_path / f"lr_finder_{method}.png")
        )

        # Create criterion
        criterion = torch.nn.CrossEntropyLoss()

        # Run LR finder
        finder = LRFinder(model, optimizer, criterion, device, lr_config)

        try:
            # FIXED: Use lr_finder config's gradient_accumulation_steps, NOT training config
            # LR finder should always use 1 (no accumulation) for accurate loss tracking
            lr_finder_accum_steps = lr_finder_config_dict.get('gradient_accumulation_steps', 1)
            results = finder.range_test(
                train_loader=train_loader,
                val_loader=val_loader if lr_config.track_validation else None,
                accumulation_steps=lr_finder_accum_steps
            )

            all_results[method] = results

            logger.info(f"\n{method.upper()} Results:")
            logger.info(f"  Suggested LR: {results['suggested_lr']:.6e}")
            logger.info(f"  Best Loss: {results['best_loss']:.6f}")
            logger.info(f"  Best LR: {results['best_lr']:.6e}")

        except Exception as e:
            logger.error(f"Error running {method} method: {e}")
            all_results[method] = {'error': str(e)}

        # Clear GPU memory between methods
        if torch.cuda.is_available():
            del model, optimizer, finder
            torch.cuda.empty_cache()
            logger.info(f"🧹 Cleared GPU cache after {method}")

    # Generate comprehensive report
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE LR FINDER REPORT")
    logger.info(f"{'='*80}\n")

    # Find consensus and variance
    suggested_lrs = {k: v.get('suggested_lr') for k, v in all_results.items() if 'suggested_lr' in v}

    if suggested_lrs:
        avg_lr = sum(suggested_lrs.values()) / len(suggested_lrs)
        min_lr = min(suggested_lrs.values())
        max_lr = max(suggested_lrs.values())
        variance = max_lr / min_lr if min_lr > 0 else float('inf')

        logger.info("📊 Summary of All Methods (Raw Results):")
        logger.info("-" * 80)
        for method, lr in suggested_lrs.items():
            diff_from_avg = ((lr - avg_lr) / avg_lr) * 100
            logger.info(f"  {method:12s}: {lr:.6e} ({diff_from_avg:+.1f}% from average)")

        logger.info(f"\n📈 Raw Statistics (Before Outlier Removal):")
        logger.info(f"  Average LR:     {avg_lr:.6e}")
        logger.info(f"  Min LR:         {min_lr:.6e}")
        logger.info(f"  Max LR:         {max_lr:.6e}")
        logger.info(f"  Variance Ratio: {variance:.2f}x")

        # AUTOMATIC SMART RECOMMENDATION WITH OUTLIER DETECTION
        import math

        logger.info(f"\n💡 Automatic Recommendation (with outlier detection):")

        # IMPROVED AUTOMATIC STRATEGY:
        # 1. Detect outliers (>5x different from median)
        # 2. Use geometric mean of non-outlier methods
        # 3. Provide clear explanation of what was excluded and why

        # Initialize variables for outlier detection (needed for all paths)
        median_lr = None
        iqr = None

        if len(suggested_lrs) >= 2:
            # Calculate median
            sorted_lrs = sorted(suggested_lrs.values())
            median_lr = sorted_lrs[len(sorted_lrs) // 2]

            # Calculate IQR (Interquartile Range) for outlier detection
            q1_idx = len(sorted_lrs) // 4
            q3_idx = 3 * len(sorted_lrs) // 4
            q1 = sorted_lrs[q1_idx] if q1_idx < len(sorted_lrs) else sorted_lrs[0]
            q3 = sorted_lrs[q3_idx] if q3_idx < len(sorted_lrs) else sorted_lrs[-1]
            iqr = q3 - q1 if q3 > q1 else 1.0  # Avoid division by zero

            # Detect outliers (methods >5x away from median)
            non_outliers = {}
            outliers = {}

            for method, lr in suggested_lrs.items():
                ratio_to_median = max(lr, median_lr) / min(lr, median_lr)
                if ratio_to_median > 5.0:
                    outliers[method] = lr
                    logger.info(f"  🚫 EXCLUDED {method}: {lr:.6e} ({ratio_to_median:.1f}x from median - outlier)")
                else:
                    non_outliers[method] = lr

            # If we have at least 2 non-outliers, use them
            if len(non_outliers) >= 2:
                # Calculate geometric mean of non-outliers
                non_outlier_values = list(non_outliers.values())
                recommended_lr = math.exp(sum(math.log(lr) for lr in non_outlier_values) / len(non_outlier_values))

                # Calculate variance of non-outliers
                non_outlier_min = min(non_outlier_values)
                non_outlier_max = max(non_outlier_values)
                non_outlier_variance = non_outlier_max / non_outlier_min if non_outlier_min > 0 else float('inf')

                strategy = f"geometric_mean({', '.join(non_outliers.keys())})"

                logger.info(f"\n  ✅ USED (non-outliers):")
                for method, lr in non_outliers.items():
                    logger.info(f"     • {method}: {lr:.6e}")
                logger.info(f"  📊 Non-outlier variance: {non_outlier_variance:.2f}x")

            elif len(non_outliers) == 1:
                # Only one non-outlier, use it
                method, lr = list(non_outliers.items())[0]
                recommended_lr = lr
                strategy = f"{method} only (others were outliers)"
                logger.info(f"  ✅ Using {method}: {lr:.6e} (only non-outlier)")

            else:
                # All methods are outliers from each other - use median
                recommended_lr = median_lr
                strategy = "median (all methods are outliers)"
                logger.info(f"  ⚠️  All methods are outliers - using median")

        elif len(suggested_lrs) == 1:
            # Only one method available
            method, lr = list(suggested_lrs.items())[0]
            recommended_lr = lr
            strategy = f"{method} only (single method)"

        else:
            # Fallback
            recommended_lr = min_lr * 3
            strategy = "3x minimum (fallback)"

        # Calculate final variance for confidence assessment
        import numpy as np
        # Use median_lr and iqr if they were calculated (when len >= 2), otherwise use all values
        if median_lr is not None and iqr is not None:
            non_outlier_values_list = [v for v in suggested_lrs.values() if abs(v - median_lr) <= 1.5 * iqr]
        else:
            non_outlier_values_list = list(suggested_lrs.values())
        non_outlier_variance_calculated = np.var(non_outlier_values_list) if non_outlier_values_list else variance
        if len(non_outlier_values_list) >= 2:
            final_variance = non_outlier_variance_calculated
        else:
            final_variance = variance

        # Assess confidence and provide guidance based on final variance (after outlier removal)
        if final_variance < 2.0:
            confidence = "VERY HIGH"
            icon = "✅✅✅"
            guidance = "Excellent agreement! Methods converged perfectly."
            guidance += f"\n      • Start with recommended LR: {recommended_lr:.2e}"
            guidance += f"\n      • Safe to use - very reliable result"
        elif final_variance < 3.0:
            confidence = "HIGH"
            icon = "✅✅"
            guidance = "Strong agreement between methods. Recommended LR is reliable."
            guidance += f"\n      • Start with recommended LR: {recommended_lr:.2e}"
            guidance += f"\n      • Monitor first 1000 steps for stability"
        elif final_variance < 5.0:
            confidence = "GOOD"
            icon = "✅"
            guidance = f"Good agreement (variance={final_variance:.1f}x after outlier removal)."
            guidance += f"\n      • Start with recommended LR: {recommended_lr:.2e}"
            guidance += f"\n      • Acceptable result - safe to use"
        elif final_variance < 10.0:
            confidence = "MEDIUM"
            icon = "⚠️"
            guidance = f"Moderate agreement (variance={final_variance:.1f}x). Some noise remains."
            guidance += f"\n      • Start with recommended LR: {recommended_lr:.2e}"
            guidance += f"\n      • Watch loss curve closely for first 1000 steps"
            guidance += f"\n      • If unstable, reduce to: {recommended_lr * 0.7:.2e}"
        else:
            # High variance even after outlier removal
            confidence = "LOW"
            icon = "⚠️⚠️"
            guidance = f"High variance ({final_variance:.1f}x) even after outlier removal."
            guidance += f"\n      • Start conservatively with: {recommended_lr:.2e}"
            guidance += f"\n      • Consider running with more smoothing or ensemble averaging"
            guidance += f"\n      • If unstable, reduce to: {recommended_lr * 0.5:.2e}"

        logger.info(f"  {icon} Strategy: {strategy}")
        logger.info(f"  {icon} Recommended LR: {recommended_lr:.6e}")
        logger.info(f"  {icon} Confidence: {confidence}")
        logger.info(f"  {icon} Guidance: {guidance}")

        # Save results to JSON
        results_file = output_path / "lr_finder_results.json"
        with open(results_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {
                'methods': suggested_lrs,
                'statistics': {
                    'average': float(avg_lr),
                    'min': float(min_lr),
                    'max': float(max_lr),
                    'variance_ratio': float(variance)
                },
                'recommendation': {
                    'lr': float(recommended_lr),
                    'strategy': strategy,
                    'confidence': confidence,
                    'alternative_higher': float(recommended_lr * 3),
                    'alternative_lower': float(recommended_lr * 0.3)
                }
            }
            json.dump(json_results, f, indent=2)

        logger.info(f"\n✅ Results saved to: {results_file}")
        logger.info(f"✅ Plots saved to: {output_path}/")

        return json_results

    else:
        logger.error("❌ No successful results from any method")
        return {}


def update_config_with_lr(config_path: str, suggested_lr: float, backup: bool = True):
    """
    Update config file with suggested learning rate and lr_end.

    Args:
        config_path: Path to config file
        suggested_lr: Suggested learning rate
        backup: Whether to create backup of original config
    """
    from datetime import datetime
    import shutil

    config_file = Path(config_path)
    backup_file = None

    # Create timestamped backup if requested
    if backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = config_file.parent / f"{config_file.stem}_backup_{timestamp}.yaml"
        logger.info(f"💾 Creating backup: {backup_file}")
        shutil.copy2(config_file, backup_file)

    # Load and update config
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Update learning_rate
    old_lr = config_data.get('training', {}).get('learning_rate', 'unknown')
    config_data['training']['learning_rate'] = float(suggested_lr)

    # Calculate and update lr_end (must be LOWER than start LR for cosine decay)
    # Use 1/300 of peak LR (safe minimum for cosine schedule)
    lr_end = max(suggested_lr / 300.0, 1e-7)
    old_lr_end = config_data.get('training', {}).get('lr_end', 'unknown')
    config_data['training']['lr_end'] = float(lr_end)

    # Add optimization note to wandb if it exists
    if 'wandb' in config_data:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        optimization_note = f"\n\n[Auto-optimized {timestamp}] LR: {suggested_lr:.2e}, lr_end: {lr_end:.2e}"

        old_notes = config_data['wandb'].get('notes', '')
        if isinstance(old_notes, str):
            config_data['wandb']['notes'] = old_notes + optimization_note
        else:
            config_data['wandb']['notes'] = optimization_note

    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, width=120)

    # Validate that lr_end < learning_rate
    if lr_end >= suggested_lr:
        logger.error(f"❌ VALIDATION ERROR: lr_end ({lr_end:.2e}) >= learning_rate ({suggested_lr:.2e})")
        logger.error(f"   This will break cosine scheduler! lr_end must be LOWER than learning_rate.")
        raise ValueError(f"Invalid LR schedule: lr_end must be < learning_rate")

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ CONFIG UPDATED")
    logger.info(f"{'='*80}")
    logger.info(f"📝 Changes made to {config_file}:")
    logger.info(f"   learning_rate: {old_lr} → {suggested_lr:.2e}")
    logger.info(f"   lr_end: {old_lr_end} → {lr_end:.2e} (1/300 of peak, ensures proper decay)")
    logger.info(f"   ✓ Validation: lr_end < learning_rate ({lr_end:.2e} < {suggested_lr:.2e})")
    if backup and backup_file:
        logger.info(f"   backup: {backup_file}")
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Learning Rate Finder (1000 iterations for maximum accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='lr_results',
        help='Output directory for results (default: lr_results)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of runs to average (default: 1)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        choices=['fastai', 'steepest', 'minimum', 'valley', 'combined'],
        default=['fastai', 'valley', 'steepest'],
        help='Suggestion methods to try (default: fastai, valley, steepest)'
    )
    parser.add_argument(
        '--auto-update-config',
        action='store_true',
        default=True,
        help='Automatically update config file with suggested LR (default: True)'
    )
    parser.add_argument(
        '--no-auto-update',
        action='store_true',
        help='Skip auto-updating config file (only generate report)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup when updating config'
    )

    args = parser.parse_args()

    # Run LR finder
    try:
        results = run_enhanced_lr_finder(
            config_path=args.config,
            output_dir=args.output,
            num_runs=args.num_runs,
            methods=args.methods
        )

        # Auto-update config by default (unless --no-auto-update specified)
        if not args.no_auto_update and results and 'recommendation' in results:
            suggested_lr = results['recommendation']['lr']
            logger.info("\n🔧 Auto-updating config with optimal learning rate...")
            update_config_with_lr(
                config_path=args.config,
                suggested_lr=suggested_lr,
                backup=not args.no_backup
            )
        elif args.no_auto_update:
            logger.info("\n⏭️  Skipping config update (--no-auto-update flag set)")
            if results and 'recommendation' in results and 'lr' in results['recommendation']:
                logger.info(f"   To manually update, set learning_rate to: {results['recommendation']['lr']:.2e}")
            else:
                logger.info("   No valid recommendation available")
        elif not results or 'recommendation' not in results:
            logger.warning("\n⚠️  No valid results to update config with")

        logger.info("\n✅ Enhanced LR Finder completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error running LR finder: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
