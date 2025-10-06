#!/usr/bin/env python3
"""
Configuration Validation Script for Optimized Training Pipeline

This script validates that all optimizations are correctly configured
and provides a summary of expected improvements.
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


class ConfigValidator:
    """Validates training configuration against optimization checklist."""

    def __init__(self, config_path: str):
        """Initialize validator with config file."""
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        self.issues = []
        self.warnings = []
        self.successes = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating Optimized Configuration")
        print("=" * 60)
        print(f"Config: {self.config_path.name}\n")

        # Run all checks
        self.check_dataloader_settings()
        self.check_gradient_settings()
        self.check_optimizer_settings()
        self.check_progressive_training()
        self.check_evaluation_alignment()
        self.check_moe_settings()
        self.check_data_pipeline()
        self.check_memory_settings()

        # Print results
        self.print_results()

        return len(self.issues) == 0

    def check_dataloader_settings(self):
        """Check dataloader configuration."""
        training = self.config.get('training', {})
        data_loading = self.config.get('data_loading', {})

        # Check workers consistency
        train_workers = training.get('dataloader_num_workers', 0)
        data_workers = data_loading.get('num_workers', 0)

        if train_workers == data_workers == 4:
            self.successes.append("✓ Dataloader workers unified at 4 (optimal)")
        elif train_workers != data_workers:
            self.issues.append(f"✗ Worker mismatch: training={train_workers}, data_loading={data_workers}")

        # Check prefetch_factor
        if training.get('prefetch_factor'):
            self.successes.append(f"✓ Prefetch enabled: {training['prefetch_factor']} batches")
        else:
            self.warnings.append("⚠ Prefetch factor not set (recommend 2-4)")

        # Check buffer size
        buffer_size = data_loading.get('buffer_size', 0)
        if buffer_size >= 10000:
            self.successes.append(f"✓ Large buffer size: {buffer_size}")
        else:
            self.warnings.append(f"⚠ Small buffer size: {buffer_size} (recommend 10000)")

    def check_gradient_settings(self):
        """Check gradient and optimizer settings."""
        training = self.config.get('training', {})

        # Check gradient clipping
        grad_norm = training.get('max_gradient_norm', 0)
        if grad_norm >= 10.0:
            self.successes.append(f"✓ Gradient clipping optimized: {grad_norm} (MoE stable)")
        else:
            self.warnings.append(f"⚠ Low gradient clipping: {grad_norm} (recommend 10.0 for MoE)")

        # Check gradient checkpointing
        grad_checkpoint = training.get('gradient_checkpointing', True)
        memory_checkpoint = self.config.get('memory', {}).get('gradient_checkpointing', True)

        if grad_checkpoint == memory_checkpoint == False:
            self.successes.append("✓ Gradient checkpointing disabled for speed (unified)")
        elif grad_checkpoint != memory_checkpoint:
            self.issues.append(f"✗ Gradient checkpointing mismatch: training={grad_checkpoint}, memory={memory_checkpoint}")

    def check_optimizer_settings(self):
        """Check optimizer hyperparameters."""
        training = self.config.get('training', {})

        # Weight initialization (CRITICAL for gradient flow)
        model = self.config.get('model', {})
        init_range = model.get('initializer_range', 0.005)
        if init_range >= 0.02:
            self.successes.append(f"✓ Initialization range optimal: {init_range} (prevents gradient vanishing)")
        elif init_range >= 0.01:
            self.warnings.append(f"⚠ Initialization range: {init_range} (recommend 0.02+ for MoE)")
        else:
            self.issues.append(f"✗ Initialization range TOO SMALL: {init_range} (WILL cause gradient vanishing!)")

        # Weight decay
        weight_decay = training.get('weight_decay', 0.1)
        if weight_decay <= 0.02:
            self.successes.append(f"✓ Weight decay low: {weight_decay} (allows gradient flow)")
        elif 0.02 < weight_decay <= 0.08:
            self.warnings.append(f"⚠ Weight decay: {weight_decay} (may dampen gradients)")
        else:
            self.issues.append(f"✗ Weight decay TOO HIGH: {weight_decay} (causes gradient vanishing)")

        # Learning rate
        lr = training.get('learning_rate', 0)
        if lr >= 0.0005:
            self.successes.append(f"✓ Learning rate high: {lr} (strong gradient signals)")
        elif lr >= 0.0003:
            self.warnings.append(f"⚠ Learning rate: {lr} (consider 0.0005-0.0006)")

        # Dropout (prevents saturation)
        attn_dropout = model.get('attention_dropout', 0.0)
        hidden_dropout = model.get('hidden_dropout', 0.0)
        if attn_dropout > 0 or hidden_dropout > 0:
            self.successes.append(f"✓ Dropout enabled: attn={attn_dropout}, hidden={hidden_dropout}")
        else:
            self.warnings.append("⚠ No dropout (risk of activation saturation)")

    def check_progressive_training(self):
        """Check progressive training features."""
        progressive = self.config.get('training', {}).get('progressive', {})
        dynamic_batch = self.config.get('training', {}).get('dynamic_batching', {})

        # Curriculum learning
        if progressive.get('enable_curriculum', False):
            metric = progressive.get('curriculum_metric', 'unknown')
            self.successes.append(f"✓ Curriculum learning enabled (metric: {metric})")
        else:
            self.warnings.append("⚠ Curriculum learning disabled (30% speedup possible)")

        # Dynamic batching - REMOVED FOR STABILITY
        if dynamic_batch.get('enabled', False):
            self.warnings.append("⚠ Dynamic batching enabled (not recommended - can cause instability)")
        else:
            self.successes.append("✓ Dynamic batching disabled (stable, predictable training)")

        # Sequence growth
        growth_epochs = progressive.get('length_growth_epochs', 0)
        if growth_epochs >= 5:
            self.successes.append(f"✓ Sequence growth extended: {growth_epochs} epochs")
        elif growth_epochs > 0:
            self.warnings.append(f"⚠ Sequence growth: {growth_epochs} epochs (recommend 5+)")

    def check_evaluation_alignment(self):
        """Check evaluation and checkpointing alignment."""
        training = self.config.get('training', {})

        eval_steps = training.get('eval_steps', 0)
        save_steps = training.get('save_steps', 0)

        if save_steps == 2 * eval_steps:
            self.successes.append(f"✓ Aligned intervals: eval={eval_steps}, save={save_steps}")
        else:
            self.warnings.append(f"⚠ Misaligned intervals: eval={eval_steps}, save={save_steps}")

        # Early stopping patience
        patience = training.get('early_stopping_patience', 0)
        if patience >= 6:
            self.successes.append(f"✓ Early stopping patience: {patience} (MoE stable)")
        else:
            self.warnings.append(f"⚠ Low patience: {patience} (recommend 6+ for MoE)")

    def check_moe_settings(self):
        """Check MoE-specific settings."""
        losses = self.config.get('enhanced_features', {}).get('losses', {})
        evaluation = self.config.get('evaluation', {})

        # Multi-token prediction weight
        mtp_weight = losses.get('mtp_weight', 0)
        if mtp_weight >= 0.10:
            self.successes.append(f"✓ MTP weight optimized: {mtp_weight}")
        else:
            self.warnings.append(f"⚠ Low MTP weight: {mtp_weight} (recommend 0.10-0.15)")

        # Label smoothing
        label_smooth = losses.get('label_smoothing', 0.12)
        if label_smooth <= 0.10:
            self.successes.append(f"✓ Label smoothing reduced: {label_smooth} (sharper routing)")
        else:
            self.warnings.append(f"⚠ High label smoothing: {label_smooth} (may blur MoE routing)")

        # Gradient balance weight
        grad_balance = losses.get('gradient_balance_weight', 0)
        if grad_balance >= 0.08:
            self.successes.append(f"✓ Gradient balance weight: {grad_balance}")
        else:
            self.warnings.append(f"⚠ Low gradient balance: {grad_balance} (recommend 0.08+)")

        # MoE metrics
        moe_metrics = evaluation.get('moe_metrics', {})
        if moe_metrics.get('track_expert_utilization'):
            tracked = sum([
                moe_metrics.get('track_expert_utilization', False),
                moe_metrics.get('track_routing_entropy', False),
                moe_metrics.get('track_load_balance', False)
            ])
            self.successes.append(f"✓ MoE metrics enabled ({tracked}/3 core metrics)")
        else:
            self.warnings.append("⚠ MoE metrics not configured")

    def check_data_pipeline(self):
        """Check data pipeline optimizations."""
        data_loading = self.config.get('data_loading', {})

        # Bucketing
        if data_loading.get('enable_bucketing'):
            boundaries = data_loading.get('bucket_boundaries', [])
            self.successes.append(f"✓ Length bucketing enabled ({len(boundaries)} buckets)")
        else:
            self.warnings.append("⚠ Length bucketing disabled (GPU util improvement possible)")

    def check_memory_settings(self):
        """Check memory management settings."""
        memory = self.config.get('memory', {})

        # Cache clearing
        cache_freq = memory.get('clear_cache_frequency', 100)
        if cache_freq >= 500:
            self.successes.append(f"✓ Cache clearing optimized: every {cache_freq} steps")
        elif cache_freq >= 200:
            self.warnings.append(f"⚠ Frequent cache clearing: {cache_freq} (recommend 500+)")

    def print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60 + "\n")

        if self.successes:
            print(f"✅ SUCCESSES ({len(self.successes)}):")
            for success in self.successes:
                print(f"  {success}")
            print()

        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
            print()

        if self.issues:
            print(f"❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
            print()

        # Summary
        print("=" * 60)
        total_checks = len(self.successes) + len(self.warnings) + len(self.issues)
        pass_rate = (len(self.successes) / total_checks * 100) if total_checks > 0 else 0

        print(f"SUMMARY: {len(self.successes)}/{total_checks} checks passed ({pass_rate:.1f}%)")

        if len(self.issues) == 0 and len(self.warnings) == 0:
            print("✅ Configuration is fully optimized!")
        elif len(self.issues) == 0:
            print("✅ Configuration is valid (with minor warnings)")
        else:
            print("❌ Configuration has issues that need fixing")

        print("=" * 60)


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate optimized training configuration")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gpu/small_optimized.yaml',
        help='Path to config file to validate'
    )

    args = parser.parse_args()

    # Run validation
    validator = ConfigValidator(args.config)
    is_valid = validator.validate_all()

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
