#!/usr/bin/env python3
"""
Config Validation Script

Validates training configs for common bugs and issues that can cause training failure.

Usage:
    python validate_config.py --config configs/gpu/small.yaml
    python validate_config.py --config configs/gpu/small.yaml --fix

Checks:
    1. LR Schedule: lr_end < learning_rate (cosine scheduler requirement)
    2. Penalty weights: Not excessive (prevents repetition collapse)
    3. Warmup steps: Reasonable (prevents instability)
    4. Batch size: Valid for GPU memory
    5. Gradient clipping: Set properly
    6. Data paths: Exist and accessible
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime
import shutil


class ConfigValidator:
    """Validates training configuration files."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.errors = []
        self.warnings = []
        self.fixes_applied = []

        # Load config
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def validate_lr_schedule(self):
        """Validate learning rate schedule configuration."""
        training = self.config.get('training', {})
        lr = training.get('learning_rate')
        lr_end = training.get('lr_end')
        scheduler_type = training.get('lr_scheduler_type', 'cosine')

        if lr is None:
            self.errors.append("❌ training.learning_rate is not set")
            return

        if lr_end is None:
            self.warnings.append("⚠️  training.lr_end is not set (will use default)")
            return

        # Critical bug: inverted schedule
        if lr_end >= lr:
            self.errors.append(
                f"❌ CRITICAL: lr_end ({lr_end:.2e}) >= learning_rate ({lr:.2e})\n"
                f"   Cosine scheduler requires lr_end < learning_rate\n"
                f"   This will cause LR to INCREASE instead of DECREASE!"
            )

        # Check for too-low LR (common after collapse)
        if lr < 1e-5:
            self.warnings.append(
                f"⚠️  learning_rate ({lr:.2e}) is very low (< 1e-5)\n"
                f"   Training may be too slow. Typical range: 1e-4 to 5e-4"
            )

        # Check for too-high lr_end (prevents proper decay)
        if lr_end and lr_end > lr * 0.1:
            self.warnings.append(
                f"⚠️  lr_end ({lr_end:.2e}) is high (> 10% of peak LR)\n"
                f"   Recommend: {lr / 300:.2e} (1/300 of peak for proper decay)"
            )

    def validate_penalties(self):
        """Validate repetition penalty configuration."""
        training = self.config.get('training', {})

        # Check for excessive EOS penalties
        eos_penalty = training.get('eos_penalty_weight', 0)
        eos_bias = training.get('eos_logit_bias', 0)

        if eos_penalty > 5.0 or eos_bias > 3.0:
            self.warnings.append(
                f"⚠️  Excessive EOS penalties detected:\n"
                f"   eos_penalty_weight: {eos_penalty} (recommend: 0-3)\n"
                f"   eos_logit_bias: {eos_bias} (recommend: 0-2)\n"
                f"   High penalties can cause repetition collapse!"
            )

        # Check for excessive repetition penalties
        rep_penalty = training.get('repetition_penalty_weight', 1.0)
        imm_rep_penalty = training.get('immediate_repetition_weight', 1.0)

        if rep_penalty > 1.3 or imm_rep_penalty > 2.5:
            self.warnings.append(
                f"⚠️  High repetition penalties detected:\n"
                f"   repetition_penalty_weight: {rep_penalty} (recommend: 1.0-1.2)\n"
                f"   immediate_repetition_weight: {imm_rep_penalty} (recommend: 0.5-2.0)\n"
                f"   Consider reducing to let model learn naturally"
            )

        # Check enhanced losses
        enhanced = self.config.get('enhanced_features', {}).get('losses', {})
        if enhanced.get('use_ngram_penalty') and enhanced.get('ngram_penalty_weight', 0) > 2.0:
            self.warnings.append(
                f"⚠️  ngram_penalty_weight ({enhanced.get('ngram_penalty_weight')}) is high\n"
                f"   Recommend: 0-1.0 for gentle guidance"
            )

    def validate_warmup(self):
        """Validate warmup configuration."""
        training = self.config.get('training', {})
        warmup_steps = training.get('warmup_steps', 0)
        max_steps = training.get('max_steps', 30000)

        if warmup_steps < 500:
            self.warnings.append(
                f"⚠️  warmup_steps ({warmup_steps}) is very short\n"
                f"   Recommend: 1000-2000 steps for stable training"
            )

        if warmup_steps > max_steps * 0.1:
            self.warnings.append(
                f"⚠️  warmup_steps ({warmup_steps}) is too long (> 10% of max_steps)\n"
                f"   This wastes training time. Recommend: {int(max_steps * 0.05)}"
            )

    def validate_batch_size(self):
        """Validate batch size configuration."""
        training = self.config.get('training', {})
        batch_size = training.get('batch_size', 16)
        grad_accum = training.get('gradient_accumulation_steps', 1)
        effective_batch = batch_size * grad_accum

        if effective_batch < 32:
            self.warnings.append(
                f"⚠️  Effective batch size ({effective_batch}) is small\n"
                f"   batch_size × grad_accum = {batch_size} × {grad_accum} = {effective_batch}\n"
                f"   Small batches can cause noisy gradients"
            )

        if effective_batch > 512:
            self.warnings.append(
                f"⚠️  Effective batch size ({effective_batch}) is large\n"
                f"   May need to increase learning_rate proportionally"
            )

    def validate_gradient_clipping(self):
        """Validate gradient clipping configuration."""
        training = self.config.get('training', {})
        max_grad_norm = training.get('max_gradient_norm', 1.0)

        if max_grad_norm is None or max_grad_norm <= 0:
            self.errors.append(
                f"❌ max_gradient_norm is not set or invalid\n"
                f"   Gradient clipping is essential for stable training!"
            )
        elif max_grad_norm < 0.5:
            self.warnings.append(
                f"⚠️  max_gradient_norm ({max_grad_norm}) is very restrictive\n"
                f"   May slow down learning. Typical: 1.0-5.0"
            )

    def validate_data_paths(self):
        """Validate data paths exist."""
        data = self.config.get('data', {})
        data_dir = data.get('data_dir')
        tokenizer_name = data.get('tokenizer_name')

        if data_dir:
            data_path = Path(data_dir)
            if not data_path.is_absolute():
                # Resolve relative to config file
                data_path = self.config_path.parent.parent.parent / data_dir

            if not data_path.exists():
                self.errors.append(f"❌ data_dir does not exist: {data_path}")

        if tokenizer_name:
            tokenizer_path = Path(tokenizer_name)
            if tokenizer_path.exists() and not any(tokenizer_path.glob('*')):
                self.warnings.append(f"⚠️  tokenizer_name path exists but is empty: {tokenizer_path}")

    def run_all_checks(self):
        """Run all validation checks."""
        print(f"\n{'='*80}")
        print(f"🔍 Validating config: {self.config_path}")
        print(f"{'='*80}\n")

        self.validate_lr_schedule()
        self.validate_penalties()
        self.validate_warmup()
        self.validate_batch_size()
        self.validate_gradient_clipping()
        self.validate_data_paths()

        # Print results
        if self.errors:
            print(f"❌ ERRORS FOUND ({len(self.errors)}):")
            print(f"{'='*80}")
            for error in self.errors:
                print(f"{error}\n")

        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            print(f"{'='*80}")
            for warning in self.warnings:
                print(f"{warning}\n")

        if not self.errors and not self.warnings:
            print(f"✅ ALL CHECKS PASSED")
            print(f"{'='*80}\n")
            return True

        return len(self.errors) == 0

    def auto_fix(self):
        """Automatically fix common issues."""
        print(f"\n{'='*80}")
        print(f"🔧 AUTO-FIX MODE")
        print(f"{'='*80}\n")

        # Create backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.config_path.parent / f"{self.config_path.stem}_backup_{timestamp}.yaml"
        shutil.copy2(self.config_path, backup_path)
        print(f"💾 Backup created: {backup_path}\n")

        # Fix inverted LR schedule
        training = self.config.get('training', {})
        lr = training.get('learning_rate')
        lr_end = training.get('lr_end')

        if lr and lr_end and lr_end >= lr:
            # Fix: set lr_end to 1/300 of lr
            new_lr_end = max(lr / 300.0, 1e-7)
            self.config['training']['lr_end'] = float(new_lr_end)
            self.fixes_applied.append(f"✓ Fixed lr_end: {lr_end:.2e} → {new_lr_end:.2e}")

        # Fix low learning rate (if ridiculously low)
        if lr and lr < 1e-6:
            new_lr = 3e-4
            self.config['training']['learning_rate'] = float(new_lr)
            new_lr_end = max(new_lr / 300.0, 1e-7)
            self.config['training']['lr_end'] = float(new_lr_end)
            self.fixes_applied.append(f"✓ Fixed learning_rate: {lr:.2e} → {new_lr:.2e}")
            self.fixes_applied.append(f"✓ Fixed lr_end: {lr_end:.2e} → {new_lr_end:.2e}")

        # Fix excessive penalties
        if training.get('eos_penalty_weight', 0) > 5.0:
            self.config['training']['eos_penalty_weight'] = 0.0
            self.fixes_applied.append(f"✓ Disabled excessive eos_penalty_weight")

        if training.get('eos_logit_bias', 0) > 3.0:
            self.config['training']['eos_logit_bias'] = 0.0
            self.fixes_applied.append(f"✓ Disabled excessive eos_logit_bias")

        # Fix short warmup
        if training.get('warmup_steps', 0) < 500:
            self.config['training']['warmup_steps'] = 1000
            self.fixes_applied.append(f"✓ Increased warmup_steps to 1000")

        # Disable problematic penalties
        enhanced = self.config.get('enhanced_features', {}).get('losses', {})
        if enhanced.get('use_ngram_penalty') and enhanced.get('ngram_penalty_weight', 0) > 2.0:
            self.config['enhanced_features']['losses']['use_ngram_penalty'] = False
            self.config['enhanced_features']['losses']['ngram_penalty_weight'] = 0.0
            self.fixes_applied.append(f"✓ Disabled excessive ngram_penalty")

        if enhanced.get('use_immediate_repetition_detector'):
            self.config['enhanced_features']['losses']['use_immediate_repetition_detector'] = False
            self.config['enhanced_features']['losses']['immediate_repetition_weight'] = 0.0
            self.fixes_applied.append(f"✓ Disabled immediate_repetition_detector")

        # Save fixed config
        if self.fixes_applied:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False, width=120)

            print(f"🔧 FIXES APPLIED:")
            for fix in self.fixes_applied:
                print(f"   {fix}")
            print(f"\n✅ Config updated: {self.config_path}")
            print(f"{'='*80}\n")
        else:
            print(f"ℹ️  No fixes needed\n")


def main():
    parser = argparse.ArgumentParser(description="Validate training config")
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--fix', action='store_true', help='Auto-fix common issues')
    args = parser.parse_args()

    validator = ConfigValidator(args.config)

    if args.fix:
        validator.auto_fix()
        # Re-run validation after fixes
        validator = ConfigValidator(args.config)

    is_valid = validator.run_all_checks()

    if not is_valid:
        if not args.fix:
            print(f"💡 TIP: Run with --fix to auto-fix common issues\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
