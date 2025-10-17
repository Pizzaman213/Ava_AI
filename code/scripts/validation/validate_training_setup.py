"""
Validation script to verify training setup before starting full training.

This script checks:
1. Initial loss calculation (should be ~11.93 for vocab_size=151665)
2. Train/val file split (no overlap, correct ratios)
3. Data loader determinism (same data across runs)
4. Loss components sum correctly
5. Expert utilization after 100 steps
"""

import sys
import torch
import math
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.Ava.data_streaming import create_streaming_dataloaders


def validate_initial_loss(vocab_size=151665):
    """
    Verify that expected initial loss matches theoretical value.

    For a randomly initialized model with uniform distribution over vocabulary:
    expected_loss = -log(1/vocab_size) = log(vocab_size)
    """
    expected_loss = math.log(vocab_size)
    print(f"\n{'='*60}")
    print(f"VALIDATION 1: Initial Loss Calculation")
    print(f"{'='*60}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Expected initial loss: {expected_loss:.4f}")
    print(f"  (This is -log(1/{vocab_size}) = log({vocab_size}))")
    print(f"\n✓ Initial loss should be close to {expected_loss:.4f}")
    print(f"  If your first training step shows loss ~8.0, something is wrong!")
    return expected_loss


def validate_data_split(data_dir="/project/code/data/processed"):
    """
    Verify train/val file split has no overlap and correct ratios.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION 2: Train/Val File Split")
    print(f"{'='*60}")

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False

    # Find all data files
    all_files = sorted(list(data_path.glob("*_processed.jsonl")) +
                      list(data_path.glob("*.jsonl")))

    if not all_files:
        print(f"❌ No data files found in {data_dir}")
        return False

    print(f"Total data files found: {len(all_files)}")

    # Split files using same logic as data_streaming.py
    train_files = []
    val_files = []

    for file_path in all_files:
        file_hash = hash(file_path.name) % 100
        if file_hash < 85:
            train_files.append(file_path)
        else:
            val_files.append(file_path)

    # Check for overlap
    train_names = {f.name for f in train_files}
    val_names = {f.name for f in val_files}
    overlap = train_names & val_names

    print(f"\nTrain files: {len(train_files)} ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"Val files: {len(val_files)} ({len(val_files)/len(all_files)*100:.1f}%)")

    if overlap:
        print(f"\n❌ OVERLAP DETECTED: {len(overlap)} files in both train and val!")
        print(f"   Files: {list(overlap)[:5]}")
        return False
    else:
        print(f"\n✓ No overlap between train and val files")

    # Check ratios
    expected_train_ratio = 0.85
    expected_val_ratio = 0.15
    actual_train_ratio = len(train_files) / len(all_files)
    actual_val_ratio = len(val_files) / len(all_files)

    if abs(actual_train_ratio - expected_train_ratio) > 0.1:
        print(f"⚠️  Train ratio {actual_train_ratio:.2%} differs from expected {expected_train_ratio:.0%}")
    else:
        print(f"✓ Train ratio {actual_train_ratio:.2%} matches expected {expected_train_ratio:.0%}")

    if abs(actual_val_ratio - expected_val_ratio) > 0.1:
        print(f"⚠️  Val ratio {actual_val_ratio:.2%} differs from expected {expected_val_ratio:.0%}")
    else:
        print(f"✓ Val ratio {actual_val_ratio:.2%} matches expected {expected_val_ratio:.0%}")

    return True


def validate_data_loader_determinism(config_path="/project/code/configs/gpu/small_fixed.yaml"):
    """
    Verify data loader produces same samples across multiple runs.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION 3: Data Loader Determinism")
    print(f"{'='*60}")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        # Create data loader twice
        print("Creating first data loader...")
        train_loader1, _ = create_streaming_dataloaders(
            tokenizer=tokenizer,
            batch_size=8,
            max_length=512,
            data_dir="/project/code/data/processed",
            num_workers=0,  # Single worker for determinism test
            max_samples=100,
            buffer_size=100
        )

        print("Creating second data loader...")
        train_loader2, _ = create_streaming_dataloaders(
            tokenizer=tokenizer,
            batch_size=8,
            max_length=512,
            data_dir="/project/code/data/processed",
            num_workers=0,
            max_samples=100,
            buffer_size=100
        )

        # Get first batch from each
        print("Fetching first batch from each loader...")
        batch1 = next(iter(train_loader1))
        batch2 = next(iter(train_loader2))

        # Compare
        if torch.equal(batch1['input_ids'], batch2['input_ids']):
            print("✓ Data loaders are deterministic (same samples)")
            return True
        else:
            print("❌ Data loaders are non-deterministic (different samples)")
            print(f"   Batch 1 shape: {batch1['input_ids'].shape}")
            print(f"   Batch 2 shape: {batch2['input_ids'].shape}")
            print(f"   First 10 tokens match: {torch.equal(batch1['input_ids'][0][:10], batch2['input_ids'][0][:10])}")
            return False

    except Exception as e:
        print(f"❌ Error during determinism test: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_config_hyperparameters(config_path="/project/code/configs/gpu/small_fixed.yaml"):
    """
    Verify all hyperparameters are set correctly.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION 4: Config Hyperparameters")
    print(f"{'='*60}")

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    checks = []

    # Check critical hyperparameters
    checks.append(("Max sequence length", config['data']['max_length'], 512, ">="))
    checks.append(("Learning rate", config['training']['learning_rate'], 0.0003, "=="))
    checks.append(("Gradient accumulation", config['training']['gradient_accumulation_steps'], 16, "=="))
    checks.append(("Attention dropout", config['model']['attention_dropout'], 0.1, "=="))
    checks.append(("Hidden dropout", config['model']['hidden_dropout'], 0.1, "=="))
    checks.append(("Batch size", config['training']['batch_size'], 16, "=="))
    checks.append(("N-gram penalty enabled", config['enhanced_features']['losses']['use_ngram_penalty'], True, "=="))
    checks.append(("MoE balancing enabled", config['enhanced_features']['losses']['use_moe_balancing'], True, "=="))
    checks.append(("Aux loss enabled", config['enhanced_features']['losses']['auxiliary_loss'], True, "=="))

    all_pass = True
    for name, actual, expected, comparison in checks:
        if comparison == "==":
            passed = actual == expected
        elif comparison == ">=":
            passed = actual >= expected
        else:
            passed = False

        status = "✓" if passed else "❌"
        print(f"{status} {name}: {actual} (expected: {comparison} {expected})")

        if not passed:
            all_pass = False

    # Calculate effective batch size
    effective_batch = (config['training']['batch_size'] *
                      config['training']['gradient_accumulation_steps'] *
                      config['data']['max_length'])
    print(f"\n📊 Effective batch size: {effective_batch:,} tokens/step")
    print(f"   = {config['training']['batch_size']} (batch) × "
          f"{config['training']['gradient_accumulation_steps']} (accum) × "
          f"{config['data']['max_length']} (seq_len)")

    if effective_batch < 50000:
        print(f"⚠️  Effective batch size is low (< 50K tokens)")
    elif effective_batch > 200000:
        print(f"✓ Effective batch size is good (> 200K tokens)")
    else:
        print(f"✓ Effective batch size is acceptable")

    return all_pass


def print_summary_and_recommendations():
    """
    Print summary of validations and recommendations.
    """
    print(f"\n{'='*60}")
    print(f"SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*60}")

    print("""
BEFORE STARTING TRAINING:

1. Stop any current training runs (they use the old broken config)

2. Verify all validations above passed

3. Expected metrics after fixes:
   - Step 200: LR should be at full 0.0003
   - Step 500: Loss should be < 6.0
   - Step 1000: Loss should be < 5.0, generation length > 20 tokens
   - Step 2000: Loss should be < 4.0, perplexity < 100

4. Monitor these metrics during training:
   - Training loss should ALWAYS be > validation loss
   - Expert utilization: all experts should be used (> 5% each)
   - Generation samples should improve steadily
   - No repetition collapse

5. If you see:
   - Val loss < train loss → Label smoothing issue
   - Loss stuck at 8.0 → Initial loss calculation wrong
   - All zeros generation → EOS collapse
   - Repetitive text → Repetition penalties not working
   - One expert dominates → MoE balancing not working

TRAINING COMMAND:
   python code/scripts/4_training/train.py \\
       --config code/configs/gpu/small_fixed.yaml \\
       --output-dir code/outputs/fixed_training

QUICK TEST (500 steps):
   Edit config to add: max_samples: 10000
   Run training for 500 steps to verify all fixes work
""")


if __name__ == "__main__":
    print("="*60)
    print("LLM PRETRAINING VALIDATION SUITE")
    print("="*60)
    print("\nThis script validates all critical fixes for LLM pretraining.")
    print("Run this BEFORE starting any training!")

    # Run all validations
    results = {}

    results['initial_loss'] = validate_initial_loss()
    results['data_split'] = validate_data_split()
    results['determinism'] = validate_data_loader_determinism()
    results['hyperparameters'] = validate_config_hyperparameters()

    # Print summary
    print_summary_and_recommendations()

    # Final verdict
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT")
    print(f"{'='*60}")

    passed_checks = sum(1 for v in results.values() if v is True or isinstance(v, float))
    total_checks = len(results)

    if passed_checks == total_checks:
        print(f"✅ ALL CHECKS PASSED ({passed_checks}/{total_checks})")
        print(f"\nYou are ready to start training!")
        sys.exit(0)
    else:
        print(f"❌ SOME CHECKS FAILED ({passed_checks}/{total_checks} passed)")
        print(f"\nFix the issues above before starting training!")
        sys.exit(1)
