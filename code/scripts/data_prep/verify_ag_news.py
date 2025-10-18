#!/usr/bin/env python3
"""
Verify AG News dataset is properly set up and ready for training.
"""

import json
from pathlib import Path
import sys

def check_file(path, name):
    """Check if a file exists and return its size."""
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name}: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ {name}: NOT FOUND")
        return False

def verify_setup():
    """Verify AG News setup is complete."""
    print("=" * 60)
    print("AG News Dataset Verification")
    print("=" * 60)

    all_good = True

    # Check directories
    print("\n1. Checking directories...")
    raw_dir = Path("/project/code/data/ag_news/raw")
    processed_dir = Path("/project/code/data/ag_news/processed")

    if raw_dir.exists():
        print("  ✓ Raw data directory exists")
    else:
        print("  ✗ Raw data directory NOT FOUND")
        all_good = False

    if processed_dir.exists():
        print("  ✓ Processed data directory exists")
    else:
        print("  ✗ Processed data directory NOT FOUND")
        all_good = False

    # Check processed files
    print("\n2. Checking processed data files...")
    train_file = processed_dir / "combined_train.jsonl"
    val_file = processed_dir / "combined_val.jsonl"
    stats_file = processed_dir / "processing_stats.json"

    all_good &= check_file(train_file, "Training data")
    all_good &= check_file(val_file, "Validation data")
    all_good &= check_file(stats_file, "Statistics")

    # Check statistics
    if stats_file.exists():
        print("\n3. Checking dataset statistics...")
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        print(f"  ✓ Training samples: {stats['train_samples']:,}")
        print(f"  ✓ Validation samples: {stats['val_samples']:,}")
        print(f"  ✓ Total tokens: {stats['total_tokens']:,}")
        print(f"  ✓ Avg tokens/sample: {stats['total_tokens'] / stats['total_samples']:.1f}")
        print(f"  ✓ Tokenizer: {stats['tokenizer']}")
        print(f"  ✓ Vocab size: {stats['vocab_size']}")

        # Check if we have reasonable amounts of data
        if stats['train_samples'] < 50000:
            print(f"  ⚠️  Warning: Low training sample count ({stats['train_samples']:,})")
        if stats['val_samples'] < 1000:
            print(f"  ⚠️  Warning: Low validation sample count ({stats['val_samples']:,})")
    else:
        print("\n3. Statistics file not found - skipping")
        all_good = False

    # Check training config
    print("\n4. Checking training configuration...")
    config_file = Path("/project/code/configs/gpu/small.yaml")

    if config_file.exists():
        with open(config_file, 'r') as f:
            config_content = f.read()

        if "ag_news/processed" in config_content:
            print("  ✓ Training config points to AG News")
        else:
            print("  ✗ Training config does NOT point to AG News")
            all_good = False
    else:
        print("  ✗ Training config NOT FOUND")
        all_good = False

    # Check tokenizer
    print("\n5. Checking tokenizer...")
    tokenizer_dir = Path("/project/code/models/tokenizer/enhanced-500")

    if tokenizer_dir.exists():
        print("  ✓ Tokenizer directory exists")
        # Check for required tokenizer files
        required_files = ["tokenizer.json", "tokenizer_config.json"]
        for fname in required_files:
            if (tokenizer_dir / fname).exists():
                print(f"  ✓ {fname} found")
            else:
                print(f"  ⚠️  {fname} not found (may be optional)")
    else:
        print("  ✗ Tokenizer directory NOT FOUND")
        all_good = False

    # Sample validation
    print("\n6. Validating sample data...")
    if train_file.exists():
        try:
            with open(train_file, 'r') as f:
                first_line = f.readline()
                sample = json.loads(first_line)

            required_fields = ['text', 'input_ids', 'attention_mask', 'num_tokens']
            for field in required_fields:
                if field in sample:
                    print(f"  ✓ Sample has '{field}' field")
                else:
                    print(f"  ✗ Sample missing '{field}' field")
                    all_good = False

            # Check if it's AG News data
            if 'dataset' in sample and sample['dataset'] == 'ag_news':
                print("  ✓ Sample confirmed as AG News data")
            else:
                print("  ⚠️  Sample dataset field unexpected")

            print(f"  ✓ Sample length: {sample['num_tokens']} tokens")

        except Exception as e:
            print(f"  ✗ Error reading sample: {e}")
            all_good = False

    # Final summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ VERIFICATION PASSED")
        print("=" * 60)
        print("\nAG News dataset is properly set up and ready for training!")
        print("\nTo start training, run:")
        print("  python scripts/5_training/train.py --config configs/gpu/small.yaml")
        print()
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        print("\nSome checks failed. Please run the setup script:")
        print("  bash scripts/data_prep/setup_ag_news.sh")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(verify_setup())
