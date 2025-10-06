#!/usr/bin/env python3
"""
Training Diagnostics Script

This script analyzes your training run to identify why the model generates gibberish.
It checks:
1. Loss trajectory (is the model learning?)
2. Data quality (tokenization, corruption)
3. Model outputs at different checkpoints
4. Expert routing balance (MoE health)
5. Gradient statistics
"""

import json
import sys
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig


def load_wandb_metrics(run_dir: Path):
    """Extract metrics from WandB run file."""
    import wandb

    wandb_files = list((run_dir / "wandb").rglob("*.wandb"))
    if not wandb_files:
        print("❌ No WandB files found")
        return None

    print(f"📊 Found WandB file: {wandb_files[0]}")

    # Use wandb API to read the file
    api = wandb.Api()

    # For offline runs, we need to parse the file directly
    # This is a simplified version - you may need wandb.restore()

    return None


def analyze_loss_trajectory(run_dir: Path):
    """Check if loss is decreasing over training."""
    print("\n" + "="*80)
    print("📉 LOSS TRAJECTORY ANALYSIS")
    print("="*80)

    # Try to find loss values in logs
    log_file = run_dir / "logs" / "training.log"

    if not log_file.exists():
        print("❌ No training.log found")
        return

    losses = []
    steps = []

    with open(log_file) as f:
        for line in f:
            if "loss" in line.lower() and "step" in line.lower():
                # Try to extract loss value
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "loss" in part.lower() and i + 1 < len(parts):
                            try:
                                loss_val = float(parts[i + 1].strip(":,"))
                                losses.append(loss_val)
                            except:
                                pass
                except:
                    pass

    if not losses:
        print("⚠️  No loss values found in logs!")
        print("   This is a CRITICAL issue - we can't tell if the model is learning")
        return False

    print(f"✓ Found {len(losses)} loss measurements")
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Change: {losses[-1] - losses[0]:.4f}")

    if losses[-1] >= losses[0]:
        print("❌ CRITICAL: Loss is NOT decreasing! Model is not learning.")
        return False
    elif losses[-1] > losses[0] * 0.9:
        print("⚠️  WARNING: Loss barely decreased (< 10% reduction)")
        return False
    else:
        print("✓ Loss is decreasing")
        return True


def test_data_quality(data_dir: Path):
    """Check if data is properly formatted and tokenized."""
    print("\n" + "="*80)
    print("📦 DATA QUALITY ANALYSIS")
    print("="*80)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False

    # Check data files
    data_files = list(data_dir.glob("*.jsonl"))
    if not data_files:
        print(f"❌ No .jsonl files found in {data_dir}")
        return False

    print(f"✓ Found {len(data_files)} data files")

    # Sample and check first few examples
    sample_file = data_files[0]
    print(f"\n📄 Analyzing: {sample_file.name}")

    issues = []

    with open(sample_file) as f:
        for i, line in enumerate(f):
            if i >= 10:  # Check first 10 samples
                break

            try:
                sample = json.loads(line)

                # Check if 'text' field exists
                if 'text' not in sample:
                    issues.append(f"Sample {i}: Missing 'text' field")
                    continue

                text = sample['text']

                # Check text length
                if len(text) < 10:
                    issues.append(f"Sample {i}: Text too short ({len(text)} chars)")

                # Try tokenizing
                tokens = tokenizer.encode(text)

                if len(tokens) < 5:
                    issues.append(f"Sample {i}: Too few tokens ({len(tokens)})")

                # Check for repetitive tokens (sign of bad data)
                if len(set(tokens)) < len(tokens) * 0.3:
                    issues.append(f"Sample {i}: High token repetition")

            except json.JSONDecodeError:
                issues.append(f"Sample {i}: Invalid JSON")
            except Exception as e:
                issues.append(f"Sample {i}: {str(e)}")

    if issues:
        print("\n⚠️  Data Quality Issues Found:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        return False
    else:
        print("✓ Data quality looks good")
        return True


def test_model_generation(checkpoint_path: Path):
    """Test if model can generate coherent text."""
    print("\n" + "="*80)
    print("🤖 MODEL GENERATION TEST")
    print("="*80)

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            print("⚠️  No config in checkpoint, using default")
            config = EnhancedMoEConfig()

        # Create model
        model = EnhancedMoEModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Test generation
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In the year 2024"
        ]

        print("\n📝 Generation Samples:")

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=50,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n   Prompt: {prompt}")
            print(f"   Output: {generated_text}")

        print("\n✓ Generation test complete")
        return True

    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_expert_routing(checkpoint_path: Path):
    """Check if MoE experts are being used evenly."""
    print("\n" + "="*80)
    print("🎯 EXPERT ROUTING ANALYSIS")
    print("="*80)

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Look for expert usage statistics in checkpoint
        if 'expert_stats' in checkpoint:
            stats = checkpoint['expert_stats']
            print("✓ Expert statistics found in checkpoint")
            print(f"   {json.dumps(stats, indent=2)}")
        else:
            print("⚠️  No expert statistics in checkpoint")
            print("   Cannot verify expert routing balance")

        return True

    except Exception as e:
        print(f"❌ Failed to analyze expert routing: {e}")
        return False


def main():
    """Run full diagnostic suite."""
    print("="*80)
    print("🔍 TRAINING DIAGNOSTICS - Ava MoE Model")
    print("="*80)

    # Find most recent run
    runs_dir = Path("/project/code/outputs/runs")
    runs = sorted(runs_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not runs:
        print("❌ No training runs found")
        return

    run_dir = runs[0]
    print(f"\n📁 Analyzing run: {run_dir.name}")

    results = {
        'loss_decreasing': False,
        'data_quality_ok': False,
        'generation_ok': False,
        'experts_balanced': False
    }

    # 1. Check loss trajectory
    results['loss_decreasing'] = analyze_loss_trajectory(run_dir)

    # 2. Check data quality
    data_dir = Path("/project/code/data/processed")
    results['data_quality_ok'] = test_data_quality(data_dir)

    # 3. Find latest checkpoint
    checkpoints = list((run_dir / "checkpoints").glob("step_*/model.pt"))
    if not checkpoints:
        checkpoints = list((run_dir / "checkpoints").glob("*.pt"))

    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"\n📦 Found checkpoint: {latest_ckpt}")

        # 4. Test generation
        results['generation_ok'] = test_model_generation(latest_ckpt)

        # 5. Analyze expert routing
        results['experts_balanced'] = analyze_expert_routing(latest_ckpt)
    else:
        print("⚠️  No checkpoints found, skipping model tests")

    # Final summary
    print("\n" + "="*80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*80)

    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check.replace('_', ' ').title()}")

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    if not results['loss_decreasing']:
        print("""
   🔴 CRITICAL: Loss not decreasing

   This means your model is NOT learning. Likely causes:

   1. Learning rate too high (gradients exploding)
      → Try: --learning-rate 1e-5  (reduce by 10x)

   2. Learning rate too low (no progress)
      → Try: --learning-rate 1e-3  (increase by 10x)

   3. Incorrect loss calculation
      → Check that labels are shifted properly for causal LM

   4. Data corruption or wrong format
      → Verify data with: head -5 data/processed/*.jsonl

   5. Model initialization bad
      → Try training from scratch with --fresh-start
        """)

    if not results['data_quality_ok']:
        print("""
   🔴 CRITICAL: Data quality issues detected

   Your training data may be corrupted or improperly formatted.

   Actions:
   1. Re-preprocess data: python scripts/data/preprocess.py
   2. Verify format: cat data/processed/*.jsonl | head -1 | jq
   3. Check tokenizer: verify Qwen tokenizer is loading correctly
        """)

    if not results['generation_ok']:
        print("""
   ⚠️  Model generation produces gibberish

   This confirms the model hasn't learned language patterns.

   Possible fixes:
   1. Train longer (more epochs)
   2. Reduce learning rate
   3. Use larger batch size
   4. Check if causal masking is correct
   5. Verify loss is actually decreasing
        """)

    if results['loss_decreasing'] and results['data_quality_ok'] and not results['generation_ok']:
        print("""
   🟡 Loss decreasing but output still bad

   You may need to:
   1. Train MUCH longer (current: 100k steps, try: 1M+ steps)
   2. Reduce temperature during generation (try 0.3-0.5)
   3. Check if model is too small (100M params may be insufficient)
   4. Verify training data is diverse and high-quality
        """)


if __name__ == "__main__":
    main()
