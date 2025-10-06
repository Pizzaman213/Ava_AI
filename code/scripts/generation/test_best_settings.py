#!/usr/bin/env python3
"""
Efficient script to find best checkpoint and temperature settings.
Loads each checkpoint once and tests multiple temperatures.
"""

import sys
import torch
from pathlib import Path

sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.generation.generator import TextGenerator
from transformers import AutoTokenizer

# Configuration
RUN_DIR = Path("/project/code/outputs/runs/run_20251006_015602_9bd4df77")
PROMPT = "Once upon a time, in a land far away, there lived"
MAX_LENGTH = 100

# Test configurations
CHECKPOINTS = [
    "step_40000",
    "step_60000",
    "step_80000",
    "step_100000",
]

TEMPERATURES = [0.5, 0.7, 0.9, 1.0]

def load_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    print(f"  Loading checkpoint...", end=" ", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        if isinstance(config_data, dict) and 'model' in config_data:
            model_config = config_data['model']
        else:
            model_config = config_data
    else:
        raise ValueError("No config found in checkpoint")

    # Initialize model
    config = EnhancedMoEConfig(**model_config)
    model = EnhancedMoEModel(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("✓")
    return model

def test_generation(generator, temperature):
    """Test generation with specific temperature."""
    try:
        output = generator.generate(
            prompt=PROMPT,
            max_length=MAX_LENGTH,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True
        )
        return output, True
    except Exception as e:
        return f"ERROR: {str(e)}", False

def main():
    print("="*80)
    print("🔍 Finding Best Checkpoint and Temperature Settings")
    print("="*80)
    print(f"Prompt: '{PROMPT}'")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Testing {len(CHECKPOINTS)} checkpoints × {len(TEMPERATURES)} temperatures")
    print("="*80)
    print()

    # Initialize tokenizer once
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print()

    all_results = []

    # Test each checkpoint
    for checkpoint_name in CHECKPOINTS:
        print(f"\n{'='*80}")
        print(f"📦 Checkpoint: {checkpoint_name}")
        print(f"{'='*80}")

        checkpoint_path = RUN_DIR / "checkpoints" / checkpoint_name / "model.pt"

        if not checkpoint_path.exists():
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            continue

        # Load checkpoint once
        try:
            model = load_checkpoint(checkpoint_path)
            generator = TextGenerator(model, tokenizer)
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")
            continue

        # Test all temperatures with this checkpoint
        for temp in TEMPERATURES:
            print(f"\n  🌡️  Temperature {temp}:")
            print(f"  {'-'*76}")

            generated, success = test_generation(generator, temp)

            if success:
                # Clean up output
                clean_output = generated.strip()
                print(f"  ✓ {clean_output}")

                all_results.append({
                    'checkpoint': checkpoint_name,
                    'temperature': temp,
                    'output': clean_output,
                    'length': len(clean_output.split())
                })
            else:
                print(f"  ❌ {generated}")

        # Clean up model to free memory
        del model, generator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY - All Results")
    print("="*80)

    for result in all_results:
        print(f"\n🔹 {result['checkpoint']} | Temp: {result['temperature']} | Words: {result['length']}")
        print(f"   {result['output'][:150]}...")

    print("\n" + "="*80)
    print("✅ Testing complete! Review the outputs above to choose the best settings.")
    print("="*80)

if __name__ == "__main__":
    main()
