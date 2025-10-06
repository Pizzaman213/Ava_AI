#!/usr/bin/env python3
"""
Script to iteratively test different checkpoints and temperature settings
to find the best generation quality.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
RUN_DIR = "/project/code/outputs/runs/run_20251006_015602_9bd4df77"
PROMPT = "Once upon a time, in a land far away, there lived"
MAX_LENGTH = 100

# Test configurations
CHECKPOINTS = [
    "step_10000",
    "step_20000",
    "step_30000",
    "step_40000",
    "step_50000",
    "step_60000",
    "step_70000",
    "step_80000",
    "step_90000",
    "step_100000",
]

TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]

def test_generation(checkpoint, temperature):
    """Test generation with specific checkpoint and temperature."""
    model_path = f"{RUN_DIR}/checkpoints/{checkpoint}/model.pt"

    cmd = [
        "python", "generate.py",
        "--model-path", model_path,
        "--prompt", PROMPT,
        "--temperature", str(temperature),
        "--max-length", str(MAX_LENGTH),
        "--cpu",
        "--no-sample"  # Use greedy for consistency
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/project/code/scripts/generation"
        )

        # Extract generated text from output
        output_lines = result.stdout.strip().split('\n')
        generated_text = ""

        # Find the "Generated:" line
        for i, line in enumerate(output_lines):
            if "Generated:" in line:
                # Get all lines after "Generated:"
                generated_text = '\n'.join(output_lines[i+1:])
                break

        return generated_text.strip(), result.returncode == 0

    except subprocess.TimeoutExpired:
        return "TIMEOUT", False
    except Exception as e:
        return f"ERROR: {str(e)}", False


def main():
    print("="*80)
    print("🔍 Testing Checkpoints and Temperatures")
    print("="*80)
    print(f"Run: {RUN_DIR}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Max Length: {MAX_LENGTH}")
    print("="*80)
    print()

    results = []

    # Test each checkpoint
    for checkpoint in CHECKPOINTS:
        print(f"\n📦 Testing checkpoint: {checkpoint}")
        print("-"*80)

        checkpoint_results = []

        for temp in TEMPERATURES:
            print(f"  🌡️  Temperature {temp}...", end=" ", flush=True)

            generated, success = test_generation(checkpoint, temp)

            if success:
                print("✓")
                checkpoint_results.append({
                    'checkpoint': checkpoint,
                    'temperature': temp,
                    'generated': generated,
                    'success': True
                })
            else:
                print("✗")
                checkpoint_results.append({
                    'checkpoint': checkpoint,
                    'temperature': temp,
                    'generated': generated,
                    'success': False
                })

        results.extend(checkpoint_results)

    # Print summary
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)

    for result in results:
        if result['success']:
            print(f"\n🔹 {result['checkpoint']} | Temp: {result['temperature']}")
            print(f"   {result['generated'][:200]}...")
        else:
            print(f"\n❌ {result['checkpoint']} | Temp: {result['temperature']} - FAILED")

    print("\n" + "="*80)
    print("✅ Testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
