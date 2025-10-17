#!/usr/bin/env python3
"""
Diagnostic script to test checkpoint generation behavior and understand
why the model is hitting EOS so quickly.
"""

import sys
sys.path.append('/project/code')

import torch
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from transformers import AutoTokenizer
from src.Ava.models.moe_model import EnhancedMoEModel
from src.Ava.generation.generator import TextGenerator

print("=" * 80)
print("🔍 Generation Diagnostic Tool - Step 42000 Checkpoint")
print("=" * 80)
print()

# Load checkpoint
checkpoint_path = Path("/project/code/outputs/runs/run_20251014_113843_3ae62fda/checkpoints/latest_model.pt")
print(f"📂 Loading: {checkpoint_path.name}")
print(f"   Size: {checkpoint_path.stat().st_size / 1024**3:.2f} GB")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
step = checkpoint.get('global_step', 'unknown')
loss = checkpoint.get('metrics', {}).get('train_loss', None)
if loss is not None:
    print(f"✅ Loaded - Step: {step}, Loss: {loss:.4f}")
else:
    print(f"✅ Loaded - Step: {step}")
print()

# Load tokenizer
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
print(f"   Vocab size: {len(tokenizer)}")
print(f"   EOS token: '{tokenizer.decode([tokenizer.eos_token_id])}' (id={tokenizer.eos_token_id})")
print(f"   PAD token: '{tokenizer.decode([tokenizer.pad_token_id])}' (id={tokenizer.pad_token_id})")
print()

# Load model
print("🏗️  Building model...")
config = checkpoint.get('config', checkpoint.get('model_config'))
from src.Ava.models.moe_model import EnhancedMoEConfig
import inspect

if isinstance(config, dict):
    # If config has 'model' key, extract it
    if 'model' in config:
        model_dict = config['model']
        # Filter to only valid parameters
        valid_params = inspect.signature(EnhancedMoEConfig.__init__).parameters
        filtered_dict = {k: v for k, v in model_dict.items() if k in valid_params}

        # Fix type conversions for numeric strings
        for key in ['layer_norm_eps', 'rope_theta']:
            if key in filtered_dict and isinstance(filtered_dict[key], str):
                filtered_dict[key] = float(filtered_dict[key])

        model_config = EnhancedMoEConfig(**filtered_dict)
    else:
        valid_params = inspect.signature(EnhancedMoEConfig.__init__).parameters
        filtered_dict = {k: v for k, v in config.items() if k in valid_params}

        # Fix type conversions
        for key in ['layer_norm_eps', 'rope_theta']:
            if key in filtered_dict and isinstance(filtered_dict[key], str):
                filtered_dict[key] = float(filtered_dict[key])

        model_config = EnhancedMoEConfig(**filtered_dict)
else:
    model_config = config

model = EnhancedMoEModel(model_config)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
print("✅ Model ready")
print()

# Create generator
generator = TextGenerator(model, tokenizer, device=torch.device('cpu'))

# Test prompts with varying difficulty
test_cases = [
    {
        "prompt": "Once upon a time",
        "params": {
            "max_length": 80,
            "min_length": 30,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
            "eos_penalty": 2.5
        }
    },
    {
        "prompt": "The quick brown",
        "params": {
            "max_length": 60,
            "min_length": 20,
            "temperature": 0.9,
            "top_p": 0.95,
            "repetition_penalty": 1.3,
            "eos_penalty": 3.0
        }
    },
    {
        "prompt": "Hello",
        "params": {
            "max_length": 50,
            "min_length": 15,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.8,
            "eos_penalty": 4.0
        }
    }
]

print("=" * 80)
print("🎯 Generation Tests")
print("=" * 80)
print()

for i, test in enumerate(test_cases, 1):
    prompt = test["prompt"]
    params = test["params"]

    print(f"Test {i}/3: \"{prompt}\"")
    print("-" * 80)
    print(f"Parameters:")
    for key, val in params.items():
        print(f"  {key:20s}: {val}")
    print()

    # Generate
    print("⏳ Generating...")
    try:
        output = generator.generate(prompt, **params)

        # Analyze output
        full_text = prompt + " " + output
        words = full_text.split()
        unique_words = len(set(words))
        total_words = len(words)
        uniqueness = unique_words / total_words if total_words > 0 else 0

        print("✅ Generated:")
        print()
        print(f"   {full_text}")
        print()
        print(f"📊 Analysis:")
        print(f"   Total words: {total_words}")
        print(f"   Unique words: {unique_words}")
        print(f"   Uniqueness: {uniqueness:.1%}")
        print(f"   Output length: {len(output.split())} words")

        if uniqueness < 0.5:
            print(f"   ⚠️  High repetition detected")
        elif len(output.split()) < 5:
            print(f"   ⚠️  Very short output - may have hit EOS early")
        else:
            print(f"   ✅ Looks reasonable")

    except Exception as e:
        print(f"❌ Error: {e}")

    print()

print("=" * 80)
print("🧪 Logits Analysis - Single Token Prediction")
print("=" * 80)
print()

# Analyze what the model predicts for a simple prompt
test_prompt = "Once upon a time"
inputs = tokenizer(test_prompt, return_tensors="pt")
input_ids = inputs['input_ids']

print(f"Prompt: \"{test_prompt}\"")
print(f"Input tokens: {input_ids.shape[1]}")
print()

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs['logits'][0, -1, :]  # Last token predictions

    # Get top 10 predictions
    probs = torch.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probs, k=10)

    print("Top 10 most likely next tokens:")
    print("-" * 80)
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_str = tokenizer.decode([idx.item()])
        is_eos = " ← EOS!" if idx.item() == tokenizer.eos_token_id else ""
        print(f"  {rank:2d}. {token_str:20s}  (prob: {prob.item():.2%}){is_eos}")

    # Check EOS probability
    eos_prob = probs[tokenizer.eos_token_id].item()
    eos_rank = (probs > eos_prob).sum().item() + 1

    print()
    print(f"EOS token rank: {eos_rank} (probability: {eos_prob:.4%})")

    if eos_rank <= 3:
        print("⚠️  EOS is in top 3 predictions - model strongly wants to stop!")
    elif eos_rank <= 10:
        print("⚠️  EOS is in top 10 - model may stop early")
    else:
        print("✅ EOS is not dominant - generation should continue")

print()
print("=" * 80)
print("✅ Diagnostic complete!")
print()
print("💡 Interpretation:")
print("   • If EOS is top prediction: Model learned to end too quickly")
print("   • If high repetition: Model needs more training diversity")
print("   • If short outputs: Increase eos_penalty or check min_length")
print("   • If gibberish: Model still learning, needs more training")
print("=" * 80)
