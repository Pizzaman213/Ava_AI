#!/usr/bin/env python3
"""Test generation with the current training run's latest checkpoint"""

import sys
sys.path.insert(0, '/project/code/src')
sys.path.insert(0, '/project/code')

import torch
from transformers import AutoTokenizer
from Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
import inspect
import os

def test_generation():
    # Use the latest checkpoint from current run
    checkpoint_path = '/project/code/outputs/runs/run_20251020_012121_802c6cf4/checkpoints/latest_model.pt'
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model config from checkpoint
    full_config = checkpoint['config']
    model_config_dict = full_config['model']
    valid_params = inspect.signature(EnhancedMoEConfig.__init__).parameters
    filtered_dict = {k: v for k, v in model_config_dict.items() if k in valid_params}

    # Fix type conversions
    if 'layer_norm_eps' in filtered_dict and isinstance(filtered_dict['layer_norm_eps'], str):
        filtered_dict['layer_norm_eps'] = float(filtered_dict['layer_norm_eps'])
    if 'rope_theta' in filtered_dict and isinstance(filtered_dict['rope_theta'], str):
        filtered_dict['rope_theta'] = float(filtered_dict['rope_theta'])

    model_config = EnhancedMoEConfig(**filtered_dict)

    # Load tokenizer - use the custom tokenizer from config
    tokenizer_path = full_config.get('data', {}).get('tokenizer_name', 'Qwen/Qwen2.5-0.5B')
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model = EnhancedMoEModel(model_config)

    # Load state dict with compatibility for architecture changes
    state_dict = checkpoint['model_state_dict']

    # Remove keys that don't exist in current model
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Keys in checkpoint but not in model (will be ignored)
    extra_keys = checkpoint_keys - model_keys
    if extra_keys:
        print(f"Ignoring extra keys from checkpoint: {extra_keys}")
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    # Keys in model but not in checkpoint (will use initialized values)
    missing_keys = model_keys - checkpoint_keys
    if missing_keys:
        print(f"Missing keys (using initialized values): {missing_keys}")

    # Load with strict=False to allow missing keys
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()

    # Get current step from checkpoint
    current_step = checkpoint.get('global_step', checkpoint.get('step', 'unknown'))
    print(f"\nModel at step: {current_step}")

    print("\n" + "=" * 60)
    print(f"Testing Generation - Current Training Run (Step {current_step})")
    print("=" * 60)

    # Test prompts
    prompts = [
        "Once upon a time there was a",
        "The weather today is",
        "In a galaxy far far away",
        "The capital of France is",
    ]

    # Test with reasonable generation parameters
    gen_config = {
        "max_length": 100,
        "temperature": 0.9,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
    }

    print(f"\nGeneration Config: {gen_config}\n")

    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

        # Generate
        with torch.no_grad():
            try:
                gen_config['pad_token_id'] = tokenizer.eos_token_id

                outputs = model.generate(
                    inputs['input_ids'],
                    **gen_config
                )

                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: \"{generated}\"")

                # Check length
                tokens = tokenizer.encode(generated)
                print(f"Length: {len(tokens)} tokens")

                # Check for repetition
                words = generated.split()
                unique_words = len(set(words))
                total_words = len(words)
                diversity = unique_words / total_words if total_words > 0 else 0
                print(f"Diversity: {diversity:.2%} ({unique_words}/{total_words} unique words)")

            except Exception as e:
                print(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()

        print("-" * 60)

if __name__ == "__main__":
    test_generation()
