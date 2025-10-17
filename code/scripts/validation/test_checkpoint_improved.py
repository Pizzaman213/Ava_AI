#!/usr/bin/env python3
"""Test generation with improved parameters to avoid early EOS"""

import sys
sys.path.insert(0, '/project/code/src')
sys.path.insert(0, '/project/code')

import torch
from transformers import AutoTokenizer
from Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
import inspect
import os

def test_generation():
    # Load checkpoint with backward compatibility
    checkpoint_path = '/project/code/outputs/runs/run_20251014_113843_3ae62fda/checkpoints/step_41000/model.pt'
    print(f"Loading checkpoint: {checkpoint_path}")

    # Create src symlink for backward compatibility
    if not os.path.exists('/project/code/src/src'):
        try:
            os.symlink('/project/code/src', '/project/code/src/src')
        except:
            pass

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

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model = EnhancedMoEModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    print("\n" + "=" * 60)
    print(f"Testing generation - Step 41000 with Improved Parameters")
    print("=" * 60)

    # Test different generation strategies
    test_configs = [
        {
            "name": "High Temperature + No Sampling",
            "max_length": 100,
            "temperature": 1.2,
            "do_sample": False,  # Greedy
        },
        {
            "name": "Very Low Temperature (Greedy)",
            "max_length": 100,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50,
        },
        {
            "name": "No Repetition Penalty",
            "max_length": 100,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,  # Disabled
        },
    ]

    prompts = [
        "Once upon a time there was a",
        "The weather today is",
    ]

    for config in test_configs:
        print(f"\n{'=' * 60}")
        print(f"Config: {config['name']}")
        print(f"{'=' * 60}")

        for prompt in prompts:
            print(f"\nPrompt: \"{prompt}\"")

            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

            # Generate
            with torch.no_grad():
                try:
                    gen_config = {k: v for k, v in config.items() if k != 'name'}
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

                except Exception as e:
                    print(f"Generation failed: {e}")
                    import traceback
                    traceback.print_exc()

            print("-" * 60)

if __name__ == "__main__":
    test_generation()
