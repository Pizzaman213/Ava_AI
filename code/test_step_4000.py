#!/usr/bin/env python3
"""Test the LLM checkpoint at step 4000"""

import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig

def test_checkpoint():
    checkpoint_path = '/project/code/outputs/runs/run_20251011_170204_0f141b67/checkpoints/step_4000/model.pt'

    print("=" * 80)
    print("Testing LLM Checkpoint at Step 4000")
    print("=" * 80)
    print(f"\nCheckpoint path: {checkpoint_path}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Print checkpoint info
    print(f"\nCheckpoint Info:")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.4f}" if isinstance(checkpoint.get('loss'), (int, float)) else f"  Loss: {checkpoint.get('loss', 'unknown')}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    # Load model
    print("\nInitializing model...")
    # Fix any config values that might be strings instead of proper types
    config_dict = checkpoint['config']['model'].copy()
    if 'layer_norm_eps' in config_dict and isinstance(config_dict['layer_norm_eps'], str):
        config_dict['layer_norm_eps'] = float(config_dict['layer_norm_eps'])

    model_config = EnhancedMoEConfig(**config_dict)
    print(f"\nModel Configuration:")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Num layers: {model_config.num_layers}")
    print(f"  Num attention heads: {model_config.num_attention_heads}")
    print(f"  Num experts: {model_config.num_experts}")
    print(f"  Experts per token: {model_config.num_experts_per_token}")
    print(f"  Router type: {model_config.router_type}")

    model = EnhancedMoEModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = model.to(device)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer_path = checkpoint['config']['data'].get('tokenizer_name', 'Qwen/Qwen2.5-0.5B')
    print(f"  Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a world where",
        "Scientists have discovered",
        "The future of technology"
    ]

    print("\n" + "=" * 80)
    print("Generating Text Samples")
    print("=" * 80)

    # Try multiple generation strategies
    generation_strategies = [
        {
            'name': 'Greedy Decoding',
            'params': {
                'max_length': 80,
                'do_sample': False,
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
        },
        {
            'name': 'Nucleus Sampling (p=0.95)',
            'params': {
                'max_length': 80,
                'temperature': 1.0,
                'top_p': 0.95,
                'do_sample': True,
                'repetition_penalty': 2.0,
                'no_repeat_ngram_size': 4,
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
        },
        {
            'name': 'High Temperature Sampling',
            'params': {
                'max_length': 80,
                'temperature': 1.5,
                'top_k': 100,
                'top_p': 0.9,
                'do_sample': True,
                'repetition_penalty': 3.0,
                'no_repeat_ngram_size': 5,
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
        },
    ]

    # Select just one prompt for testing different strategies
    test_prompt = "Once upon a time"

    with torch.no_grad():
        for strategy_idx, strategy in enumerate(generation_strategies, 1):
            print(f"\n{'='*80}")
            print(f"Strategy {strategy_idx}/3: {strategy['name']}")
            print(f"{'='*80}")
            print(f"\nParameters:")
            for key, value in strategy['params'].items():
                print(f"  {key}: {value}")

            print(f"\nPrompt: \"{test_prompt}\"")

            # Tokenize
            inputs = tokenizer(test_prompt, return_tensors='pt').to(device)

            # Generate
            try:
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    **strategy['params']
                )

                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Calculate token count
                num_tokens = outputs.shape[1]
                new_tokens = num_tokens - inputs['input_ids'].shape[1]

                print(f"\nGenerated ({new_tokens} new tokens, {num_tokens} total):")
                print(f"\"{generated_text}\"")

                # Analyze repetition
                tokens = tokenizer.convert_ids_to_tokens(outputs[0])
                unique_tokens = len(set(tokens))
                repetition_ratio = 1 - (unique_tokens / len(tokens))

                print(f"\nAnalysis:")
                print(f"  Unique tokens: {unique_tokens}/{len(tokens)}")
                print(f"  Repetition ratio: {repetition_ratio:.2%}")

            except Exception as e:
                print(f"Error during generation: {e}")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_checkpoint()
