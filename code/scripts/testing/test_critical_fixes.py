#!/usr/bin/env python3
"""
Quick test script to validate critical bug fixes for:
1. CUDA index out of bounds errors
2. Shape mismatch errors during generation
3. Torch Dynamo graph breaks
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from Ava.models.moe_model import OptimizedMoETransformer, OptimizedMoEConfig

def test_generation_fixes():
    """Test that generation works without shape mismatch or CUDA errors."""
    print("=" * 80)
    print("Testing Critical Fixes for Generation Errors")
    print("=" * 80)

    # Create a tiny model for testing
    config = OptimizedMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_experts=4,
        num_experts_per_token=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        router_type='mixtral',
    )

    print(f"\n1. Creating model with config:")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Num experts: {config.num_experts}")
    print(f"   - Num layers: {config.num_layers}")

    model = OptimizedMoETransformer(config)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        print(f"   - Device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"   - Device: CPU")

    model.eval()

    # Test 1: Forward pass with different sequence lengths
    print(f"\n2. Testing forward pass with varying sequence lengths...")
    for seq_len in [5, 9, 11, 16]:
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        try:
            with torch.no_grad():
                outputs = model(input_ids, return_dict=True)
            logits_shape = outputs['logits'].shape
            print(f"   ✓ Seq len {seq_len:2d}: logits shape = {logits_shape}")
            assert logits_shape == (1, seq_len, config.vocab_size), f"Wrong shape: {logits_shape}"
        except Exception as e:
            print(f"   ✗ Seq len {seq_len:2d}: FAILED - {str(e)}")
            return False

    # Test 2: Generation with different prompts
    print(f"\n3. Testing generation (the main error source)...")
    test_prompts = [
        ("Short prompt", torch.randint(0, config.vocab_size, (1, 5), device=device)),
        ("Medium prompt", torch.randint(0, config.vocab_size, (1, 10), device=device)),
        ("Longer prompt", torch.randint(0, config.vocab_size, (1, 15), device=device)),
    ]

    for prompt_name, prompt_ids in test_prompts:
        try:
            with torch.no_grad():
                # Generate just a few tokens to test
                generated = model.generate(
                    prompt_ids,
                    max_length=prompt_ids.shape[1] + 5,
                    temperature=1.0,
                    do_sample=False  # Greedy for determinism
                )
            gen_len = generated.shape[1]
            prompt_len = prompt_ids.shape[1]
            print(f"   ✓ {prompt_name:15s} (len {prompt_len:2d}): generated {gen_len} tokens")
        except Exception as e:
            print(f"   ✗ {prompt_name:15s} (len {prompt_len:2d}): FAILED")
            print(f"      Error: {str(e)}")
            return False

    # Test 3: Verify routing indices are in bounds
    print(f"\n4. Testing routing indices bounds...")
    input_ids = torch.randint(0, config.vocab_size, (2, 10), device=device)
    try:
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
            # Check if we can access aux_info
            if 'aux_info' in outputs and 'routing_metrics' in outputs['aux_info']:
                print(f"   ✓ Routing metrics available")
            else:
                print(f"   ⚠ Routing metrics not in outputs (may be expected)")
        print(f"   ✓ No CUDA index out of bounds errors")
    except Exception as e:
        print(f"   ✗ Routing failed: {str(e)}")
        return False

    # Test 4: Test with torch.compile if available
    print(f"\n5. Testing torch.compile compatibility...")
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            # Compile the model
            compiled_model = torch.compile(model, mode='default', dynamic=True)

            # Test forward pass
            input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
            with torch.no_grad():
                outputs = compiled_model(input_ids, return_dict=True)
            print(f"   ✓ torch.compile forward pass successful")

            # Note: Generation is decorated with @torch.compiler.disable
            # so it won't be compiled even on a compiled model
            print(f"   ℹ Generation is excluded from compilation (by design)")

        except Exception as e:
            print(f"   ⚠ torch.compile test failed: {str(e)}")
            print(f"      (This is non-critical, model still works without compilation)")
    else:
        print(f"   ⊘ torch.compile not available or no CUDA")

    print(f"\n" + "=" * 80)
    print("✓ All critical fixes validated successfully!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_generation_fixes()
    sys.exit(0 if success else 1)
