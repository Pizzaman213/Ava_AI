#!/usr/bin/env python3
"""
Quick generation test script for training checkpoints.
Tests model coherence at various temperature/top-p settings.
"""

import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.Ava.models.moe_model import EnhancedMoEModel
from transformers import AutoTokenizer


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model_state_dict", checkpoint)

    # Get model config from checkpoint
    config_dict = checkpoint.get("config")
    if config_dict is None:
        # Fallback: use small config
        import yaml
        with open("/project/code/configs/gpu/small.yaml") as f:
            config_dict = yaml.safe_load(f)

    # Get model config
    model_config = config_dict.get("model") if isinstance(config_dict, dict) else config_dict.model

    # Initialize model
    from src.Ava.models.moe_model import EnhancedMoEConfig
    moe_config = EnhancedMoEConfig(**model_config)
    model = EnhancedMoEModel(moe_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    return model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cuda"
):
    """Generate text from prompt."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Settings: temp={temperature}, top_p={top_p}, top_k={top_k}")
    print(f"{'='*60}")

    with torch.no_grad():
        generated = input_ids.clone()

        for _ in range(max_length):
            # Forward pass
            outputs = model(generated)
            # Handle dict or tensor output
            if isinstance(outputs, dict):
                logits = outputs["logits"][:, -1, :]
            else:
                logits = outputs[:, -1, :]  # Last token logits

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"\nGenerated ({len(generated[0])} tokens):")
    print(f"{generated_text}")
    print(f"{'='*60}\n")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Test model text generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Generation prompt")
    parser.add_argument("--max-length", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--test-suite", action="store_true", help="Run full test suite with multiple settings")

    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = load_checkpoint(args.checkpoint, args.device)

    if args.test_suite:
        print("\n" + "="*60)
        print("RUNNING TEST SUITE")
        print("="*60)

        test_prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In a world where",
            "def fibonacci(n):",
            "Q: What is the capital of France?\nA:",
        ]

        test_configs = [
            {"temperature": 0.7, "top_p": 0.9, "top_k": 50},  # Balanced
            {"temperature": 1.0, "top_p": 0.95, "top_k": 0},  # Creative
            {"temperature": 0.3, "top_p": 0.9, "top_k": 40},  # Focused
        ]

        for prompt in test_prompts:
            for config in test_configs:
                generate_text(
                    model, tokenizer, prompt,
                    max_length=args.max_length,
                    device=args.device,
                    **config
                )
    else:
        generate_text(
            model, tokenizer, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=args.device
        )


if __name__ == "__main__":
    main()
