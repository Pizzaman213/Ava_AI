#!/usr/bin/env python3
"""
Text generation script for trained Qwen MoE++ models.

This script provides text generation capabilities using models trained on data from
/project/code/data/pretraining/processed/. It supports various generation strategies
including greedy decoding, beam search, and nucleus sampling.

Usage:
    python generate.py --model-path outputs/checkpoint.pt --prompt "Once upon a time"
    python generate.py --model-path outputs/checkpoint.pt --interactive
    python generate.py --model-path outputs/checkpoint.pt --input-file prompts.txt --output-file responses.txt

Examples:
    # Single prompt generation
    python generate.py --model-path outputs/model.pt --prompt "The future of AI is" --max-length 100

    # Interactive mode
    python generate.py --model-path outputs/model.pt --interactive --temperature 0.8 --top-p 0.9

    # Batch generation from file
    python generate.py --model-path outputs/model.pt --input-file prompts.txt --output-file outputs.txt
"""

import argparse
import sys
import torch
import yaml
from pathlib import Path
from typing import Optional, List, Union
import json
from tqdm import tqdm

# Add project root to path
sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.generation.generator import TextGenerator
from transformers import AutoTokenizer


class GenerationPipeline:
    """
    Complete generation pipeline for Qwen MoE++ models.

    This pipeline handles model loading, tokenization, and various generation
    strategies for producing high-quality text outputs.

    Args:
        model_path (str): Path to trained model checkpoint
        config_path (str, optional): Path to model configuration file
        device (str): Device to run inference on ('cuda', 'cpu')

    Example:
        >>> pipeline = GenerationPipeline("outputs/model.pt")
        >>> text = pipeline.generate("The meaning of life is", max_length=100)
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        # Check if this is a DeepSpeed checkpoint
        is_deepspeed = 'mp_rank_00_model_states.pt' in model_path or model_path.endswith('/step_257000')

        # Handle DeepSpeed checkpoint path
        if model_path.endswith('/step_257000'):
            deepspeed_path = Path(model_path) / 'step_257000' / 'mp_rank_00_model_states.pt'
            meta_path = Path(model_path) / 'model.pt'
        else:
            deepspeed_path = Path(model_path) if is_deepspeed else None
            meta_path = Path(model_path).parent.parent / 'model.pt' if is_deepspeed else None

        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            model_config = config_dict.get('model', {})
        elif is_deepspeed and meta_path and meta_path.exists():
            # Load config from metadata checkpoint for DeepSpeed
            meta_checkpoint = torch.load(meta_path, map_location=self.device, weights_only=False)
            if 'config' in meta_checkpoint:
                model_config = meta_checkpoint['config'].get('model', meta_checkpoint['config'])
            else:
                raise ValueError("No configuration found in metadata. Please provide --config-path")
        else:
            # Try to load config from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'config' in checkpoint:
                model_config = checkpoint['config'].get('model', checkpoint['config'])
            else:
                raise ValueError("No configuration found. Please provide --config-path")

        # Initialize model
        print("📚 Loading model...")
        self.config = EnhancedMoEConfig(**model_config)
        self.model = EnhancedMoEModel(self.config)

        # Load checkpoint
        if is_deepspeed and deepspeed_path and deepspeed_path.exists():
            print(f"🔄 Loading DeepSpeed checkpoint from {deepspeed_path}")
            ds_checkpoint = torch.load(deepspeed_path, map_location=self.device, weights_only=False)
            if 'module' in ds_checkpoint:
                self.model.load_state_dict(ds_checkpoint['module'])
                print(f"✅ DeepSpeed model loaded from {deepspeed_path}")
            else:
                raise ValueError("Invalid DeepSpeed checkpoint format")
        elif Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'module' in checkpoint:  # DeepSpeed format
                self.model.load_state_dict(checkpoint['module'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ No checkpoint found at {model_path}, using random initialization")

        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize generator
        self.generator = TextGenerator(self.model, self.tokenizer)

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
        do_sample: bool = True
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).

        Args:
            prompt: Input prompt(s) for generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy decoding)

        Returns:
            Generated text(s)
        """
        single_prompt = isinstance(prompt, str)
        if single_prompt:
            prompt = [prompt]

        generated_texts = []

        for p in tqdm(prompt, desc="Generating", disable=len(prompt) == 1):
            output = self.generator.generate(
                prompt=p,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
            generated_texts.append(output)

        return generated_texts[0] if single_prompt else generated_texts

    def interactive_generation(self):
        """
        Interactive generation mode for real-time text generation.
        """
        print("\n" + "="*60)
        print("🤖 Interactive Generation Mode")
        print("="*60)
        print("Enter your prompts (type 'quit' to exit)")
        print("Commands: /settings - show settings, /set <param> <value> - update parameter")
        print("="*60 + "\n")

        # Default settings
        settings = {
            'max_length': 100,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.2
        }

        while True:
            try:
                prompt = input("\n📝 Prompt: ").strip()

                if prompt.lower() == 'quit':
                    break

                if prompt.startswith('/settings'):
                    print("\nCurrent settings:")
                    for k, v in settings.items():
                        print(f"  {k}: {v}")
                    continue

                if prompt.startswith('/set'):
                    parts = prompt.split()
                    if len(parts) == 3:
                        param, value = parts[1], parts[2]
                        if param in settings:
                            try:
                                settings[param] = type(settings[param])(value)
                                print(f"✅ Updated {param} to {value}")
                            except ValueError:
                                print(f"❌ Invalid value for {param}")
                        else:
                            print(f"❌ Unknown parameter: {param}")
                    continue

                if not prompt:
                    continue

                print("\n🔄 Generating...\n")
                response = self.generate(prompt, **settings)
                print(f"🤖 Response:\n{response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained Qwen MoE++ model')

    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-path', type=str,
                       help='Path to model configuration (if not in checkpoint)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')

    # Generation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--prompt', type=str,
                           help='Single prompt for generation')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Interactive generation mode')
    mode_group.add_argument('--input-file', type=str,
                           help='File containing prompts (one per line)')

    # Output
    parser.add_argument('--output-file', type=str,
                       help='File to save generated texts')

    # Generation parameters
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling probability threshold')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--num-beams', type=int, default=1,
                       help='Number of beams for beam search')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='Penalty for repeating tokens')
    parser.add_argument('--no-sample', action='store_true',
                       help='Use greedy decoding instead of sampling')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = GenerationPipeline(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )

    # Handle different modes
    if args.interactive:
        pipeline.interactive_generation()

    elif args.prompt:
        # Single prompt generation
        output = pipeline.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample
        )

        print(f"\n📝 Prompt: {args.prompt}")
        print(f"🤖 Generated:\n{output}")

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output)
            print(f"\n✅ Saved to {args.output_file}")

    elif args.input_file:
        # Batch generation from file
        with open(args.input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"📚 Loaded {len(prompts)} prompts from {args.input_file}")

        outputs = pipeline.generate(
            prompt=prompts,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample
        )

        if args.output_file:
            with open(args.output_file, 'w') as f:
                for prompt, output in zip(prompts, outputs):
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Response: {output}\n")
                    f.write("-" * 50 + "\n")
            print(f"✅ Saved {len(outputs)} responses to {args.output_file}")
        else:
            for prompt, output in zip(prompts, outputs):
                print(f"\n📝 Prompt: {prompt}")
                print(f"🤖 Generated: {output}")
                print("-" * 50)


if __name__ == "__main__":
    main()