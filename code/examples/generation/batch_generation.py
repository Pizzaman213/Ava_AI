#!/usr/bin/env python3
"""
Batch text generation for testing fine-tuned models
"""
import sys
import os
import torch
import yaml
import json
import argparse
from typing import List

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from src.model.moe_transformer import MoEConfig, MoEForCausalLM
from transformers import AutoTokenizer

def load_model(checkpoint_path: str, device: str = 'auto'):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load using from_pretrained if possible
    try:
        model = MoEForCausalLM.from_pretrained(checkpoint_path)
        print("Loaded using from_pretrained")
    except:
        # Fallback to manual loading
        # Load config
        config_yaml_path = os.path.join(checkpoint_path, "config.yaml")
        config_json_path = os.path.join(checkpoint_path, "config.json")
        
        if os.path.exists(config_yaml_path):
            with open(config_yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            if 'model' in config_dict:
                model_config = config_dict['model']
            else:
                model_config = config_dict
        elif os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                model_config = json.load(f)
        else:
            raise FileNotFoundError(f"No config file found in {checkpoint_path}")
        
        # Create model
        model_config.pop('size', None)
        config = MoEConfig(**model_config)
        model = MoEForCausalLM(config)
        
        # Load weights
        model_pt_path = os.path.join(checkpoint_path, "model.pt")
        pytorch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        
        if os.path.exists(model_pt_path):
            state_dict = torch.load(model_pt_path, map_location='cpu')
        elif os.path.exists(pytorch_bin_path):
            state_dict = torch.load(pytorch_bin_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
        
        model.load_state_dict(state_dict)
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    return model, device

def test_generation(model, tokenizer, prompts: List[str], device, **gen_kwargs):
    """Test generation with multiple prompts"""
    results = []
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
        results.append({
            'prompt': prompt,
            'generated': generated
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch generation with fine-tuned models")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompts-file', type=str,
                        help='File containing prompts (one per line)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--num-return-sequences', type=int, default=1,
                        help='Number of sequences to generate per prompt')
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.checkpoint, args.device)
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    max_length = min(args.max_length, model.config.max_position_embeddings)
    
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {device}")
    print(f"  Max sequence length: {model.config.max_position_embeddings}")
    print(f"  Using max_length: {max_length}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default test prompts
        prompts = [
            "The capital of France is",
            "In the beginning",
            "Once upon a time",
            "The meaning of life is",
            "To cook pasta, you should",
            "The best programming language is",
            "Hello, my name is",
            "The weather today is"
        ]
    
    print(f"\nTesting with {len(prompts)} prompts...")
    
    # Test generation
    gen_kwargs = {
        'max_length': max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'do_sample': True,
        'num_return_sequences': args.num_return_sequences,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    
    results = test_generation(model, tokenizer, prompts, device, **gen_kwargs)
    
    # Summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['prompt']}")
        print(f"   → {result['generated']}")

if __name__ == "__main__":
    main()