#!/usr/bin/env python3
"""
Simple text generation example - supports both base and fine-tuned models
"""
import sys
import os
import torch
import yaml
import argparse
import json

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# Import from the project modules
from src.model.moe_transformer import MoEConfig, MoEForCausalLM
from src.inference.generate import TextGenerator, GenerationConfig
from transformers import AutoTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Text generation with MoE++ models")
parser.add_argument('--checkpoint', type=str, default="outputs/checkpoints/best",
                    help='Path to model checkpoint (e.g., outputs/sft_model/checkpoint-best)')
parser.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cpu', 'cuda', 'mps'],
                    help='Device to run on')
args = parser.parse_args()

# Determine checkpoint path
checkpoint_path = args.checkpoint
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint path {checkpoint_path} does not exist")
    sys.exit(1)

# Load model
print(f"Loading model from {checkpoint_path}...")

# Try to load config from different formats
config_yaml_path = os.path.join(checkpoint_path, "config.yaml")
config_json_path = os.path.join(checkpoint_path, "config.json")

if os.path.exists(config_yaml_path):
    # Load config from YAML
    with open(config_yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract model config if nested
    if 'model' in config_dict:
        model_config = config_dict['model'].copy()
    else:
        model_config = config_dict
elif os.path.exists(config_json_path):
    # Load config from JSON
    with open(config_json_path, 'r') as f:
        model_config = json.load(f)
else:
    print(f"Error: No config file found in {checkpoint_path}")
    sys.exit(1)

# Remove non-MoEConfig fields
model_config.pop('size', None)  # Remove 'size' field if present

# Convert string values that should be floats
if 'rms_norm_eps' in model_config and isinstance(model_config['rms_norm_eps'], str):
    model_config['rms_norm_eps'] = float(model_config['rms_norm_eps'])

# Create config object
config = MoEConfig(**model_config)

# Create model and load weights
model = MoEForCausalLM(config)

# Look for model weights in different formats
model_pt_path = os.path.join(checkpoint_path, "model.pt")
pytorch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

if os.path.exists(model_pt_path):
    checkpoint = torch.load(model_pt_path, map_location='cpu')
elif os.path.exists(pytorch_bin_path):
    checkpoint = torch.load(pytorch_bin_path, map_location='cpu')
else:
    print(f"Error: No model weights found in {checkpoint_path}")
    sys.exit(1)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

print("Model loaded successfully!")

# Determine device
if args.device == 'auto':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
else:
    device = torch.device(args.device)

# Move model to device
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Print model info
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Model device: {device}")
print(f"Max sequence length: {config.max_position_embeddings}")

# Load tokenizer (from config, we know it's using GPT-2)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
print("Tokenizer loaded!")

# Default generation parameters (respect model's max position embeddings)
max_length = min(50, config.max_position_embeddings)
temperature = 0.8  # Higher temperature for more diverse outputs
top_k = 50
top_p = 0.9

# Create generation config
gen_config = GenerationConfig(
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    use_speculative_decoding=False,  # Disable for now
    use_rag=False,
)

# Create text generator
generator = TextGenerator(model, tokenizer, gen_config)

# Interactive generation loop
print("\n" + "=" * 60)
print("Interactive Text Generation with MoE++ Model")
print("=" * 60)
print("\nCommands:")
print("  - Type your prompt and press Enter to generate text")
print("  - Type 'quit' or 'exit' to stop")
print("  - Type 'help' for options")
print("\nNote: This is a small model trained on limited data, so outputs may not be fully coherent.")
print("=" * 60)

while True:
    try:
        # Get user input
        prompt = input("\n> ").strip()
        
        # Check for exit commands
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        # Show help
        if prompt.lower() == 'help':
            print("\nGeneration parameters:")
            print(f"  - Max length: {max_length}")
            print(f"  - Temperature: {temperature}")
            print(f"  - Top-k: {top_k}")
            print(f"  - Top-p: {top_p}")
            print("\nTo change parameters, use:")
            print("  set max_length <value>")
            print("  set temperature <value>")
            print("  set top_k <value>")
            print("  set top_p <value>")
            continue
        
        # Handle parameter changes
        if prompt.lower().startswith('set '):
            parts = prompt.split()
            if len(parts) == 3:
                param, value = parts[1], parts[2]
                try:
                    if param == 'max_length':
                        max_length = int(value)
                        print(f"Max length set to {max_length}")
                    elif param == 'temperature':
                        temperature = float(value)
                        print(f"Temperature set to {temperature}")
                    elif param == 'top_k':
                        top_k = int(value)
                        print(f"Top-k set to {top_k}")
                    elif param == 'top_p':
                        top_p = float(value)
                        print(f"Top-p set to {top_p}")
                    else:
                        print(f"Unknown parameter: {param}")
                except ValueError:
                    print(f"Invalid value for {param}")
            else:
                print("Usage: set <parameter> <value>")
            continue
        
        # Skip empty prompts
        if not prompt:
            continue
        
        # Generate text
        print(f"\nGenerating (max_length={max_length}, temp={temperature})...")
        
        # Update generation config
        gen_config.max_length = max_length
        gen_config.temperature = temperature
        gen_config.top_k = top_k
        gen_config.top_p = top_p
        
        # Generate
        outputs = generator.generate(prompt, generation_config=gen_config)
        
        # Decode response
        if isinstance(outputs, torch.Tensor):
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = outputs
        
        print("\n" + "-" * 60)
        print(response)
        print("-" * 60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Type 'quit' to exit or continue with a new prompt.")
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()