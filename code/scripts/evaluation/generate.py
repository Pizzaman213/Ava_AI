#!/usr/bin/env python3
"""
Text generation script for MoE++ model
"""
import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.moe_transformer import MoEForCausalLM as MoEModel
from src.generation.text_generator import TextGenerator
from src.utils.config import load_config
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with MoE++ model")
    
    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/finetuned_mps/checkpoint-best",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model config (if not in checkpoint)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling threshold"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    
    checkpoint_path = Path(args.checkpoint)
    
    # Load config
    if args.config:
        config = load_config(args.config)
    elif (checkpoint_path / "config.yaml").exists():
        config = load_config(checkpoint_path / "config.yaml")
    elif (checkpoint_path / "config.json").exists():
        # Load JSON config
        import json
        from src.model.moe_transformer import MoEConfig
        with open(checkpoint_path / "config.json", "r") as f:
            config_dict = json.load(f)
        # Remove problematic fields if they exist
        config_dict.pop('_optimization_configs', None)
        config_dict.pop('_name_or_path', None)
        # Create config object directly
        config = MoEConfig(**config_dict)
        # Wrap in SimpleConfig for compatibility
        class SimpleConfig:
            def __init__(self, model_config):
                self.model = model_config
                self.data = {"tokenizer": "gpt2"}  # Default tokenizer
        config = SimpleConfig(config)
    else:
        # Try to find config in parent directory
        config_path = checkpoint_path.parent.parent / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            raise ValueError("Could not find model config. Please specify --config")
    
    # Initialize model
    model = MoEModel(config.model)
    
    # Check if this is a LoRA fine-tuned model
    state_dict_path = checkpoint_path / "pytorch_model.bin"
    if state_dict_path.exists():
        # Peek at the state dict to see if it's a PEFT model
        state_dict = torch.load(state_dict_path, map_location="cpu")
        is_peft_model = any("lora" in k.lower() or "base_model" in k for k in state_dict.keys())
        
        if is_peft_model:
            print("Detected LoRA fine-tuned model, loading with PEFT...")
            from peft import PeftModel, PeftConfig
            
            # First, we need to load the base model
            # Try to find base model info
            adapter_config_path = checkpoint_path / "adapter_config.json"
            if adapter_config_path.exists():
                # Load PEFT config to get base model info
                peft_config = PeftConfig.from_pretrained(str(checkpoint_path))
                base_model_path = getattr(peft_config, 'base_model_name_or_path', None)
                
                if base_model_path and Path(base_model_path).exists():
                    print(f"Loading base model from {base_model_path}")
                    base_model = MoEModel.from_pretrained(base_model_path)
                else:
                    # Try to find base model in parent directories
                    possible_base_paths = [
                        checkpoint_path.parent.parent / "moe_small_experiment" / "epoch_3",
                        checkpoint_path.parent.parent / "checkpoints" / "best",
                        Path("outputs/moe_small_experiment/epoch_3"),
                    ]
                    
                    base_model = None
                    for base_path in possible_base_paths:
                        if base_path.exists() and (base_path / "pytorch_model.bin").exists():
                            print(f"Found base model at {base_path}")
                            base_model = MoEModel.from_pretrained(str(base_path))
                            break
                    
                    if base_model is None:
                        print("Could not find base model, creating new model from config")
                        base_model = MoEModel(config.model)
            else:
                # No adapter config, just create model from config
                print("No adapter config found, creating new model from config")
                base_model = MoEModel(config.model)
            
            # Make base model compatible with PEFT if needed
            if hasattr(base_model, 'config') and not hasattr(base_model.config, 'model_type'):
                # Add model_type attribute for PEFT compatibility
                base_model.config.model_type = "moe"
            
            # Now load the LoRA weights
            try:
                model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
                model = model.merge_and_unload()  # Merge LoRA weights into base model
                print("LoRA weights merged successfully")
            except Exception as e:
                print(f"Error loading PEFT model: {e}")
                print("Attempting manual LoRA merge...")
                
                # Manual merge as fallback
                from .merge_lora_weights import merge_lora_weights
                merged_weights = merge_lora_weights(state_dict)
                base_model.load_state_dict(merged_weights, strict=False)
                model = base_model
                print("Manual LoRA merge completed")
        else:
            # Regular model loading
            model.load_state_dict(state_dict)
    else:
        # Try other file formats
        if (checkpoint_path / "model.pt").exists():
            state_dict = torch.load(
                checkpoint_path / "model.pt",
                map_location=device
            )
        else:
            # Look for any .pt or .bin file
            model_files = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.bin"))
            if model_files:
                state_dict = torch.load(model_files[0], map_location=device)
            else:
                raise ValueError(f"No model file found in {checkpoint_path}")
        
        # Handle state dict format
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        # Remove module. prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer_name = config.data.get("tokenizer", "gpt2")
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize text generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("-" * 80)
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i+1}:")
        
        response = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.temperature > 0
        )
        
        print(response)
        
        if i < args.num_samples - 1:
            print("-" * 40)


if __name__ == "__main__":
    main()