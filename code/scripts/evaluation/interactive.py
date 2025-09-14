#!/usr/bin/env python3
"""
Interactive text generation with MoE++ model
"""
import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.moe_transformer import MoEForCausalLM as MoEModel, MoEConfig
from src.generation.text_generator import TextGenerator
from src.utils.config import load_config, DotDict
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Interactive text generation with MoE++")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (use --demo for random weights)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode with random weights (no checkpoint needed)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model config"
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
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
    
    if args.demo:
        print("Demo mode: Using random weights (output will be nonsense)")
        
        # Create a small demo config
        model_config = MoEConfig(
            vocab_size=50257,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=512,
            num_experts=4,
            num_experts_per_tok=2,
            use_mod=False,
        )
        
        config = DotDict({
            'model': model_config,
            'data': {
                'tokenizer': 'gpt2'
            }
        })
        
        # Initialize model with random weights
        model = MoEModel(model_config)
        model.to(device)
        model.eval()
    else:
        print("Loading model... (this may take a moment)")
        
        if not args.checkpoint:
            print("\nError: No checkpoint specified!")
            print("Use --checkpoint path/to/checkpoint or --demo for demo mode")
            return
        
        # Load model and config
        checkpoint_path = Path(args.checkpoint)
        
        # Load config
        if args.config:
            config = load_config(args.config)
        elif (checkpoint_path / "config.yaml").exists():
            config = load_config(checkpoint_path / "config.yaml")
        else:
            # Try parent directory (for checkpoints like checkpoint_8000)
            config_path = checkpoint_path.parent / "config.yaml"
            if config_path.exists():
                config = load_config(config_path)
            else:
                # Try parent's parent directory
                config_path = checkpoint_path.parent.parent / "config.yaml"
                if config_path.exists():
                    config = load_config(config_path)
                else:
                    raise ValueError(f"Could not find model config. Looked in:\n"
                                   f"  - {checkpoint_path / 'config.yaml'}\n"
                                   f"  - {checkpoint_path.parent / 'config.yaml'}\n"
                                   f"  - {checkpoint_path.parent.parent / 'config.yaml'}")
        
        # Always use hierarchical for trained models
        expert_type = 'hierarchical'
            
        # Add any missing default values to config
        model_defaults = {
            'expert_type': expert_type,
            'aux_loss_coef': 0.01,
            'mlp_bias': False,
            'router_z_loss_coef': 0.0,
            'mod_capacity_factor': 1.25,
            'router_aux_loss_coef': 0.01,
            'tie_word_embeddings': False,
            'dropout': 0.0,
            'attention_dropout': 0.0,
            'activation_dropout': 0.0,
            'layer_norm_epsilon': 1e-6,
            'use_return_dict': True,
            'output_hidden_states': False,
            'output_attentions': False,
            'torchscript': False,
            'torch_dtype': 'float32',
            'use_bfloat16': False,
            'tf_legacy_loss': False,
            'pruned_heads': {},
            'tie_encoder_decoder': False,
            'is_encoder_decoder': False,
            'is_decoder': False,
            'cross_layer_interval': 1,
            'prune_heads': {},
            'chunk_size_feed_forward': 0,
            'output_router_probs': False,
        }
        
        for key, value in model_defaults.items():
            if key not in config.model:
                config.model[key] = value
        
        # Force expert_type based on checkpoint detection
        config.model['expert_type'] = expert_type
        print(f"Detected expert_type: {expert_type}")
            
        # Convert config dict to MoEConfig object
        # Filter out unknown fields
        from dataclasses import fields
        valid_fields = {f.name for f in fields(MoEConfig)}
        config_dict = {k: v for k, v in config.model.items() if k in valid_fields}
        model_config = MoEConfig(**config_dict)
        
        # Initialize model
        model = MoEModel(model_config)
        
        # Load checkpoint
        if (checkpoint_path / "pytorch_model.bin").exists():
            state_dict = torch.load(
                checkpoint_path / "pytorch_model.bin",
                map_location=device
            )
        elif (checkpoint_path / "model.pt").exists():
            state_dict = torch.load(
                checkpoint_path / "model.pt",
                map_location=device
            )
        else:
            model_files = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.bin"))
            if model_files:
                state_dict = torch.load(model_files[0], map_location=device)
            else:
                raise ValueError(f"No model file found in {checkpoint_path}")
        
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        # Remove module. prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Try loading with strict=False first
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print("Warning: Model architecture mismatch. Loading with strict=False...")
                model.load_state_dict(new_state_dict, strict=False)
            else:
                raise e
        model.to(device)
        model.eval()
    
    # Initialize tokenizer
    tokenizer_name = config.data.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize generator
    generator = TextGenerator(model, tokenizer, device)
    
    print("\nMoE++ Interactive Generation")
    print("=" * 50)
    print("Commands:")
    print("  /help     - Show this help")
    print("  /temp N   - Set temperature (current: {:.1f})".format(args.temperature))
    print("  /len N    - Set max length (current: {})".format(args.max_length))
    print("  /quit     - Exit")
    print("=" * 50)
    print("\nEnter your prompt (or /quit to exit):\n")
    
    temperature = args.temperature
    max_length = args.max_length
    
    while True:
        try:
            # Get user input
            prompt = input("\n> ").strip()
            
            if not prompt:
                continue
            
            # Handle commands
            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0].lower()
                
                if cmd == "/quit":
                    print("Goodbye!")
                    break
                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /help     - Show this help")
                    print("  /temp N   - Set temperature (current: {:.1f})".format(temperature))
                    print("  /len N    - Set max length (current: {})".format(max_length))
                    print("  /quit     - Exit")
                elif cmd == "/temp" and len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"Temperature set to {temperature:.1f}")
                    except ValueError:
                        print("Invalid temperature value")
                elif cmd == "/len" and len(parts) > 1:
                    try:
                        max_length = int(parts[1])
                        print(f"Max length set to {max_length}")
                    except ValueError:
                        print("Invalid length value")
                else:
                    print("Unknown command. Type /help for help.")
                continue
            
            # Generate response
            print("\nGenerating...\n")
            
            response = generator.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=temperature > 0
            )
            
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()