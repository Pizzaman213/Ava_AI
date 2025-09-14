#!/usr/bin/env python3
"""
Interactive text generation for MoE++ models
Works with both base and fine-tuned models
"""
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import yaml
import argparse
from typing import Optional, Dict, Any

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# Import model and tokenizer
from src.model.moe_transformer import MoEForCausalLM, MoEConfig
from transformers import AutoTokenizer


class InteractiveGenerator:
    """Interactive text generator for MoE++ models"""
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._get_device(device)
        
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # Model info
        self.max_length = self.model.config.max_position_embeddings
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Model loaded successfully!")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Max sequence length: {self.max_length}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self) -> MoEForCausalLM:
        """Load model from checkpoint"""
        # Try direct loading first
        try:
            model = MoEForCausalLM.from_pretrained(str(self.checkpoint_path))
            model.to(self.device)
            model.eval()
            return model
        except:
            pass
        
        # Manual loading
        # Find config file
        config_yaml = self.checkpoint_path / "config.yaml"
        config_json = self.checkpoint_path / "config.json"
        
        if config_yaml.exists():
            with open(config_yaml, 'r') as f:
                config_dict = yaml.safe_load(f)
            # Extract model config if nested
            if 'model' in config_dict:
                model_config = config_dict['model']
            else:
                model_config = config_dict
        elif config_json.exists():
            with open(config_json, 'r') as f:
                model_config = json.load(f)
        else:
            raise FileNotFoundError(f"No config file found in {self.checkpoint_path}")
        
        # Clean config
        model_config.pop('size', None)
        
        # Create model
        config = MoEConfig(**model_config)
        model = MoEForCausalLM(config)
        
        # Load weights
        model_pt = self.checkpoint_path / "model.pt"
        pytorch_bin = self.checkpoint_path / "pytorch_model.bin"
        
        if model_pt.exists():
            state_dict = torch.load(model_pt, map_location=self.device)
        elif pytorch_bin.exists():
            state_dict = torch.load(pytorch_bin, map_location=self.device)
        else:
            raise FileNotFoundError(f"No model weights found in {self.checkpoint_path}")
        
        # Handle different checkpoint formats
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        # Try loading from checkpoint first
        tokenizer_path = self.checkpoint_path / "tokenizer_config.json"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        else:
            # Default to GPT-2 tokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate text from prompt"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Determine max tokens
        prompt_length = input_ids.shape[1]
        if max_new_tokens is None:
            max_new_tokens = self.max_length - prompt_length
        else:
            max_new_tokens = min(max_new_tokens, self.max_length - prompt_length)
        
        if max_new_tokens <= 0:
            return prompt  # Already at max length
        
        # Generate
        generated = input_ids
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Prepare inputs
                if past_key_values is None:
                    model_inputs = {"input_ids": generated, "attention_mask": attention_mask}
                else:
                    model_inputs = {
                        "input_ids": generated[:, -1:],
                        "attention_mask": attention_mask,
                        "past_key_values": past_key_values,
                        "use_cache": True
                    }
                
                # Forward pass
                outputs = self.model(**model_inputs)
                
                # Get logits
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("lm_logits"))
                    past_key_values = outputs.get("past_key_values", None)
                else:
                    logits = outputs.logits
                    past_key_values = getattr(outputs, "past_key_values", None)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        next_token_logits[0, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append token
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                ], dim=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text
    
    def interactive_loop(self):
        """Run interactive generation loop"""
        print("\n" + "="*60)
        print("Interactive Text Generation")
        print("="*60)
        print("\nCommands:")
        print("  /help     - Show this help")
        print("  /set      - Show current settings")
        print("  /set <param> <value> - Change generation parameter")
        print("  /quit     - Exit")
        print("\nAvailable parameters:")
        print("  temperature (0.1-2.0) - Randomness of generation")
        print("  top_k (1-100) - Top-k sampling")
        print("  top_p (0.1-1.0) - Nucleus sampling")
        print("  max_tokens (1-{}) - Max new tokens".format(self.max_length))
        print("  repetition_penalty (1.0-2.0) - Penalize repetition")
        print("="*60)
        
        # Default settings
        settings = {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "max_tokens": min(50, self.max_length),
            "repetition_penalty": 1.0,
            "do_sample": True
        }
        
        while True:
            try:
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
                        self.interactive_loop()  # Show help again
                        return
                    
                    elif cmd == "/set":
                        if len(parts) == 1:
                            # Show current settings
                            print("\nCurrent settings:")
                            for k, v in settings.items():
                                print(f"  {k}: {v}")
                        elif len(parts) == 3:
                            # Set parameter
                            param, value = parts[1], parts[2]
                            try:
                                if param == "temperature":
                                    settings[param] = float(value)
                                elif param == "top_p":
                                    settings[param] = float(value)
                                elif param == "repetition_penalty":
                                    settings[param] = float(value)
                                elif param in ["top_k", "max_tokens"]:
                                    settings[param] = int(value)
                                else:
                                    print(f"Unknown parameter: {param}")
                                    continue
                                print(f"Set {param} = {settings[param]}")
                            except ValueError:
                                print(f"Invalid value for {param}")
                        else:
                            print("Usage: /set <param> <value>")
                    
                    else:
                        print(f"Unknown command: {cmd}")
                    
                    continue
                
                # Generate text
                print("\nGenerating...", end="", flush=True)
                
                generated = self.generate(
                    prompt,
                    max_new_tokens=settings["max_tokens"],
                    temperature=settings["temperature"],
                    top_k=settings["top_k"],
                    top_p=settings["top_p"],
                    do_sample=settings["do_sample"],
                    repetition_penalty=settings["repetition_penalty"]
                )
                
                print("\r" + " "*20 + "\r", end="")  # Clear "Generating..."
                print(f"\n{generated}")
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Interactive text generation for MoE++ models")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use (default: auto)")
    parser.add_argument("--test", action="store_true",
                        help="Run test generation instead of interactive mode")
    
    args = parser.parse_args()
    
    # Create generator
    generator = InteractiveGenerator(args.checkpoint, args.device)
    
    if args.test:
        # Run test prompts
        test_prompts = [
            "The capital of France is",
            "Once upon a time",
            "The meaning of life is",
            "In the beginning",
            "Hello, my name is",
        ]
        
        print("\nRunning test generation...")
        print("="*60)
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            result = generator.generate(prompt, max_new_tokens=20)
            print(f"Generated: {result}")
            print("-"*40)
    else:
        # Run interactive mode
        generator.interactive_loop()


if __name__ == "__main__":
    main()