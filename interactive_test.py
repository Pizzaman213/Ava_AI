#!/usr/bin/env python3
"""
Interactive testing script for the trained LLM checkpoint
"""
import torch
import sys
import yaml
from pathlib import Path
import time

# Add project root to path
sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from transformers import AutoTokenizer

class InteractiveLLMTester:
    def __init__(self, checkpoint_path, config_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")

        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config = config_dict.get('model', {})

        # Load model from DeepSpeed checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint['module']

        self.config = EnhancedMoEConfig(**model_config)
        self.model = EnhancedMoEModel(self.config)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def generate_text(self, prompt, max_length=100, temperature=0.8, top_p=0.9, top_k=50):
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_tokens = len(generated_ids[0]) - len(input_ids[0])

        return generated_text, generation_time, new_tokens

    def interactive_mode(self):
        """Interactive generation mode"""
        print("\n" + "="*60)
        print("🤖 Interactive LLM Testing Mode")
        print("="*60)
        print("Commands:")
        print("  /help - Show this help")
        print("  /settings - Show current settings")
        print("  /set <param> <value> - Change parameter")
        print("  /quit or /exit - Exit")
        print("  Just type text to generate!")
        print("="*60 + "\n")

        # Default settings
        settings = {
            'max_length': 50,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50
        }

        while True:
            try:
                user_input = input("\n📝 Prompt: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['/quit', '/exit']:
                    print("👋 Goodbye!")
                    break

                if user_input == '/help':
                    print("\nAvailable commands:")
                    print("  /settings - Show current generation settings")
                    print("  /set max_length <num> - Set max generation length")
                    print("  /set temperature <float> - Set sampling temperature (0.1-2.0)")
                    print("  /set top_p <float> - Set nucleus sampling threshold (0.1-1.0)")
                    print("  /set top_k <int> - Set top-k sampling (1-100)")
                    print("  /quit or /exit - Exit interactive mode")
                    continue

                if user_input == '/settings':
                    print("\nCurrent settings:")
                    for k, v in settings.items():
                        print(f"  {k}: {v}")
                    continue

                if user_input.startswith('/set'):
                    parts = user_input.split()
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
                    else:
                        print("❌ Usage: /set <parameter> <value>")
                    continue

                # Generate text
                print("\n🔄 Generating...")
                generated_text, gen_time, new_tokens = self.generate_text(
                    user_input, **settings
                )

                print(f"\n🤖 Generated ({new_tokens} tokens in {gen_time:.2f}s):")
                print(f"📄 {generated_text}")
                print(f"⚡ Speed: {new_tokens/gen_time:.1f} tokens/sec")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    print("🚀 Interactive LLM Testing")

    checkpoint_path = "/project/code/outputs/checkpoint_step_5000/step_5000/mp_rank_00_model_states.pt"
    config_path = "/project/code/configs/gpu/small.yaml"

    try:
        tester = InteractiveLLMTester(checkpoint_path, config_path)
        tester.interactive_mode()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()