#!/usr/bin/env python3
"""
Interactive text generation with trained model
"""

import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.moe_transformer import MoEConfig, MoEForCausalLM
from src.utils.path_utils import get_outputs_dir

def generate_text(model, tokenizer, prompt, device, max_tokens=50, temperature=0.8, top_k=50, show_tokens=False):
    """Generate text from prompt"""
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    if show_tokens:
        print("Generating: ", end="", flush=True)
    
    # Generate tokens
    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Apply temperature and get probabilities
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                min_value = top_k_values[-1]
                next_token_logits[next_token_logits < min_value] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Show progress
            if show_tokens:
                token_text = tokenizer.decode(next_token)
                print(token_text, end="", flush=True)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    if show_tokens:
        print()  # New line after generation
    
    # Decode final text
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated

def main():
    # Load model
    print("🤖 Loading MoE model...")
    print("-" * 60)
    
    # Check for command line argument
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"📁 Using model from command line: {model_path}")
    else:
        model_path = os.environ.get('MODEL_PATH', str(get_outputs_dir('run_20250831_155903/checkpoint_step_4000.pt')))
    
    # Check if model exists
    if not Path(model_path).exists():
        print("❌ Model not found! Looking for alternatives...")
        outputs_dir = get_outputs_dir()
        # Look for checkpoints and best models
        model_files = list(outputs_dir.glob('*/checkpoint_step_*.pt')) + \
                     list(outputs_dir.glob('*/best_model.pt')) + \
                     list(outputs_dir.glob('*/final_model.pt'))
        if model_files:
            # Sort by modification time to get most recent
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = str(model_files[0])
            print(f"✅ Found most recent: {model_path}")
        else:
            print("❌ No model files found in outputs directory!")
            return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = MoEConfig(**checkpoint['config'])
    model = MoEForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"📊 Config: {config.num_layers} layers, {config.num_experts} experts")
    print(f"📝 Training loss: {checkpoint.get('loss', 'N/A')}")
    print("-" * 60)
    
    # Interactive loop
    print("\n🎮 INTERACTIVE TEXT GENERATION")
    print("=" * 60)
    print("Commands:")
    print("  - Type any prompt to generate text")
    print("  - 'settings' to adjust generation parameters")
    print("  - 'examples' for example prompts")
    print("  - 'quit' to exit")
    print("=" * 60)
    
    # Default settings
    temperature = 0.8
    max_tokens = 50
    top_k = 50
    show_tokens = False
    
    # Example prompts
    examples = [
        "The future of artificial intelligence",
        "Machine learning is",
        "In the next decade, technology will",
        "The most important discovery",
        "Once upon a time",
        "Natural language processing",
        "The key to success is",
        "When I think about the future"
    ]
    
    while True:
        print()
        user_input = input("📝 Enter prompt > ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        elif user_input.lower() == 'settings':
            print("\n⚙️  Current Settings:")
            print(f"  Temperature: {temperature} (0.1-2.0, higher=more random)")
            print(f"  Max tokens: {max_tokens} (1-200)")
            print(f"  Top-k: {top_k} (0-100, 0=disabled)")
            print(f"  Show tokens: {show_tokens}")
            
            try:
                new_temp = input(f"  New temperature [{temperature}]: ").strip()
                if new_temp:
                    temperature = float(new_temp)
                
                new_max = input(f"  New max tokens [{max_tokens}]: ").strip()
                if new_max:
                    max_tokens = int(new_max)
                
                new_topk = input(f"  New top-k [{top_k}]: ").strip()
                if new_topk:
                    top_k = int(new_topk)
                
                show_input = input(f"  Show tokens as generated? (y/n) [{show_tokens}]: ").strip().lower()
                if show_input:
                    show_tokens = show_input == 'y'
                
                print("✅ Settings updated!")
            except ValueError:
                print("❌ Invalid input, keeping current settings")
        
        elif user_input.lower() == 'examples':
            print("\n📚 Example prompts:")
            for i, example in enumerate(examples, 1):
                print(f"  {i}. {example}")
            
            choice = input("\nSelect number (or press Enter to skip): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(examples):
                user_input = examples[int(choice) - 1]
                print(f"\n📝 Using: {user_input}")
            else:
                continue
        
        elif not user_input:
            continue
        
        else:
            user_input = user_input  # Use the prompt as-is
        
        # Generate text
        if user_input and user_input.lower() not in ['settings', 'examples']:
            print(f"\n🔮 Generating with temperature={temperature}, max_tokens={max_tokens}, top_k={top_k}...")
            print("-" * 60)
            
            try:
                generated = generate_text(
                    model, tokenizer, user_input, device,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    show_tokens=show_tokens
                )
                
                if not show_tokens:
                    print(f"✨ Generated text:\n")
                    print(generated)
                else:
                    print(f"\n\n✨ Complete text: {generated}")
                
                print("-" * 60)
                
                # Ask for feedback
                feedback = input("Rate this generation (1-5, or Enter to skip): ").strip()
                if feedback.isdigit() and 1 <= int(feedback) <= 5:
                    print(f"Thanks for the feedback! {'⭐' * int(feedback)}")
                    
            except Exception as e:
                print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    main()