#!/usr/bin/env python3
"""
Test a specific checkpoint quickly
"""

import torch
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.moe_transformer import MoEConfig, MoEForCausalLM
from src.utils.path_utils import get_outputs_dir

def test_checkpoint(model_path):
    """Test a checkpoint with various prompts"""
    
    print(f"🔍 Testing checkpoint: {model_path}")
    print("="*60)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = MoEConfig(**checkpoint['config'])
    model = MoEForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Model info
    print(f"✅ Model loaded on {device}")
    print(f"📊 Config: {config.num_layers} layers, {config.num_experts} experts")
    print(f"📈 Training step: {checkpoint.get('step', 'N/A')}")
    print(f"📉 Training loss: {checkpoint.get('loss', 'N/A'):.4f}" if checkpoint.get('loss') else "📉 Training loss: N/A")
    print("-"*60)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Once upon a time",
        "def fibonacci(n):",
        "The most important thing about",
    ]
    
    print("\n🧪 Testing generation with sample prompts:")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-"*40)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        # Generate
        with torch.no_grad():
            # Use model's generate method if available, otherwise simple generation
            try:
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=50,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            except:
                # Simple generation fallback with repetition penalty
                input_ids = inputs['input_ids']
                generated_tokens = []
                
                for _ in range(30):
                    outputs = model(input_ids)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    next_token_logits = logits[0, -1, :]
                    
                    # Apply repetition penalty
                    repetition_penalty = 1.3
                    for token_id in set(input_ids[0].tolist()):
                        if next_token_logits[token_id] > 0:
                            next_token_logits[token_id] /= repetition_penalty
                        else:
                            next_token_logits[token_id] *= repetition_penalty
                    
                    # Extra penalty for recent tokens
                    if len(generated_tokens) > 0:
                        for token_id in generated_tokens[-5:]:
                            next_token_logits[token_id] -= 2.0
                    
                    # Temperature (higher = more random, lower = more focused)
                    temperature = 0.9
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k and top-p filtering
                    top_k = 40
                    top_p = 0.9
                    
                    # Filter to top-k
                    top_k_values, _ = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                    next_token_logits[next_token_logits < top_k_values[-1]] = -float('Inf')
                    
                    # Apply top-p
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    generated_tokens.append(next_token.item())
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        print(f"✨ Generated: {generated}")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    
    # Performance metrics
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"  • Parameters: {param_count:,}")
    print(f"  • Memory usage: ~{param_count * 4 / 1024**2:.1f} MB (FP32)")
    
    return model, tokenizer

def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default to the checkpoint you mentioned
        model_path = os.environ.get('MODEL_PATH', str(get_outputs_dir('run_20250906_123813/checkpoint_step_10000.pt')))

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        
        # Look for alternatives
        outputs_dir = get_outputs_dir()
        checkpoints = list(outputs_dir.glob("*/checkpoint_*.pt"))
        if checkpoints:
            print("\n📁 Available checkpoints:")
            for cp in sorted(checkpoints)[-5:]:  # Show last 5
                print(f"  • {cp}")
            print("\nUsage: python3 test_checkpoint.py <checkpoint_path>")
        return
    
    test_checkpoint(model_path)

if __name__ == "__main__":
    main()