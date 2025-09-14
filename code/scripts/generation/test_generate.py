#!/usr/bin/env python3
"""
Simple script to test generation from a trained model
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

def generate(model, tokenizer, prompt, device, max_length=50, temperature=0.8, top_k=50, top_p=0.95, 
             repetition_penalty=1.2, no_repeat_ngram_size=3):
    """
    Generate text from a prompt with repetition penalty.
    
    Args:
        repetition_penalty: Penalty for repeating tokens (>1.0 = discourage repetition)
        no_repeat_ngram_size: Block n-grams of this size from repeating
    """
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    # Track generated n-grams to prevent repetition
    def get_ngrams(tokens, n):
        """Get all n-grams from a list of tokens"""
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i:i+n]))
        return ngrams
    
    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            # Get model output
            outputs = model(input_ids)
            
            # Extract logits
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs
            
            # Get next token logits and apply temperature
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply repetition penalty to tokens that have appeared
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    # Reduce probability of tokens that have appeared
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
            
            # Block n-gram repetition
            if no_repeat_ngram_size > 0 and len(input_ids[0]) >= no_repeat_ngram_size:
                # Get all existing n-grams
                prev_tokens = input_ids[0].tolist()
                if len(prev_tokens) >= no_repeat_ngram_size - 1:
                    # Get the last n-1 tokens
                    last_ngram_prefix = tuple(prev_tokens[-(no_repeat_ngram_size-1):])
                    
                    # Check each possible next token
                    for token_id in range(len(next_token_logits)):
                        # Would this create a repeated n-gram?
                        potential_ngram = last_ngram_prefix + (token_id,)
                        
                        # Check if this n-gram exists earlier in the sequence
                        for i in range(len(prev_tokens) - no_repeat_ngram_size + 1):
                            if tuple(prev_tokens[i:i+no_repeat_ngram_size]) == potential_ngram:
                                # Block this token
                                next_token_logits[token_id] = -float('Inf')
                                break
            
            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_value = top_k_values[-1]
                next_token_logits[next_token_logits < min_value] = -float('Inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Stop if EOS token generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode and return
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def test_generation():
    # Path to your trained model - configurable via environment variable
    model_path = os.environ.get('MODEL_PATH', str(get_outputs_dir('run_20250911_155555/checkpoint_step_20000.pt')))
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Looking for other models...")
        outputs_dir = get_outputs_dir()
        model_files = list(outputs_dir.glob('*/best_model.pt')) + list(outputs_dir.glob('*/final_model.pt'))
        if model_files:
            model_path = str(model_files[0])
            print(f"Using {model_path}")
        else:
            print("No model files found!")
            return
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = MoEConfig(**checkpoint['config'])
    model = MoEForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to best available device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts
    prompts = [
        "The future of AI is",
        "Machine learning can help",
        "In the next decade, technology will",
        "Natural language processing is",
        "The most important thing about"
    ]
    
    print("\n" + "="*60)
    print("Testing Generation")
    print("="*60)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        # Generate text with repetition penalty
        generated_text = generate(model, tokenizer, prompt, device, 
                                max_length=50, temperature=0.8, top_k=50, top_p=0.95,
                                repetition_penalty=1.3, no_repeat_ngram_size=3)
        
        print(f"Generated: {generated_text}")
    
    # Skip interactive mode for automated testing
    print("\n" + "="*60)
    print("Testing completed successfully!")

if __name__ == "__main__":
    test_generation()