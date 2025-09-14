#!/usr/bin/env python3
"""
Quick test of trained model - single generation
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

# Load model
print("Loading model (this may take a moment due to size)...")
# Try to find the best available model
model_path = os.environ.get('MODEL_PATH')
if not model_path:
    # Look for a recent best model
    outputs_dir = get_outputs_dir()
    model_candidates = list(outputs_dir.glob('*/best_model.pt')) + list(outputs_dir.glob('*/final_model.pt'))
    if model_candidates:
        # Use the most recently modified model
        model_path = str(max(model_candidates, key=lambda x: x.stat().st_mtime))
    else:
        model_path = str(get_outputs_dir('run_20250831_133933/best_model.pt'))  # fallback
checkpoint = torch.load(model_path, map_location='cpu')
config = MoEConfig(**checkpoint['config'])
model = MoEForCausalLM(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Simple test
prompt = "Solve for x: 2x + 5 = 13"
print(f"\nPrompt: {prompt}")

# Tokenize
inputs = tokenizer(prompt, return_tensors='pt').to(device)
input_ids = inputs['input_ids']

# Generate 20 tokens
print("Generating...")
with torch.no_grad():
    for i in range(20):
        outputs = model(input_ids)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Get next token
        next_token_logits = logits[0, -1, :] / 0.8  # temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        # Decode current sequence
        current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"  Token {i+1}: {tokenizer.decode(next_token)} -> {current_text}")
        
        if next_token.item() == tokenizer.eos_token_id:
            break

print(f"\nFinal: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")