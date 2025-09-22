#!/usr/bin/env python3
"""
Quick generation tests for various prompts
"""
import torch
import sys
import yaml

# Add project root to path
sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from transformers import AutoTokenizer

def quick_test_generation():
    """Test generation with various prompts"""
    print("🚀 Quick Generation Tests")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")

    # Load model
    checkpoint_path = "/project/code/outputs/checkpoint_step_5000/step_5000/mp_rank_00_model_states.pt"
    config_path = "/project/code/configs/gpu/small.yaml"

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    model_config = config_dict.get('model', {})

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['module']

    config = EnhancedMoEConfig(**model_config)
    model = EnhancedMoEModel(config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The benefits of machine learning include",
        "In the year 2030, technology will",
        "Python is a programming language that",
        "The most important thing to remember is"
    ]

    print(f"\n🧪 Testing {len(test_prompts)} different prompts:")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"📝 Prompt: '{prompt}'")

        # Generate
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 30,  # Short generations for testing
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"🤖 Generated: '{generated_text}'")

    print(f"\n✅ All generation tests completed!")

if __name__ == "__main__":
    quick_test_generation()