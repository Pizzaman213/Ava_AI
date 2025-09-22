#!/usr/bin/env python3
"""
Test script for the trained LLM checkpoint at step 5000
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

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict.get('model', {})

def load_model_from_deepspeed_checkpoint(checkpoint_path, config_dict):
    """Load model from DeepSpeed checkpoint format"""
    print(f"🔧 Loading DeepSpeed checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model state dict from DeepSpeed format
    if 'module' in checkpoint:
        model_state_dict = checkpoint['module']
        print(f"✅ Found model state dict with {len(model_state_dict)} parameters")
    else:
        raise ValueError("No 'module' key found in checkpoint")

    # Initialize model
    config = EnhancedMoEConfig(**config_dict)
    model = EnhancedMoEModel(config)

    # Load state dict
    model.load_state_dict(model_state_dict)

    print(f"✅ Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config

def test_forward_pass(model, tokenizer, device='cpu'):
    """Test basic forward pass"""
    print(f"\n🧪 Testing forward pass on {device}...")

    model.to(device)
    model.eval()

    # Create test input
    test_text = "The future of artificial intelligence"
    inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)

    print(f"📝 Input text: '{test_text}'")
    print(f"🔢 Input shape: {input_ids.shape}")

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids)
    inference_time = time.time() - start_time

    print(f"⚡ Inference time: {inference_time:.3f}s")
    print(f"📊 Output logits shape: {outputs['logits'].shape}")
    print(f"🎯 Logits range: [{outputs['logits'].min():.3f}, {outputs['logits'].max():.3f}]")

    return outputs

def test_text_generation(model, tokenizer, device='cpu', max_length=50):
    """Test text generation"""
    print(f"\n🎨 Testing text generation...")

    model.to(device)
    model.eval()

    prompt = "The future of artificial intelligence is"
    print(f"📝 Prompt: '{prompt}'")

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)

    # Generate
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"🤖 Generated text: '{generated_text}'")
    print(f"⚡ Generation time: {generation_time:.3f}s")
    print(f"📏 Generated tokens: {len(generated_ids[0]) - len(input_ids[0])}")

    return generated_text

def extract_model_info(checkpoint_path, model, config):
    """Extract and display model information"""
    print(f"\n📋 Model Information")
    print("=" * 50)

    # Configuration
    print(f"🔧 Model Configuration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Num experts: {config.num_experts}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max position: {config.max_position_embeddings}")

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / (1024**2):.1f} MB (FP32)")

    # Checkpoint info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\n🎯 Training Information:")
    if 'global_steps' in checkpoint:
        print(f"  - Global steps: {checkpoint['global_steps']}")
    if 'global_samples' in checkpoint:
        print(f"  - Global samples: {checkpoint['global_samples']}")
    if 'ds_version' in checkpoint:
        print(f"  - DeepSpeed version: {checkpoint['ds_version']}")

def main():
    print("🚀 Testing LLM Model Checkpoint at Step 5000")
    print("=" * 60)

    # Paths
    checkpoint_path = "/project/code/outputs/checkpoint_step_5000/step_5000/mp_rank_00_model_states.pt"
    config_path = "/project/code/configs/gpu/small.yaml"

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")

    try:
        # Load configuration
        print(f"\n📄 Loading configuration from: {config_path}")
        config_dict = load_config(config_path)

        # Load model
        model, config = load_model_from_deepspeed_checkpoint(checkpoint_path, config_dict)

        # Initialize tokenizer
        print(f"\n🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Extract model information
        extract_model_info(checkpoint_path, model, config)

        # Test forward pass
        outputs = test_forward_pass(model, tokenizer, device)

        # Test text generation
        generated_text = test_text_generation(model, tokenizer, device)

        print(f"\n✅ All tests passed successfully!")
        print(f"🎉 Model is ready for inference and generation!")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)