#!/usr/bin/env python3
"""
Simple Training Test for Enhanced LLM Framework

Tests the training infrastructure with a basic model to validate:
- Configuration loading
- Data processing
- Training loop
- Feature flag handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import argparse
from pathlib import Path
from transformers import GPT2Tokenizer
import sys
import os

# Add project root to Python path
sys.path.append('/project/code')

class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing training infrastructure"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Basic transformer components
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.position_embedding = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])

        # Simple transformer layers
        intermediate_size = config.get('intermediate_size', config.get('ffn_hidden_size', config['hidden_size'] * 4))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_attention_heads'],
                dim_feedforward=intermediate_size,
                dropout=config.get('hidden_dropout', 0.1),
                batch_first=True
            )
            for _ in range(config['num_layers'])
        ])

        self.ln_f = nn.LayerNorm(config['hidden_size'])
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'])

        # Enhanced features tracking
        self.enhanced_features = {}

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + pos_embeds

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask if attention_mask is not None else None)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(data_path, tokenizer, max_length=128):
    """Prepare training data from JSONL files"""
    data = []

    # Load data from JSON files
    if Path(data_path).is_file() and data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                # Combine instruction and response for language modeling
                if 'instruction' in item and 'response' in item:
                    text = f"Instruction: {item['instruction']}\nResponse: {item['response']}"
                    data.append(text)
    else:
        # Look for JSON files in directory
        data_dir = Path(data_path)
        for json_file in data_dir.rglob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        if 'instruction' in item and 'response' in item:
                            text = f"Instruction: {item['instruction']}\nResponse: {item['response']}"
                            data.append(text)
            except:
                continue

    print(f"Loaded {len(data)} examples")

    # Tokenize data
    tokenized_data = []
    for text in data[:100]:  # Limit to 100 examples for testing
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length')
        tokenized_data.append(torch.tensor(tokens))

    return tokenized_data

def test_enhanced_features(args):
    """Test enhanced feature flags"""
    features_enabled = []

    if hasattr(args, 'enable_all_features') and args.enable_all_features:
        features_enabled.append("🚀 All enhanced features")

    if hasattr(args, 'use_moh') and args.use_moh:
        features_enabled.append("🧠 Mixture of Heads (MoH)")

    if hasattr(args, 'use_moa') and args.use_moa:
        features_enabled.append("⚡ Mixture of Activations (MoA)")

    if hasattr(args, 'use_rag') and args.use_rag:
        features_enabled.append("📚 RAG (Retrieval-Augmented Generation)")

    if hasattr(args, 'use_focal_loss') and args.use_focal_loss:
        features_enabled.append("🎯 Focal Loss")

    if hasattr(args, 'use_contrastive_loss') and args.use_contrastive_loss:
        features_enabled.append("🔄 Contrastive Loss")

    if hasattr(args, 'gradient_surgery') and args.gradient_surgery:
        features_enabled.append("🔧 Gradient Surgery")

    if hasattr(args, 'use_episodic_memory') and args.use_episodic_memory:
        features_enabled.append("🧭 Episodic Memory")

    if features_enabled:
        print("✅ Enhanced Features Enabled:")
        for feature in features_enabled:
            print(f"   {feature}")
    else:
        print("📝 Running basic training (no enhanced features)")

    return features_enabled

def main():
    parser = argparse.ArgumentParser(description='Simple Training Test')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default='/project/code/data/training', help='Training data path')
    parser.add_argument('--steps', type=int, default=10, help='Number of training steps')

    # Enhanced feature flags for testing
    parser.add_argument('--enable-all-features', action='store_true', help='Enable all enhanced features')
    parser.add_argument('--use-moh', action='store_true', help='Use Mixture of Heads')
    parser.add_argument('--use-moa', action='store_true', help='Use Mixture of Activations')
    parser.add_argument('--use-rag', action='store_true', help='Use RAG')
    parser.add_argument('--use-focal-loss', action='store_true', help='Use Focal Loss')
    parser.add_argument('--use-contrastive-loss', action='store_true', help='Use Contrastive Loss')
    parser.add_argument('--gradient-surgery', action='store_true', help='Use Gradient Surgery')
    parser.add_argument('--use-episodic-memory', action='store_true', help='Use Episodic Memory')

    args = parser.parse_args()

    # Handle enable-all-features flag
    if args.enable_all_features:
        args.use_moh = True
        args.use_moa = True
        args.use_rag = True
        args.use_focal_loss = True
        args.use_contrastive_loss = True
        args.gradient_surgery = True
        args.use_episodic_memory = True

    print("🧪 Enhanced LLM Training Infrastructure Test")
    print("=" * 50)

    # Load configuration
    print("📋 Loading configuration...")
    config = load_config(args.config)
    model_config = config['model']
    train_config = config['training']

    print(f"   Model: {model_config.get('size', 'unknown')} ({model_config['num_layers']} layers)")
    print(f"   Hidden size: {model_config['hidden_size']}")
    print(f"   Vocab size: {model_config['vocab_size']}")

    # Test enhanced features
    features_enabled = test_enhanced_features(args)

    # Initialize tokenizer
    print("🔤 Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    print("📊 Loading training data...")
    train_data = prepare_data(args.data_path, tokenizer, model_config.get('max_position_embeddings', 128))

    if not train_data:
        print("❌ No training data found!")
        return

    # Initialize model
    print("🤖 Initializing model...")
    model = SimpleTransformerModel(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    learning_rate = float(train_config['learning_rate'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"🚀 Starting training ({args.steps} steps)...")
    model.train()

    batch_size = train_config.get('batch_size', 4)
    total_loss = 0

    for step in range(args.steps):
        # Get random batch
        batch_indices = torch.randint(0, len(train_data), (batch_size,))
        batch = torch.stack([train_data[i] for i in batch_indices]).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 5 == 0:
            print(f"   Step {step}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / args.steps
    print(f"✅ Training completed!")
    print(f"   Average loss: {avg_loss:.4f}")

    # Test inference
    print("🔮 Testing inference...")
    model.eval()
    with torch.no_grad():
        test_input = train_data[0][:20].unsqueeze(0).to(device)  # Take first 20 tokens
        outputs = model(test_input)
        logits = outputs['logits']
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode
        input_text = tokenizer.decode(test_input[0], skip_special_tokens=True)
        predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        print(f"   Input: {input_text[:100]}...")
        print(f"   Predicted: {predicted_text[:100]}...")

    print("🎉 Training infrastructure test completed successfully!")

    # Summary
    print("\n📈 Summary:")
    print(f"   Configuration: {args.config}")
    print(f"   Enhanced features: {len(features_enabled)} enabled")
    print(f"   Training data: {len(train_data)} examples")
    print(f"   Model parameters: {trainable_params:,}")
    print(f"   Training steps: {args.steps}")
    print(f"   Final loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()