#!/usr/bin/env python3
"""
Simple MLX training script that uses existing configs
Optimized for Apple Silicon - 5-10x faster than PyTorch MPS
"""

import mlx.core as mx
import mlx.nn as mxnn
import mlx.optimizers as mxopt
from mlx.utils import tree_flatten
import numpy as np
import yaml
import json
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.moe_transformer import MoEConfig
from src.model.mlx_model import MLXTransformer, count_parameters

def load_data_files(data_path, tokenizer, max_samples=10000, max_length=256):
    """Load data from JSON files"""
    print(f"📂 Loading data from {data_path}...")
    
    samples = []
    data_files = list(Path(data_path).glob("*.json"))[:3]  # First 3 files
    
    if not data_files:
        print("  ⚠️ No data files found, using sample data")
        samples = ["Once upon a time", "The quick brown fox", "Machine learning is"] * 100
    else:
        for file_path in data_files:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                for item in file_data[:max_samples // len(data_files)]:
                    text = item.get('text', '') if isinstance(item, dict) else str(item)
                    if text:
                        samples.append(text)
    
    print(f"  ✅ Loaded {len(samples)} samples")
    
    # Tokenize
    print("  🔤 Tokenizing...")
    tokenized = []
    for text in tqdm(samples[:max_samples], desc="Tokenizing"):
        tokens = tokenizer(text, max_length=max_length, truncation=True, 
                          padding='max_length', return_tensors='np')
        tokenized.append(tokens['input_ids'][0])
    
    return np.array(tokenized)

def train_step(model, optimizer, batch):
    """Single training step"""
    def loss_fn(model):
        outputs = model(batch, labels=batch, training=True)
        return outputs['loss']
    
    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mps/small.yaml")
    parser.add_argument("--data-path", type=str, default="data/pretraining/raw/downloaded/train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="outputs/mlx_model")
    args = parser.parse_args()
    
    print("🚀 MLX Training Script")
    print("=" * 60)
    print(f"📄 Config: {args.config}")
    print(f"📁 Data: {args.data_path}")
    print(f"🎯 Batch size: {args.batch_size}")
    print(f"📊 Max samples: {args.max_samples}")
    print("=" * 60)
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_params = config_dict.get('model', {})
    training_params = config_dict.get('training', {})
    
    # Create config
    config = MoEConfig(
        vocab_size=model_params.get('vocab_size', 50257),
        hidden_size=model_params.get('hidden_size', 512),
        num_layers=model_params.get('num_layers', 6),
        num_attention_heads=model_params.get('num_attention_heads', 8),
        num_key_value_heads=model_params.get('num_key_value_heads', 4),
        intermediate_size=model_params.get('intermediate_size', 1792),
        max_position_embeddings=model_params.get('max_position_embeddings', 512),
        num_experts=model_params.get('num_experts', 3),
        num_experts_per_tok=model_params.get('num_experts_per_tok', 1),
        hidden_dropout=model_params.get('hidden_dropout', 0.1),
        attention_dropout=model_params.get('attention_dropout', 0.1),
        expert_dropout=model_params.get('expert_dropout', 0.1),
    )
    
    print("\n📊 Model Configuration:")
    print(f"  • Hidden: {config.hidden_size}")
    print(f"  • Layers: {config.num_layers}")
    print(f"  • Heads: {config.num_attention_heads}")
    print(f"  • Experts: {config.num_experts}")
    
    # Create model
    print("\n🤖 Creating model...")
    model = MLXTransformer(config)
    num_params = count_parameters(model)
    print(f"  ✅ Model created: {num_params:,} parameters")
    
    # Create optimizer
    learning_rate = training_params.get('learning_rate', 3e-5)
    # Convert to float if it's a string
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    optimizer = mxopt.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    print(f"  📈 Learning rate: {learning_rate}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    data = load_data_files(args.data_path, tokenizer, args.max_samples, 
                          config.max_position_embeddings)
    print(f"  📊 Data shape: {data.shape}")
    
    # Training
    print(f"\n🏃 Training for {args.epochs} epochs...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        
        # Create batches
        indices = np.random.permutation(len(data))
        data = data[indices]
        
        epoch_losses = []
        batches = range(0, len(data) - args.batch_size, args.batch_size)
        
        for step, i in enumerate(batches):
            batch = mx.array(data[i:i + args.batch_size])
            loss = train_step(model, optimizer, batch)
            
            loss_val = float(loss)
            epoch_losses.append(loss_val)
            
            avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
            
            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"    Step {step+1}: loss={loss_val:.4f}, avg={avg_loss:.4f}")
            
            # Evaluate periodically
            if (i // args.batch_size + 1) % 50 == 0:
                mx.eval(model.parameters())
        
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"  ✅ Epoch {epoch + 1}: Avg loss = {avg_epoch_loss:.4f}")
    
    train_time = time.time() - start_time
    samples_per_sec = (args.max_samples * args.epochs) / train_time
    
    print(f"\n⏱️  Training complete!")
    print(f"  • Time: {train_time:.2f} seconds")
    print(f"  • Speed: {samples_per_sec:.1f} samples/second")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving model to {output_dir}...")
    mx.save(str(output_dir / "model.npz"), dict(tree_flatten(model.parameters())))
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    print("✅ Model saved!")
    
    # Test generation
    print("\n🧪 Testing generation...")
    model.eval()
    
    test_prompt = "Once upon a time"
    tokens = tokenizer(test_prompt, return_tensors='np')['input_ids'][0]
    input_ids = mx.array(tokens).reshape(1, -1)
    
    print(f"  Prompt: '{test_prompt}'")
    
    for _ in range(30):
        outputs = model(input_ids, training=False)
        logits = outputs['logits']
        next_token = mx.argmax(logits[0, -1, :])
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)
    
    generated = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    print(f"  Generated: '{generated}'")
    
    print("\n" + "=" * 60)
    print("🎉 MLX training successful!")
    print(f"✨ MLX is optimized for Apple Silicon - much faster than PyTorch MPS!")
    print("=" * 60)

if __name__ == "__main__":
    main()