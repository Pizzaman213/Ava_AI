#!/usr/bin/env python3

# Set MPS memory settings BEFORE importing torch
import os
if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' not in os.environ:
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit

"""
Streaming training script for LLM with proper memory management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import json
import glob
from transformers import AutoTokenizer
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.moe_transformer import MoEConfig, MoEForCausalLM

class StreamingDataset(IterableDataset):
    """Dataset that streams data from shards without loading everything into memory"""
    
    def __init__(self, shard_paths, tokenizer, max_length=512, max_samples_per_epoch=50000):
        self.shard_paths = sorted(shard_paths)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples_per_epoch = max_samples_per_epoch
        self.current_shard = None
        self.current_shard_idx = 0
        
        print(f"📊 Streaming from {len(self.shard_paths)} shards")
        print(f"🎯 Max samples per epoch: {max_samples_per_epoch}")
    
    def __iter__(self):
        """Stream samples from shards"""
        from datasets import load_from_disk
        
        samples_yielded = 0
        
        for shard_path in self.shard_paths:
            if samples_yielded >= self.max_samples_per_epoch:
                break
                
            try:
                # Load one shard at a time
                print(f"\n📁 Streaming from shard: {Path(shard_path).name}")
                shard = load_from_disk(shard_path)
                shard.set_format(type='torch', columns=['input_ids', 'attention_mask'])
                
                # Stream samples from this shard
                for idx in range(len(shard)):
                    if samples_yielded >= self.max_samples_per_epoch:
                        break
                    
                    item = shard[idx]
                    yield {
                        'input_ids': item['input_ids'][:self.max_length],
                        'attention_mask': item.get('attention_mask', torch.ones_like(item['input_ids']))[:self.max_length],
                        'labels': item['input_ids'][:self.max_length]
                    }
                    samples_yielded += 1
                
                # Clear the shard from memory
                del shard
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
            except Exception as e:
                print(f"⚠️ Error loading shard {shard_path}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mps/small.yaml')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max-samples', type=int, default=10000, help='Max samples per epoch')
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Memory settings for MPS
    if device.type == 'mps':
        print("🍎 MPS Memory optimizations:")
        print("  • PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (unlimited)")
        print(f"  • Batch size: {args.batch_size}")
        print(f"  • Gradient accumulation: {args.gradient_accumulation}")
        print(f"  • Effective batch: {args.batch_size * args.gradient_accumulation}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Find shard files
    shard_pattern = "data/pretraining/processed/train/shard_*"
    shard_paths = glob.glob(shard_pattern)
    
    if not shard_paths:
        print("❌ No shard files found!")
        return
    
    print(f"✅ Found {len(shard_paths)} shards")
    
    # Create streaming dataset
    dataset = StreamingDataset(
        shard_paths=shard_paths,
        tokenizer=tokenizer,
        max_length=512,
        max_samples_per_epoch=args.max_samples
    )
    
    # Create dataloader with NO workers for MPS
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # CRITICAL: No workers for MPS
        pin_memory=False  # Not needed for MPS
    )
    
    # Load config
    config = {}
    if Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    # Create model with smaller config for memory
    model_config = MoEConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.get('model', {}).get('hidden_size', 256),
        num_hidden_layers=config.get('model', {}).get('num_layers', 4),
        num_attention_heads=config.get('model', {}).get('num_attention_heads', 8),
        intermediate_size=config.get('model', {}).get('intermediate_size', 1024),
        num_experts=config.get('model', {}).get('num_experts', 2),
        num_experts_per_tok=config.get('model', {}).get('num_experts_per_tok', 1),
        max_position_embeddings=512
    )
    
    model = MoEForCausalLM(model_config)
    model.to(device)
    
    print(f"\n📊 Model stats:")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Layers: {model_config.num_hidden_layers}")
    print(f"  • Experts: {model_config.num_experts}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\n🏃 Starting training for {args.epochs} epochs...")
    print("="*60)
    
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        model.train()
        
        epoch_loss = 0
        batch_count = 0
        
        # Progress bar for samples
        progress = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}",
            total=args.max_samples // args.batch_size
        )
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress):
            try:
                # Move to device
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Periodic memory cleanup
                    if (batch_idx + 1) % 100 == 0 and device.type == 'mps':
                        torch.mps.empty_cache()
                
                # Track metrics
                epoch_loss += loss.item()
                batch_count += 1
                global_step += 1
                
                # Update progress
                avg_loss = epoch_loss / batch_count
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Save checkpoint
                if global_step % args.save_every == 0:
                    output_dir = Path("outputs") / f"streaming_run_{global_step}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': global_step,
                        'loss': avg_loss
                    }, output_dir / "checkpoint.pt")
                    
                    print(f"\n💾 Saved checkpoint at step {global_step}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM at batch {batch_idx}, clearing cache...")
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    optimizer.zero_grad()
                    gc.collect()
                    continue
                else:
                    raise e
        
        print(f"✅ Epoch {epoch+1} complete - Avg Loss: {epoch_loss/batch_count:.4f}")
    
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()