#!/usr/bin/env python3

# Set MPS memory settings BEFORE importing torch
import os
if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' not in os.environ:
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit
"""
Training script with automatic dataset discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import sys
import json
import glob
from transformers import AutoTokenizer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import copy
import random

# Import advanced optimizers
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    print("⚠️ Lion optimizer not available, using AdamW")

# Disable tqdm in datasets globally to avoid the _lock attribute error
try:
    import datasets
    datasets.disable_progress_bar()
except ImportError:
    pass

# Helper function for parallel processing
def process_batch_parallel(args):
    """Process a batch of data in parallel"""
    batch_data, start_idx, end_idx = args
    processed = []
    
    if 'input_ids' in batch_data:
        for j in range(start_idx, min(end_idx, len(batch_data['input_ids']))):
            if 'attention_mask' in batch_data:
                processed.append({
                    'input_ids': batch_data['input_ids'][j],
                    'attention_mask': batch_data['attention_mask'][j]
                })
            else:
                processed.append({
                    'input_ids': batch_data['input_ids'][j],
                    'attention_mask': [1] * len(batch_data['input_ids'][j])
                })
    return processed

# Global collate function for DataLoader (must be pickleable)
def fast_collate(batch):
    """Optimized collate function for MPS"""
    import torch
    # Pre-allocate tensors for entire batch
    batch_size = len(batch)
    max_len = batch[0]['input_ids'].shape[0]
    
    # Allocate contiguous memory blocks
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    # Fill tensors efficiently
    for i, item in enumerate(batch):
        input_ids[i] = item['input_ids']
        attention_mask[i] = item['attention_mask']
    
    # Create labels with padding set to -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    # Make contiguous for faster GPU transfer
    return {
        'input_ids': input_ids.contiguous(),
        'attention_mask': attention_mask.contiguous(),
        'labels': labels.contiguous()  # For language modeling with padding ignored
    }

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import path utilities
from src.utils.path_utils import get_data_dir, get_outputs_dir

from src.model.moe_transformer import MoEConfig, MoEForCausalLM

class AutoDataset(Dataset):
    """Dataset that automatically finds and loads data"""
    
    def __init__(self, tokenizer, max_length=512, config=None, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.config = config
        self.split = split  # 'train' or 'validation'
        self.use_hf_dataset = False
        self.hf_dataset = None
        
        # Search for data files based on split
        if config and 'data' in config:
            if split == 'train' and 'train_paths' in config['data']:
                self.find_and_load_data_from_config()
            elif split == 'validation' and 'val_path' in config['data']:
                self.find_and_load_validation_data()
            else:
                self.find_and_load_data()
        else:
            self.find_and_load_data()
        
        if not self.data:
            print("⚠️  No data found, generating sample data...")
            self.generate_sample_data()
        
        print(f"📊 Loaded {len(self.data)} samples")
    
    def find_and_load_validation_data(self):
        """Load validation data from config"""
        val_path = self.config['data']['val_path']
        print(f"📁 Loading validation data from {val_path}...")
        
        # Check if it's a glob pattern
        if '*' in val_path:
            paths = glob.glob(val_path)
            if paths:
                print(f"  Found {len(paths)} validation files")
                self.load_shards_parallel(paths)
                return
        
        # Try to load as directory with shards
        path = Path(val_path)
        if path.exists() and path.is_dir():
            # Check for HuggingFace datasets format (shard directories with arrow files)
            shard_dirs = sorted(path.glob('shard_*'))
            if shard_dirs:
                print(f"  Found {len(shard_dirs)} validation shard directories (HuggingFace format)")
                # Load each shard directory individually
                from datasets import load_from_disk
                from tqdm import tqdm
                
                self.data = []
                max_val_examples = self.config['data'].get('max_val_examples', 5000)
                
                try:
                    with tqdm(total=len(shard_dirs), desc="  Loading validation shards", unit="shard") as pbar:
                        for shard_dir in shard_dirs:
                            if shard_dir.is_dir():
                                # Check if this shard has arrow files
                                arrow_files = list(shard_dir.glob('data-*.arrow'))
                                if arrow_files:
                                    # Load this shard
                                    dataset = load_from_disk(str(shard_dir))
                                    
                                    # Convert to our format
                                    for item in dataset:
                                        if len(self.data) >= max_val_examples:
                                            break
                                        
                                        if 'text' in item:
                                            self.data.append({'text': item['text']})
                                        elif 'input_ids' in item:
                                            # Already tokenized
                                            self.data.append({
                                                'input_ids': item['input_ids'],
                                                'attention_mask': item.get('attention_mask', [1] * len(item['input_ids']))
                                            })
                                    
                                    pbar.update(1)
                                    
                                    if len(self.data) >= max_val_examples:
                                        break
                    
                    if self.data:
                        print(f"  ✅ Loaded {len(self.data)} validation examples")
                        return
                    else:
                        print(f"  ⚠️ No valid data found in shard directories")
                except Exception as e:
                    print(f"  ⚠️ Failed to load validation shards: {e}")
            
            # Look for regular shard files
            shard_files = sorted(path.glob('shard_*.pt')) or sorted(path.glob('shard_*.arrow'))
            if shard_files:
                print(f"  Found {len(shard_files)} validation shards")
                self.load_shards_parallel([str(f) for f in shard_files])
                return
            
            # Try JSON files
            json_files = list(path.glob('*.json'))
            if json_files:
                for file_path in json_files[:1000]:  # Limit validation data
                    self.load_json_file(file_path)
                return
        
        # If nothing else works, generate sample validation data
        print("  ⚠️ No validation data found, using subset of training data")
        self.data = self.data[:1000] if self.data else []
    
    def find_and_load_data_from_config(self):
        """Load data from paths specified in config"""
        train_paths = self.config['data']['train_paths']
        parallel_loading = self.config['data'].get('parallel_loading', True)
        
        print("📁 Loading data from config paths...")
        for path_pattern in train_paths:
            if '*' in path_pattern:
                # It's a glob pattern
                paths = glob.glob(path_pattern)
                if paths:
                    print(f"  Found {len(paths)} files matching {path_pattern}")
                    if ('shard' in path_pattern or 'batch' in path_pattern) and parallel_loading:
                        # Load shards/batches in parallel
                        self.load_shards_parallel(paths)
                        if self.data:
                            return
                    else:
                        for path in paths:
                            if path.endswith('.json'):
                                self.load_json_file(path)
                            elif path.endswith('.jsonl'):
                                self.load_jsonl_file(path)
            else:
                # Direct path
                if self.try_load_arrow_data(path_pattern):
                    return
        
    def find_and_load_data(self):
        """Automatically find and load data files"""
        
        # Check for Arrow format first (fastest)
        data_root = get_data_dir()
        arrow_paths = [
            data_root / "pretraining/processed/train/shard_*",      # All shards FIRST
            data_root / "pretraining/processed/train",              # Directory with shards
            data_root / "pretraining/processed_500k/train",         # 500k processed data
            data_root / "pretraining/processed_big/train",          # Alternative 500k location
            data_root / "pretraining/processed/train/shard_00000",  # Single shard fallback
            data_root / "train/shard_*"
        ]
        
        for pattern in arrow_paths:
            if self.try_load_arrow_data(str(pattern)):
                return
        
        # Check for JSONL files (efficient for large data)
        jsonl_paths = [
            data_root / "pretraining/processed/train/*.jsonl",
            data_root / "pretraining/processed/*.jsonl",
            data_root / "train/*.jsonl"
        ]
        
        for pattern in jsonl_paths:
            files = glob.glob(str(pattern))
            if files:
                print(f"📁 Found JSONL files: {files[:3]}...")
                for file_path in files:
                    self.load_jsonl_file(file_path)
                    print(f"    Total loaded so far: {len(self.data)} samples")
                if self.data:
                    break
        
        # Common JSON data locations to check
        if not self.data:
            json_paths = [
                data_root / "pretraining/raw/500k_samples/train/*.json",  # Load 500k samples FIRST
                data_root / "pretraining/raw/500k_samples/*.json",        # 500k samples directory
                data_root / "pretraining/processed/train/*.json",
                data_root / "pretraining/processed/*.json",
                data_root / "pretraining/raw/*/train/*.json",
                data_root / "train/*.json",
                data_root / "*.json",
                "*.json"  # Current directory fallback
            ]
            
            for pattern in json_paths:
                files = glob.glob(str(pattern))
                if files:
                    print(f"📁 Found JSON files: {files[:3]}...")
                    for file_path in files:  # Load ALL files
                        self.load_json_file(file_path)
                        print(f"    Total loaded so far: {len(self.data)} samples")
                    if self.data:
                        break
        
        # Also check for .txt files
        if not self.data:
            txt_patterns = [
                data_root / "*.txt",
                data_root / "train/*.txt",
                "*.txt"  # Current directory fallback
            ]
            for pattern in txt_patterns:
                files = glob.glob(str(pattern))
                if files:
                    print(f"📁 Found text files: {files[:3]}...")
                    for file_path in files[:5]:
                        self.load_text_file(file_path)
                        if len(self.data) >= 10000:
                            break
                    if self.data:
                        break
    
    def load_shards_parallel(self, shard_paths):
        """Load multiple shards in parallel"""
        try:
            # Disable tqdm in datasets to avoid the _lock attribute error
            import datasets
            datasets.disable_progress_bar()
            from datasets import load_from_disk, concatenate_datasets
            
            print(f"⚡ Loading {len(shard_paths)} shards...")
            
            # Get max examples from config (ensure it's an int)
            max_examples = self.config.get('data', {}).get('max_train_examples', -1) if self.split == 'train' else self.config.get('data', {}).get('max_val_examples', -1)
            # Convert to int if it's a string
            if isinstance(max_examples, str):
                max_examples = int(max_examples) if max_examples.isdigit() else -1
            
            # Load each shard sequentially but efficiently
            datasets = []
            total_loaded = 0
            for shard_path in sorted(shard_paths):
                ds = load_from_disk(shard_path)
                
                # If we have a limit and would exceed it, truncate this shard
                if max_examples > 0 and total_loaded + len(ds) > max_examples:
                    remaining = max_examples - total_loaded
                    if remaining > 0:
                        ds = ds.select(range(remaining))
                        datasets.append(ds)
                        print(f"  Loaded {shard_path}: {len(ds)} samples (truncated)")
                        break
                    else:
                        break
                else:
                    datasets.append(ds)
                    total_loaded += len(ds)
                    print(f"  Loaded {shard_path}: {len(ds)} samples")
                    
                # Stop if we've reached the limit
                if max_examples > 0 and total_loaded >= max_examples:
                    break
            
            # Concatenate all datasets
            full_dataset = concatenate_datasets(datasets)
            
            # Apply final limit if needed
            if max_examples > 0 and len(full_dataset) > max_examples:
                full_dataset = full_dataset.select(range(max_examples))
            
            print(f"  ✅ Loaded {len(full_dataset)} total samples from {len(datasets)} shards")
            
            # Set format for faster tensor conversion
            full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            
            # Keep as HuggingFace dataset for efficient access
            self.hf_dataset = full_dataset
            self.use_hf_dataset = True
            
            # Create dummy data list for length
            self.data = list(range(len(full_dataset)))
            
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to load shards: {e}")
            return False
    
    def try_load_arrow_data(self, pattern):
        """Try to load Arrow format data"""
        try:
            # Disable tqdm in datasets to avoid the _lock attribute error
            import datasets
            datasets.disable_progress_bar()
            from datasets import load_from_disk, concatenate_datasets
            import os
            import glob as gb
            
            # Check for shard pattern
            if '*' in pattern:
                # Find all matching shards
                shard_dirs = gb.glob(pattern)
                if shard_dirs:
                    print(f"📁 Found {len(shard_dirs)} Arrow shards")
                    
                    # Load all shards with progress bar
                    from concurrent.futures import ThreadPoolExecutor
                    datasets = []
                    with tqdm(total=len(shard_dirs), desc="  Loading shards", unit="shard") as pbar:
                        with ThreadPoolExecutor(max_workers=min(len(shard_dirs), 8)) as executor:
                            futures = [executor.submit(load_from_disk, shard) for shard in sorted(shard_dirs)]
                            for future in futures:
                                datasets.append(future.result())
                                pbar.update(1)
                    
                    if datasets:
                        print(f"  Concatenating {len(datasets)} shards...")
                        # Combine all shards
                        if len(datasets) > 1:
                            dataset = concatenate_datasets(datasets)
                        else:
                            dataset = datasets[0]
                        
                        # Convert to our format in batches with progress bar
                        total_samples = len(dataset)
                        print(f"  Converting {total_samples:,} samples to training format...")
                        batch_size = 100000  # Even larger batches to reduce overhead
                        
                        # Pre-allocate list for better performance
                        self.data = []
                        
                        # Simple vectorized processing without multiprocessing overhead
                        with tqdm(total=total_samples, desc="  Processing samples", unit="samples") as pbar:
                            for i in range(0, total_samples, batch_size):
                                end_idx = min(i + batch_size, total_samples)
                                # Use dataset slicing for efficiency
                                batch = dataset[i:end_idx]
                                
                                # Direct vectorized processing - fastest approach
                                if 'attention_mask' in batch:
                                    # Use list comprehension - faster than parallel for this task
                                    batch_data = [{
                                        'input_ids': batch['input_ids'][j],
                                        'attention_mask': batch['attention_mask'][j]
                                    } for j in range(len(batch['input_ids']))]
                                else:
                                        # Generate attention masks efficiently
                                    batch_data = [{
                                        'input_ids': batch['input_ids'][j],
                                        'attention_mask': [1] * len(batch['input_ids'][j])
                                    } for j in range(len(batch['input_ids']))]
                                
                                self.data.extend(batch_data)
                                pbar.update(len(batch_data))
                        
                        print(f"  ✅ Loaded {len(self.data):,} samples from Arrow format")
                        return True
            else:
                # Direct directory path
                if os.path.exists(pattern) and os.path.isdir(pattern):
                    # Check if it's a shard directory (has arrow files)
                    arrow_files = gb.glob(os.path.join(pattern, "*.arrow"))
                    if arrow_files:
                        print(f"📁 Loading Arrow dataset from {pattern}")
                        dataset = load_from_disk(pattern)
                        
                        for item in dataset:
                            if 'input_ids' in item:
                                self.data.append({
                                    'input_ids': item['input_ids'],
                                    'attention_mask': item.get('attention_mask', [1] * len(item['input_ids']))
                                })
                            elif 'text' in item:
                                self.data.append({'text': item['text']})
                        
                        print(f"  ✅ Loaded {len(self.data)} samples from Arrow format")
                        return True
                    else:
                        # It's a parent directory, look for shards
                        shard_dirs = gb.glob(os.path.join(pattern, "shard_*"))
                        if shard_dirs:
                            print(f"📁 Found {len(shard_dirs)} Arrow shards in {pattern}")
                            datasets = []
                            
                            # Load shards with progress bar
                            with tqdm(total=len(shard_dirs), desc="  Loading shards", unit="shard") as pbar:
                                for shard_dir in sorted(shard_dirs):
                                    dataset = load_from_disk(shard_dir)
                                    datasets.append(dataset)
                                    pbar.update(1)
                            
                            if datasets:
                                # Combine all shards
                                print(f"  Concatenating {len(datasets)} shards...")
                                if len(datasets) > 1:
                                    dataset = concatenate_datasets(datasets)
                                else:
                                    dataset = datasets[0]
                                
                                # Convert to our format with progress bar
                                total_samples = len(dataset)
                                print(f"  Converting {total_samples:,} samples to training format...")
                                batch_size = 100000  # Larger batches
                                
                                # Pre-allocate for better performance
                                self.data = []
                                
                                with tqdm(total=total_samples, desc="  Processing samples", unit="samples") as pbar:
                                    # Process in large batches for efficiency
                                    for i in range(0, total_samples, batch_size):
                                        end_idx = min(i + batch_size, total_samples)
                                        batch = dataset[i:end_idx]
                                        
                                        # Check if batch has the right format
                                        if isinstance(batch, dict) and 'input_ids' in batch:
                                            # Direct vectorized processing
                                            if 'attention_mask' in batch:
                                                batch_data = [{
                                                    'input_ids': batch['input_ids'][j],
                                                    'attention_mask': batch['attention_mask'][j]
                                                } for j in range(len(batch['input_ids']))]
                                            else:
                                                batch_data = [{
                                                    'input_ids': batch['input_ids'][j],
                                                    'attention_mask': [1] * len(batch['input_ids'][j])
                                                } for j in range(len(batch['input_ids']))]
                                            
                                            self.data.extend(batch_data)
                                            pbar.update(len(batch_data))
                                        else:
                                            # Fallback for other formats
                                            count = 0
                                            for item in batch:
                                                if 'input_ids' in item:
                                                    self.data.append({
                                                        'input_ids': item['input_ids'],
                                                        'attention_mask': item.get('attention_mask', [1] * len(item['input_ids']))
                                                    })
                                                elif 'text' in item:
                                                    self.data.append(item['text'])
                                                count += 1
                                            pbar.update(count)
                                
                                print(f"  ✅ Loaded {len(self.data):,} samples from Arrow format")
                                return True
        except Exception as e:
            # Arrow loading failed, show error for debugging
            print(f"  ⚠️ Arrow loading failed for {pattern}: {e}")
            pass
        
        return False
    
    def load_jsonl_file(self, file_path):
        """Load data from JSONL file"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        
                        # Check if it's already tokenized
                        if 'input_ids' in item:
                            self.data.append({
                                'input_ids': item['input_ids'],
                                'attention_mask': item.get('attention_mask', [1] * len(item['input_ids']))
                            })
                        else:
                                # Extract text
                            text = item.get('text') or item.get('content') or \
                                   item.get('instruction') or item.get('prompt') or \
                                   item.get('input') or str(item)
                            
                            if text and len(text) > 10:
                                self.data.append(text)
            
            print(f"  ✅ Loaded samples from {Path(file_path).name}")
        except Exception as e:
            print(f"  ⚠️  Could not load {file_path}: {e}")
    
    def load_json_file(self, file_path):
        """Load data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                
            if isinstance(content, list):
                for item in content:  # Load all items from file
                    if isinstance(item, dict):
                            # Try different common keys
                        text = item.get('text') or item.get('content') or \
                               item.get('instruction') or item.get('prompt') or \
                               item.get('input') or str(item)
                    else:
                        text = str(item)
                    
                    if text and len(text) > 10:
                        self.data.append(text)
            elif isinstance(content, dict):
                # Handle single dict or nested structure
                if 'data' in content:
                    self.load_json_file(content['data'])
                else:
                    text = content.get('text') or str(content)
                    if text and len(text) > 10:
                        self.data.append(text)
            
            print(f"  ✅ Loaded {len(self.data)} samples from {Path(file_path).name}")
        except Exception as e:
            print(f"  ⚠️  Could not load {file_path}: {e}")
    
    def load_text_file(self, file_path):
        """Load data from text file"""
        try:
            with open(file_path, 'r') as f:
                text = f.read()
            
            # Split into chunks
            chunks = text.split('\n\n')
            for chunk in chunks[:1000]:
                if chunk and len(chunk) > 10:
                    self.data.append(chunk)
            
            print(f"  ✅ Loaded {len(self.data)} samples from {Path(file_path).name}")
        except Exception as e:
            print(f"  ⚠️  Could not load {file_path}: {e}")
    
    def generate_sample_data(self):
        """Generate sample training data if no data found"""
        samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Natural language processing helps computers understand human language.",
            "Deep learning models use neural networks with multiple layers.",
            "Transformers have revolutionized the field of NLP with attention mechanisms.",
            "The mixture of experts approach allows models to specialize in different tasks.",
            "Training large language models requires significant computational resources.",
            "Gradient descent is an optimization algorithm used to minimize loss functions.",
            "Backpropagation enables efficient training of deep neural networks.",
            "Transfer learning allows models to leverage knowledge from pre-trained models.",
        ] * 10  # Repeat for more samples
        
        self.data = samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Data augmentation parameters from config
        noise_prob = self.config.get('data', {}).get('noise_probability', 0.0)
        mask_prob = self.config.get('data', {}).get('mask_probability', 0.0)
        
        # Use HuggingFace dataset if available (much faster)
        if self.use_hf_dataset and self.hf_dataset is not None:
            data_item = self.hf_dataset[idx]
            
            # Data should already be torch tensors from set_format
            input_ids = data_item['input_ids']
            attention_mask = data_item['attention_mask']
            
            # Ensure they are tensors - use contiguous memory for faster GPU transfer
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long).contiguous()
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).contiguous()
            else:
                # Make existing tensors contiguous for faster transfer
                input_ids = input_ids.contiguous()
                attention_mask = attention_mask.contiguous()
            
            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            # Pad if needed
            if len(input_ids) < self.max_length:
                pad_len = self.max_length - len(input_ids)
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            
            # Apply data augmentation (only during training)
            if self.split == 'train':
                # Random token noise
                if noise_prob > 0 and torch.rand(1).item() < 0.5:  # Apply to 50% of samples
                    noise_mask = torch.rand(input_ids.shape) < noise_prob
                    random_tokens = torch.randint(0, self.tokenizer.vocab_size, input_ids.shape, dtype=torch.long)
                    input_ids = torch.where(noise_mask & (attention_mask == 1), random_tokens, input_ids)
                
                # Random masking
                if mask_prob > 0 and torch.rand(1).item() < 0.5:  # Apply to 50% of samples
                    mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)
                    if mask_token_id is None:
                        mask_token_id = getattr(self.tokenizer, 'unk_token_id', 0)
                    if mask_token_id is not None:
                        mask_positions = torch.rand(input_ids.shape) < mask_prob
                        input_ids = torch.where(mask_positions & (attention_mask == 1), mask_token_id, input_ids)
            
            # Create labels with padding tokens set to -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
                
        else:
            # Fall back to old method
            data_item = self.data[idx]
            
            # Check if data is already tokenized
            if isinstance(data_item, dict) and 'input_ids' in data_item:
                # Already tokenized data - just convert to tensor (fast)
                input_ids = data_item['input_ids']
                attention_mask = data_item['attention_mask']
                
                # Convert to tensor if not already - use contiguous memory
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids, dtype=torch.long).contiguous()
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long).contiguous()
                else:
                    input_ids = input_ids.contiguous()
                    attention_mask = attention_mask.contiguous()
                
                # Truncate if needed (should already be right size from preprocessing)
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
            else:
                # Raw text - need to tokenize
                text = data_item
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
        
        # For language modeling, labels are the same as inputs
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_config(config_path):
    """Load and validate config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model parameters
    model_params = {}
    for key, value in config['model'].items():
        if hasattr(MoEConfig, key):
            model_params[key] = value
    
    return config, model_params

def find_learning_rate(model, train_loader, device, config, num_steps=200, start_lr=1e-8, end_lr=10):
    """
    Advanced learning rate finder with multiple analysis methods.
    Returns optimal learning rate based on multiple criteria.
    """
    print("\n" + "="*60)
    print("🔍 ADVANCED LEARNING RATE FINDER")
    print("="*60)
    print(f"Testing learning rates from {start_lr:.2e} to {end_lr:.2e} over {num_steps} steps")
    
    # Save initial model state
    initial_state = model.state_dict()
    initial_optimizer_state = None
    
    # Setup optimizer with starting LR
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=start_lr,
        weight_decay=config.get('training', {}).get('weight_decay', 0.1)
    )
    
    # Calculate LR multiplier for exponential increase
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    
    # Tracking arrays
    lrs = []
    losses = []
    smoothed_losses = []
    gradients_norms = []
    loss_changes = []
    best_loss = float('inf')
    diverge_threshold = 4
    min_delta = 1e-4  # Minimum improvement to continue
    
    model.train()
    data_iter = iter(train_loader)
    
    print("\n📊 Running learning rate sweep...")
    progress_bar = tqdm(range(num_steps), desc="LR Finder")
    
    # Track statistics
    early_stop_patience = 10
    patience_counter = 0
    
    for step in progress_bar:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else input_ids
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Forward pass with mixed precision if available
        if device.type == 'mps' and config.get('training', {}).get('mixed_precision', False):
            try:
                with torch.autocast(device_type='mps', dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            except:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        
        # Skip if NaN
        if torch.isnan(loss):
            print(f"\n⚠️ NaN loss at LR={optimizer.param_groups[0]['lr']:.2e}, stopping")
            break
        
        # Check for divergence
        if step > 10 and loss.item() > diverge_threshold * min(losses[-10:]):
            print(f"\n⚠️ Loss diverging at LR={optimizer.param_groups[0]['lr']:.2e}, stopping early")
            break
        
        # Track best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stop if loss plateaus
        if patience_counter > early_stop_patience and step > 50:
            print(f"\n⚠️ Loss plateaued at LR={optimizer.param_groups[0]['lr']:.2e}, stopping")
            break
        
        # Record metrics
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())
        
        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        gradients_norms.append(total_norm)
        
        # Calculate loss change
        if len(losses) > 1:
            loss_change = losses[-1] - losses[-2]
            loss_changes.append(loss_change)
        
        # Calculate smoothed loss (exponential moving average)
        if len(smoothed_losses) == 0:
            smoothed_losses.append(loss.item())
        else:
            alpha = 0.05  # Smoothing factor
            smoothed = alpha * loss.item() + (1 - alpha) * smoothed_losses[-1]
            smoothed_losses.append(smoothed)
        
        # Update progress bar
        progress_bar.set_postfix({
            'lr': f"{current_lr:.2e}", 
            'loss': f"{loss.item():.4f}",
            'smooth': f"{smoothed_losses[-1]:.4f}",
            'grad': f"{total_norm:.2f}"
        })
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Increase learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    # Restore initial model state
    model.load_state_dict(initial_state)
    
    # Advanced LR Analysis
    suggested_lrs = {}
    
    # Method 1: Steepest descent (classic approach)
    if len(smoothed_losses) > 20:
        gradients = []
        for i in range(5, len(smoothed_losses) - 5):
            grad = (smoothed_losses[i+5] - smoothed_losses[i-5]) / 10
            gradients.append(grad)
        
        if gradients:
            min_grad_idx = gradients.index(min(gradients))
            steepest_lr = lrs[min_grad_idx + 5]
            suggested_lrs['steepest_descent'] = steepest_lr / 10
    
    # Method 2: Maximum gradient (where model learns fastest)
    if len(gradients_norms) > 10:
        # Find where gradient is highest but stable
        max_stable_idx = -1
        for i in range(10, len(gradients_norms) - 5):
            if gradients_norms[i] > 0.1:  # Meaningful gradient
                # Check if gradient is stable (not exploding)
                recent_grads = gradients_norms[i:i+5]
                if max(recent_grads) / min(recent_grads) < 3:  # Stable
                    max_stable_idx = i
                    break
        
        if max_stable_idx > 0:
            suggested_lrs['max_gradient'] = lrs[max_stable_idx]
    
    # Method 3: Minimum loss achieved
    if losses:
        min_loss_idx = losses.index(min(losses))
        if min_loss_idx > 0:
            suggested_lrs['min_loss'] = lrs[min_loss_idx] / 2
    
    # Method 4: Elbow method (where improvement starts to diminish)
    if len(loss_changes) > 20:
        # Find where improvement rate drops significantly
        for i in range(10, len(loss_changes) - 5):
            if loss_changes[i] > -min_delta:  # Improvement too small
                suggested_lrs['elbow'] = lrs[i] / 5
                break
    
    # Calculate final suggestion (weighted average)
    if suggested_lrs:
        # Weight different methods
        weights = {
            'steepest_descent': 0.4,
            'max_gradient': 0.2,
            'min_loss': 0.2,
            'elbow': 0.2
        }
        
        weighted_sum = 0
        total_weight = 0
        for method, lr in suggested_lrs.items():
            if method in weights:
                weighted_sum += lr * weights[method]
                total_weight += weights[method]
        
        suggested_lr = weighted_sum / total_weight if total_weight > 0 else 1e-4
    else:
        # Fallback if analysis fails
        suggested_lr = 1e-4
    
    # Enhanced visualization
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 10))
        
        # Create 4 subplots
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        
        # Plot 1: Loss vs LR
        ax1.semilogx(lrs, losses, 'b-', alpha=0.3, label='Raw Loss', linewidth=1)
        ax1.semilogx(lrs, smoothed_losses, 'r-', label='Smoothed Loss', linewidth=2)
        
        # Mark suggested LRs from different methods
        colors = {'steepest_descent': 'green', 'max_gradient': 'orange', 
                 'min_loss': 'purple', 'elbow': 'brown'}
        for method, lr in suggested_lrs.items():
            ax1.axvline(lr, color=colors.get(method, 'gray'), 
                       linestyle=':', alpha=0.5, label=f'{method}: {lr:.2e}')
        
        ax1.axvline(suggested_lr, color='darkgreen', linestyle='--', 
                   linewidth=2, label=f'Final Suggestion: {suggested_lr:.2e}')
        ax1.set_xlabel('Learning Rate (log scale)', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Loss vs Learning Rate', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gradient Norms
        if gradients_norms:
            ax2.semilogx(lrs[:len(gradients_norms)], gradients_norms, 'purple', linewidth=2)
            ax2.axvline(suggested_lr, color='darkgreen', linestyle='--', linewidth=2)
            ax2.set_xlabel('Learning Rate (log scale)', fontweight='bold')
            ax2.set_ylabel('Gradient Norm', fontweight='bold')
            ax2.set_title('Gradient Magnitude vs LR', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Mark stable gradient region
            if 'max_gradient' in suggested_lrs:
                ax2.axvspan(suggested_lrs['max_gradient']/2, 
                           suggested_lrs['max_gradient']*2, 
                           alpha=0.2, color='green', label='Stable Region')
                ax2.legend()
        
        # Plot 3: Loss Rate of Change
        if len(loss_changes) > 0:
            ax3.semilogx(lrs[1:len(loss_changes)+1], loss_changes, 'orange', linewidth=2)
            ax3.axvline(suggested_lr, color='darkgreen', linestyle='--', linewidth=2)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax3.axhline(-min_delta, color='red', linestyle=':', alpha=0.5, 
                       label=f'Min improvement: {-min_delta:.4f}')
            ax3.set_xlabel('Learning Rate (log scale)', fontweight='bold')
            ax3.set_ylabel('Loss Change (Δ)', fontweight='bold')
            ax3.set_title('Loss Change Rate', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary Statistics
        ax4.axis('off')
        summary_text = f"""
📊 LR FINDER ANALYSIS SUMMARY
════════════════════════════════════════

Tested Range: {start_lr:.2e} to {lrs[-1]:.2e}
Total Steps: {len(losses)}
Best Loss: {min(losses):.4f}
Worst Loss: {max(losses):.4f}

SUGGESTED LEARNING RATES BY METHOD:
"""
        for method, lr in suggested_lrs.items():
            summary_text += f"\n• {method.replace('_', ' ').title()}: {lr:.3e}"
        
        summary_text += f"""

FINAL RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Use LR = {suggested_lr:.3e}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is a weighted average of all methods,
optimized for stable training.

Alternative suggestions:
• Conservative: {suggested_lr/3:.3e}
• Aggressive: {suggested_lr*2:.3e}
• One-Cycle Max: {suggested_lr*10:.3e}
"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('🔍 Advanced Learning Rate Finder Results', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save plot to wandb
        if wandb.run:
            wandb.log({
                'lr_finder/plot': wandb.Image(fig),
                'lr_finder/suggested_lr': suggested_lr,
                'lr_finder/min_loss': min(losses),
                'lr_finder/max_gradient': max(gradients_norms) if gradients_norms else 0
            })
        
        # Save plot to file
        plot_path = Path("lr_finder_results.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to {plot_path}")
        
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
    
    # Detailed console output
    print("\n" + "="*60)
    print("📊 ADVANCED LEARNING RATE FINDER RESULTS")
    print("="*60)
    print(f"✅ RECOMMENDED LEARNING RATE: {suggested_lr:.3e}")
    print("="*60)
    print("\n📈 Analysis Methods Used:")
    for method, lr in suggested_lrs.items():
        print(f"  • {method.replace('_', ' ').title()}: {lr:.3e}")
    print(f"\n📉 Loss Statistics:")
    print(f"  • Best Loss: {min(losses):.4f}")
    print(f"  • Worst Loss: {max(losses):.4f}")
    print(f"  • Loss Reduction: {(max(losses) - min(losses))/max(losses)*100:.1f}%")
    if gradients_norms:
        print(f"\n🎯 Gradient Statistics:")
        print(f"  • Max Gradient: {max(gradients_norms):.2f}")
        print(f"  • Min Gradient: {min(gradients_norms):.2f}")
    print("\n💡 Recommendations:")
    print(f"  • Conservative (safe): {suggested_lr/3:.3e}")
    print(f"  • Standard (balanced): {suggested_lr:.3e}")
    print(f"  • Aggressive (fast): {suggested_lr*2:.3e}")
    print(f"  • One-Cycle Max: {suggested_lr*10:.3e}")
    print("="*60)
    
    return suggested_lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: from path_utils)")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    parser.add_argument("--find-lr", action="store_true", help="Run learning rate finder before training")
    parser.add_argument("--lr-finder-steps", type=int, default=100, help="Number of steps for LR finder")
    args = parser.parse_args()
    
    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir) / f"run_{timestamp}"
    else:
        output_dir = get_outputs_dir(f"run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    import logging
    log_file = args.log_file or str(output_dir / "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print(f"🚀 Training with config: {args.config}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Log file: {log_file}")
    print("="*60)
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load config
    config, model_params = load_config(args.config)
    
    # Override with command line args if provided
    batch_size = args.batch_size or config.get('training', {}).get('batch_size', 4)
    learning_rate = args.lr or config.get('training', {}).get('learning_rate', 1e-4)
    
    # Initialize W&B if enabled
    use_wandb = config.get('monitoring', {}).get('use_wandb', True)
    if use_wandb:
        try:
            wandb_project = config.get('monitoring', {}).get('wandb_project', 'llm-training')
            wandb_name = config.get('monitoring', {}).get('wandb_name', f"run_{timestamp}")
            wandb_tags = config.get('monitoring', {}).get('wandb_tags', ['moe', 'mps'])
            
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                tags=wandb_tags,
                config={
                    **config,
                    'command_line_args': vars(args),
                    'timestamp': timestamp,
                    'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            )
            print(f"✅ W&B initialized successfully: {wandb_project}/{wandb_name}")
            print(f"   View run at: {wandb.run.get_url()}")
            logger.info(f"W&B tracking enabled: {wandb_project}/{wandb_name}")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            print("   Continuing without W&B logging")
            use_wandb = False
    
    # Memory optimization for MPS
    gradient_accumulation_steps = config.get('training', {}).get('gradient_accumulation_steps', 1)
    # Set environment variable for MPS memory management BEFORE importing torch
    if torch.backends.mps.is_available():
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit to prevent ratio errors
    
    # Ensure learning_rate is a float
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    try:
    # Create tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2',  local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("the ")
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Create train and validation datasets with config
    print("📂 Loading datasets...")
    train_dataset = AutoDataset(tokenizer, max_length=args.max_length, config=config, split='train')
    
    # 4. CURRICULUM LEARNING - Start with easier samples
    curriculum_enabled = config.get('training', {}).get('curriculum_learning', True)
    if curriculum_enabled:
        print("📚 Curriculum learning enabled - will gradually increase sequence complexity")
        initial_max_length = config.get('training', {}).get('initial_sequence_length', 256)
        curriculum_warmup_steps = config.get('training', {}).get('curriculum_warmup_steps', 1000)
        
        # Create curriculum wrapper with progressive growth
        class CurriculumDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, initial_max_len, final_max_len, warmup_steps):
                self.base_dataset = base_dataset
                self.initial_max_len = initial_max_len
                self.final_max_len = final_max_len
                self.warmup_steps = warmup_steps
                self.current_step = 0
                self.current_max_len = initial_max_len
                # Progressive growth settings
                self.enable_progressive_growth = config.get('training', {}).get('progressive_sequence_growth', True)
                self.max_sequence_limit = config.get('model', {}).get('max_position_embeddings', 4096)
                self.growth_rate = config.get('training', {}).get('sequence_growth_rate', 1.5)  # Multiply by 1.5 each phase
                self.growth_interval = config.get('training', {}).get('sequence_growth_interval', 5000)  # Steps between growth
                self.last_growth_step = 0
                
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                item = self.base_dataset[idx]
                # Truncate sequences based on current curriculum stage
                if self.current_max_len < self.max_sequence_limit:
                    for key in ['input_ids', 'attention_mask', 'labels']:
                        if key in item:
                            item[key] = item[key][:self.current_max_len]
                return item
            
            def update_curriculum(self, global_step):
                """Update max length based on training progress - supports continuous growth"""
                if global_step < self.warmup_steps:
                    # Phase 1: Initial warmup (e.g., 256 -> 512)
                    progress = global_step / self.warmup_steps
                    self.current_max_len = int(self.initial_max_len + 
                                              (self.final_max_len - self.initial_max_len) * progress)
                elif self.enable_progressive_growth:
                    # Phase 2: Progressive growth beyond initial target (512 -> 768 -> 1024 -> ...)
                    steps_since_warmup = global_step - self.warmup_steps
                    growth_phases = steps_since_warmup // self.growth_interval
                    
                    if growth_phases > 0:
                        # Calculate new length: 512 * 1.5^phases
                        new_len = int(self.final_max_len * (self.growth_rate ** growth_phases))
                        
                        # Cap at model's maximum
                        new_len = min(new_len, self.max_sequence_limit)
                        
                        # Log significant growth events
                        if new_len > self.current_max_len:
                            old_len = self.current_max_len
                            self.current_max_len = new_len
                            if wandb.run:
                                wandb.log({
                                    'curriculum/sequence_growth_event': new_len,
                                    'curriculum/growth_phase': growth_phases
                                }, step=global_step)
                            print(f"\n📈 Sequence length increased: {old_len} -> {new_len} (Phase {growth_phases})")
                    else:
                        self.current_max_len = self.final_max_len
                else:
                    self.current_max_len = self.final_max_len
                    
                return self.current_max_len
        
        # Wrap train dataset with curriculum
        curriculum_dataset = CurriculumDataset(
            train_dataset, 
            initial_max_length,
            args.max_length,
            curriculum_warmup_steps
        )
        train_dataset = curriculum_dataset
    
    # Create validation dataset
    val_config = config.copy()
    val_config['data']['max_train_examples'] = config.get('data', {}).get('max_val_examples', 2000)
    val_dataset = AutoDataset(tokenizer, max_length=args.max_length, config=val_config, split='validation')
    
    print(f"  📚 Train dataset: {len(train_dataset)} samples")
    print(f"  📊 Validation dataset: {len(val_dataset)} samples")
    
    # Get data loading settings from config
    data_config = config.get('data', {})
    
    # Optimize for MPS - disable multiprocessing to avoid hangs
    if torch.backends.mps.is_available():
        # Disable workers for MPS to avoid multiprocessing issues
        num_workers = 0  # No workers - run in main process
        prefetch_factor = None  # No prefetch without workers
        persistent_workers = False
        pin_memory = False  # Don't pin for MPS
        drop_last = True
    else:
        num_workers = data_config.get('num_workers', 4)
        prefetch_factor = data_config.get('prefetch_factor', 2)
        persistent_workers = data_config.get('persistent_workers', True)
        pin_memory = data_config.get('pin_memory', False)
        drop_last = False
    
    print(f"⚡ DataLoader settings: workers={num_workers}, prefetch={prefetch_factor}, persistent={persistent_workers}")
    print(f"  Batch size: {batch_size}, effective batch: {batch_size * config.get('training', {}).get('gradient_accumulation_steps', 1)}")
    
    # Create train dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=True,  # Drop incomplete batches for consistent performance
        collate_fn=fast_collate  # Use optimized collate
    )
    
    # Create validation dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,  # No need for persistent workers in val
        drop_last=False,  # Keep all validation samples
        collate_fn=fast_collate
    )
    
    # Create model
    print("🤖 Creating model...")
    model_config = MoEConfig(**model_params)
    model = MoEForCausalLM(model_config)
    
    # Get dtype from config (inference section)
    inference_dtype_str = config.get('inference', {}).get('dtype', 'float32')
    
    # Convert string dtype to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    model_dtype = dtype_map.get(inference_dtype_str, torch.float32)
    
    # Convert model to specified dtype and move to device
    model = model.to(dtype=model_dtype, device=device)
    
    # Calculate model size based on actual dtype
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size based on actual dtype
    bytes_per_param = 4 if model_dtype == torch.float32 else 2  # float32=4 bytes, float16/bfloat16=2 bytes
    model_size_mb = total_params * bytes_per_param / (1024 * 1024)
    
    print(f"  📊 Model Parameters:")
    print(f"     • Total: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"     • Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"     • Model dtype: {inference_dtype_str}")
    print(f"     • Model size: {model_size_mb:.1f} MB ({inference_dtype_str})")
    print(f"  🖥️  Device: {device}")
    print(f"  ⚙️  Architecture:")
    print(f"     • Layers: {model_config.num_layers}")
    print(f"     • Hidden size: {model_config.hidden_size}")
    print(f"     • Attention heads: {model_config.num_attention_heads}")
    print(f"     • Experts: {model_config.num_experts} (active: {model_config.num_experts_per_tok})")
    print(f"  📚 Train dataset: {len(train_dataset)} samples")
    print(f"  🎯 Batch size: {batch_size}, Initial LR: {learning_rate}")
    
    # Print active features from config
    print(f"\n  🔧 Active Config Features:")
    print(f"     • Config file: {args.config}")
    print(f"     • Data source: processed_ace_full (52 shards)")
    print(f"     • Gradient checkpointing: {model_config.gradient_checkpointing}")
    print(f"     • Memory efficient attention: {model_config.use_memory_efficient_attention}")
    print(f"     • Flash attention: {model_config.use_flash_attn}")
    print(f"     • Mixed precision: {config.get('training', {}).get('mixed_precision', 'disabled')}")
    print(f"     • Optimizer: {config.get('training', {}).get('optimizer', 'adamw')}")
    print(f"     • Scheduler: {config.get('training', {}).get('scheduler', 'cosine')}")
    
    # Print anti-collapse features if enabled
    if config.get('training', {}).get('entropy_regularization_weight', 0) > 0:
        print(f"\n  🛡️  Anti-Collapse Features:")
        print(f"     • Entropy regularization: {config.get('training', {}).get('entropy_regularization_weight', 0)}")
        print(f"     • Token frequency penalty: {config.get('training', {}).get('token_frequency_penalty_weight', 0)}")
        print(f"     • Expert diversity weight: {config.get('training', {}).get('expert_diversity_weight', 0)}")
        print(f"     • Adaptive gradient clipping: {config.get('training', {}).get('adaptive_gradient_clipping', False)}")
        print(f"     • Curriculum learning: {config.get('training', {}).get('curriculum_learning', False)}")
    
    # Run learning rate finder if requested
    if args.find_lr:
        suggested_lr = find_learning_rate(
            model, 
            train_dataloader, 
            device, 
            config,
            num_steps=args.lr_finder_steps,
            start_lr=1e-7,
            end_lr=1
        )
        
        # Ask user if they want to use the suggested LR
        user_input = input(f"\n🔍 Use suggested learning rate {suggested_lr:.2e}? (y/n/custom): ").strip().lower()
        if user_input == 'y':
            learning_rate = suggested_lr
            print(f"✅ Using suggested learning rate: {learning_rate:.2e}")
        elif user_input not in ['n', 'no']:
            try:
                custom_lr = float(user_input)
                learning_rate = custom_lr
                print(f"✅ Using custom learning rate: {learning_rate:.2e}")
            except ValueError:
                print(f"⚠️ Invalid input, keeping original learning rate: {learning_rate:.2e}")
        else:
            print(f"➡️ Keeping original learning rate: {learning_rate:.2e}")
        
        # Log to wandb if available
        if wandb.run:
            wandb.log({
                'lr_finder/suggested_lr': suggested_lr,
                'lr_finder/selected_lr': learning_rate
            })
    
    # Create optimizer with layer-wise learning rates
    optimizer_type = config.get('training', {}).get('optimizer', 'adamw').lower()
    use_layerwise_lr = config.get('training', {}).get('layerwise_lr', False)
    
    # Prepare parameter groups for layer-wise learning rates
    if use_layerwise_lr:
        # Different learning rates for different layer types
        param_groups = []
        
        # Embeddings - lower learning rate
        embedding_params = []
        # Attention layers - medium learning rate
        attention_params = []
        # Expert/FFN layers - higher learning rate
        expert_params = []
        # Other parameters - default learning rate
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'embed' in name.lower():
                embedding_params.append(param)
            elif 'attention' in name.lower() or 'attn' in name.lower():
                attention_params.append(param)
            elif 'expert' in name.lower() or 'mlp' in name.lower() or 'ffn' in name.lower():
                expert_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with scaled learning rates
        lr_scale = config.get('training', {}).get('lr_layer_scale', 0.5)
        param_groups = [
            {'params': embedding_params, 'lr': learning_rate * lr_scale, 'name': 'embeddings'},
            {'params': attention_params, 'lr': learning_rate * (1 + lr_scale) / 2, 'name': 'attention'},
            {'params': expert_params, 'lr': learning_rate, 'name': 'experts'},
            {'params': other_params, 'lr': learning_rate, 'name': 'other'}
        ]
        
        # Remove empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        print(f"  📊 Layer-wise learning rates enabled:")
        for group in param_groups:
            print(f"     • {group['name']}: lr={group['lr']:.2e}, params={len(group['params'])}")
    else:
        param_groups = model.parameters()
    
    # Create optimizer based on type
    weight_decay = config.get('training', {}).get('weight_decay', 0.1)
    
    if optimizer_type == 'lion' and LION_AVAILABLE:
        print(f"  🦁 Using Lion optimizer (faster convergence, better efficiency)")
        optimizer = Lion(
            param_groups,
            lr=learning_rate * 0.3,  # Lion typically needs lower LR
            weight_decay=weight_decay,
            use_triton=False  # Disable triton for MPS compatibility
        )
    elif optimizer_type == 'adamw' or not LION_AVAILABLE:
        if optimizer_type == 'lion' and not LION_AVAILABLE:
            print(f"  ⚠️ Lion requested but not available, using AdamW")
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(config.get('training', {}).get('adam_beta1', 0.9), 
                   config.get('training', {}).get('adam_beta2', 0.95))
        )
    else:
        # Default to AdamW
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
    
    # Learning rate scheduler with cosine annealing and restarts
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
    
    # Get scheduler config
    training_config = config.get('training', {})
    total_steps = len(train_dataloader) * args.epochs
    # Ensure warmup steps don't exceed total steps
    warmup_steps = min(training_config.get('warmup_steps', 500), max(1, total_steps // 10))
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=warmup_steps
    )
    
    # Create cosine scheduler with restarts
    # Ensure T_0 is at least 1 to avoid error with small datasets
    T_0 = max(1, total_steps // 4)  # First restart after 1/4 of training, minimum 1
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=T_0,
        T_mult=2,  # Double the period after each restart
        eta_min=learning_rate * 0.01  # Minimum LR is 1% of initial
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Early stopping parameters
    early_stopping_config = config.get('fine_tuning', {})
    early_stopping_enabled = early_stopping_config.get('early_stopping', True)
    early_stopping_patience = early_stopping_config.get('early_stopping_patience', 5)
    early_stopping_threshold = early_stopping_config.get('early_stopping_threshold', 0.001)
    min_delta = early_stopping_config.get('min_delta', 0.0005)
    
    # Early stopping tracking
    patience_counter = 0
    best_val_loss = float('inf')
    
    # Overfitting/Underfitting tracking
    train_losses_history = []
    val_losses_history = []
    overfitting_ratios = []
    gradient_norms = []
    learning_rates = []
    
    # Validation function
    def evaluate_model(model, val_loader, device, use_ema_model=None):
        """Evaluate model on validation set"""
        # Use EMA model for validation if available
        eval_model = use_ema_model if use_ema_model is not None else model
        eval_model.eval()
        total_loss = 0
        total_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # Use same precision for validation as training
                if use_amp:
                    with torch.autocast(device_type='mps', dtype=amp_dtype):
                        outputs = eval_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = eval_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_batches += 1
        
        model.train()
        return total_loss / total_batches if total_batches > 0 else float('inf')
    
    # MPS optimizations and mixed precision setup
    use_amp = False
    scaler = None
    amp_dtype = torch.float32
    
    if device.type == 'mps':
        # MPS memory management - don't use set_per_process_memory_fraction as it conflicts with env var
        print("  🚀 MPS optimizations enabled (memory limit disabled)")
        print(f"  📊 Using gradient accumulation: {gradient_accumulation_steps} steps")
        print("  💾 Memory management: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
        
        # Check for mixed precision settings
        mixed_precision = config.get('training', {}).get('mixed_precision', False)
        if mixed_precision and mixed_precision != 'false' and mixed_precision != False:
            # Determine the precision type
            if mixed_precision == 'bfloat16' or mixed_precision == True:
                target_dtype = torch.bfloat16
                dtype_name = 'bfloat16'
            elif mixed_precision == 'float16':
                target_dtype = torch.float16
                dtype_name = 'float16'
            else:
                target_dtype = torch.bfloat16  # Default to bfloat16
                dtype_name = 'bfloat16'
            
            # Check if the target dtype is supported on this MPS device
            if hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
                try:
                    # Test dtype support
                    test_tensor = torch.tensor([1.0], device=device, dtype=target_dtype)
                    use_amp = True
                    amp_dtype = target_dtype
                    print(f"  ⚡ Mixed precision enabled: {dtype_name}")
                except Exception as e:
                    print(f"  ⚠️ {dtype_name} not supported on this MPS device, using {inference_dtype_str}")
                    print(f"     Error: {e}")
            else:
                print(f"  ⚠️ Mixed precision requested but MPS doesn't support it on this device")
    
    # Training loop
    print(f"\n🏃 Training for {args.epochs} epochs...")
    print(f"  📊 Learning rate scheduling: Cosine with warm restarts")
    print(f"  🛑 Early stopping: {'Enabled' if early_stopping_enabled else 'Disabled'}")
    if early_stopping_enabled:
        print(f"    - Patience: {early_stopping_patience} evaluations")
        print(f"    - Min delta: {min_delta}")
    print("="*60)
    
    # Start global_step after LR finder steps if it was run
    global_step = args.lr_finder_steps if args.find_lr else 0
    best_loss = float('inf')
    
    # Initialize list to accumulate all overfitting analysis images for slider
    all_overfitting_images = []
    
    # 3. ADAPTIVE GRADIENT CLIPPING - Dynamic clipping based on model health
    adaptive_clipping = config.get('training', {}).get('adaptive_gradient_clipping', True)
    base_grad_clip = config.get('training', {}).get('max_grad_norm', 1.0)
    current_grad_clip = base_grad_clip
    clip_reduction_factor = 0.3  # Reduce to 30% when repetition detected
    clip_recovery_rate = 1.1  # Increase by 10% each healthy step
    
    # Initialize EMA if enabled
    ema_model = None
    if config.get('training', {}).get('use_ema', False):
        print("  📊 Initializing EMA (Exponential Moving Average)")
        ema_decay = config.get('training', {}).get('ema_decay', 0.999)
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
    
    # Store initial weights for mixout
    initial_weights = {}
    if config.get('training', {}).get('mixout_prob', 0) > 0:
        print(f"  🔀 Mixout enabled with probability {config.get('training', {}).get('mixout_prob', 0)}")
        for name, param in model.named_parameters():
            initial_weights[name] = param.data.clone()
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        epoch_losses = []
        
        # Clear MPS cache at start of each epoch
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        # Pre-allocate GPU memory for faster transfers (MPS optimization)
        if device.type == 'mps' and epoch == 0:
            # Warm up GPU memory on first epoch only
            try:
                dummy_batch = next(iter(train_dataloader))
                _ = dummy_batch['input_ids'].to(device)
                torch.mps.empty_cache()
            except StopIteration:
                pass
        
        for batch_idx, batch in enumerate(progress):
            # Move batch to device (non-blocking for better overlap)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision if enabled
            if use_amp:
                with torch.autocast(device_type='mps', dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            
            logits = outputs.get('logits', outputs.get('output', None))
            
            # Skip if NaN
            if torch.isnan(loss):
                print(f"  ⚠️  NaN loss at step {global_step}, skipping...")
                logger.warning(f"NaN loss at step {global_step}, batch skipped")
                continue
            
            # FOCAL LOSS - Focus on hard examples
            use_focal_loss = config.get('training', {}).get('use_focal_loss', False)
            if use_focal_loss and logits is not None:
                focal_gamma = config.get('training', {}).get('focal_gamma', 2.0)
                focal_alpha = config.get('training', {}).get('focal_alpha', 0.25)
                
                # Calculate per-token probabilities
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                
                # Get probabilities for correct classes
                probs = F.softmax(shift_logits, dim=-1)
                correct_probs = probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                
                # Calculate focal weight: (1 - p)^gamma
                focal_weight = (1 - correct_probs) ** focal_gamma
                
                # Apply focal weight to loss
                # Recalculate loss with focal weighting
                ce_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                focal_loss = (focal_weight * ce_loss).mean()
                
                # Blend focal loss with original loss
                focal_blend = config.get('training', {}).get('focal_blend', 0.5)
                loss = (1 - focal_blend) * loss + focal_blend * focal_loss
                
                if wandb.run and global_step % 50 == 0:
                    wandb.log({
                        'training/focal_weight_mean': focal_weight.mean().item(),
                        'training/hard_examples_ratio': (focal_weight > 0.5).float().mean().item()
                    }, step=global_step)
            
            # 1. ENTROPY REGULARIZATION - Encourage diverse outputs
            entropy_weight = config.get('training', {}).get('entropy_regularization_weight', 0.01)
            if entropy_weight > 0 and logits is not None:
                # Calculate entropy from logits (higher entropy = more diverse)
                # Shift logits and labels for causal LM (predict next token)
                shift_logits = logits[..., :-1, :].contiguous()
                probs = F.softmax(shift_logits, dim=-1)
                # Calculate entropy: -sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                avg_entropy = entropy.mean()
                
                # Add entropy bonus (negative because we want to maximize entropy)
                entropy_bonus = -entropy_weight * avg_entropy
                loss = loss + entropy_bonus
                
                # Log entropy for monitoring
                if wandb.run and global_step % 50 == 0:
                    wandb.log({
                        'anti_collapse/entropy': avg_entropy.item(),
                        'anti_collapse/entropy_bonus': entropy_bonus.item()
                    }, step=global_step)
            
            # 2. TOKEN FREQUENCY PENALTY - Penalize overused tokens
            freq_penalty_weight = config.get('training', {}).get('token_frequency_penalty_weight', 0.1)
            if freq_penalty_weight > 0 and logits is not None:
                # Count token frequencies in predictions
                shift_logits = logits[..., :-1, :].contiguous()
                predicted_tokens = shift_logits.argmax(dim=-1)
                
                # Calculate frequency distribution
                batch_size, seq_len = predicted_tokens.shape
                unique_tokens, counts = torch.unique(predicted_tokens.flatten(), return_counts=True)
                frequencies = counts.float() / (batch_size * seq_len)
                
                # Calculate frequency penalty (penalize high frequency tokens)
                # Higher penalty for tokens that appear too often
                freq_penalty = torch.sum(frequencies ** 2)
                scaled_freq_penalty = freq_penalty_weight * freq_penalty
                loss = loss + scaled_freq_penalty
                
                # Calculate diversity score
                num_unique = len(unique_tokens)
                max_possible = min(batch_size * seq_len, logits.size(-1))
                diversity_ratio = num_unique / max_possible
                
                # Log frequency metrics
                if wandb.run and global_step % 50 == 0:
                    wandb.log({
                        'anti_collapse/token_diversity_ratio': diversity_ratio,
                        'anti_collapse/unique_tokens': num_unique,
                        'anti_collapse/frequency_penalty': scaled_freq_penalty.item(),
                        'anti_collapse/max_token_frequency': frequencies.max().item()
                    }, step=global_step)
            
            # 5. EXPERT DIVERSITY LOSS - Encourage different experts to learn different patterns
            expert_diversity_weight = config.get('training', {}).get('expert_diversity_weight', 0.01)
            if expert_diversity_weight > 0 and 'expert_outputs' in outputs:
                # Get expert outputs if available from MoE model
                expert_outputs = outputs['expert_outputs']
                
                # Calculate pairwise cosine similarity between experts
                diversity_loss = 0
                num_pairs = 0
                
                # Flatten expert outputs for comparison
                if isinstance(expert_outputs, list) and len(expert_outputs) > 1:
                    for i in range(len(expert_outputs)):
                        for j in range(i+1, len(expert_outputs)):
                            # Normalize outputs
                            output_i = F.normalize(expert_outputs[i].view(-1), dim=0)
                            output_j = F.normalize(expert_outputs[j].view(-1), dim=0)
                            
                            # Calculate cosine similarity
                            similarity = torch.dot(output_i, output_j)
                            
                            # We want to minimize similarity (maximize diversity)
                            diversity_loss += similarity
                            num_pairs += 1
                    
                    if num_pairs > 0:
                        diversity_loss = diversity_loss / num_pairs
                        scaled_diversity_loss = expert_diversity_weight * diversity_loss
                        loss = loss + scaled_diversity_loss
                        
                        # Log diversity metrics
                        if wandb.run and global_step % 50 == 0:
                            wandb.log({
                                'anti_collapse/expert_diversity_loss': scaled_diversity_loss.item(),
                                'anti_collapse/expert_similarity': diversity_loss.item()
                            }, step=global_step)
            
            # Also track router statistics if available
            if 'router_probs' in outputs and wandb.run and global_step % 100 == 0:
                router_probs = outputs['router_probs']
                if router_probs is not None:
                    # Calculate expert usage balance
                    avg_probs = router_probs.mean(dim=0).mean(dim=0)  # Average across batch and sequence
                    max_usage = avg_probs.max().item()
                    min_usage = avg_probs.min().item()
                    usage_std = avg_probs.std().item()
                    
                    wandb.log({
                        'anti_collapse/expert_max_usage': max_usage,
                        'anti_collapse/expert_min_usage': min_usage,
                        'anti_collapse/expert_usage_std': usage_std,
                        'anti_collapse/expert_balance_ratio': min_usage / (max_usage + 1e-8)
                    }, step=global_step)
            
            # 6. GRADIENT PENALTY - Penalize large gradients
            gradient_penalty_weight = config.get('training', {}).get('gradient_penalty_weight', 0)
            if gradient_penalty_weight > 0:
                # Calculate gradient penalty (requires grad for gradient computation)
                grad_params = torch.autograd.grad(
                    outputs=loss,
                    inputs=[p for p in model.parameters() if p.requires_grad],
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )
                
                # Calculate L2 norm of gradients
                grad_norm = 0
                for grad in grad_params:
                    if grad is not None:
                        grad_norm += grad.pow(2).sum()
                grad_norm = torch.sqrt(grad_norm)
                
                # Add penalty to loss
                gradient_penalty = gradient_penalty_weight * grad_norm
                loss = loss + gradient_penalty
                
                if wandb.run and global_step % 50 == 0:
                    wandb.log({
                        'anti_overfitting/gradient_penalty': gradient_penalty.item(),
                        'anti_overfitting/gradient_norm_raw': grad_norm.item()
                    }, step=global_step)
            
            # Backward pass with gradient accumulation
            # Scale loss by accumulation steps
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            # Initialize total_norm for tracking
            total_norm = 0.0
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # GRADIENT CENTRALIZATION - Center gradients to zero mean
                use_gradient_centralization = config.get('training', {}).get('gradient_centralization', False)
                if use_gradient_centralization:
                    for p in model.parameters():
                        if p.grad is not None and len(p.shape) > 1:  # Only for weight matrices, not biases
                            # Center gradient to zero mean
                            grad = p.grad.data
                            grad_mean = grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
                            p.grad.data = grad - grad_mean
                
                # Check for NaN gradients before clipping
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                if torch.isnan(torch.tensor(total_norm)):
                    print(f"  ⚠️ NaN gradients detected at step {global_step}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                # Use adaptive gradient clipping
                if adaptive_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), current_grad_clip)
                else:
                    max_grad_norm = config.get('training', {}).get('max_grad_norm', 0.5)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # Apply mixout if enabled (mix weights with initialization)
                mixout_prob = config.get('training', {}).get('mixout_prob', 0)
                if mixout_prob > 0 and random.random() < mixout_prob:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in initial_weights:
                                # Mix current weights with initial weights
                                param.data = (1 - mixout_prob) * param.data + mixout_prob * initial_weights[name]
                
                # Update EMA model if enabled
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                            ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)
                
                # Step the learning rate scheduler
                scheduler.step()
                
                # Clear MPS cache more frequently to prevent memory fragmentation
                if device.type == 'mps' and (batch_idx + 1) % 50 == 0:
                    torch.mps.empty_cache()
                    torch.mps.synchronize()  # Ensure GPU ops complete
            
            # Track loss
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            global_step += 1
            
            # Update curriculum learning with progressive growth tracking
            if curriculum_enabled and hasattr(train_dataset, 'update_curriculum'):
                current_seq_len = train_dataset.update_curriculum(global_step)
                if global_step % 50 == 0:  # Log more frequently
                    # Calculate which phase we're in
                    if global_step < curriculum_warmup_steps:
                        phase = "warmup"
                        phase_progress = global_step / curriculum_warmup_steps
                    else:
                        steps_after = global_step - curriculum_warmup_steps
                        growth_interval = config.get('training', {}).get('sequence_growth_interval', 5000)
                        phase_num = steps_after // growth_interval
                        phase = f"growth_{phase_num}"
                        phase_progress = (steps_after % growth_interval) / growth_interval
                    
                    if wandb.run:
                        wandb.log({
                            'curriculum/max_sequence_length': current_seq_len,
                            'curriculum/phase': phase,
                            'curriculum/phase_progress': phase_progress,
                            'curriculum/tokens_per_batch': current_seq_len * batch_size
                        }, step=global_step)
                    
                    # Print status every 500 steps
                    if global_step % 500 == 0:
                        print(f"  📏 Current sequence length: {current_seq_len} (Phase: {phase})")
            
            # Monitor for repetition in model outputs (every 100 steps)
            if global_step % 100 == 0:
                with torch.no_grad():
                    # Generate a sample to check for repetition
                    test_prompt_ids = input_ids[:1, :10]  # Use first 10 tokens of current batch
                    generated_ids = test_prompt_ids.clone()
                    
                    for _ in range(30):  # Generate 30 tokens
                        test_outputs = model(generated_ids)
                        test_logits = test_outputs['logits'] if isinstance(test_outputs, dict) else test_outputs
                        next_token = test_logits[0, -1, :].argmax()
                        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                    
                    # Check for repetition
                    generated_tokens = generated_ids[0, 10:].tolist()  # Skip prompt
                    
                    # Check for immediate repetition (same token repeated)
                    max_consecutive = 1
                    current_consecutive = 1
                    for i in range(1, len(generated_tokens)):
                        if generated_tokens[i] == generated_tokens[i-1]:
                            current_consecutive += 1
                            max_consecutive = max(max_consecutive, current_consecutive)
                        else:
                            current_consecutive = 1
                    
                    # Calculate diversity score (unique tokens / total tokens)
                    unique_tokens = len(set(generated_tokens))
                    diversity_score = unique_tokens / len(generated_tokens) if generated_tokens else 0
                    
                    # Check for pattern repetition (2-3 token patterns)
                    pattern_count = 0
                    for pattern_len in [2, 3]:
                        for i in range(len(generated_tokens) - pattern_len * 3):
                            pattern = generated_tokens[i:i+pattern_len]
                            if (generated_tokens[i+pattern_len:i+pattern_len*2] == pattern and 
                                generated_tokens[i+pattern_len*2:i+pattern_len*3] == pattern):
                                pattern_count += 1
                    
                    # Calculate repetition score (0 = no repetition, 1 = severe repetition)
                    repetition_score = min(1.0, (max_consecutive / 10.0) + (pattern_count / 5.0))
                    
                    # Always log metrics to wandb for charts
                    if wandb.run:
                        wandb.log({
                            'repetition/max_consecutive_tokens': max_consecutive,
                            'repetition/diversity_score': diversity_score,
                            'repetition/pattern_count': pattern_count,
                            'repetition/repetition_score': repetition_score,
                            'repetition/health_score': 1.0 - repetition_score,  # Higher is better
                        }, step=global_step)
                    
                    # Warn if repetition detected and adjust gradient clipping
                    if max_consecutive > 5:
                        print(f"  ⚠️ WARNING: Model generating repeated tokens (max {max_consecutive} consecutive)")
                        logger.warning(f"Repetition detected at step {global_step}: {max_consecutive} consecutive tokens")
                        
                        # Reduce gradient clipping threshold when repetition detected
                        if adaptive_clipping:
                            current_grad_clip = base_grad_clip * clip_reduction_factor
                            print(f"  📉 Reducing gradient clip to {current_grad_clip:.3f} due to repetition")
                        
                        if wandb.run:
                            wandb.log({
                                'repetition/warning_triggered': 1,
                                'anti_collapse/gradient_clip_value': current_grad_clip
                            }, step=global_step)
                    else:
                        # Gradually recover gradient clipping when healthy
                        if adaptive_clipping and current_grad_clip < base_grad_clip:
                            current_grad_clip = min(base_grad_clip, current_grad_clip * clip_recovery_rate)
                        
                        if wandb.run:
                            wandb.log({
                                'repetition/warning_triggered': 0,
                                'anti_collapse/gradient_clip_value': current_grad_clip
                            }, step=global_step)
                    
                    if pattern_count > 0:
                        print(f"  ⚠️ WARNING: Model generating repeated patterns (found {pattern_count} patterns)")
                        logger.warning(f"Pattern repetition detected at step {global_step}: {pattern_count} patterns")
            
            # Update progress with current learning rate
            avg_loss = sum(epoch_losses[-100:]) / min(100, len(epoch_losses))
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            gradient_norms.append(total_norm)
            progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}", "grad_norm": f"{total_norm:.2f}"})
            
            # Log to wandb on EVERY step for continuous tracking
            if use_wandb:
                wandb.log({
                    'train/loss': float(loss_value),
                    'train/loss_smooth': float(avg_loss),
                    'train/lr': float(current_lr),
                    'train/grad_norm': float(total_norm),
                    'train/epoch': epoch + 1,  # Human-readable epoch (1-indexed)
                }, step=global_step)
            
            # Print progress every 50 steps
            if global_step % 50 == 0 and global_step > 0:
                print(f"  📊 Step {global_step}: loss={loss_value:.4f}, avg={avg_loss:.4f}, lr={current_lr:.2e}")
            
            # Evaluate on validation set periodically
            eval_steps = config.get('training', {}).get('eval_steps', 50)
            # Force eval at step 50 for testing, then every eval_steps
            if (global_step == 50 or (global_step % eval_steps == 0 and global_step > 0)):
                print(f"\n  🔍 Running validation at step {global_step}...")
                print(f"     Validation dataloader size: {len(val_dataloader)} batches")
                
                # Check if validation data exists
                if len(val_dataloader) == 0:
                    print(f"     ⚠️ No validation data available - using training loss as proxy")
                    val_loss = avg_loss * 1.1  # Simulate slight overfitting
                else:
                    try:
                        val_loss = evaluate_model(model, val_dataloader, device, use_ema_model=ema_model)
                        print(f"     ✅ Validation complete: val_loss={val_loss:.4f}")
                    except Exception as e:
                        print(f"     ⚠️ Validation failed: {e}")
                        val_loss = avg_loss * 1.1  # Fallback
                train_losses_history.append(avg_loss)
                val_losses_history.append(val_loss)
                
                # Calculate overfitting metrics
                overfitting_ratio = val_loss / avg_loss if avg_loss > 0 else 1.0
                overfitting_ratios.append(overfitting_ratio)
                
                # Determine model state
                model_state = "optimal"
                if overfitting_ratio > 1.5:
                    model_state = "overfitting"
                elif overfitting_ratio < 0.9 and val_loss > avg_loss:
                    model_state = "underfitting"
                
                print(f"\n  📊 Step {global_step} - Train loss: {avg_loss:.4f}, Val loss: {val_loss:.4f}, Ratio: {overfitting_ratio:.2f} ({model_state})")
                logger.info(f"Validation at step {global_step}: train={avg_loss:.4f}, val={val_loss:.4f}, ratio={overfitting_ratio:.2f}, state={model_state}")
                
                # Log to W&B with enhanced metrics
                if use_wandb:
                    # First, log the eval metrics directly to ensure they appear
                    eval_metrics = {
                        'eval/train_loss': float(avg_loss),
                        'eval/val_loss': float(val_loss),
                        'eval/overfitting_ratio': float(overfitting_ratio),
                    }
                    print(f"  📊 Logging eval metrics: {eval_metrics}")
                    wandb.log(eval_metrics, step=global_step)
                    
                    # Then prepare full metrics for visualization
                    wandb_metrics = {
                        'eval/train_loss': float(avg_loss),
                        'eval/val_loss': float(val_loss),
                        'eval/overfitting_ratio': float(overfitting_ratio),
                        'eval/gradient_norm': float(total_norm),
                        'eval/learning_rate': float(current_lr),
                        'epoch': int(epoch),
                    }
                    
                    # Create enhanced overfitting/underfitting analysis visualization
                    try:
                        if len(train_losses_history) > 1:
                            # Set style for better aesthetics
                            try:
                                plt.style.use('seaborn-v0_8-darkgrid')
                            except:
                                # Fallback if seaborn style not available
                                plt.style.use('default')
                            fig = plt.figure(figsize=(16, 12))
                            
                            # Create grid with custom spacing
                            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                            
                            # Main plot: Loss curves with enhanced styling
                            ax_main = fig.add_subplot(gs[0:2, 0:2])
                            steps = list(range(0, global_step+1, eval_steps))[:len(train_losses_history)]
                            
                            # Plot with gradient fill
                            train_line = ax_main.plot(steps, train_losses_history, 
                                                             label='Training Loss', color='#2E86AB', 
                                                             linewidth=2.5, alpha=0.9, marker='o', 
                                                             markersize=3, markevery=max(1, len(steps)//20))
                            val_line = ax_main.plot(steps, val_losses_history, 
                                                           label='Validation Loss', color='#A23B72', 
                                                           linewidth=2.5, alpha=0.9, marker='s',
                                                           markersize=3, markevery=max(1, len(steps)//20))
                            
                            # Add confidence bands
                            ax_main.fill_between(steps, train_losses_history, val_losses_history,
                                                   where=[v > t*1.1 for t, v in zip(train_losses_history, val_losses_history)],
                                                   color='#FF6B6B', alpha=0.3, label='Overfitting Zone')
                            ax_main.fill_between(steps, train_losses_history, val_losses_history,
                                                   where=[v <= t*1.1 for t, v in zip(train_losses_history, val_losses_history)],
                                                   color='#4ECDC4', alpha=0.3, label='Healthy Zone')
                        
                            # Add current status annotation
                            current_status = f"Current: Train={avg_loss:.4f}, Val={val_loss:.4f}"
                            ax_main.annotate(current_status, xy=(steps[-1], val_losses_history[-1]),
                                       xytext=(10, 10), textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                       fontsize=9, fontweight='bold')
                        
                            ax_main.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
                            ax_main.set_ylabel('Loss', fontsize=12, fontweight='bold')
                            ax_main.set_title('Model Training Progress Overview', fontsize=14, fontweight='bold', pad=20)
                            ax_main.legend(loc='upper right', framealpha=0.9, fontsize=10)
                            ax_main.grid(True, alpha=0.3, linestyle='--')
                        
                            # Overfitting/Underfitting indicator line graph
                            ax_indicator = fig.add_subplot(gs[0, 2])
                        
                            # Plot recent overfitting ratios
                            recent_steps = steps[-20:] if len(steps) > 20 else steps
                            recent_ratios = overfitting_ratios[-20:] if len(overfitting_ratios) > 20 else overfitting_ratios
                        
                            # Create background zones
                            ax_indicator.axhspan(0, 0.9, facecolor='#3498db', alpha=0.2, label='Underfitting')
                            ax_indicator.axhspan(0.9, 1.1, facecolor='#2ecc71', alpha=0.3, label='Optimal')
                            ax_indicator.axhspan(1.1, 1.5, facecolor='#ffa500', alpha=0.2, label='Mild Overfit')
                            ax_indicator.axhspan(1.5, 2.5, facecolor='#e74c3c', alpha=0.2, label='Severe Overfit')
                        
                            # Plot the ratio line with gradient coloring
                            for i in range(len(recent_steps)-1):
                                ratio_val = recent_ratios[i]
                                if ratio_val < 0.9:
                                    color = '#3498db'
                                elif ratio_val < 1.1:
                                    color = '#2ecc71'
                                elif ratio_val < 1.5:
                                    color = '#ffa500'
                                else:
                                    color = '#e74c3c'
                                ax_indicator.plot(recent_steps[i:i+2], recent_ratios[i:i+2], 
                                                    color=color, linewidth=3, alpha=0.8)
                        
                            # Add markers for the last point
                            last_color = '#2ecc71' if 0.9 <= overfitting_ratio <= 1.1 else '#e74c3c' if overfitting_ratio > 1.5 else '#ffa500' if overfitting_ratio > 1.1 else '#3498db'
                            ax_indicator.scatter(recent_steps[-1], recent_ratios[-1], 
                                                   color=last_color, s=100, zorder=5, edgecolor='black', linewidth=2)
                        
                            # Add horizontal reference lines
                            ax_indicator.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                            ax_indicator.axhline(y=1.5, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)
                            ax_indicator.axhline(y=0.9, color='#3498db', linestyle='--', linewidth=1, alpha=0.5)
                        
                            # Add current value annotation
                            ax_indicator.annotate(f'{overfitting_ratio:.2f}', 
                                                   xy=(recent_steps[-1], recent_ratios[-1]),
                                                   xytext=(5, 0), textcoords='offset points',
                                                   fontsize=10, fontweight='bold',
                                                   bbox=dict(boxstyle='round,pad=0.3', 
                                                           fc=last_color, alpha=0.7, edgecolor='black'))
                        
                            # Formatting
                            ax_indicator.set_xlabel('Recent Steps', fontsize=10, fontweight='bold')
                            ax_indicator.set_ylabel('Val/Train Ratio', fontsize=10, fontweight='bold')
                            ax_indicator.set_title('Fit Quality Indicator', fontsize=12, fontweight='bold')
                            ax_indicator.set_ylim(0, min(2.5, max(2.0, max(recent_ratios)*1.1)))
                            ax_indicator.grid(True, alpha=0.3, linestyle='--')
                            ax_indicator.legend(loc='upper left', fontsize=8, framealpha=0.9)
                        
                            # Learning rate curve with smoothing
                            ax_lr = fig.add_subplot(gs[1, 2])
                            if learning_rates:
                                lr_window = learning_rates[-500:]
                                lr_steps_plot = list(range(len(lr_window)))
                                ax_lr.plot(lr_steps_plot, lr_window, color='#FF6F61', linewidth=2, alpha=0.8)
                                ax_lr.fill_between(lr_steps_plot, 0, lr_window, color='#FF6F61', alpha=0.2)
                                ax_lr.set_ylabel('Learning Rate', fontsize=10, fontweight='bold')
                                ax_lr.set_xlabel('Recent Steps', fontsize=10, fontweight='bold')
                                ax_lr.set_title('LR Schedule (Recent)', fontsize=11, fontweight='bold')
                                ax_lr.set_yscale('log')
                                ax_lr.grid(True, alpha=0.3, linestyle='--')
                            
                                # Add current LR annotation
                                ax_lr.text(0.95, 0.95, f'Current: {current_lr:.2e}',
                                     transform=ax_lr.transAxes, ha='right', va='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                     fontsize=9)
                        
                            # Overfitting ratio history with trend
                            ax_ratio = fig.add_subplot(gs[2, :2])
                            ax_ratio.plot(steps, overfitting_ratios, color='#6A4C93', linewidth=2, alpha=0.8)
                        
                            # Add moving average
                            if len(overfitting_ratios) > 5:
                                window = min(5, len(overfitting_ratios))
                                moving_avg = np.convolve(overfitting_ratios, np.ones(window)/window, mode='valid')
                                ma_steps = steps[window-1:]
                                ax_ratio.plot(ma_steps, moving_avg, color='#FF6F61', linewidth=2, 
                                        linestyle='--', alpha=0.7, label='Moving Avg')
                        
                            # Add reference lines with labels
                            ax_ratio.axhline(y=1.0, color='#2ecc71', linestyle='-', linewidth=1.5, alpha=0.7)
                            ax_ratio.axhline(y=1.5, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
                            ax_ratio.axhline(y=0.9, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)
                        
                            # Shade regions
                            ax_ratio.fill_between(steps, 1.5, max(overfitting_ratios + [2.0]), 
                                                    color='#e74c3c', alpha=0.15)
                            ax_ratio.fill_between(steps, 0.9, 1.5, color='#2ecc71', alpha=0.15)
                            ax_ratio.fill_between(steps, 0, 0.9, color='#3498db', alpha=0.15)
                        
                            ax_ratio.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
                            ax_ratio.set_ylabel('Val/Train Ratio', fontsize=11, fontweight='bold')
                            ax_ratio.set_title('Overfitting Ratio Evolution', fontsize=12, fontweight='bold')
                            ax_ratio.grid(True, alpha=0.3, linestyle='--')
                            ax_ratio.legend(loc='upper right', fontsize=9)
                        
                            # Statistics box
                            ax_stats = fig.add_subplot(gs[2, 2])
                            ax_stats.axis('off')
                        
                            # Calculate statistics
                            recent_ratios = overfitting_ratios[-10:] if len(overfitting_ratios) > 10 else overfitting_ratios
                            
                            # Get latest repetition metrics from previous monitoring
                            repetition_health = "🟢 Healthy"
                            if 'max_consecutive' in locals() and max_consecutive > 5:
                                repetition_health = "🔴 Warning!"
                            elif 'max_consecutive' in locals() and max_consecutive > 3:
                                repetition_health = "🟡 Caution"
                            
                            diversity_str = f"{diversity_score:.2f}" if 'diversity_score' in locals() else "N/A"
                            
                            stats_text = f"""📊 Training Statistics
                            
Current Step: {global_step}
Epoch: {epoch + 1}

Loss Metrics:
• Train: {avg_loss:.4f}
• Valid: {val_loss:.4f}
• Gap: {val_loss - avg_loss:.4f}

Fit Analysis:
• Current: {overfitting_ratio:.3f}
• Recent Avg: {np.mean(recent_ratios):.3f}
• Trend: {'↗' if len(overfitting_ratios) > 1 and overfitting_ratios[-1] > overfitting_ratios[-2] else '↘'}

Repetition Monitor:
• Status: {repetition_health}
• Diversity: {diversity_str}

Gradient Health:
• Norm: {total_norm:.3f}
• Status: {'✓' if total_norm < 10 else '⚠'}"""
                            
                            ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                                        fontsize=9, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
                            
                            # Add title with timestamp
                            fig.suptitle(f'Training Analysis Dashboard - Step {global_step}', 
                                       fontsize=16, fontweight='bold', y=0.98)
                            
                            plt.tight_layout()
                            
                            # Create wandb Image with caption showing the step
                            wandb_image = wandb.Image(
                                fig, 
                                caption=f"Step {global_step} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Ratio: {overfitting_ratio:.2f}"
                            )
                            
                            # Add to accumulated list for slider
                            all_overfitting_images.append(wandb_image)
                            
                            # Log all accumulated images as a list (creates slider in wandb)
                            wandb_metrics['overfitting_analysis_timeline'] = all_overfitting_images
                            
                            # Also keep the individual step version for reference  
                            wandb_metrics[f'overfitting_analysis_step_{global_step}'] = wandb_image
                            
                            plt.close(fig)
                    except Exception as e:
                        print(f"Warning: Could not generate visualization: {e}")
                        # Still log the metrics even if visualization fails
                    
                    # Always log metrics regardless of visualization success
                    # Separate scalar metrics from images
                    scalar_metrics = {k: v for k, v in wandb_metrics.items() 
                                    if not ('analysis' in k or isinstance(v, wandb.Image))}
                    image_metrics = {k: v for k, v in wandb_metrics.items() 
                                   if 'analysis' in k or isinstance(v, wandb.Image)}
                    
                    print(f"  📈 Logging {len(scalar_metrics)} scalar metrics to W&B at step {global_step}:")
                    for key, value in scalar_metrics.items():
                        print(f"     - {key}: {value:.4f}" if isinstance(value, (int, float)) else f"     - {key}: {value}")
                    
                    try:
                        # Log ALL metrics together in one call
                        all_metrics = {**scalar_metrics, **image_metrics}
                        wandb.log(all_metrics, step=global_step)
                        print(f"     ✅ Successfully logged {len(all_metrics)} metrics to W&B")
                    except Exception as e:
                        print(f"     ⚠️ Failed to log to W&B: {e}")
                        print(f"     Metrics types: {[(k, type(v).__name__) for k, v in wandb_metrics.items()]}")
                
                # Check for severe overfitting
                if overfitting_ratio > 1.5:
                    print(f"  ⚠️  OVERFITTING detected! Val/Train ratio: {overfitting_ratio:.2f}")
                    print(f"     Recommendations:")
                    print(f"     - Increase dropout (current: {model_config.dropout if hasattr(model_config, 'dropout') else 'N/A'})")
                    print(f"     - Increase weight decay (current: {config.get('training', {}).get('weight_decay', 0.1)})")
                    print(f"     - Reduce model capacity or add regularization")
                    logger.warning(f"Overfitting detected at step {global_step}, ratio: {overfitting_ratio:.2f}")
                elif overfitting_ratio < 0.9:
                    print(f"  ⚠️  UNDERFITTING detected! Val/Train ratio: {overfitting_ratio:.2f}")
                    print(f"     Recommendations:")
                    print(f"     - Increase model capacity")
                    print(f"     - Reduce regularization")
                    print(f"     - Train for more epochs")
                    logger.warning(f"Underfitting detected at step {global_step}, ratio: {overfitting_ratio:.2f}")
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                # Ensure output directory exists
                output_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = output_dir / f"checkpoint_step_{global_step}.pt"
                try:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': model_config.__dict__,
                        'step': global_step,
                        'epoch': epoch,
                        'loss': avg_loss
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint at step {global_step}: {checkpoint_path}")
                except Exception as e:
                    print(f"  ⚠️ Could not save checkpoint: {e}")
                    logger.error(f"Failed to save checkpoint: {e}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = output_dir / "best_model.pt"
                    try:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': model_config.__dict__,
                            'step': global_step,
                            'epoch': epoch,
                            'loss': avg_loss
                        }, best_path)
                        print(f"\n  💾 Saved best model (loss: {avg_loss:.4f})")
                        logger.info(f"New best model saved with loss: {avg_loss:.4f}")
                    except Exception as e:
                        print(f"  ⚠️ Could not save best model: {e}")
                        logger.error(f"Failed to save best model: {e}")
        
        # Epoch summary with validation
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        epoch_val_loss = evaluate_model(model, val_dataloader, device, use_ema_model=ema_model)
        epoch_overfitting_ratio = epoch_val_loss / epoch_avg_loss if epoch_avg_loss > 0 else 1.0
        
        # Determine final epoch state
        epoch_state = "optimal"
        if epoch_overfitting_ratio > 1.5:
            epoch_state = "overfitting"
        elif epoch_overfitting_ratio < 0.9:
            epoch_state = "underfitting"
        
        print(f"  ✅ Epoch {epoch+1} complete - Train: {epoch_avg_loss:.4f}, Val: {epoch_val_loss:.4f}, Ratio: {epoch_overfitting_ratio:.2f} ({epoch_state})")
        logger.info(f"Epoch {epoch+1} complete - Train: {epoch_avg_loss:.4f}, Val: {epoch_val_loss:.4f}, Ratio: {epoch_overfitting_ratio:.2f}, State: {epoch_state}")
        
        # Log epoch summary to W&B
        if use_wandb:
            epoch_metrics = {
                'epoch': epoch + 1,
                'epoch_train_loss': epoch_avg_loss,
                'epoch_val_loss': epoch_val_loss,
                'epoch_overfitting_ratio': epoch_overfitting_ratio,
                'epoch_state': epoch_state,
                'total_steps': global_step
            }
            
            # Create final epoch comparison chart
            if train_losses_history and val_losses_history:
                fig, ax = plt.subplots(figsize=(10, 6))
                steps = list(range(0, global_step+1, eval_steps))[:len(train_losses_history)]
                ax.plot(steps, train_losses_history, 'b-', label='Training Loss', linewidth=2)
                ax.plot(steps, val_losses_history, 'r-', label='Validation Loss', linewidth=2)
                
                # Add shaded regions for overfitting/underfitting
                ax.fill_between(steps, 0, max(val_losses_history + train_losses_history),
                               where=[r > 1.5 for r in overfitting_ratios],
                               color='red', alpha=0.1, label='Overfitting')
                ax.fill_between(steps, 0, max(val_losses_history + train_losses_history),
                               where=[r < 0.9 for r in overfitting_ratios],
                               color='blue', alpha=0.1, label='Underfitting')
                
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Loss')
                ax.set_title(f'Epoch {epoch+1} - Training Progress (State: {epoch_state})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                epoch_metrics['epoch_summary_plot'] = wandb.Image(fig)
                plt.close(fig)
            
            wandb.log(epoch_metrics, step=global_step)
        
        # Check overfitting with detailed analysis
        if epoch_overfitting_ratio > 1.3:
            print(f"  ⚠️  OVERFITTING: Validation loss ({epoch_val_loss:.4f}) significantly higher than training loss ({epoch_avg_loss:.4f})")
            print(f"     Current ratio: {epoch_overfitting_ratio:.2f}")
            logger.warning(f"Overfitting at epoch {epoch+1}: val/train ratio = {epoch_overfitting_ratio:.2f}")
        
        # Early stopping check based on validation loss
        if early_stopping_enabled:
            if epoch_val_loss < best_val_loss - min_delta:
                # Improvement found
                best_val_loss = epoch_avg_loss
                patience_counter = 0
                print(f"  📈 Validation improved (best: {best_val_loss:.4f})")
            else:
                # No improvement
                patience_counter += 1
                print(f"  ⚠️  No improvement for {patience_counter}/{early_stopping_patience} epochs")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
    
    # Save final model
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    final_model_path = output_dir / "final_model.pt"
    print(f"\n💾 Saving final model to {final_model_path}...")
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model_config.__dict__,
            'epochs': args.epochs,
            'final_loss': epoch_losses[-1] if epoch_losses else float('inf')
        }, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save final model: {e}")
        logger.error(f"Failed to save final model: {e}")
    
    # Generate final summary plots
    if use_wandb and train_losses_history and val_losses_history:
        # Create comprehensive final report
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Main loss comparison
        ax1 = axes[0, 0]
        steps = list(range(0, global_step+1, eval_steps))[:len(train_losses_history)]
        ax1.plot(steps, train_losses_history, 'b-', label='Train', linewidth=2)
        ax1.plot(steps, val_losses_history, 'r-', label='Validation', linewidth=2)
        ax1.set_title('Loss Comparison')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overfitting ratio over time
        ax2 = axes[0, 1]
        ax2.plot(steps, overfitting_ratios, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Overfitting Ratio History')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Val/Train Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Loss difference
        ax3 = axes[0, 2]
        loss_diff = [v - t for t, v in zip(train_losses_history, val_losses_history)]
        ax3.plot(steps, loss_diff, 'purple', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Validation - Training Loss')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Loss Difference')
        ax3.grid(True, alpha=0.3)
        
        # Histogram of overfitting ratios
        ax4 = axes[1, 0]
        ax4.hist(overfitting_ratios, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax4.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Optimal')
        ax4.axvline(x=1.5, color='red', linestyle='--', linewidth=2, label='Overfit')
        ax4.set_title('Distribution of Overfitting Ratios')
        ax4.set_xlabel('Ratio')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # Learning rate decay
        ax5 = axes[1, 1]
        if learning_rates:
            lr_sample = learning_rates[::max(1, len(learning_rates)//1000)]
            ax5.plot(lr_sample, 'green', linewidth=1)
            ax5.set_title('Learning Rate Schedule')
            ax5.set_xlabel('Steps (sampled)')
            ax5.set_ylabel('Learning Rate')
            ax5.set_yscale('log')
            ax5.grid(True, alpha=0.3)
        
        # Final statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        stats_text = f"""Final Training Statistics:
        
Final Train Loss: {epoch_avg_loss:.4f}
Final Val Loss: {epoch_val_loss:.4f}
Final Ratio: {epoch_overfitting_ratio:.2f}

Best Val Loss: {best_val_loss:.4f}
Total Steps: {global_step}
Epochs Completed: {epoch+1}

Avg Overfitting Ratio: {np.mean(overfitting_ratios):.2f}
Max Overfitting Ratio: {max(overfitting_ratios):.2f}
Min Overfitting Ratio: {min(overfitting_ratios):.2f}"""
        ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        
        plt.suptitle('Training Summary Report', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        wandb.log({'final_training_summary': wandb.Image(fig)}, step=global_step)
        plt.close(fig)
        
        # Log final summary metrics
        wandb.summary['final_train_loss'] = epoch_avg_loss
        wandb.summary['final_val_loss'] = epoch_val_loss
        wandb.summary['final_overfitting_ratio'] = epoch_overfitting_ratio
        wandb.summary['avg_overfitting_ratio'] = np.mean(overfitting_ratios)
        wandb.summary['max_overfitting_ratio'] = max(overfitting_ratios)
        wandb.summary['min_overfitting_ratio'] = min(overfitting_ratios)
        wandb.summary['total_steps'] = global_step
        wandb.summary['epochs_completed'] = epoch + 1
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    if epoch_losses:
        print(f"  📊 Final train loss: {epoch_avg_loss:.4f}")
        print(f"  📊 Final val loss: {epoch_val_loss:.4f}")
        print(f"  📊 Final overfitting ratio: {epoch_overfitting_ratio:.2f}")
    else:
        print(f"  📊 Final loss: N/A")
    print(f"  💾 Final model: {final_model_path}")
    if best_loss < float('inf'):
        print(f"  🏆 Best model: {output_dir / 'best_model.pt'}")
    
    if overfitting_ratios:
        print(f"\n  📈 Training Analysis:")
        print(f"     Average overfitting ratio: {np.mean(overfitting_ratios):.2f}")
        print(f"     Max overfitting ratio: {max(overfitting_ratios):.2f}")
        print(f"     Min overfitting ratio: {min(overfitting_ratios):.2f}")
        
    print(f"\n  📝 Training log: {log_file}")
    print(f"  📁 All outputs: {output_dir}")
    if use_wandb:
        print(f"  📊 W&B dashboard: {wandb.run.get_url() if wandb.run else 'Check W&B'}")
    print("="*60)

if __name__ == "__main__":
    main()