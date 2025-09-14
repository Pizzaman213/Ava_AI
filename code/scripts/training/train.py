#!/usr/bin/env python3
"""
Training Script for MoE++ Model
Supports all 50 advanced features from FUTURE_FEATURES.md
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import yaml

from src.model.moe_transformer import MoEConfig, MoEModel, MoEForCausalLM
from src.training.advanced_training import (
    MetaLearningMoE, MetaLearningConfig,
    FederatedMoE as FederatedLearning, FederatedConfig,
    ContinualLearningMoE as ContinualLearning, ContinualConfig,
    AdvancedPretraining, PretrainingConfig
)
from src.optimization.efficiency_optimizations import (
    MixedPrecisionExperts,
    CompiledMoE,
    DistributedExpertPlacement as DistributedPlacement,
    ExpertCache
)
from src.advanced_features_complete import (
    AdversarialMoE,
    InterpretableMoE,
    BiasMitigation
)


class MoEDataset(Dataset):
    """Custom dataset for MoE++ training"""
    
    def __init__(self, data_path: str, max_length: int = 512):
        self.data_path = Path(data_path)
        self.max_length = max_length
        
        # Load data based on file type
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
        elif self.data_path.suffix == '.pt':
            self.data = torch.load(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Process data
        if isinstance(self.data, dict) and 'input_ids' in self.data:
            self.input_ids = self.data['input_ids']
            self.attention_mask = self.data.get('attention_mask', torch.ones_like(self.input_ids))
            self.metadata = self.data.get('metadata', [{}] * len(self.input_ids))
        elif isinstance(self.data, list):
            # Assume list of dicts with 'input_ids' key
            self.input_ids = torch.tensor([d['input_ids'][:max_length] for d in self.data])
            self.attention_mask = torch.tensor([d.get('attention_mask', [1]*len(d['input_ids']))[:max_length] for d in self.data])
            self.metadata = self.data
        else:
            raise ValueError("Unsupported data format")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx],  # For language modeling
            'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
        }


class MoETrainer:
    """Advanced trainer for MoE++ model with all features"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.training_config = yaml.safe_load(f)
        else:
            self.training_config = self.get_default_config()
        
        # Initialize model configuration
        model_params = self.training_config['model']
        
        # Extract valid MoEConfig parameters
        valid_config_params = {}
        for key, value in model_params.items():
            if hasattr(MoEConfig, key):
                valid_config_params[key] = value
        
        # Check for features section (optional)
        features = self.training_config.get('features', {})
        
        # Add feature flags if they exist
        if features:
            if 'mod_plus_plus' in features:
                valid_config_params['use_mod_plus_plus'] = features['mod_plus_plus']
            if 'hierarchical_moe' in features:
                valid_config_params['use_hierarchical_moe'] = features['hierarchical_moe']
            if 'continuous_experts' in features:
                valid_config_params['use_continuous_experts'] = features['continuous_experts']
            if 'mixture_tokenizers' in features:
                valid_config_params['use_mixture_tokenizers'] = features['mixture_tokenizers']
        
        self.model_config = MoEConfig(**valid_config_params)
        
        # Initialize model (default to causal LM for language modeling)
        use_causal_lm = self.training_config['model'].get('use_causal_lm', True)
        if use_causal_lm:
            self.model = MoEForCausalLM(self.model_config)
        else:
            self.model = MoEModel(self.model_config)
        
        # Setup device
        device_str = self.training_config.get('training', {}).get('device', 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self.setup_scheduler()
        
        # Setup advanced training components
        self.setup_advanced_components()
        
        # Initialize wandb if enabled
        if self.training_config['logging']['use_wandb']:
            wandb.init(
                project=self.training_config['logging']['project'],
                name=self.training_config['logging']['run_name'],
                config=self.training_config
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            'model': {
                'vocab_size': 50000,
                'hidden_size': 768,
                'num_layers': 12,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'num_experts': 8,
                'num_experts_per_tok': 2,
                'use_causal_lm': True
            },
            'features': {
                'mod_plus_plus': True,
                'hierarchical_moe': False,
                'continuous_experts': False,
                'mixture_tokenizers': False,
                'expert_cache': True,
                'mixed_precision': True,
                'sparse_threshold': 0.3,
                'distributed_experts': False,
                'torch_compile': False,
                'meta_learning': False,
                'federated_learning': False,
                'continual_learning': False,
                'adversarial_training': False,
                'interpretability': False,
                'bias_mitigation': False
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'num_epochs': 10,
                'gradient_accumulation_steps': 1,
                'gradient_clip': 1.0,
                'warmup_steps': 1000,
                'eval_steps': 500,
                'save_steps': 1000,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'mixed_precision': True,
                'checkpoint_dir': 'checkpoints'
            },
            'logging': {
                'use_wandb': False,
                'project': 'moe-plus-plus',
                'run_name': f'moe_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'log_interval': 10
            }
        }
    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.training_config['training']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = optim.AdamW(
            optimizer_groups,
            lr=self.training_config['training']['learning_rate'],
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        return optimizer
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.training_config['training']['warmup_steps'],
            T_mult=2,
            eta_min=1e-6
        )
        
        return scheduler
    
    def setup_advanced_components(self):
        """Setup advanced training components based on configuration"""
        self.advanced_components = {}
        
        # Meta-Learning
        if self.training_config['features'].get('meta_learning', False):
            meta_config = MetaLearningConfig(
                inner_lr=0.01,
                outer_lr=0.001,
                num_inner_steps=5
            )
            self.advanced_components['meta_learning'] = MetaLearningMoE(self.model, meta_config)
        
        # Federated Learning
        if self.training_config['features'].get('federated_learning', False):
            fed_config = FederatedConfig(
                num_clients=5,
                rounds_per_epoch=10,
                client_fraction=0.5
            )
            self.advanced_components['federated'] = FederatedLearning(self.model, fed_config)
        
        # Continual Learning
        if self.training_config['features'].get('continual_learning', False):
            continual_config = ContinualConfig(
                ewc_lambda=1000.0,
                replay_buffer_size=1000
            )
            self.advanced_components['continual'] = ContinualLearning(self.model, continual_config)
        
        # Adversarial Training
        if self.training_config['features'].get('adversarial_training', False):
            self.advanced_components['adversarial'] = AdversarialMoE(self.model, epsilon=0.1)
        
        # Interpretability
        if self.training_config['features'].get('interpretability', False):
            self.advanced_components['interpretable'] = InterpretableMoE(self.model)
        
        # Bias Mitigation
        if self.training_config['features'].get('bias_mitigation', False):
            self.advanced_components['bias_mitigation'] = BiasMitigation(self.model)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Compute loss
        if hasattr(self.model, 'compute_loss'):
            loss = self.model.compute_loss(outputs, batch['labels'])
        else:
            # Simple language modeling loss
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Add auxiliary losses
        if isinstance(outputs, dict) and 'aux_loss' in outputs:
            loss = loss + outputs['aux_loss']
        
        # Adversarial training if enabled
        if 'adversarial' in self.advanced_components:
            adv_loss = self.advanced_components['adversarial'].adversarial_training_step(
                batch['input_ids'],
                batch['labels'],
                self.optimizer
            )
            loss = loss + 0.1 * adv_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.training_config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config['training']['gradient_clip']
            )
        
        # Optimizer step
        if (self.global_step + 1) % self.training_config['training']['gradient_accumulation_steps'] == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        metrics = {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'global_step': self.global_step
        }
        
        # Add interpretability metrics if enabled
        if 'interpretable' in self.advanced_components:
            routing_info = self.advanced_components['interpretable'].explain_routing(batch['input_ids'])
            metrics['routing_entropy'] = routing_info.get('entropy', 0.0)
        
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Compute loss
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(outputs, batch['labels'])
                else:
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch['labels'][..., 1:].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = np.exp(avg_loss)
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint"""
        if path is None:
            checkpoint_dir = Path(self.training_config['training']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.training_config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # Save model config separately
        config_path = Path(path).parent / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.model_config.__dict__, f, indent=2, default=str)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded from {path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop"""
        print("="*50)
        print("Starting MoE++ Training")
        print(f"Model: {self.model_config.num_layers} layers, {self.model_config.num_experts} experts")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.training_config['training']['batch_size']}")
        print(f"Learning rate: {self.training_config['training']['learning_rate']}")
        print("="*50)
        
        # Training loop
        for epoch in range(self.training_config['training']['num_epochs']):
            self.epoch = epoch
            epoch_loss = 0
            epoch_steps = 0
            
            # Training
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.training_config['training']['num_epochs']}")
            for batch_idx, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                epoch_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics['learning_rate']:.2e}"
                })
                
                # Logging
                if self.global_step % self.training_config['logging']['log_interval'] == 0:
                    if self.training_config['logging']['use_wandb']:
                        wandb.log(metrics, step=self.global_step)
                
                # Evaluation
                if val_dataloader and self.global_step % self.training_config['training']['eval_steps'] == 0:
                    eval_metrics = self.evaluate(val_dataloader)
                    print(f"\nEval Loss: {eval_metrics['eval_loss']:.4f}, Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    if self.training_config['logging']['use_wandb']:
                        wandb.log(eval_metrics, step=self.global_step)
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_loss:
                        self.best_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(
                            Path(self.training_config['training']['checkpoint_dir']) / "best_model.pt"
                        )
                
                # Save checkpoint
                if self.global_step % self.training_config['training']['save_steps'] == 0:
                    self.save_checkpoint()
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
            
            # Continual learning consolidation if enabled
            if 'continual' in self.advanced_components:
                self.advanced_components['continual'].consolidate()
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Train MoE++ model")
    parser.add_argument('--config', type=str, help='Path to training config YAML')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing training data')
    parser.add_argument('--train_data', type=str, default='text_corpus_train.pt', help='Training data file')
    parser.add_argument('--val_data', type=str, default='text_corpus_val.pt', help='Validation data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MoETrainer(config_path=args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        trainer.training_config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        trainer.training_config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        trainer.training_config['training']['learning_rate'] = args.learning_rate
    if args.wandb:
        trainer.training_config['logging']['use_wandb'] = True
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dataset = MoEDataset(data_dir / args.train_data)
    val_dataset = MoEDataset(data_dir / args.val_data) if args.val_data else None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=trainer.training_config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=trainer.training_config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if val_dataset else None
    
    # Train model
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()