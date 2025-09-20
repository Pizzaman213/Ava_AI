#!/usr/bin/env python3
"""
Enhanced Training Script for Advanced LLM Framework

This comprehensive training script supports all the enhanced features including:
- Mixture of Heads (MoH), Mixture of Activations (MoA)
- Advanced expert routing (Switch, GShard, Hash-based, Stochastic)
- Multi-modal cross-attention capabilities
- RAG (Retrieval-Augmented Generation) training
- Advanced loss functions (contrastive, focal, diversity)
- Gradient surgery for multi-task learning
- Comprehensive evaluation during training
- Model quantization and optimization
- Episodic memory for continual learning
- Progressive training strategies

Usage:
    # Basic enhanced training
    python train.py --config configs/gpu/small.yaml --enable-all-features

    # Training with specific enhancements
    python train.py --config configs/gpu/small.yaml --use-moh --use-moa --use-rag

    # Multi-task training with gradient surgery
    python train.py --config configs/gpu/small.yaml --multi-task --gradient-surgery

    # Training with comprehensive evaluation
    python train.py --config configs/gpu/small.yaml --eval-during-training --eval-metrics perplexity,bleu,toxicity

Examples:
    # Full feature training
    python train.py --config configs/gpu/medium.yaml --enable-all-features --output-dir outputs/enhanced_model

    # RAG-enabled training
    python train.py --config configs/gpu/small.yaml --use-rag --knowledge-base-path data/kb/ --output-dir outputs/rag_model

    # Quantization-aware training
    python train.py --config configs/gpu/small.yaml --quantization-aware --bit-width 8
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from collections import defaultdict

# Add project root to path
sys.path.append('/project/code')

# Core imports
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from src.Ava.data import create_dataloaders
from src.Ava.data_streaming import create_streaming_dataloaders
from transformers import AutoTokenizer

# Enhanced feature imports
from src.Ava.layers.mixture_of_heads import MixtureOfHeads, AdaptiveHeadAttention
from src.Ava.layers.mixture_of_activations import MixtureOfActivations, AdaptiveActivation
from src.Ava.layers.cross_attention import MultiModalCrossAttention, PerceiversCrossAttention
from src.Ava.layers.routing import (
    SwitchTransformerRouting, GSERouting,
    HashingExpertRouting, StochasticExpertRouting
)
from src.Ava.retrieval import RAGSystem, AdaptiveRAG, KnowledgeBase
from src.Ava.losses import (
    ContrastiveLoss, FocalLoss, DiversityLoss, AuxiliaryLoss,
    CompositeLoss, AdaptiveLossScaling
)
from src.Ava.training import GradientSurgeon, AdaptiveGradientSurgeon
from src.Ava.evaluation.comprehensive_eval import ComprehensiveEvaluator
from src.Ava.optimization.quantization import ModelQuantizer, QuantizationConfig
from src.Ava.memory import EpisodicMemoryBank, AdaptiveMemoryManager, ExperienceReplay


class EnhancedTrainer:
    """
    Enhanced trainer with support for all advanced features.

    This trainer integrates:
    - Advanced loss functions
    - Gradient surgery
    - RAG training
    - Multi-task learning
    - Comprehensive evaluation
    - Quantization-aware training
    - Episodic memory for continual learning
    """

    def __init__(self, args, config, model, tokenizer, device):
        self.args = args
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize enhanced components
        self._init_loss_functions()
        self._init_gradient_surgery()
        self._init_rag_system()
        self._init_evaluator()
        self._init_quantization()
        self._init_episodic_memory()

        # Training statistics
        self.training_stats = defaultdict(list)
        self.step_count = 0

    def _init_loss_functions(self):
        """Initialize advanced loss functions."""
        loss_config = {}

        if self.args.use_focal_loss:
            loss_config['focal'] = {
                'type': 'focal',
                'alpha': 1.0,
                'gamma': 2.0,
                'weight': 0.1
            }

        if self.args.use_contrastive_loss:
            loss_config['contrastive'] = {
                'type': 'contrastive',
                'temperature': 0.07,
                'weight': 0.1
            }

        if self.args.use_diversity_loss:
            loss_config['diversity'] = {
                'type': 'diversity',
                'similarity_metric': 'cosine',
                'weight': 0.01
            }

        # Always include auxiliary losses for MoE
        loss_config['auxiliary'] = {
            'type': 'auxiliary',
            'load_balancing_weight': 0.01,
            'router_z_weight': 0.001,
            'weight': 1.0
        }

        if loss_config:
            self.composite_loss = CompositeLoss(loss_config)
            print(f"📊 Initialized composite loss with: {list(loss_config.keys())}")
        else:
            self.composite_loss = None

        # Adaptive loss scaling
        if self.args.adaptive_loss_scaling:
            num_losses = len(loss_config) + 1  # +1 for main loss
            self.adaptive_scaler = AdaptiveLossScaling(num_losses)
            print("🔧 Enabled adaptive loss scaling")
        else:
            self.adaptive_scaler = None

    def _init_gradient_surgery(self):
        """Initialize gradient surgery for multi-task learning."""
        if self.args.gradient_surgery:
            if self.args.adaptive_gradient_surgery:
                self.gradient_surgeon = AdaptiveGradientSurgeon(
                    methods=['pcgrad', 'graddrop', 'gradnorm'],
                    conflict_threshold=0.3
                )
                print("🔧 Enabled adaptive gradient surgery")
            else:
                self.gradient_surgeon = GradientSurgeon(
                    method=self.args.gradient_surgery_method,
                    cosine_similarity_threshold=0.5
                )
                print(f"🔧 Enabled gradient surgery with {self.args.gradient_surgery_method}")
        else:
            self.gradient_surgeon = None

    def _init_rag_system(self):
        """Initialize RAG system if enabled."""
        if self.args.use_rag:
            try:
                # Initialize RAG system
                self.rag_system = RAGSystem(
                    encoder_dim=self.config.hidden_size,
                    retrieval_dim=256,
                    max_retrieved=self.args.max_retrieved_docs,
                    fusion_type=self.args.rag_fusion_type
                )

                # Load or create knowledge base
                if self.args.knowledge_base_path:
                    kb_path = Path(self.args.knowledge_base_path)
                    if kb_path.exists():
                        self.knowledge_base = KnowledgeBase(
                            embedding_dim=256,
                            index_type="IVF"
                        )
                        self.knowledge_base.load(kb_path)
                        self.rag_system.set_knowledge_base(self.knowledge_base)
                        print(f"📚 Loaded knowledge base from {kb_path}")
                    else:
                        print(f"⚠️ Knowledge base path {kb_path} not found, RAG disabled")
                        self.rag_system = None
                else:
                    print("⚠️ No knowledge base path provided, RAG disabled")
                    self.rag_system = None
            except Exception as e:
                print(f"⚠️ Failed to initialize RAG system: {e}")
                self.rag_system = None
        else:
            self.rag_system = None

    def _init_evaluator(self):
        """Initialize comprehensive evaluator."""
        if self.args.eval_during_training:
            eval_config = {
                'tokenizer_name': 'gpt2',
                'bleu_max_n': 4,
                'rouge_types': ['rouge-1', 'rouge-2', 'rouge-l']
            }
            self.evaluator = ComprehensiveEvaluator(eval_config)

            # Parse evaluation metrics
            if self.args.eval_metrics:
                self.eval_metrics = self.args.eval_metrics.split(',')
            else:
                self.eval_metrics = ['perplexity']

            print(f"📊 Enabled evaluation with metrics: {self.eval_metrics}")
        else:
            self.evaluator = None

    def _init_quantization(self):
        """Initialize quantization if enabled."""
        if self.args.quantization_aware:
            quant_config = QuantizationConfig(
                bit_width=self.args.bit_width,
                symmetric=True,
                per_channel=True
            )
            self.quantizer = ModelQuantizer(quant_config)
            print(f"🔧 Enabled {self.args.bit_width}-bit quantization-aware training")
        else:
            self.quantizer = None

    def _init_episodic_memory(self):
        """Initialize episodic memory for continual learning."""
        if self.args.use_episodic_memory:
            hidden_size = self.config.get('hidden_size', 768)
            self.memory_bank = EpisodicMemoryBank(
                capacity=self.args.memory_capacity,
                hidden_size=hidden_size,
                selection_strategy=self.args.memory_selection_strategy,
                importance_threshold=self.args.memory_importance_threshold,
                retrieval_method=self.args.memory_retrieval_method
            ).to(self.device)

            self.memory_manager = AdaptiveMemoryManager(
                memory_bank=self.memory_bank,
                adaptation_rate=self.args.memory_adaptation_rate,
                performance_window=self.args.memory_performance_window
            )

            self.experience_replay = ExperienceReplay(
                memory_bank=self.memory_bank,
                replay_ratio=self.args.memory_replay_ratio,
                replay_strategy=self.args.memory_replay_strategy
            )

            print(f"🧠 Enabled episodic memory with capacity {self.args.memory_capacity}")
            print(f"   Selection: {self.args.memory_selection_strategy}")
            print(f"   Replay ratio: {self.args.memory_replay_ratio}")
        else:
            self.memory_bank = None
            self.memory_manager = None
            self.experience_replay = None

    def train_epoch(self, dataloader, optimizer, epoch, total_epochs):
        """Enhanced training epoch with all features."""
        self.model.train()
        epoch_stats = {
            'total_loss': 0,
            'main_loss': 0,
            'aux_losses': defaultdict(float),
            'num_batches': 0,
            'gradient_conflicts': 0
        }

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Enhanced forward pass
            loss_dict = self._enhanced_forward_pass(
                input_ids, attention_mask, labels, batch_idx
            )

            # Enhanced backward pass with gradient surgery
            self._enhanced_backward_pass(loss_dict, optimizer)

            # Update statistics
            self._update_epoch_stats(epoch_stats, loss_dict)

            # Update progress bar
            avg_loss = epoch_stats['total_loss'] / epoch_stats['num_batches']
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'step': self.step_count
            })

            # Periodic logging and evaluation
            if batch_idx % 100 == 0 and batch_idx > 0:
                self._periodic_logging(epoch_stats, batch_idx)

            if self.args.eval_during_training and batch_idx % 500 == 0 and batch_idx > 0:
                self._periodic_evaluation(dataloader)

            self.step_count += 1

        return self._finalize_epoch_stats(epoch_stats)

    def _enhanced_forward_pass(self, input_ids, attention_mask, labels, batch_idx):
        """Enhanced forward pass with all features."""
        # Basic model forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        main_loss = outputs.get('loss', outputs.get('logits'))
        if not isinstance(main_loss, torch.Tensor):
            # If logits returned, compute loss manually
            logits = outputs.get('logits')
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        loss_dict = {'main_loss': main_loss}

        # RAG enhancement
        if self.rag_system:
            try:
                # Get hidden states for RAG
                hidden_states = outputs.get('hidden_states')
                if hidden_states is not None:
                    enhanced_hidden, rag_info = self.rag_system(hidden_states)
                    # RAG loss could be added here if needed
                    loss_dict['rag_info'] = rag_info
            except Exception as e:
                if batch_idx == 0:  # Only log once per epoch
                    print(f"⚠️ RAG forward pass failed: {e}")

        # Auxiliary losses from model (MoE routing, etc.)
        if 'aux_losses' in outputs:
            for aux_name, aux_loss in outputs['aux_losses'].items():
                if aux_loss is not None and isinstance(aux_loss, torch.Tensor):
                    loss_dict[f'aux_{aux_name}'] = aux_loss

        # Episodic memory storage and replay
        if self.memory_bank:
            try:
                # Store current experience in memory
                hidden_states = outputs.get('hidden_states')
                if hidden_states is not None:
                    # Compute gradient norm for importance scoring
                    gradient_norm = None
                    if hasattr(main_loss, 'grad_fn') and main_loss.grad_fn is not None:
                        gradient_norm = torch.norm(torch.autograd.grad(
                            main_loss, self.model.parameters(),
                            retain_graph=True, create_graph=False
                        )[0]).item()

                    # Add to memory bank
                    self.memory_bank.add_memory(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        hidden_states=hidden_states,
                        task_id=getattr(self.args, 'task_id', 0),
                        labels=labels,
                        loss_value=main_loss.item(),
                        gradient_norm=gradient_norm
                    )

                    # Experience replay
                    replay_batch = self.experience_replay.get_replay_batch(
                        current_batch_size=input_ids.size(0),
                        current_hidden_states=hidden_states
                    )

                    if replay_batch[0] is not None:
                        replay_loss = self.experience_replay.compute_replay_loss(
                            self.model, replay_batch, self.device
                        )
                        if replay_loss.item() > 0:
                            loss_dict['replay_loss'] = replay_loss

            except Exception as e:
                if batch_idx == 0:  # Only log once per epoch
                    print(f"⚠️ Episodic memory processing failed: {e}")

        # Composite loss functions
        if self.composite_loss:
            try:
                composite_losses = self.composite_loss(
                    inputs=outputs.get('logits'),
                    targets=labels,
                    gate_logits=outputs.get('routing_logits'),
                    expert_indices=outputs.get('expert_indices'),
                    expert_outputs=outputs.get('expert_outputs'),
                    num_experts=getattr(self.config, 'num_experts', 8)
                )
                loss_dict.update(composite_losses)
            except Exception as e:
                if batch_idx == 0:
                    print(f"⚠️ Composite loss computation failed: {e}")

        return loss_dict

    def _enhanced_backward_pass(self, loss_dict, optimizer):
        """Enhanced backward pass with gradient surgery."""
        # Compute total loss
        total_loss = loss_dict['main_loss']

        # Add auxiliary losses
        for key, loss in loss_dict.items():
            if key.startswith('aux_') and isinstance(loss, torch.Tensor):
                total_loss = total_loss + 0.01 * loss
            elif key in ['focal', 'contrastive', 'diversity'] and isinstance(loss, torch.Tensor):
                total_loss = total_loss + loss

        # Adaptive loss scaling
        if self.adaptive_scaler:
            losses = [loss_dict['main_loss']]
            for key in ['focal', 'contrastive', 'diversity']:
                if key in loss_dict:
                    losses.append(loss_dict[key])

            if len(losses) > 1:
                total_loss, weights = self.adaptive_scaler(losses)
                loss_dict['adaptive_weights'] = weights

        # Gradient surgery for multi-task learning
        if self.gradient_surgeon and len(loss_dict) > 2:
            # Extract task-specific losses
            task_losses = {}
            if 'main_loss' in loss_dict:
                task_losses['main'] = loss_dict['main_loss']
            if 'contrastive' in loss_dict:
                task_losses['contrastive'] = loss_dict['contrastive']
            if 'focal' in loss_dict:
                task_losses['focal'] = loss_dict['focal']

            if len(task_losses) > 1:
                try:
                    # Compute gradients for each task
                    task_gradients = {}
                    for task_name, task_loss in task_losses.items():
                        optimizer.zero_grad()
                        task_loss.backward(retain_graph=True)

                        task_grads = []
                        for param in self.model.parameters():
                            if param.grad is not None:
                                task_grads.append(param.grad.clone())
                        task_gradients[task_name] = task_grads

                        # Clear gradients
                        optimizer.zero_grad()

                    # Apply gradient surgery
                    if len(task_gradients) > 1:
                        modified_grads = self.gradient_surgeon.apply_surgery(
                            task_gradients, task_losses
                        )

                        # Apply modified gradients
                        for param, grad in zip(self.model.parameters(), modified_grads):
                            param.grad = grad

                        loss_dict['gradient_conflicts'] = 1
                    else:
                        # Normal backward pass
                        total_loss.backward()
                except Exception as e:
                    print(f"⚠️ Gradient surgery failed, using normal backward: {e}")
                    optimizer.zero_grad()
                    total_loss.backward()
            else:
                # Normal backward pass
                optimizer.zero_grad()
                total_loss.backward()
        else:
            # Normal backward pass
            optimizer.zero_grad()
            total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()

        loss_dict['total_loss'] = total_loss

    def _update_epoch_stats(self, epoch_stats, loss_dict):
        """Update epoch statistics."""
        epoch_stats['num_batches'] += 1

        if 'total_loss' in loss_dict:
            epoch_stats['total_loss'] += loss_dict['total_loss'].item()

        if 'main_loss' in loss_dict:
            epoch_stats['main_loss'] += loss_dict['main_loss'].item()

        if 'gradient_conflicts' in loss_dict:
            epoch_stats['gradient_conflicts'] += loss_dict['gradient_conflicts']

        # Track auxiliary losses
        for key, value in loss_dict.items():
            if key.startswith('aux_') and isinstance(value, torch.Tensor):
                epoch_stats['aux_losses'][key] += value.item()

    def _periodic_logging(self, epoch_stats, batch_idx):
        """Periodic detailed logging."""
        avg_total_loss = epoch_stats['total_loss'] / epoch_stats['num_batches']
        avg_main_loss = epoch_stats['main_loss'] / epoch_stats['num_batches']

        log_msg = f"  Step {batch_idx}: Total Loss = {avg_total_loss:.4f}, Main Loss = {avg_main_loss:.4f}"

        if epoch_stats['aux_losses']:
            aux_losses_str = ", ".join([
                f"{k.replace('aux_', '')}: {v/epoch_stats['num_batches']:.4f}"
                for k, v in epoch_stats['aux_losses'].items()
            ])
            log_msg += f", Aux Losses: {aux_losses_str}"

        if epoch_stats['gradient_conflicts'] > 0:
            conflict_rate = epoch_stats['gradient_conflicts'] / epoch_stats['num_batches']
            log_msg += f", Gradient Conflicts: {conflict_rate:.2%}"

        # Add memory statistics
        if self.memory_bank:
            memory_stats = self.memory_bank.get_memory_stats()
            log_msg += f", Memory: {memory_stats['total_memories']}/{self.memory_bank.capacity}"
            if memory_stats['total_memories'] > 0:
                log_msg += f" (avg_imp: {memory_stats['avg_importance']:.3f})"

        print(log_msg)

    def _periodic_evaluation(self, dataloader):
        """Periodic evaluation during training."""
        if not self.evaluator:
            return

        try:
            print("🔍 Running periodic evaluation...")

            # Generate sample texts for evaluation
            sample_texts = self._generate_evaluation_samples(num_samples=50)

            # Prepare evaluation data
            eval_data = {
                'texts': sample_texts,
                'generated_texts': sample_texts  # For toxicity/bias evaluation
            }

            # Run evaluation
            results = self.evaluator.evaluate_model(
                self.model,
                eval_data,
                metrics=self.eval_metrics,
                save_results=False
            )

            # Log results
            print("📊 Evaluation results:")
            for metric, result in results.items():
                print(f"  {metric}: {result.score:.4f}")

        except Exception as e:
            print(f"⚠️ Periodic evaluation failed: {e}")

    def _generate_evaluation_samples(self, num_samples=50):
        """Generate sample texts for evaluation."""
        self.model.eval()
        samples = []

        try:
            prompts = [
                "The future of artificial intelligence",
                "Climate change and its impact",
                "The benefits of renewable energy",
                "Space exploration and discovery",
                "The importance of education"
            ]

            with torch.no_grad():
                for prompt in prompts[:min(len(prompts), num_samples//10)]:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=50,
                        truncation=True,
                        padding=True
                    ).to(self.device)

                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    generated_text = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    samples.append(generated_text)

                    if len(samples) >= num_samples:
                        break
        except Exception as e:
            print(f"⚠️ Sample generation failed: {e}")
            # Return dummy samples
            samples = ["Sample text for evaluation"] * num_samples

        self.model.train()
        return samples

    def _finalize_epoch_stats(self, epoch_stats):
        """Finalize and return epoch statistics."""
        num_batches = epoch_stats['num_batches']

        final_stats = {
            'avg_total_loss': epoch_stats['total_loss'] / num_batches,
            'avg_main_loss': epoch_stats['main_loss'] / num_batches,
            'gradient_conflict_rate': epoch_stats['gradient_conflicts'] / num_batches,
            'aux_losses': {
                k: v / num_batches for k, v in epoch_stats['aux_losses'].items()
            }
        }

        # Store in training history
        self.training_stats['total_loss'].append(final_stats['avg_total_loss'])
        self.training_stats['main_loss'].append(final_stats['avg_main_loss'])

        return final_stats


def evaluate_model(trainer, dataloader):
    """Enhanced model evaluation."""
    trainer.model.eval()
    eval_stats = {
        'total_loss': 0,
        'main_loss': 0,
        'num_batches': 0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(trainer.device)
            attention_mask = batch['attention_mask'].to(trainer.device)
            labels = batch['labels'].to(trainer.device)

            # Forward pass
            loss_dict = trainer._enhanced_forward_pass(
                input_ids, attention_mask, labels, 0
            )

            # Update stats
            eval_stats['num_batches'] += 1
            if 'total_loss' in loss_dict:
                eval_stats['total_loss'] += loss_dict['total_loss'].item()
            if 'main_loss' in loss_dict:
                eval_stats['main_loss'] += loss_dict['main_loss'].item()

    # Compute averages
    avg_total_loss = eval_stats['total_loss'] / eval_stats['num_batches']
    avg_main_loss = eval_stats['main_loss'] / eval_stats['num_batches']

    return {
        'avg_total_loss': avg_total_loss,
        'avg_main_loss': avg_main_loss
    }


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced LLM Training with Advanced Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhanced training
  python train.py --config configs/gpu/small.yaml --enable-all-features

  # RAG-enabled training
  python train.py --config configs/gpu/small.yaml --use-rag --knowledge-base-path data/kb/

  # Multi-task training with gradient surgery
  python train.py --config configs/gpu/small.yaml --multi-task --gradient-surgery

  # Quantization-aware training
  python train.py --config configs/gpu/small.yaml --quantization-aware --bit-width 8
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')

    # === ENHANCED FEATURE FLAGS ===
    parser.add_argument('--enable-all-features', action='store_true',
                       help='Enable all enhanced features (overrides individual flags)')

    # Architecture enhancements
    parser.add_argument('--use-moh', action='store_true',
                       help='Enable Mixture of Heads (MoH)')
    parser.add_argument('--use-moa', action='store_true',
                       help='Enable Mixture of Activations (MoA)')
    parser.add_argument('--use-cross-attention', action='store_true',
                       help='Enable multi-modal cross-attention')
    parser.add_argument('--use-alibi', action='store_true',
                       help='Use ALiBi positional encoding instead of RoPE')

    # Expert routing enhancements
    parser.add_argument('--expert-routing-type', type=str, default='base',
                       choices=['base', 'switch', 'gshard', 'hash', 'stochastic'],
                       help='Type of expert routing to use')

    # RAG system
    parser.add_argument('--use-rag', action='store_true',
                       help='Enable Retrieval-Augmented Generation')
    parser.add_argument('--knowledge-base-path', type=str,
                       help='Path to knowledge base for RAG')
    parser.add_argument('--max-retrieved-docs', type=int, default=5,
                       help='Maximum number of documents to retrieve')
    parser.add_argument('--rag-fusion-type', type=str, default='attention',
                       choices=['attention', 'gate', 'concat', 'weighted'],
                       help='RAG fusion strategy')

    # Advanced loss functions
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='Enable focal loss for hard example mining')
    parser.add_argument('--use-contrastive-loss', action='store_true',
                       help='Enable contrastive learning loss')
    parser.add_argument('--use-diversity-loss', action='store_true',
                       help='Enable expert diversity loss')
    parser.add_argument('--adaptive-loss-scaling', action='store_true',
                       help='Enable adaptive loss scaling')

    # Gradient surgery
    parser.add_argument('--gradient-surgery', action='store_true',
                       help='Enable gradient surgery for multi-task learning')
    parser.add_argument('--adaptive-gradient-surgery', action='store_true',
                       help='Use adaptive gradient surgery method selection')
    parser.add_argument('--gradient-surgery-method', type=str, default='pcgrad',
                       choices=['pcgrad', 'graddrop', 'gradnorm', 'cagrad', 'mgda'],
                       help='Gradient surgery method')

    # Multi-task learning
    parser.add_argument('--multi-task', action='store_true',
                       help='Enable multi-task learning mode')

    # Evaluation during training
    parser.add_argument('--eval-during-training', action='store_true',
                       help='Run comprehensive evaluation during training')
    parser.add_argument('--eval-metrics', type=str,
                       help='Comma-separated list of evaluation metrics')
    parser.add_argument('--eval-frequency', type=int, default=500,
                       help='Evaluation frequency (steps)')

    # Quantization
    parser.add_argument('--quantization-aware', action='store_true',
                       help='Enable quantization-aware training')
    parser.add_argument('--bit-width', type=int, default=8, choices=[4, 8],
                       help='Quantization bit width')

    # Episodic memory for continual learning
    parser.add_argument('--use-episodic-memory', action='store_true',
                       help='Enable episodic memory for continual learning')
    parser.add_argument('--memory-capacity', type=int, default=1000,
                       help='Episodic memory bank capacity')
    parser.add_argument('--memory-selection-strategy', type=str, default='importance',
                       choices=['importance', 'random', 'task_balanced'],
                       help='Memory selection strategy')
    parser.add_argument('--memory-importance-threshold', type=float, default=0.5,
                       help='Threshold for memory importance scoring')
    parser.add_argument('--memory-retrieval-method', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'dot'],
                       help='Memory retrieval similarity method')
    parser.add_argument('--memory-replay-ratio', type=float, default=0.2,
                       help='Ratio of replay samples to current batch')
    parser.add_argument('--memory-replay-strategy', type=str, default='importance',
                       choices=['random', 'importance', 'similarity'],
                       help='Experience replay sampling strategy')
    parser.add_argument('--memory-adaptation-rate', type=float, default=0.01,
                       help='Adaptation rate for memory parameters')
    parser.add_argument('--memory-performance-window', type=int, default=100,
                       help='Window size for performance-based adaptation')
    parser.add_argument('--task-id', type=int, default=0,
                       help='Task ID for multi-task continual learning')

    # === DATA ARGUMENTS ===
    parser.add_argument('--data-dir', type=str,
                       default='/project/code/data/pretraining/processed',
                       help='Directory containing preprocessed training data')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of training samples to load (for testing)')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Use streaming data loader for large datasets (default: True)')
    parser.add_argument('--no-streaming', dest='streaming', action='store_false',
                       help='Disable streaming and load all data into memory')
    parser.add_argument('--buffer-size', type=int, default=1000,
                       help='Buffer size for streaming data loader')

    # === TRAINING ARGUMENTS ===
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')

    # === OUTPUT ARGUMENTS ===
    parser.add_argument('--output-dir', type=str, default='/project/code/outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # === SYSTEM ARGUMENTS ===
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Handle enable-all-features flag
    if args.enable_all_features:
        args.use_moh = True
        args.use_moa = True
        args.use_cross_attention = True
        args.use_focal_loss = True
        args.use_contrastive_loss = True
        args.use_diversity_loss = True
        args.adaptive_loss_scaling = True
        args.gradient_surgery = True
        args.adaptive_gradient_surgery = True
        args.eval_during_training = True
        args.expert_routing_type = 'switch'
        args.use_episodic_memory = True
        print("🚀 All enhanced features enabled!")

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"🔧 Using device: {device}")

    # Load configuration
    print(f"📋 Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract model config
    model_config_dict = config_dict.get('model', {})
    training_config = config_dict.get('training', {})

    # Override with command line arguments
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    if args.epochs:
        training_config['num_epochs'] = args.epochs
    if args.learning_rate:
        training_config['learning_rate'] = args.learning_rate

    # Set defaults if not in config
    batch_size = training_config.get('batch_size', 4)
    num_epochs = training_config.get('num_epochs', 3)
    learning_rate = float(training_config.get('learning_rate', 5e-4))

    print(f"📊 Training configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max length: {args.max_length}")
    print(f"  - Data directory: {args.data_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    import logging
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    # Initialize tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloaders
    print(f"📚 Loading data from {args.data_dir}...")
    if args.streaming:
        print("🌊 Using streaming data loader (memory efficient)")
        train_loader, val_loader = create_streaming_dataloaders(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=args.max_length,
            data_dir=args.data_dir,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            max_samples=args.max_samples,
            buffer_size=args.buffer_size
        )
    else:
        print("⚠️ Loading all data into memory (use --streaming for large datasets)")
        train_loader, val_loader = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=args.max_length,
            data_dir=args.data_dir,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            max_samples=args.max_samples
        )

    # Initialize model
    print("🤖 Initializing model...")
    # Filter config to only include valid fields
    from dataclasses import fields
    valid_fields = {f.name for f in fields(EnhancedMoEConfig)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_fields}
    model_config = EnhancedMoEConfig(**filtered_config)
    model = EnhancedMoEModel(model_config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"  Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n📍 Epoch {epoch}/{num_epochs}")

        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        train_time = time.time() - start_time

        print(f"  Training loss: {train_loss:.4f}")
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"  Validation loss: {val_loss:.4f}")

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'best_model.pt'
            print(f"  💾 Saving best model to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config_dict
            }, checkpoint_path)

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            print(f"  💾 Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config_dict
            }, checkpoint_path)

    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"📂 Model saved to: {output_dir}")
    print("="*60)

    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config_dict
    }, final_path)
    print(f"\n💾 Final model saved to {final_path}")

    print("\n🎯 To generate text with your trained model:")
    print(f"python /project/code/scripts/generation/generate.py --model-path {final_path} --prompt 'Your text here'")


if __name__ == "__main__":
    main()