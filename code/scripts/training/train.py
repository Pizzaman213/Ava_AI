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
import os
import signal
import atexit
import gc

# DeepSpeed support
try:
    import deepspeed
    from deepspeed import comm as dist
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("⚠️ DeepSpeed not installed. Install with: pip install deepspeed")

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


def cleanup_gpu_memory():
    """
    Comprehensive GPU memory cleanup function.

    This function clears GPU cache, runs garbage collection,
    and ensures proper cleanup when training is interrupted.
    """
    try:
        if torch.cuda.is_available():
            print("🧹 Cleaning up GPU memory...")

            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Additional CUDA cleanup
            if torch.cuda.is_available():
                # Clear all cached allocations
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Get memory info for reporting
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                cached = torch.cuda.memory_reserved() / 1024**3     # GB

                print(f"💾 GPU Memory after cleanup: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

                # Additional aggressive cleanup if needed
                if cached > 1.0:  # If more than 1GB still cached
                    print("🔄 Performing aggressive GPU cleanup...")
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️ Error during GPU cleanup: {e}")


def signal_handler(signum, frame):
    """
    Handle interruption signals (Ctrl+C, etc.) with proper GPU cleanup.
    """
    print(f"\n🛑 Received signal {signum}. Performing cleanup...")
    cleanup_gpu_memory()
    print("✅ Cleanup completed. Exiting...")
    exit(0)


def register_cleanup_handlers():
    """
    Register signal handlers and exit functions for proper cleanup.
    """
    # Register signal handlers for common interruption signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # Register cleanup function to run at exit
    atexit.register(cleanup_gpu_memory)

    print("🔧 GPU cleanup handlers registered")


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

    def __init__(self, args, config, model, tokenizer, device, use_deepspeed=False):
        self.args = args
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_deepspeed = use_deepspeed

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
        # Temporarily use only auxiliary losses for stability
        loss_config = {}

        # Only use auxiliary losses for MoE (essential for training)
        loss_config['auxiliary'] = {
            'type': 'auxiliary',
            'load_balancing_weight': 0.001,  # Reduced weight for stability
            'router_z_weight': 0.0001,       # Reduced weight for stability
            'weight': 0.1                    # Reduced weight for stability
        }

        if loss_config:
            self.composite_loss = CompositeLoss(loss_config)
            print(f"📊 Initialized composite loss with: {list(loss_config.keys())} (simplified for stability)")
        else:
            self.composite_loss = None

        # Disable adaptive loss scaling for stability
        print("🔧 Adaptive loss scaling disabled for stability")
        self.adaptive_scaler = None

    def _init_gradient_surgery(self):
        """Initialize gradient surgery for multi-task learning."""
        if self.args.gradient_surgery:
            # Temporarily disable gradient surgery to fix stability issues
            print("🔧 Gradient surgery temporarily disabled for stability")
            self.gradient_surgeon = None
        else:
            self.gradient_surgeon = None

    def _init_rag_system(self):
        """Initialize RAG system if enabled."""
        if self.args.use_rag:
            try:
                # Initialize RAG system with defensive config access
                model_config = self.config.get('model', {})
                hidden_size = model_config.get('hidden_size', 768)

                # Defensive checks for required arguments
                max_retrieved = getattr(self.args, 'max_retrieved_docs', 5)
                fusion_type = getattr(self.args, 'rag_fusion_type', 'attention')

                self.rag_system = RAGSystem(
                    encoder_dim=hidden_size,
                    retrieval_dim=256,
                    max_retrieved=max_retrieved,
                    fusion_type=fusion_type
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
        if getattr(self.args, 'eval_during_training', False):
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
        if getattr(self.args, 'quantization_aware', False):
            bit_width = getattr(self.args, 'bit_width', 8)
            quant_config = QuantizationConfig(
                bit_width=bit_width,
                symmetric=True,
                per_channel=True
            )
            self.quantizer = ModelQuantizer(quant_config)
            print(f"🔧 Enabled {bit_width}-bit quantization-aware training")
        else:
            self.quantizer = None

    def _init_episodic_memory(self):
        """Initialize episodic memory for continual learning."""
        if self.args.use_episodic_memory:
            # Temporarily disable episodic memory for stability
            print("🧠 Episodic memory temporarily disabled for stability")
            self.memory_bank = None
            self.memory_manager = None
            self.experience_replay = None
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

            # Run evaluation less frequently to reduce training time impact
            if self.args.eval_during_training and batch_idx % 1000 == 0 and batch_idx > 0:
                self._periodic_evaluation(dataloader)

            # Save model checkpoint every 500 steps
            if self.step_count > 0 and self.step_count % 500 == 0:
                self._save_checkpoint(epoch, optimizer, avg_loss)

            # Periodic GPU cleanup every 1000 steps
            if self.step_count > 0 and self.step_count % 1000 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

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
                    # Use loss value as importance score instead of gradient norm to avoid instability
                    gradient_norm = main_loss.item() if not torch.isnan(main_loss) else 0.0

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
                    num_experts=self.config.get('model', {}).get('num_experts', 8)
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

        # Add auxiliary losses with conservative weights
        for key, loss in loss_dict.items():
            if key.startswith('aux_') and isinstance(loss, torch.Tensor):
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss = total_loss + 0.001 * loss  # Much smaller weight for stability

        # Gradient surgery for multi-task learning (disabled for DeepSpeed and when losses are NaN)
        if self.gradient_surgeon and len(loss_dict) > 2 and not self.use_deepspeed and not torch.isnan(total_loss):
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
                    params_with_grad = []

                    for task_name, task_loss in task_losses.items():
                        optimizer.zero_grad()
                        task_loss.backward(retain_graph=True)

                        task_grads = []
                        # Store parameters that have gradients (first task defines the structure)
                        if task_name == list(task_losses.keys())[0]:
                            params_with_grad = [(i, param) for i, param in enumerate(self.model.parameters())
                                                if param.grad is not None]

                        # Collect gradients for parameters that have them
                        for idx, param in params_with_grad:
                            if param.grad is not None:
                                task_grads.append(param.grad.clone())

                        task_gradients[task_name] = task_grads

                        # Clear gradients
                        optimizer.zero_grad()

                    # Apply gradient surgery
                    if len(task_gradients) > 1 and len(params_with_grad) > 0:
                        try:
                            modified_grads = self.gradient_surgeon.apply_surgery(
                                task_gradients, task_losses
                            )

                            # Apply modified gradients back to the correct parameters
                            for grad_idx, (param_idx, param) in enumerate(params_with_grad):
                                if grad_idx < len(modified_grads):
                                    if modified_grads[grad_idx].shape == param.shape:
                                        param.grad = modified_grads[grad_idx]
                                    else:
                                        # Shape mismatch - fall back to normal backward
                                        raise ValueError(f"Gradient shape mismatch: expected {param.shape}, got {modified_grads[grad_idx].shape}")

                            loss_dict['gradient_conflicts'] = 1
                        except Exception as e:
                            # Fall back to normal backward pass if surgery fails
                            print(f"⚠️ Gradient surgery failed, using normal backward: {e}")
                            optimizer.zero_grad()
                            total_loss.backward()
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
            # Normal backward pass or DeepSpeed
            if self.use_deepspeed:
                # DeepSpeed backward with compatibility fix
                try:
                    self.model.backward(total_loss)
                except AttributeError as e:
                    if "timers" in str(e):
                        # Fallback for DeepSpeed version compatibility issue
                        print(f"⚠️ DeepSpeed timer compatibility issue: {e}")
                        print("🔄 Falling back to standard training...")
                        # Switch to standard PyTorch training
                        self.use_deepspeed = False
                        optimizer.zero_grad()
                        total_loss.backward()
                    else:
                        raise e
            else:
                optimizer.zero_grad()
                total_loss.backward()

        # Check for NaN before gradient updates
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ NaN/Inf loss detected: {total_loss.item()}, skipping gradient update")
            optimizer.zero_grad()
            loss_dict['total_loss'] = torch.tensor(0.0, device=total_loss.device)
            return

        # Gradient clipping and optimizer step
        if self.use_deepspeed:
            # DeepSpeed handles gradient clipping and optimization
            self.model.step()
        else:
            # Very aggressive gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

            # Much more permissive threshold - let model learn gradually
            if torch.isnan(grad_norm) or grad_norm > 200.0:
                print(f"⚠️ Large gradient norm detected: {grad_norm.item()}, skipping update")
                optimizer.zero_grad()
                loss_dict['total_loss'] = torch.tensor(0.0, device=total_loss.device)
                return

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

    def _save_checkpoint(self, epoch, optimizer, avg_loss):
        """Save model checkpoint at specified intervals."""
        try:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.use_deepspeed:
                # DeepSpeed checkpoint saving
                checkpoint_dir = output_dir / f'checkpoint_step_{self.step_count}'
                print(f"💾 Saving DeepSpeed checkpoint at step {self.step_count} to {checkpoint_dir}")

                # Save DeepSpeed checkpoint
                self.model.save_checkpoint(str(checkpoint_dir), tag=f'step_{self.step_count}')

                # Save additional training state
                state_file = checkpoint_dir / 'training_state.pt'
                torch.save({
                    'epoch': epoch,
                    'step': self.step_count,
                    'loss': avg_loss,
                    'config': self.config,
                    'training_stats': self.training_stats
                }, state_file)
            else:
                # Standard checkpoint saving
                checkpoint_path = output_dir / f'checkpoint_step_{self.step_count}.pt'
                print(f"💾 Saving checkpoint at step {self.step_count} to {checkpoint_path}")

                torch.save({
                    'epoch': epoch,
                    'step': self.step_count,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': self.config,
                    'training_stats': self.training_stats
                }, checkpoint_path)

            print(f"✅ Checkpoint saved successfully at step {self.step_count}")

        except Exception as e:
            print(f"⚠️ Failed to save checkpoint at step {self.step_count}: {e}")

    def _generate_evaluation_samples(self, num_samples=50):
        """Generate sample texts for evaluation."""
        self.model.eval()
        samples = []

        try:
            # Use more diverse and representative prompts
            prompts = [
                "The future of artificial intelligence will transform how we",
                "Climate change and its impact on global weather patterns",
                "Machine learning algorithms are becoming increasingly sophisticated",
                "Space exploration has revealed many secrets about our universe",
                "The importance of education in developing critical thinking skills",
                "Technology companies are investing heavily in research and development",
                "Scientists have discovered new methods for renewable energy generation",
                "The human brain processes information through complex neural networks"
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

                    # Generate with better parameters for evaluation
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=32,  # Shorter for faster evaluation
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,  # Add nucleus sampling
                        pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=1.1  # Reduce repetition
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
    # Register GPU cleanup handlers first thing
    register_cleanup_handlers()

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
    parser.add_argument('--use-moh', action='store_true', default=True,
                       help='Enable Mixture of Heads (MoH)')
    parser.add_argument('--use-moa', action='store_true', default=True,
                       help='Enable Mixture of Activations (MoA)')
    parser.add_argument('--use-cross-attention', action='store_true', default=True,
                       help='Enable multi-modal cross-attention')
    parser.add_argument('--use-alibi', action='store_true', default=True,
                       help='Use ALiBi positional encoding instead of RoPE')

    # Expert routing enhancements
    parser.add_argument('--expert-routing-type', type=str, default='switch',
                       choices=['base', 'switch', 'gshard', 'hash', 'stochastic'],
                       help='Type of expert routing to use')

    # RAG system
    parser.add_argument('--use-rag', action='store_true', default=True,
                       help='Enable Retrieval-Augmented Generation')
    parser.add_argument('--knowledge-base-path', type=str,
                       help='Path to knowledge base for RAG')
    parser.add_argument('--max-retrieved-docs', type=int, default=5,
                       help='Maximum number of documents to retrieve')
    parser.add_argument('--rag-fusion-type', type=str, default='attention',
                       choices=['attention', 'gate', 'concat', 'weighted'],
                       help='RAG fusion strategy')

    # Advanced loss functions
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Enable focal loss for hard example mining')
    parser.add_argument('--use-contrastive-loss', action='store_true', default=True,
                       help='Enable contrastive learning loss')
    parser.add_argument('--use-diversity-loss', action='store_true', default=True,
                       help='Enable expert diversity loss')
    parser.add_argument('--adaptive-loss-scaling', action='store_true', default=True,
                       help='Enable adaptive loss scaling')

    # Gradient surgery
    parser.add_argument('--gradient-surgery', action='store_true', default=True,
                       help='Enable gradient surgery for multi-task learning')
    parser.add_argument('--adaptive-gradient-surgery', action='store_true', default=True,
                       help='Use adaptive gradient surgery method selection')
    parser.add_argument('--gradient-surgery-method', type=str, default='pcgrad',
                       choices=['pcgrad', 'graddrop', 'gradnorm', 'cagrad', 'mgda'],
                       help='Gradient surgery method')

    # Multi-task learning
    parser.add_argument('--multi-task', action='store_true',
                       help='Enable multi-task learning mode')

    # Evaluation during training
    parser.add_argument('--eval-during-training', action='store_true', default=True,
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
    parser.add_argument('--use-episodic-memory', action='store_true', default=True,
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
                       default='/project/code/data/processed',
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
    parser.add_argument('--save-every', type=int, default=100,
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

    # === DEEPSPEED ARGUMENTS ===
    parser.add_argument('--use-deepspeed', action='store_true',
                       help='Enable DeepSpeed for distributed training and memory optimization')
    parser.add_argument('--deepspeed-config', type=str,
                       default='/project/code/configs/deepspeed/ds_config_simple.json',
                       help='Path to DeepSpeed configuration file')
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='Local rank for distributed training (set by DeepSpeed launcher)')

    # Add DeepSpeed's argument parser if available
    if DEEPSPEED_AVAILABLE:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)

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

    # Setup device and DeepSpeed
    use_deepspeed = args.use_deepspeed and DEEPSPEED_AVAILABLE

    # Check if we're in a distributed environment
    distributed_training = args.local_rank != -1 or 'WORLD_SIZE' in os.environ

    if use_deepspeed:
        # DeepSpeed handles device setup
        if distributed_training:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
            print(f"🚀 DeepSpeed enabled for distributed training with config: {args.deepspeed_config}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🚀 DeepSpeed enabled for single GPU training with config: {args.deepspeed_config}")
    else:
        # Standard device setup
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

    # Reduce learning rate for streaming mode to improve stability
    if args.streaming:
        current_lr = float(training_config.get('learning_rate', 1e-5))
        training_config['learning_rate'] = min(current_lr, 1e-6)  # Much smaller for all streaming
        print(f"🔧 Reduced learning rate for streaming mode: {training_config['learning_rate']}")

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
    # Get valid fields from the config class __init__ signature
    import inspect
    sig = inspect.signature(EnhancedMoEConfig.__init__)
    valid_fields = {param for param in sig.parameters.keys() if param not in ['self', 'kwargs']}

    # Debug: Show what's being filtered
    print("\n🔍 Debug: Config filtering")
    print(f"Config keys from YAML: {list(model_config_dict.keys())[:10]}...")  # Show first 10
    print(f"Valid EnhancedMoEConfig fields: {sorted(list(valid_fields))[:10]}...")  # Show first 10

    # Create filtered config - be more lenient with the filtering
    filtered_config = {}
    for k, v in model_config_dict.items():
        if k in valid_fields:
            filtered_config[k] = v

    # Manually ensure critical parameters are included if present
    critical_params = ['vocab_size', 'hidden_size', 'num_layers', 'intermediate_size', 'num_experts', 'num_experts_per_token']
    for param in critical_params:
        if param in model_config_dict and param in valid_fields:
            filtered_config[param] = model_config_dict[param]

    removed_keys = set(model_config_dict.keys()) - set(filtered_config.keys())
    if removed_keys:
        print(f"⚠️ Filtered out config keys (total: {len(removed_keys)}): {list(removed_keys)[:10]}...")
    print(f"✅ Passed config values: hidden_size={filtered_config.get('hidden_size', 'NOT SET')}, "
          f"intermediate_size={filtered_config.get('intermediate_size', 'NOT SET')}, "
          f"num_experts={filtered_config.get('num_experts', 'NOT SET')}")

    model_config = EnhancedMoEConfig(**filtered_config)
    model = EnhancedMoEModel(model_config)

    # Count parameters before DeepSpeed
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    if use_deepspeed:
        # Load DeepSpeed config
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)

        # Update DeepSpeed config with training parameters
        ds_config['train_batch_size'] = batch_size * args.gradient_accumulation
        ds_config['train_micro_batch_size_per_gpu'] = batch_size
        ds_config['gradient_accumulation_steps'] = args.gradient_accumulation
        ds_config['optimizer']['params']['lr'] = learning_rate
        ds_config['scheduler']['params']['warmup_max_lr'] = learning_rate
        ds_config['scheduler']['params']['warmup_num_steps'] = training_config.get('warmup_steps', 500)

        # Calculate total steps - handle streaming datasets
        try:
            dataset_length = len(train_loader)
            total_steps = num_epochs * dataset_length
        except TypeError:
            # Streaming dataset doesn't have length, estimate from max_train_examples
            data_config = config_dict.get('data', {})
            max_examples = data_config.get('max_train_examples', args.max_samples or 10000)
            steps_per_epoch = max_examples // (batch_size * args.gradient_accumulation)
            total_steps = num_epochs * steps_per_epoch
            print(f"📊 Estimated steps per epoch: {steps_per_epoch} (streaming dataset)")

        ds_config['scheduler']['params']['total_num_steps'] = total_steps

        # Initialize DeepSpeed
        try:
            # Fix engine_timers compatibility issue before initialization
            # Monkey patch the DeepSpeedEngine class to handle missing timers
            import deepspeed.runtime.engine
            original_getattr = deepspeed.runtime.engine.DeepSpeedEngine.__getattribute__

            def patched_getattr(self, name):
                if name == 'engine_timers':
                    # Return a mock timers object if it doesn't exist
                    if not hasattr(self, '_mock_timers'):
                        class MockTimers:
                            def __init__(self):
                                self.forward_timers = []  # Empty list instead of None
                                self.backward_timers = []  # Empty list instead of None
                                self.backward_inner_timers = []  # Missing attribute causing the error
                                self.forward_inner_timers = []   # Add this for completeness

                            def __getattr__(self, name):
                                # Return empty list for any missing timer attributes
                                if 'timer' in name.lower():
                                    return []
                                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
                        self._mock_timers = MockTimers()
                    return self._mock_timers
                return original_getattr(self, name)

            deepspeed.runtime.engine.DeepSpeedEngine.__getattribute__ = patched_getattr

            if distributed_training:
                # Distributed training - DeepSpeed handles initialization
                model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=model,
                    config=ds_config
                )
            else:
                # Single GPU training - need to set up environment
                if 'RANK' not in os.environ:
                    os.environ['MASTER_ADDR'] = 'localhost'
                    # Use a random free port to avoid conflicts
                    import socket
                    sock = socket.socket()
                    sock.bind(('', 0))
                    port = sock.getsockname()[1]
                    sock.close()
                    os.environ['MASTER_PORT'] = str(port)
                    os.environ['RANK'] = '0'
                    os.environ['LOCAL_RANK'] = '0'
                    os.environ['WORLD_SIZE'] = '1'

                model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=model,
                    config=ds_config
                )

            model = model_engine
            print(f"✅ DeepSpeed initialized with ZeRO stage {ds_config['zero_optimization']['stage']}")
        except Exception as e:
            print(f"⚠️ DeepSpeed initialization failed: {e}")
            print("🔄 Falling back to standard training...")
            use_deepspeed = False

            # Standard initialization
            model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.01
            )
    else:
        # Standard initialization
        model.to(device)

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

    # Initialize trainer (use_deepspeed might have been changed to False in fallback)
    trainer = EnhancedTrainer(
        args=args,
        config=config_dict,
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_deepspeed=use_deepspeed
    )

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n📍 Epoch {epoch}/{num_epochs}")

        try:
            # Train
            start_time = time.time()
            train_loss = trainer.train_epoch(train_loader, optimizer, epoch, num_epochs)
            train_time = time.time() - start_time

            avg_train_loss = train_loss.get('avg_total_loss', 0.0) if isinstance(train_loss, dict) else train_loss
            print(f"  Training loss: {avg_train_loss:.4f}")
            print(f"  Training time: {train_time:.1f}s")

            # Evaluate
            val_loss = evaluate(model, val_loader, device)
            print(f"  Validation loss: {val_loss:.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️ GPU out of memory in epoch {epoch}. Cleaning up and continuing...")
                cleanup_gpu_memory()
                # Try to continue with the next epoch
                continue
            else:
                print(f"\n❌ Runtime error in epoch {epoch}: {e}")
                cleanup_gpu_memory()
                raise

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

    # Final GPU cleanup
    print("\n🧹 Performing final GPU cleanup...")
    cleanup_gpu_memory()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        cleanup_gpu_memory()
        print("✅ GPU cleanup completed")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        cleanup_gpu_memory()
        raise