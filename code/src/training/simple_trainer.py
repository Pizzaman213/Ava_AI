"""
Simplified trainer for MoE++ model - for testing and development
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import json
from tqdm.auto import tqdm
from dataclasses import dataclass
import time
import os
import sys
from LLM.src.utils.path_utils import get_data_dir, get_outputs_dir, get_cache_dir, get_checkpoints_dir

logger = logging.getLogger(__name__)

@dataclass
class SimpleTrainingConfig:
    """Simplified training configuration"""
    # Model
    model_config: Any = None
    
    # Data
    train_dataset: Any = None
    val_dataset: Optional[Any] = None
    batch_size: int = 8
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Training
    num_epochs: int = 3
    max_steps: int = -1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    gradient_accumulation_steps: int = 1
    
    # Output
    output_dir: str = None
    experiment_name: str = "moe_train"
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = get_outputs_dir()
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "bf16"

class SimpleMoETrainer:
    """Simplified trainer for MoE++ model"""
    
    def __init__(self, config: SimpleTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        logger.info("Initializing model...")
        from ..model.moe_transformer import MoEForCausalLM
        self.model = MoEForCausalLM(config.model_config)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {total_params/1e6:.1f}M parameters")
        
        # Setup training components
        self._setup_optimizer()
        self._setup_dataloaders()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb tracking
        self.wandb_run = None
        
        # Setup mixed precision scaler
        self.use_amp = getattr(self.config, 'mixed_precision', 'no') in ['fp16', 'bf16']
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
    
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        optimizer_type = getattr(self.config, 'optimizer', 'adamw').lower()
        
        # Get optimizer parameters from config
        learning_rate = self.config.learning_rate
        weight_decay = self.config.weight_decay
        beta1 = getattr(self.config, 'adam_beta1', 0.9)
        beta2 = getattr(self.config, 'adam_beta2', 0.999)
        eps = getattr(self.config, 'adam_epsilon', 1e-8)
        
        logger.info(f"  Optimizer: {optimizer_type}")
        
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'fused_adamw':
            # Use PyTorch's fused AdamW implementation for better performance
            if hasattr(torch.optim, 'AdamW') and torch.cuda.is_available():
                try:
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=learning_rate,
                        betas=(beta1, beta2),
                        eps=eps,
                        weight_decay=weight_decay,
                        fused=True  # Enable fused implementation
                    )
                    logger.info("  Using fused AdamW implementation")
                except Exception as e:
                    logger.warning(f"Fused AdamW not available ({e}). Falling back to standard AdamW")
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=learning_rate,
                        betas=(beta1, beta2),
                        eps=eps,
                        weight_decay=weight_decay
                    )
            else:
                logger.warning("Fused AdamW requires CUDA. Using standard AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'lion':
            try:
                from lion_pytorch import Lion
                self.optimizer = Lion(
                    self.model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("Lion optimizer not installed. Install with: pip install lion-pytorch")
                logger.warning("Falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
        elif optimizer_type == 'sophia':
            try:
                from sophia import SophiaG
                self.optimizer = SophiaG(
                    self.model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    rho=getattr(self.config, 'sophia_rho', 0.04),
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("Sophia optimizer not installed. Install with: pip install sophia-optimizer")
                logger.warning("Falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
        elif optimizer_type == 'adafactor':
            try:
                from transformers import Adafactor
                self.optimizer = Adafactor(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    scale_parameter=True,
                    relative_step=False,
                    warmup_init=False
                )
            except ImportError:
                logger.warning("Adafactor requires transformers. Falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
        elif optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                alpha=0.99,
                eps=eps,
                weight_decay=weight_decay,
                momentum=getattr(self.config, 'rmsprop_momentum', 0.9)
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=getattr(self.config, 'sgd_momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}. Using standard AdamW.")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
        
        # Setup learning rate scheduler
        self.lr_scheduler = None
        if hasattr(self.config, 'lr_scheduler_type') and self.config.lr_scheduler_type:
            from transformers import get_scheduler
            
            # Calculate total training steps
            # Try to get actual dataset size
            if hasattr(self.config, 'train_dataset') and hasattr(self.config.train_dataset, '__len__'):
                dataset_size = len(self.config.train_dataset)
            else:
                dataset_size = getattr(self.config, 'num_train_examples', 100000)
            
            num_training_steps = self.config.num_epochs * (dataset_size // self.config.batch_size)
            if hasattr(self.config, 'max_steps') and self.config.max_steps > 0:
                num_training_steps = self.config.max_steps
            
            # Get warmup steps
            warmup_steps = getattr(self.config, 'warmup_steps', 0)
            if warmup_steps == 0 and hasattr(self.config, 'warmup_ratio'):
                warmup_steps = int(num_training_steps * self.config.warmup_ratio)
            
            self.lr_scheduler = get_scheduler(
                name=self.config.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(f"  Learning rate scheduler: {self.config.lr_scheduler_type}")
            logger.info(f"  Warmup steps: {warmup_steps}")
            logger.info(f"  Total training steps: {num_training_steps}")
            logger.info(f"  Starting LR: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def _setup_dataloaders(self):
        """Setup data loaders"""
        from ..data.datasets import collate_fn
        from torch.utils.data import IterableDataset
        
        # Check if dataset is iterable (streaming)
        is_streaming = isinstance(self.config.train_dataset, IterableDataset)
        
        # Get number of workers from config or use CPU count
        num_workers = getattr(self.config, 'dataloader_num_workers', 2)  # Default to 2 for better memory usage
        if num_workers == -1:  # Auto-detect
            num_workers = min(2, os.cpu_count() // 2)  # Use half of CPU cores
        
        # For streaming datasets, limit workers to prevent issues
        if is_streaming:
            num_workers = min(num_workers, 1)  # Use single worker for streaming
        
        # Other dataloader settings - optimize for GPU training
        pin_memory = getattr(self.config, 'dataloader_pin_memory', True)  # Enable by default for GPU
        prefetch_factor = getattr(self.config, 'dataloader_prefetch_factor', 2) if num_workers > 0 else None
        persistent_workers = num_workers > 0
        
        self.train_dataloader = DataLoader(
            self.config.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=not is_streaming,  # Don't shuffle streaming datasets
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )
        
        if self.config.val_dataset:
            val_is_streaming = isinstance(self.config.val_dataset, IterableDataset)
            self.eval_dataloader = DataLoader(
                self.config.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"  Model: {self.model.__class__.__name__}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Max sequence length: {getattr(self.config, 'max_length', 'N/A')}")
        self.gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {getattr(self.config, 'mixed_precision', 'none')}")
        self.model.train()
        
        # Calculate total batches if possible
        try:
            # Try to get length directly
            total_batches = len(self.train_dataloader)
            batches_per_epoch = total_batches
        except:
            # For streaming datasets, try to calculate from dataset info
            try:
                if hasattr(self.train_dataloader.dataset, 'num_examples'):
                    # StreamingTensorDataset has num_examples
                    num_examples = self.train_dataloader.dataset.num_examples
                    batch_size = self.train_dataloader.batch_size or self.config.batch_size
                    total_batches = (num_examples + batch_size - 1) // batch_size  # Ceiling division
                    batches_per_epoch = total_batches
                elif hasattr(self.train_dataloader.dataset, 'metadata'):
                    # Alternative: get from metadata
                    num_examples = self.train_dataloader.dataset.metadata.get('num_examples', 0)
                    batch_size = self.train_dataloader.batch_size or self.config.batch_size
                    total_batches = (num_examples + batch_size - 1) // batch_size
                    batches_per_epoch = total_batches
                else:
                    total_batches = None
                    batches_per_epoch = "unknown"
            except:
                total_batches = None
                batches_per_epoch = "unknown"
        
        # Check if we're in an interactive environment
        disable_tqdm = not sys.stdout.isatty() or os.environ.get('DISABLE_TQDM', '0') == '1'
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            # Only log epoch info if not showing progress bar
            if disable_tqdm:
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            num_batches = 0
            
            # Create progress bar with total if known
            if total_batches:
                progress_bar = tqdm(self.train_dataloader, 
                                  desc=f"Epoch {epoch + 1}/{self.config.num_epochs} ({total_batches} batches)", 
                                  total=total_batches,
                                  disable=disable_tqdm)
            else:
                progress_bar = tqdm(self.train_dataloader, 
                                  desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", 
                                  disable=disable_tqdm)
            
            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                use_amp = getattr(self.config, 'mixed_precision', 'no') in ['fp16', 'bf16']
                if use_amp:
                    dtype = torch.float16 if self.config.mixed_precision == 'fp16' else torch.bfloat16
                    with torch.autocast(device_type=self.device.type, dtype=dtype):
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    loss = outputs.get('loss')
                    if loss is None:
                        # Compute loss if not provided
                        logits = outputs.get('logits')
                        if logits is not None:
                            # Shift for causal LM
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = batch['labels'][..., 1:].contiguous()
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        # Gradient clipping with scaler
                        if self.config.max_grad_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                        
                        # Optimizer step with scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Regular gradient clipping and optimizer step
                        if self.config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Learning rate scheduler step
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar with loss and remaining info
                avg_loss = epoch_loss / num_batches
                if total_batches:
                    remaining = total_batches - (batch_idx + 1)
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'remaining': f'{remaining}'
                    })
                else:
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    
                    # Get batch information
                    batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 0
                    seq_length = batch['input_ids'].shape[1] if 'input_ids' in batch else 0
                    total_tokens = batch_size * seq_length
                    
                    # Calculate learning rate (get actual LR from optimizer which scheduler updates)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Get memory usage if on GPU
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                        memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
                        memory_str = f" | gpu_mem={memory_used:.1f}/{memory_reserved:.1f}GB"
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self.device.type == 'mps':
                        # MPS memory tracking is limited
                        memory_str = " | device=MPS"
                    else:
                        memory_str = ""
                    
                    # Calculate progress info
                    if total_batches:
                        batches_done = batch_idx + 1
                        batches_remaining = total_batches - batches_done
                        percent_complete = (batches_done / total_batches) * 100
                        
                        # Estimate time remaining based on current speed
                        if hasattr(progress_bar, 'format_dict'):
                            rate = progress_bar.format_dict.get('rate', 0)
                            if rate and rate > 0:
                                eta_seconds = batches_remaining / rate
                                eta_str = f" | ETA: {eta_seconds/60:.1f}min"
                            else:
                                eta_str = ""
                        else:
                            eta_str = ""
                        
                        # Add warmup indicator
                        warmup_str = ""
                        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, '_step_count'):
                            if self.lr_scheduler._step_count <= getattr(self.lr_scheduler, 'num_warmup_steps', 0):
                                warmup_str = " [WARMUP]"
                        
                        logger.info(
                            f"Step {self.global_step}: loss={avg_loss:.4f} | batch_size={batch_size} | "
                            f"seq_len={seq_length} | tokens={total_tokens} | lr={current_lr:.2e}{memory_str} | "
                            f"[{percent_complete:.0f}%{eta_str}]{warmup_str}"
                        )
                    else:
                        # Streaming dataset - can't show exact progress
                        # Add warmup indicator
                        warmup_str = ""
                        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, '_step_count'):
                            if self.lr_scheduler._step_count <= getattr(self.lr_scheduler, 'num_warmup_steps', 0):
                                warmup_str = " [WARMUP]"
                        
                        logger.info(
                            f"Step {self.global_step}: loss={avg_loss:.4f} | batch_size={batch_size} | "
                            f"seq_len={seq_length} | tokens={total_tokens} | lr={current_lr:.2e}{memory_str}{warmup_str}"
                        )
                    
                    # WandB logging
                    if self.wandb_run:
                        metrics = {
                            # Core training metrics
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/global_step": self.global_step,
                            "train/epoch": self.epoch,
                            
                            # Batch metrics
                            "train/batch_size": batch_size,
                            "train/sequence_length": seq_length,
                            "train/tokens_per_step": total_tokens,
                            "train/tokens_seen": self.global_step * total_tokens,
                            
                            # Performance metrics
                            "performance/tokens_per_second": total_tokens / (time.time() - batch_start_time) if 'batch_start_time' in locals() else 0,
                            "performance/steps_per_second": 1 / (time.time() - batch_start_time) if 'batch_start_time' in locals() else 0,
                        }
                        
                        # Memory metrics
                        if torch.cuda.is_available() and self.device.type == 'cuda':
                            metrics.update({
                                "memory/gpu_memory_allocated_gb": memory_used,
                                "memory/gpu_memory_reserved_gb": memory_reserved,
                                "memory/gpu_memory_utilization": memory_used / memory_reserved if memory_reserved > 0 else 0,
                            })
                        
                        # Expert-specific metrics if available
                        if isinstance(outputs, dict) and 'aux_loss' in outputs and outputs['aux_loss'] is not None:
                            metrics["train/auxiliary_loss"] = outputs['aux_loss'].item() if hasattr(outputs['aux_loss'], 'item') else outputs['aux_loss']
                        
                        # Collect expert routing statistics from model layers
                        if self.global_step % (self.config.logging_steps * 5) == 0 and hasattr(self.model, 'model'):
                            try:
                                with torch.no_grad():
                                    # Get a forward pass with hidden states to analyze routing
                                    temp_outputs = self.model.model(
                                        input_ids=batch['input_ids'],
                                        attention_mask=batch.get('attention_mask'),
                                        output_hidden_states=True,
                                        return_dict=True
                                    )
                                    
                                    if 'hidden_states' in temp_outputs:
                                        all_hidden_states = temp_outputs['hidden_states']
                                        
                                        # Get expert statistics from each layer
                                        for layer_idx, layer in enumerate(self.model.model.layers):
                                            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router') and layer_idx < len(all_hidden_states) - 1:
                                                # Get router gate weights
                                                router = layer.mlp.router
                                                hidden_state = all_hidden_states[layer_idx]
                                                
                                                # Apply layer norm as the model would
                                                normed_hidden = layer.post_attention_layernorm(hidden_state)
                                                
                                                # Get routing decisions
                                                router_logits = router.gate(normed_hidden)
                                                router_probs = F.softmax(router_logits, dim=-1)
                                                
                                                # Expert utilization (average probability per expert)
                                                expert_probs = router_probs.mean(dim=[0, 1])  # Average over batch and sequence
                                                
                                                # Only log a subset of experts to avoid too many metrics
                                                num_experts_to_log = min(4, router.num_experts)
                                                for expert_idx in range(num_experts_to_log):
                                                    metrics[f"experts/layer_{layer_idx}/expert_{expert_idx}_prob"] = expert_probs[expert_idx].item()
                                                
                                                # Load balancing metrics
                                                expert_counts = router_probs.argmax(dim=-1).flatten().bincount(minlength=router.num_experts).float()
                                                expert_counts = expert_counts / expert_counts.sum()
                                                
                                                # Compute entropy for load balancing
                                                entropy = -(expert_counts * torch.log(expert_counts + 1e-9)).sum()
                                                metrics[f"experts/layer_{layer_idx}/routing_entropy"] = entropy.item()
                                                
                                                # Max vs min expert usage ratio (higher means more imbalanced)
                                                max_usage = expert_counts.max().item()
                                                min_usage = expert_counts[expert_counts > 0].min().item() if (expert_counts > 0).any() else 0
                                                metrics[f"experts/layer_{layer_idx}/load_balance_ratio"] = max_usage / (min_usage + 1e-9)
                            except Exception as e:
                                # Log but don't fail training if expert metrics collection fails
                                logger.debug(f"Failed to collect expert metrics: {e}")
                        
                        # Gradient metrics
                        if self.global_step % (self.config.logging_steps * 10) == 0:
                            grad_norm = 0
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    grad_norm += p.grad.data.norm(2).item() ** 2
                            grad_norm = grad_norm ** 0.5
                            metrics["gradients/global_norm"] = grad_norm
                        
                        self.wandb_run.log(metrics, step=self.global_step)
                    
                    batch_start_time = time.time()
                
                # Evaluation
                if self.config.val_dataset and self.global_step % self.config.eval_steps == 0:
                    self.evaluate()
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check max steps
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    logger.info("Reached max steps. Stopping training.")
                    return
            
            # End of epoch evaluation
            if self.config.val_dataset:
                eval_loss = self.evaluate()
                if num_batches > 0:
                    logger.info(f"Epoch {epoch + 1} complete. Train: {epoch_loss/num_batches:.4f}, Val: {eval_loss:.4f}")
                else:
                    logger.warning(f"Epoch {epoch + 1} complete. No training batches processed.")
                # Save best model based on validation loss
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint("best")
                    logger.info(f"New best model saved with validation loss: {eval_loss:.4f}")
            else:
                # Use train loss for best model selection
                avg_train_loss = epoch_loss / num_batches
                if avg_train_loss < self.best_loss:
                    self.best_loss = avg_train_loss
                    self.save_checkpoint("best")
                    logger.info(f"New best model saved with train loss: {avg_train_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
    
    def evaluate(self) -> float:
        """Evaluate model on validation set"""
        if not self.config.val_dataset:
            return 0.0
        
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            disable_tqdm = not sys.stdout.isatty() or os.environ.get('DISABLE_TQDM', '0') == '1'
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=disable_tqdm):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    loss = outputs.get('loss')
                    if loss is None:
                        # Compute loss if not provided
                        logits = outputs.get('logits')
                        if logits is not None:
                            # Shift for causal LM
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = batch['labels'][..., 1:].contiguous()
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Log evaluation metrics to WandB
        if self.wandb_run:
            self.wandb_run.log({
                "eval/loss": avg_loss,
                "eval/perplexity": torch.exp(torch.tensor(avg_loss)).item(),
                "eval/global_step": self.global_step,
            }, step=self.global_step)
        
        return avg_loss
    
    def save_checkpoint(self, name: Optional[str] = None):
        """Save model checkpoint"""
        if name is None:
            name = f"checkpoint_{self.global_step}"
        
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt"
        )
        
        # Also save as pytorch_model.bin for compatibility
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "pytorch_model.bin"
        )
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config.__dict__
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save model config in the format expected by from_pretrained
        model_config_dict = self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else self.model.config.__dict__
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(model_config_dict, f, indent=2)
        
        # Also save as YAML for compatibility with the training configs
        import yaml
        config_yaml = {
            'model': model_config_dict,
            'data': {
                'dataset_type': 'tensor',
                'train_path': os.path.join(get_data_dir(), 'pretraining', 'tensor', 'train'),
                'val_path': os.path.join(get_data_dir(), 'pretraining', 'tensor', 'validation'),
                'tokenizer': 'gpt2',
                'max_length': getattr(self.config.model_config, 'max_position_embeddings', 256),
                'stride': getattr(self.config.model_config, 'max_position_embeddings', 256) // 2,
                'dataset_args': {
                    'text_column': 'text',
                    'streaming': True,
                    'num_proc': 1
                }
            },
            'training': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'eval_steps': self.config.eval_steps,
                'save_steps': self.config.save_steps,
                'logging_steps': self.config.logging_steps,
                'weight_decay': self.config.weight_decay,
                'max_grad_norm': self.config.max_grad_norm,
                'mixed_precision': self.config.mixed_precision,
                'gradient_accumulation_steps': 1,
                'warmup_steps': 500,
                'dataloader_num_workers': 0
            }
        }
        
        with open(checkpoint_dir / "config.yaml", 'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False)
        
        logger.info(f"Checkpoint saved: {name}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model state
        model_path = checkpoint_path / "model.pt"
        optimizer_path = checkpoint_path / "optimizer.pt"
        state_path = checkpoint_path / "training_state.json"
        
        loaded_components = []
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            loaded_components.append("model")
        
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            loaded_components.append("optimizer")
        
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.global_step = state.get('global_step', 0)
                self.epoch = state.get('epoch', 0)
            loaded_components.append("state")
        
        if loaded_components:
            logger.info(f"Loaded checkpoint: {checkpoint_path.name} (step {self.global_step})")