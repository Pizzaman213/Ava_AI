"""
Advanced Training Pipeline for MoE++ Model
Implements distributed training, mixed precision, and advanced optimizations
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import wandb
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from ..model.moe_transformer import MoEForCausalLM, MoEConfig
from ..data.fast_dataloader import create_optimized_dataloader
from .optimizer import create_optimizer, create_scheduler
from ..memory.zero3_offload import ZeRO3Wrapper
from ..utils.parallel_utils import setup_distributed, cleanup_distributed

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model
    model_config: MoEConfig = None
    
    # Data
    train_dataset: Any = None
    val_dataset: Any = None
    batch_size: int = 8
    num_workers: int = 4
    dataloader_num_workers: int = 4
    use_dynamic_batching: bool = False
    max_length: int = 512
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training
    num_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    # Mixed precision
    mixed_precision: str = "bf16"  # no, fp16, bf16
    gradient_checkpointing: bool = True
    
    # Distributed
    distributed: bool = True
    local_rank: int = -1
    world_size: int = 1
    
    # Memory optimization
    use_zero3: bool = True
    cpu_offload: bool = True
    nvme_offload: bool = False
    activation_checkpointing: bool = True
    
    # Checkpointing
    output_dir: str = "outputs"
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "moe-llm"
    wandb_run_name: Optional[str] = None
    
    # Advanced
    use_compile: bool = False  # PyTorch 2.0 compile
    compile_mode: str = "default"
    dataloader_num_workers: int = 4
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = MoEConfig()

class MoETrainer:
    """Advanced trainer for MoE++ models"""
    def __init__(
        self,
        model: MoEForCausalLM,
        config: TrainingConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup distributed training
        self._setup_distributed()
        
        # Setup accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None,
            project_dir=config.output_dir,
        )
        
        # Setup device
        self.device = self.accelerator.device
        
        # Initialize components
        self._setup_model()
        self._setup_optimizers()
        self._setup_dataloaders()
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Metrics
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "throughput": [],
        }
    
    def _setup_distributed(self):
        """Setup distributed training environment"""
        if self.config.distributed:
            if self.config.local_rank == -1:
                # Setup distributed from environment
                setup_distributed()
                self.config.local_rank = dist.get_rank()
                self.config.world_size = dist.get_world_size()
            
            # Set device
            torch.cuda.set_device(self.config.local_rank)
    
    def _setup_model(self):
        """Setup model with optimizations"""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            # Check if model has gradient_checkpointing_enable method
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
                # For PEFT models
                self.model.base_model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing'):
                # For MoE models with gradient_checkpointing as attribute
                self.model.model.gradient_checkpointing = True
            else:
                logger.warning("Model does not support gradient checkpointing")
        
        # Compile model (PyTorch 2.0+)
        if self.config.use_compile and hasattr(torch, 'compile'):
            logger.info(f"Compiling model with mode: {self.config.compile_mode}")
            self.model = torch.compile(self.model, mode=self.config.compile_mode)
        
        # Wrap with ZeRO-3 if enabled
        if self.config.use_zero3:
            self.model = ZeRO3Wrapper(
                self.model,
                cpu_offload=self.config.cpu_offload,
                nvme_offload=self.config.nvme_offload,
            )
        
        # Prepare with accelerator
        self.model = self.accelerator.prepare(self.model)
    
    def _setup_optimizers(self):
        """Setup optimizer and scheduler"""
        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
        )
        
        # Create scheduler
        num_training_steps = self._calculate_training_steps()
        self.scheduler = create_scheduler(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_type="cosine",
        )
        
        # Prepare with accelerator
        self.optimizer, self.scheduler = self.accelerator.prepare(
            self.optimizer, self.scheduler
        )
        
        # Setup gradient scaler for mixed precision
        if self.config.mixed_precision == "fp16":
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def _setup_dataloaders(self):
        """Setup data loaders"""
        # Check if we're using a streaming dataset
        from torch.utils.data import IterableDataset
        is_streaming = isinstance(self.train_dataset, IterableDataset)
        
        # For streaming datasets, use standard DataLoader to avoid prefetch issues
        if is_streaming:
            from torch.utils.data import DataLoader
            from ..data.datasets import collate_fn
            
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_fn,
                num_workers=0,  # Streaming works better with 0 workers
                pin_memory=False,  # Disable pin_memory for MPS
            )
        else:
            # Create train dataloader with optimization for non-streaming
            # When using dynamic batching, batch_size is actually max_tokens_per_batch
            batch_size_param = self.config.batch_size * self.config.max_length if hasattr(self.config, 'max_length') else self.config.batch_size * 512
            
            self.train_dataloader = create_optimized_dataloader(
                self.train_dataset,
                batch_size=batch_size_param if self.config.use_dynamic_batching else self.config.batch_size,
                num_workers=self.config.dataloader_num_workers,
                device=self.device,
                distributed=self.config.distributed,
                use_dynamic_batching=self.config.use_dynamic_batching if hasattr(self.config, 'use_dynamic_batching') else False,
            )
        
        # Create eval dataloader
        if self.eval_dataset:
            eval_is_streaming = isinstance(self.eval_dataset, IterableDataset)
            if eval_is_streaming:
                from torch.utils.data import DataLoader
                from ..data.datasets import collate_fn
                
                self.eval_dataloader = DataLoader(
                    self.eval_dataset,
                    batch_size=self.config.batch_size,
                    collate_fn=collate_fn,
                    num_workers=0,
                    pin_memory=False,
                )
            else:
                self.eval_dataloader = create_optimized_dataloader(
                    self.eval_dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.dataloader_num_workers,
                    device=self.device,
                    distributed=self.config.distributed,
                )
        else:
            self.eval_dataloader = None
        
        # Prepare with accelerator
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
    
    def _setup_logging(self):
        """Setup logging and tracking"""
        if self.accelerator.is_main_process:
            # Setup wandb
            if self.config.use_wandb:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=asdict(self.config),
                )
            
            # Create output directory
            self.output_dir = Path(self.config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(asdict(self.config), f, indent=2)
    
    def _calculate_training_steps(self) -> int:
        """Calculate total training steps"""
        if self.config.max_steps > 0:
            return self.config.max_steps
        
        # Check if dataloader exists
        if hasattr(self, 'train_dataloader'):
            try:
                steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
                return steps_per_epoch * self.config.num_epochs
            except TypeError:
                pass
        
        # Try to calculate from dataset directly
        dataset_size = None
        
        # Try different ways to get dataset size
        if hasattr(self.train_dataset, '__len__'):
            try:
                dataset_size = len(self.train_dataset)
            except:
                pass
        
        if dataset_size is None and hasattr(self.train_dataset, 'num_examples'):
            dataset_size = self.train_dataset.num_examples
        elif dataset_size is None and hasattr(self.train_dataset, 'metadata') and 'num_examples' in self.train_dataset.metadata:
            dataset_size = self.train_dataset.metadata['num_examples']
        
        if dataset_size:
            steps_per_epoch = dataset_size // (self.config.batch_size * self.config.gradient_accumulation_steps)
            return steps_per_epoch * self.config.num_epochs
        
        # Default to -1 (unknown)
        return -1
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Check if dataset has length (non-streaming)
        try:
            num_examples = len(self.train_dataset)
            logger.info(f"  Num examples = {num_examples}")
        except TypeError:
            # For streaming datasets, try to get count from metadata
            if hasattr(self.train_dataset, 'num_examples'):
                num_examples = self.train_dataset.num_examples
                logger.info(f"  Num examples = {num_examples} (streaming dataset)")
            elif hasattr(self.train_dataset, 'metadata') and 'num_examples' in self.train_dataset.metadata:
                num_examples = self.train_dataset.metadata['num_examples']
                logger.info(f"  Num examples = {num_examples} (from metadata)")
            else:
                logger.info(f"  Num examples = Unknown (streaming dataset)")
        
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        
        total_steps = self._calculate_training_steps()
        if total_steps > 0:
            logger.info(f"  Total optimization steps = {total_steps}")
            effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
            logger.info(f"  Effective batch size = {effective_batch_size}")
        else:
            logger.info(f"  Total optimization steps = Unknown (streaming dataset)")
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        
        # Training loop
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            train_loss = self._train_epoch()
            
            # Evaluation
            if self.eval_dataloader:
                eval_loss = self._evaluate()
                logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}, Eval loss: {eval_loss:.4f}")
                
                # Save best model based on eval loss
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("best")
                    logger.info(f"New best model saved with eval loss: {eval_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")
                # If no eval set, use train loss for best model selection
                if train_loss < self.best_eval_loss:
                    self.best_eval_loss = train_loss
                    self._save_checkpoint("best")
                    logger.info(f"New best model saved with train loss: {train_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch}")
            
            # Check if should stop
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
        
        # Final save
        self._save_checkpoint("final")
        
        # If no "best" checkpoint exists (no eval was done), copy final as best
        best_checkpoint_path = self.output_dir / "checkpoint-best"
        if not best_checkpoint_path.exists():
            logger.info("No best checkpoint found (no evaluation data). Using final checkpoint as best.")
            import shutil
            final_checkpoint_path = self.output_dir / "checkpoint-final"
            if final_checkpoint_path.exists():
                shutil.copytree(final_checkpoint_path, best_checkpoint_path)
                logger.info(f"Copied final checkpoint to best: {best_checkpoint_path}")
        else:
            logger.info(f"Best checkpoint already exists at: {best_checkpoint_path}")
        
        # Cleanup
        if self.config.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")
    
    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            disable=not self.accelerator.is_main_process,
        )
        
        batch_start_time = time.time()
        for step, batch in enumerate(progress_bar):
            # Training step
            loss = self._training_step(batch)
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            self.accelerator.backward(loss / self.config.gradient_accumulation_steps)
            
            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.metrics["grad_norm"].append(grad_norm.item())
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics({
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.item() if self.config.max_grad_norm > 0 else 0.0,
                    })
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    if self.eval_dataloader:
                        eval_loss = self._evaluate()
                        self.model.train()
                        
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self._save_checkpoint("best")
                            logger.info(f"New best model saved at step {self.global_step} with eval loss: {eval_loss:.4f}")
                    else:
                        # Use current average loss as proxy for best model selection
                        current_avg_loss = total_loss / num_batches
                        if current_avg_loss < self.best_eval_loss:
                            self.best_eval_loss = current_avg_loss
                            self._save_checkpoint("best")
                            logger.info(f"New best model saved at step {self.global_step} with train loss: {current_avg_loss:.4f}")
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
                
                # Calculate additional stats
                avg_loss = total_loss / num_batches
                tokens_per_sec = 0
                if hasattr(batch, 'get') and 'input_ids' in batch:
                    batch_tokens = batch['input_ids'].numel()
                    elapsed_time = time.time() - batch_start_time if 'batch_start_time' in locals() else 1.0
                    tokens_per_sec = batch_tokens / elapsed_time
                
                # Memory stats for MPS/CUDA
                memory_str = ""
                if torch.cuda.is_available():
                    memory_gb = torch.cuda.memory_allocated() / 1024**3
                    memory_str = f", mem: {memory_gb:.1f}GB"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    memory_str = ", mem: MPS"
                
                # Update progress bar with comprehensive stats
                progress_bar.set_postfix({
                    "step": self.global_step,
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "grad_norm": f"{grad_norm.item():.2f}" if self.config.max_grad_norm > 0 and 'grad_norm' in locals() else "N/A",
                    "tokens/s": f"{tokens_per_sec:.0f}" if tokens_per_sec > 0 else "N/A",
                    "acc_steps": f"{(step + 1) % self.config.gradient_accumulation_steps}/{self.config.gradient_accumulation_steps}",
                } | ({"memory": memory_str} if memory_str else {}))
                
                batch_start_time = time.time()
        
        # Handle case where no batches were processed
        if num_batches == 0:
            logger.warning("No batches were processed in this epoch")
            return 0.0
            
        return total_loss / num_batches
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step"""
        # Forward pass
        with self.accelerator.autocast():
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Add auxiliary losses
            if "aux_loss" in outputs:
                loss = loss + outputs["aux_loss"]
        
        return loss
    
    def _evaluate(self) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                with self.accelerator.autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self._log_metrics({"eval_loss": avg_loss})
        
        return avg_loss
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics"""
        # Update internal metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Log to wandb
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.log(metrics, step=self.global_step)
        
        # Log to console
        if self.accelerator.is_main_process:
            # Include max steps if known
            if hasattr(self.config, 'max_steps') and self.config.max_steps > 0:
                log_str = f"[Step {self.global_step}/{self.config.max_steps}] "
            else:
                log_str = f"[Step {self.global_step}] "
            log_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(log_str)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-{name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Handle different model types
        if hasattr(unwrapped_model, 'save_pretrained'):
            # For HuggingFace models
            unwrapped_model.save_pretrained(checkpoint_dir)
        elif hasattr(unwrapped_model, 'module') and hasattr(unwrapped_model.module, 'save_pretrained'):
            # For wrapped models
            unwrapped_model.module.save_pretrained(checkpoint_dir)
        else:
            # Fallback: save state dict
            model_to_save = unwrapped_model
            # Check for PEFT model
            if hasattr(unwrapped_model, 'module'):
                model_to_save = unwrapped_model.module
            elif hasattr(unwrapped_model, 'model'):
                model_to_save = unwrapped_model.model
                
            # Save the model state dict
            torch.save(model_to_save.state_dict(), checkpoint_dir / "pytorch_model.bin")
            
            # Save config if available
            if hasattr(model_to_save, 'config'):
                config_to_save = model_to_save.config
                # Handle wrapped configs
                if hasattr(config_to_save, '_config'):
                    config_to_save = config_to_save._config
                    
                import json
                config_dict = config_to_save.__dict__ if hasattr(config_to_save, '__dict__') else {}
                with open(checkpoint_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": self.metrics,
            "config": asdict(self.config),
        }
        
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        # Manage checkpoint limit
        self._cleanup_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_dir}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        self.model = MoEForCausalLM.from_pretrained(str(checkpoint_dir))
        self.model = self.model.to(self.device)
        
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_eval_loss = state["best_eval_loss"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.metrics = state["metrics"]
            
            logger.info(f"Resumed from checkpoint: {checkpoint_dir}")
            logger.info(f"  Global step: {self.global_step}")
            logger.info(f"  Epoch: {self.epoch}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain limit"""
        if self.config.save_total_limit <= 0:
            return
        
        # Get all checkpoints
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        
        # Keep best and final
        keep_checkpoints = ["checkpoint-best", "checkpoint-final"]
        
        # Sort by modification time
        checkpoints = sorted(
            [cp for cp in checkpoints if cp.name not in keep_checkpoints],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.config.save_total_limit - len(keep_checkpoints):]:
            import shutil
            shutil.rmtree(checkpoint)
            logger.info(f"Removed old checkpoint: {checkpoint}")

class DistributedTrainer(MoETrainer):
    """Distributed trainer with advanced parallelism"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional distributed setup
        self._setup_model_parallelism()
    
    def _setup_model_parallelism(self):
        """Setup model parallelism if configured"""
        if hasattr(self.config, 'model_parallel_size') and self.config.model_parallel_size > 1:
            from ..model.mtp import MTPWrapper, MTPConfig
            
            mtp_config = MTPConfig(
                data_parallel_size=self.config.world_size // self.config.model_parallel_size,
                model_parallel_size=self.config.model_parallel_size,
                expert_parallel_size=getattr(self.config, 'expert_parallel_size', 1),
                pipeline_parallel_size=getattr(self.config, 'pipeline_parallel_size', 1),
            )
            
            self.model = MTPWrapper(self.model, mtp_config)

def train_moe_model(
    model_config: MoEConfig,
    training_config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
) -> MoEForCausalLM:
    """
    Train MoE model with advanced features
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        
    Returns:
        Trained model
    """
    # Create model
    model = MoEForCausalLM(model_config)
    
    # Create trainer
    if training_config.distributed:
        trainer = DistributedTrainer(
            model=model,
            config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        trainer = MoETrainer(
            model=model,
            config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    
    # Train
    trainer.train()
    
    # Return trained model
    return trainer.model