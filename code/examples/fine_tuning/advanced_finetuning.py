#!/usr/bin/env python3
"""
Advanced fine-tuning script for MoE++ model
Integrates LoRA experts, constitutional training, active learning, and more
"""
import argparse
import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.moe_transformer import MoEConfig, MoETransformer
from src.model.lora_experts import LoRAMoEModel, LoRAConfig
from src.model.uncertainty_quantification import UncertaintyAwareMoE, UncertaintyConfig
from src.training.curriculum_learning import CurriculumScheduler, CurriculumConfig
from src.training.constitutional_training import ConstitutionalTrainer, ConstitutionalConfig
from src.training.active_learning import ActiveLearningSelector, ActiveLearningConfig
from src.optimization.model_pruning import ModelPruner, PruningConfig
from src.monitoring.expert_health import ExpertHealthMonitor, HealthConfig
from src.data.datasets import get_dataset
from src.utils.logging_utils import setup_logging
from transformers import AutoTokenizer
import wandb

@dataclass
class AdvancedFinetuningConfig:
    """Configuration for advanced fine-tuning"""
    # Basic settings
    base_model_path: str
    output_dir: str = "./finetuned_models"
    experiment_name: Optional[str] = None
    
    # Task settings
    task_name: str = "general"
    num_tasks: int = 1
    multi_task: bool = False
    
    # Training settings
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    num_lora_experts: int = 4
    lora_target_modules: List[str] = None
    
    # Constitutional AI settings
    use_constitutional: bool = True
    safety_principles: List[str] = None
    num_revisions: int = 2
    safety_loss_weight: float = 0.1
    
    # Active learning settings
    use_active_learning: bool = True
    active_learning_strategy: str = "hybrid"
    selection_batch_size: int = 32
    pool_size: int = 10000
    
    # Curriculum learning settings
    use_curriculum: bool = True
    curriculum_type: str = "adaptive"
    initial_difficulty: float = 0.3
    
    # Uncertainty settings
    use_uncertainty: bool = True
    uncertainty_regularization: float = 0.01
    
    # Pruning settings
    prune_after_training: bool = True
    pruning_sparsity: float = 0.3
    
    # Monitoring
    use_health_monitoring: bool = True
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 500
    
    # Hardware
    device: str = "cuda"
    mixed_precision: str = "bf16"
    
    def __post_init__(self):
        if self.safety_principles is None:
            self.safety_principles = [
                "Be helpful, harmless, and honest",
                "Avoid generating harmful, biased, or offensive content",
                "Respect user privacy and confidentiality",
                "Provide accurate information and acknowledge limitations"
            ]
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate", "up_proj", "down_proj"]
        if self.experiment_name is None:
            self.experiment_name = f"advanced_finetune_{self.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class AdvancedFineTuner:
    """Advanced fine-tuning orchestrator"""
    
    def __init__(self, config: AdvancedFinetuningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_model()
        self._setup_training_components()
        self._setup_monitoring()
        
    def _setup_model(self):
        """Load and setup model with advanced features"""
        self.logger.info(f"Loading base model from {self.config.base_model_path}")
        
        # Load base model
        checkpoint = torch.load(self.config.base_model_path, map_location='cpu')
        model_config = checkpoint['config'].model_config if 'config' in checkpoint else checkpoint['model_config']
        
        base_model = MoETransformer(model_config)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Wrap with LoRA if enabled
        if self.config.use_lora:
            self.logger.info("Setting up LoRA experts...")
            lora_config = LoRAConfig(
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                num_lora_experts=self.config.num_lora_experts,
                target_modules=self.config.lora_target_modules,
                task_embedding_dim=64 if self.config.multi_task else 0,
                share_base_weights=True
            )
            self.model = LoRAMoEModel(base_model, lora_config)
            
            # Freeze base model weights
            for name, param in self.model.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
        else:
            self.model = base_model
            
        # Wrap with uncertainty if enabled
        if self.config.use_uncertainty:
            self.logger.info("Setting up uncertainty quantification...")
            uncertainty_config = UncertaintyConfig(
                use_monte_carlo_dropout=True,
                mc_dropout_samples=5,
                use_temperature_scaling=True
            )
            self.model = UncertaintyAwareMoE(self.model, uncertainty_config)
            
        self.model = self.model.to(self.config.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
    def _setup_training_components(self):
        """Setup training components"""
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision != "no" else None
        self.amp_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        
        # Curriculum learning
        if self.config.use_curriculum:
            self.logger.info("Setting up curriculum learning...")
            curriculum_config = CurriculumConfig(
                curriculum_type=self.config.curriculum_type,
                initial_difficulty=self.config.initial_difficulty,
                warmup_steps=self.config.warmup_steps
            )
            self.curriculum_scheduler = CurriculumScheduler(curriculum_config)
        else:
            self.curriculum_scheduler = None
            
        # Constitutional training
        if self.config.use_constitutional:
            self.logger.info("Setting up constitutional training...")
            constitutional_config = ConstitutionalConfig(
                principles=self.config.safety_principles,
                num_revisions=self.config.num_revisions,
                safety_weight=self.config.safety_loss_weight
            )
            self.constitutional_trainer = ConstitutionalTrainer(self.model, constitutional_config)
        else:
            self.constitutional_trainer = None
            
        # Active learning
        if self.config.use_active_learning:
            self.logger.info("Setting up active learning...")
            al_config = ActiveLearningConfig(
                strategy=self.config.active_learning_strategy,
                batch_size=self.config.selection_batch_size,
                pool_size=self.config.pool_size
            )
            self.active_selector = ActiveLearningSelector(self.model, al_config)
        else:
            self.active_selector = None
            
    def _setup_monitoring(self):
        """Setup monitoring components"""
        
        # Expert health monitoring
        if self.config.use_health_monitoring:
            self.logger.info("Setting up health monitoring...")
            health_config = HealthConfig(
                check_interval=self.config.log_every_n_steps,
                detailed_check_interval=self.config.eval_every_n_steps
            )
            self.health_monitor = ExpertHealthMonitor(self.model, health_config)
        else:
            self.health_monitor = None
            
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'uncertainty': []
        }
        
    def fine_tune(
        self,
        train_dataset,
        eval_dataset=None,
        task_id: Optional[int] = None
    ):
        """Run fine-tuning with all advanced features"""
        
        self.logger.info(f"Starting fine-tuning for task: {self.config.task_name}")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
        # Training loop
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            self.model.train()
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Active learning selection
                if self.active_selector and global_step % 100 == 0:
                    # Update training samples based on uncertainty
                    selected_indices, _ = self.active_selector.select_batch()
                    
                # Curriculum learning
                if self.curriculum_scheduler:
                    curriculum_params = self.curriculum_scheduler.step()
                    # Filter batch by difficulty if needed
                    
                # Training step
                loss = self._training_step(batch, task_id)
                epoch_loss += loss
                
                # Logging
                if global_step % self.config.log_every_n_steps == 0:
                    self.logger.info(
                        f"Step {global_step}, Loss: {loss:.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                    
                # Evaluation
                if eval_loader and global_step % self.config.eval_every_n_steps == 0:
                    eval_metrics = self.evaluate(eval_loader, task_id)
                    self.logger.info(f"Eval metrics: {eval_metrics}")
                    
                    # Save best model
                    if eval_metrics['loss'] < best_eval_loss:
                        best_eval_loss = eval_metrics['loss']
                        self.save_model("best_model.pt")
                        
                # Save checkpoint
                if global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{global_step}.pt")
                    
                # Health monitoring
                if self.health_monitor:
                    self.health_monitor.step()
                    
                global_step += 1
                
            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
        # Final evaluation
        if eval_loader:
            final_metrics = self.evaluate(eval_loader, task_id)
            self.logger.info(f"Final evaluation metrics: {final_metrics}")
            
        # Pruning if enabled
        if self.config.prune_after_training:
            self._prune_model(eval_loader)
            
        # Save final model
        self.save_model("final_model.pt")
        
        # Save health report
        if self.health_monitor:
            self.health_monitor.save_report(str(self.output_dir / "health_report.json"))
            
    def _training_step(self, batch: Dict[str, torch.Tensor], task_id: Optional[int] = None) -> float:
        """Single training step with advanced features"""
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.config.device)
        labels = batch.get('labels', input_ids).to(self.config.device)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision != "no", dtype=self.amp_dtype):
            if isinstance(self.model, LoRAMoEModel) and task_id is not None:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    task_id=task_id
                )
            else:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
            loss = outputs.loss
            
            # Add constitutional loss
            if self.constitutional_trainer:
                safety_loss = self.constitutional_trainer.compute_safety_loss(
                    input_ids,
                    outputs.logits if hasattr(outputs, 'logits') else outputs
                )
                loss = loss + safety_loss * self.config.safety_loss_weight
                
            # Add uncertainty regularization
            if self.config.use_uncertainty and hasattr(outputs, 'uncertainty'):
                uncertainty_loss = outputs['uncertainty']['total'].mean()
                loss = loss + uncertainty_loss * self.config.uncertainty_regularization
                
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
        self.optimizer.zero_grad()
        
        return loss.item()
        
    def evaluate(self, eval_loader, task_id: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model with advanced metrics"""
        
        self.model.eval()
        total_loss = 0
        total_uncertainty = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)
                labels = batch.get('labels', input_ids).to(self.config.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision != "no", dtype=self.amp_dtype):
                    if isinstance(self.model, LoRAMoEModel) and task_id is not None:
                        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            task_id=task_id,
                            return_uncertainty=self.config.use_uncertainty
                        )
                    else:
                        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_uncertainty=self.config.use_uncertainty
                        )
                        
                total_loss += outputs.loss.item()
                
                if self.config.use_uncertainty and 'uncertainty' in outputs:
                    total_uncertainty += outputs['uncertainty']['total'].mean().item()
                    
                num_batches += 1
                
        metrics = {
            'loss': total_loss / num_batches,
            'perplexity': torch.exp(torch.tensor(total_loss / num_batches)).item()
        }
        
        if self.config.use_uncertainty:
            metrics['avg_uncertainty'] = total_uncertainty / num_batches
            
        return metrics
        
    def _prune_model(self, eval_loader):
        """Prune model after training"""
        
        self.logger.info(f"Pruning model to {self.config.pruning_sparsity:.0%} sparsity...")
        
        pruning_config = PruningConfig(
            target_sparsity=self.config.pruning_sparsity,
            prune_experts=True,
            expert_merge_threshold=0.95,
            finetune_epochs=1,
            finetune_lr=self.config.learning_rate * 0.1
        )
        
        pruner = ModelPruner(self.model, pruning_config)
        
        def eval_fn(model):
            metrics = self.evaluate(eval_loader)
            return metrics['loss']
            
        results = pruner.prune(eval_loader, eval_fn)
        
        self.logger.info(f"Pruning complete. Final sparsity: {results['final_sparsity']:.2%}")
        
    def save_model(self, filename: str):
        """Save model with all components"""
        
        save_path = self.output_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        
        if self.health_monitor:
            checkpoint['health_summary'] = self.health_monitor.get_health_summary()
            
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved model to {save_path}")
        
    def save_checkpoint(self, filename: str):
        """Save full training checkpoint"""
        
        save_path = self.output_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        
        if self.curriculum_scheduler:
            checkpoint['curriculum_state'] = self.curriculum_scheduler.get_state()
        if self.active_selector:
            checkpoint['active_learning_state'] = self.active_selector.selection_history
            
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced fine-tuning for MoE++ models")
    
    # Model
    parser.add_argument("--base-model", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--task-name", type=str, default="general", help="Task name")
    
    # Data
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval-data", type=str, help="Path to evaluation data")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    # Training
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--num-lora-experts", type=int, default=4, help="Number of LoRA experts")
    
    # Features
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--no-constitutional", action="store_true", help="Disable constitutional training")
    parser.add_argument("--no-active-learning", action="store_true", help="Disable active learning")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--no-uncertainty", action="store_true", help="Disable uncertainty")
    parser.add_argument("--no-pruning", action="store_true", help="Disable post-training pruning")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./finetuned_models", help="Output directory")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--wandb-project", type=str, default="moe-finetuning", help="W&B project")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create config
    config = AdvancedFinetuningConfig(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        task_name=args.task_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        num_lora_experts=args.num_lora_experts,
        use_lora=not args.no_lora,
        use_constitutional=not args.no_constitutional,
        use_active_learning=not args.no_active_learning,
        use_curriculum=not args.no_curriculum,
        use_uncertainty=not args.no_uncertainty,
        prune_after_training=not args.no_pruning
    )
    
    # Save config
    config_path = Path(config.output_dir) / config.experiment_name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(vars(config), f)
        
    # Initialize W&B if requested
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=config.experiment_name,
            config=vars(config)
        )
        
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = get_dataset(
        args.train_data,
        'text',
        split='train',
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    eval_dataset = None
    if args.eval_data:
        eval_dataset = get_dataset(
            args.eval_data,
            'text',
            split='validation',
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        
    # Create fine-tuner
    fine_tuner = AdvancedFineTuner(config)
    
    # Run fine-tuning
    logger.info("Starting advanced fine-tuning...")
    fine_tuner.fine_tune(train_dataset, eval_dataset)
    
    logger.info("Fine-tuning completed!")
    
    # Log final metrics to W&B
    if args.wandb:
        wandb.log(fine_tuner.metrics)
        wandb.finish()

if __name__ == "__main__":
    main()