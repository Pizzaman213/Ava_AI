"""
Direct Preference Optimization (DPO) Trainer
Implements preference-based training without reward models
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import wandb
from pathlib import Path
from accelerate import Accelerator
from LLM.src.utils.path_utils import get_outputs_dir

logger = logging.getLogger(__name__)

@dataclass
class DPOConfig:
    """Configuration for DPO training"""
    # Model settings
    model_name_or_path: str = None
    ref_model_name_or_path: Optional[str] = None
    
    # Training hyperparameters
    beta: float = 0.1  # KL regularization coefficient
    learning_rate: float = 5e-7
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_length: int = 512
    max_prompt_length: int = 128
    
    # Loss settings
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    label_smoothing: float = 0.0
    ipo_tau: float = 0.05  # For IPO loss
    reference_free: bool = False  # For reference-free DPO
    
    # Optimization
    warmup_steps: int = 150
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # Data settings
    max_samples: Optional[int] = None
    remove_unused_columns: bool = True
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    
    # Output
    output_dir: str = None
    run_name: Optional[str] = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = os.path.join(get_outputs_dir(), "dpo")
    
    # Advanced features
    use_peft: bool = False  # Parameter-efficient fine-tuning
    peft_config: Optional[Dict[str, Any]] = None
    mixed_precision: str = "bf16"  # no, fp16, bf16
    gradient_checkpointing: bool = True
    
    # Logging
    report_to: str = "wandb"
    push_to_hub: bool = False

class DPODataCollator:
    """Data collator for DPO training"""
    def __init__(self, tokenizer, max_length: int, max_prompt_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into batch for DPO training
        
        Expected features format:
        - prompt: The input prompt
        - chosen: The preferred response
        - rejected: The dispreferred response
        """
        batch = {
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
        }
        
        for feature in features:
            # Tokenize prompt
            prompt_tokens = self.tokenizer(
                feature["prompt"],
                max_length=self.max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize chosen response
            chosen_tokens = self.tokenizer(
                feature["prompt"] + feature["chosen"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize rejected response
            rejected_tokens = self.tokenizer(
                feature["prompt"] + feature["rejected"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Add to batch
            batch["prompt_input_ids"].append(prompt_tokens["input_ids"])
            batch["prompt_attention_mask"].append(prompt_tokens["attention_mask"])
            batch["chosen_input_ids"].append(chosen_tokens["input_ids"])
            batch["chosen_attention_mask"].append(chosen_tokens["attention_mask"])
            batch["rejected_input_ids"].append(rejected_tokens["input_ids"])
            batch["rejected_attention_mask"].append(rejected_tokens["attention_mask"])
        
        # Stack tensors
        for key in batch:
            batch[key] = torch.cat(batch[key], dim=0)
        
        return batch

class DPOTrainer:
    """Direct Preference Optimization trainer"""
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module],
        config: DPOConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        tokenizer: Any = None,
        data_collator: Optional[DPODataCollator] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=config.report_to,
        )
        
        # Models
        self.model = model
        self.ref_model = ref_model or model  # Use same model if no ref provided
        
        # Move to device
        self.device = self.accelerator.device
        
        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Freeze reference model
        if ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # Data
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or DPODataCollator(
            tokenizer, config.max_length, config.max_prompt_length
        )
        
        # Create data loaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        
        if eval_dataset:
            self.eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                drop_last=False,
            )
        else:
            self.eval_dataloader = None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Metrics
        self.train_metrics = {
            "loss": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
        }
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Compute log probabilities for sequences"""
        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits
        
        # Shift for autoregressive
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask prompt tokens
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, prompt_length:] = True
        selected_log_probs = selected_log_probs * mask
        
        # Sum over sequence
        return selected_log_probs.sum(dim=-1)
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute DPO loss"""
        # Compute rewards (log ratios)
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # Compute loss based on type
        if self.config.loss_type == "sigmoid":
            # Standard DPO loss
            losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        
        elif self.config.loss_type == "hinge":
            # Hinge loss variant
            losses = torch.relu(1 - (chosen_rewards - rejected_rewards))
        
        elif self.config.loss_type == "ipo":
            # IPO (Identity Preference Optimization) loss
            losses = (chosen_rewards - rejected_rewards - self.config.ipo_tau) ** 2
        
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            losses = losses * (1 - self.config.label_smoothing) + \
                    self.config.label_smoothing * 0.5
        
        # Metrics
        chosen_rewards_mean = chosen_rewards.detach().mean()
        rejected_rewards_mean = rejected_rewards.detach().mean()
        reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
        reward_margins = (chosen_rewards - rejected_rewards).detach().mean()
        
        metrics = {
            "rewards/chosen": chosen_rewards_mean,
            "rewards/rejected": rejected_rewards_mean,
            "rewards/accuracies": reward_accuracies,
            "rewards/margins": reward_margins,
            "logps/chosen": policy_chosen_logps.detach().mean(),
            "logps/rejected": policy_rejected_logps.detach().mean(),
        }
        
        return losses.mean(), metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single training step"""
        # Get prompt length
        prompt_length = batch["prompt_input_ids"].shape[1]
        
        # Compute policy log probs
        policy_chosen_logps = self.compute_log_probs(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            prompt_length,
        )
        
        policy_rejected_logps = self.compute_log_probs(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            prompt_length,
        )
        
        # Compute reference log probs
        if self.config.reference_free:
            # Reference-free DPO
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            reference_chosen_logps = self.compute_log_probs(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                prompt_length,
            )
            
            reference_rejected_logps = self.compute_log_probs(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                prompt_length,
            )
        
        # Compute loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        metrics["loss"] = loss
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting DPO training...")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Total optimization steps = {len(self.train_dataloader) * self.config.num_epochs}")
        
        # Initialize tracking
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="dpo-training",
                config=self.config.__dict__,
                init_kwargs={"wandb": {"name": self.config.run_name}} if self.config.run_name else {}
            )
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            train_loss = 0.0
            
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    metrics = self.training_step(batch)
                    loss = metrics["loss"]
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update metrics
                train_loss += loss.item()
                for key, value in metrics.items():
                    if key not in self.train_metrics:
                        self.train_metrics[key] = []
                    self.train_metrics[key].append(value.item() if torch.is_tensor(value) else value)
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_metrics = {
                        f"train/{key}": np.mean(values[-self.config.logging_steps:])
                        for key, values in self.train_metrics.items()
                    }
                    avg_metrics["train/global_step"] = self.global_step
                    
                    self.accelerator.log(avg_metrics, step=self.global_step)
                    
                    progress_bar.set_postfix({
                        "loss": avg_metrics["train/loss"],
                        "acc": avg_metrics["train/rewards/accuracies"],
                    })
                
                # Evaluation
                if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    
                    eval_metrics_log = {f"eval/{key}": value for key, value in eval_metrics.items()}
                    self.accelerator.log(eval_metrics_log, step=self.global_step)
                    
                    logger.info(
                        f"Step {self.global_step} - "
                        f"Eval loss: {eval_metrics['loss']:.4f}, "
                        f"Eval accuracy: {eval_metrics['rewards/accuracies']:.4f}"
                    )
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            # End of epoch logging
            avg_train_loss = train_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch + 1} - Average train loss: {avg_train_loss:.4f}")
        
        # Final save
        self.save_checkpoint("final")
        
        # End tracking
        self.accelerator.end_training()
        
        logger.info("DPO training completed!")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        eval_metrics = {key: [] for key in self.train_metrics}
        
        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process,
            ):
                metrics = self.training_step(batch)
                
                for key, value in metrics.items():
                    if key in eval_metrics:
                        eval_metrics[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Average metrics
        avg_eval_metrics = {key: np.mean(values) for key, values in eval_metrics.items()}
        
        self.model.train()
        return avg_eval_metrics
    
    def save_checkpoint(self, tag: Optional[str] = None):
        """Save model checkpoint"""
        tag = tag or f"step_{self.global_step}"
        output_dir = Path(self.config.output_dir) / f"checkpoint-{tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if self.accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir)
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
            
            # Save training state
            training_state = {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "config": self.config.__dict__,
                "metrics": self.train_metrics,
            }
            torch.save(training_state, output_dir / "training_state.pt")
        
        logger.info(f"Saved checkpoint to {output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path)
            self.global_step = training_state["global_step"]
            self.current_epoch = training_state["current_epoch"]
            self.train_metrics = training_state["metrics"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

def create_dpo_trainer(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    config: Optional[DPOConfig] = None,
    tokenizer: Optional[Any] = None,
) -> DPOTrainer:
    """
    Create DPO trainer instance
    
    Args:
        model: Policy model to train
        ref_model: Reference model (optional)
        train_dataset: Training dataset with preference pairs
        eval_dataset: Evaluation dataset (optional)
        config: DPO configuration
        tokenizer: Tokenizer
        
    Returns:
        DPOTrainer instance
    """
    config = config or DPOConfig()
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    return trainer