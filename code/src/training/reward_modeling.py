"""
Multi-objective Reward Modeling
Trains reward models for helpfulness, harmlessness, and honesty
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
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer
from LLM.src.utils.path_utils import get_outputs_dir

logger = logging.getLogger(__name__)

@dataclass
class RewardModelConfig:
    """Configuration for reward model training"""
    # Model settings
    base_model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_objectives: int = 3  # helpfulness, harmlessness, honesty
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_length: int = 512
    
    # Multi-objective settings
    objective_names: List[str] = None
    objective_weights: List[float] = None
    use_multi_task_learning: bool = True
    shared_encoder: bool = True
    
    # Loss settings
    loss_type: str = "ranking"  # ranking, regression, classification
    margin: float = 0.5  # For ranking loss
    label_smoothing: float = 0.1
    
    # Data settings
    balance_objectives: bool = True
    augment_data: bool = True
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # Output
    output_dir: str = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = os.path.join(get_outputs_dir(), "reward_model")
    
    def __post_init__(self):
        if self.objective_names is None:
            self.objective_names = ["helpfulness", "harmlessness", "honesty"]
        
        if self.objective_weights is None:
            self.objective_weights = [1.0] * len(self.objective_names)

class RewardHead(nn.Module):
    """Reward prediction head for a single objective"""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MultiObjectiveRewardModel(nn.Module):
    """Multi-objective reward model with shared encoder"""
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        
        # Base encoder
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        
        # Freeze lower layers for efficiency
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Reward heads for each objective
        self.reward_heads = nn.ModuleDict({
            obj_name: RewardHead(config.hidden_size)
            for obj_name in config.objective_names
        })
        
        # Optional: Separate encoders for each objective
        if not config.shared_encoder:
            self.objective_encoders = nn.ModuleDict({
                obj_name: nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config.hidden_size,
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.1,
                    ),
                    num_layers=2,
                )
                for obj_name in config.objective_names
            })
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        objective: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            objective: Specific objective to compute (None for all)
            
        Returns:
            Dictionary of rewards for each objective
        """
        # Encode input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Use [CLS] token representation
        sequence_output = encoder_outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Compute rewards for specified objective(s)
        rewards = {}
        
        objectives_to_compute = [objective] if objective else self.config.objective_names
        
        for obj_name in objectives_to_compute:
            if obj_name not in self.reward_heads:
                continue
            
            # Apply objective-specific encoder if not shared
            if not self.config.shared_encoder:
                obj_features = self.objective_encoders[obj_name](
                    sequence_output.transpose(0, 1)
                ).transpose(0, 1)[:, 0, :]
            else:
                obj_features = cls_output
            
            # Compute reward
            reward = self.reward_heads[obj_name](obj_features)
            rewards[obj_name] = reward.squeeze(-1)
        
        return rewards

class RewardModelTrainer:
    """Trainer for multi-objective reward models"""
    def __init__(
        self,
        model: MultiObjectiveRewardModel,
        config: RewardModelConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.base_model_name)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Data loaders
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None
        
        # Metrics
        self.train_metrics = {obj: [] for obj in config.objective_names}
        self.eval_metrics = {obj: [] for obj in config.objective_names}
        
    def _create_dataloader(self, dataset: Any, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create data loader"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    
    def _collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        # Expected format:
        # - text_a: First text
        # - text_b: Second text (for comparison)
        # - labels: Dict of labels for each objective
        
        texts_a = [ex["text_a"] for ex in examples]
        texts_b = [ex.get("text_b", "") for ex in examples]
        
        # Tokenize
        if any(texts_b):
            # Pairwise comparison
            encodings = self.tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
        else:
            # Single text
            encodings = self.tokenizer(
                texts_a,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
        
        # Prepare labels
        labels = {}
        for obj_name in self.config.objective_names:
            obj_labels = [ex["labels"].get(obj_name, 0) for ex in examples]
            labels[obj_name] = torch.tensor(obj_labels, dtype=torch.float)
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    
    def compute_loss(
        self,
        rewards: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-objective loss"""
        losses = {}
        total_loss = 0.0
        
        for obj_name in self.config.objective_names:
            if obj_name not in rewards or obj_name not in labels:
                continue
            
            obj_rewards = rewards[obj_name]
            obj_labels = labels[obj_name]
            
            if self.config.loss_type == "ranking":
                # Pairwise ranking loss
                # Assume labels are preferences: 1 if first is better, 0 if second is better
                # Split rewards into pairs
                batch_size = obj_rewards.shape[0] // 2
                rewards_a = obj_rewards[:batch_size]
                rewards_b = obj_rewards[batch_size:]
                
                # Compute ranking loss
                diff = rewards_a - rewards_b
                loss = -F.logsigmoid(diff * (2 * obj_labels[:batch_size] - 1) * self.config.margin)
                
            elif self.config.loss_type == "regression":
                # MSE loss for regression
                loss = F.mse_loss(obj_rewards, obj_labels)
                
            elif self.config.loss_type == "classification":
                # Binary cross-entropy for classification
                loss = F.binary_cross_entropy_with_logits(obj_rewards, obj_labels)
                
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")
            
            # Apply label smoothing
            if self.config.label_smoothing > 0:
                loss = loss * (1 - self.config.label_smoothing) + \
                       self.config.label_smoothing * 0.5
            
            # Weight by objective importance
            weighted_loss = loss.mean() * self.config.objective_weights[
                self.config.objective_names.index(obj_name)
            ]
            
            losses[obj_name] = loss.mean()
            total_loss += weighted_loss
        
        return total_loss, losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
        
        # Forward pass
        rewards = self.model(input_ids, attention_mask)
        
        # Compute loss
        loss, objective_losses = self.compute_loss(rewards, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Prepare metrics
        metrics = {"loss": loss.item()}
        for obj_name, obj_loss in objective_losses.items():
            metrics[f"loss/{obj_name}"] = obj_loss.item()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model"""
        if not self.eval_dataloader:
            return {}
        
        self.model.eval()
        
        all_rewards = {obj: [] for obj in self.config.objective_names}
        all_labels = {obj: [] for obj in self.config.objective_names}
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
                
                # Forward pass
                rewards = self.model(input_ids, attention_mask)
                
                # Compute loss
                loss, _ = self.compute_loss(rewards, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and labels
                for obj_name in self.config.objective_names:
                    if obj_name in rewards:
                        all_rewards[obj_name].extend(rewards[obj_name].cpu().numpy())
                        all_labels[obj_name].extend(labels[obj_name].cpu().numpy())
        
        # Compute metrics
        metrics = {
            "eval_loss": total_loss / num_batches,
        }
        
        for obj_name in self.config.objective_names:
            if all_rewards[obj_name]:
                rewards_np = np.array(all_rewards[obj_name])
                labels_np = np.array(all_labels[obj_name])
                
                if self.config.loss_type == "ranking":
                    # Compute ranking accuracy
                    batch_size = len(rewards_np) // 2
                    rewards_a = rewards_np[:batch_size]
                    rewards_b = rewards_np[batch_size:]
                    predictions = (rewards_a > rewards_b).astype(float)
                    accuracy = accuracy_score(labels_np[:batch_size], predictions)
                    metrics[f"accuracy/{obj_name}"] = accuracy
                    
                elif self.config.loss_type == "classification":
                    # Compute classification metrics
                    predictions = (rewards_np > 0).astype(float)
                    accuracy = accuracy_score(labels_np, predictions)
                    try:
                        auc = roc_auc_score(labels_np, rewards_np)
                        metrics[f"auc/{obj_name}"] = auc
                    except:
                        pass
                    metrics[f"accuracy/{obj_name}"] = accuracy
                
                elif self.config.loss_type == "regression":
                    # Compute regression metrics
                    mse = np.mean((rewards_np - labels_np) ** 2)
                    metrics[f"mse/{obj_name}"] = mse
        
        self.model.train()
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting reward model training...")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Training epoch {epoch + 1}")
            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                
                # Update metrics
                for obj_name in self.config.objective_names:
                    if f"loss/{obj_name}" in metrics:
                        self.train_metrics[obj_name].append(metrics[f"loss/{obj_name}"])
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    progress_bar.set_postfix({"loss": metrics["loss"]})
                
                # Evaluation
                if global_step % self.config.eval_steps == 0 and self.eval_dataloader:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {global_step}: {eval_metrics}")
                    
                    # Save best model
                    if eval_metrics.get("eval_loss", float('inf')) < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        self.save_model("best")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"step_{global_step}")
                
                global_step += 1
            
            # End of epoch evaluation
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")
            
            if self.eval_dataloader:
                eval_metrics = self.evaluate()
                logger.info(f"Epoch {epoch + 1} eval metrics: {eval_metrics}")
        
        # Save final model
        self.save_model("final")
        logger.info("Training completed!")
    
    def save_model(self, tag: str):
        """Save model checkpoint"""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), output_dir / "model.pt")
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Saved model to {output_dir}")
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model weights
        self.model.load_state_dict(
            torch.load(checkpoint_path / "model.pt", map_location=self.device)
        )
        
        logger.info(f"Loaded model from {checkpoint_path}")

def create_reward_training_data(
    responses: List[Dict[str, Any]],
    objective_labelers: Dict[str, Callable],
) -> List[Dict[str, Any]]:
    """
    Create training data for reward models
    
    Args:
        responses: List of response dictionaries with 'prompt' and 'response' keys
        objective_labelers: Dictionary of labeling functions for each objective
        
    Returns:
        List of training examples
    """
    training_data = []
    
    # Create pairwise comparisons
    for i in range(0, len(responses) - 1, 2):
        response_a = responses[i]
        response_b = responses[i + 1]
        
        # Skip if different prompts
        if response_a["prompt"] != response_b["prompt"]:
            continue
        
        # Label each objective
        labels = {}
        for obj_name, labeler in objective_labelers.items():
            # Get scores for each response
            score_a = labeler(response_a["prompt"], response_a["response"])
            score_b = labeler(response_b["prompt"], response_b["response"])
            
            # Create preference label (1 if A is better, 0 if B is better)
            labels[obj_name] = 1.0 if score_a > score_b else 0.0
        
        # Create training example
        example = {
            "text_a": response_a["prompt"] + " " + response_a["response"],
            "text_b": response_b["prompt"] + " " + response_b["response"],
            "labels": labels,
        }
        
        training_data.append(example)
    
    return training_data