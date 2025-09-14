"""
PPO (Proximal Policy Optimization) Training Pipeline
Implements RLHF with PPO for language model alignment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import wandb
from collections import deque
import time

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Model settings
    learning_rate: float = 1.4e-5
    batch_size: int = 128
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # PPO hyperparameters
    ppo_epochs: int = 4
    gamma: float = 1.0  # Discount factor
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.1
    
    # KL penalty
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    kl_horizon: int = 10000
    
    # Generation settings
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Training settings
    max_steps: int = 100000
    warmup_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Reward settings
    reward_model_path: Optional[str] = None
    use_reward_scaling: bool = True
    reward_scale: float = 1.0
    use_reward_baseline: bool = True
    
    # Memory optimization
    clear_cache_every_n_steps: int = 10
    use_gradient_checkpointing: bool = True
    
    # Logging
    log_with: str = "wandb"
    log_interval: int = 10

class AdaptiveKLController:
    """Adaptive KL penalty controller"""
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.value = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
        
    def update(self, current_kl: float, n_steps: int):
        """Update KL coefficient based on current KL divergence"""
        proportional_error = np.clip(current_kl / self.target_kl - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class RunningMeanStd:
    """Running mean and standard deviation"""
    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.epsilon = epsilon
        
    def update(self, x: np.ndarray):
        """Update running statistics"""
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values"""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

class PPOTrainer:
    """PPO trainer for RLHF"""
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        tokenizer: Any,
        config: PPOConfig,
        value_model: Optional[nn.Module] = None,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Value model (can be separate or shared with policy)
        self.value_model = value_model or self._create_value_head(policy_model)
        
        # Move models to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model = self.policy_model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.value_model = self.value_model.to(self.device)
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # KL controller
        self.kl_controller = AdaptiveKLController(
            config.init_kl_coef,
            config.target_kl,
            config.kl_horizon
        )
        
        # Reward statistics
        self.reward_stats = RunningMeanStd()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_model.parameters()),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Metrics
        self.metrics = {
            "rewards": [],
            "kl_divergence": [],
            "entropy": [],
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
        }
        
        # Experience buffer
        self.experience_buffer = []
    
    def _create_value_head(self, policy_model: nn.Module) -> nn.Module:
        """Create value head for the policy model"""
        class ValueHead(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.linear = nn.Linear(hidden_size, 1)
                
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # Use last hidden state for value prediction
                return self.linear(hidden_states[:, -1, :])
        
        hidden_size = policy_model.config.hidden_size
        return ValueHead(hidden_size)
    
    def generate_experience(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate experiences using the policy model"""
        max_length = max_length or self.config.max_length
        experiences = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                
                prompt_length = inputs["input_ids"].shape[1]
                
                # Generate response
                outputs = self.policy_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                response_ids = outputs.sequences[0, prompt_length:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Get log probabilities
                logits = torch.stack(outputs.scores, dim=1)
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(
                    -1, response_ids.unsqueeze(0).unsqueeze(-1)
                ).squeeze()
                
                # Get value estimates
                value_outputs = self.policy_model(
                    input_ids=outputs.sequences,
                    return_dict=True,
                    output_hidden_states=True,
                )
                values = self.value_model(value_outputs.hidden_states[-1])
                
                # Get rewards
                rewards = self._compute_rewards(
                    prompt,
                    response_text,
                    outputs.sequences[0]
                )
                
                # Store experience
                experience = {
                    "prompt": prompt,
                    "response": response_text,
                    "prompt_ids": inputs["input_ids"][0],
                    "response_ids": response_ids,
                    "values": values[0, prompt_length:].cpu(),
                    "log_probs": action_log_probs.cpu(),
                    "rewards": rewards,
                    "advantages": None,  # Computed later
                    "returns": None,  # Computed later
                }
                
                experiences.append(experience)
        
        return experiences
    
    def _compute_rewards(
        self,
        prompt: str,
        response: str,
        full_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rewards for generated response"""
        # Get reward from reward model
        with torch.no_grad():
            reward_outputs = self.reward_model(
                input_ids=full_ids.unsqueeze(0),
                return_dict=True,
            )
            
            # Assume reward model outputs scores
            reward_score = reward_outputs.logits[0, -1].item()
        
        # Get KL penalty
        with torch.no_grad():
            # Policy model log probs
            policy_outputs = self.policy_model(
                input_ids=full_ids.unsqueeze(0),
                return_dict=True,
            )
            policy_logits = policy_outputs.logits[0]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            
            # Reference model log probs
            ref_outputs = self.ref_model(
                input_ids=full_ids.unsqueeze(0),
                return_dict=True,
            )
            ref_logits = ref_outputs.logits[0]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            # KL divergence
            kl_div = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(-1)
        
        # Combine rewards
        prompt_length = len(self.tokenizer(prompt)["input_ids"])
        response_length = len(full_ids) - prompt_length
        
        rewards = torch.zeros(response_length)
        rewards[-1] = reward_score  # Terminal reward
        
        # Add KL penalty
        kl_penalty = -self.kl_controller.value * kl_div[prompt_length:]
        rewards = rewards + kl_penalty.cpu()
        
        return rewards
    
    def compute_advantages(self, experiences: List[Dict[str, Any]]) -> None:
        """Compute advantages using GAE"""
        for exp in experiences:
            rewards = exp["rewards"]
            values = exp["values"]
            
            # Add bootstrap value for GAE
            next_values = torch.cat([values[1:], torch.zeros(1)])
            
            # TD residuals
            deltas = rewards + self.config.gamma * next_values - values
            
            # GAE
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            
            for t in reversed(range(len(rewards))):
                advantages[t] = deltas[t] + self.config.gamma * self.config.lam * last_advantage
                last_advantage = advantages[t]
            
            # Compute returns
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            exp["advantages"] = advantages
            exp["returns"] = returns
    
    def train_step(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Single PPO training step"""
        # Prepare batch data
        batch_size = len(experiences)
        
        # Concatenate all data
        all_prompt_ids = []
        all_response_ids = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []
        
        for exp in experiences:
            response_len = len(exp["response_ids"])
            all_prompt_ids.extend([exp["prompt_ids"]] * response_len)
            all_response_ids.extend(exp["response_ids"].tolist())
            all_old_log_probs.extend(exp["log_probs"].tolist())
            all_advantages.extend(exp["advantages"].tolist())
            all_returns.extend(exp["returns"].tolist())
            all_values.extend(exp["values"].tolist())
        
        # Convert to tensors
        all_advantages = torch.tensor(all_advantages, device=self.device)
        all_returns = torch.tensor(all_returns, device=self.device)
        all_old_log_probs = torch.tensor(all_old_log_probs, device=self.device)
        all_values = torch.tensor(all_values, device=self.device)
        
        # PPO epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        for ppo_epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(all_response_ids))
            
            for start in range(0, len(indices), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_prompts = [all_prompt_ids[i] for i in batch_indices]
                batch_responses = [all_response_ids[i] for i in batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_returns = all_returns[batch_indices]
                batch_old_log_probs = all_old_log_probs[batch_indices]
                batch_old_values = all_values[batch_indices]
                
                # Forward pass
                # ... (construct input sequences)
                
                # Compute losses
                loss_dict = self._compute_loss(
                    batch_prompts,
                    batch_responses,
                    batch_advantages,
                    batch_returns,
                    batch_old_log_probs,
                    batch_old_values,
                )
                
                # Backward pass
                loss = loss_dict["total_loss"] / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (start // self.config.mini_batch_size + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy_model.parameters()) + list(self.value_model.parameters()),
                        1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Accumulate metrics
                total_loss += loss_dict["total_loss"].item()
                total_policy_loss += loss_dict["policy_loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                total_entropy += loss_dict["entropy"].item()
                total_kl += loss_dict["kl_divergence"].item()
        
        # Average metrics
        num_updates = self.config.ppo_epochs * (len(all_response_ids) // self.config.mini_batch_size)
        
        metrics = {
            "loss/total": total_loss / num_updates,
            "loss/policy": total_policy_loss / num_updates,
            "loss/value": total_value_loss / num_updates,
            "policy/entropy": total_entropy / num_updates,
            "policy/kl_divergence": total_kl / num_updates,
            "policy/kl_coef": self.kl_controller.value,
        }
        
        # Update KL controller
        self.kl_controller.update(metrics["policy/kl_divergence"], len(experiences))
        
        return metrics
    
    def _compute_loss(
        self,
        prompt_ids: List[torch.Tensor],
        response_ids: List[int],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss"""
        # This is simplified - full implementation would properly construct sequences
        # and compute losses for each token in the response
        
        # Placeholder losses
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        entropy = torch.tensor(0.0, device=self.device)
        kl_divergence = torch.tensor(0.0, device=self.device)
        
        total_loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss -
            self.config.entropy_coef * entropy
        )
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
        }
    
    def train(
        self,
        train_prompts: List[str],
        eval_prompts: Optional[List[str]] = None,
    ):
        """Main PPO training loop"""
        logger.info("Starting PPO training...")
        
        global_step = 0
        best_reward = float("-inf")
        
        # Initialize wandb
        if self.config.log_with == "wandb":
            wandb.init(project="ppo-training", config=self.config.__dict__)
        
        with tqdm(total=self.config.max_steps, desc="PPO Training") as pbar:
            while global_step < self.config.max_steps:
                # Sample prompts
                batch_prompts = np.random.choice(
                    train_prompts,
                    size=self.config.batch_size,
                    replace=True
                ).tolist()
                
                # Generate experiences
                experiences = self.generate_experience(batch_prompts)
                
                # Compute advantages
                self.compute_advantages(experiences)
                
                # Update reward statistics
                all_rewards = []
                for exp in experiences:
                    all_rewards.extend(exp["rewards"].numpy())
                self.reward_stats.update(np.array(all_rewards))
                
                # Normalize rewards if configured
                if self.config.use_reward_scaling:
                    for exp in experiences:
                        exp["rewards"] = torch.tensor(
                            self.reward_stats.normalize(exp["rewards"].numpy())
                        )
                        exp["returns"] = exp["advantages"] + exp["values"]
                
                # PPO update
                metrics = self.train_step(experiences)
                
                # Update metrics
                for key, value in metrics.items():
                    if key not in self.metrics:
                        self.metrics[key] = []
                    self.metrics[key].append(value)
                
                # Logging
                if global_step % self.config.log_interval == 0:
                    avg_reward = np.mean([exp["rewards"].sum().item() for exp in experiences])
                    metrics["reward/mean"] = avg_reward
                    
                    if self.config.log_with == "wandb":
                        wandb.log(metrics, step=global_step)
                    
                    logger.info(
                        f"Step {global_step}: "
                        f"reward={avg_reward:.3f}, "
                        f"loss={metrics['loss/total']:.3f}, "
                        f"kl={metrics['policy/kl_divergence']:.3f}"
                    )
                
                # Evaluation
                if eval_prompts and global_step % self.config.eval_steps == 0:
                    eval_reward = self.evaluate(eval_prompts)
                    
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        self.save_checkpoint(f"best_model_step_{global_step}")
                    
                    if self.config.log_with == "wandb":
                        wandb.log({"eval/reward": eval_reward}, step=global_step)
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{global_step}")
                
                # Clear cache
                if global_step % self.config.clear_cache_every_n_steps == 0:
                    torch.cuda.empty_cache()
                
                global_step += 1
                pbar.update(1)
        
        logger.info("PPO training completed!")
        
        if self.config.log_with == "wandb":
            wandb.finish()
    
    def evaluate(self, eval_prompts: List[str]) -> float:
        """Evaluate model on prompts"""
        total_reward = 0.0
        
        with torch.no_grad():
            for prompt in eval_prompts:
                experiences = self.generate_experience([prompt])
                reward = experiences[0]["rewards"].sum().item()
                total_reward += reward
        
        return total_reward / len(eval_prompts)
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint = {
            "policy_model": self.policy_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "kl_controller": self.kl_controller.__dict__,
            "reward_stats": self.reward_stats.__dict__,
            "metrics": self.metrics,
            "config": self.config,
        }
        
        torch.save(checkpoint, f"{name}.pt")
        logger.info(f"Saved checkpoint: {name}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_model.load_state_dict(checkpoint["policy_model"])
        self.value_model.load_state_dict(checkpoint["value_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        for key, value in checkpoint["kl_controller"].items():
            setattr(self.kl_controller, key, value)
        
        for key, value in checkpoint["reward_stats"].items():
            setattr(self.reward_stats, key, value)
        
        self.metrics = checkpoint["metrics"]
        
        logger.info(f"Loaded checkpoint: {path}")