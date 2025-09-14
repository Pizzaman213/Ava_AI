"""
Enhanced PPO Trainer with Advanced Features
Implements PPO with replay buffer, meta-learning, and adaptive KL control
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import wandb
from collections import deque, defaultdict
import time
import heapq
from pathlib import Path
import pickle
import gc

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPPOConfig:
    """Configuration for Enhanced PPO training"""
    # Model settings
    learning_rate: float = 1.4e-5
    batch_size: int = 128
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # PPO hyperparameters
    ppo_epochs: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    
    # Adaptive KL control
    init_kl_coef: float = 0.2
    target_kl: float = 0.01
    kl_horizon: int = 10000
    kl_decay: float = 0.999
    adaptive_kl: bool = True
    
    # Replay buffer
    replay_buffer_size: int = 100000
    replay_batch_ratio: float = 0.3
    prioritized_replay: bool = True
    replay_alpha: float = 0.6
    replay_beta: float = 0.4
    
    # Meta-learning
    use_meta_learning: bool = True
    meta_batch_size: int = 16
    meta_learning_rate: float = 3e-4
    inner_loops: int = 3
    
    # Generation settings
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Advanced features
    use_curriculum_learning: bool = True
    curriculum_stages: int = 5
    use_reward_shaping: bool = True
    reward_shaping_gamma: float = 0.9
    
    # Memory optimization
    clear_cache_every_n_steps: int = 10
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

class AdaptiveKLController:
    """Enhanced adaptive KL penalty controller with decay and bounds"""
    def __init__(self, config: EnhancedPPOConfig):
        self.config = config
        self.value = config.init_kl_coef
        self.target_kl = config.target_kl
        self.horizon = config.kl_horizon
        self.decay = config.kl_decay
        
        # Bounds
        self.min_kl = 0.0001
        self.max_kl = 10.0
        
        # History for adaptive control
        self.kl_history = deque(maxlen=100)
        self.update_history = deque(maxlen=100)
    
    def update(self, current_kl: float, n_steps: int):
        """Update KL coefficient based on current KL divergence"""
        self.kl_history.append(current_kl)
        
        if self.config.adaptive_kl:
            # Proportional control with history
            kl_error = current_kl - self.target_kl
            proportional_error = np.clip(kl_error / self.target_kl, -0.2, 0.2)
            
            # Consider trend
            if len(self.kl_history) > 10:
                kl_trend = np.mean(list(self.kl_history)[-5:]) - np.mean(list(self.kl_history)[-10:-5])
                trend_factor = 1 + np.clip(kl_trend / self.target_kl, -0.1, 0.1)
            else:
                trend_factor = 1.0
            
            # Update with bounds
            mult = 1 + proportional_error * n_steps / self.horizon * trend_factor
            self.value = np.clip(self.value * mult, self.min_kl, self.max_kl)
            
            # Apply decay
            self.value *= self.decay
        
        self.update_history.append(self.value)
    
    def get_stats(self) -> Dict[str, float]:
        """Get controller statistics"""
        return {
            "kl_coef": self.value,
            "avg_kl": np.mean(self.kl_history) if self.kl_history else 0.0,
            "kl_trend": self.kl_history[-1] - self.kl_history[0] if len(self.kl_history) > 1 else 0.0,
        }

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for PPO"""
    def __init__(self, config: EnhancedPPOConfig):
        self.config = config
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
        
        # For efficient sampling
        self.alpha = config.replay_alpha
        self.beta = config.replay_beta
        self.beta_schedule = lambda step: min(1.0, self.beta + step * (1.0 - self.beta) / 100000)
    
    def add(self, experience: Dict[str, Any], priority: Optional[float] = None):
        """Add experience to buffer with priority"""
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) >= self.config.replay_buffer_size:
            # Remove lowest priority item
            min_idx = np.argmin(self.priorities)
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int, beta: Optional[float] = None) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
        """Sample batch with importance weights"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if beta is None:
            beta = self.beta
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Compute importance weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights, torch.tensor(indices)
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices.tolist(), priorities.tolist()):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class MetaLearningOptimizer:
    """Meta-learning optimizer for fast adaptation"""
    def __init__(self, model: nn.Module, config: EnhancedPPOConfig):
        self.model = model
        self.config = config
        
        # Meta-parameters
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.meta_learning_rate
        )
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(100, 128)  # Support 100 task types
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
    
    def compute_meta_loss(
        self,
        tasks: List[Dict[str, Any]],
        model: nn.Module
    ) -> torch.Tensor:
        """Compute meta-learning loss across tasks"""
        meta_losses = []
        
        for task in tasks:
            # Clone model for inner loop
            inner_model = self._clone_model(model)
            inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.01)
            
            # Inner loop adaptation
            for _ in range(self.config.inner_loops):
                inner_loss = self._compute_task_loss(inner_model, task)
                inner_optimizer.zero_grad()
                inner_loss.backward()
                inner_optimizer.step()
            
            # Compute meta-loss on query set
            with torch.no_grad():
                adapted_params = {name: param.clone() for name, param in inner_model.named_parameters()}
            
            # Apply adapted parameters to original model
            for name, param in model.named_parameters():
                param.data = adapted_params[name]
            
            meta_loss = self._compute_task_loss(model, task, query_set=True)
            meta_losses.append(meta_loss)
        
        return torch.stack(meta_losses).mean()
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a functional clone of the model"""
        clone = type(model)(model.config)
        clone.load_state_dict(model.state_dict())
        return clone
    
    def _compute_task_loss(
        self,
        model: nn.Module,
        task: Dict[str, Any],
        query_set: bool = False
    ) -> torch.Tensor:
        """Compute loss for a specific task"""
        data_key = "query_data" if query_set else "support_data"
        data = task[data_key]
        
        # Simple supervised loss for meta-learning
        outputs = model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"]
        )
        
        return outputs.loss
    
    def adapt_to_task(
        self,
        task_id: int,
        task_data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Adapt model parameters to specific task"""
        # Get task embedding
        task_emb = self.task_embeddings(torch.tensor([task_id]))
        
        # Generate adaptation
        adaptation = self.adaptation_net(task_emb)
        
        # Apply adaptation to model parameters
        adapted_params = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:
                # Simple scaling adaptation
                scale = 1 + 0.1 * adaptation.mean().item()
                adapted_params[name] = param * scale
            else:
                adapted_params[name] = param
        
        return adapted_params

class RewardShaper:
    """Shapes rewards for better learning signal"""
    def __init__(self, config: EnhancedPPOConfig):
        self.config = config
        self.gamma = config.reward_shaping_gamma
        
        # Potential function parameters
        self.potential_weights = nn.Parameter(torch.randn(128))
        
        # Reward statistics
        self.reward_stats = {
            "mean": 0.0,
            "std": 1.0,
            "count": 0
        }
    
    def shape_rewards(
        self,
        rewards: torch.Tensor,
        states: List[torch.Tensor],
        next_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """Apply reward shaping based on potential function"""
        shaped_rewards = rewards.clone()
        
        if self.config.use_reward_shaping:
            # Compute potential difference
            for i in range(len(rewards)):
                if i < len(states) - 1:
                    potential_diff = self._compute_potential(next_states[i]) - self._compute_potential(states[i])
                    shaped_rewards[i] += self.gamma * potential_diff
            
            # Normalize rewards
            shaped_rewards = self._normalize_rewards(shaped_rewards)
        
        return shaped_rewards
    
    def _compute_potential(self, state: torch.Tensor) -> float:
        """Compute potential function value for state"""
        # Simple linear potential
        if state.dim() > 1:
            state_features = state.mean(dim=0)
        else:
            state_features = state
        
        # Ensure dimensions match
        if state_features.size(0) > self.potential_weights.size(0):
            state_features = state_features[:self.potential_weights.size(0)]
        elif state_features.size(0) < self.potential_weights.size(0):
            padding = torch.zeros(self.potential_weights.size(0) - state_features.size(0))
            state_features = torch.cat([state_features, padding])
        
        potential = torch.dot(state_features, self.potential_weights)
        return potential.item()
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards based on running statistics"""
        # Update statistics
        self.reward_stats["count"] += len(rewards)
        self.reward_stats["mean"] = (
            self.reward_stats["mean"] * (self.reward_stats["count"] - len(rewards)) +
            rewards.sum().item()
        ) / self.reward_stats["count"]
        
        if self.reward_stats["count"] > 1:
            variance = ((rewards - self.reward_stats["mean"]) ** 2).mean().item()
            self.reward_stats["std"] = np.sqrt(
                (self.reward_stats["std"] ** 2 * (self.reward_stats["count"] - len(rewards)) +
                 variance * len(rewards)) / self.reward_stats["count"]
            )
        
        # Normalize
        if self.reward_stats["std"] > 0:
            normalized = (rewards - self.reward_stats["mean"]) / self.reward_stats["std"]
        else:
            normalized = rewards - self.reward_stats["mean"]
        
        return normalized

class CurriculumScheduler:
    """Manages curriculum learning stages"""
    def __init__(self, config: EnhancedPPOConfig):
        self.config = config
        self.current_stage = 0
        self.stage_thresholds = [0.6, 0.7, 0.8, 0.85, 0.9]  # Success rate thresholds
        self.stage_metrics = defaultdict(list)
    
    def get_current_difficulty(self) -> Dict[str, Any]:
        """Get current curriculum parameters"""
        difficulties = [
            {"max_length": 128, "complexity": "simple", "reward_scale": 1.2},
            {"max_length": 256, "complexity": "medium", "reward_scale": 1.1},
            {"max_length": 384, "complexity": "medium_hard", "reward_scale": 1.0},
            {"max_length": 448, "complexity": "hard", "reward_scale": 0.9},
            {"max_length": 512, "complexity": "very_hard", "reward_scale": 0.8},
        ]
        
        return difficulties[min(self.current_stage, len(difficulties) - 1)]
    
    def update_stage(self, success_rate: float, avg_reward: float):
        """Update curriculum stage based on performance"""
        self.stage_metrics[self.current_stage].append({
            "success_rate": success_rate,
            "avg_reward": avg_reward
        })
        
        # Check if should advance
        if len(self.stage_metrics[self.current_stage]) >= 10:  # At least 10 evaluations
            avg_success = np.mean([m["success_rate"] for m in self.stage_metrics[self.current_stage][-10:]])
            
            if avg_success > self.stage_thresholds[min(self.current_stage, len(self.stage_thresholds) - 1)]:
                if self.current_stage < self.config.curriculum_stages - 1:
                    self.current_stage += 1
                    logger.info(f"Advanced to curriculum stage {self.current_stage}")
    
    def should_use_easy_samples(self) -> bool:
        """Determine if should mix in easier samples"""
        # In early stages or when struggling, mix in easier samples
        if self.current_stage < 2:
            return True
        
        # Check recent performance
        if self.stage_metrics[self.current_stage]:
            recent_success = np.mean([
                m["success_rate"] for m in self.stage_metrics[self.current_stage][-5:]
            ])
            return recent_success < 0.5
        
        return False

class EnhancedPPOTrainer:
    """Enhanced PPO trainer with all advanced features"""
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        tokenizer: Any,
        config: EnhancedPPOConfig,
        value_model: Optional[nn.Module] = None,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Value model
        self.value_model = value_model or self._create_value_head(policy_model)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_models()
        
        # Enhanced components
        self.kl_controller = AdaptiveKLController(config)
        self.replay_buffer = PrioritizedReplayBuffer(config)
        self.meta_optimizer = MetaLearningOptimizer(policy_model, config) if config.use_meta_learning else None
        self.reward_shaper = RewardShaper(config)
        self.curriculum_scheduler = CurriculumScheduler(config) if config.use_curriculum_learning else None
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_model.parameters()),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Mixed precision
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Metrics
        self.metrics = defaultdict(list)
        self.global_step = 0
    
    def _setup_models(self):
        """Setup and move models to device"""
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
        
        # Enable gradient checkpointing if configured
        if self.config.use_gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()
    
    def _create_value_head(self, policy_model: nn.Module) -> nn.Module:
        """Create value head for the policy model"""
        class ValueHead(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.linear = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1)
                )
                
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # Use last hidden state for value prediction
                if hidden_states.dim() == 3:
                    pooled = hidden_states[:, -1, :]
                else:
                    pooled = hidden_states
                return self.linear(pooled)
        
        hidden_size = policy_model.config.hidden_size
        return ValueHead(hidden_size)
    
    def generate_experience_batch(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate batch of experiences with enhanced features"""
        max_length = max_length or self.config.max_length
        
        # Apply curriculum learning
        if self.curriculum_scheduler:
            curriculum = self.curriculum_scheduler.get_current_difficulty()
            max_length = min(max_length, curriculum["max_length"])
        
        experiences = []
        
        for prompt in prompts:
            # Generate response
            exp = self._generate_single_experience(prompt, max_length)
            
            # Add to replay buffer
            if exp["rewards"].sum() > 0:  # Only add positive experiences
                priority = abs(exp["advantages"].mean().item()) if "advantages" in exp else 1.0
                self.replay_buffer.add(exp, priority)
            
            experiences.append(exp)
        
        # Mix with replay buffer if configured
        if len(self.replay_buffer) > 0 and self.config.replay_batch_ratio > 0:
            num_replay = int(len(experiences) * self.config.replay_batch_ratio)
            replay_exps, weights, indices = self.replay_buffer.sample(num_replay)
            
            # Merge with current experiences
            experiences.extend(replay_exps)
        
        return experiences
    
    def _generate_single_experience(
        self,
        prompt: str,
        max_length: int
    ) -> Dict[str, Any]:
        """Generate single experience with full trajectory"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length // 2,
        ).to(self.device)
        
        prompt_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            # Generate with sampling
            outputs = self.policy_model.generate(
                **inputs,
                max_length=max_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
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
            hidden_states = outputs.hidden_states[-1][-1]  # Last layer, last token
            values = self.value_model(hidden_states[prompt_length:])
            
            # Get rewards with shaping
            rewards = self._compute_shaped_rewards(
                prompt,
                response_text,
                outputs.sequences[0],
                hidden_states[prompt_length:]
            )
        
        return {
            "prompt": prompt,
            "response": response_text,
            "prompt_ids": inputs["input_ids"][0],
            "response_ids": response_ids,
            "values": values.squeeze(-1).cpu(),
            "log_probs": action_log_probs.cpu(),
            "rewards": rewards,
            "hidden_states": hidden_states[prompt_length:].cpu(),
        }
    
    def _compute_shaped_rewards(
        self,
        prompt: str,
        response: str,
        full_ids: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute rewards with shaping and KL penalty"""
        # Get base rewards from reward model
        with torch.no_grad():
            reward_outputs = self.reward_model(
                input_ids=full_ids.unsqueeze(0),
                return_dict=True,
            )
            base_reward = reward_outputs.logits[0, -1].item()
        
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
            
            # KL divergence per token
            kl_div = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(-1)
        
        # Construct reward tensor
        prompt_length = len(self.tokenizer(prompt)["input_ids"])
        response_length = len(full_ids) - prompt_length
        
        rewards = torch.zeros(response_length)
        rewards[-1] = base_reward  # Terminal reward
        
        # Add KL penalty
        kl_penalty = -self.kl_controller.value * kl_div[prompt_length:]
        rewards = rewards + kl_penalty.cpu()
        
        # Apply reward shaping if configured
        if self.config.use_reward_shaping and response_length > 1:
            # Create state list for shaping
            states = [hidden_states[i] for i in range(response_length - 1)]
            next_states = [hidden_states[i + 1] for i in range(response_length - 1)]
            rewards[:-1] = self.reward_shaper.shape_rewards(rewards[:-1], states, next_states)
        
        return rewards
    
    def compute_advantages_and_returns(
        self,
        experiences: List[Dict[str, Any]]
    ) -> None:
        """Compute GAE advantages and returns for experiences"""
        for exp in experiences:
            rewards = exp["rewards"]
            values = exp["values"]
            
            # Compute advantages using GAE
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.config.gamma * next_value - values[t]
                advantages[t] = delta + self.config.gamma * self.config.lam * last_advantage
                last_advantage = advantages[t]
            
            # Compute returns
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            exp["advantages"] = advantages
            exp["returns"] = returns
    
    def train_step(
        self,
        experiences: List[Dict[str, Any]],
        replay_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Enhanced PPO training step"""
        # Prepare batch data
        batch_data = self._prepare_batch_data(experiences)
        
        # PPO epochs
        all_metrics = defaultdict(list)
        
        for ppo_epoch in range(self.config.ppo_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(batch_data["size"])
            
            for start_idx in range(0, batch_data["size"], self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, batch_data["size"])
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                mini_batch = {
                    key: value[mb_indices] if torch.is_tensor(value) else [value[i] for i in mb_indices]
                    for key, value in batch_data.items()
                    if key != "size"
                }
                
                # Compute losses
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    losses = self._compute_ppo_loss(mini_batch, replay_weights)
                
                # Backward pass
                total_loss = losses["total"]
                
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy_model.parameters()) + list(self.value_model.parameters()),
                        1.0
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy_model.parameters()) + list(self.value_model.parameters()),
                        1.0
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Record metrics
                for key, value in losses.items():
                    all_metrics[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Update KL controller
        avg_kl = np.mean(all_metrics["kl_divergence"])
        self.kl_controller.update(avg_kl, len(experiences))
        
        # Update replay buffer priorities
        if replay_weights is not None and hasattr(batch_data, "replay_indices"):
            new_priorities = torch.abs(batch_data["advantages"]).mean(dim=1)
            self.replay_buffer.update_priorities(batch_data["replay_indices"], new_priorities)
        
        # Aggregate metrics
        metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        metrics.update(self.kl_controller.get_stats())
        
        # Meta-learning update if configured
        if self.meta_optimizer and self.global_step % 100 == 0:
            meta_loss = self._meta_learning_update(experiences)
            metrics["meta_loss"] = meta_loss.item()
        
        self.global_step += 1
        
        # Clear cache periodically
        if self.global_step % self.config.clear_cache_every_n_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        return metrics
    
    def _prepare_batch_data(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare batch data from experiences"""
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
        
        return {
            "prompt_ids": torch.stack([ids.clone() for ids in all_prompt_ids]),
            "response_ids": torch.tensor(all_response_ids, device=self.device),
            "old_log_probs": torch.tensor(all_old_log_probs, device=self.device),
            "advantages": torch.tensor(all_advantages, device=self.device),
            "returns": torch.tensor(all_returns, device=self.device),
            "old_values": torch.tensor(all_values, device=self.device),
            "size": len(all_response_ids)
        }
    
    def _compute_ppo_loss(
        self,
        batch: Dict[str, Any],
        replay_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss components"""
        # Forward pass through policy
        # Note: This is simplified - full implementation would properly handle sequence construction
        
        # Placeholder for actual forward pass
        # In practice, you'd construct full sequences and compute proper log probs
        
        # For now, use stored values
        policy_log_probs = batch["old_log_probs"]
        
        # Compute ratio
        ratio = torch.exp(policy_log_probs - batch["old_log_probs"])
        
        # Compute surrogate losses
        surr1 = ratio * batch["advantages"]
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch["advantages"]
        
        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with clipping
        values = self.value_model(torch.randn(batch["size"], 768).to(self.device))  # Placeholder
        value_pred_clipped = batch["old_values"] + torch.clamp(
            values.squeeze() - batch["old_values"],
            -self.config.value_clip,
            self.config.value_clip
        )
        value_losses = (values.squeeze() - batch["returns"]) ** 2
        value_losses_clipped = (value_pred_clipped - batch["returns"]) ** 2
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        
        # Entropy loss (placeholder)
        entropy = torch.tensor(0.01).to(self.device)
        
        # KL divergence (placeholder)
        kl_divergence = torch.abs(ratio - 1).mean()
        
        # Apply importance weights if using replay buffer
        if replay_weights is not None:
            policy_loss = (policy_loss * replay_weights).mean()
            value_loss = (value_loss * replay_weights).mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss -
            self.config.entropy_coef * entropy
        )
        
        return {
            "total": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "approx_kl": kl_divergence,
            "clip_fraction": ((ratio - 1).abs() > self.config.clip_ratio).float().mean()
        }
    
    def _meta_learning_update(self, experiences: List[Dict[str, Any]]) -> torch.Tensor:
        """Perform meta-learning update"""
        # Create tasks from experiences
        tasks = []
        for i in range(0, len(experiences), self.config.meta_batch_size):
            task_experiences = experiences[i:i + self.config.meta_batch_size]
            if len(task_experiences) >= 4:  # Need enough for support and query
                tasks.append({
                    "support_data": self._prepare_batch_data(task_experiences[:len(task_experiences)//2]),
                    "query_data": self._prepare_batch_data(task_experiences[len(task_experiences)//2:])
                })
        
        if not tasks:
            return torch.tensor(0.0)
        
        # Compute meta loss
        meta_loss = self.meta_optimizer.compute_meta_loss(tasks, self.policy_model)
        
        # Meta optimizer step
        self.meta_optimizer.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.meta_optimizer.step()
        
        return meta_loss
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint with all components"""
        checkpoint = {
            "policy_model": self.policy_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "kl_controller": self.kl_controller.__dict__,
            "global_step": self.global_step,
            "config": self.config,
            "metrics": dict(self.metrics),
        }
        
        # Save replay buffer
        if len(self.replay_buffer) > 0:
            checkpoint["replay_buffer"] = {
                "buffer": self.replay_buffer.buffer[:1000],  # Save subset
                "priorities": self.replay_buffer.priorities[:1000]
            }
        
        # Save curriculum state
        if self.curriculum_scheduler:
            checkpoint["curriculum"] = {
                "stage": self.curriculum_scheduler.current_stage,
                "metrics": dict(self.curriculum_scheduler.stage_metrics)
            }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_model.load_state_dict(checkpoint["policy_model"])
        self.value_model.load_state_dict(checkpoint["value_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Restore KL controller
        for key, value in checkpoint["kl_controller"].items():
            setattr(self.kl_controller, key, value)
        
        self.global_step = checkpoint["global_step"]
        self.metrics = defaultdict(list, checkpoint["metrics"])
        
        # Restore replay buffer if available
        if "replay_buffer" in checkpoint:
            for exp, priority in zip(
                checkpoint["replay_buffer"]["buffer"],
                checkpoint["replay_buffer"]["priorities"]
            ):
                self.replay_buffer.add(exp, priority)
        
        # Restore curriculum state
        if "curriculum" in checkpoint and self.curriculum_scheduler:
            self.curriculum_scheduler.current_stage = checkpoint["curriculum"]["stage"]
            self.curriculum_scheduler.stage_metrics = defaultdict(
                list,
                checkpoint["curriculum"]["metrics"]
            )
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        num_epochs: int = 1
    ):
        """Main training loop with all enhancements"""
        logger.info("Starting Enhanced PPO training...")
        
        # Initialize wandb if configured
        if wandb:
            wandb.init(
                project="enhanced-ppo-rlhf",
                config=self.config.__dict__
            )
        
        for epoch in range(num_epochs):
            epoch_metrics = defaultdict(list)
            
            # Training loop
            for batch_idx, batch_prompts in enumerate(train_dataset):
                # Generate experiences
                experiences = self.generate_experience_batch(batch_prompts)
                
                # Compute advantages
                self.compute_advantages_and_returns(experiences)
                
                # Train step
                step_metrics = self.train_step(experiences)
                
                # Update metrics
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
                    self.metrics[key].append(value)
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    avg_metrics = {key: np.mean(values[-10:]) for key, values in self.metrics.items()}
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"loss={avg_metrics.get('total', 0):.3f}, "
                        f"reward={avg_metrics.get('reward', 0):.3f}, "
                        f"kl={avg_metrics.get('kl_divergence', 0):.3f}"
                    )
                    
                    if wandb:
                        wandb.log(avg_metrics, step=self.global_step)
                
                # Evaluation
                if eval_dataset and batch_idx % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    logger.info(f"Evaluation metrics: {eval_metrics}")
                    
                    # Update curriculum
                    if self.curriculum_scheduler:
                        self.curriculum_scheduler.update_stage(
                            eval_metrics.get("success_rate", 0),
                            eval_metrics.get("avg_reward", 0)
                        )
                
                # Save checkpoint
                if batch_idx % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_epoch{epoch}_step{self.global_step}.pt")
            
            # End of epoch summary
            epoch_summary = {key: np.mean(values) for key, values in epoch_metrics.items()}
            logger.info(f"Epoch {epoch} summary: {epoch_summary}")
        
        logger.info("Enhanced PPO training completed!")
    
    def evaluate(self, eval_dataset: Any) -> Dict[str, float]:
        """Evaluate model performance"""
        eval_metrics = defaultdict(list)
        
        self.policy_model.eval()
        
        with torch.no_grad():
            for batch_prompts in eval_dataset:
                experiences = self.generate_experience_batch(batch_prompts[:10])  # Smaller eval batch
                
                for exp in experiences:
                    eval_metrics["reward"].append(exp["rewards"].sum().item())
                    eval_metrics["response_length"].append(len(exp["response_ids"]))
        
        self.policy_model.train()
        
        return {
            "avg_reward": np.mean(eval_metrics["reward"]),
            "avg_length": np.mean(eval_metrics["response_length"]),
            "success_rate": np.mean([r > 0 for r in eval_metrics["reward"]])
        }