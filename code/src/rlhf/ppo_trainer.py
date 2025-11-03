"""
PPO (Proximal Policy Optimization) Trainer for RLHF

Implements PPO algorithm for fine-tuning language models with rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 128
    mini_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    gamma: float = 1.0  # Discount factor
    lambda_: float = 0.95  # GAE lambda

    # KL divergence penalty
    kl_penalty: str = 'kl'  # 'kl', 'abs', or 'mse'
    target_kl: float = 0.01
    init_kl_coef: float = 0.2
    adaptive_kl: bool = True

    # Generation settings
    max_gen_length: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # Training settings
    num_train_epochs: int = 1
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Advanced settings
    use_score_scaling: bool = True
    use_score_norm: bool = True
    whiten_rewards: bool = True
    clip_reward: Optional[float] = 10.0

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = 'bf16'  # 'no', 'fp16', 'bf16'


class PPOTrainer:
    """
    PPO Trainer for RLHF fine-tuning.

    This trainer implements Proximal Policy Optimization for fine-tuning
    language models using reward signals.
    """

    def __init__(
        self,
        config: PPOConfig,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: Any,
        tokenizer: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: str = 'cuda'
    ):
        """
        Initialize PPO trainer.

        Args:
            config: PPO configuration
            model: Policy model to train
            ref_model: Reference model (frozen) for KL penalty
            reward_model: Reward model for scoring outputs
            tokenizer: Tokenizer
            optimizer: Optimizer (created if not provided)
            lr_scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        # Adaptive KL coefficient
        self.kl_coef = config.init_kl_coef

        # Statistics tracking
        self.stats = {
            'rewards': deque(maxlen=100),
            'kl_div': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100)
        }

        # Step counter
        self.global_step = 0

        # Mixed precision
        self.scaler = None
        if config.mixed_precision == 'fp16':
            self.scaler = torch.cuda.amp.GradScaler()

        logger.info(f"Initialized PPO Trainer with config: {config}")

    def generate_responses(
        self,
        prompts: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Generate responses for given prompts.

        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length

        Returns:
            Tuple of (input_ids, attention_mask, decoded_responses)
        """
        max_length = max_length or self.config.max_gen_length

        # Encode prompts
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Generate
        self.model.eval()
        with torch.no_grad():
            gen_output = self.model.generate(  # type: ignore[call-arg]
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode responses
        responses = self.tokenizer.batch_decode(gen_output, skip_special_tokens=True)

        return gen_output, attention_mask, responses

    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.

        Args:
            input_ids: Generated token IDs
            attention_mask: Attention mask
            prompts: Original prompts
            responses: Generated responses

        Returns:
            Reward tensor [batch_size]
        """
        # Get rewards from reward model
        if hasattr(self.reward_model, 'rate_responses'):
            # ModelToModelReward
            rewards = self.reward_model.rate_responses(prompts, responses)
        else:
            # Standard RewardModel
            with torch.no_grad():
                reward_output = self.reward_model(input_ids, attention_mask)
                if isinstance(reward_output, dict):
                    rewards = reward_output['rewards']
                else:
                    rewards = reward_output

        # Apply reward processing
        if self.config.clip_reward is not None:
            rewards = torch.clamp(rewards, -self.config.clip_reward, self.config.clip_reward)

        return rewards

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).

        Args:
            rewards: Reward values [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            masks: Attention masks [batch_size, seq_len]

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Ensure values is the same size as rewards
        if values.size(1) != seq_len:
            # Pad or trim values to match seq_len
            if values.size(1) < seq_len:
                padding = torch.zeros(batch_size, seq_len - values.size(1), device=values.device)
                values = torch.cat([values, padding], dim=1)
            else:
                values = values[:, :seq_len]

        gae = torch.zeros(batch_size, device=rewards.device)
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=rewards.device)
            else:
                next_value = values[:, t + 1]

            # Compute TD error
            delta = rewards[:, t] + self.config.gamma * next_value * masks[:, t] - values[:, t]

            # Compute GAE
            gae = delta + self.config.gamma * self.config.lambda_ * masks[:, t] * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]

        # Normalize advantages (only over valid positions)
        if self.config.whiten_rewards:
            valid_advantages = advantages[masks.bool()]
            if valid_advantages.numel() > 0:
                mean = valid_advantages.mean()
                std = valid_advantages.std()
                advantages = (advantages - mean) / (std + 1e-8)

        return advantages, returns

    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO policy loss with clipping.

        Args:
            logprobs: New log probabilities
            old_logprobs: Old log probabilities
            advantages: Advantage estimates
            mask: Attention mask

        Returns:
            Tuple of (loss, stats_dict)
        """
        # Compute ratio
        ratio = torch.exp(logprobs - old_logprobs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages

        # Policy loss (negative because we want to maximize)
        policy_loss = -torch.min(surr1, surr2)

        # Apply mask and average
        policy_loss = (policy_loss * mask).sum() / mask.sum()

        # Compute statistics
        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > self.config.clip_range).float().mean().item()
            approx_kl = (old_logprobs - logprobs).mean().item()

        stats = {
            'policy_loss': policy_loss.item(),
            'clipfrac': clipfrac,
            'approx_kl': approx_kl
        }

        return policy_loss, stats

    def compute_kl_penalty(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty.

        Args:
            logprobs: Policy log probabilities
            ref_logprobs: Reference model log probabilities
            mask: Attention mask

        Returns:
            KL penalty value
        """
        if self.config.kl_penalty == 'kl':
            # KL(policy || ref) = E[log(policy) - log(ref)]
            kl = logprobs - ref_logprobs
        elif self.config.kl_penalty == 'abs':
            # Absolute difference
            kl = (logprobs - ref_logprobs).abs()
        elif self.config.kl_penalty == 'mse':
            # Mean squared error
            kl = (logprobs - ref_logprobs).pow(2)
        else:
            raise ValueError(f"Unknown KL penalty type: {self.config.kl_penalty}")

        # Apply mask and average
        kl = (kl * mask).sum() / mask.sum()

        return kl

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform a single PPO training step.

        Args:
            batch: Batch of experience data

        Returns:
            Dictionary of training statistics
        """
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        old_logprobs = batch['logprobs'].to(self.device)
        old_values = batch['values'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        ref_logprobs = batch['ref_logprobs'].to(self.device)

        # Compute advantages
        # Expand rewards to match sequence length, but only where we have valid tokens
        batch_size, seq_len = input_ids.size()
        expanded_rewards = torch.zeros(batch_size, seq_len, device=rewards.device)
        # Put the reward at the last valid token position for each sequence
        for i in range(batch_size):
            valid_len: int = int(attention_mask[i].sum().item())
            if valid_len > 0:
                idx: int = valid_len - 1
                expanded_rewards[i, idx] = rewards[i]

        advantages, returns = self.compute_advantages(
            expanded_rewards,
            old_values,
            attention_mask
        )

        stats_accum = {}

        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch training
            indices = torch.randperm(input_ids.size(0))

            for i in range(0, input_ids.size(0), self.config.mini_batch_size):
                batch_indices = indices[i:i + self.config.mini_batch_size]

                mini_input_ids = input_ids[batch_indices]
                mini_attention_mask = attention_mask[batch_indices]
                mini_old_logprobs = old_logprobs[batch_indices]
                mini_advantages = advantages[batch_indices]
                mini_returns = returns[batch_indices]
                mini_ref_logprobs = ref_logprobs[batch_indices]

                # Forward pass
                outputs = self.model(
                    input_ids=mini_input_ids,
                    attention_mask=mini_attention_mask
                )

                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Compute log probabilities
                logprobs = F.log_softmax(logits, dim=-1)
                action_logprobs = torch.gather(
                    logprobs,
                    2,
                    mini_input_ids.unsqueeze(-1)
                ).squeeze(-1)

                # Compute policy loss
                policy_loss, policy_stats = self.compute_policy_loss(
                    action_logprobs,
                    mini_old_logprobs,
                    mini_advantages,
                    mini_attention_mask
                )

                # Compute KL penalty
                kl_penalty = self.compute_kl_penalty(
                    action_logprobs,
                    mini_ref_logprobs,
                    mini_attention_mask
                )

                # Entropy bonus
                entropy = -(logprobs * logprobs.exp()).sum(dim=-1)
                entropy = (entropy * mini_attention_mask).sum() / mini_attention_mask.sum()

                # Total loss
                loss = (
                    policy_loss
                    + self.kl_coef * kl_penalty
                    - self.config.entropy_coef * entropy
                )

                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update stats
                for k, v in policy_stats.items():
                    if k not in stats_accum:
                        stats_accum[k] = []
                    stats_accum[k].append(v)

                if 'kl_div' not in stats_accum:
                    stats_accum['kl_div'] = []
                stats_accum['kl_div'].append(kl_penalty.item())

                if 'entropy' not in stats_accum:
                    stats_accum['entropy'] = []
                stats_accum['entropy'].append(entropy.item())

                # Gradient accumulation
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.optimizer.zero_grad()

        # Average stats
        stats = {k: np.mean(v) for k, v in stats_accum.items()}

        # Adaptive KL coefficient
        if self.config.adaptive_kl:
            if stats['kl_div'] > self.config.target_kl * 1.5:
                self.kl_coef *= 1.5
            elif stats['kl_div'] < self.config.target_kl / 1.5:
                self.kl_coef /= 1.5
            self.kl_coef = np.clip(self.kl_coef, 0.01, 1.0)

        stats['kl_coef'] = self.kl_coef
        self.global_step += 1

        return stats

    @torch.no_grad()
    def evaluate(self, eval_prompts: List[str]) -> Dict[str, float]:
        """
        Evaluate model on a set of prompts.

        Args:
            eval_prompts: List of evaluation prompts

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        # Generate responses
        _, _, responses = self.generate_responses(eval_prompts)

        # Compute rewards
        encodings = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        rewards = self.compute_rewards(
            encodings['input_ids'],
            encodings['attention_mask'],
            eval_prompts,
            responses
        )

        return {
            'eval_reward_mean': rewards.mean().item(),
            'eval_reward_std': rewards.std().item(),
            'eval_reward_min': rewards.min().item(),
            'eval_reward_max': rewards.max().item()
        }
