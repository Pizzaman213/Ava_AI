"""
Reward Model for RLHF Training

This module implements reward models that can evaluate and score model outputs.
Includes a model-to-model rating system where one model evaluates another's responses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for reward model."""
    hidden_size: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    value_head_hidden_size: int = 256
    use_pairwise_loss: bool = True
    margin: float = 0.5  # Margin for pairwise ranking loss
    temperature: float = 1.0  # Temperature for softmax in rating


class RewardModel(nn.Module):
    """
    Reward Model that scores model outputs.

    This can be trained on preference data or initialized from a pretrained model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: RewardModelConfig,
        freeze_base: bool = False
    ):
        """
        Initialize reward model.

        Args:
            base_model: Base language model to build reward model from
            config: Reward model configuration
            freeze_base: Whether to freeze the base model weights
        """
        super().__init__()
        self.config = config
        self.base_model = base_model

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Value head to produce scalar reward
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.value_head_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_head_hidden_size, config.value_head_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_head_hidden_size // 2, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass to compute reward.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return a dictionary

        Returns:
            Reward scores [batch_size] or dict with rewards and hidden states
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get last hidden state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback: assume outputs is the hidden state
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # Pool hidden states (use last token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)

        # Compute reward value
        rewards = self.value_head(pooled).squeeze(-1)

        if return_dict:
            return {
                'rewards': rewards,
                'hidden_states': hidden_states,
                'pooled_states': pooled
            }
        return rewards

    def compute_pairwise_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Args:
            chosen_rewards: Rewards for chosen/preferred responses [batch_size]
            rejected_rewards: Rewards for rejected responses [batch_size]

        Returns:
            Loss value
        """
        # Pairwise ranking loss: chosen should have higher reward than rejected
        if self.config.use_pairwise_loss:
            # Margin ranking loss
            loss = F.margin_ranking_loss(
                chosen_rewards,
                rejected_rewards,
                torch.ones_like(chosen_rewards),
                margin=self.config.margin
            )
        else:
            # Simple MSE loss (assumes chosen should be 1, rejected should be 0)
            chosen_loss = F.mse_loss(chosen_rewards, torch.ones_like(chosen_rewards))
            rejected_loss = F.mse_loss(rejected_rewards, torch.zeros_like(rejected_rewards))
            loss = chosen_loss + rejected_loss

        return loss


class ModelToModelReward:
    """
    Model-to-model rating system where one model evaluates another's outputs.

    This uses a judge model to rate the quality of responses from a policy model.
    """

    def __init__(
        self,
        judge_model: nn.Module,
        tokenizer,
        rating_prompt_template: Optional[str] = None,
        device: str = 'cuda',
        temperature: float = 1.0,
        rating_scale: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize model-to-model reward system.

        Args:
            judge_model: Model that acts as the judge/rater
            tokenizer: Tokenizer for encoding prompts
            rating_prompt_template: Template for creating rating prompts
            device: Device to run on
            temperature: Temperature for reward scaling
            rating_scale: Min and max values for rating scale
        """
        self.judge_model = judge_model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.rating_scale = rating_scale

        # Default rating prompt template
        if rating_prompt_template is None:
            self.rating_prompt_template = (
                "Rate the quality of the following response on a scale from 0 to 10.\n"
                "Consider helpfulness, accuracy, coherence, and relevance.\n\n"
                "Prompt: {prompt}\n"
                "Response: {response}\n\n"
                "Rating:"
            )
        else:
            self.rating_prompt_template = rating_prompt_template

        self.judge_model.eval()
        self.judge_model.to(device)

    @torch.no_grad()
    def rate_responses(
        self,
        prompts: List[str],
        responses: List[str],
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Rate a batch of responses using the judge model.

        Args:
            prompts: List of prompts that generated the responses
            responses: List of responses to rate
            batch_size: Batch size for processing

        Returns:
            Tensor of reward scores [num_responses]
        """
        all_rewards = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]

            # Create rating prompts
            rating_texts = [
                self.rating_prompt_template.format(
                    prompt=prompt,
                    response=response
                )
                for prompt, response in zip(batch_prompts, batch_responses)
            ]

            # Encode prompts
            encodings = self.tokenizer(
                rating_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # Get model outputs
            outputs = self.judge_model(**encodings, output_hidden_states=True)

            # Extract reward from model logits
            # We can use various strategies here:
            # 1. Use specific tokens for ratings (0-10)
            # 2. Use a linear probe on hidden states
            # 3. Use the model's generation to extract numeric ratings

            # Strategy: Extract logits for numeric tokens and compute expected rating
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Get logits for tokens representing numbers 0-10
            # This assumes tokenizer has single tokens for 0-9
            numeric_token_ids = [
                self.tokenizer.encode(str(i), add_special_tokens=False)[0]
                for i in range(11)  # 0 to 10
            ]
            numeric_token_ids = torch.tensor(numeric_token_ids, device=self.device)

            # Get logits for these tokens
            numeric_logits = logits[:, numeric_token_ids]

            # Compute weighted average rating
            probs = F.softmax(numeric_logits / self.temperature, dim=-1)
            ratings = torch.arange(11, dtype=torch.float32, device=self.device)
            expected_ratings = (probs * ratings).sum(dim=-1)

            # Normalize to reward scale
            rewards = self._normalize_ratings(expected_ratings)
            all_rewards.append(rewards)

        return torch.cat(all_rewards, dim=0)

    def _normalize_ratings(self, ratings: torch.Tensor) -> torch.Tensor:
        """
        Normalize ratings from 0-10 scale to desired reward scale.

        Args:
            ratings: Raw ratings [batch_size]

        Returns:
            Normalized rewards [batch_size]
        """
        # Normalize from 0-10 to rating_scale
        normalized = (ratings / 10.0) * (self.rating_scale[1] - self.rating_scale[0]) + self.rating_scale[0]
        return normalized

    @torch.no_grad()
    def compare_responses(
        self,
        prompts: List[str],
        responses_a: List[str],
        responses_b: List[str]
    ) -> torch.Tensor:
        """
        Compare pairs of responses and return preference scores.

        Args:
            prompts: List of prompts
            responses_a: First set of responses
            responses_b: Second set of responses

        Returns:
            Preference scores: positive means A is better, negative means B is better
        """
        rewards_a = self.rate_responses(prompts, responses_a)
        rewards_b = self.rate_responses(prompts, responses_b)

        # Return difference (positive = A better, negative = B better)
        return rewards_a - rewards_b


class EnsembleRewardModel:
    """
    Ensemble of multiple reward models for more robust evaluation.
    """

    def __init__(
        self,
        reward_models: List[Union[RewardModel, ModelToModelReward]],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble reward model.

        Args:
            reward_models: List of reward models
            weights: Optional weights for each model (defaults to equal weights)
        """
        self.reward_models = reward_models

        if weights is None:
            self.weights = [1.0 / len(reward_models)] * len(reward_models)
        else:
            assert len(weights) == len(reward_models)
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def get_rewards(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        responses: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Get ensemble rewards.

        Args:
            input_ids: Input token IDs (for RewardModel)
            attention_mask: Attention mask (for RewardModel)
            prompts: Prompts (for ModelToModelReward)
            responses: Responses (for ModelToModelReward)

        Returns:
            Weighted average rewards
        """
        all_rewards = []

        for model, weight in zip(self.reward_models, self.weights):
            if isinstance(model, RewardModel):
                assert input_ids is not None
                rewards = model(input_ids, attention_mask)
                if isinstance(rewards, dict):
                    rewards = rewards['rewards']
            elif isinstance(model, ModelToModelReward):
                assert prompts is not None and responses is not None
                rewards = model.rate_responses(prompts, responses)
            else:
                raise ValueError(f"Unknown reward model type: {type(model)}")

            all_rewards.append(rewards * weight)

        return torch.stack(all_rewards).sum(dim=0) if all_rewards else torch.tensor(0.0)